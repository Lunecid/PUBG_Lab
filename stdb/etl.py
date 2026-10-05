"""
stdb/etl.py
===========
Ingest raw `pubg.*` telemetry into the `stdb.*` STDB layer.

Pipeline (per match):
  1. matches              ← pubg.matches + pubg.telem_match_start
  2. teams                ← pubg.rosters
  3. moving_objects       ← pubg.telem_positions (resampled at 10s grid)
                            • player rows (one per account_id × snapshot)
                            • team rows   (plain-mean centroid per team × snapshot)
  4. dynamic_regions      ← pubg.telem_game_states (safe + poison)
                            phase_idx joined from pubg.telem_phase_changes
  5. spatial_events       ← pubg.telem_kills / telem_damage / telem_groggy /
                                  telem_parachute_landing
  6. team_phase_features  ← derived (SQL window aggregation)
  7. team_proximity       ← derived (per-snapshot pairwise distances)

Coordinates are converted from CM (raw) → METERS at ingest time.
The whole ingest for one match runs inside a single transaction so a failure
rolls back cleanly.  Re-running on the same match is idempotent (PK upserts +
DELETE-then-INSERT for the derived tables).

Usage:
    python -m stdb.etl --match-id <uuid>
    python -m stdb.etl --map Baltic_Main --mode squad-fpp --limit 50
    python -m stdb.etl --all
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import defaultdict
from typing import Iterable

import psycopg2.extras

from stdb.conn import connect

logger = logging.getLogger("stdb.etl")

# ── Constants ────────────────────────────────────────────────────────────────
CM_TO_M = 100.0           # raw cm → meters
SNAPSHOT_INTERVAL = 10    # seconds (10s grid)
SNAPSHOT_HALF_WIN = 5     # ±5s match window for resampling
PROXIMITY_TOP_K = 5       # k-NN edges per team per snapshot (saved)
ENCOUNTER_WINDOW = 10.0   # seconds before snapshot to count as "recent" event


# ── Helpers ──────────────────────────────────────────────────────────────────


def event_time_to_elapsed(event_time, match_start_time) -> float | None:
    if event_time is None or match_start_time is None:
        return None
    return (event_time - match_start_time).total_seconds()


def _floor_to_grid(t_s: float, interval: int = SNAPSHOT_INTERVAL) -> int:
    return int(round(t_s / interval) * interval)


# ── Per-match ETL ────────────────────────────────────────────────────────────


def _select_match_meta(cur, match_id: str) -> dict | None:
    cur.execute(
        """
        SELECT m.match_id, m.duration, m.game_mode, m.map_name, m.is_custom,
               ms.event_time AS started_at, ms.team_size,
               (SELECT count(*) FROM pubg.rosters r WHERE r.match_id = m.match_id) AS num_teams,
               (SELECT count(*) FROM pubg.participants p WHERE p.match_id = m.match_id) AS num_players
          FROM pubg.matches m
          LEFT JOIN pubg.telem_match_start ms ON ms.match_id = m.match_id
         WHERE m.match_id = %s
        """,
        (match_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def _upsert_match(cur, meta: dict) -> None:
    cur.execute(
        """
        INSERT INTO stdb.matches (match_id, started_at, duration_s, map_name,
                                  game_mode, team_size, is_custom, num_teams,
                                  num_players)
        VALUES (%(match_id)s, %(started_at)s, %(duration)s, %(map_name)s,
                %(game_mode)s, %(team_size)s, %(is_custom)s, %(num_teams)s,
                %(num_players)s)
        ON CONFLICT (match_id) DO UPDATE SET
            started_at  = EXCLUDED.started_at,
            duration_s  = EXCLUDED.duration_s,
            map_name    = EXCLUDED.map_name,
            game_mode   = EXCLUDED.game_mode,
            team_size   = EXCLUDED.team_size,
            is_custom   = EXCLUDED.is_custom,
            num_teams   = EXCLUDED.num_teams,
            num_players = EXCLUDED.num_players
        """,
        meta,
    )


def _upsert_teams(cur, match_id: str) -> None:
    cur.execute(
        """
        INSERT INTO stdb.teams (match_id, team_id, final_rank, num_players, won)
        SELECT r.match_id, r.team_id, r.rank,
               (SELECT count(*) FROM pubg.participants p
                 WHERE p.match_id = r.match_id AND p.roster_id = r.roster_id),
               r.won
          FROM pubg.rosters r
         WHERE r.match_id = %s AND r.team_id IS NOT NULL
        ON CONFLICT (match_id, team_id) DO UPDATE SET
            final_rank  = EXCLUDED.final_rank,
            num_players = EXCLUDED.num_players,
            won         = EXCLUDED.won
        """,
        (match_id,),
    )


def _ingest_moving_objects(
    cur, match_id: str, duration_s: int, *, team_only: bool = False
) -> tuple[int, int]:
    """Resample telem_positions to 10s grid → moving_objects (player + team rows).

    ``team_only=True`` computes the same grid but inserts only the team centroid rows
    (used by stdb.base_stage for the Fix B reruns, which never read player rows).

    Returns (n_player_rows, n_team_rows).  With ``team_only`` the first element is 0.
    """
    cur.execute(
        """
        SELECT account_id, team_id, elapsed_time, health,
               pos_x, pos_y, pos_z, vehicle_type
          FROM pubg.telem_positions
         WHERE match_id = %s
           AND account_id IS NOT NULL
           AND pos_x IS NOT NULL
           AND elapsed_time IS NOT NULL
           AND is_game >= 0.1
         ORDER BY account_id, elapsed_time
        """,
        (match_id,),
    )

    rows = cur.fetchall()
    if not rows:
        return 0, 0

    by_player: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for account_id, team_id, et, health, px, py, pz, veh in rows:
        by_player[(account_id, team_id)].append(
            {
                "t": float(et),
                "health": float(health) if health is not None else None,
                "x": float(px) / CM_TO_M,
                "y": float(py) / CM_TO_M,
                "z": float(pz) / CM_TO_M,
                "veh": veh,
            }
        )

    grid_ts = list(range(0, int(duration_s) + SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL))

    player_rows: list[tuple] = []
    team_buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for (account_id, team_id), pts in by_player.items():
        pts.sort(key=lambda r: r["t"])
        # for each grid timestep, take the closest sample within the half-window
        i = 0
        for gt in grid_ts:
            while i + 1 < len(pts) and abs(pts[i + 1]["t"] - gt) < abs(pts[i]["t"] - gt):
                i += 1
            best = pts[i]
            if abs(best["t"] - gt) > SNAPSHOT_HALF_WIN:
                continue
            in_vehicle = bool(best["veh"]) and best["veh"] not in (
                "TransportAircraft",
                "DummyTransportAircraft_C",
            )
            health_v = best["health"]
            is_alive = (health_v is None) or (health_v > 0)
            player_rows.append(
                (
                    match_id,
                    "player",
                    account_id,
                    team_id,
                    gt,
                    best["x"], best["y"], best["z"],
                    health_v,
                    in_vehicle,
                    is_alive,
                )
            )
            team_buckets[(team_id, gt)].append(best | {"alive": is_alive})

    team_rows: list[tuple] = []
    for (team_id, gt), bucket in team_buckets.items():
        alive_bucket = [b for b in bucket if b["alive"]]
        use = alive_bucket if alive_bucket else bucket
        cx = sum(b["x"] for b in use) / len(use)
        cy = sum(b["y"] for b in use) / len(use)
        cz = sum(b["z"] for b in use) / len(use)
        team_alive = any(b["alive"] for b in bucket)
        team_rows.append(
            (
                match_id,
                "team",
                str(team_id),
                team_id,
                gt,
                cx, cy, cz,
                None,             # team-level health is undefined
                False,            # in_vehicle aggregated false
                team_alive,
            )
        )

    if not player_rows and not team_rows:
        return 0, 0

    insert_sql = """
        INSERT INTO stdb.moving_objects
            (match_id, agent_kind, agent_id, team_id, snapshot_t, pos,
             health, in_vehicle, is_alive)
        VALUES (%s, %s, %s, %s, %s,
                ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0),
                %s, %s, %s)
        ON CONFLICT (match_id, agent_kind, agent_id, snapshot_t) DO UPDATE SET
            team_id    = EXCLUDED.team_id,
            pos        = EXCLUDED.pos,
            health     = EXCLUDED.health,
            in_vehicle = EXCLUDED.in_vehicle,
            is_alive   = EXCLUDED.is_alive
    """
    if team_only:
        player_rows = []
    psycopg2.extras.execute_batch(cur, insert_sql, player_rows, page_size=500)
    psycopg2.extras.execute_batch(cur, insert_sql, team_rows, page_size=500)
    return len(player_rows), len(team_rows)


def _ingest_dynamic_regions(cur, match_id: str) -> int:
    """telem_game_states → dynamic_regions (safe + poison).  Phase joined later."""
    cur.execute(
        """
        SELECT elapsed_time,
               safe_zone_x, safe_zone_y, safe_zone_z, safe_zone_radius,
               poison_zone_x, poison_zone_y, poison_zone_z, poison_zone_radius
          FROM pubg.telem_game_states
         WHERE match_id = %s
           AND elapsed_time IS NOT NULL
         ORDER BY elapsed_time
        """,
        (match_id,),
    )
    states = cur.fetchall()
    if not states:
        return 0

    rows: list[tuple] = []
    for (et, sx, sy, sz, sr, px, py, pz, pr) in states:
        gt = _floor_to_grid(int(et))
        for kind, (cx, cy, cz, r) in (
            ("safe",   (sx, sy, sz, sr)),
            ("poison", (px, py, pz, pr)),
        ):
            if cx is None or r is None:
                continue
            cx_m = float(cx) / CM_TO_M
            cy_m = float(cy) / CM_TO_M
            cz_m = float(cz or 0) / CM_TO_M
            r_m = float(r) / CM_TO_M
            if r_m <= 0:
                continue
            rows.append((match_id, kind, gt, cx_m, cy_m, cz_m, r_m))

    # Keep last value per (kind, snapshot) — already monotonically increasing in elapsed_time
    deduped: dict[tuple[str, int], tuple] = {}
    for r in rows:
        deduped[(r[1], r[2])] = r
    rows = list(deduped.values())

    insert_sql = """
        INSERT INTO stdb.dynamic_regions
            (match_id, region_kind, snapshot_t, center_pos, radius_m, region)
        VALUES (%s, %s, %s,
                ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0),
                %s,
                ST_Buffer(ST_Force2D(ST_MakePoint(%s, %s)), %s, 64)::geometry(Polygon, 0))
        ON CONFLICT (match_id, region_kind, snapshot_t) DO UPDATE SET
            center_pos = EXCLUDED.center_pos,
            radius_m   = EXCLUDED.radius_m,
            region     = EXCLUDED.region
    """
    batched = [(m, k, t, x, y, z, rm, x, y, rm) for (m, k, t, x, y, z, rm) in rows]
    psycopg2.extras.execute_batch(cur, insert_sql, batched, page_size=200)

    # Phase tagging from telem_phase_changes.  Some raw rows have NULL
    # elapsed_time, so fall back to event_time - match_start_time.
    cur.execute(
        """
        SELECT coalesce(
                   pc.elapsed_time::real,
                   extract(epoch FROM (pc.event_time - ms.event_time))::real
               ) AS elapsed_s,
               pc.phase
          FROM pubg.telem_phase_changes pc
          JOIN pubg.telem_match_start ms
            ON ms.match_id = pc.match_id
         WHERE pc.match_id = %s
           AND pc.phase IS NOT NULL
           AND coalesce(
                   pc.elapsed_time::real,
                   extract(epoch FROM (pc.event_time - ms.event_time))::real
               ) IS NOT NULL
         ORDER BY elapsed_s
        """,
        (match_id,),
    )
    phases = cur.fetchall()
    if phases:
        boundaries = [(int(et), int(float(ph))) for (et, ph) in phases]
        cur.execute(
            "SELECT snapshot_t FROM stdb.dynamic_regions WHERE match_id = %s ORDER BY snapshot_t",
            (match_id,),
        )
        snaps = [r[0] for r in cur.fetchall()]
        updates = []
        for t in snaps:
            phase_idx = 0
            for bt, bp in boundaries:
                if bt <= t:
                    phase_idx = bp
                else:
                    break
            updates.append((phase_idx, match_id, t))
        psycopg2.extras.execute_batch(
            cur,
            "UPDATE stdb.dynamic_regions SET phase_idx = %s WHERE match_id = %s AND snapshot_t = %s",
            updates,
            page_size=500,
        )

    return len(rows)


def _team_lookup_from_positions(cur, match_id: str) -> dict[str, int]:
    """account_id → team_id (from any pubg.telem_positions row)."""
    cur.execute(
        """
        SELECT DISTINCT ON (account_id) account_id, team_id
          FROM pubg.telem_positions
         WHERE match_id = %s AND account_id IS NOT NULL AND team_id IS NOT NULL
        """,
        (match_id,),
    )
    return {a: t for a, t in cur.fetchall()}


def _ingest_spatial_events(cur, match_id: str, started_at) -> int:
    """telem_kills + telem_damage + telem_groggy + telem_parachute_landing → spatial_events."""
    team_lookup = _team_lookup_from_positions(cur, match_id)

    def tlookup(account_id: str | None) -> int | None:
        if account_id is None:
            return None
        return team_lookup.get(account_id)

    rows: list[tuple] = []

    # --- kills ---
    cur.execute(
        """
        SELECT event_time, killer_id, killer_x, killer_y, killer_z,
               victim_id, victim_x, victim_y, victim_z, victim_team_id,
               damage_reason, damage_type, damage_causer, distance, is_suicide
          FROM pubg.telem_kills
         WHERE match_id = %s
        """,
        (match_id,),
    )
    for r in cur.fetchall():
        (et, kid, kx, ky, kz, vid, vx, vy, vz, vteam,
         dreason, dtype, dcauser, dist, is_sui) = r
        rows.append((
            "kill",
            event_time_to_elapsed(et, started_at),
            kid, tlookup(kid),
            (float(kx) / CM_TO_M, float(ky) / CM_TO_M, float(kz) / CM_TO_M) if kx is not None else None,
            vid, vteam if vteam is not None else tlookup(vid),
            (float(vx) / CM_TO_M, float(vy) / CM_TO_M, float(vz) / CM_TO_M) if vx is not None else None,
            None,
            dtype,
            dcauser,
            float(dist) if dist is not None else None,
            psycopg2.extras.Json({"reason": dreason, "is_suicide": is_sui}),
        ))

    # --- damage ---
    cur.execute(
        """
        SELECT event_time, attacker_id, attacker_x, attacker_y, attacker_z,
               victim_id, victim_x, victim_y, victim_z,
               damage, damage_reason, damage_type, damage_causer, distance, is_through_wall
          FROM pubg.telem_damage
         WHERE match_id = %s
        """,
        (match_id,),
    )
    for r in cur.fetchall():
        (et, aid, ax, ay, az, vid, vx, vy, vz, dmg,
         dreason, dtype, dcauser, dist, through) = r
        rows.append((
            "damage",
            event_time_to_elapsed(et, started_at),
            aid, tlookup(aid),
            (float(ax) / CM_TO_M, float(ay) / CM_TO_M, float(az) / CM_TO_M) if ax is not None else None,
            vid, tlookup(vid),
            (float(vx) / CM_TO_M, float(vy) / CM_TO_M, float(vz) / CM_TO_M) if vx is not None else None,
            float(dmg) if dmg is not None else None,
            dtype,
            dcauser,
            float(dist) if dist is not None else None,
            psycopg2.extras.Json({"reason": dreason, "through_wall": through}),
        ))

    # --- groggy ---
    cur.execute(
        """
        SELECT event_time, attacker_id, attacker_x, attacker_y, attacker_z,
               victim_id, victim_x, victim_y, victim_z,
               damage_reason, damage_type, damage_causer, distance, is_attacker_in_vehicle
          FROM pubg.telem_groggy
         WHERE match_id = %s
        """,
        (match_id,),
    )
    for r in cur.fetchall():
        (et, aid, ax, ay, az, vid, vx, vy, vz,
         dreason, dtype, dcauser, dist, in_veh) = r
        rows.append((
            "groggy",
            event_time_to_elapsed(et, started_at),
            aid, tlookup(aid),
            (float(ax) / CM_TO_M, float(ay) / CM_TO_M, float(az) / CM_TO_M) if ax is not None else None,
            vid, tlookup(vid),
            (float(vx) / CM_TO_M, float(vy) / CM_TO_M, float(vz) / CM_TO_M) if vx is not None else None,
            None, dtype, dcauser,
            float(dist) if dist is not None else None,
            psycopg2.extras.Json({"reason": dreason, "attacker_in_vehicle": in_veh}),
        ))

    # --- parachute landing ---
    cur.execute(
        """
        SELECT event_time, account_id, team_id, pos_x, pos_y, pos_z, distance
          FROM pubg.telem_parachute_landing
         WHERE match_id = %s
        """,
        (match_id,),
    )
    for r in cur.fetchall():
        (et, aid, tid, px, py, pz, dist) = r
        rows.append((
            "landing",
            event_time_to_elapsed(et, started_at),
            None, None, None,
            aid, tid,
            (float(px) / CM_TO_M, float(py) / CM_TO_M, float(pz) / CM_TO_M) if px is not None else None,
            None, None, None,
            float(dist) if dist is not None else None,
            None,
        ))

    if not rows:
        return 0

    # Idempotent: delete-then-insert
    cur.execute("DELETE FROM stdb.spatial_events WHERE match_id = %s", (match_id,))

    insert_sql = """
        INSERT INTO stdb.spatial_events
            (match_id, event_kind, event_t,
             attacker_id, attacker_team, attacker_pos,
             victim_id, victim_team, victim_pos,
             damage, damage_type, damage_causer, distance_m, extra)
        VALUES (
            %s, %s, %s,
            %s, %s, CASE WHEN %s IS NULL THEN NULL ELSE ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0) END,
            %s, %s, CASE WHEN %s IS NULL THEN NULL ELSE ST_MakePoint(%s, %s, %s)::geometry(PointZ, 0) END,
            %s, %s, %s, %s, %s
        )
    """
    flat = []
    for (kind, t, aid, ateam, apos, vid, vteam, vpos, dmg, dtype, dcauser, dist, extra) in rows:
        if t is None:
            continue
        ax, ay, az = (apos if apos is not None else (None, None, None))
        vx, vy, vz = (vpos if vpos is not None else (None, None, None))
        flat.append((
            match_id, kind, t,
            aid, ateam, ax, ax, ay, az,
            vid, vteam, vx, vx, vy, vz,
            dmg, dtype, dcauser, dist, extra,
        ))
    psycopg2.extras.execute_batch(cur, insert_sql, flat, page_size=500)
    return len(flat)


def _build_team_proximity(cur, match_id: str) -> int:
    """Pairwise team distances per snapshot, with recent-encounter flag."""
    cur.execute(
        "DELETE FROM stdb.team_proximity WHERE match_id = %s", (match_id,)
    )
    cur.execute(
        """
        WITH centroids AS (
            SELECT team_id, snapshot_t, pos, is_alive
              FROM stdb.moving_objects
             WHERE match_id = %s AND agent_kind = 'team'
        ),
        pairs AS (
            SELECT a.snapshot_t,
                   LEAST(a.team_id, b.team_id) AS team_a,
                   GREATEST(a.team_id, b.team_id) AS team_b,
                   ST_3DDistance(a.pos, b.pos) AS distance_m,
                   (a.is_alive AND b.is_alive) AS both_alive
              FROM centroids a
              JOIN centroids b
                ON a.snapshot_t = b.snapshot_t AND a.team_id < b.team_id
        )
        INSERT INTO stdb.team_proximity
            (match_id, snapshot_t, team_a, team_b, distance_m, encounter_recently)
        SELECT %s, snapshot_t, team_a, team_b, distance_m,
               FALSE
          FROM pairs
         WHERE both_alive
        """,
        (match_id, match_id),
    )

    # Mark only already-observable encounters.  Earlier floor-bucketing could
    # assign a t+epsilon event to the prior grid cut.
    cur.execute(
        """
        UPDATE stdb.team_proximity tp
           SET encounter_recently = TRUE
          FROM (
            SELECT DISTINCT
                   ceil(event_t / %s) * %s AS snap_t,
                   LEAST(attacker_team, victim_team) AS team_a,
                   GREATEST(attacker_team, victim_team) AS team_b
              FROM stdb.spatial_events
             WHERE match_id = %s
               AND event_kind IN ('kill','damage','groggy')
               AND attacker_team IS NOT NULL
               AND victim_team   IS NOT NULL
               AND attacker_team <> victim_team
          ) e
         WHERE tp.match_id   = %s
           AND tp.snapshot_t = e.snap_t::int
           AND tp.team_a     = e.team_a
           AND tp.team_b     = e.team_b
        """,
        (SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL, match_id, match_id),
    )

    cur.execute(
        "SELECT count(*) FROM stdb.team_proximity WHERE match_id = %s", (match_id,)
    )
    return int(cur.fetchone()[0])


def _build_team_phase_features(cur, match_id: str) -> int:
    """Aggregate moving_objects + dynamic_regions + spatial_events per (team, phase)."""
    cur.execute(
        "DELETE FROM stdb.team_phase_features WHERE match_id = %s", (match_id,)
    )
    cur.execute(
        """
        WITH snaps AS (
            SELECT mo.team_id, mo.snapshot_t, mo.pos,
                   dr_poison.region AS poison_region,
                   dr_poison.center_pos AS poison_center,
                   dr_poison.radius_m  AS poison_r,
                   dr_safe.region AS safe_region,
                   dr_safe.center_pos AS safe_center,
                   dr_safe.radius_m AS safe_r,
                   dr_poison.phase_idx
              FROM stdb.moving_objects mo
         LEFT JOIN stdb.dynamic_regions dr_poison
                ON dr_poison.match_id = mo.match_id
               AND dr_poison.region_kind = 'poison'
               AND dr_poison.snapshot_t = mo.snapshot_t
         LEFT JOIN stdb.dynamic_regions dr_safe
                ON dr_safe.match_id = mo.match_id
               AND dr_safe.region_kind = 'safe'
               AND dr_safe.snapshot_t = mo.snapshot_t
             WHERE mo.match_id = %s
               AND mo.agent_kind = 'team'
               AND mo.is_alive
        ),
        prox_raw AS (
            SELECT snapshot_t, team_a AS team_id, count(*) AS n_close
              FROM stdb.team_proximity
             WHERE match_id = %s AND distance_m < 100.0
             GROUP BY snapshot_t, team_a
            UNION ALL
            SELECT snapshot_t, team_b AS team_id, count(*) AS n_close
              FROM stdb.team_proximity
             WHERE match_id = %s AND distance_m < 100.0
             GROUP BY snapshot_t, team_b
        ),
        prox AS (
            SELECT snapshot_t, team_id, sum(n_close) AS n_close
              FROM prox_raw
             GROUP BY snapshot_t, team_id
        ),
        combat AS (
            SELECT victim_team AS team_id,
                   floor(event_t / %s)::int * %s AS snapshot_t,
                   sum(coalesce(damage, 0))::real AS dmg_taken
              FROM stdb.spatial_events
             WHERE match_id = %s
               AND event_kind = 'damage'
               AND victim_team IS NOT NULL
             GROUP BY victim_team, snapshot_t
        ),
        joined AS (
            SELECT s.team_id, s.snapshot_t, s.phase_idx,
                   ST_Distance(ST_Force2D(s.pos), ST_Boundary(s.poison_region)) AS dist_to_poison,
                   CASE WHEN ST_Contains(s.safe_region, ST_Force2D(s.pos)) THEN 1 ELSE 0 END AS in_safe,
                   ST_X(s.pos) AS x, ST_Y(s.pos) AS y,
                   coalesce(p.n_close, 0) AS n_close,
                   coalesce(c.dmg_taken, 0)::real AS dmg_taken
              FROM snaps s
         LEFT JOIN prox p
                ON p.snapshot_t = s.snapshot_t AND p.team_id = s.team_id
         LEFT JOIN combat c
                ON c.snapshot_t = s.snapshot_t AND c.team_id = s.team_id
        )
        INSERT INTO stdb.team_phase_features
            (match_id, team_id, phase_idx,
             mean_dist_to_boundary_m, enemy_density_100m,
             recent_combat_exposure, position_stability, inside_safe_ratio)
        SELECT %s,
               team_id,
               coalesce(phase_idx, 0) AS phase_idx,
               avg(dist_to_poison),
               avg(n_close),
               sum(dmg_taken),
               CASE WHEN count(*) > 1 THEN
                    1.0 / (1.0 + sqrt(var_pop(x) + var_pop(y)))
                    ELSE NULL END,
               avg(in_safe)::real
          FROM joined
         GROUP BY team_id, coalesce(phase_idx, 0)
        """,
        (match_id, match_id, match_id, SNAPSHOT_INTERVAL, SNAPSHOT_INTERVAL, match_id, match_id),
    )

    cur.execute(
        "SELECT count(*) FROM stdb.team_phase_features WHERE match_id = %s", (match_id,)
    )
    return int(cur.fetchone()[0])


# ── Match selection ─────────────────────────────────────────────────────────


def list_match_ids(map_name: str | None, mode: str | None, limit: int | None) -> list[str]:
    conn = connect()
    cur = conn.cursor()
    sql = """
        SELECT m.match_id
          FROM pubg.matches m
          JOIN pubg.collection_log c ON c.match_id = m.match_id
         WHERE c.telemetry_fetched = TRUE
    """
    params: list = []
    if map_name:
        sql += " AND m.map_name = %s"
        params.append(map_name)
    if mode:
        sql += " AND m.game_mode = %s"
        params.append(mode)
    sql += " ORDER BY m.created_at"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    cur.execute(sql, params)
    ids = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()
    return ids


def ingest_match(match_id: str) -> dict:
    t0 = time.time()
    conn = connect()
    cur = conn.cursor()
    try:
        meta = _select_match_meta(cur, match_id)
        if meta is None:
            return {"match_id": match_id, "ok": False, "reason": "match not found"}

        _upsert_match(cur, meta)
        _upsert_teams(cur, match_id)
        n_player, n_team = _ingest_moving_objects(cur, match_id, meta["duration"])
        n_region = _ingest_dynamic_regions(cur, match_id)
        n_event = _ingest_spatial_events(cur, match_id, meta["started_at"])
        n_prox = _build_team_proximity(cur, match_id)
        n_feat = _build_team_phase_features(cur, match_id)

        conn.commit()
        return {
            "match_id": match_id, "ok": True,
            "elapsed_s": round(time.time() - t0, 2),
            "player_rows": n_player, "team_rows": n_team,
            "region_rows": n_region, "event_rows": n_event,
            "proximity_rows": n_prox, "phase_feature_rows": n_feat,
        }
    except Exception as exc:
        conn.rollback()
        logger.exception("ingest failed for %s", match_id)
        return {"match_id": match_id, "ok": False, "reason": repr(exc)}
    finally:
        cur.close(); conn.close()


def verify_etl(match_id: str) -> dict:
    """Cross-check stdb.* row counts vs pubg.* sources."""
    conn = connect()
    cur = conn.cursor()
    out = {"match_id": match_id}
    try:
        cur.execute("SELECT count(*) FROM stdb.moving_objects WHERE match_id=%s AND agent_kind='team'", (match_id,))
        out["stdb_team_rows"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM stdb.moving_objects WHERE match_id=%s AND agent_kind='player'", (match_id,))
        out["stdb_player_rows"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM stdb.dynamic_regions WHERE match_id=%s", (match_id,))
        out["stdb_region_rows"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM stdb.spatial_events WHERE match_id=%s AND event_kind='kill'", (match_id,))
        out["stdb_kill_events"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pubg.telem_kills WHERE match_id=%s", (match_id,))
        out["pubg_kill_rows"] = cur.fetchone()[0]
    finally:
        cur.close(); conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", help="Single match UUID/text id")
    ap.add_argument("--map", help="Filter by map_name (e.g. Baltic_Main)")
    ap.add_argument("--mode", help="Filter by game_mode (e.g. squad-fpp)")
    ap.add_argument("--limit", type=int, help="Limit number of matches")
    ap.add_argument("--all", action="store_true", help="Ingest all telemetry-fetched matches")
    ap.add_argument("--verify", action="store_true", help="Run verify_etl after ingest")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    match_ids: Iterable[str]
    if args.match_id:
        match_ids = [args.match_id]
    elif args.all or args.map or args.mode:
        match_ids = list_match_ids(args.map, args.mode, args.limit)
    else:
        ap.error("must pass --match-id / --all / --map / --mode")
        return 2

    n_ok = n_fail = 0
    for mid in match_ids:
        out = ingest_match(mid)
        if out["ok"]:
            n_ok += 1
            logger.info(
                "OK %s  player=%d team=%d region=%d event=%d prox=%d feat=%d  (%.1fs)",
                mid, out["player_rows"], out["team_rows"],
                out["region_rows"], out["event_rows"],
                out["proximity_rows"], out["phase_feature_rows"], out["elapsed_s"],
            )
            if args.verify:
                logger.info("verify: %s", verify_etl(mid))
        else:
            n_fail += 1
            logger.error("FAIL %s  reason=%s", mid, out.get("reason"))

    logger.info("Done.  ok=%d  fail=%d", n_ok, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
