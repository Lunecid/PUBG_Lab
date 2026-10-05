"""
Phase-level trajectory and exposure feature builder.

This script materializes the spatial features that are derived from the base
STDB objects rather than directly from raw telemetry:

    stdb.moving_objects + stdb.dynamic_regions + stdb.team_proximity
        -> stdb.team_phase_trajectory
        -> stdb.team_phase_exposure

It is intentionally match-scoped and idempotent so large batches can be resumed.

Fix B + single clock (2026-10-05): every join to the 10 s grid (stdb.moving_objects and, through
the shared snapshot_t, stdb.dynamic_regions / stdb.team_proximity) runs on the game clock window
``[phase_start_s - delta, phase_end_s - delta - SNAPSHOT_INTERVAL]``: the phase bounds are
LogMatchStart-clock numbers, the grid is on the elapsed_time clock that lags by the per-match
``delta`` (see stdb/causal_cut.py).  Event buckets (combat_points) are moved onto the same grid.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from stdb.causal_cut import (
    CUT_COLUMN,
    WIN_START_COLUMN,
    fetch_clock_delta,
    sql_delta_literal,
    sql_window_columns,
)
from stdb.conn import apply_sql_file, connect

SNAPSHOT_INTERVAL = 10.0

logger = logging.getLogger("stdb.phase_spatial_feature_etl")


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_phase_trajectory (
    match_id                    TEXT NOT NULL,
    team_id                     INTEGER NOT NULL,
    phase_idx                   INTEGER NOT NULL,
    waypoints                   INTEGER NOT NULL,
    phase_start_s               REAL NOT NULL,
    phase_end_s                 REAL NOT NULL,
    phase_duration_s            REAL NOT NULL,
    any_alive                   BOOLEAN NOT NULL,
    trajectory                  geometry(LineStringZ, 0),
    path_length_m               REAL,
    displacement_m              REAL,
    sinuosity                   REAL,
    avg_speed_mps               REAL,
    rotation_timing             REAL,
    safe_alignment              REAL,
    zone_entry_time             REAL,
    outside_safe_duration_s     REAL,
    boundary_distance_slope     REAL,
    PRIMARY KEY (match_id, team_id, phase_idx)
);

CREATE INDEX IF NOT EXISTS ix_tpt_team
    ON stdb.team_phase_trajectory (match_id, team_id);
CREATE INDEX IF NOT EXISTS ix_tpt_phase
    ON stdb.team_phase_trajectory (phase_idx);

CREATE TABLE IF NOT EXISTS stdb.team_phase_exposure (
    match_id                    TEXT NOT NULL,
    team_id                     INTEGER NOT NULL,
    phase_idx                   INTEGER NOT NULL,
    mean_dist_to_boundary_m     REAL,
    min_dist_to_boundary_m      REAL,
    outside_safe_ratio          REAL,
    avg_enemy_density_100m      REAL,
    min_nearest_enemy_m         REAL,
    path_near_enemy_ratio       REAL,
    recent_damage_taken         REAL,
    combat_zone_pass_count      INTEGER,
    PRIMARY KEY (match_id, team_id, phase_idx)
);

CREATE INDEX IF NOT EXISTS ix_tpe_team
    ON stdb.team_phase_exposure (match_id, team_id);

CREATE INDEX IF NOT EXISTS ix_dr_match_kind_snap
    ON stdb.dynamic_regions (match_id, region_kind, snapshot_t);
CREATE INDEX IF NOT EXISTS ix_mo_match_kind_team_snap
    ON stdb.moving_objects (match_id, agent_kind, team_id, snapshot_t);
CREATE INDEX IF NOT EXISTS ix_tp_match_snap_team_a
    ON stdb.team_proximity (match_id, snapshot_t, team_a);
CREATE INDEX IF NOT EXISTS ix_tp_match_snap_team_b
    ON stdb.team_proximity (match_id, snapshot_t, team_b);
"""


def apply_schema() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def build_match(cur, match_id: str) -> dict[str, int]:
    delta_s = fetch_clock_delta(cur, match_id)
    dsql = sql_delta_literal(delta_s)
    win_cols = sql_window_columns(dsql)
    cur.execute("DELETE FROM stdb.team_phase_trajectory WHERE match_id = %s", (match_id,))
    cur.execute("DELETE FROM stdb.team_phase_exposure WHERE match_id = %s", (match_id,))

    cur.execute(
        f"""
        WITH phase_bounds AS (
            SELECT match_id, phase_idx, phase_start_s, phase_end_s, phase_duration_s,
                   {win_cols}
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
               AND phase_idx > 0
        ),
        snaps AS (
            SELECT mo.team_id,
                   pb.phase_idx,
                   pb.phase_start_s,
                   pb.phase_end_s,
                   pb.phase_duration_s,
                   pb.{WIN_START_COLUMN},
                   mo.snapshot_t::real AS snapshot_t,
                   mo.is_alive,
                   mo.pos AS pos3d,
                   ST_Force2D(mo.pos) AS pos2d,
                   dr.region AS safe_region,
                   ST_Force2D(dr.center_pos) AS safe_center,
                   CASE
                       WHEN dr.region IS NULL THEN NULL
                       ELSE ST_Distance(ST_Force2D(mo.pos), ST_Boundary(dr.region))
                   END AS dist_to_boundary,
                   CASE
                       WHEN dr.region IS NULL THEN NULL
                       ELSE ST_Contains(dr.region, ST_Force2D(mo.pos))
                   END AS in_safe
              FROM phase_bounds pb
              JOIN stdb.moving_objects mo
                ON mo.match_id = pb.match_id
               AND mo.agent_kind = 'team'
               -- Fix B + one clock: grid window on the game clock, capped at the end
               AND mo.snapshot_t BETWEEN pb.{WIN_START_COLUMN} AND pb.{CUT_COLUMN}
               AND mo.is_alive
         LEFT JOIN stdb.dynamic_regions dr
                ON dr.match_id = mo.match_id
               AND dr.region_kind = 'safe'
               AND dr.snapshot_t = mo.snapshot_t
        ),
        summary AS (
            SELECT %s::text AS match_id,
                   team_id,
                   phase_idx,
                   count(*)::integer AS waypoints,
                   min(phase_start_s)::real AS phase_start_s,
                   max(phase_end_s)::real AS phase_end_s,
                   max(phase_duration_s)::real AS phase_duration_s,
                   min({WIN_START_COLUMN}) AS {WIN_START_COLUMN},
                   bool_or(is_alive)::boolean AS any_alive,
                   CASE
                       WHEN count(*) >= 2 THEN ST_MakeLine(pos3d ORDER BY snapshot_t)
                       ELSE NULL
                   END AS trajectory,
                   (array_agg(pos2d ORDER BY snapshot_t))[1] AS first_pos,
                   (array_agg(pos2d ORDER BY snapshot_t DESC))[1] AS last_pos,
                   (array_agg(safe_center ORDER BY snapshot_t DESC)
                       FILTER (WHERE safe_center IS NOT NULL))[1] AS last_safe_center,
                   min(snapshot_t) FILTER (WHERE in_safe) AS first_safe_t,
                   sum(CASE WHEN in_safe = FALSE THEN %s ELSE 0 END)::real
                       AS outside_safe_duration_s,
                   regr_slope(dist_to_boundary, snapshot_t)::real
                       AS boundary_distance_slope
              FROM snaps
             GROUP BY team_id, phase_idx
        ),
        geom AS (
            SELECT s.*,
                   CASE
                       WHEN trajectory IS NOT NULL THEN ST_Length(trajectory)::real
                       ELSE 0::real
                   END AS path_length_m,
                   ST_Distance(first_pos, last_pos)::real AS displacement_m,
                   (ST_X(last_pos) - ST_X(first_pos))::real AS move_dx,
                   (ST_Y(last_pos) - ST_Y(first_pos))::real AS move_dy,
                   CASE
                       WHEN last_safe_center IS NULL THEN NULL
                       ELSE (ST_X(last_safe_center) - ST_X(first_pos))::real
                   END AS safe_dx,
                   CASE
                       WHEN last_safe_center IS NULL THEN NULL
                       ELSE (ST_Y(last_safe_center) - ST_Y(first_pos))::real
                   END AS safe_dy
              FROM summary s
        ),
        rot AS (
            SELECT g.match_id,
                   g.team_id,
                   g.phase_idx,
                   min(sn.snapshot_t) AS rot_t
              FROM geom g
              JOIN snaps sn
                ON sn.team_id = g.team_id
               AND sn.phase_idx = g.phase_idx
             WHERE g.displacement_m > 1.0
               AND ST_Distance(sn.pos2d, g.first_pos) >= 0.3 * g.displacement_m
             GROUP BY g.match_id, g.team_id, g.phase_idx
        )
        INSERT INTO stdb.team_phase_trajectory (
            match_id, team_id, phase_idx, waypoints, phase_start_s, phase_end_s,
            phase_duration_s, any_alive, trajectory, path_length_m, displacement_m,
            sinuosity, avg_speed_mps, rotation_timing, safe_alignment,
            zone_entry_time, outside_safe_duration_s, boundary_distance_slope
        )
        SELECT g.match_id,
               g.team_id,
               g.phase_idx,
               g.waypoints,
               g.phase_start_s,
               g.phase_end_s,
               g.phase_duration_s,
               g.any_alive,
               g.trajectory,
               g.path_length_m,
               g.displacement_m,
               CASE
                   WHEN g.displacement_m > 1.0 THEN g.path_length_m / g.displacement_m
                   ELSE NULL
               END::real AS sinuosity,
               CASE
                   WHEN g.phase_duration_s > 0 THEN g.path_length_m / g.phase_duration_s
                   ELSE NULL
               END::real AS avg_speed_mps,
               CASE
                   WHEN g.phase_duration_s > 0 AND r.rot_t IS NOT NULL
                   THEN greatest(0, least(1, (r.rot_t - g.{WIN_START_COLUMN}) / g.phase_duration_s))
                   ELSE NULL
               END::real AS rotation_timing,
               CASE
                   WHEN sqrt(g.move_dx*g.move_dx + g.move_dy*g.move_dy) > 1.0
                    AND sqrt(g.safe_dx*g.safe_dx + g.safe_dy*g.safe_dy) > 1.0
                   THEN (
                       (g.move_dx*g.safe_dx + g.move_dy*g.safe_dy)
                       / (
                           sqrt(g.move_dx*g.move_dx + g.move_dy*g.move_dy)
                           * sqrt(g.safe_dx*g.safe_dx + g.safe_dy*g.safe_dy)
                       )
                   )
                   ELSE NULL
               END::real AS safe_alignment,
               CASE
                   WHEN g.phase_duration_s > 0 AND g.first_safe_t IS NOT NULL
                   THEN greatest(0, least(1, (g.first_safe_t - g.{WIN_START_COLUMN}) / g.phase_duration_s))
                   ELSE NULL
               END::real AS zone_entry_time,
               g.outside_safe_duration_s,
               g.boundary_distance_slope
          FROM geom g
     LEFT JOIN rot r
            ON r.match_id = g.match_id
           AND r.team_id = g.team_id
           AND r.phase_idx = g.phase_idx
        """,
        (match_id, match_id, SNAPSHOT_INTERVAL),
    )
    trajectory_rows = cur.rowcount

    cur.execute(
        f"""
        WITH phase_bounds AS (
            SELECT match_id, phase_idx, phase_start_s, phase_end_s, phase_duration_s,
                   {win_cols}
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
               AND phase_idx > 0
        ),
        snaps AS (
            SELECT mo.team_id,
                   pb.phase_idx,
                   mo.snapshot_t,
                   ST_Force2D(mo.pos) AS pos2d,
                   dr.region AS safe_region,
                   CASE
                       WHEN dr.region IS NULL THEN NULL
                       ELSE ST_Distance(ST_Force2D(mo.pos), ST_Boundary(dr.region))
                   END AS dist_to_boundary,
                   CASE
                       WHEN dr.region IS NULL THEN NULL
                       ELSE ST_Contains(dr.region, ST_Force2D(mo.pos))
                   END AS in_safe
              FROM phase_bounds pb
              JOIN stdb.moving_objects mo
                ON mo.match_id = pb.match_id
               AND mo.agent_kind = 'team'
               -- Fix B + one clock: grid window on the game clock, capped at the end
               AND mo.snapshot_t BETWEEN pb.{WIN_START_COLUMN} AND pb.{CUT_COLUMN}
               AND mo.is_alive
         LEFT JOIN stdb.dynamic_regions dr
                ON dr.match_id = mo.match_id
               AND dr.region_kind = 'safe'
               AND dr.snapshot_t = mo.snapshot_t
        ),
        prox AS (
            SELECT tp.snapshot_t, tp.team_a AS team_id, tp.team_b AS other_team, tp.distance_m
              FROM stdb.team_proximity tp
             WHERE tp.match_id = %s
            UNION ALL
            SELECT tp.snapshot_t, tp.team_b AS team_id, tp.team_a AS other_team, tp.distance_m
              FROM stdb.team_proximity tp
             WHERE tp.match_id = %s
        ),
        prox_agg AS (
            SELECT s.team_id,
                   s.phase_idx,
                   s.snapshot_t,
                   count(p.other_team) FILTER (WHERE p.distance_m < 100.0)::integer
                       AS n_enemy_100m,
                   min(p.distance_m)::real AS nearest_enemy_m
              FROM snaps s
         LEFT JOIN prox p
                ON p.team_id = s.team_id
               AND p.snapshot_t = s.snapshot_t
             GROUP BY s.team_id, s.phase_idx, s.snapshot_t
        ),
        damage AS (
            SELECT victim_team AS team_id,
                   pb.phase_idx,
                   sum(coalesce(damage, 0))::real AS recent_damage_taken
              FROM stdb.spatial_events e
              JOIN phase_bounds pb
                ON pb.match_id = e.match_id
               AND e.event_t BETWEEN pb.phase_start_s AND pb.phase_end_s
             WHERE e.match_id = %s
               AND e.event_kind = 'damage'
               AND e.victim_team IS NOT NULL
             GROUP BY victim_team, pb.phase_idx
        ),
        combat_points AS (
            SELECT e.event_id,
                   e.event_t,
                   -- events are on the LogMatchStart clock, the grid on the game clock
                   floor((e.event_t - {dsql}) / %s)::integer * %s::integer AS snapshot_t,
                   COALESCE(ST_Force2D(e.victim_pos), ST_Force2D(e.attacker_pos)) AS event_pos
              FROM stdb.spatial_events e
             WHERE e.match_id = %s
               AND e.event_kind IN ('kill', 'damage', 'groggy')
               AND COALESCE(e.victim_pos, e.attacker_pos) IS NOT NULL
        ),
        combat_pass AS (
            -- LEAK FIX: the floor() bucket can pull an event from the NEXT phase
            -- into this phase's last snapshot. Clamp event_t to the phase window
            -- so a cut-phase feature never sees post-cut combat.
            SELECT s.team_id,
                   s.phase_idx,
                   count(DISTINCT cp.event_id)::integer AS combat_zone_pass_count
              FROM snaps s
              JOIN phase_bounds pb
                ON pb.phase_idx = s.phase_idx
              JOIN combat_points cp
                ON cp.snapshot_t = s.snapshot_t
               AND cp.event_t BETWEEN pb.phase_start_s AND pb.phase_end_s
               AND ST_DWithin(s.pos2d, cp.event_pos, 200.0)
             GROUP BY s.team_id, s.phase_idx
        )
        INSERT INTO stdb.team_phase_exposure (
            match_id, team_id, phase_idx, mean_dist_to_boundary_m,
            min_dist_to_boundary_m, outside_safe_ratio, avg_enemy_density_100m,
            min_nearest_enemy_m, path_near_enemy_ratio, recent_damage_taken,
            combat_zone_pass_count
        )
        SELECT %s::text AS match_id,
               s.team_id,
               s.phase_idx,
               avg(s.dist_to_boundary)::real AS mean_dist_to_boundary_m,
               min(s.dist_to_boundary)::real AS min_dist_to_boundary_m,
               avg(CASE WHEN s.in_safe = FALSE THEN 1.0 ELSE 0.0 END)::real
                   AS outside_safe_ratio,
               avg(coalesce(pa.n_enemy_100m, 0))::real AS avg_enemy_density_100m,
               min(pa.nearest_enemy_m)::real AS min_nearest_enemy_m,
               avg(CASE WHEN coalesce(pa.n_enemy_100m, 0) > 0 THEN 1.0 ELSE 0.0 END)::real
                   AS path_near_enemy_ratio,
               coalesce(max(d.recent_damage_taken), 0)::real AS recent_damage_taken,
               coalesce(max(cp.combat_zone_pass_count), 0)::integer AS combat_zone_pass_count
          FROM snaps s
     LEFT JOIN prox_agg pa
            ON pa.team_id = s.team_id
           AND pa.phase_idx = s.phase_idx
           AND pa.snapshot_t = s.snapshot_t
     LEFT JOIN damage d
            ON d.team_id = s.team_id
           AND d.phase_idx = s.phase_idx
     LEFT JOIN combat_pass cp
            ON cp.team_id = s.team_id
           AND cp.phase_idx = s.phase_idx
         GROUP BY s.team_id, s.phase_idx
        """,
        (
            match_id,
            match_id,
            match_id,
            match_id,
            SNAPSHOT_INTERVAL,
            SNAPSHOT_INTERVAL,
            match_id,
            match_id,
        ),
    )
    exposure_rows = cur.rowcount
    return {"trajectory": int(trajectory_rows), "exposure": int(exposure_rows)}


def select_match_ids(*, limit: int | None, refresh: bool, match_id: str | None) -> list[str]:
    if match_id:
        return [match_id]

    sql = """
        SELECT DISTINCT match_id
          FROM stdb.team_phase_labels
    """
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id
              FROM target t
             WHERE NOT EXISTS (
                   SELECT 1 FROM stdb.team_phase_trajectory tr
                    WHERE tr.match_id = t.match_id
             )
                OR NOT EXISTS (
                   SELECT 1 FROM stdb.team_phase_exposure ex
                    WHERE ex.match_id = t.match_id
             )
             ORDER BY t.match_id
        """
    else:
        sql += " ORDER BY match_id"

    params: tuple[object, ...]
    if limit:
        sql += " LIMIT %s"
        params = (limit,)
    else:
        params = ()

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [r[0] for r in cur.fetchall()]


def status() -> dict[str, int]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_labels")
            total = int(cur.fetchone()[0])
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_trajectory")
            trajectory_done = int(cur.fetchone()[0])
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_exposure")
            exposure_done = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT count(*)
                  FROM (
                        SELECT DISTINCT match_id FROM stdb.team_phase_labels
                       ) m
                 WHERE EXISTS (
                        SELECT 1 FROM stdb.team_phase_trajectory tr
                         WHERE tr.match_id = m.match_id
                       )
                   AND EXISTS (
                        SELECT 1 FROM stdb.team_phase_exposure ex
                         WHERE ex.match_id = m.match_id
                       )
                """
            )
            both_done = int(cur.fetchone()[0])
    return {
        "total_matches": total,
        "trajectory_done": trajectory_done,
        "exposure_done": exposure_done,
        "both_done": both_done,
        "remaining": total - both_done,
    }


def build_batch(match_ids: list[str]) -> dict[str, object]:
    started = time.time()
    ok = 0
    failed = 0
    failures: list[dict[str, object]] = []

    for idx, match_id in enumerate(match_ids, start=1):
        t0 = time.time()
        conn = connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    counts = build_match(cur, match_id)
            ok += 1
            logger.info(
                "[%s/%s] OK %s elapsed=%.2fs rows=%s",
                idx,
                len(match_ids),
                match_id,
                time.time() - t0,
                counts,
            )
        except Exception as exc:  # keep large resumable batches moving
            conn.rollback()
            failed += 1
            failures.append({"match_id": match_id, "error": repr(exc)})
            logger.exception("[%s/%s] FAILED %s", idx, len(match_ids), match_id)
        finally:
            conn.close()

    return {
        "ok": ok,
        "failed": failed,
        "failures": failures,
        "elapsed_s": round(time.time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--match-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.apply_schema:
        apply_schema()

    logger.info("status before: %s", status())

    if args.status:
        return 0

    match_ids = select_match_ids(
        limit=args.limit,
        refresh=args.refresh,
        match_id=args.match_id,
    )
    logger.info("selected %s matches", len(match_ids))
    result = build_batch(match_ids)
    logger.info("batch result: %s", result)
    logger.info("status after: %s", status())
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
