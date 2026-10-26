"""
F0 official-raw extras (Tier-1: stdb-base derivable) — FEATURE_DESIGN_V2 §4.

Builds stdb.team_phase_f0_extra with the raw-aggregate columns that the F0
baseline needs but the long view does not yet expose, computed from stdb base
objects (no raw telem re-aggregation, no base rebuild):

    zone centers   safe/poison center x/y at phase start/end   (dynamic_regions)
    team centroid  team centroid x/y at phase start/end        (moving_objects)
    raw kinematics team_displacement_raw_m, team_avg_speed_raw_mps (centroid)

The remaining F0 raw columns (assists_count, suicide_count from telem_kills;
landing_* from telem_parachute_landing) require event_time->elapsed mapping via
matches.started_at and are built in a second pass (see f0_landing_combat_etl,
TODO). The F0 view joins base + this table + (pass 2).

Leakage: all columns are phase-local observed state (start/end snapshot within
the phase). Centroid uses alive team rows; zone center is public game state.
match-scoped, resumable — mirrors stdb/habitat_feature_etl.py.

Fix B + single clock (2026-10-05): the grid is on the game clock, the phase bounds on the
LogMatchStart clock, so every grid read uses the window
``[phase_start_s - delta, phase_end_s - delta - SNAPSHOT_INTERVAL]`` on the game clock
(stdb/causal_cut.py).  The ``*_end`` columns (zone centres and team centroid at the end of the phase)
are the last grid point of that window; the ``*_start`` columns its first grid point.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[5]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import hashlib
import logging
import time

from pubg_lab.data.stdb.temporal.causal_cut import (
    CUT_COLUMN,
    WIN_START_COLUMN,
    fetch_clock_delta,
    sql_window_columns,
)
from pubg_lab.data.stdb.connection.conn import connect

logger = logging.getLogger("pubg_lab.data.stdb.pipelines.f0_feature_etl")


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_phase_f0_extra (
    match_id                  TEXT    NOT NULL,
    team_id                   INTEGER NOT NULL,
    phase_idx                 INTEGER NOT NULL,

    safe_center_x_start       REAL,
    safe_center_y_start       REAL,
    safe_center_x_end         REAL,
    safe_center_y_end         REAL,
    poison_center_x_start     REAL,
    poison_center_y_start     REAL,
    poison_center_x_end       REAL,
    poison_center_y_end       REAL,

    team_centroid_x_start     REAL,
    team_centroid_y_start     REAL,
    team_centroid_x_end       REAL,
    team_centroid_y_end       REAL,
    team_displacement_raw_m   REAL,
    team_avg_speed_raw_mps    REAL,

    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);

CREATE INDEX IF NOT EXISTS ix_tpf0_match_phase
    ON stdb.team_phase_f0_extra (match_id, phase_idx);
"""


BUILD_SQL = f"""
WITH bounds AS (
    SELECT phase_idx, phase_start_s, phase_end_s, phase_duration_s,
           {sql_window_columns("%(delta_s)s")}
      FROM stdb.v_match_phase_bounds
     WHERE match_id = %(match_id)s AND phase_idx > 0
),
safe AS (
    -- Fix B + one clock: grid window on the game clock, capped at the end
    SELECT b.phase_idx,
           (array_agg(ST_X(dr.center_pos) ORDER BY dr.snapshot_t))[1]      AS cx_s,
           (array_agg(ST_Y(dr.center_pos) ORDER BY dr.snapshot_t))[1]      AS cy_s,
           (array_agg(ST_X(dr.center_pos) ORDER BY dr.snapshot_t DESC))[1] AS cx_e,
           (array_agg(ST_Y(dr.center_pos) ORDER BY dr.snapshot_t DESC))[1] AS cy_e
      FROM bounds b
      JOIN stdb.dynamic_regions dr
        ON dr.match_id = %(match_id)s AND dr.region_kind = 'safe'
       AND dr.snapshot_t BETWEEN b.{WIN_START_COLUMN} AND b.{CUT_COLUMN}
     GROUP BY b.phase_idx
),
poison AS (
    -- Fix B + one clock: grid window on the game clock, capped at the end
    SELECT b.phase_idx,
           (array_agg(ST_X(dr.center_pos) ORDER BY dr.snapshot_t))[1]      AS cx_s,
           (array_agg(ST_Y(dr.center_pos) ORDER BY dr.snapshot_t))[1]      AS cy_s,
           (array_agg(ST_X(dr.center_pos) ORDER BY dr.snapshot_t DESC))[1] AS cx_e,
           (array_agg(ST_Y(dr.center_pos) ORDER BY dr.snapshot_t DESC))[1] AS cy_e
      FROM bounds b
      JOIN stdb.dynamic_regions dr
        ON dr.match_id = %(match_id)s AND dr.region_kind = 'poison'
       AND dr.snapshot_t BETWEEN b.{WIN_START_COLUMN} AND b.{CUT_COLUMN}
     GROUP BY b.phase_idx
),
centroid AS (
    -- Fix B: the team centroid at phase end = last alive grid point <= phase_end_s - SNAPSHOT_INTERVAL
    SELECT b.phase_idx, mo.team_id,
           (array_agg(ST_X(ST_Force2D(mo.pos)) ORDER BY mo.snapshot_t))[1]      AS cx_s,
           (array_agg(ST_Y(ST_Force2D(mo.pos)) ORDER BY mo.snapshot_t))[1]      AS cy_s,
           (array_agg(ST_X(ST_Force2D(mo.pos)) ORDER BY mo.snapshot_t DESC))[1] AS cx_e,
           (array_agg(ST_Y(ST_Force2D(mo.pos)) ORDER BY mo.snapshot_t DESC))[1] AS cy_e
      FROM bounds b
      JOIN stdb.moving_objects mo
        ON mo.match_id = %(match_id)s AND mo.agent_kind = 'team' AND mo.is_alive
       AND mo.snapshot_t BETWEEN b.{WIN_START_COLUMN} AND b.{CUT_COLUMN}
     GROUP BY b.phase_idx, mo.team_id
),
universe AS (
    SELECT t.team_id, b.phase_idx, b.phase_duration_s
      FROM stdb.teams t CROSS JOIN bounds b
     WHERE t.match_id = %(match_id)s
)
INSERT INTO stdb.team_phase_f0_extra (
    match_id, team_id, phase_idx,
    safe_center_x_start, safe_center_y_start, safe_center_x_end, safe_center_y_end,
    poison_center_x_start, poison_center_y_start, poison_center_x_end, poison_center_y_end,
    team_centroid_x_start, team_centroid_y_start, team_centroid_x_end, team_centroid_y_end,
    team_displacement_raw_m, team_avg_speed_raw_mps
)
SELECT
    %(match_id)s::text, u.team_id, u.phase_idx,
    s.cx_s, s.cy_s, s.cx_e, s.cy_e,
    p.cx_s, p.cy_s, p.cx_e, p.cy_e,
    c.cx_s, c.cy_s, c.cx_e, c.cy_e,
    sqrt(power(c.cx_e - c.cx_s, 2) + power(c.cy_e - c.cy_s, 2))::real AS displ,
    (sqrt(power(c.cx_e - c.cx_s, 2) + power(c.cy_e - c.cy_s, 2))
        / nullif(u.phase_duration_s, 0))::real AS speed
FROM universe u
LEFT JOIN safe     s ON s.phase_idx = u.phase_idx
LEFT JOIN poison   p ON p.phase_idx = u.phase_idx
LEFT JOIN centroid c ON c.phase_idx = u.phase_idx AND c.team_id = u.team_id
"""


# F0 official-raw baseline view (FEATURE_DESIGN_V2 §4). 78/85 manifest columns:
# base (match_phase_global/state/combat/loot, incl. previously-unexposed cols)
# + team_phase_f0_extra (zone centers, centroid, raw kinematics). The remaining
# 7 (assists_count, suicide_count, landing_*) need event_time->elapsed raw
# aggregation (pass 2) and are added later.
F0_VIEW_SQL = """
CREATE OR REPLACE VIEW stdb.v_team_phase_official_raw_features AS
SELECT
    t.match_id, t.team_id, b.phase_idx,
    -- meta
    m.map_name, m.game_mode, m.team_size AS match_team_size,
    g.weather_id, g.camera_view_type,
    -- phase / population
    g.phase_start_s, g.phase_end_s, g.phase_duration_s,
    g.avg_alive_teams, g.start_alive_teams, g.end_alive_teams,
    g.avg_alive_players, g.start_alive_players, g.end_alive_players,
    -- zone radius (base) + center (f0_extra)
    g.safe_radius_start_m, g.safe_radius_end_m, g.safe_radius_delta_m,
    g.poison_radius_start_m, g.poison_radius_end_m, g.poison_radius_delta_m,
    f0.safe_center_x_start, f0.safe_center_y_start, f0.safe_center_x_end, f0.safe_center_y_end,
    f0.poison_center_x_start, f0.poison_center_y_start, f0.poison_center_x_end, f0.poison_center_y_end,
    g.red_zone_active_ratio, g.black_zone_active_ratio,
    -- team state (base) + centroid/kinematics (f0_extra)
    s.team_size, s.position_rows,
    s.alive_players_start, s.alive_players_end, s.alive_players_min, s.alive_players_avg,
    s.mean_health_avg, s.min_health_min, s.health_slope,
    s.team_spread_avg_m, s.team_spread_max_m,
    s.vehicle_ratio, s.vehicle_speed_mean, s.vehicle_speed_max,
    f0.team_centroid_x_start, f0.team_centroid_y_start, f0.team_centroid_x_end, f0.team_centroid_y_end,
    f0.team_displacement_raw_m, f0.team_avg_speed_raw_mps,
    -- combat (base)
    c.damage_dealt, c.damage_taken, c.damage_events_dealt, c.damage_events_taken,
    c.groggy_dealt, c.groggy_taken, c.kills_dealt, c.kills_taken,
    c.distinct_opponents_damaged, c.distinct_opponents_damaged_by, c.combat_partner_count,
    c.first_damage_dealt_s, c.first_damage_taken_s, c.combat_initiated,
    -- loot (base)
    l.pickup_count, l.equip_count, l.use_count, l.unique_item_count,
    l.weapon_pickup_count, l.weapon_equip_count, l.ammo_pickup_count, l.armor_pickup_count,
    l.throwable_pickup_count, l.heal_use_count, l.boost_use_count,
    l.carepackage_pickup_count, l.lootbox_pickup_count,
    -- labels (carried for ablation convenience; forbidden as input by manifest)
    lab.target_defined, lab.hazard_next, lab.final_rank_norm, lab.final_top5, lab.final_top10
FROM stdb.teams t
JOIN stdb.matches m ON m.match_id = t.match_id
JOIN stdb.v_match_phase_bounds b ON b.match_id = t.match_id
LEFT JOIN stdb.match_phase_global g
       ON g.match_id = t.match_id AND g.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_state s
       ON s.match_id = t.match_id AND s.team_id = t.team_id AND s.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_combat c
       ON c.match_id = t.match_id AND c.team_id = t.team_id AND c.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_loot l
       ON l.match_id = t.match_id AND l.team_id = t.team_id AND l.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_f0_extra f0
       ON f0.match_id = t.match_id AND f0.team_id = t.team_id AND f0.phase_idx = b.phase_idx
LEFT JOIN stdb.team_phase_labels lab
       ON lab.match_id = t.match_id AND lab.team_id = t.team_id AND lab.cut_phase = b.phase_idx;
"""


def apply_schema() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def apply_view() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(F0_VIEW_SQL)


def build_match(cur, match_id: str) -> int:
    cur.execute("DELETE FROM stdb.team_phase_f0_extra WHERE match_id = %s", (match_id,))
    cur.execute(BUILD_SQL, {"match_id": match_id, "delta_s": fetch_clock_delta(cur, match_id)})
    return int(cur.rowcount)


def select_match_ids(*, limit, refresh, match_id, shards=1, shard_index=0):
    if match_id:
        return [match_id]
    sql = "SELECT DISTINCT match_id FROM stdb.team_phase_labels"
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id FROM target t
             WHERE NOT EXISTS (SELECT 1 FROM stdb.team_phase_f0_extra f WHERE f.match_id = t.match_id)
             ORDER BY t.match_id
        """
    else:
        sql += " ORDER BY match_id"
    params: tuple = ()
    if limit:
        sql += " LIMIT %s"
        params = (limit,)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            ids = [r[0] for r in cur.fetchall()]
    if shards > 1:
        ids = [
            m
            for m in ids
            if int(hashlib.md5(m.encode()).hexdigest()[:8], 16) % shards == shard_index
        ]
    return ids


def status() -> dict:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_labels")
            total = int(cur.fetchone()[0])
            cur.execute("SELECT count(*), count(DISTINCT match_id) FROM stdb.team_phase_f0_extra")
            rows, done = cur.fetchone()
    return {
        "label_matches": total,
        "f0_done": int(done),
        "remaining": total - int(done),
        "rows": int(rows),
    }


def build_batch(match_ids: list[str]) -> dict:
    started = time.time()
    ok = failed = 0
    failures = []
    for idx, mid in enumerate(match_ids, 1):
        conn = connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    rows = build_match(cur, mid)
            ok += 1
            if idx % 200 == 0 or idx == len(match_ids):
                logger.info("[%s/%s] OK %s rows=%s", idx, len(match_ids), mid, rows)
        except Exception as exc:
            conn.rollback()
            failed += 1
            failures.append({"match_id": mid, "error": repr(exc)})
            logger.exception("[%s/%s] FAILED %s", idx, len(match_ids), mid)
        finally:
            conn.close()
    return {
        "ok": ok,
        "failed": failed,
        "failures": failures,
        "elapsed_s": round(time.time() - started, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply-schema", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--match-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.apply_schema:
        apply_schema()
    logger.info("status before: %s", status())
    if args.status:
        return 0
    ids = select_match_ids(
        limit=args.limit,
        refresh=args.refresh,
        match_id=args.match_id,
        shards=args.shards,
        shard_index=args.shard_index,
    )
    logger.info("selected %s matches (shard %s/%s)", len(ids), args.shard_index, args.shards)
    result = build_batch(ids)
    logger.info("batch result: %s", result)
    logger.info("status after: %s", status())
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
