"""
Snapshot-resolution (10s) team node-feature builder for the multi-resolution
ST-GNN (MR-STGNN Tier-S). Materializes one row per alive (match, team,
snapshot_t):

    stdb.team_snapshot_features
        (match_id, team_id, snapshot_t, phase_idx, <node features>)

Why materialize (vs on-the-fly): full DB is ~42.7M rows but lazy per-batch
fetch during training would re-run the PostGIS zone joins every epoch over
17k matches — prohibitive. Built once, fetched fast.

Leakage: every column is an INSTANTANEOUS or PAST-only quantity (lag over
snapshot_t). No future look. The cumulative cut (snapshot_t <= phase_end_s(k))
is applied at FETCH time in stdb.queries, not here. The grid-aligned join key
is snapshot_t (NEVER floor(event_t/10)*10) so no cross-cut event bleed.

Node features (compact — the snapshot encoder learns dynamics from the
sequence, so we store instantaneous state + one kinematic hint):
    pos_x/y/z              raw team centroid (z = altitude)
    rel_zone_x/y           (pos - safe_center) / safe_radius  (zone-relative)
    speed_mps              ||pos_t - pos_{t-1}|| / dt   (past-only, lag)
    dist_blue_boundary_m   signed distance to current blue-zone boundary (+ inside)
    inside_blue            bool
    dist_target_center_m   distance to white target (poison) center, NULL if none
    mean_health/min_health team mean / min player health  (player rows)
    alive_n                alive players in team at snapshot
    vehicle_ratio          fraction of players in a vehicle

team rows carry NULL health (etl.py team aggregation), so health/alive/vehicle
come from agent_kind='player' rows aggregated per (team, snapshot).

match-scoped, resumable — mirrors stdb/habitat_feature_etl.py.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
import time

from pubg_lab.data.stdb.conn import connect

logger = logging.getLogger("stdb.snapshot_feature_etl")


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_snapshot_features (
    match_id                TEXT    NOT NULL,
    team_id                 INTEGER NOT NULL,
    snapshot_t              INTEGER NOT NULL,
    phase_idx               INTEGER,

    pos_x                   REAL,
    pos_y                   REAL,
    pos_z                   REAL,
    rel_zone_x              REAL,
    rel_zone_y              REAL,
    speed_mps               REAL,
    dist_blue_boundary_m    REAL,
    inside_blue             BOOLEAN,
    dist_target_center_m    REAL,
    mean_health             REAL,
    min_health              REAL,
    alive_n                 INTEGER,
    vehicle_ratio           REAL,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, snapshot_t)
);

CREATE INDEX IF NOT EXISTS ix_tsf_match_snap
    ON stdb.team_snapshot_features (match_id, snapshot_t);
CREATE INDEX IF NOT EXISTS ix_tsf_match_phase
    ON stdb.team_snapshot_features (match_id, phase_idx);
"""


BUILD_SQL = """
WITH team_pos AS (
    SELECT team_id, snapshot_t,
           ST_Force2D(pos) AS p2, ST_X(pos) AS x, ST_Y(pos) AS y, ST_Z(pos) AS z
      FROM stdb.moving_objects
     WHERE match_id = %(match_id)s AND agent_kind = 'team' AND is_alive
),
player_agg AS (
    SELECT team_id, snapshot_t,
           avg(health)::real             AS mean_health,
           min(health)::real             AS min_health,
           count(*) FILTER (WHERE is_alive)::int AS alive_n,
           avg(in_vehicle::int)::real    AS vehicle_ratio
      FROM stdb.moving_objects
     WHERE match_id = %(match_id)s AND agent_kind = 'player'
     GROUP BY 1, 2
),
safe AS (
    SELECT snapshot_t, region, ST_Force2D(center_pos) AS c, radius_m, phase_idx
      FROM stdb.dynamic_regions
     WHERE match_id = %(match_id)s AND region_kind = 'safe'
),
target AS (
    SELECT snapshot_t, ST_Force2D(center_pos) AS c
      FROM stdb.dynamic_regions
     WHERE match_id = %(match_id)s AND region_kind = 'poison'
),
joined AS (
    SELECT tp.team_id, tp.snapshot_t, tp.p2, tp.x, tp.y, tp.z,
           s.region AS blue, s.c AS safe_c, s.radius_m AS safe_r, s.phase_idx,
           t.c AS target_c
      FROM team_pos tp
 LEFT JOIN safe s   ON s.snapshot_t = tp.snapshot_t
 LEFT JOIN target t ON t.snapshot_t = tp.snapshot_t
)
INSERT INTO stdb.team_snapshot_features (
    match_id, team_id, snapshot_t, phase_idx,
    pos_x, pos_y, pos_z, rel_zone_x, rel_zone_y, speed_mps,
    dist_blue_boundary_m, inside_blue, dist_target_center_m,
    mean_health, min_health, alive_n, vehicle_ratio
)
SELECT
    %(match_id)s::text,
    j.team_id,
    j.snapshot_t,
    j.phase_idx,
    j.x, j.y, j.z,
    CASE WHEN j.safe_c IS NULL OR j.safe_r IS NULL OR j.safe_r = 0 THEN NULL
         ELSE ((j.x - ST_X(j.safe_c)) / j.safe_r)::real END,
    CASE WHEN j.safe_c IS NULL OR j.safe_r IS NULL OR j.safe_r = 0 THEN NULL
         ELSE ((j.y - ST_Y(j.safe_c)) / j.safe_r)::real END,
    -- past-only instantaneous speed; first snapshot of a team -> NULL
    (ST_Distance(j.p2, lag(j.p2) OVER w)
        / NULLIF(j.snapshot_t - lag(j.snapshot_t) OVER w, 0))::real,
    -- signed: + inside (cushion), - outside (exposure)
    CASE WHEN j.blue IS NULL THEN NULL
         WHEN ST_Contains(j.blue, j.p2)
            THEN  ST_Distance(j.p2, ST_Boundary(j.blue))::real
         ELSE    -ST_Distance(j.p2, ST_Boundary(j.blue))::real END,
    CASE WHEN j.blue IS NULL THEN NULL ELSE ST_Contains(j.blue, j.p2) END,
    CASE WHEN j.target_c IS NULL THEN NULL
         ELSE ST_Distance(j.p2, j.target_c)::real END,
    pa.mean_health, pa.min_health, pa.alive_n, pa.vehicle_ratio
FROM joined j
LEFT JOIN player_agg pa
       ON pa.team_id = j.team_id AND pa.snapshot_t = j.snapshot_t
WINDOW w AS (PARTITION BY j.team_id ORDER BY j.snapshot_t)
"""


def apply_schema() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def build_match(cur, match_id: str) -> int:
    cur.execute("DELETE FROM stdb.team_snapshot_features WHERE match_id = %s", (match_id,))
    cur.execute(BUILD_SQL, {"match_id": match_id})
    return int(cur.rowcount)


def select_match_ids(
    *,
    limit: int | None,
    refresh: bool,
    match_id: str | None,
    shards: int = 1,
    shard_index: int = 0,
) -> list[str]:
    if match_id:
        return [match_id]
    if shards < 1:
        raise ValueError("--shards must be >= 1")
    if shard_index < 0 or shard_index >= shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shards")

    sql = "SELECT DISTINCT match_id FROM stdb.team_phase_labels"
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id
              FROM target t
             WHERE NOT EXISTS (
                   SELECT 1 FROM stdb.team_snapshot_features f
                    WHERE f.match_id = t.match_id
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
            match_ids = [r[0] for r in cur.fetchall()]
    if shards > 1:
        match_ids = [
            mid
            for mid in match_ids
            if int(hashlib.md5(mid.encode("utf-8")).hexdigest()[:8], 16) % shards == shard_index
        ]
    return match_ids


def status() -> dict[str, object]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_labels")
            total = int(cur.fetchone()[0])
            cur.execute(
                "SELECT count(*), count(DISTINCT match_id) FROM stdb.team_snapshot_features"
            )
            rows, done = cur.fetchone()
    return {
        "label_matches": total,
        "snapshot_done": int(done),
        "remaining": total - int(done),
        "snapshot_rows": int(rows),
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
                    rows = build_match(cur, match_id)
            ok += 1
            if idx % 200 == 0 or idx == len(match_ids):
                logger.info(
                    "[%s/%s] OK %s elapsed=%.2fs rows=%s",
                    idx,
                    len(match_ids),
                    match_id,
                    time.time() - t0,
                    rows,
                )
        except Exception as exc:
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
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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
        shards=args.shards,
        shard_index=args.shard_index,
    )
    logger.info("selected %s matches (shard %s/%s)", len(match_ids), args.shard_index, args.shards)
    result = build_batch(match_ids)
    logger.info("batch result: %s", result)
    logger.info("status after: %s", status())
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
