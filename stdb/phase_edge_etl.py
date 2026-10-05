"""
Phase-level edge cache builder for DynamicSTGNN.

Materializes graph edges in a model-friendly long format:

    stdb.team_phase_edge_cache
        (match_id, phase_idx, edge_type, src_team_id, dst_team_id, weight, ...)

Edge types:
    proximity  : symmetric phase-level distance edge from stdb.team_proximity
    combat     : directed attacker -> victim combat-intensity edge
    copressure : symmetric edge when teams are close while both under zone pressure

The script is match-scoped and resumable.

Fix B + single clock (2026-10-05): the proximity and co-pressure edges aggregate the 10 s grid
(stdb.team_proximity / stdb.moving_objects) that lives on the game clock, while the phase bounds are
LogMatchStart-clock numbers.  Their grid window is therefore
``[phase_start_s - delta, phase_end_s - delta - SNAPSHOT_INTERVAL]`` (stdb/causal_cut.py).  Combat
edges use exact event timestamps on the LogMatchStart clock and keep the ``[phase_start_s,
phase_end_s]`` window unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from stdb.causal_cut import (
    CUT_COLUMN,
    WIN_START_COLUMN,
    fetch_clock_delta,
    sql_delta_literal,
    sql_window_columns,
)
from stdb.conn import connect

logger = logging.getLogger("stdb.phase_edge_etl")


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_phase_edge_cache (
    match_id             TEXT    NOT NULL,
    phase_idx            INTEGER NOT NULL,
    edge_type            TEXT    NOT NULL,
    src_team_id          INTEGER NOT NULL,
    dst_team_id          INTEGER NOT NULL,
    weight               REAL    NOT NULL,
    support_count        INTEGER NOT NULL DEFAULT 0,
    avg_distance_m       REAL,
    ratio                REAL,
    damage_sum           REAL,
    event_count          INTEGER NOT NULL DEFAULT 0,
    kill_count           INTEGER NOT NULL DEFAULT 0,
    groggy_count         INTEGER NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, phase_idx, edge_type, src_team_id, dst_team_id),
    CHECK (src_team_id <> dst_team_id),
    CHECK (edge_type IN ('proximity', 'combat', 'copressure'))
);

CREATE INDEX IF NOT EXISTS ix_tpec_match_phase_type
    ON stdb.team_phase_edge_cache (match_id, phase_idx, edge_type);
CREATE INDEX IF NOT EXISTS ix_tpec_src
    ON stdb.team_phase_edge_cache (match_id, src_team_id);
CREATE INDEX IF NOT EXISTS ix_tpec_dst
    ON stdb.team_phase_edge_cache (match_id, dst_team_id);
"""


def apply_schema() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def build_match(cur, match_id: str) -> dict[str, int]:
    delta_s = fetch_clock_delta(cur, match_id)
    win_cols = sql_window_columns(sql_delta_literal(delta_s))
    cur.execute("DELETE FROM stdb.team_phase_edge_cache WHERE match_id = %s", (match_id,))

    cur.execute(
        f"""
        WITH phase_bounds AS (
            SELECT match_id, phase_idx, phase_start_s, phase_end_s, phase_duration_s,
                   {win_cols}
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
               AND phase_idx > 0
        ),
        pair_phase AS (
            SELECT pb.phase_idx,
                   tp.team_a,
                   tp.team_b,
                   avg(tp.distance_m)::real AS avg_distance_m,
                   count(*)::integer AS support_count
              FROM phase_bounds pb
              JOIN stdb.team_proximity tp
                ON tp.match_id = pb.match_id
               -- Fix B + one clock: grid window on the game clock, capped at the end
               AND tp.snapshot_t BETWEEN pb.{WIN_START_COLUMN} AND pb.{CUT_COLUMN}
             GROUP BY pb.phase_idx, tp.team_a, tp.team_b
        ),
        directed AS (
            SELECT phase_idx, team_a AS src_team_id, team_b AS dst_team_id,
                   avg_distance_m, support_count
              FROM pair_phase
            UNION ALL
            SELECT phase_idx, team_b AS src_team_id, team_a AS dst_team_id,
                   avg_distance_m, support_count
              FROM pair_phase
        )
        INSERT INTO stdb.team_phase_edge_cache (
            match_id, phase_idx, edge_type, src_team_id, dst_team_id, weight,
            support_count, avg_distance_m
        )
        SELECT %s::text AS match_id,
               phase_idx,
               'proximity' AS edge_type,
               src_team_id,
               dst_team_id,
               exp(-avg_distance_m / 200.0)::real AS weight,
               support_count,
               avg_distance_m
          FROM directed
         WHERE src_team_id <> dst_team_id
           AND avg_distance_m IS NOT NULL
        """,
        (match_id, match_id),
    )
    proximity_rows = int(cur.rowcount)

    cur.execute(
        """
        WITH phase_bounds AS (
            SELECT match_id, phase_idx, phase_start_s, phase_end_s
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
               AND phase_idx > 0
        ),
        events AS (
            SELECT pb.phase_idx,
                   e.attacker_team AS src_team_id,
                   e.victim_team AS dst_team_id,
                   sum(CASE WHEN e.event_kind = 'damage' THEN coalesce(e.damage, 0) ELSE 0 END)::real
                       AS damage_sum,
                   count(*)::integer AS event_count,
                   count(*) FILTER (WHERE e.event_kind = 'kill')::integer AS kill_count,
                   count(*) FILTER (WHERE e.event_kind = 'groggy')::integer AS groggy_count
              FROM stdb.spatial_events e
              JOIN phase_bounds pb
                ON pb.match_id = e.match_id
               AND e.event_t BETWEEN pb.phase_start_s AND pb.phase_end_s
             WHERE e.match_id = %s
               AND e.event_kind IN ('damage', 'kill', 'groggy')
               AND e.attacker_team IS NOT NULL
               AND e.victim_team IS NOT NULL
               AND e.attacker_team <> e.victim_team
             GROUP BY pb.phase_idx, e.attacker_team, e.victim_team
        )
        INSERT INTO stdb.team_phase_edge_cache (
            match_id, phase_idx, edge_type, src_team_id, dst_team_id, weight,
            support_count, damage_sum, event_count, kill_count, groggy_count
        )
        SELECT %s::text AS match_id,
               phase_idx,
               'combat' AS edge_type,
               src_team_id,
               dst_team_id,
               ln(1.0 + greatest(0, coalesce(damage_sum, 0))
                    + 100.0 * kill_count
                    + 50.0 * groggy_count)::real AS weight,
               event_count AS support_count,
               coalesce(damage_sum, 0)::real AS damage_sum,
               event_count,
               kill_count,
               groggy_count
          FROM events
         WHERE event_count > 0
           AND (
                greatest(0, coalesce(damage_sum, 0))
                + 100.0 * kill_count
                + 50.0 * groggy_count
           ) > 0
        """,
        (match_id, match_id, match_id),
    )
    combat_rows = int(cur.rowcount)

    cur.execute(
        f"""
        WITH phase_bounds AS (
            SELECT match_id, phase_idx, phase_start_s, phase_end_s,
                   {win_cols}
              FROM stdb.v_match_phase_bounds
             WHERE match_id = %s
               AND phase_idx > 0
        ),
        pressure AS (
            SELECT mo.team_id,
                   mo.snapshot_t,
                   CASE
                       WHEN dr.region IS NULL THEN FALSE
                       WHEN NOT ST_Contains(dr.region, ST_Force2D(mo.pos)) THEN TRUE
                       WHEN ST_Distance(ST_Force2D(mo.pos), ST_Boundary(dr.region)) <= 150.0 THEN TRUE
                       ELSE FALSE
                   END AS under_pressure
              FROM stdb.moving_objects mo
         LEFT JOIN stdb.dynamic_regions dr
                ON dr.match_id = mo.match_id
               AND dr.region_kind = 'safe'
               AND dr.snapshot_t = mo.snapshot_t
             WHERE mo.match_id = %s
               AND mo.agent_kind = 'team'
               AND mo.is_alive
        ),
        pair_phase AS (
            SELECT pb.phase_idx,
                   tp.team_a,
                   tp.team_b,
                   count(*)::integer AS support_count,
                   count(*) FILTER (
                       WHERE tp.distance_m <= 200.0
                         AND pa.under_pressure
                         AND pbp.under_pressure
                   )::integer AS copress_count,
                   avg(tp.distance_m) FILTER (
                       WHERE tp.distance_m <= 200.0
                         AND pa.under_pressure
                         AND pbp.under_pressure
                   )::real AS avg_copress_distance_m
              FROM phase_bounds pb
              JOIN stdb.team_proximity tp
                ON tp.match_id = pb.match_id
               -- Fix B + one clock: grid window on the game clock, capped at the end
               AND tp.snapshot_t BETWEEN pb.{WIN_START_COLUMN} AND pb.{CUT_COLUMN}
              JOIN pressure pa
                ON pa.team_id = tp.team_a
               AND pa.snapshot_t = tp.snapshot_t
              JOIN pressure pbp
                ON pbp.team_id = tp.team_b
               AND pbp.snapshot_t = tp.snapshot_t
             GROUP BY pb.phase_idx, tp.team_a, tp.team_b
        ),
        weighted AS (
            SELECT phase_idx,
                   team_a,
                   team_b,
                   support_count,
                   copress_count,
                   avg_copress_distance_m,
                   CASE
                       WHEN support_count > 0 THEN copress_count::real / support_count
                       ELSE 0::real
                   END AS ratio,
                   CASE
                       WHEN avg_copress_distance_m IS NULL THEN 0::real
                       ELSE exp(-avg_copress_distance_m / 200.0)
                   END AS distance_weight
              FROM pair_phase
             WHERE copress_count > 0
        ),
        directed AS (
            SELECT phase_idx, team_a AS src_team_id, team_b AS dst_team_id,
                   support_count, copress_count, avg_copress_distance_m,
                   ratio, distance_weight
              FROM weighted
            UNION ALL
            SELECT phase_idx, team_b AS src_team_id, team_a AS dst_team_id,
                   support_count, copress_count, avg_copress_distance_m,
                   ratio, distance_weight
              FROM weighted
        )
        INSERT INTO stdb.team_phase_edge_cache (
            match_id, phase_idx, edge_type, src_team_id, dst_team_id, weight,
            support_count, avg_distance_m, ratio
        )
        SELECT %s::text AS match_id,
               phase_idx,
               'copressure' AS edge_type,
               src_team_id,
               dst_team_id,
               (ratio * distance_weight)::real AS weight,
               copress_count AS support_count,
               avg_copress_distance_m,
               ratio
          FROM directed
         WHERE src_team_id <> dst_team_id
        """,
        (match_id, match_id, match_id),
    )
    copressure_rows = int(cur.rowcount)

    return {
        "proximity": proximity_rows,
        "combat": combat_rows,
        "copressure": copressure_rows,
    }


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
                   SELECT 1
                     FROM stdb.team_phase_edge_cache e
                    WHERE e.match_id = t.match_id
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


def status() -> dict[str, int]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_labels")
            total = int(cur.fetchone()[0])
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_edge_cache")
            edge_done = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT edge_type, count(*), count(DISTINCT match_id)
                  FROM stdb.team_phase_edge_cache
                 GROUP BY edge_type
                 ORDER BY edge_type
                """
            )
            by_type = {
                row[0]: {"rows": int(row[1]), "matches": int(row[2])}
                for row in cur.fetchall()
            }
    return {
        "total_matches": total,
        "edge_done": edge_done,
        "remaining": total - edge_done,
        "by_type": by_type,
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
