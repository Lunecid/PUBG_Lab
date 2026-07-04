"""
O6 Z-ladder spatial-affordance feature builder.

Materializes the corrected PUBG O6 Z1-Z3 proxy features:

    stdb.team_phase_z_affordance_features
        (match_id, team_id, phase_idx, z_* ...)

The table is intentionally separate from habitat features.  Habitat belongs to
OM3 as O2 x O6 pressure/habitat; this table owns only the corrected O6
spatial-affordance sensitivity block used by `om6z_spatial_affordance`.

Definitions
-----------
Z1 absolute / percentile vertical position:
    team_z_mean_m, team_z_range_m, team_z_percentile_alive

Z2 contact-gated relative height:
    contact_delta_z_mean_m, contact_delta_z_positive_ratio,
    high_ground_contact_advantage

Z3 Z2 under contraction pressure:
    high_ground_x_boundary_pressure,
    z_percentile_x_hab_crowding_pressure,
    contact_delta_z_x_outside_safe_ratio

The source signal is `stdb.team_snapshot_features.pos_z`, which is an
instantaneous team centroid altitude.  This is a PUBG telemetry proxy, not a
terrain mesh or true cover/LOS measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from stdb.conn import connect

logger = logging.getLogger("stdb.z_affordance_feature_etl")

CONTACT_DISTANCE_M = 200.0
HEIGHT_ADVANTAGE_SCALE_M = 50.0


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_phase_z_affordance_features (
    match_id                                TEXT    NOT NULL,
    team_id                                 INTEGER NOT NULL,
    phase_idx                               INTEGER NOT NULL,

    -- Z1: absolute / percentile vertical position
    team_z_mean_m                           REAL,
    team_z_range_m                          REAL,
    team_z_percentile_alive                 REAL,

    -- Z2: contact-gated relative height advantage
    contact_delta_z_mean_m                  REAL,
    contact_delta_z_positive_ratio          REAL,
    high_ground_contact_advantage           REAL,

    -- Z3: Z2 under contraction pressure
    high_ground_x_boundary_pressure         REAL,
    z_percentile_x_hab_crowding_pressure    REAL,
    contact_delta_z_x_outside_safe_ratio    REAL,

    created_at                              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);

CREATE INDEX IF NOT EXISTS ix_tpzaf_match_phase
    ON stdb.team_phase_z_affordance_features (match_id, phase_idx);
"""


BUILD_SQL = """
WITH bounds AS (
    SELECT phase_idx, phase_start_s, phase_end_s
      FROM stdb.v_match_phase_bounds
     WHERE match_id = %(match_id)s
       AND phase_idx BETWEEN 1 AND 7
),
universe AS (
    SELECT t.team_id, b.phase_idx
      FROM stdb.teams t
     CROSS JOIN bounds b
     WHERE t.match_id = %(match_id)s
),
team_z AS (
    SELECT s.phase_idx,
           s.team_id,
           avg(s.pos_z)::real AS team_z_mean_m,
           (max(s.pos_z) - min(s.pos_z))::real AS team_z_range_m
      FROM stdb.team_snapshot_features s
     WHERE s.match_id = %(match_id)s
       AND s.phase_idx BETWEEN 1 AND 7
       AND s.alive_n > 0
       AND s.pos_z IS NOT NULL
     GROUP BY 1, 2
),
team_z_rank AS (
    SELECT z.*,
           percent_rank() OVER (
               PARTITION BY z.phase_idx
               ORDER BY z.team_z_mean_m
           )::real AS team_z_percentile_alive
      FROM team_z z
),
prox_pairs AS (
    SELECT b.phase_idx,
           p.snapshot_t,
           p.team_a,
           p.team_b,
           a.pos_z AS za,
           c.pos_z AS zb
      FROM bounds b
      JOIN stdb.team_proximity p
        ON p.match_id = %(match_id)s
       AND p.snapshot_t BETWEEN b.phase_start_s AND b.phase_end_s
       AND p.distance_m <= %(contact_distance_m)s
      JOIN stdb.team_snapshot_features a
        ON a.match_id = %(match_id)s
       AND a.snapshot_t = p.snapshot_t
       AND a.team_id = p.team_a
       AND a.alive_n > 0
       AND a.pos_z IS NOT NULL
      JOIN stdb.team_snapshot_features c
        ON c.match_id = %(match_id)s
       AND c.snapshot_t = p.snapshot_t
       AND c.team_id = p.team_b
       AND c.alive_n > 0
       AND c.pos_z IS NOT NULL
),
directed_delta AS (
    SELECT phase_idx, team_a AS team_id, (za - zb)::real AS delta_z_m
      FROM prox_pairs
    UNION ALL
    SELECT phase_idx, team_b AS team_id, (zb - za)::real AS delta_z_m
      FROM prox_pairs
),
contact_z AS (
    SELECT phase_idx,
           team_id,
           avg(delta_z_m)::real AS contact_delta_z_mean_m,
           avg(CASE WHEN delta_z_m > 0 THEN 1.0 ELSE 0.0 END)::real
               AS contact_delta_z_positive_ratio,
           avg(greatest(-1.0, least(1.0, delta_z_m / %(height_advantage_scale_m)s)))::real
               AS high_ground_contact_advantage
      FROM directed_delta
     GROUP BY 1, 2
)
INSERT INTO stdb.team_phase_z_affordance_features (
    match_id, team_id, phase_idx,
    team_z_mean_m, team_z_range_m, team_z_percentile_alive,
    contact_delta_z_mean_m, contact_delta_z_positive_ratio,
    high_ground_contact_advantage,
    high_ground_x_boundary_pressure,
    z_percentile_x_hab_crowding_pressure,
    contact_delta_z_x_outside_safe_ratio
)
SELECT
    %(match_id)s::text,
    u.team_id,
    u.phase_idx,
    zr.team_z_mean_m,
    zr.team_z_range_m,
    zr.team_z_percentile_alive,
    cz.contact_delta_z_mean_m,
    cz.contact_delta_z_positive_ratio,
    cz.high_ground_contact_advantage,
    (
        cz.high_ground_contact_advantage
        * coalesce(h.hab_edge_pressure_ratio, ex.outside_safe_ratio, 0.0)
    )::real AS high_ground_x_boundary_pressure,
    (
        zr.team_z_percentile_alive
        * h.hab_crowding_pressure
    )::real AS z_percentile_x_hab_crowding_pressure,
    (
        cz.contact_delta_z_mean_m
        * coalesce(ex.outside_safe_ratio, h.hab_outside_safe_ratio, 0.0)
    )::real AS contact_delta_z_x_outside_safe_ratio
FROM universe u
LEFT JOIN team_z_rank zr
       ON zr.team_id = u.team_id AND zr.phase_idx = u.phase_idx
LEFT JOIN contact_z cz
       ON cz.team_id = u.team_id AND cz.phase_idx = u.phase_idx
LEFT JOIN stdb.team_phase_habitat_features h
       ON h.match_id = %(match_id)s AND h.team_id = u.team_id
      AND h.phase_idx = u.phase_idx
LEFT JOIN stdb.team_phase_exposure ex
       ON ex.match_id = %(match_id)s AND ex.team_id = u.team_id
      AND ex.phase_idx = u.phase_idx
"""


def apply_schema() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def build_match(cur, match_id: str) -> int:
    cur.execute(
        "DELETE FROM stdb.team_phase_z_affordance_features WHERE match_id = %s",
        (match_id,),
    )
    cur.execute(
        BUILD_SQL,
        {
            "match_id": match_id,
            "contact_distance_m": CONTACT_DISTANCE_M,
            "height_advantage_scale_m": HEIGHT_ADVANTAGE_SCALE_M,
        },
    )
    return int(cur.rowcount)


def select_match_ids(
    *,
    limit: int | None,
    refresh: bool,
    match_id: str | None,
    map_name: str | None,
    mode: str | None,
    shards: int = 1,
    shard_index: int = 0,
) -> list[str]:
    if match_id:
        return [match_id]
    if shards < 1:
        raise ValueError("--shards must be >= 1")
    if shard_index < 0 or shard_index >= shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shards")

    sql = """
        SELECT DISTINCT l.match_id
          FROM stdb.team_phase_labels l
          JOIN stdb.matches m ON m.match_id = l.match_id
          JOIN stdb.team_snapshot_features s ON s.match_id = l.match_id
         WHERE (%s IS NULL OR m.map_name = %s)
           AND (%s IS NULL OR m.game_mode = %s)
    """
    params: list[object] = [map_name, map_name, mode, mode]
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id
              FROM target t
             WHERE NOT EXISTS (
                   SELECT 1
                     FROM stdb.team_phase_z_affordance_features f
                    WHERE f.match_id = t.match_id
             )
             ORDER BY t.match_id
        """
    else:
        sql += " ORDER BY l.match_id"

    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
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
                "SELECT count(*), count(DISTINCT match_id) FROM stdb.team_phase_z_affordance_features"
            )
            rows, done = cur.fetchone()
    return {
        "label_matches": total,
        "z_affordance_done": int(done),
        "remaining": total - int(done),
        "z_affordance_rows": int(rows),
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
            if idx % 100 == 0 or idx == len(match_ids):
                logger.info(
                    "[%s/%s] OK %s elapsed=%.2fs rows=%s",
                    idx, len(match_ids), match_id, time.time() - t0, rows,
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
    parser.add_argument("--map-name")
    parser.add_argument("--mode")
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
        logger.info("schema applied")

    if args.status:
        print(status())
        return 0

    match_ids = select_match_ids(
        limit=args.limit,
        refresh=args.refresh,
        match_id=args.match_id,
        map_name=args.map_name,
        mode=args.mode,
        shards=args.shards,
        shard_index=args.shard_index,
    )
    logger.info("selected %s matches", len(match_ids))
    if not match_ids:
        print({"ok": 0, "failed": 0, "failures": [], "elapsed_s": 0.0})
        return 0
    result = build_batch(match_ids)
    print(result)
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
