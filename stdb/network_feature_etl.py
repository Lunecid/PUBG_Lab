"""
F6 Tier-1 network role feature builder (FEATURE_DESIGN_V2 §7).

Summarizes stdb.team_phase_edge_cache into node-level network role features:

    stdb.team_phase_network_features
        (match_id, team_id, phase_idx, net_* ..., cum_* ...)

Column contract = family `f6_network_phase_role_tier1` in
model_stgcn/feature_manifests/pubg_representation_v2.json (43 columns).

Semantics
---------
* All metrics use raw edge-cache weights, i.e. BEFORE DynamicSTGNN
  row-normalization. Combat weight is the cache's log-damped intensity
  ln(1 + damage + 100*kills + 50*groggy); "strength" sums these per-pair
  weights, so it is breadth-weighted intensity, not raw damage.
* `net_*` columns are phase-local: built only from edges of that phase.
* `cum_*` columns accumulate phases 1..k of the same match (never beyond
  the cut phase, never full-match) — see leakage rules in §7.5.
* Combat / co-pressure are event channels: absence means the event did
  not happen, so counts and strengths coalesce to 0.
* Proximity distance aggregates (mean/min/isolation/top-k/entropy) are
  NULL when the team has no proximity edges in the phase (unobserved,
  not zero). Count/sum-style proximity columns coalesce to 0.
* Contract redundancies kept verbatim from the design doc:
  unique_targets == out_degree and unique_attackers == in_degree at
  phase level (the edge cache is already pair-aggregated per phase);
  isolation_m == min_distance_m; copress top1_weight == max_weight.
  The cum_* variants are NOT redundant (edge-row sums vs distinct teams).

The script is match-scoped and resumable, mirroring phase_edge_etl.py.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from stdb.conn import connect

logger = logging.getLogger("stdb.network_feature_etl")


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_phase_network_features (
    match_id                                TEXT    NOT NULL,
    team_id                                 INTEGER NOT NULL,
    phase_idx                               INTEGER NOT NULL,

    -- combat local (phase k edges only)
    net_combat_out_degree                   INTEGER NOT NULL DEFAULT 0,
    net_combat_in_degree                    INTEGER NOT NULL DEFAULT 0,
    net_combat_total_degree                 INTEGER NOT NULL DEFAULT 0,
    net_combat_out_strength                 REAL    NOT NULL DEFAULT 0,
    net_combat_in_strength                  REAL    NOT NULL DEFAULT 0,
    net_combat_total_strength               REAL    NOT NULL DEFAULT 0,
    net_combat_balance                      REAL    NOT NULL DEFAULT 0,
    net_combat_unique_targets               INTEGER NOT NULL DEFAULT 0,
    net_combat_unique_attackers             INTEGER NOT NULL DEFAULT 0,
    net_combat_reciprocal_pairs             INTEGER NOT NULL DEFAULT 0,
    net_combat_reciprocity_ratio            REAL    NOT NULL DEFAULT 0,
    net_combat_out_strength_per_min         REAL,
    net_combat_in_strength_per_min          REAL,

    -- proximity local
    net_prox_degree_100m                    INTEGER NOT NULL DEFAULT 0,
    net_prox_degree_200m                    INTEGER NOT NULL DEFAULT 0,
    net_prox_degree_500m                    INTEGER NOT NULL DEFAULT 0,
    net_prox_strength_decay                 REAL    NOT NULL DEFAULT 0,
    net_prox_mean_distance_m                REAL,
    net_prox_min_distance_m                 REAL,
    net_prox_top1_weight                    REAL,
    net_prox_top3_strength                  REAL,
    net_prox_isolation_m                    REAL,
    net_prox_neighbor_entropy               REAL,

    -- co-pressure local
    net_copress_degree                      INTEGER NOT NULL DEFAULT 0,
    net_copress_strength                    REAL    NOT NULL DEFAULT 0,
    net_copress_max_weight                  REAL    NOT NULL DEFAULT 0,
    net_copress_mean_ratio                  REAL    NOT NULL DEFAULT 0,
    net_copress_shared_pressure_count       INTEGER NOT NULL DEFAULT 0,
    net_copress_top1_weight                 REAL    NOT NULL DEFAULT 0,

    -- cumulative through phases 1..k (same match only)
    cum_combat_out_degree                   INTEGER NOT NULL DEFAULT 0,
    cum_combat_in_degree                    INTEGER NOT NULL DEFAULT 0,
    cum_combat_out_strength                 REAL    NOT NULL DEFAULT 0,
    cum_combat_in_strength                  REAL    NOT NULL DEFAULT 0,
    cum_combat_total_strength               REAL    NOT NULL DEFAULT 0,
    cum_combat_reciprocal_pairs             INTEGER NOT NULL DEFAULT 0,
    cum_unique_targets                      INTEGER NOT NULL DEFAULT 0,
    cum_unique_attackers                    INTEGER NOT NULL DEFAULT 0,
    cum_copress_strength                    REAL    NOT NULL DEFAULT 0,
    cum_prox_close_exposure_200m            INTEGER NOT NULL DEFAULT 0,

    -- phase-normalized
    net_combat_out_strength_per_alive_team  REAL,
    net_combat_in_strength_per_alive_team   REAL,
    net_prox_degree_200m_per_alive_team     REAL,
    net_copress_degree_per_alive_team       REAL,

    created_at                              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);

CREATE INDEX IF NOT EXISTS ix_tpnf_match_phase
    ON stdb.team_phase_network_features (match_id, phase_idx);
"""


BUILD_SQL = """
WITH edges AS (
    SELECT phase_idx, edge_type, src_team_id, dst_team_id,
           weight, support_count, avg_distance_m, ratio
      FROM stdb.team_phase_edge_cache
     WHERE match_id = %(match_id)s
),
universe AS (
    -- same (team x phase) universe as v_team_phase_feature_long
    SELECT t.team_id,
           b.phase_idx,
           b.phase_duration_s,
           g.avg_alive_teams
      FROM stdb.teams t
      JOIN stdb.v_match_phase_bounds b
        ON b.match_id = t.match_id
 LEFT JOIN stdb.match_phase_global g
        ON g.match_id = t.match_id AND g.phase_idx = b.phase_idx
     WHERE t.match_id = %(match_id)s
),
combat_out AS (
    SELECT phase_idx, src_team_id AS team_id,
           count(*)::int     AS out_degree,
           sum(weight)::real AS out_strength
      FROM edges
     WHERE edge_type = 'combat'
     GROUP BY 1, 2
),
combat_in AS (
    SELECT phase_idx, dst_team_id AS team_id,
           count(*)::int     AS in_degree,
           sum(weight)::real AS in_strength
      FROM edges
     WHERE edge_type = 'combat'
     GROUP BY 1, 2
),
combat_recip AS (
    -- opponents with combat edges in BOTH directions within the phase
    SELECT a.phase_idx, a.src_team_id AS team_id,
           count(*)::int AS reciprocal_pairs
      FROM edges a
      JOIN edges b
        ON b.edge_type = 'combat'
       AND b.phase_idx = a.phase_idx
       AND b.src_team_id = a.dst_team_id
       AND b.dst_team_id = a.src_team_id
     WHERE a.edge_type = 'combat'
     GROUP BY 1, 2
),
prox AS (
    SELECT phase_idx, src_team_id AS team_id,
           count(*) FILTER (WHERE avg_distance_m < 100)::int AS degree_100m,
           count(*) FILTER (WHERE avg_distance_m < 200)::int AS degree_200m,
           count(*) FILTER (WHERE avg_distance_m < 500)::int AS degree_500m,
           sum(weight)::real                                 AS strength_decay,
           avg(avg_distance_m)::real                         AS mean_distance_m,
           min(avg_distance_m)::real                         AS min_distance_m,
           max(weight)::real                                 AS top1_weight,
           sum(weight * ln(weight)) FILTER (WHERE weight > 0) AS w_lnw_sum
      FROM edges
     WHERE edge_type = 'proximity'
     GROUP BY 1, 2
),
prox_top3 AS (
    SELECT phase_idx, team_id, sum(weight)::real AS top3_strength
      FROM (
            SELECT phase_idx, src_team_id AS team_id, weight,
                   row_number() OVER (
                       PARTITION BY phase_idx, src_team_id
                       ORDER BY weight DESC
                   ) AS rn
              FROM edges
             WHERE edge_type = 'proximity'
      ) ranked
     WHERE rn <= 3
     GROUP BY 1, 2
),
copress AS (
    SELECT phase_idx, src_team_id AS team_id,
           count(*)::int           AS degree,
           sum(weight)::real       AS strength,
           max(weight)::real       AS max_weight,
           avg(ratio)::real        AS mean_ratio,
           sum(support_count)::int AS shared_pressure_count
      FROM edges
     WHERE edge_type = 'copressure'
     GROUP BY 1, 2
),
cum_out AS (
    SELECT u.team_id, u.phase_idx,
           count(e.weight)::int               AS cum_out_degree,
           coalesce(sum(e.weight), 0)::real   AS cum_out_strength,
           count(DISTINCT e.dst_team_id)::int AS cum_unique_targets
      FROM universe u
 LEFT JOIN edges e
        ON e.edge_type = 'combat'
       AND e.src_team_id = u.team_id
       AND e.phase_idx <= u.phase_idx
     GROUP BY 1, 2
),
cum_in AS (
    SELECT u.team_id, u.phase_idx,
           count(e.weight)::int               AS cum_in_degree,
           coalesce(sum(e.weight), 0)::real   AS cum_in_strength,
           count(DISTINCT e.src_team_id)::int AS cum_unique_attackers
      FROM universe u
 LEFT JOIN edges e
        ON e.edge_type = 'combat'
       AND e.dst_team_id = u.team_id
       AND e.phase_idx <= u.phase_idx
     GROUP BY 1, 2
),
cum_recip AS (
    -- opponents attacked in some phase <= k AND attacking back in some phase <= k
    SELECT u.team_id, u.phase_idx,
           count(DISTINCT a.dst_team_id)::int AS cum_reciprocal_pairs
      FROM universe u
      JOIN edges a
        ON a.edge_type = 'combat'
       AND a.src_team_id = u.team_id
       AND a.phase_idx <= u.phase_idx
      JOIN edges b
        ON b.edge_type = 'combat'
       AND b.src_team_id = a.dst_team_id
       AND b.dst_team_id = u.team_id
       AND b.phase_idx <= u.phase_idx
     GROUP BY 1, 2
),
cum_copress AS (
    SELECT u.team_id, u.phase_idx,
           coalesce(sum(e.weight), 0)::real AS cum_copress_strength
      FROM universe u
 LEFT JOIN edges e
        ON e.edge_type = 'copressure'
       AND e.src_team_id = u.team_id
       AND e.phase_idx <= u.phase_idx
     GROUP BY 1, 2
),
cum_prox AS (
    SELECT u.team_id, u.phase_idx,
           count(e.weight)::int AS cum_prox_close_exposure_200m
      FROM universe u
 LEFT JOIN edges e
        ON e.edge_type = 'proximity'
       AND e.src_team_id = u.team_id
       AND e.phase_idx <= u.phase_idx
       AND e.avg_distance_m < 200
     GROUP BY 1, 2
)
INSERT INTO stdb.team_phase_network_features (
    match_id, team_id, phase_idx,
    net_combat_out_degree, net_combat_in_degree, net_combat_total_degree,
    net_combat_out_strength, net_combat_in_strength, net_combat_total_strength,
    net_combat_balance,
    net_combat_unique_targets, net_combat_unique_attackers,
    net_combat_reciprocal_pairs, net_combat_reciprocity_ratio,
    net_combat_out_strength_per_min, net_combat_in_strength_per_min,
    net_prox_degree_100m, net_prox_degree_200m, net_prox_degree_500m,
    net_prox_strength_decay, net_prox_mean_distance_m, net_prox_min_distance_m,
    net_prox_top1_weight, net_prox_top3_strength, net_prox_isolation_m,
    net_prox_neighbor_entropy,
    net_copress_degree, net_copress_strength, net_copress_max_weight,
    net_copress_mean_ratio, net_copress_shared_pressure_count,
    net_copress_top1_weight,
    cum_combat_out_degree, cum_combat_in_degree,
    cum_combat_out_strength, cum_combat_in_strength, cum_combat_total_strength,
    cum_combat_reciprocal_pairs, cum_unique_targets, cum_unique_attackers,
    cum_copress_strength, cum_prox_close_exposure_200m,
    net_combat_out_strength_per_alive_team, net_combat_in_strength_per_alive_team,
    net_prox_degree_200m_per_alive_team, net_copress_degree_per_alive_team
)
SELECT
    %(match_id)s::text,
    u.team_id,
    u.phase_idx,

    -- combat local
    coalesce(co.out_degree, 0),
    coalesce(ci.in_degree, 0),
    coalesce(co.out_degree, 0) + coalesce(ci.in_degree, 0),
    coalesce(co.out_strength, 0),
    coalesce(ci.in_strength, 0),
    coalesce(co.out_strength, 0) + coalesce(ci.in_strength, 0),
    (
        (coalesce(co.out_strength, 0) - coalesce(ci.in_strength, 0))
        / (coalesce(co.out_strength, 0) + coalesce(ci.in_strength, 0) + 1e-6)
    )::real,
    coalesce(co.out_degree, 0),
    coalesce(ci.in_degree, 0),
    coalesce(cr.reciprocal_pairs, 0),
    CASE
        WHEN coalesce(co.out_degree, 0) + coalesce(ci.in_degree, 0)
             - coalesce(cr.reciprocal_pairs, 0) > 0
        THEN (
            coalesce(cr.reciprocal_pairs, 0)::real
            / (coalesce(co.out_degree, 0) + coalesce(ci.in_degree, 0)
               - coalesce(cr.reciprocal_pairs, 0))
        )::real
        ELSE 0
    END,
    (coalesce(co.out_strength, 0) * 60.0 / nullif(u.phase_duration_s, 0))::real,
    (coalesce(ci.in_strength, 0) * 60.0 / nullif(u.phase_duration_s, 0))::real,

    -- proximity local
    coalesce(p.degree_100m, 0),
    coalesce(p.degree_200m, 0),
    coalesce(p.degree_500m, 0),
    coalesce(p.strength_decay, 0),
    p.mean_distance_m,
    p.min_distance_m,
    p.top1_weight,
    p3.top3_strength,
    p.min_distance_m,
    CASE
        WHEN p.strength_decay > 0
        THEN (ln(p.strength_decay) - p.w_lnw_sum / p.strength_decay)::real
        ELSE NULL
    END,

    -- co-pressure local
    coalesce(cp.degree, 0),
    coalesce(cp.strength, 0),
    coalesce(cp.max_weight, 0),
    coalesce(cp.mean_ratio, 0),
    coalesce(cp.shared_pressure_count, 0),
    coalesce(cp.max_weight, 0),

    -- cumulative 1..k
    coalesce(cuo.cum_out_degree, 0),
    coalesce(cui.cum_in_degree, 0),
    coalesce(cuo.cum_out_strength, 0),
    coalesce(cui.cum_in_strength, 0),
    coalesce(cuo.cum_out_strength, 0) + coalesce(cui.cum_in_strength, 0),
    coalesce(cur.cum_reciprocal_pairs, 0),
    coalesce(cuo.cum_unique_targets, 0),
    coalesce(cui.cum_unique_attackers, 0),
    coalesce(cuc.cum_copress_strength, 0),
    coalesce(cup.cum_prox_close_exposure_200m, 0),

    -- phase-normalized
    (coalesce(co.out_strength, 0) / nullif(u.avg_alive_teams, 0))::real,
    (coalesce(ci.in_strength, 0) / nullif(u.avg_alive_teams, 0))::real,
    (coalesce(p.degree_200m, 0) / nullif(u.avg_alive_teams, 0))::real,
    (coalesce(cp.degree, 0) / nullif(u.avg_alive_teams, 0))::real
FROM universe u
LEFT JOIN combat_out   co  ON co.team_id  = u.team_id AND co.phase_idx  = u.phase_idx
LEFT JOIN combat_in    ci  ON ci.team_id  = u.team_id AND ci.phase_idx  = u.phase_idx
LEFT JOIN combat_recip cr  ON cr.team_id  = u.team_id AND cr.phase_idx  = u.phase_idx
LEFT JOIN prox         p   ON p.team_id   = u.team_id AND p.phase_idx   = u.phase_idx
LEFT JOIN prox_top3    p3  ON p3.team_id  = u.team_id AND p3.phase_idx  = u.phase_idx
LEFT JOIN copress      cp  ON cp.team_id  = u.team_id AND cp.phase_idx  = u.phase_idx
LEFT JOIN cum_out      cuo ON cuo.team_id = u.team_id AND cuo.phase_idx = u.phase_idx
LEFT JOIN cum_in       cui ON cui.team_id = u.team_id AND cui.phase_idx = u.phase_idx
LEFT JOIN cum_recip    cur ON cur.team_id = u.team_id AND cur.phase_idx = u.phase_idx
LEFT JOIN cum_copress  cuc ON cuc.team_id = u.team_id AND cuc.phase_idx = u.phase_idx
LEFT JOIN cum_prox     cup ON cup.team_id = u.team_id AND cup.phase_idx = u.phase_idx
"""


def apply_schema() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def build_match(cur, match_id: str) -> int:
    cur.execute(
        "DELETE FROM stdb.team_phase_network_features WHERE match_id = %s",
        (match_id,),
    )
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

    # Universe = matches with a built edge cache (the only F6 input).
    sql = "SELECT DISTINCT match_id FROM stdb.team_phase_edge_cache"
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id
              FROM target t
             WHERE NOT EXISTS (
                   SELECT 1
                     FROM stdb.team_phase_network_features f
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
            cur.execute("SELECT count(DISTINCT match_id) FROM stdb.team_phase_edge_cache")
            total = int(cur.fetchone()[0])
            cur.execute(
                "SELECT count(*), count(DISTINCT match_id) FROM stdb.team_phase_network_features"
            )
            rows, done = cur.fetchone()
    return {
        "edge_cache_matches": total,
        "network_done": int(done),
        "remaining": total - int(done),
        "network_rows": int(rows),
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
