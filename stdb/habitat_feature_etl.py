"""
F7 Tier-1 habitat-inspired pressure feature builder (FEATURE_DESIGN_V2 §8).

Quantifies shrinking-playable-space pressure per team-phase:

    stdb.team_phase_habitat_features
        (match_id, team_id, phase_idx, hab_* ...)

Column contract = family `f7_habitat_pressure_tier1` in
model_stgcn/feature_manifests/pubg_representation_v2.json (32 columns).

Zone semantics (verified empirically 2026-06-12, alive-containment test):
    dynamic_regions 'safe'   = current blue zone (damaging boundary; alive
                               teams sit inside: 77%/95%/97% at phases 2/5/7)
    dynamic_regions 'poison' = announced white target circle (next safe area)

Measurement conventions
-----------------------
* Global block broadcasts (match, phase) zone state from match_phase_global:
  areas are pi*r^2, `hab_alive_teams` = end_alive_teams (population entering
  the squeeze), crowding = alive / area_end. `hab_zone_gap_*` copies the
  existing mpg convention (poison_radius - safe_radius, usually negative
  while the blue zone is still closing); `hab_zone_gap_pressure` normalizes
  the end gap by the target radius (0 = closed onto target).
* The white-circle announcement does NOT align with stdb phase boundaries
  (target radius shifts mid-phase). All target-relative features therefore
  use the target region at the team's CUT time — the last poison snapshot of
  the phase, i.e. exactly the circle a player sees on the map at the cut.
* `hab_required_speed_to_safe_mps` = distance-to-target at cut / this
  phase's duration. The true closing countdown belongs to the next phase
  (future), so the current cycle length is used as a causal pace proxy.
  `hab_migration_deficit_mps` = required - observed (trajectory avg speed).
* `hab_rotation_delay_ratio` = normalized first-entry time into the cut-time
  target region; teams that never enter during the phase get 1.0 (maximal
  delay), teams already inside get 0.0.
* Boundary block measures the CURRENT blue zone at the first/last snapshot;
  signed distance is positive inside (cushion), negative outside (exposure).
  `hab_edge_pressure_ratio` uses the co-pressure convention: outside OR
  within 150 m inside the boundary.
* Local block counts enemy teams from team_proximity at the team's last
  snapshot; `hab_safe_entry_*` evaluates the entry snapshot into the target
  region (NULL when the team never enters during the phase).
  `hab_refuge_scarcity_simple` = entry enemy count / (blue area_end + 1 m^2)
  per the design doc's pragmatic first implementation.
* Reused quantities are joined, not recomputed: `hab_observed_speed_mps` =
  team_phase_trajectory.avg_speed_mps, `hab_outside_safe_ratio` =
  team_phase_exposure.outside_safe_ratio.
* NULL policy: dead/snapshot-less team-phases keep the global block (zone
  state is public) and NULL for team-relative features.

The script is match-scoped and resumable, mirroring phase_edge_etl.py.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from stdb.conn import connect

logger = logging.getLogger("stdb.habitat_feature_etl")

EDGE_PRESSURE_M = 150.0  # same threshold as the co-pressure edge channel


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.team_phase_habitat_features (
    match_id                                TEXT    NOT NULL,
    team_id                                 INTEGER NOT NULL,
    phase_idx                               INTEGER NOT NULL,

    -- global shrinking-space block (match, phase)
    hab_safe_area_start_m2                  REAL,
    hab_safe_area_end_m2                    REAL,
    hab_safe_area_delta_m2                  REAL,
    hab_loss_ratio                          REAL,
    hab_shrink_rate_m2ps                    REAL,
    hab_alive_teams                         INTEGER,
    hab_safe_area_per_alive_team            REAL,
    hab_crowding_pressure                   REAL,
    hab_center_shift_m                      REAL,
    hab_center_shift_per_radius             REAL,
    hab_zone_gap_start_m                    REAL,
    hab_zone_gap_end_m                      REAL,
    hab_zone_gap_pressure                   REAL,

    -- team movement-pressure block
    hab_dist_to_safe_boundary_start_m       REAL,
    hab_dist_to_safe_boundary_end_m         REAL,
    hab_signed_boundary_distance_start_m    REAL,
    hab_signed_boundary_distance_end_m      REAL,
    hab_distance_to_target_safe_m           REAL,
    hab_required_speed_to_safe_mps          REAL,
    hab_observed_speed_mps                  REAL,
    hab_migration_deficit_mps               REAL,
    hab_outside_safe_ratio                  REAL,
    hab_edge_pressure_ratio                 REAL,
    hab_rotation_delay_ratio                REAL,
    hab_safe_alignment_to_target            REAL,

    -- local competition / refuge scarcity block
    hab_local_enemy_count_100m              INTEGER,
    hab_local_enemy_count_200m              INTEGER,
    hab_local_enemy_count_500m              INTEGER,
    hab_local_enemy_density_200m            REAL,
    hab_safe_entry_enemy_count_200m         INTEGER,
    hab_safe_entry_congestion               INTEGER,
    hab_refuge_scarcity_simple              REAL,

    created_at                              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, phase_idx)
);

CREATE INDEX IF NOT EXISTS ix_tphf_match_phase
    ON stdb.team_phase_habitat_features (match_id, phase_idx);
"""


BUILD_SQL = """
WITH bounds AS (
    SELECT phase_idx, phase_start_s, phase_end_s, phase_duration_s
      FROM stdb.v_match_phase_bounds
     WHERE match_id = %(match_id)s
       AND phase_idx > 0
),
universe AS (
    SELECT t.team_id, b.phase_idx, b.phase_start_s, b.phase_end_s, b.phase_duration_s
      FROM stdb.teams t
     CROSS JOIN bounds b
     WHERE t.match_id = %(match_id)s
),
team_snaps AS (
    SELECT b.phase_idx,
           mo.team_id,
           mo.snapshot_t,
           ST_Force2D(mo.pos) AS pos,
           CASE WHEN sr.region IS NULL THEN NULL
                ELSE ST_Contains(sr.region, ST_Force2D(mo.pos)) END AS in_blue,
           CASE WHEN sr.region IS NULL THEN NULL
                ELSE ST_Distance(ST_Force2D(mo.pos), ST_Boundary(sr.region)) END AS d_blue
      FROM bounds b
      JOIN stdb.moving_objects mo
        ON mo.match_id = %(match_id)s
       AND mo.agent_kind = 'team'
       AND mo.is_alive
       AND mo.snapshot_t BETWEEN b.phase_start_s AND b.phase_end_s
 LEFT JOIN stdb.dynamic_regions sr
        ON sr.match_id = %(match_id)s
       AND sr.region_kind = 'safe'
       AND sr.snapshot_t = mo.snapshot_t
),
cut_target AS (
    -- white circle visible at the cut: last poison snapshot within the phase
    SELECT b.phase_idx, dr.region AS target_region, ST_Force2D(dr.center_pos) AS target_center
      FROM bounds b
      JOIN LATERAL (
          SELECT region, center_pos
            FROM stdb.dynamic_regions
           WHERE match_id = %(match_id)s
             AND region_kind = 'poison'
             AND snapshot_t BETWEEN b.phase_start_s AND b.phase_end_s
           ORDER BY snapshot_t DESC
           LIMIT 1
      ) dr ON TRUE
),
team_phase AS (
    SELECT phase_idx, team_id,
           count(*) AS n_snaps,
           max(snapshot_t) AS t_cut,
           (array_agg(pos ORDER BY snapshot_t))[1]      AS first_pos,
           (array_agg(pos ORDER BY snapshot_t DESC))[1] AS last_pos,
           (array_agg(d_blue ORDER BY snapshot_t)
                FILTER (WHERE d_blue IS NOT NULL))[1]      AS d_blue_start,
           (array_agg(d_blue ORDER BY snapshot_t DESC)
                FILTER (WHERE d_blue IS NOT NULL))[1]      AS d_blue_end,
           (array_agg(in_blue ORDER BY snapshot_t)
                FILTER (WHERE in_blue IS NOT NULL))[1]     AS in_blue_start,
           (array_agg(in_blue ORDER BY snapshot_t DESC)
                FILTER (WHERE in_blue IS NOT NULL))[1]     AS in_blue_end,
           avg(CASE WHEN in_blue = FALSE
                      OR (in_blue AND d_blue <= %(edge_pressure_m)s)
                    THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE in_blue IS NOT NULL)           AS edge_pressure_ratio
      FROM team_snaps
     GROUP BY 1, 2
),
entry AS (
    SELECT ts.phase_idx, ts.team_id, min(ts.snapshot_t) AS entry_t
      FROM team_snaps ts
      JOIN cut_target ct ON ct.phase_idx = ts.phase_idx
     WHERE ST_Contains(ct.target_region, ts.pos)
     GROUP BY 1, 2
),
local_cut AS (
    SELECT x.phase_idx, x.team_id,
           count(*) FILTER (WHERE x.d < 100)::int AS cnt100,
           count(*) FILTER (WHERE x.d < 200)::int AS cnt200,
           count(*) FILTER (WHERE x.d < 500)::int AS cnt500
      FROM (
        SELECT tp.phase_idx, tp.team_id, pr.distance_m AS d
          FROM team_phase tp
          JOIN stdb.team_proximity pr
            ON pr.match_id = %(match_id)s
           AND pr.snapshot_t = tp.t_cut
           AND (pr.team_a = tp.team_id OR pr.team_b = tp.team_id)
      ) x
     GROUP BY 1, 2
),
entry_local AS (
    SELECT e.phase_idx, e.team_id,
           count(*) FILTER (WHERE pr.distance_m < 200)::int AS entry_cnt200
      FROM entry e
      JOIN stdb.team_proximity pr
        ON pr.match_id = %(match_id)s
       AND pr.snapshot_t = e.entry_t
       AND (pr.team_a = e.team_id OR pr.team_b = e.team_id)
     GROUP BY 1, 2
),
entry_congestion AS (
    SELECT e.phase_idx, e.team_id, count(*)::int AS inside_cnt
      FROM entry e
      JOIN cut_target ct ON ct.phase_idx = e.phase_idx
      JOIN stdb.moving_objects mo
        ON mo.match_id = %(match_id)s
       AND mo.agent_kind = 'team'
       AND mo.is_alive
       AND mo.snapshot_t = e.entry_t
       AND mo.team_id <> e.team_id
     WHERE ST_Contains(ct.target_region, ST_Force2D(mo.pos))
     GROUP BY 1, 2
)
INSERT INTO stdb.team_phase_habitat_features (
    match_id, team_id, phase_idx,
    hab_safe_area_start_m2, hab_safe_area_end_m2, hab_safe_area_delta_m2,
    hab_loss_ratio, hab_shrink_rate_m2ps, hab_alive_teams,
    hab_safe_area_per_alive_team, hab_crowding_pressure,
    hab_center_shift_m, hab_center_shift_per_radius,
    hab_zone_gap_start_m, hab_zone_gap_end_m, hab_zone_gap_pressure,
    hab_dist_to_safe_boundary_start_m, hab_dist_to_safe_boundary_end_m,
    hab_signed_boundary_distance_start_m, hab_signed_boundary_distance_end_m,
    hab_distance_to_target_safe_m, hab_required_speed_to_safe_mps,
    hab_observed_speed_mps, hab_migration_deficit_mps,
    hab_outside_safe_ratio, hab_edge_pressure_ratio,
    hab_rotation_delay_ratio, hab_safe_alignment_to_target,
    hab_local_enemy_count_100m, hab_local_enemy_count_200m,
    hab_local_enemy_count_500m, hab_local_enemy_density_200m,
    hab_safe_entry_enemy_count_200m, hab_safe_entry_congestion,
    hab_refuge_scarcity_simple
)
SELECT
    %(match_id)s::text,
    u.team_id,
    u.phase_idx,

    -- global shrinking-space block
    (pi() * g.safe_radius_start_m ^ 2)::real,
    (pi() * g.safe_radius_end_m ^ 2)::real,
    (pi() * (g.safe_radius_end_m ^ 2 - g.safe_radius_start_m ^ 2))::real,
    (1.0 - (g.safe_radius_end_m ^ 2) / nullif(g.safe_radius_start_m ^ 2, 0))::real,
    (pi() * (g.safe_radius_start_m ^ 2 - g.safe_radius_end_m ^ 2)
        / nullif(u.phase_duration_s, 0))::real,
    g.end_alive_teams,
    (pi() * g.safe_radius_end_m ^ 2 / greatest(g.end_alive_teams, 1))::real,
    (g.end_alive_teams / nullif(pi() * g.safe_radius_end_m ^ 2, 0))::real,
    g.safe_center_shift_m,
    (g.safe_center_shift_m / nullif(g.safe_radius_start_m, 0))::real,
    g.zone_gap_start_m,
    g.zone_gap_end_m,
    (g.zone_gap_end_m / nullif(g.poison_radius_end_m, 0))::real,

    -- team movement-pressure block
    tp.d_blue_start,
    tp.d_blue_end,
    (CASE WHEN tp.in_blue_start THEN tp.d_blue_start ELSE -tp.d_blue_start END)::real,
    (CASE WHEN tp.in_blue_end THEN tp.d_blue_end ELSE -tp.d_blue_end END)::real,
    ST_Distance(tp.last_pos, ct.target_region)::real,
    (ST_Distance(tp.last_pos, ct.target_region)
        / nullif(u.phase_duration_s, 0))::real,
    tr.avg_speed_mps,
    (ST_Distance(tp.last_pos, ct.target_region) / nullif(u.phase_duration_s, 0)
        - tr.avg_speed_mps)::real,
    ex.outside_safe_ratio,
    tp.edge_pressure_ratio::real,
    CASE
        WHEN tp.n_snaps IS NULL OR ct.phase_idx IS NULL THEN NULL
        WHEN en.entry_t IS NULL THEN 1.0
        ELSE greatest(0, least(1,
            (en.entry_t - u.phase_start_s) / nullif(u.phase_duration_s, 0)))
    END::real,
    CASE
        WHEN tp.last_pos IS NULL OR ct.target_center IS NULL THEN NULL
        WHEN ST_Distance(tp.first_pos, tp.last_pos) <= 1.0 THEN NULL
        WHEN ST_Distance(tp.first_pos, ct.target_center) <= 1.0 THEN NULL
        ELSE (
            ((ST_X(tp.last_pos) - ST_X(tp.first_pos))
                * (ST_X(ct.target_center) - ST_X(tp.first_pos))
             + (ST_Y(tp.last_pos) - ST_Y(tp.first_pos))
                * (ST_Y(ct.target_center) - ST_Y(tp.first_pos)))
            / (ST_Distance(tp.first_pos, tp.last_pos)
               * ST_Distance(tp.first_pos, ct.target_center))
        )
    END::real,

    -- local competition block
    CASE WHEN tp.t_cut IS NOT NULL THEN coalesce(lc.cnt100, 0) END,
    CASE WHEN tp.t_cut IS NOT NULL THEN coalesce(lc.cnt200, 0) END,
    CASE WHEN tp.t_cut IS NOT NULL THEN coalesce(lc.cnt500, 0) END,
    CASE WHEN tp.t_cut IS NOT NULL
         THEN coalesce(lc.cnt200, 0) / (pi() * 0.2 ^ 2)
    END::real,
    CASE WHEN en.entry_t IS NOT NULL THEN coalesce(el.entry_cnt200, 0) END,
    CASE WHEN en.entry_t IS NOT NULL THEN coalesce(ec.inside_cnt, 0) END,
    CASE WHEN en.entry_t IS NOT NULL
         THEN coalesce(el.entry_cnt200, 0)
              / (pi() * g.safe_radius_end_m ^ 2 + 1.0)
    END::real
FROM universe u
LEFT JOIN stdb.match_phase_global g
       ON g.match_id = %(match_id)s AND g.phase_idx = u.phase_idx
LEFT JOIN team_phase tp
       ON tp.team_id = u.team_id AND tp.phase_idx = u.phase_idx
LEFT JOIN cut_target ct
       ON ct.phase_idx = u.phase_idx
LEFT JOIN entry en
       ON en.team_id = u.team_id AND en.phase_idx = u.phase_idx
LEFT JOIN local_cut lc
       ON lc.team_id = u.team_id AND lc.phase_idx = u.phase_idx
LEFT JOIN entry_local el
       ON el.team_id = u.team_id AND el.phase_idx = u.phase_idx
LEFT JOIN entry_congestion ec
       ON ec.team_id = u.team_id AND ec.phase_idx = u.phase_idx
LEFT JOIN stdb.team_phase_trajectory tr
       ON tr.match_id = %(match_id)s AND tr.team_id = u.team_id
      AND tr.phase_idx = u.phase_idx
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
        "DELETE FROM stdb.team_phase_habitat_features WHERE match_id = %s",
        (match_id,),
    )
    cur.execute(BUILD_SQL, {"match_id": match_id, "edge_pressure_m": EDGE_PRESSURE_M})
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

    # Universe = the modeling population (same as phase_edge_etl).
    sql = "SELECT DISTINCT match_id FROM stdb.team_phase_labels"
    if not refresh:
        sql = f"""
            WITH target AS ({sql})
            SELECT t.match_id
              FROM target t
             WHERE NOT EXISTS (
                   SELECT 1
                     FROM stdb.team_phase_habitat_features f
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
                "SELECT count(*), count(DISTINCT match_id) FROM stdb.team_phase_habitat_features"
            )
            rows, done = cur.fetchone()
    return {
        "label_matches": total,
        "habitat_done": int(done),
        "remaining": total - int(done),
        "habitat_rows": int(rows),
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
