"""Build leakage-safe historical outcome proxy tiers.

The current PUBG telemetry schema does not contain official ranked tier/MMR.
This ETL derives a proxy from each player's matches that occurred before the
target match, then aggregates those player priors to team-level context.

Important leakage rule:
    The target match outcome is never used to build its own proxy.

The generated global quantile bucket is a convenient exploratory slice. For a
final model run, buckets should be refit on the training split or replaced with
the continuous `team_prior_strength_raw` feature.

Important limitation:
    This proxy is not opponent-adjusted. A strong result in a weak lobby and a
    strong result in a hard lobby can receive similar scores. For tier-like
    absolute skill context, prefer `stdb.skill_rating_proxy_etl`.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import logging
import time
from typing import Any

import psycopg2.extras

from pubg_lab.data.stdb.conn import connect


logger = logging.getLogger("stdb.skill_proxy_etl")


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.player_skill_proxy (
    match_id                    TEXT        NOT NULL,
    player_id                   TEXT        NOT NULL,
    roster_id                   TEXT,
    team_id                     INTEGER,
    match_created_at            TIMESTAMPTZ NOT NULL,
    game_mode                   TEXT        NOT NULL,
    map_name                    TEXT        NOT NULL,
    tier_source                 TEXT        NOT NULL DEFAULT 'historical_proxy',
    tier_policy                 TEXT        NOT NULL,
    min_prior_matches           INTEGER     NOT NULL,

    prior_matches               INTEGER     NOT NULL,
    prior_placement_pct         REAL,
    prior_top10_rate            REAL,
    prior_win_rate              REAL,
    prior_kills_avg             REAL,
    prior_assists_avg           REAL,
    prior_damage_avg            REAL,
    prior_survival_pct          REAL,
    prior_strength_raw          REAL,
    proxy_ready                 BOOLEAN     NOT NULL,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, player_id, tier_source, tier_policy)
);

CREATE INDEX IF NOT EXISTS ix_player_skill_proxy_player_time
    ON stdb.player_skill_proxy (player_id, match_created_at);
CREATE INDEX IF NOT EXISTS ix_player_skill_proxy_match_team
    ON stdb.player_skill_proxy (match_id, team_id);
CREATE INDEX IF NOT EXISTS ix_player_skill_proxy_policy
    ON stdb.player_skill_proxy (tier_policy, proxy_ready);

CREATE TABLE IF NOT EXISTS stdb.team_skill_proxy (
    match_id                    TEXT        NOT NULL,
    team_id                     INTEGER     NOT NULL,
    match_created_at            TIMESTAMPTZ NOT NULL,
    game_mode                   TEXT        NOT NULL,
    map_name                    TEXT        NOT NULL,
    tier_source                 TEXT        NOT NULL DEFAULT 'historical_proxy',
    tier_policy                 TEXT        NOT NULL,
    min_prior_matches           INTEGER     NOT NULL,

    roster_players              INTEGER     NOT NULL,
    eligible_players            INTEGER     NOT NULL,
    coverage_ratio              REAL        NOT NULL,
    player_prior_matches_avg    REAL,
    player_prior_matches_max    INTEGER,
    team_prior_placement_pct    REAL,
    team_prior_top10_rate       REAL,
    team_prior_win_rate         REAL,
    team_prior_kills_avg        REAL,
    team_prior_damage_avg       REAL,
    team_prior_survival_pct     REAL,
    team_prior_strength_raw     REAL,
    team_skill_bucket_q_global  INTEGER,
    team_skill_bucket_global    TEXT,
    proxy_ready                 BOOLEAN     NOT NULL,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, tier_source, tier_policy)
);

CREATE INDEX IF NOT EXISTS ix_team_skill_proxy_match
    ON stdb.team_skill_proxy (match_id, team_id);
CREATE INDEX IF NOT EXISTS ix_team_skill_proxy_policy_bucket
    ON stdb.team_skill_proxy (tier_policy, team_skill_bucket_q_global);
CREATE INDEX IF NOT EXISTS ix_team_skill_proxy_time
    ON stdb.team_skill_proxy (match_created_at);
"""


PLAYER_SQL = """
WITH roster_counts AS (
    SELECT match_id, count(*)::real AS n_teams
      FROM pubg.rosters
     GROUP BY match_id
),
base AS (
    SELECT
        p.match_id,
        p.player_id,
        p.roster_id,
        r.team_id,
        m.created_at AS match_created_at,
        m.game_mode,
        m.map_name,
        greatest(coalesce(m.duration, 0), 1)::real AS duration_s,
        greatest(coalesce(rc.n_teams, 1), 1)::real AS n_teams,
        r.rank::real AS final_rank,
        CASE
            WHEN rc.n_teams > 1 AND r.rank IS NOT NULL
            THEN ((rc.n_teams - r.rank::real) / greatest(rc.n_teams - 1.0, 1.0))::real
            ELSE NULL
        END AS placement_pct,
        CASE WHEN r.rank IS NOT NULL AND r.rank <= LEAST(10, rc.n_teams)::integer
             THEN 1.0 ELSE 0.0 END::real AS top10,
        CASE WHEN coalesce(r.won, false) THEN 1.0 ELSE 0.0 END::real AS win,
        coalesce(p.kills, 0)::real AS kills,
        coalesce(p.assists, 0)::real AS assists,
        coalesce(p.damage_dealt, 0)::real AS damage,
        LEAST(
            1.0,
            GREATEST(0.0, coalesce(p.time_survived, 0)::real / greatest(coalesce(m.duration, 0), 1)::real)
        )::real AS survival_pct
      FROM pubg.participants p
      JOIN pubg.matches m
        ON m.match_id = p.match_id
      LEFT JOIN pubg.rosters r
        ON r.match_id = p.match_id
       AND r.roster_id = p.roster_id
      LEFT JOIN roster_counts rc
        ON rc.match_id = p.match_id
     WHERE p.player_id IS NOT NULL
       AND r.team_id IS NOT NULL
       AND m.created_at IS NOT NULL
),
hist AS (
    SELECT
        *,
        count(*) OVER w AS prior_matches,
        avg(placement_pct) OVER w AS prior_placement_pct,
        avg(top10) OVER w AS prior_top10_rate,
        avg(win) OVER w AS prior_win_rate,
        avg(kills) OVER w AS prior_kills_avg,
        avg(assists) OVER w AS prior_assists_avg,
        avg(damage) OVER w AS prior_damage_avg,
        avg(survival_pct) OVER w AS prior_survival_pct
      FROM base
      WINDOW w AS (
        PARTITION BY {partition_cols}
        ORDER BY match_created_at, match_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
      )
),
scored AS (
    SELECT
        *,
        CASE
            WHEN prior_matches >= %(min_prior_matches)s THEN (
                0.35 * coalesce(prior_placement_pct, 0.0)
              + 0.20 * coalesce(prior_top10_rate, 0.0)
              + 0.20 * (1.0 - exp(-greatest(coalesce(prior_damage_avg, 0.0), 0.0) / 250.0))
              + 0.15 * (1.0 - exp(-greatest(coalesce(prior_kills_avg, 0.0), 0.0) / 2.0))
              + 0.10 * coalesce(prior_survival_pct, 0.0)
            )::real
            ELSE NULL
        END AS prior_strength_raw,
        (prior_matches >= %(min_prior_matches)s) AS proxy_ready
      FROM hist
)
INSERT INTO stdb.player_skill_proxy (
    match_id, player_id, roster_id, team_id, match_created_at, game_mode, map_name,
    tier_source, tier_policy, min_prior_matches,
    prior_matches, prior_placement_pct, prior_top10_rate, prior_win_rate,
    prior_kills_avg, prior_assists_avg, prior_damage_avg, prior_survival_pct,
    prior_strength_raw, proxy_ready
)
SELECT
    match_id, player_id, roster_id, team_id, match_created_at, game_mode, map_name,
    'historical_proxy', %(tier_policy)s, %(min_prior_matches)s,
    prior_matches::integer,
    prior_placement_pct::real,
    prior_top10_rate::real,
    prior_win_rate::real,
    prior_kills_avg::real,
    prior_assists_avg::real,
    prior_damage_avg::real,
    prior_survival_pct::real,
    prior_strength_raw::real,
    proxy_ready
  FROM scored
ON CONFLICT (match_id, player_id, tier_source, tier_policy) DO UPDATE SET
    roster_id = EXCLUDED.roster_id,
    team_id = EXCLUDED.team_id,
    match_created_at = EXCLUDED.match_created_at,
    game_mode = EXCLUDED.game_mode,
    map_name = EXCLUDED.map_name,
    min_prior_matches = EXCLUDED.min_prior_matches,
    prior_matches = EXCLUDED.prior_matches,
    prior_placement_pct = EXCLUDED.prior_placement_pct,
    prior_top10_rate = EXCLUDED.prior_top10_rate,
    prior_win_rate = EXCLUDED.prior_win_rate,
    prior_kills_avg = EXCLUDED.prior_kills_avg,
    prior_assists_avg = EXCLUDED.prior_assists_avg,
    prior_damage_avg = EXCLUDED.prior_damage_avg,
    prior_survival_pct = EXCLUDED.prior_survival_pct,
    prior_strength_raw = EXCLUDED.prior_strength_raw,
    proxy_ready = EXCLUDED.proxy_ready,
    created_at = now();
"""


TEAM_SQL = """
WITH team_raw AS (
    SELECT
        p.match_id,
        p.team_id,
        min(p.match_created_at) AS match_created_at,
        min(p.game_mode) AS game_mode,
        min(p.map_name) AS map_name,
        count(*)::integer AS roster_players,
        count(*) FILTER (WHERE p.proxy_ready)::integer AS eligible_players,
        (count(*) FILTER (WHERE p.proxy_ready)::real / greatest(count(*)::real, 1.0))::real AS coverage_ratio,
        avg(p.prior_matches::real) FILTER (WHERE p.proxy_ready)::real AS player_prior_matches_avg,
        max(p.prior_matches) FILTER (WHERE p.proxy_ready)::integer AS player_prior_matches_max,
        avg(p.prior_placement_pct) FILTER (WHERE p.proxy_ready)::real AS team_prior_placement_pct,
        avg(p.prior_top10_rate) FILTER (WHERE p.proxy_ready)::real AS team_prior_top10_rate,
        avg(p.prior_win_rate) FILTER (WHERE p.proxy_ready)::real AS team_prior_win_rate,
        avg(p.prior_kills_avg) FILTER (WHERE p.proxy_ready)::real AS team_prior_kills_avg,
        avg(p.prior_damage_avg) FILTER (WHERE p.proxy_ready)::real AS team_prior_damage_avg,
        avg(p.prior_survival_pct) FILTER (WHERE p.proxy_ready)::real AS team_prior_survival_pct,
        avg(p.prior_strength_raw) FILTER (WHERE p.proxy_ready)::real AS team_prior_strength_raw,
        (count(*) FILTER (WHERE p.proxy_ready) > 0) AS proxy_ready
      FROM stdb.player_skill_proxy p
     WHERE p.tier_source = 'historical_proxy'
       AND p.tier_policy = %(tier_policy)s
     GROUP BY p.match_id, p.team_id
),
eligible_bucket AS (
    SELECT
        match_id,
        team_id,
        ntile(5) OVER (
            PARTITION BY game_mode
            ORDER BY team_prior_strength_raw
        )::integer AS team_skill_bucket_q_global
      FROM team_raw
     WHERE proxy_ready
       AND team_prior_strength_raw IS NOT NULL
),
bucketed AS (
    SELECT
        tr.*,
        eb.team_skill_bucket_q_global,
        CASE eb.team_skill_bucket_q_global
            WHEN 1 THEN 'q1_low'
            WHEN 2 THEN 'q2_mid_low'
            WHEN 3 THEN 'q3_mid'
            WHEN 4 THEN 'q4_mid_high'
            WHEN 5 THEN 'q5_high'
            ELSE 'unknown'
        END AS team_skill_bucket_global
      FROM team_raw tr
      LEFT JOIN eligible_bucket eb
        ON eb.match_id = tr.match_id
       AND eb.team_id = tr.team_id
)
INSERT INTO stdb.team_skill_proxy (
    match_id, team_id, match_created_at, game_mode, map_name,
    tier_source, tier_policy, min_prior_matches,
    roster_players, eligible_players, coverage_ratio,
    player_prior_matches_avg, player_prior_matches_max,
    team_prior_placement_pct, team_prior_top10_rate, team_prior_win_rate,
    team_prior_kills_avg, team_prior_damage_avg, team_prior_survival_pct,
    team_prior_strength_raw, team_skill_bucket_q_global, team_skill_bucket_global,
    proxy_ready
)
SELECT
    match_id, team_id, match_created_at, game_mode, map_name,
    'historical_proxy', %(tier_policy)s, %(min_prior_matches)s,
    roster_players, eligible_players, coverage_ratio,
    player_prior_matches_avg, player_prior_matches_max,
    team_prior_placement_pct, team_prior_top10_rate, team_prior_win_rate,
    team_prior_kills_avg, team_prior_damage_avg, team_prior_survival_pct,
    team_prior_strength_raw, team_skill_bucket_q_global, team_skill_bucket_global,
    proxy_ready
  FROM bucketed
ON CONFLICT (match_id, team_id, tier_source, tier_policy) DO UPDATE SET
    match_created_at = EXCLUDED.match_created_at,
    game_mode = EXCLUDED.game_mode,
    map_name = EXCLUDED.map_name,
    min_prior_matches = EXCLUDED.min_prior_matches,
    roster_players = EXCLUDED.roster_players,
    eligible_players = EXCLUDED.eligible_players,
    coverage_ratio = EXCLUDED.coverage_ratio,
    player_prior_matches_avg = EXCLUDED.player_prior_matches_avg,
    player_prior_matches_max = EXCLUDED.player_prior_matches_max,
    team_prior_placement_pct = EXCLUDED.team_prior_placement_pct,
    team_prior_top10_rate = EXCLUDED.team_prior_top10_rate,
    team_prior_win_rate = EXCLUDED.team_prior_win_rate,
    team_prior_kills_avg = EXCLUDED.team_prior_kills_avg,
    team_prior_damage_avg = EXCLUDED.team_prior_damage_avg,
    team_prior_survival_pct = EXCLUDED.team_prior_survival_pct,
    team_prior_strength_raw = EXCLUDED.team_prior_strength_raw,
    team_skill_bucket_q_global = EXCLUDED.team_skill_bucket_q_global,
    team_skill_bucket_global = EXCLUDED.team_skill_bucket_global,
    proxy_ready = EXCLUDED.proxy_ready,
    created_at = now();
"""


SUMMARY_SQL = """
SELECT
    %(tier_policy)s AS tier_policy,
    count(*)::integer AS team_rows,
    count(*) FILTER (WHERE proxy_ready)::integer AS ready_team_rows,
    avg(coverage_ratio)::real AS avg_coverage_ratio,
    avg(team_prior_strength_raw)::real AS avg_strength,
    min(team_prior_strength_raw)::real AS min_strength,
    max(team_prior_strength_raw)::real AS max_strength
  FROM stdb.team_skill_proxy
 WHERE tier_source = 'historical_proxy'
   AND tier_policy = %(tier_policy)s;
"""


BUCKET_SQL = """
SELECT
    game_mode,
    coalesce(team_skill_bucket_global, 'unknown') AS bucket,
    count(*)::integer AS team_rows,
    avg(team_prior_strength_raw)::real AS avg_strength
  FROM stdb.team_skill_proxy
 WHERE tier_source = 'historical_proxy'
   AND tier_policy = %(tier_policy)s
 GROUP BY game_mode, coalesce(team_skill_bucket_global, 'unknown')
 ORDER BY game_mode, bucket;
"""


def _tier_policy(mode_scope: str, min_prior_matches: int) -> str:
    return f"historical_{mode_scope}_v1_min{min_prior_matches}"


def _partition_cols(mode_scope: str) -> str:
    if mode_scope == "all":
        return "player_id"
    if mode_scope == "same-mode":
        return "player_id, game_mode"
    raise ValueError(f"unknown mode_scope: {mode_scope}")


def build_proxy(
    *,
    mode_scope: str,
    min_prior_matches: int,
    refresh: bool,
) -> dict[str, Any]:
    tier_policy = _tier_policy(mode_scope, min_prior_matches)
    params = {"tier_policy": tier_policy, "min_prior_matches": int(min_prior_matches)}
    started = time.time()
    conn = connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(SCHEMA_SQL)
                if refresh:
                    cur.execute(
                        """
                        DELETE FROM stdb.team_skill_proxy
                         WHERE tier_source = 'historical_proxy'
                           AND tier_policy = %(tier_policy)s
                        """,
                        params,
                    )
                    cur.execute(
                        """
                        DELETE FROM stdb.player_skill_proxy
                         WHERE tier_source = 'historical_proxy'
                           AND tier_policy = %(tier_policy)s
                        """,
                        params,
                    )

                player_sql = PLAYER_SQL.format(partition_cols=_partition_cols(mode_scope))
                logger.info("building player skill proxy policy=%s", tier_policy)
                cur.execute(player_sql, params)
                logger.info("building team skill proxy policy=%s", tier_policy)
                cur.execute(TEAM_SQL, params)
                cur.execute(SUMMARY_SQL, params)
                summary = dict(cur.fetchone() or {})
                cur.execute(BUCKET_SQL, params)
                buckets = [dict(row) for row in cur.fetchall()]

        summary["elapsed_s"] = round(time.time() - started, 3)
        summary["buckets"] = buckets
        return summary
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode-scope",
        choices=["same-mode", "all"],
        default="same-mode",
        help="Whether a player's prior history is mode-specific or pooled across modes.",
    )
    parser.add_argument(
        "--min-prior-matches",
        type=int,
        default=3,
        help="Minimum prior matches required before a player contributes to team proxy.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete and rebuild rows for the selected tier policy.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = build_proxy(
        mode_scope=args.mode_scope,
        min_prior_matches=args.min_prior_matches,
        refresh=args.refresh,
    )
    print(summary)


if __name__ == "__main__":
    main()
