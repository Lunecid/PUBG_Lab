"""Build opponent-adjusted historical skill rating proxies.

This is the preferred current-data fallback when official PUBG tier/MMR is not
available. It processes matches chronologically and records each team's
pre-match rating before updating player ratings from the match outcome.

The rating is not an official PUBG tier. It is an internal latent skill scale
that adjusts outcome by lobby strength:

    expected(team i beats team j) = logistic(rating_i - rating_j)
    delta_i = K * (observed_placement_score_i - expected_score_i)

The target match itself is never used to construct its pre-match rating.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg2.extras

from pubg_lab.data.stdb.conn import connect


logger = logging.getLogger("stdb.skill_rating_proxy_etl")


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS stdb;

CREATE TABLE IF NOT EXISTS stdb.player_skill_rating_proxy (
    match_id                    TEXT        NOT NULL,
    player_id                   TEXT        NOT NULL,
    roster_id                   TEXT,
    team_id                     INTEGER     NOT NULL,
    match_created_at            TIMESTAMPTZ NOT NULL,
    game_mode                   TEXT        NOT NULL,
    map_name                    TEXT        NOT NULL,
    rating_source               TEXT        NOT NULL DEFAULT 'opponent_adjusted_history',
    rating_policy               TEXT        NOT NULL,
    min_prior_matches           INTEGER     NOT NULL,

    pre_rating                  REAL        NOT NULL,
    prior_matches               INTEGER     NOT NULL,
    team_pre_rating             REAL        NOT NULL,
    lobby_pre_rating            REAL,
    expected_score              REAL,
    observed_score              REAL,
    rating_delta                REAL,
    proxy_ready                 BOOLEAN     NOT NULL,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, player_id, rating_source, rating_policy)
);

CREATE INDEX IF NOT EXISTS ix_psrp_player_time
    ON stdb.player_skill_rating_proxy (player_id, match_created_at);
CREATE INDEX IF NOT EXISTS ix_psrp_match_team
    ON stdb.player_skill_rating_proxy (match_id, team_id);
CREATE INDEX IF NOT EXISTS ix_psrp_policy
    ON stdb.player_skill_rating_proxy (rating_policy, proxy_ready);

CREATE TABLE IF NOT EXISTS stdb.team_skill_rating_proxy (
    match_id                    TEXT        NOT NULL,
    team_id                     INTEGER     NOT NULL,
    match_created_at            TIMESTAMPTZ NOT NULL,
    game_mode                   TEXT        NOT NULL,
    map_name                    TEXT        NOT NULL,
    rating_source               TEXT        NOT NULL DEFAULT 'opponent_adjusted_history',
    rating_policy               TEXT        NOT NULL,
    min_prior_matches           INTEGER     NOT NULL,
    min_coverage_ratio          REAL        NOT NULL,

    roster_players              INTEGER     NOT NULL,
    eligible_players            INTEGER     NOT NULL,
    coverage_ratio              REAL        NOT NULL,
    player_prior_matches_avg    REAL,
    player_prior_matches_max    INTEGER,
    team_pre_rating             REAL        NOT NULL,
    lobby_pre_rating            REAL,
    lobby_rating_std            REAL,
    expected_score              REAL,
    observed_score              REAL,
    rating_delta                REAL,
    team_skill_bucket_q_global  INTEGER,
    team_skill_bucket_global    TEXT,
    proxy_ready                 BOOLEAN     NOT NULL,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id, rating_source, rating_policy)
);

CREATE INDEX IF NOT EXISTS ix_tsrp_match
    ON stdb.team_skill_rating_proxy (match_id, team_id);
CREATE INDEX IF NOT EXISTS ix_tsrp_policy_bucket
    ON stdb.team_skill_rating_proxy (rating_policy, team_skill_bucket_q_global);
CREATE INDEX IF NOT EXISTS ix_tsrp_time
    ON stdb.team_skill_rating_proxy (match_created_at);
"""


FETCH_SQL = """
SELECT
    m.match_id,
    m.created_at AS match_created_at,
    m.game_mode,
    m.map_name,
    p.player_id,
    p.roster_id,
    r.team_id,
    r.rank::integer AS final_rank
  FROM pubg.matches m
  JOIN pubg.rosters r
    ON r.match_id = m.match_id
  JOIN pubg.participants p
    ON p.match_id = m.match_id
   AND p.roster_id = r.roster_id
 WHERE m.created_at IS NOT NULL
   AND p.player_id IS NOT NULL
   AND r.team_id IS NOT NULL
   AND r.rank IS NOT NULL
 ORDER BY m.created_at, m.match_id, r.team_id, p.player_id
"""


PLAYER_INSERT_SQL = """
INSERT INTO stdb.player_skill_rating_proxy (
    match_id, player_id, roster_id, team_id, match_created_at, game_mode, map_name,
    rating_source, rating_policy, min_prior_matches,
    pre_rating, prior_matches, team_pre_rating, lobby_pre_rating,
    expected_score, observed_score, rating_delta, proxy_ready
)
VALUES %s
ON CONFLICT (match_id, player_id, rating_source, rating_policy) DO UPDATE SET
    roster_id = EXCLUDED.roster_id,
    team_id = EXCLUDED.team_id,
    match_created_at = EXCLUDED.match_created_at,
    game_mode = EXCLUDED.game_mode,
    map_name = EXCLUDED.map_name,
    min_prior_matches = EXCLUDED.min_prior_matches,
    pre_rating = EXCLUDED.pre_rating,
    prior_matches = EXCLUDED.prior_matches,
    team_pre_rating = EXCLUDED.team_pre_rating,
    lobby_pre_rating = EXCLUDED.lobby_pre_rating,
    expected_score = EXCLUDED.expected_score,
    observed_score = EXCLUDED.observed_score,
    rating_delta = EXCLUDED.rating_delta,
    proxy_ready = EXCLUDED.proxy_ready,
    created_at = now()
"""


TEAM_INSERT_SQL = """
INSERT INTO stdb.team_skill_rating_proxy (
    match_id, team_id, match_created_at, game_mode, map_name,
    rating_source, rating_policy, min_prior_matches, min_coverage_ratio,
    roster_players, eligible_players, coverage_ratio,
    player_prior_matches_avg, player_prior_matches_max,
    team_pre_rating, lobby_pre_rating, lobby_rating_std,
    expected_score, observed_score, rating_delta, proxy_ready
)
VALUES %s
ON CONFLICT (match_id, team_id, rating_source, rating_policy) DO UPDATE SET
    match_created_at = EXCLUDED.match_created_at,
    game_mode = EXCLUDED.game_mode,
    map_name = EXCLUDED.map_name,
    min_prior_matches = EXCLUDED.min_prior_matches,
    min_coverage_ratio = EXCLUDED.min_coverage_ratio,
    roster_players = EXCLUDED.roster_players,
    eligible_players = EXCLUDED.eligible_players,
    coverage_ratio = EXCLUDED.coverage_ratio,
    player_prior_matches_avg = EXCLUDED.player_prior_matches_avg,
    player_prior_matches_max = EXCLUDED.player_prior_matches_max,
    team_pre_rating = EXCLUDED.team_pre_rating,
    lobby_pre_rating = EXCLUDED.lobby_pre_rating,
    lobby_rating_std = EXCLUDED.lobby_rating_std,
    expected_score = EXCLUDED.expected_score,
    observed_score = EXCLUDED.observed_score,
    rating_delta = EXCLUDED.rating_delta,
    proxy_ready = EXCLUDED.proxy_ready,
    created_at = now()
"""


BUCKET_SQL = """
WITH bucket AS (
    SELECT
        match_id,
        team_id,
        ntile(5) OVER (
            PARTITION BY game_mode
            ORDER BY team_pre_rating
        )::integer AS q
      FROM stdb.team_skill_rating_proxy
     WHERE rating_source = 'opponent_adjusted_history'
       AND rating_policy = %(rating_policy)s
       AND proxy_ready
),
named AS (
    SELECT
        match_id,
        team_id,
        q,
        CASE q
            WHEN 1 THEN 'q1_low'
            WHEN 2 THEN 'q2_mid_low'
            WHEN 3 THEN 'q3_mid'
            WHEN 4 THEN 'q4_mid_high'
            WHEN 5 THEN 'q5_high'
            ELSE 'unknown'
        END AS name
      FROM bucket
)
UPDATE stdb.team_skill_rating_proxy t
   SET team_skill_bucket_q_global = n.q,
       team_skill_bucket_global = n.name
  FROM named n
 WHERE t.match_id = n.match_id
   AND t.team_id = n.team_id
   AND t.rating_source = 'opponent_adjusted_history'
   AND t.rating_policy = %(rating_policy)s
"""


SUMMARY_SQL = """
SELECT
    %(rating_policy)s AS rating_policy,
    count(*)::integer AS team_rows,
    count(*) FILTER (WHERE proxy_ready)::integer AS ready_team_rows,
    avg(coverage_ratio)::real AS avg_coverage_ratio,
    avg(team_pre_rating)::real AS avg_team_rating,
    min(team_pre_rating)::real AS min_team_rating,
    max(team_pre_rating)::real AS max_team_rating
  FROM stdb.team_skill_rating_proxy
 WHERE rating_source = 'opponent_adjusted_history'
   AND rating_policy = %(rating_policy)s
"""


BUCKET_SUMMARY_SQL = """
SELECT
    game_mode,
    coalesce(team_skill_bucket_global, 'unknown') AS bucket,
    count(*)::integer AS team_rows,
    avg(team_pre_rating)::real AS avg_team_rating
  FROM stdb.team_skill_rating_proxy
 WHERE rating_source = 'opponent_adjusted_history'
   AND rating_policy = %(rating_policy)s
 GROUP BY game_mode, coalesce(team_skill_bucket_global, 'unknown')
 ORDER BY game_mode, bucket
"""


@dataclass
class Participant:
    player_id: str
    roster_id: str | None
    team_id: int
    final_rank: int


@dataclass
class MatchBlock:
    match_id: str
    match_created_at: Any
    game_mode: str
    map_name: str
    participants: list[Participant]


def _rating_policy(
    *,
    mode_scope: str,
    min_prior_matches: int,
    min_coverage_ratio: float,
    k_factor: float,
    rating_scale: float,
) -> str:
    cov = int(round(min_coverage_ratio * 100))
    return (
        f"rating_{mode_scope}_v1_min{min_prior_matches}"
        f"_cov{cov}_k{int(k_factor)}_s{int(rating_scale)}"
    )


def _rating_key(player_id: str, game_mode: str, mode_scope: str) -> tuple[str, ...]:
    if mode_scope == "all":
        return (player_id,)
    if mode_scope == "same-mode":
        return (player_id, game_mode)
    raise ValueError(f"unknown mode_scope: {mode_scope}")


def _expected_score(
    team_rating: float, opponent_ratings: list[float], scale: float
) -> float | None:
    if not opponent_ratings:
        return None
    vals = [1.0 / (1.0 + math.pow(10.0, (opp - team_rating) / scale)) for opp in opponent_ratings]
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float | None:
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def _iter_match_blocks(cur):
    cur.execute(FETCH_SQL)
    current_key = None
    current: MatchBlock | None = None
    for row in cur:
        key = row["match_id"]
        if key != current_key:
            if current is not None:
                yield current
            current_key = key
            current = MatchBlock(
                match_id=row["match_id"],
                match_created_at=row["match_created_at"],
                game_mode=row["game_mode"],
                map_name=row["map_name"],
                participants=[],
            )
        assert current is not None
        current.participants.append(
            Participant(
                player_id=row["player_id"],
                roster_id=row["roster_id"],
                team_id=int(row["team_id"]),
                final_rank=int(row["final_rank"]),
            )
        )
    if current is not None:
        yield current


def _flush(cur, player_rows: list[tuple], team_rows: list[tuple]) -> None:
    if player_rows:
        psycopg2.extras.execute_values(cur, PLAYER_INSERT_SQL, player_rows, page_size=5000)
        player_rows.clear()
    if team_rows:
        psycopg2.extras.execute_values(cur, TEAM_INSERT_SQL, team_rows, page_size=5000)
        team_rows.clear()


def build_rating_proxy(
    *,
    mode_scope: str,
    min_prior_matches: int,
    min_coverage_ratio: float,
    default_rating: float,
    k_factor: float,
    rating_scale: float,
    refresh: bool,
    flush_every_matches: int,
) -> dict[str, Any]:
    rating_policy = _rating_policy(
        mode_scope=mode_scope,
        min_prior_matches=min_prior_matches,
        min_coverage_ratio=min_coverage_ratio,
        k_factor=k_factor,
        rating_scale=rating_scale,
    )
    params = {"rating_policy": rating_policy}
    started = time.time()

    ratings: dict[tuple[str, ...], float] = defaultdict(lambda: float(default_rating))
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    player_rows: list[tuple] = []
    team_rows: list[tuple] = []

    conn = connect()
    read_conn = connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(SCHEMA_SQL)
                if refresh:
                    cur.execute(
                        """
                        DELETE FROM stdb.team_skill_rating_proxy
                         WHERE rating_source = 'opponent_adjusted_history'
                           AND rating_policy = %(rating_policy)s
                        """,
                        params,
                    )
                    cur.execute(
                        """
                        DELETE FROM stdb.player_skill_rating_proxy
                         WHERE rating_source = 'opponent_adjusted_history'
                           AND rating_policy = %(rating_policy)s
                        """,
                        params,
                    )

                read_cur = read_conn.cursor(
                    name="skill_rating_source",
                    cursor_factory=psycopg2.extras.RealDictCursor,
                )
                read_cur.itersize = 20000
                logger.info("processing matches policy=%s", rating_policy)

                try:
                    for idx, block in enumerate(_iter_match_blocks(read_cur), start=1):
                        teams: dict[int, list[Participant]] = defaultdict(list)
                        for part in block.participants:
                            teams[part.team_id].append(part)
                        if len(teams) <= 1:
                            continue

                        team_pre: dict[int, float] = {}
                        team_prior_counts: dict[int, list[int]] = {}
                        ranks: dict[int, int] = {}
                        for team_id, parts in teams.items():
                            pre_vals = []
                            prior_vals = []
                            rank_vals = []
                            for part in parts:
                                key = _rating_key(part.player_id, block.game_mode, mode_scope)
                                pre_vals.append(float(ratings[key]))
                                prior_vals.append(int(counts[key]))
                                rank_vals.append(part.final_rank)
                            team_pre[team_id] = sum(pre_vals) / len(pre_vals)
                            team_prior_counts[team_id] = prior_vals
                            ranks[team_id] = min(rank_vals)

                        all_team_ratings = list(team_pre.values())
                        n_teams = len(teams)
                        team_records: dict[int, dict[str, Any]] = {}
                        for team_id, parts in teams.items():
                            opponents = [r for tid, r in team_pre.items() if tid != team_id]
                            expected = _expected_score(team_pre[team_id], opponents, rating_scale)
                            observed = (n_teams - ranks[team_id]) / max(n_teams - 1, 1)
                            delta = float(k_factor) * (
                                observed - (expected if expected is not None else 0.5)
                            )
                            prior_vals = team_prior_counts[team_id]
                            eligible = sum(1 for c in prior_vals if c >= min_prior_matches)
                            coverage = eligible / max(len(prior_vals), 1)
                            proxy_ready = coverage >= min_coverage_ratio
                            lobby_rating = sum(all_team_ratings) / len(all_team_ratings)
                            record = {
                                "expected": expected,
                                "observed": observed,
                                "delta": delta,
                                "eligible": eligible,
                                "coverage": coverage,
                                "proxy_ready": proxy_ready,
                                "lobby_rating": lobby_rating,
                                "lobby_std": _std(all_team_ratings),
                                "prior_avg": sum(prior_vals) / len(prior_vals),
                                "prior_max": max(prior_vals),
                            }
                            team_records[team_id] = record
                            team_rows.append(
                                (
                                    block.match_id,
                                    team_id,
                                    block.match_created_at,
                                    block.game_mode,
                                    block.map_name,
                                    "opponent_adjusted_history",
                                    rating_policy,
                                    int(min_prior_matches),
                                    float(min_coverage_ratio),
                                    len(parts),
                                    eligible,
                                    coverage,
                                    record["prior_avg"],
                                    record["prior_max"],
                                    team_pre[team_id],
                                    lobby_rating,
                                    record["lobby_std"],
                                    expected,
                                    observed,
                                    delta,
                                    proxy_ready,
                                )
                            )

                        for team_id, parts in teams.items():
                            record = team_records[team_id]
                            for part in parts:
                                key = _rating_key(part.player_id, block.game_mode, mode_scope)
                                prior = int(counts[key])
                                pre = float(ratings[key])
                                proxy_ready = prior >= min_prior_matches
                                player_rows.append(
                                    (
                                        block.match_id,
                                        part.player_id,
                                        part.roster_id,
                                        team_id,
                                        block.match_created_at,
                                        block.game_mode,
                                        block.map_name,
                                        "opponent_adjusted_history",
                                        rating_policy,
                                        int(min_prior_matches),
                                        pre,
                                        prior,
                                        team_pre[team_id],
                                        record["lobby_rating"],
                                        record["expected"],
                                        record["observed"],
                                        record["delta"],
                                        proxy_ready,
                                    )
                                )
                                ratings[key] = pre + record["delta"]
                                counts[key] = prior + 1

                        if idx % flush_every_matches == 0:
                            _flush(cur, player_rows, team_rows)
                            logger.info("processed %d matches", idx)
                finally:
                    read_cur.close()

                _flush(cur, player_rows, team_rows)
                cur.execute(BUCKET_SQL, params)
                cur.execute(SUMMARY_SQL, params)
                summary = dict(cur.fetchone() or {})
                cur.execute(BUCKET_SUMMARY_SQL, params)
                buckets = [dict(row) for row in cur.fetchall()]

        summary["elapsed_s"] = round(time.time() - started, 3)
        summary["buckets"] = buckets
        return summary
    finally:
        read_conn.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-scope", choices=["same-mode", "all"], default="same-mode")
    parser.add_argument("--min-prior-matches", type=int, default=3)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.5)
    parser.add_argument("--default-rating", type=float, default=1500.0)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--rating-scale", type=float, default=400.0)
    parser.add_argument("--flush-every-matches", type=int, default=500)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = build_rating_proxy(
        mode_scope=args.mode_scope,
        min_prior_matches=args.min_prior_matches,
        min_coverage_ratio=args.min_coverage_ratio,
        default_rating=args.default_rating,
        k_factor=args.k_factor,
        rating_scale=args.rating_scale,
        refresh=args.refresh,
        flush_every_matches=args.flush_every_matches,
    )
    print(summary)


if __name__ == "__main__":
    main()
