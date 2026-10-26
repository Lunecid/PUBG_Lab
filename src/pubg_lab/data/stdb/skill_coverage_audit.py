"""Audit whether current PUBG data can support repeated-player skill proxies.

This is a read-only diagnostic. It does not build a tier proxy. Its purpose is
to decide whether tier should remain future work because repeated player
coverage is too low for a stable latent rating.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json

from pubg_lab.data.stdb.conn import get_cur


AUDIT_SQL = """
WITH player_matches AS (
    SELECT
        p.player_id,
        m.game_mode,
        count(DISTINCT p.match_id)::integer AS n_matches
      FROM pubg.participants p
      JOIN pubg.matches m
        ON m.match_id = p.match_id
     WHERE p.player_id IS NOT NULL
       AND m.created_at IS NOT NULL
     GROUP BY p.player_id, m.game_mode
),
overall AS (
    SELECT
        count(*)::integer AS player_mode_rows,
        count(*) FILTER (WHERE n_matches >= 2)::integer AS repeated_player_mode_rows,
        count(*) FILTER (WHERE n_matches >= 3)::integer AS ge3_player_mode_rows,
        count(*) FILTER (WHERE n_matches >= 5)::integer AS ge5_player_mode_rows,
        avg(n_matches)::real AS avg_matches_per_player_mode,
        max(n_matches)::integer AS max_matches_per_player_mode
      FROM player_matches
),
by_mode AS (
    SELECT
        game_mode,
        count(*)::integer AS player_rows,
        count(*) FILTER (WHERE n_matches >= 2)::integer AS repeated_players,
        count(*) FILTER (WHERE n_matches >= 3)::integer AS ge3_players,
        count(*) FILTER (WHERE n_matches >= 5)::integer AS ge5_players,
        avg(n_matches)::real AS avg_matches,
        max(n_matches)::integer AS max_matches
      FROM player_matches
     GROUP BY game_mode
),
participant_coverage AS (
    SELECT
        count(*)::integer AS participant_rows,
        count(*) FILTER (WHERE pm.n_matches >= 2)::integer AS participant_rows_with_repeated_player,
        count(*) FILTER (WHERE pm.n_matches >= 3)::integer AS participant_rows_with_ge3_player,
        count(*) FILTER (WHERE pm.n_matches >= 5)::integer AS participant_rows_with_ge5_player
      FROM pubg.participants p
      JOIN pubg.matches m
        ON m.match_id = p.match_id
      JOIN player_matches pm
        ON pm.player_id = p.player_id
       AND pm.game_mode = m.game_mode
     WHERE p.player_id IS NOT NULL
       AND m.created_at IS NOT NULL
)
SELECT
    (SELECT row_to_json(overall) FROM overall) AS overall,
    (SELECT row_to_json(participant_coverage) FROM participant_coverage) AS participant_coverage,
    (SELECT json_agg(row_to_json(by_mode) ORDER BY player_rows DESC) FROM by_mode) AS by_mode;
"""


def run_audit() -> dict:
    with get_cur(dict_cursor=True) as (_conn, cur):
        cur.execute(AUDIT_SQL)
        row = cur.fetchone()
    return {
        "overall": row["overall"],
        "participant_coverage": row["participant_coverage"],
        "by_mode": row["by_mode"] or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    out = run_audit()
    print(json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
