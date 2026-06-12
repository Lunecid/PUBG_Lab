"""Update map/mode data availability summary for phase-level GNN runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stdb.conn import connect
from stdb.queries import PHASE_N_MAX


DEFAULT_OUT = Path("model_stgcn/checkpoints/domain_runs/data_availability_summary.csv")

FIELDS = [
    "map_name",
    "game_mode",
    "raw_fetched",
    "base_done",
    "base_with_team_positions",
    "phase_feature_done",
    "target_label_done",
    "edge_cache_done",
    f"graph_eligible_n{PHASE_N_MAX}",
    "not_phase_featured",
    "missing_edges_after_features",
]


def fetch_counts(cur, sql: str, params: tuple[object, ...] = ()) -> dict[tuple[str, str], int]:
    cur.execute(sql, params)
    return {(row[0], row[1]): int(row[2]) for row in cur.fetchall()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with connect() as conn:
        cur = conn.cursor()

        raw = fetch_counts(
            cur,
            """
            SELECT m.map_name, m.game_mode, count(DISTINCT m.match_id)
              FROM pubg.matches m
              JOIN pubg.collection_log c ON c.match_id = m.match_id
             WHERE c.telemetry_fetched = TRUE
             GROUP BY m.map_name, m.game_mode
            """,
        )
        base = fetch_counts(
            cur,
            """
            SELECT map_name, game_mode, count(*)
              FROM stdb.matches
             GROUP BY map_name, game_mode
            """,
        )
        base_with_positions = fetch_counts(
            cur,
            """
            SELECT m.map_name, m.game_mode, count(*)
              FROM stdb.matches m
             WHERE EXISTS (
                   SELECT 1
                     FROM stdb.moving_objects mo
                    WHERE mo.match_id = m.match_id
                      AND mo.agent_kind = 'team'
             )
             GROUP BY m.map_name, m.game_mode
            """,
        )
        phase_features = fetch_counts(
            cur,
            """
            SELECT m.map_name, m.game_mode, count(DISTINCT l.match_id)
              FROM stdb.team_phase_labels l
              JOIN stdb.matches m ON m.match_id = l.match_id
             GROUP BY m.map_name, m.game_mode
            """,
        )
        target_labels = fetch_counts(
            cur,
            """
            SELECT m.map_name, m.game_mode, count(DISTINCT l.match_id)
              FROM stdb.team_phase_labels l
              JOIN stdb.matches m ON m.match_id = l.match_id
             WHERE l.target_defined
             GROUP BY m.map_name, m.game_mode
            """,
        )
        edge_cache = fetch_counts(
            cur,
            """
            SELECT m.map_name, m.game_mode, count(DISTINCT e.match_id)
              FROM stdb.team_phase_edge_cache e
              JOIN stdb.matches m ON m.match_id = e.match_id
             GROUP BY m.map_name, m.game_mode
            """,
        )
        graph_eligible = fetch_counts(
            cur,
            """
            SELECT m.map_name, m.game_mode, count(DISTINCT m.match_id)
              FROM stdb.matches m
             WHERE m.num_teams >= 4
               AND m.num_teams <= %s
               AND EXISTS (
                   SELECT 1
                     FROM stdb.team_phase_labels l
                    WHERE l.match_id = m.match_id
                      AND l.target_defined
               )
               AND EXISTS (
                   SELECT 1
                     FROM stdb.team_phase_edge_cache e
                    WHERE e.match_id = m.match_id
               )
             GROUP BY m.map_name, m.game_mode
            """,
            (PHASE_N_MAX,),
        )
        cur.close()

    keys = sorted(
        set(raw)
        | set(base)
        | set(base_with_positions)
        | set(phase_features)
        | set(target_labels)
        | set(edge_cache)
        | set(graph_eligible),
        key=lambda k: (-raw.get(k, 0), k[0], k[1]),
    )

    rows = []
    for key in keys:
        raw_fetched = raw.get(key, 0)
        phase_done = phase_features.get(key, 0)
        edges_done = edge_cache.get(key, 0)
        rows.append(
            {
                "map_name": key[0],
                "game_mode": key[1],
                "raw_fetched": raw_fetched,
                "base_done": base.get(key, 0),
                "base_with_team_positions": base_with_positions.get(key, 0),
                "phase_feature_done": phase_done,
                "target_label_done": target_labels.get(key, 0),
                "edge_cache_done": edges_done,
                f"graph_eligible_n{PHASE_N_MAX}": graph_eligible.get(key, 0),
                "not_phase_featured": raw_fetched - phase_done,
                "missing_edges_after_features": phase_done - edges_done,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = {
        "raw_fetched": sum(row["raw_fetched"] for row in rows),
        "base_done": sum(row["base_done"] for row in rows),
        "phase_feature_done": sum(row["phase_feature_done"] for row in rows),
        "target_label_done": sum(row["target_label_done"] for row in rows),
        "edge_cache_done": sum(row["edge_cache_done"] for row in rows),
        f"graph_eligible_n{PHASE_N_MAX}": sum(row[f"graph_eligible_n{PHASE_N_MAX}"] for row in rows),
    }
    print(f"wrote {args.out}")
    print(total)


if __name__ == "__main__":
    main()
