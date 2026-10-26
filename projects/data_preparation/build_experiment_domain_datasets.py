"""Create experiment-ready domain datasets from the raw PUBG population.

The raw corpus is grouped by map, perspective, and team size.  For each domain,
this script writes:

* raw match ids from ``pubg.matches`` with fetched telemetry,
* phase-feature eligible ids,
* graph-eligible ids for DynamicSTGNN,
* deterministic train/validation/test split manifests.

The output preserves the raw population denominator while making the actual
experiment inputs explicit and reproducible.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[2]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pubg_lab.survival.data.split_access import split_ids, write_split_manifest
from pubg_lab.data.stdb.connection.conn import connect
from pubg_lab.data.stdb.queries.queries import PHASE_N_MAX


DEFAULT_OUT_DIR = Path("model_stgcn/experiment_datasets/raw_domain_population_seed42")
DEFAULT_DOC = Path("docs/EXPERIMENT_DATASETS_BY_DOMAIN.md")


def perspective(game_mode: str) -> str:
    return "FPP" if game_mode.endswith("-fpp") else "TPP"


def team_size(game_mode: str) -> str:
    base = game_mode.removesuffix("-fpp")
    if base in {"solo", "duo", "squad"}:
        return base
    return base or "unknown"


def domain_key(map_name: str, game_mode: str) -> tuple[str, str, str, str]:
    return (map_name, perspective(game_mode), team_size(game_mode), game_mode)


def slug(parts: Iterable[str]) -> str:
    text = "__".join(parts)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def fetch_domain_ids() -> tuple[
    dict[tuple[str, str, str, str], list[str]],
    dict[tuple[str, str, str, str], list[str]],
    dict[tuple[str, str, str, str], list[str]],
]:
    raw: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    feature: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    graph: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)

    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.map_name, m.game_mode, m.match_id
              FROM pubg.matches m
              JOIN pubg.collection_log c ON c.match_id = m.match_id
             WHERE c.telemetry_fetched = TRUE
             ORDER BY m.map_name, m.game_mode, m.created_at, m.match_id
            """
        )
        for map_name, game_mode, match_id in cur.fetchall():
            raw[domain_key(map_name, game_mode)].append(match_id)

        cur.execute(
            """
            SELECT DISTINCT f.map_name, f.game_mode, f.match_id
              FROM stdb.v_team_phase_feature_long f
             WHERE f.target_defined
             ORDER BY f.map_name, f.game_mode, f.match_id
            """
        )
        for map_name, game_mode, match_id in cur.fetchall():
            feature[domain_key(map_name, game_mode)].append(match_id)

        cur.execute(
            """
            SELECT m.map_name, m.game_mode, m.match_id
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
             ORDER BY m.map_name, m.game_mode, m.started_at, m.match_id
            """,
            (PHASE_N_MAX,),
        )
        for map_name, game_mode, match_id in cur.fetchall():
            graph[domain_key(map_name, game_mode)].append(match_id)
        cur.close()

    return raw, feature, graph


def write_id_file(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def write_split(
    path: Path, ids: list[str], *, seed: int, val_frac: float, test_frac: float, metadata: dict
) -> tuple[int, int, int]:
    if not ids:
        return (0, 0, 0)
    train_ids, val_ids, test_ids = split_ids(ids, seed=seed, val_frac=val_frac, test_frac=test_frac)
    write_split_manifest(
        path,
        ids=ids,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
        metadata=metadata,
    )
    return (len(train_ids), len(val_ids), len(test_ids))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "domain_slug",
        "map_name",
        "perspective",
        "team_size",
        "game_mode",
        "raw_count",
        "feature_eligible_count",
        "graph_eligible_count",
        "raw_to_graph_rate",
        "graph_train",
        "graph_val",
        "graph_test",
        "feature_train",
        "feature_val",
        "feature_test",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def status_for(raw_count: int, graph_count: int, min_graph: int) -> str:
    if graph_count >= min_graph:
        return "experiment_ready"
    if graph_count > 0:
        return "small_auxiliary"
    if raw_count > 0:
        return "raw_only_not_ready"
    return "empty"


def markdown_table(
    rows: list[dict[str, object]], fields: list[str], *, limit: int | None = None
) -> str:
    selected = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| "
        + " | ".join(
            "---:"
            if field.endswith("_count")
            or field in {"raw_count", "graph_train", "graph_val", "graph_test"}
            else "---"
            for field in fields
        )
        + " |",
    ]
    for row in selected:
        vals = []
        for field in fields:
            val = row[field]
            if isinstance(val, int):
                vals.append(f"{val:,}")
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, object]], out_dir: Path, min_graph: int) -> None:
    total = {
        "domain_slug": "TOTAL",
        "raw_count": sum(int(r["raw_count"]) for r in rows),
        "feature_eligible_count": sum(int(r["feature_eligible_count"]) for r in rows),
        "graph_eligible_count": sum(int(r["graph_eligible_count"]) for r in rows),
    }
    ready = [r for r in rows if r["status"] == "experiment_ready"]
    small = [r for r in rows if r["status"] == "small_auxiliary"]
    raw_only = [r for r in rows if r["status"] == "raw_only_not_ready"]

    lines = [
        "# Experiment Datasets by Raw Domain",
        "",
        "이 문서는 raw fetched 전체 모집단을 map / perspective / team size 도메인으로 나누고, 각 도메인에서 실제 실험 가능한 match id와 split manifest를 생성한 결과이다.",
        "",
        f"- Output root: `{out_dir}`",
        f"- Minimum graph-eligible threshold for `experiment_ready`: `{min_graph}` matches",
        "",
        "## Overall",
        "",
        markdown_table(
            [total], ["domain_slug", "raw_count", "feature_eligible_count", "graph_eligible_count"]
        ),
        "",
        "## Experiment-Ready Domains",
        "",
        markdown_table(
            ready,
            [
                "domain_slug",
                "raw_count",
                "feature_eligible_count",
                "graph_eligible_count",
                "graph_train",
                "graph_val",
                "graph_test",
                "status",
            ],
        ),
        "",
        "## Small Auxiliary Domains",
        "",
        markdown_table(
            small,
            [
                "domain_slug",
                "raw_count",
                "feature_eligible_count",
                "graph_eligible_count",
                "graph_train",
                "graph_val",
                "graph_test",
                "status",
            ],
            limit=80,
        ),
        "",
        "## Raw-Only Domains Not Ready for Current Experiments",
        "",
        markdown_table(
            raw_only,
            [
                "domain_slug",
                "raw_count",
                "feature_eligible_count",
                "graph_eligible_count",
                "status",
            ],
        ),
        "",
        "## How to Use",
        "",
        "- `raw_match_ids.txt`: raw fetched 모집단 ID.",
        "- `feature_match_ids.txt`: feature ablation/tabular experiment에 사용할 수 있는 ID.",
        "- `graph_match_ids.txt`: DynamicSTGNN 학습/평가에 사용할 수 있는 ID.",
        "- `feature_split_seed*.json`: feature-level experiment split.",
        "- `graph_split_seed*.json`: graph/STGNN experiment split.",
        "- 논문용 main 재실험은 `experiment_ready` 도메인 또는 전체 `graph_eligible_count` 기준으로 수행한다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-graph", type=int, default=300)
    args = parser.parse_args()

    raw, feature, graph = fetch_domain_ids()
    keys = sorted(set(raw) | set(feature) | set(graph), key=lambda k: (k[0], k[1], k[2], k[3]))

    summary_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "phase_n_max": PHASE_N_MAX,
        "domains": {},
    }

    for key in keys:
        map_name, persp, size, game_mode = key
        domain_slug = slug([map_name, persp, size])
        domain_dir = args.out_dir / "domains" / domain_slug
        raw_ids = raw.get(key, [])
        feature_ids = feature.get(key, [])
        graph_ids = graph.get(key, [])

        write_id_file(domain_dir / "raw_match_ids.txt", raw_ids)
        write_id_file(domain_dir / "feature_match_ids.txt", feature_ids)
        write_id_file(domain_dir / "graph_match_ids.txt", graph_ids)

        metadata = {
            "map_name": map_name,
            "perspective": persp,
            "team_size": size,
            "game_mode": game_mode,
            "raw_count": len(raw_ids),
            "feature_eligible_count": len(feature_ids),
            "graph_eligible_count": len(graph_ids),
            "source": "projects/data_preparation/build_experiment_domain_datasets.py",
        }
        feature_split_counts = write_split(
            domain_dir / f"feature_split_seed{args.seed}.json",
            feature_ids,
            seed=args.seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            metadata={**metadata, "split_type": "feature"},
        )
        graph_split_counts = write_split(
            domain_dir / f"graph_split_seed{args.seed}.json",
            graph_ids,
            seed=args.seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            metadata={**metadata, "split_type": "graph"},
        )

        graph_count = len(graph_ids)
        raw_count = len(raw_ids)
        row = {
            "domain_slug": domain_slug,
            "map_name": map_name,
            "perspective": persp,
            "team_size": size,
            "game_mode": game_mode,
            "raw_count": raw_count,
            "feature_eligible_count": len(feature_ids),
            "graph_eligible_count": graph_count,
            "raw_to_graph_rate": (graph_count / raw_count) if raw_count else 0.0,
            "graph_train": graph_split_counts[0],
            "graph_val": graph_split_counts[1],
            "graph_test": graph_split_counts[2],
            "feature_train": feature_split_counts[0],
            "feature_val": feature_split_counts[1],
            "feature_test": feature_split_counts[2],
            "status": status_for(raw_count, graph_count, args.min_graph),
        }
        summary_rows.append(row)
        manifest["domains"][domain_slug] = {
            **metadata,
            "status": row["status"],
            "paths": {
                "raw_match_ids": str(domain_dir / "raw_match_ids.txt"),
                "feature_match_ids": str(domain_dir / "feature_match_ids.txt"),
                "graph_match_ids": str(domain_dir / "graph_match_ids.txt"),
                "feature_split": str(domain_dir / f"feature_split_seed{args.seed}.json")
                if feature_ids
                else None,
                "graph_split": str(domain_dir / f"graph_split_seed{args.seed}.json")
                if graph_ids
                else None,
            },
        }

    summary_rows.sort(
        key=lambda row: (
            str(row["status"]) != "experiment_ready",
            -int(row["graph_eligible_count"]),
            -int(row["raw_count"]),
            str(row["domain_slug"]),
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "domain_experiment_summary.csv"
    manifest_path = args.out_dir / "domain_experiment_manifest.json"
    write_summary_csv(summary_path, summary_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.doc, summary_rows, args.out_dir, args.min_graph)

    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
