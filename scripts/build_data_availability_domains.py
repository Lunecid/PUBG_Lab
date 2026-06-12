"""Build availability tables by perspective, map, and team size.

The raw availability summary is keyed by ``map_name`` and ``game_mode``.
For paper planning, it is easier to reason in terms of:

* perspective: TPP / FPP
* team_size: solo / duo / squad / other modes
* map_name

This script derives those fields, writes detailed and aggregated CSV files,
and emits a short Markdown report.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_IN = Path("model_stgcn/checkpoints/domain_runs/data_availability_summary.csv")
DEFAULT_OUT_DIR = Path("model_stgcn/checkpoints/domain_runs")
DEFAULT_DOC = Path("docs/DATA_AVAILABILITY_BY_DOMAIN.md")

COUNT_FIELDS = [
    "raw_fetched",
    "base_done",
    "base_with_team_positions",
    "phase_feature_done",
    "target_label_done",
    "edge_cache_done",
    "graph_eligible_n128",
    "not_phase_featured",
    "missing_edges_after_features",
]


def perspective(game_mode: str) -> str:
    return "FPP" if game_mode.endswith("-fpp") else "TPP"


def team_size(game_mode: str) -> str:
    base = game_mode.removesuffix("-fpp")
    if base in {"solo", "duo", "squad"}:
        return base
    return base or "unknown"


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, object] = {
                "perspective": perspective(row["game_mode"]),
                "map_name": row["map_name"],
                "team_size": team_size(row["game_mode"]),
                "game_mode": row["game_mode"],
            }
            for field in COUNT_FIELDS:
                out[field] = int(row[field])
            rows.append(out)
    return rows


def aggregate(rows: list[dict[str, object]], group_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        if key not in grouped:
            grouped[key] = {field: row[field] for field in group_fields}
            for count_field in COUNT_FIELDS:
                grouped[key][count_field] = 0
        for count_field in COUNT_FIELDS:
            grouped[key][count_field] = int(grouped[key][count_field]) + int(row[count_field])

    out = list(grouped.values())
    out.sort(
        key=lambda row: (
            -int(row["graph_eligible_n128"]),
            -int(row["phase_feature_done"]),
            tuple(str(row[field]) for field in group_fields),
        )
    )
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], fields: list[str], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---:" if field in COUNT_FIELDS else "---" for field in fields) + " |",
    ]
    for row in selected:
        vals = []
        for field in fields:
            val = row[field]
            vals.append(f"{val:,}" if isinstance(val, int) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    detail_rows: list[dict[str, object]],
    by_perspective_team: list[dict[str, object]],
    by_map_perspective_team: list[dict[str, object]],
) -> None:
    totals = aggregate(detail_rows, [])
    total = totals[0] if totals else {field: 0 for field in COUNT_FIELDS}
    raw_total = int(total["raw_fetched"])
    graph_total = int(total["graph_eligible_n128"])
    eligible_rows = [row for row in by_map_perspective_team if int(row["graph_eligible_n128"]) > 0]
    raw_only_rows = [
        row
        for row in by_map_perspective_team
        if int(row["raw_fetched"]) > 0 and int(row["graph_eligible_n128"]) == 0
    ]

    lines = [
        "# Data Availability by Perspective, Map, and Team Size",
        "",
        "이 문서는 raw fetched match를 논문/재실험에 바로 사용할 수 있는 phase/graph eligible 데이터로 나누기 위해 생성한 availability 요약이다.",
        "",
        "## Overall Funnel",
        "",
        markdown_table([total], COUNT_FIELDS),
        "",
        "## Perspective x Team Size",
        "",
        markdown_table(
            by_perspective_team,
            ["perspective", "team_size", "raw_fetched", "phase_feature_done", "target_label_done", "edge_cache_done", "graph_eligible_n128"],
        ),
        "",
        "## Map x Perspective x Team Size: Graph-Eligible Domains",
        "",
        markdown_table(
            eligible_rows,
            ["map_name", "perspective", "team_size", "raw_fetched", "phase_feature_done", "target_label_done", "edge_cache_done", "graph_eligible_n128"],
            limit=80,
        ),
        "",
        "## Raw Domains Without Current Graph-Eligible Data",
        "",
        markdown_table(
            raw_only_rows,
            ["map_name", "perspective", "team_size", "game_mode", "raw_fetched", "phase_feature_done", "graph_eligible_n128", "not_phase_featured"],
            limit=80,
        ),
        "",
        "## Interpretation",
        "",
        "- `raw_fetched`는 telemetry를 가져온 원천 match 수이다.",
        "- `phase_feature_done`은 phase 단위 feature가 생성된 match 수이다.",
        "- `target_label_done`은 next-phase hazard target이 정의된 match 수이다.",
        "- `edge_cache_done`은 phase-level graph edge cache가 있는 match 수이다.",
        "- `graph_eligible_n128`은 현재 DynamicSTGNN 재실험에 바로 쓸 수 있는 match 수이다.",
        f"- 현재 논문용 전체 재실험의 최대 모집단은 raw {raw_total:,}건이 아니라 `graph_eligible_n128` 기준 {graph_total:,}건이다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    detail_rows = read_rows(args.input)
    detail_rows.sort(
        key=lambda row: (
            -int(row["graph_eligible_n128"]),
            -int(row["raw_fetched"]),
            str(row["map_name"]),
            str(row["perspective"]),
            str(row["team_size"]),
        )
    )

    by_perspective_team = aggregate(detail_rows, ["perspective", "team_size"])
    by_map_perspective_team = aggregate(detail_rows, ["map_name", "perspective", "team_size", "game_mode"])
    by_map = aggregate(detail_rows, ["map_name"])

    detail_path = args.out_dir / "data_availability_by_perspective_map_team_size.csv"
    perspective_team_path = args.out_dir / "data_availability_by_perspective_team_size.csv"
    map_path = args.out_dir / "data_availability_by_map.csv"

    write_csv(detail_path, detail_rows, ["perspective", "map_name", "team_size", "game_mode", *COUNT_FIELDS])
    write_csv(
        perspective_team_path,
        by_perspective_team,
        ["perspective", "team_size", *COUNT_FIELDS],
    )
    write_csv(map_path, by_map, ["map_name", *COUNT_FIELDS])
    write_report(
        args.doc,
        detail_rows=detail_rows,
        by_perspective_team=by_perspective_team,
        by_map_perspective_team=by_map_perspective_team,
    )

    print(f"wrote {detail_path}")
    print(f"wrote {perspective_team_path}")
    print(f"wrote {map_path}")
    print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
