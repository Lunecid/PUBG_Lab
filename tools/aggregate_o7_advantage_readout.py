"""Aggregate O7 advantage-edge A/B/C phase-group readouts across cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


DEFAULT_COHORTS = [
    "erangel_tpp_squad",
    "taego_tpp_squad",
    "erangel_tpp_solo",
    "miramar_desert_tpp_squad",
    "rondo_tpp_squad",
    "sanhok_tpp_squad",
    "vikendi_tpp_squad",
    "paramo_tpp_squad",
    "karakin_tpp_squad",
    "erangel_tpp_duo",
    "taego_tpp_duo",
    "taego_fpp_squad",
    "erangel_fpp_squad",
]


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _int(row: dict, key: str) -> int:
    return int(float(row[key]))


def _weighted_mean(rows: Iterable[dict], key: str) -> float:
    rows = list(rows)
    total = sum(_int(row, "n") for row in rows)
    if total == 0:
        return float("nan")
    return sum(_float(row, key) * _int(row, "n") for row in rows) / total


def _summarize_delta(rows: list[dict], delta_key: str, lo_key: str, hi_key: str) -> dict:
    vals = [_float(row, delta_key) for row in rows]
    return {
        "positive": sum(v > 0 for v in vals),
        "negative": sum(v < 0 for v in vals),
        "zero": sum(v == 0 for v in vals),
        "ci_positive": sum(_float(row, lo_key) > 0 for row in rows),
        "ci_negative": sum(_float(row, hi_key) < 0 for row in rows),
        "median": median(vals),
        "mean": sum(vals) / len(vals),
        "weighted_mean_by_n": _weighted_mean(rows, delta_key),
        "min": min(vals),
        "max": max(vals),
    }


def _load_cohort(root: Path, slug: str, input_name: str) -> list[dict]:
    path = root / slug / input_name
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["cohort_slug"] = slug
            rows.append(row)
    return rows


def aggregate(root: Path, cohorts: list[str], input_name: str) -> tuple[list[dict], list[dict]]:
    detail_rows: list[dict] = []
    for slug in cohorts:
        detail_rows.extend(_load_cohort(root, slug, input_name))

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in detail_rows:
        by_group[row["phase_group"]].append(row)

    summary_rows: list[dict] = []
    for group_name in sorted(by_group.keys(), key=lambda k: ["all", "early_1_2", "mid_3_4", "late_5_6", "final_7"].index(k)):
        rows = by_group[group_name]
        if len(rows) != len(cohorts):
            raise ValueError(f"{group_name} has {len(rows)} rows; expected {len(cohorts)}")
        bc = _summarize_delta(
            rows,
            "delta_B_minus_C",
            "delta_B_minus_C_ci_low",
            "delta_B_minus_C_ci_high",
        )
        ba = _summarize_delta(
            rows,
            "delta_B_minus_A",
            "delta_B_minus_A_ci_low",
            "delta_B_minus_A_ci_high",
        )
        summary_rows.append(
            {
                "phase_group": group_name,
                "phases": rows[0]["phases"],
                "n_cohorts": len(rows),
                "total_n": sum(_int(row, "n") for row in rows),
                "total_n_matches": sum(_int(row, "n_matches") for row in rows),
                "B_minus_C": bc,
                "B_minus_A": ba,
            }
        )
    return detail_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="model_stgcn/checkpoints/br_o7_advantage_edge_required_v1")
    parser.add_argument("--cohorts", default="|".join(DEFAULT_COHORTS))
    parser.add_argument("--input-name", default="o7_early_readout_phase_group_deltas.csv")
    parser.add_argument("--out-prefix", default="o7_required_13cohort")
    args = parser.parse_args()

    root = Path(args.root)
    cohorts = [part.strip() for part in args.cohorts.split("|") if part.strip()]
    detail_rows, summary_rows = aggregate(root, cohorts, args.input_name)

    detail_path = root / f"{args.out_prefix}_phase_group_deltas.csv"
    summary_path = root / f"{args.out_prefix}_summary.json"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(detail_rows[0].keys())
        if "cohort_slug" in fields:
            fields = ["cohort_slug"] + [field for field in fields if field != "cohort_slug"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in detail_rows)
    summary_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "input_name": args.input_name,
                "cohorts": cohorts,
                "n_cohorts": len(cohorts),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")
    for row in summary_rows:
        bc = row["B_minus_C"]
        ba = row["B_minus_A"]
        print(
            row["phase_group"],
            f"B-C +={bc['positive']}/{row['n_cohorts']}",
            f"ci+={bc['ci_positive']}/{row['n_cohorts']}",
            f"median={bc['median']:+.4f}",
            f"weighted={bc['weighted_mean_by_n']:+.4f}",
            f"| B-A +={ba['positive']}/{row['n_cohorts']}",
            f"ci+={ba['ci_positive']}/{row['n_cohorts']}",
            f"median={ba['median']:+.4f}",
        )


if __name__ == "__main__":
    main()
