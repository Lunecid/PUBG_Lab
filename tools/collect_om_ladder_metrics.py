"""Collect per-cohort OM ladder matrix metrics into one run-level table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


METADATA_FIELDS = [
    "cohort",
    "cohort_slug",
    "map",
    "mode",
    "run_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path, metadata: dict[str, str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{**metadata, **row} for row in reader]


def collect(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(root.glob("*/matrix_metrics.csv")):
        run_dir = metrics_path.parent
        manifest = _read_json(run_dir / "matrix_manifest.json")
        metadata = {
            "cohort": str(manifest.get("cohort", "")),
            "cohort_slug": str(manifest.get("cohort_slug", run_dir.name)),
            "map": str(manifest.get("map", "")),
            "mode": str(manifest.get("mode", "")),
            "run_dir": str(run_dir),
        }
        rows.extend(_read_rows(metrics_path, metadata))

    cohort_counts = Counter(row["cohort"] for row in rows)
    complete_cohorts = sorted(cohort for cohort, count in cohort_counts.items() if count >= 7)
    summary = {
        "root": str(root),
        "rows": len(rows),
        "cohorts": len(cohort_counts),
        "complete_cohorts": complete_cohorts,
        "cohort_row_counts": dict(sorted(cohort_counts.items())),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(METADATA_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="model_stgcn/checkpoints/br_om_ladder_required_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-out", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "om_ladder_metrics_all_cohorts.csv"
    summary_out = Path(args.summary_out) if args.summary_out else root / "om_ladder_metrics_summary.json"
    rows, summary = collect(root)
    write_csv(out, rows)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary_out": str(summary_out), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
