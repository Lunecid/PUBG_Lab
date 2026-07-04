"""Collect phase-stratified metrics for the required-cohort OM ladder.

The source eval JSONs already contain `phase_metrics`, computed inside each
map / perspective / team-size cohort.  This collector flattens those metrics
and computes sequential OM deltas without mixing cohorts during training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


PRESET_ORDER = [
    "om0_context",
    "om1_survival_capital",
    "om2_movement",
    "om3_pressure_habitat",
    "om4_threat_exposure",
    "om5_relation_structure",
    "om6_spatial_proxy",
]

STEP_PAIRS = list(zip(PRESET_ORDER[:-1], PRESET_ORDER[1:]))

METRIC_PATHS = {
    "hazard_auc": ("hazard", "auc"),
    "hazard_brier": ("hazard", "brier"),
    "hazard_ece": ("hazard", "ece"),
    "top5_auc": ("top5", "auc"),
    "top10_auc": ("top10", "auc"),
    "rank_spearman": ("rank", "spearman"),
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _get_nested(data: dict[str, Any], path: tuple[str, str]) -> float:
    node: Any = data
    for key in path:
        if not isinstance(node, dict):
            return float("nan")
        node = node.get(key)
    return _float(node)


def phase_group(cut_phase: int) -> str:
    if cut_phase <= 2:
        return "early_1_2"
    if cut_phase <= 4:
        return "mid_3_4"
    return "late_5_plus"


def flatten_phase_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eval_json = Path(row["eval_json"])
            data = _read_json(eval_json)
            phase_metrics = data.get("phase_metrics", {})
            for phase_key, phase_data in sorted(phase_metrics.items(), key=lambda kv: int(kv[0])):
                cut_phase = int(phase_data.get("cut_phase", phase_key))
                out: dict[str, Any] = {
                    "cohort": row["cohort"],
                    "cohort_slug": row["cohort_slug"],
                    "map": row["map"],
                    "mode": row["mode"],
                    "feature_preset": row["feature_preset"],
                    "cut_phase": cut_phase,
                    "phase_group": phase_group(cut_phase),
                    "n_matches": int(phase_data.get("n_matches", 0)),
                    "hazard_n": int(phase_data.get("hazard", {}).get("n", 0)),
                    "hazard_pos_rate": _get_nested(phase_data, ("hazard", "pos_rate")),
                    "top5_n": int(phase_data.get("top5", {}).get("n", 0)),
                    "top10_n": int(phase_data.get("top10", {}).get("n", 0)),
                    "rank_n": int(phase_data.get("rank", {}).get("n", 0)),
                    "eval_json": row["eval_json"],
                }
                for metric, path in METRIC_PATHS.items():
                    out[metric] = _get_nested(phase_data, path)
                rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def delta_rows(phase_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in phase_rows:
        keyed[(str(row["cohort"]), int(row["cut_phase"]), str(row["feature_preset"]))] = row

    out: list[dict[str, Any]] = []
    for (cohort, cut_phase, _preset), row in sorted(keyed.items()):
        if _preset != PRESET_ORDER[0]:
            continue
        for before, after in STEP_PAIRS:
            left = keyed.get((cohort, cut_phase, before))
            right = keyed.get((cohort, cut_phase, after))
            if not left or not right:
                continue
            delta: dict[str, Any] = {
                "cohort": cohort,
                "cohort_slug": row["cohort_slug"],
                "map": row["map"],
                "mode": row["mode"],
                "cut_phase": cut_phase,
                "phase_group": row["phase_group"],
                "step": f"{before}->{after}",
                "before": before,
                "after": after,
                "n_matches": right["n_matches"],
                "hazard_n": right["hazard_n"],
                "hazard_pos_rate": right["hazard_pos_rate"],
            }
            for metric in METRIC_PATHS:
                lval = _float(left.get(metric))
                rval = _float(right.get(metric))
                delta[f"delta_{metric}"] = rval - lval if _is_finite(lval) and _is_finite(rval) else float("nan")
            out.append(delta)
    return out


def _summarize_values(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if _is_finite(v)]
    if not finite:
        return {
            "n": 0,
            "positive": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": len(finite),
        "positive": sum(1 for v in finite if v > 0),
        "median": median(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def summarize_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metric": "delta_hazard_auc",
        "by_step_phase": {},
        "by_step_phase_group": {},
        "om2_to_om3_by_phase": {},
        "om2_to_om3_by_phase_group": {},
    }

    by_step_phase: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_step_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get("delta_hazard_auc"))
        if not _is_finite(value):
            continue
        by_step_phase[(str(row["step"]), int(row["cut_phase"]))].append(value)
        by_step_group[(str(row["step"]), str(row["phase_group"]))].append(value)

    for (step, cut_phase), values in sorted(by_step_phase.items()):
        summary["by_step_phase"].setdefault(step, {})[str(cut_phase)] = _summarize_values(values)
    for (step, group), values in sorted(by_step_group.items()):
        summary["by_step_phase_group"].setdefault(step, {})[group] = _summarize_values(values)

    target_step = "om2_movement->om3_pressure_habitat"
    summary["om2_to_om3_by_phase"] = summary["by_step_phase"].get(target_step, {})
    summary["om2_to_om3_by_phase_group"] = summary["by_step_phase_group"].get(target_step, {})
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="model_stgcn/checkpoints/br_om_ladder_required_v1")
    ap.add_argument("--metrics-csv", default=None)
    ap.add_argument("--phase-out", default=None)
    ap.add_argument("--delta-out", default=None)
    ap.add_argument("--summary-out", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else root / "om_ladder_metrics_all_cohorts.csv"
    phase_out = Path(args.phase_out) if args.phase_out else root / "phase_stratified_metrics.csv"
    delta_out = Path(args.delta_out) if args.delta_out else root / "phase_stratified_deltas.csv"
    summary_out = Path(args.summary_out) if args.summary_out else root / "phase_stratified_delta_summary.json"

    phase_rows = flatten_phase_rows(metrics_csv)
    deltas = delta_rows(phase_rows)
    summary = summarize_deltas(deltas)
    summary.update(
        {
            "root": str(root),
            "metrics_csv": str(metrics_csv),
            "phase_rows": len(phase_rows),
            "delta_rows": len(deltas),
            "cohorts": len({row["cohort"] for row in phase_rows}),
            "phase_out": str(phase_out),
            "delta_out": str(delta_out),
        }
    )

    write_csv(phase_out, phase_rows)
    write_csv(delta_out, deltas)
    summary_out.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"phase_out": str(phase_out), "delta_out": str(delta_out), "summary_out": str(summary_out), "phase_rows": len(phase_rows), "delta_rows": len(deltas)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

