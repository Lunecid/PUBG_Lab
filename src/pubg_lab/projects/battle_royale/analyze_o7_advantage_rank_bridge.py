"""Hazard-implied rank bridge for O7 advantage-edge A/B/C predictions.

The O7 advantage-edge model is hazard-only.  This bridge evaluates whether its
phase-cut hazard probabilities also order teams by final-rank labels.  It is
therefore a hazard-implied rank Spearman, not a separate rank-head metric.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from pubg_lab.survival.provenance import git_provenance
from pubg_lab.survival.v3.factorized_data import fetch_factorized_phase_match


CONDITIONS = {
    "A": "A_none_observed",
    "B": "B_knn_observed",
    "C": "C_knn_edge_shuffle",
}

PHASE_GROUPS = {
    "all": None,
    "early_1_2": (1, 2),
    "mid_3_4": (3, 4),
    "late_5_6": (5, 6),
    "final_7": (7,),
}


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2 or np.unique(y).size < 2 or np.unique(p).size < 2:
        return float("nan")
    yr = rankdata(y.astype(np.float64), method="average")
    pr = rankdata(p.astype(np.float64), method="average")
    yc = yr - yr.mean()
    pc = pr - pr.mean()
    denom = np.sqrt(np.sum(yc * yc) * np.sum(pc * pc))
    if denom == 0:
        return float("nan")
    return float(np.sum(yc * pc) / denom)


def _load(path: Path) -> dict[tuple[str, int, int], tuple[int, float]]:
    out: dict[tuple[str, int, int], tuple[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["match_id"], int(row["cut_phase"]), int(row["team_idx"]))
            out[key] = (int(row["hazard_true"]), float(row["hazard_prob"]))
    return out


def _group_keys(
    keys: Iterable[tuple[str, int, int]], phases: tuple[int, ...] | None
) -> list[tuple[str, int, int]]:
    if phases is None:
        return list(keys)
    phase_set = set(phases)
    return [key for key in keys if key[1] in phase_set]


def _cluster_indices(match_ids: np.ndarray) -> list[np.ndarray]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for idx, match_id in enumerate(match_ids):
        clusters[str(match_id)].append(idx)
    return [np.asarray(v, dtype=np.int64) for v in clusters.values()]


def _bootstrap_delta(
    y: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    match_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    delta = _spearman(y, p_left) - _spearman(y, p_right)
    clusters = _cluster_indices(match_ids)
    if not clusters or n_boot <= 0:
        return delta, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in picked])
        if np.unique(y[idx]).size < 2:
            continue
        val = _spearman(y[idx], p_left[idx]) - _spearman(y[idx], p_right[idx])
        if not np.isnan(val):
            samples.append(val)
    if not samples:
        return delta, float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(samples), [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def _rank_labels_for_matches(
    match_ids: Iterable[str], feature_preset: str
) -> dict[tuple[str, int], float]:
    labels: dict[tuple[str, int], float] = {}
    for match_id in sorted(set(match_ids)):
        m = fetch_factorized_phase_match(match_id, feature_preset=feature_preset)
        if m is None:
            continue
        for team_idx, value in enumerate(m.y_rank):
            labels[(match_id, team_idx)] = float(value)
    return labels


def analyze(root: Path, *, n_boot: int, seed: int, feature_preset: str) -> dict:
    preds = {arm: _load(root / dirname / "pred_test.csv") for arm, dirname in CONDITIONS.items()}
    common = sorted(set.intersection(*(set(v) for v in preds.values())))
    if not common:
        raise SystemExit(f"no aligned prediction rows found under {root}")
    labels = _rank_labels_for_matches((key[0] for key in common), feature_preset)
    keys = [key for key in common if (key[0], key[2]) in labels]
    if not keys:
        raise SystemExit(f"no rank labels found for aligned predictions under {root}")

    rows: list[dict] = []
    for group_name, phases in PHASE_GROUPS.items():
        group_keys = _group_keys(keys, phases)
        y_rank = np.asarray([labels[(key[0], key[2])] for key in group_keys], dtype=np.float64)
        match_ids = np.asarray([key[0] for key in group_keys])
        probs = {
            arm: np.asarray([preds[arm][key][1] for key in group_keys], dtype=np.float64)
            for arm in CONDITIONS
        }
        spearman = {arm: _spearman(y_rank, p) for arm, p in probs.items()}
        bc, bc_lo, bc_hi = _bootstrap_delta(
            y_rank, probs["B"], probs["C"], match_ids, n_boot=n_boot, seed=seed
        )
        ba, ba_lo, ba_hi = _bootstrap_delta(
            y_rank, probs["B"], probs["A"], match_ids, n_boot=n_boot, seed=seed + 1
        )
        rows.append(
            {
                "phase_group": group_name,
                "phases": "all" if phases is None else ",".join(str(p) for p in phases),
                "n": int(y_rank.size),
                "n_matches": int(len(set(match_ids.tolist()))),
                "rank_spearman_A_none_observed": spearman["A"],
                "rank_spearman_B_knn_observed": spearman["B"],
                "rank_spearman_C_knn_edge_shuffle": spearman["C"],
                "delta_B_minus_C": bc,
                "delta_B_minus_C_ci_low": bc_lo,
                "delta_B_minus_C_ci_high": bc_hi,
                "delta_B_minus_A": ba,
                "delta_B_minus_A_ci_low": ba_lo,
                "delta_B_minus_A_ci_high": ba_hi,
            }
        )
    return {
        "root": str(root),
        "conditions": CONDITIONS,
        "feature_preset": feature_preset,
        "metric": "hazard_implied_final_placement_spearman",
        "metric_note": (
            "Spearman(final_rank_norm, hazard_prob). final_rank_norm increases "
            "with worse placement, so positive values mean higher hazard orders "
            "teams toward worse final rank. O7 has no rank head."
        ),
        "metric_is_headline_safe": False,
        "provenance": git_provenance(),
        "n_aligned_rows": len(keys),
        "n_boot": n_boot,
        "seed": seed,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-preset", default="om5_relation_structure")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(
        root, n_boot=args.bootstrap, seed=args.seed, feature_preset=args.feature_preset
    )

    json_path = out_dir / "o7_rank_bridge_summary.json"
    csv_path = out_dir / "o7_rank_bridge_phase_group_deltas.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(result["rows"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for row in result["rows"]:
        print(
            row["phase_group"],
            f"B-C={row['delta_B_minus_C']:+.4f}",
            f"[{row['delta_B_minus_C_ci_low']:+.4f},{row['delta_B_minus_C_ci_high']:+.4f}]",
            f"B-A={row['delta_B_minus_A']:+.4f}",
            f"[{row['delta_B_minus_A_ci_low']:+.4f},{row['delta_B_minus_A_ci_high']:+.4f}]",
        )


if __name__ == "__main__":
    main()
