"""Analyze O7 advantage-edge A/B/C early-readout predictions."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


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


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


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
    delta = _auc(y, p_left) - _auc(y, p_right)
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
        samples.append(_auc(y[idx], p_left[idx]) - _auc(y[idx], p_right[idx]))
    if not samples:
        return delta, float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(samples), [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def analyze(root: Path, *, n_boot: int, seed: int) -> dict:
    preds = {arm: _load(root / dirname / "pred_test.csv") for arm, dirname in CONDITIONS.items()}
    keys = sorted(set.intersection(*(set(v) for v in preds.values())))
    if not keys:
        raise SystemExit(f"no aligned prediction rows found under {root}")

    rows: list[dict] = []
    for group_name, phases in PHASE_GROUPS.items():
        group_keys = _group_keys(keys, phases)
        y = np.asarray([preds["A"][key][0] for key in group_keys], dtype=np.int64)
        match_ids = np.asarray([key[0] for key in group_keys])
        probs = {
            arm: np.asarray([preds[arm][key][1] for key in group_keys], dtype=np.float64)
            for arm in CONDITIONS
        }
        aucs = {arm: _auc(y, p) for arm, p in probs.items()}
        bc, bc_lo, bc_hi = _bootstrap_delta(
            y, probs["B"], probs["C"], match_ids, n_boot=n_boot, seed=seed
        )
        ba, ba_lo, ba_hi = _bootstrap_delta(
            y, probs["B"], probs["A"], match_ids, n_boot=n_boot, seed=seed + 1
        )
        rows.append(
            {
                "phase_group": group_name,
                "phases": "all" if phases is None else ",".join(str(p) for p in phases),
                "n": int(y.size),
                "n_matches": int(len(set(match_ids.tolist()))),
                "auc_A_none_observed": aucs["A"],
                "auc_B_knn_observed": aucs["B"],
                "auc_C_knn_edge_shuffle": aucs["C"],
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
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(root, n_boot=args.bootstrap, seed=args.seed)

    json_path = out_dir / "o7_early_readout_summary.json"
    csv_path = out_dir / "o7_early_readout_phase_group_deltas.csv"
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
