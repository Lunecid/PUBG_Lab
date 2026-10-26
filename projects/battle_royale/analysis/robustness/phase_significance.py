"""Per-phase match-level (cluster) bootstrap significance for hazard AUC.

Answers, for the cut-phase trajectory interpretation:
  (1) is each phase's hazard AUC significantly above chance, and how tight?
  (2) is the model's lift over baseline significant PER cut phase (paired)?

Match-level resampling preserves within-match correlation (bootstrap_ci.py C7).
Reads existing prediction CSVs (model,match_id,team_id,phase_idx,y,prob).
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
import csv
import sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
SEED = 42


def load_pair(path, model_a, model_b=None):
    """Return per-phase dict: phase -> (y, pa, pb, match_id) aligned arrays."""
    want = {model_a} | ({model_b} if model_b else set())
    # key (match,team,phase) -> {y, a, b}
    cells = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m not in want:
                continue
            k = (row["match_id"], row["team_id"], row["phase_idx"])
            d = cells[k]
            d["y"] = int(float(row["y"]))
            d["mid"] = row["match_id"]
            d["ph"] = int(row["phase_idx"])
            if m == model_a:
                d["a"] = float(row["prob"])
            else:
                d["b"] = float(row["prob"])
    per = defaultdict(lambda: {"y": [], "a": [], "b": [], "mid": []})
    for k, d in cells.items():
        if "a" not in d:
            continue
        if model_b and "b" not in d:
            continue
        p = per[d["ph"]]
        p["y"].append(d["y"])
        p["a"].append(d["a"])
        p["mid"].append(d["mid"])
        if model_b:
            p["b"].append(d["b"])
    return {ph: {k: np.asarray(v) for k, v in p.items()} for ph, p in per.items()}


def cluster_rows(groups):
    uniq = np.unique(groups)
    order = np.argsort(groups, kind="stable")
    sg = groups[order]
    s = np.searchsorted(sg, uniq, "left")
    e = np.searchsorted(sg, uniq, "right")
    return uniq, [order[a:b] for a, b in zip(s, e)]


def auc_ci(y, p, groups, n_boot=N_BOOT):
    uniq, rows = cluster_rows(groups)
    rng = np.random.default_rng(SEED)
    point = roc_auc_score(y, p) if y.min() != y.max() else float("nan")
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows[k] for k in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        stats.append(roc_auc_score(yy, p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (np.nan, np.nan)
    return point, lo, hi


def paired_delta_ci(y, pa, pb, groups, n_boot=N_BOOT):
    """Bootstrap CI of AUC(a) - AUC(b), paired on resampled matches."""
    uniq, rows = cluster_rows(groups)
    rng = np.random.default_rng(SEED)
    pt = (roc_auc_score(y, pa) - roc_auc_score(y, pb)) if y.min() != y.max() else float("nan")
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows[k] for k in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        deltas.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas.size else (np.nan, np.nan)
    p_le0 = float((deltas <= 0).mean()) if deltas.size else float("nan")  # ~one-sided boot p
    return pt, lo, hi, p_le0


def report_single(path, model, title):
    print(f"\n{'=' * 86}\n{title}\n  file={path}\n  model={model}\n{'=' * 86}")
    per = load_pair(path, model)
    print(
        f"{'phase':>5}{'n_cells':>9}{'n_match':>8}{'pos%':>7}{'AUC':>8}{'95% CI':>20}{'sig>0.5':>9}"
    )
    for ph in sorted(per):
        d = per[ph]
        pt, lo, hi = auc_ci(d["y"], d["a"], d["mid"])
        sig = "YES" if lo > 0.5 else "no"
        print(
            f"{ph:>5}{d['y'].size:>9}{len(np.unique(d['mid'])):>8}"
            f"{d['y'].mean() * 100:>6.1f}%{pt:>8.4f}  [{lo:.4f}, {hi:.4f}]{sig:>9}"
        )


def report_paired(path, model_a, model_b, title):
    print(f"\n{'=' * 86}\n{title}\n  file={path}\n  {model_a}  vs  {model_b}\n{'=' * 86}")
    per = load_pair(path, model_a, model_b)
    print(
        f"{'phase':>5}{'n_match':>8}{'AUC_a':>8}{'AUC_b':>8}{'dAUC':>9}{'95% CI of dAUC':>22}{'sig':>6}"
    )
    for ph in sorted(per):
        d = per[ph]
        a = roc_auc_score(d["y"], d["a"])
        b = roc_auc_score(d["y"], d["b"])
        pt, lo, hi, p0 = paired_delta_ci(d["y"], d["a"], d["b"], d["mid"])
        sig = "YES" if lo > 0 else ("(neg)" if hi < 0 else "ns")
        print(
            f"{ph:>5}{len(np.unique(d['mid'])):>8}{a:>8.4f}{b:>8.4f}{pt:>+9.4f}"
            f"  [{lo:+.4f}, {hi:+.4f}]{sig:>6}"
        )


if __name__ == "__main__":
    FA = "model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/feature_ablation_predictions.csv"
    PR = "model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv"
    # (1) full representation: per-phase hazard AUC CI (is each phase significant + tight?)
    report_single(
        FA,
        "a11_full_representation",
        "(1) FULL representation hazard AUC per cut phase (full DB, match-level CI)",
    )
    # (2) does representation beat official-raw baseline, per phase?
    report_paired(
        FA,
        "a11_full_representation",
        "a0_official_raw",
        "(2) FULL vs official-raw baseline, paired per cut phase",
    )
    # (3) graph contribution per phase: dynamic full graph vs tabular HGB, and vs no-graph
    report_paired(
        PR,
        "dynamic_full_tuned",
        "tabular_baseline_hgb",
        "(3) dynamic full GNN vs tabular HGB baseline, paired per cut phase",
    )
    report_paired(
        PR,
        "dynamic_full_e12",
        "dynamic_none_e12",
        "(4) graph structure contribution: full graph vs no-graph, paired per cut phase",
    )
