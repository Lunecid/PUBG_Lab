"""Paired match-level bootstrap contrasts of the K3 test (pooled and within-phase AUC), with random-effects pooling.

    python -m adaptation.k3_contrasts [--n-boot 2000] [--out reports/krafton_baseline_2026-10-06/k3_contrasts.csv]

For every cohort the arms are scored on the SAME bootstrap resamples of whole test matches (``metrics.resample_counts``),
so shared match-level noise cancels in the differences:

* ``pooled``        AUC over all risk-set rows,
* ``within_phase``  unweighted mean over the cut phases of the per-phase AUC; the phases are those that hold both classes and
                    >= 30 rows in the full test set (fixed, so every resample uses the same phases).

Per cohort: point difference, bootstrap SE and 95% percentile interval.  Over the cohorts: DerSimonian-Laird random-effects
pooling of the per-cohort differences (``metrics.dersimonian_laird``): pooled estimate, 95% CI, tau^2, one-sided p (H0: <= 0).
Contrasts (A - B): K3a - K2 (objective), K3 - K3a (phase baseline), K3 - K2, M01 - K3a, M01 - K3.  Rows of all arms are
asserted identical (``metrics.align_predictions``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptation.phase_auc import COHORTS  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402  (read-only use)

ARMS = {"K2": "KRAFTON-tf-ourfeat", "K3a": "KRAFTON-tf-ourfeat-bce", "K3": "KRAFTON-tf-ourfeat-hazard", "M01": "M01"}
CONTRASTS = [("K3a", "K2"), ("K3", "K3a"), ("K3", "K2"), ("M01", "K3a"), ("M01", "K3")]
MIN_ROWS = 30


def cohort_draws(cohort: str, ckpt: Path, seed: int, n_boot: int, boot_seed: int) -> dict:
    frames = {k: M.read_predictions(ckpt / cohort / arm / f"seed{seed}" / "pred_test.csv") for k, arm in ARMS.items()}
    base = frames["K2"]
    for k, df in frames.items():
        M.align_predictions(base, df)  # identical rows and labels
    match_ids, codes = np.unique(base["match_id"].to_numpy(), return_inverse=True)
    counts = M.resample_counts(len(match_ids), n_boot, boot_seed)  # [B, n_matches]
    y = base["y"].to_numpy()
    phase = base["phase"].to_numpy()
    phases = [k for k in sorted(set(phase.tolist())) if (phase == k).sum() >= MIN_ROWS and len(np.unique(y[phase == k])) == 2]
    rows_k = {k: np.flatnonzero(phase == k) for k in phases}
    out = {}
    for name, df in frames.items():
        p = df["p_hazard"].to_numpy()
        pooled = M.SortedBinary(y, p)
        per = {k: M.SortedBinary(y[r], p[r]) for k, r in rows_k.items()}
        d_pool, d_within = np.empty(n_boot), np.empty(n_boot)
        full_w = np.ones(len(y))
        point = (pooled.auc(full_w), float(np.mean([per[k].auc(full_w[rows_k[k]]) for k in phases])))
        for b in range(n_boot):
            w = counts[b][codes].astype(np.float64)
            d_pool[b] = pooled.auc(w)
            d_within[b] = np.mean([per[k].auc(w[rows_k[k]]) for k in phases])
        out[name] = {"pooled": (point[0], d_pool), "within_phase": (point[1], d_within)}
    return {"arms": out, "n_matches": len(match_ids), "phases": phases}


def summarise(est: float, draws: np.ndarray) -> dict:
    d = draws[np.isfinite(draws)]
    return {"estimate": float(est), "se": float(d.std(ddof=1)), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "reports/krafton_baseline_2026-10-06/k3_contrasts.csv")
    a = ap.parse_args(argv)
    rows = []
    for c in COHORTS:
        if not all((a.ckpt_dir / c / arm / f"seed{a.seed}" / "pred_test.csv").is_file() for arm in ARMS.values()):
            print("skip (arms missing):", c)
            continue
        r = cohort_draws(c, a.ckpt_dir, a.seed, a.n_boot, a.boot_seed)
        for x, y in CONTRASTS:
            for metric in ("pooled", "within_phase"):
                est = r["arms"][x][metric][0] - r["arms"][y][metric][0]
                draws = r["arms"][x][metric][1] - r["arms"][y][metric][1]
                rows.append({"cohort": c, "contrast": f"{x} - {y}", "metric": metric, "level": "cohort", **summarise(est, draws)})
    cohort_df = pd.DataFrame(rows)
    pooled_rows = []
    for (contrast, metric), g in cohort_df.groupby(["contrast", "metric"], sort=False):
        re = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        pooled_rows.append({"cohort": f"RE pooled (k={re['k']})", "contrast": contrast, "metric": metric, "level": "pooled", "estimate": re["theta"], "se": re["se"],
                            "ci_low": re["ci95"][0], "ci_high": re["ci95"][1], "tau2": re["tau2"], "p_one_sided": re["p_one_sided"],
                            "mean_unweighted": float(g["estimate"].mean()), "n_cohorts_positive": int((g["estimate"] > 0).sum())})  # fmt: skip
    out = pd.concat([cohort_df, pd.DataFrame(pooled_rows)], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(pooled_rows)[["contrast", "metric", "estimate", "ci_low", "ci_high", "tau2", "mean_unweighted", "n_cohorts_positive", "p_one_sided"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
