"""Diagnostic (not part of the baseline protocol): how much of the pooled hazard-AUC gap is the missing phase base rate.

The Cox-family loss identifies the score only up to a constant per sample (match, time), so the raw score carries no
information about the phase's base hazard (0.21 ... 0.56 over cuts 1-7), while the PTR arms are trained with a hazard
BCE per phase.  For every cohort this script reports, on the test rows of seed ``--seed``:

* pooled AUC of the KRAFTON score (what the main table reports), the mean within-cut-phase AUC (row weighted);
* pooled AUC after adding a phase-specific intercept fitted on the VAL rows (logistic regression on the raw score +
  one-hot cut phase; val rows only);
* the same pooled / within-phase AUC for M11R and HGB-cut for reference.

    python -m adaptation.diagnostics [--seed 42] [--arm KRAFTON-tf] [--out reports/krafton_baseline_2026-10-05/phase_offset_diagnostic_tf.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from model_stgcn.ptr import metrics as M  # noqa: E402  (read-only use)

COHORTS = ["erangel_tpp_squad", "taego_tpp_squad", "miramar_desert_tpp_squad", "rondo_tpp_squad", "sanhok_tpp_squad", "erangel_tpp_solo", "erangel_tpp_duo",
           "taego_tpp_duo", "taego_fpp_squad", "erangel_fpp_squad", "karakin_tpp_squad", "paramo_tpp_squad"]  # fmt: skip


def within_phase_auc(df: pd.DataFrame, col: str) -> float:
    aucs, w = [], []
    for _k, g in df.groupby("phase"):
        if g["y"].nunique() == 2:
            aucs.append(roc_auc_score(g["y"], g[col]))
            w.append(len(g))
    return float(np.average(aucs, weights=w))


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df["phase"].astype(int), prefix="p").reindex(columns=[f"p_{k}" for k in range(1, 8)], fill_value=0).astype(float)
    X["g"] = df["raw_score"].to_numpy()
    return X


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", default="KRAFTON-tf")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = []
    for c in COHORTS:
        d = a.ckpt_dir / c / a.arm / f"seed{a.seed}"
        if not (d / "pred_test.csv").is_file():
            continue
        kv, kt = pd.read_csv(d / "pred_val.csv"), pd.read_csv(d / "pred_test.csv")
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(_design(kv), kv["y"].astype(int))
        row = {"cohort": c, "arm": a.arm, "n_test_rows": len(kt), "auc_pooled": roc_auc_score(kt["y"], kt["raw_score"]),
               "auc_within_phase": within_phase_auc(kt, "raw_score"),
               "auc_pooled_phase_offset": roc_auc_score(kt["y"], lr.predict_proba(_design(kt))[:, 1])}  # fmt: skip
        for ref in ("M11R", "HGB-cut"):
            p = a.ckpt_dir / c / ref / "seed42" / "pred_test.csv"
            if p.is_file():
                df = M.read_predictions(p)
                row[f"{ref}_auc_pooled"] = roc_auc_score(df["y"], df["p_hazard"])
                row[f"{ref}_auc_within_phase"] = within_phase_auc(df, "p_hazard")
        rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.round(4).to_string(index=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False, float_format="%.6g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
