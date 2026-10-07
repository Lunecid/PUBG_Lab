"""Pooled and within-phase test AUC of the K-arms and reference arms (the K3 table of the paper's headline test).

    python -m adaptation.phase_auc [--arms KRAFTON-tf-ourfeat,KRAFTON-tf-ourfeat-bce,KRAFTON-tf-ourfeat-hazard,M01] [--out-dir reports/krafton_baseline_v2]

For every cohort and arm (seed 42, test split, ``p_hazard`` of ``pred_test.csv``):

* ``pooled``        hazard AUC over all risk-set rows (the paper's metric; equals ``test.hazard.auc`` of metrics.json),
* ``within_phase``  UNWEIGHTED mean over the cut phases k of the per-phase AUC, using the phases that hold both classes and
                    at least ``--min-rows`` (30) rows; ``n_phases`` = how many phases that is,
* ``phase_cal_gap`` row-weighted mean over the cut phases of |mean predicted p - observed hazard rate| (calibration in the large
                    per phase, on ``p_hazard``; extra diagnostic, not an AUC).

The set of qualifying phases depends only on the rows, so it is identical for every arm of a cohort.  Writes
``k3_phase_auc.csv`` (one row per cohort and arm), ``k3_phase_auc_by_phase.csv`` (one row per cohort, arm and phase) and prints
the markdown tables with the 12-cohort means and the paired differences between arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sklearn.metrics import roc_auc_score  # noqa: E402

from model_stgcn.ptr import metrics as M  # noqa: E402  (read-only use)

COHORTS = ["erangel_tpp_squad", "taego_tpp_squad", "miramar_desert_tpp_squad", "rondo_tpp_squad", "sanhok_tpp_squad", "erangel_tpp_solo", "erangel_tpp_duo",
           "taego_tpp_duo", "taego_fpp_squad", "erangel_fpp_squad", "karakin_tpp_squad", "paramo_tpp_squad"]  # fmt: skip
SHORT = {"KRAFTON-tf-ourfeat": "K2", "KRAFTON-tf-ourfeat-bce": "K3a", "KRAFTON-tf-ourfeat-hazard": "K3", "KRAFTON-tf": "K1", "KRAFTON-mlp": "K1-mlp",
         "HGB-kraftonfeat": "HGB-kf", "HGB-cut": "HGB-cut"}  # fmt: skip
DEFAULT_ARMS = "KRAFTON-tf-ourfeat,KRAFTON-tf-ourfeat-bce,KRAFTON-tf-ourfeat-hazard,M01"


def per_phase_auc(df: pd.DataFrame, col: str = "p_hazard", min_rows: int = 30) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby("phase"):
        both = g["y"].nunique() == 2
        rows.append({"phase": int(k), "n_rows": len(g), "n_pos": int(g["y"].sum()), "qualifies": bool(both and len(g) >= min_rows),
                     "auc": float(roc_auc_score(g["y"], g[col])) if both else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v2")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-rows", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports/krafton_baseline_v2")
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    rows, byphase = [], []
    for c in COHORTS:
        for arm in arms:
            d = a.ckpt_dir / c / arm / f"seed{a.seed}"
            if not (d / "pred_test.csv").is_file() or not (d / "metrics.json").is_file():
                continue
            df = M.read_predictions(d / "pred_test.csv")
            pooled = float(roc_auc_score(df["y"], df["p_hazard"]))
            ref = json.loads((d / "metrics.json").read_text())["test"]["hazard"]["auc"]
            if abs(pooled - ref) > 1e-5:
                raise AssertionError(f"{c}/{arm}: pooled AUC {pooled:.7f} from the CSV differs from metrics.json {ref:.7f}")
            ph = per_phase_auc(df, "p_hazard", a.min_rows)
            used = ph[ph["qualifies"]]
            gap = sum(abs(g["p_hazard"].mean() - g["y"].mean()) * len(g) for _k, g in df.groupby("phase")) / len(df)
            rows.append({"cohort": c, "arm": arm, "n_rows": len(df), "pooled": pooled, "within_phase": float(used["auc"].mean()), "n_phases": int(len(used)),
                         "phase_cal_gap": float(gap)})  # fmt: skip
            ph.insert(0, "arm", arm)
            ph.insert(0, "cohort", c)
            byphase.append(ph)
    res = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out_dir / "k3_phase_auc.csv", index=False, float_format="%.9g")
    pd.concat(byphase).to_csv(a.out_dir / "k3_phase_auc_by_phase.csv", index=False, float_format="%.9g")

    wide = res.pivot(index="cohort", columns="arm", values=["pooled", "within_phase"]).reindex(COHORTS)
    name = lambda arm: SHORT.get(arm, arm)  # noqa: E731
    for metric, title in (("pooled", "pooled test AUC"), ("within_phase", f"within-phase test AUC (mean of per-phase AUCs, phases with both classes and >= {a.min_rows} rows)")):
        print(f"\n{title}\n")
        print("| cohort | " + " | ".join(name(x) for x in arms) + " |")
        print("| --- | " + " | ".join("---:" for _ in arms) + " |")
        for c in COHORTS:
            if c in wide.index and wide.loc[c, metric].notna().any():
                print(f"| {c} | " + " | ".join(f"{wide.loc[c, (metric, x)]:.3f}" if (metric, x) in wide.columns and pd.notna(wide.loc[c, (metric, x)]) else "-" for x in arms) + " |")
        full = wide[metric].dropna()
        print("| **mean (%d cohorts)** | " % len(full) + " | ".join(f"{full[x].mean():.3f}" if x in full.columns else "-" for x in arms) + " |")
    print("\nmean over the cohorts of the per-phase calibration-in-the-large gap |mean p - mean y| (row weighted over phases)\n")
    cal = res.pivot(index="cohort", columns="arm", values="phase_cal_gap").reindex(COHORTS).dropna()
    print("| " + " | ".join(name(x) for x in arms) + " |")
    print("| " + " | ".join("---:" for _ in arms) + " |")
    print("| " + " | ".join(f"{cal[x].mean():.4f}" if x in cal.columns else "-" for x in arms) + " |")
    print("\npaired differences over the cohorts (mean, min, max, cohorts > 0)\n")
    pairs = [("KRAFTON-tf-ourfeat-bce", "KRAFTON-tf-ourfeat"), ("KRAFTON-tf-ourfeat-hazard", "KRAFTON-tf-ourfeat-bce"), ("KRAFTON-tf-ourfeat-hazard", "KRAFTON-tf-ourfeat"),
             ("M01", "KRAFTON-tf-ourfeat-bce"), ("M01", "KRAFTON-tf-ourfeat-hazard")]  # fmt: skip
    print("| difference | pooled mean | min | max | n > 0 | within-phase mean | min | max | n > 0 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for x, y in pairs:
        if x not in arms or y not in arms:
            continue
        cells = []
        for metric in ("pooled", "within_phase"):
            w = wide[metric][[x, y]].dropna()
            d = w[x] - w[y]
            cells += [f"{d.mean():+.3f}", f"{d.min():+.3f}", f"{d.max():+.3f}", f"{int((d > 0).sum())}/{len(d)}"]
        print(f"| {name(x)} - {name(y)} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
