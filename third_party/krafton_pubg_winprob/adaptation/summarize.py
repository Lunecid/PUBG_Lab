"""Summary table of the KRAFTON baseline runs (KRAFTON-tf, KRAFTON-mlp, KRAFTON-tf-ourfeat) next to the PTR reference arms.

    python -m adaptation.summarize [--seed 42] [--out reports/krafton_baseline_2026-10-05/summary_seed42.csv]

Reads ``<ckpt>/<cohort>/KRAFTON-tf|KRAFTON-mlp|KRAFTON-tf-ourfeat/seed<seed>/metrics.json`` and ``reports/
bigcomp2027_results/arm_metrics_per_cohort.csv`` (HGB-cut, M01, M11R, seed 42, stratum all).  Prints one row per
cohort; facts only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
COHORTS = ["erangel_tpp_squad", "taego_tpp_squad", "miramar_desert_tpp_squad", "rondo_tpp_squad", "sanhok_tpp_squad", "erangel_tpp_solo", "erangel_tpp_duo",
           "taego_tpp_duo", "taego_fpp_squad", "erangel_fpp_squad", "karakin_tpp_squad", "paramo_tpp_squad"]  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1")
    ap.add_argument("--ref", type=Path, default=REPO / "reports/bigcomp2027_results/arm_metrics_per_cohort.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    a = ap.parse_args(argv)
    ref = pd.read_csv(a.ref)
    ref = ref[(ref.seed == 42) & (ref.stratum == "all")]
    rows = []
    for c in COHORTS:
        row = {"cohort": c}
        for arm, key in (("KRAFTON-tf", "tf"), ("KRAFTON-mlp", "mlp"), ("KRAFTON-tf-ourfeat", "k2"), ("HGB-kraftonfeat", "hkf")):
            p = a.ckpt_dir / c / arm / f"seed{a.seed}" / "metrics.json"
            if not p.is_file():
                continue
            m = json.loads(p.read_text())
            t = m["test"]
            nat = (m.get("krafton_native") or {}).get("test", {}).get("all", {})
            row.update({
                f"{key}_auc": t["hazard"]["auc"], f"{key}_ap": t["hazard"]["ap"], f"{key}_nll": t["hazard"]["nll"], f"{key}_brier": t["hazard"]["brier"],
                f"{key}_spearman": t["rank"]["spearman"], f"{key}_top5_auc": t["rank"]["top5_auc"], f"{key}_cindex": t["rank"]["cindex"],
                f"{key}_win_acc": nat.get("winner_accuracy"), f"{key}_win_acc_chance": nat.get("winner_accuracy_chance"), f"{key}_native_cindex": nat.get("c_index"),
                f"{key}_epochs": m.get("epochs_run"), f"{key}_best_epoch": m.get("best_epoch"), f"{key}_stopped_early": m.get("stopped_early"),
                f"{key}_train_sec": m.get("train_sec", m.get("fit_sec")), f"{key}_total_sec": m.get("elapsed_sec"),
                f"{key}_platt_a": (m.get("baseline") or {}).get("platt", {}).get("a"),
            })  # fmt: skip
            row["n_test_rows"] = t["hazard"]["n"]
        for arm in ("HGB-cut", "M01", "M11R"):
            for metric in ("auc", "spearman"):
                r = ref[(ref.cohort == c) & (ref.arm == arm) & (ref.metric == metric)]
                if len(r):
                    row[f"{arm.replace('-', '_')}_{metric}"] = float(r.value.iloc[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    cols = ["cohort", "n_test_rows", "tf_auc", "mlp_auc", "k2_auc", "hkf_auc", "M01_auc", "HGB_cut_auc", "M11R_auc", "tf_epochs", "mlp_epochs", "k2_epochs", "tf_total_sec", "mlp_total_sec", "k2_total_sec", "hkf_total_sec"]
    print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
    if a.md:
        f = lambda v: "-" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
        print("| cohort | test rows | KRAFTON-tf | KRAFTON-mlp | KRAFTON-tf-ourfeat | HGB-kraftonfeat | M01 | HGB-cut | M11R |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in df.iterrows():
            g = lambda k: r.get(k, float("nan"))  # noqa: E731
            n = g("n_test_rows")
            print(f"| {r['cohort']} | {int(n) if not pd.isna(n) else '-'} | {f(g('tf_auc'))} | {f(g('mlp_auc'))} | {f(g('k2_auc'))} | {f(g('hkf_auc'))} | {f(g('M01_auc'))} | {f(g('HGB_cut_auc'))} | {f(g('M11R_auc'))} |")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False, float_format="%.9g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
