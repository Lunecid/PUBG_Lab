"""Run facts of K2 / K3a / K3 (epochs, early stopping, Platt scaling, intercept movement, run time) from the metrics.json files.

    python -m adaptation.k3_facts [--out reports/krafton_baseline_2026-10-06/k3_run_facts.csv]

Per cohort and arm: epochs run / best epoch / early stopping, steps per epoch, run seconds; for K3a / K3 also whether the samples
equal K2's (``same_samples_as_k2``), the Platt (a, b) fitted on val, the raw-logit and Platt test NLL, and how far the learned
intercepts moved from their initial values (head bias; the 7 phase offsets of K3).  Prints the summaries.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptation.phase_auc import COHORTS  # noqa: E402

ARMS = (("K2", "KRAFTON-tf-ourfeat"), ("K3a", "KRAFTON-tf-ourfeat-bce"), ("K3", "KRAFTON-tf-ourfeat-hazard"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "model_stgcn/checkpoints/bigcomp2027_ptr_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "reports/krafton_baseline_2026-10-06/k3_run_facts.csv")
    a = ap.parse_args(argv)
    rows = []
    for c in COHORTS:
        for key, arm in ARMS:
            p = a.ckpt_dir / c / arm / f"seed{a.seed}" / "metrics.json"
            if not p.is_file():
                continue
            m = json.loads(p.read_text())
            r = {"cohort": c, "arm": key, "epochs": m["epochs_run"], "best_epoch": m["best_epoch"], "stopped_early": m["stopped_early"], "epoch_cap": m["max_epochs_effective"],
                 "steps_per_epoch": m["steps_per_epoch"], "train_sec": m["train_sec"], "total_sec": m["elapsed_sec"],
                 "platt_a": m["baseline"]["platt"]["a"], "platt_b": m["baseline"]["platt"]["b"]}  # fmt: skip
            if key != "K2":
                r["same_samples_as_k2"] = all(m["same_samples_as_k2"].values()) if m.get("same_samples_as_k2") else None
                r["nll_raw_test"], r["nll_platt_test"] = m["uncalibrated"]["test"]["nll"], m["test"]["hazard"]["nll"]
                r["nll_raw_val"], r["nll_platt_val"] = m["uncalibrated"]["val"]["nll"], m["val"]["hazard"]["nll"]
                r["best_val_nll"] = m["best_val_nll"]
                r["head_bias_move"] = m["baseline"]["final_head_bias"] - m["baseline"]["intercept_init"]["head_bias"]
                if key == "K3":
                    init, fin = np.array(m["baseline"]["intercept_init"]["phase_offsets"]), np.array(m["baseline"]["final_phase_offsets"])
                    r["phase_offset_move_max"] = float(np.abs(fin - init).max())
                    r["phase_offset_init_range"] = float(init.max() - init.min())
            rows.append(r)
    d = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(d.groupby("arm").agg(runs=("cohort", "size"), epochs_min=("epochs", "min"), epochs_max=("epochs", "max"), best_min=("best_epoch", "min"), best_max=("best_epoch", "max"),
                               all_early_stopped=("stopped_early", "all"), run_minutes=("total_sec", lambda s: round(s.sum() / 60, 1))).to_string())
    b = d[d.arm != "K2"]
    print("same samples as K2 in every run:", bool(b["same_samples_as_k2"].dropna().all()), f"({int(b['same_samples_as_k2'].notna().sum())} runs checked)")
    for k in ("K3a", "K3"):
        x = d[d.arm == k]
        if len(x):
            print(f"{k}: Platt a {x.platt_a.mean():.3f} ({x.platt_a.min():.3f}-{x.platt_a.max():.3f}), b {x.platt_b.mean():+.3f} ({x.platt_b.min():+.3f} to {x.platt_b.max():+.3f}); "
                  f"test NLL raw logit {x.nll_raw_test.mean():.4f} -> Platt {x.nll_platt_test.mean():.4f}; head bias moved {x.head_bias_move.abs().mean():.3f} on average")
    x = d[d.arm == "K3"]
    if len(x):
        print(f"K3 phase offsets: largest move from the initial value {x.phase_offset_move_max.max():.3f} (mean over cohorts {x.phase_offset_move_max.mean():.3f}); "
              f"initial spread max-min {x.phase_offset_init_range.mean():.2f} on average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
