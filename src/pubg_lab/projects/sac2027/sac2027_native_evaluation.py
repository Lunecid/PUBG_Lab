#!/usr/bin/env python3
"""Original-task discrimination of the new patch predictions, on matched rows."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import importlib.util
import json
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from pubg_lab.projects.sac2027.sac2027_fair_tune import BASE, DUMP, M, save
from pubg_lab.survival.ptr import data as D

src = REPO / "reports/bigcomp2027_results_v2/fidelity/krafton_fidelity.py"
spec = importlib.util.spec_from_file_location("native_helpers", src)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
ARMS = (
    "HGB-cut",
    "M11R",
    "KRAFTON-tf",
    "KRAFTON-tf-h0",
    "KRAFTON-tf-ourfeat",
    "KRAFTON-tf-ourfeat-bce",
    "KRAFTON-tf-ourfeat-hazard",
    "HGB-union",
)


def main():
    from pubg_lab.projects.sac2027.sac2027_followup_run import wait_for

    wait_for([BASE / "main_complete.json"])
    out = BASE / "native_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = counts.loc[counts.analysis_role == "main", "cohort"].tolist()
    rows = []
    draws = {a: {} for a in ARMS}
    checks = []
    for c in cohorts:
        mats = {str(m["match_id"]): m for m in D.load_dump(DUMP / c / "test.pt")["matches"]}
        reference = None
        for arm in ARMS:
            root = BASE / "checkpoints/ptr_v2" / c / arm / "seed42"
            frame = M.read_predictions(root / "pred_test.csv")
            if reference is None:
                reference = frame
            else:
                assert frame[M.KEY_COLUMNS + ["y"]].equals(reference[M.KEY_COLUMNS + ["y"]])
            P, Y, V, ph, mids = F.padded(frame, frame.rank_score.to_numpy(), mats)
            st = F.per_sample_stats(P, Y, V)
            point = dict(
                winner_accuracy=float(st["acc"].mean()),
                c_index=float(st["conc"].sum() / st["comp"].sum()),
                random_winner_accuracy=float(np.mean(1 / st["n"])),
            )
            u, inv = np.unique(mids.astype(str), return_inverse=True)
            sample_n = np.bincount(inv)
            acc = np.bincount(inv, weights=st["acc"])
            conc = np.bincount(inv, weights=st["conc"])
            comp = np.bincount(inv, weights=st["comp"])
            w = M.resample_counts(len(u), 2000, 20261008 + cohorts.index(c))
            draws[arm][c] = dict(
                winner_accuracy=(w @ acc) / (w @ sample_n), c_index=(w @ conc) / (w @ comp)
            )
            rows.append(dict(cohort=c, arm=arm, **point))
            if arm == "KRAFTON-tf":
                stored = json.loads((root / "metrics.json").read_text())["krafton_native"]["test"][
                    "all"
                ]
                delta = {k: abs(point[k] - stored[k]) for k in ("winner_accuracy", "c_index")}
                assert max(delta.values()) < 2e-6, (c, delta)
                checks.append(
                    dict(
                        cohort=c, recomputed=point, stored=stored, max_abs_delta=max(delta.values())
                    )
                )
            print(c, arm, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_cohort.csv", index=False)
    df.groupby("arm")[["winner_accuracy", "c_index", "random_winner_accuracy"]].mean().to_csv(
        out / "means.csv"
    )
    comparisons = []
    for plus, minus in (
        ("KRAFTON-tf", "HGB-cut"),
        ("KRAFTON-tf", "M11R"),
        ("KRAFTON-tf-ourfeat", "KRAFTON-tf-ourfeat-bce"),
    ):
        for metric in ("winner_accuracy", "c_index"):
            d = np.mean([draws[plus][c][metric] - draws[minus][c][metric] for c in cohorts], axis=0)
            estimate = float(df[df.arm == plus][metric].mean() - df[df.arm == minus][metric].mean())
            lo, hi = np.quantile(d, [0.025, 0.975])
            comparisons.append(
                dict(
                    contrast=plus + "_minus_" + minus,
                    metric=metric,
                    estimate=estimate,
                    ci_low=float(lo),
                    ci_high=float(hi),
                    weighting="equal cohort; paired match resampling",
                )
            )
    pd.DataFrame(comparisons).to_csv(out / "contrasts.csv", index=False)
    save(out / "producer_check.json", checks)
    save(
        out / "complete.json",
        dict(
            cohorts=cohorts,
            helper_source=str(src),
            task="winner selection and ordering by observed/censored elimination times; no softmax probability comparison",
        ),
    )


if __name__ == "__main__":
    main()
