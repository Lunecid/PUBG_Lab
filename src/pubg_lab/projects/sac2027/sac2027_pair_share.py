#!/usr/bin/env python3
"""Share of same-match pairs among the (eliminated, surviving) pairs of within-phase AUC, per primary cohort.

Within one phase, a pair either shares a match or comes from two matches; the share of same-match pairs is about 1/M
for M test matches.  Pair composition depends only on labels, phases, and match ids, which every design shares, so the
reference GBDT test predictions supply them.  Eligible phases follow WithinPhaseAUC (at least MIN_PHASE_ROWS rows and
both classes); ``share_mean_over_phases`` averages per-phase shares as within-phase AUC averages per-phase AUCs.
Output: reports/sac2027_patch_v14/control_evaluation/pair_share/pair_share.csv
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pubg_lab.survival.ptr import metrics as M  # noqa: E402
from pubg_lab.survival.ptr.metrics import MIN_PHASE_ROWS  # noqa: E402

BASE = REPO / "reports/sac2027_patch_v14"
OUT = BASE / "control_evaluation/pair_share"


def shares(frame: pd.DataFrame) -> dict:
    per_phase, same_total, all_total = [], 0.0, 0.0
    for _, g in frame.groupby("phase"):
        y = g["y"].to_numpy()
        if len(g) < MIN_PHASE_ROWS or y.min() == y.max():
            continue
        t = g.groupby("match_id")["y"].agg(["sum", "count"])
        same = float((t["sum"] * (t["count"] - t["sum"])).sum())
        pairs = float(y.sum() * (len(y) - y.sum()))
        per_phase.append(same / pairs)
        same_total += same
        all_total += pairs
    return dict(
        n_matches=int(frame["match_id"].nunique()),
        n_phases=len(per_phase),
        share_mean_over_phases=float(np.mean(per_phase)),
        share_pooled=same_total / all_total,
    )


def main() -> None:
    counts = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = sorted(counts.loc[counts.analysis_role == "main", "cohort"])
    rows = []
    for cohort in cohorts:
        frame = M.read_predictions(
            BASE / "checkpoints/ptr_v2" / cohort / "HGB-cut" / "seed42" / "pred_test.csv"
        )
        rows.append(dict(cohort=cohort, **shares(frame)))
    out = pd.DataFrame(rows)
    out["inverse_matches"] = 1.0 / out["n_matches"]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "pair_share.csv", index=False)
    print(out.to_string(index=False))
    print(
        "range of share_mean_over_phases:",
        out.share_mean_over_phases.min(),
        out.share_mean_over_phases.max(),
    )


if __name__ == "__main__":
    main()
