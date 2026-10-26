#!/usr/bin/env python3
"""Ceiling-aware review metrics for the SAC 2027 manuscript (stored seed-42 test predictions; nothing is retrained).

recall@3 (``tools/sac2027_extra_evaluation.py``) pools, over boundaries with at least one elimination, the eliminations among
the three highest-risk teams divided by all eliminations.  Its attainable maximum differs by boundary: min(3, E_g) of E_g
eliminations when more than three teams are at risk, and all of them otherwise, so boundaries with many eliminations cap it
and boundaries with few teams saturate it.  Two ceiling-aware variants are reported per design:

* normalized recall@3: captured eliminations divided by the attainable maximum, pooled over boundaries;
* R-precision: at each boundary the top E_g teams are flagged (k = number of eliminations), pooled as captured / sum E_g.

Ties are broken at random in expectation, as in recall@3.  Per-cohort values are averaged over the 11 primary cohorts, like
the recall@3 values quoted in the text (``control_evaluation/extra_v16/arm_means.csv``).

Design differences get the intervals of the recall@3 differences: the same match-bootstrap draws (``extra_v16`` seeds,
2,000 resamples) per cohort and DerSimonian-Laird pooling over the 11 cohorts.  The recall@3 rows repeat the stored
``extra_v16`` contrasts as a check.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pubg_lab.survival.ptr.evaluation.metrics as M  # noqa: E402
import projects.sac2027.evaluation.metrics.sac2027_extra_evaluation as X  # noqa: E402

OUT = X.BASE / "control_evaluation/recall_norm"
NAMES = (
    "GBDT",
    "PTR",
    "Logistic",
    "KRAFTON",
    "KRAFTON+phase",
    "Sagong",
    "Union",
    "Members",
    "Phase-only",
)
CONTRASTS = (
    ("GBDT", "Members"),
    ("GBDT", "KRAFTON+phase"),
    ("PTR", "KRAFTON+phase"),
    ("PTR", "GBDT"),
    ("Union", "GBDT"),
)


def expected_captured(yg: np.ndarray, sg: np.ndarray, k: int) -> float:
    """Expected eliminations among the k highest scores, ties broken at random."""
    if len(yg) <= k:
        return float(yg.sum())
    cut = np.sort(sg)[::-1][k - 1]
    above, tie = sg > cut, sg == cut
    return float(yg[above].sum() + (k - above.sum()) / tie.sum() * yg[tie].sum())


def per_match(frame: pd.DataFrame, risk: np.ndarray) -> np.ndarray:
    """Per-match sums over boundaries with at least one elimination: eliminations, top-3 captures, attainable top-3
    maxima, and top-E_g captures (rows), with matches in sorted match_id order like the bootstrap weights."""
    codes = np.unique(frame["match_id"].to_numpy(), return_inverse=True)[1]
    y = frame["y"].to_numpy(float)
    groups = frame.groupby(["match_id", "phase"], sort=True).ngroup().to_numpy()
    order = np.argsort(groups, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(groups))]
    sums = np.zeros((4, codes.max() + 1))
    for g in range(len(bounds) - 1):
        idx = order[bounds[g] : bounds[g + 1]]
        yg, sg = y[idx], risk[idx]
        e = yg.sum()
        if e == 0:
            continue
        sums[:, codes[idx[0]]] += (
            e,
            expected_captured(yg, sg, X.K),
            e if len(idx) <= X.K else min(X.K, e),
            expected_captured(yg, sg, int(e)),
        )
    return sums


def metrics(sums: np.ndarray, w: np.ndarray) -> dict:
    """Pooled metrics for match weights ``w`` ([n_matches] for the estimate, [n_boot, n_matches] for the draws)."""
    events, cap3, maxcap, caprp = np.moveaxis(np.asarray(w, dtype=np.float64) @ sums.T, -1, 0)
    return dict(recall3=cap3 / events, recall3_norm=cap3 / maxcap, r_precision=caprp / events)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = (
        pd.read_csv(X.BASE / "cohort_counts.csv")
        .query("analysis_role == 'main'")["cohort"]
        .tolist()
    )
    rows, contrasts = [], []
    for cohort in cohorts:
        frames, _ = X.load(cohort)
        n_matches = frames["GBDT"]["match_id"].nunique()
        counts = M.resample_counts(n_matches, X.N_BOOT, X.seed_of(cohort, "all"))
        point, draws = {}, {}
        for name in NAMES:
            if name == "Members":
                f = frames["GBDT"]
                risk = -f["members"].to_numpy(float)
            else:
                f = frames[name]
                risk = f["p_hazard"].to_numpy(float)
            sums = per_match(f, risk)
            assert sums.shape[1] == n_matches, (cohort, name)
            point[name] = {k: float(v) for k, v in metrics(sums, np.ones(n_matches)).items()}
            draws[name] = metrics(sums, counts)
            rows.append(dict(cohort=cohort, arm=name, **point[name]))
        contrasts += X.contrast_rows(
            cohort, "all", point, draws, CONTRASTS, ("recall3", "r_precision")
        )
        print("DONE", cohort, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_cohort.csv", index=False)
    means = (
        per.groupby("arm")[["recall3", "recall3_norm", "r_precision"]].mean().reindex(list(NAMES))
    )
    means.to_csv(OUT / "arm_means.csv")
    print(means.round(4).to_string())
    per_c = pd.DataFrame(contrasts)
    per_c.to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    pooled = []
    for (name, m), g in per_c.groupby(["contrast", "metric"]):
        dl = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        pooled.append(
            dict(
                contrast=name,
                metric=m,
                same_sign=int((np.sign(g["estimate"]) == np.sign(dl["theta"])).sum()),
                **dl,
            )
        )
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            pooled[["contrast", "metric", "same_sign", "theta", "ci95", "ci90"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
