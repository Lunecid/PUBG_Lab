#!/usr/bin/env python3
"""Objective controls in all 11 primary cohorts: per-cohort contrasts, random-effects pooling, and the recorded predictions.

Plan recorded before training: reports/sac2027_patch_v14/OBJECTIVE_REPLICATION_PLAN_KO_2026-10-09.md.  For each
checkpoint-selection rule (validation within-phase AUC: ``objective_controls``; each objective's own validation loss:
``objective_controls_ownloss``) and cohort, test predictions of the six losses x three seeds are scored with the
match-offset decomposition and within-match AUC on resampled matches (the resampling seed of the objective suite).
Seed-averaged contrasts are pooled with DerSimonian-Laird over (a) the nine cohorts not used to form the explanation
and (b) all 11 cohorts.  The seven recorded predictions are checked on (a) under the original selection rule.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import hashlib
import json
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pubg_lab.survival.ptr import metrics as M  # noqa: E402
from pubg_lab.survival.ptr.supplement import RiskSetAUC  # noqa: E402
from pubg_lab.projects.sac2027.sac2027_match_offset import (
    BASE,
    CONDITIONS,
    N_BOOT,
    SEEDS,
    Decomposition,
)  # noqa: E402

OUT = BASE / "control_evaluation/objective_all_cohorts"
SELECTIONS = {"within_phase_auc": "objective_controls", "own_loss": "objective_controls_ownloss"}
HYPOTHESIS_COHORTS = ("erangel_tpp_squad", "taego_tpp_squad")
PAIRS = (
    ("bce", "cox_truncated"),
    ("bce", "cox_full"),
    ("cox_truncated", "cox_full"),
    ("bce", "cox_truncated_bce"),
    ("cox_truncated_bce", "cox_truncated"),
    ("weighted_truncated", "weighted_full"),
    ("bce", "weighted_truncated"),
)
METRICS = (
    "original_within_phase",
    "centred_within_phase",
    "match_mean_between_match",
    "within_match",
    "within_match_phase_mean",
    "within_match_set_mean",
)
SESOI = 0.005


def cohort_contrasts(selection: str, cohort: str) -> list[dict]:
    root = BASE / SELECTIONS[selection] / cohort
    frames = {
        (c, s): M.read_predictions(root / c / f"seed{s}" / "pred_test.csv")
        for c in CONDITIONS
        for s in SEEDS
    }
    seed = int(hashlib.sha256(f"20261008|objective|{cohort}|all".encode()).hexdigest()[:8], 16)
    n_matches = frames[("bce", SEEDS[0])]["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    point, draws = {}, {}
    for key, frame in frames.items():
        dec = Decomposition(frame)
        point[key] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
        draws[key] = dec.evaluate(counts)
        rs = RiskSetAUC(frame)
        point[key]["within_match"], draws[key]["within_match"] = (
            rs.evaluate(np.ones(n_matches)),
            rs.evaluate(counts),
        )
    mean = lambda cond, src, m: np.mean([src[(cond, s)][m] for s in SEEDS], axis=0)
    rows = []
    for plus, minus in PAIRS:
        for m in METRICS:
            d = mean(plus, draws, m) - mean(minus, draws, m)
            d = d[np.isfinite(d)]
            rows.append(
                dict(
                    selection=selection,
                    cohort=cohort,
                    contrast=f"{plus}_minus_{minus}",
                    metric=m,
                    estimate=float(mean(plus, point, m) - mean(minus, point, m)),
                    se=float(d.std(ddof=1)),
                )
            )
    for c in CONDITIONS:
        for m in METRICS:
            rows.append(
                dict(
                    selection=selection,
                    cohort=cohort,
                    contrast=f"level:{c}",
                    metric=m,
                    estimate=float(mean(c, point, m)),
                    se=float("nan"),
                )
            )
    return rows


def pool(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (selection, contrast, metric), g in per[~per.contrast.str.startswith("level:")].groupby(
        ["selection", "contrast", "metric"]
    ):
        for scope, cohorts in (("nine_new", None), ("all_11", "all")):
            h = g[~g.cohort.isin(HYPOTHESIS_COHORTS)] if scope == "nine_new" else g
            dl = M.dersimonian_laird(h["estimate"].to_numpy(), h["se"].to_numpy())
            same = int((np.sign(h["estimate"]) == np.sign(dl["theta"])).sum())
            out.append(
                dict(
                    selection=selection,
                    scope=scope,
                    contrast=contrast,
                    metric=metric,
                    same_sign=same,
                    **dl,
                )
            )
    return pd.DataFrame(out)


def interval(r, key):
    lo, hi = (float(x) for x in str(r[key]).strip("()[]").split(","))
    return lo, hi


def predictions(pooled: pd.DataFrame) -> pd.DataFrame:
    p = pooled[(pooled.selection == "within_phase_auc") & (pooled.scope == "nine_new")].set_index(
        ["contrast", "metric"]
    )
    tests = [
        (1, "bce_minus_cox_truncated", "within_match", "equivalent"),
        (2, "bce_minus_cox_truncated", "original_within_phase", "positive"),
        (3, "bce_minus_cox_truncated", "centred_within_phase", "equivalent"),
        (4, "cox_truncated_bce_minus_cox_truncated", "original_within_phase", "positive"),
        (5, "bce_minus_cox_truncated_bce", "original_within_phase", "equivalent"),
        (6, "cox_truncated_bce_minus_cox_truncated", "within_match", "equivalent"),
        (7, "cox_truncated_minus_cox_full", "within_match", "positive"),
    ]
    rows = []
    for k, contrast, metric, kind in tests:
        r = p.loc[(contrast, metric)]
        lo95, hi95 = interval(r, "ci95")
        lo90, hi90 = interval(r, "ci90")
        held = (lo95 > 0) if kind == "positive" else (lo90 > -SESOI and hi90 < SESOI)
        rows.append(
            dict(
                prediction=k,
                contrast=contrast,
                metric=metric,
                kind=kind,
                theta=float(r["theta"]),
                ci95_low=lo95,
                ci95_high=hi95,
                ci90_low=lo90,
                ci90_high=hi90,
                held=bool(held),
                k=int(r["k"]),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", nargs="+", choices=tuple(SELECTIONS), default=tuple(SELECTIONS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = (
        pd.read_csv(BASE / "cohort_counts.csv").query("analysis_role == 'main'")["cohort"].tolist()
    )
    rows = []
    for selection in args.selections:
        for cohort in cohorts:
            print("START", selection, cohort, flush=True)
            rows += cohort_contrasts(selection, cohort)
            pd.DataFrame(rows).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    per = pd.DataFrame(rows)
    pooled = pool(per)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    if "within_phase_auc" in args.selections:
        pred = predictions(pooled)
        pred.to_csv(OUT / "predictions.csv", index=False)
        print(pred.to_string(index=False))
    (OUT / "complete.json").write_text(
        json.dumps(
            dict(
                cohorts=cohorts,
                selections=SELECTIONS,
                pairs=PAIRS,
                n_boot=N_BOOT,
                hypothesis_cohorts=HYPOTHESIS_COHORTS,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
