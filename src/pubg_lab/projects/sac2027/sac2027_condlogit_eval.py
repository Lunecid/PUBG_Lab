#!/usr/bin/env python3
"""Conditional-logit control and BCE-weight sweep: contrasts, closure shares, phase-minus-match differences, predictions.

Plan recorded before training: reports/sac2027_patch_v14/CONDLOGIT_PLAN_KO_2026-10-09.md (hash and launch time in
reports/sac2027_patch_v14/objective_controls_condlogit.log).  Stored test predictions of the objective suite
(``objective_controls/<cohort>/<condition>/seed*/pred_test.csv``, checkpoint selected by validation within-phase AUC) are
scored on resampled matches with the resampling seed of ``tools/sac2027_objective_all_cohorts.py``, so every interval
shares its resampled matches with the earlier objective analyses.  Nothing is retrained here.

Reported per cohort and pooled (DerSimonian-Laird) over the nine cohorts not used to form the explanation and over all 11:

* pairwise contrasts on within-phase AUC (original, centred, match-mean) and within-match AUC (pair-pooled, equal phase
  weights) for BCE - cond_logit, cond_logit - next-phase Cox, and BCE / next-phase Cox against Cox + w*BCE, w in
  {0.25, 0.5, 1, 2, 4};
* the closure share (Cox + w*BCE - Cox) / (BCE - Cox) of the within-phase gap, per draw, for every w;
* the phase-minus-match difference D = (within-phase difference) - (within-match difference with equal phase weights),
  per draw, so that "the gap depends on the unit" is itself an estimate with an interval (review item S2);
* the seven predictions of the plan.
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
from pubg_lab.projects.sac2027.sac2027_match_offset import BASE, N_BOOT, SEEDS, Decomposition  # noqa: E402

OUT = BASE / "control_evaluation/condlogit"
PLAN = BASE / "CONDLOGIT_PLAN_KO_2026-10-09.md"
HYPOTHESIS_COHORTS = ("erangel_tpp_squad", "taego_tpp_squad")
WEIGHTS = {
    "cox_truncated_bce_w025": 0.25,
    "cox_truncated_bce_w05": 0.5,
    "cox_truncated_bce": 1.0,
    "cox_truncated_bce_w2": 2.0,
    "cox_truncated_bce_w4": 4.0,
}
CONDITIONS = ("bce", "cox_truncated", "cond_logit") + tuple(WEIGHTS)
PAIRS = (("bce", "cond_logit"), ("cond_logit", "cox_truncated"), ("bce", "cox_truncated")) + tuple(
    p for v in WEIGHTS for p in (("bce", v), (v, "cox_truncated"))
)
METRICS = (
    "original_within_phase",
    "centred_within_phase",
    "match_mean_between_match",
    "within_match",
    "within_match_phase_mean",
)
DID_PAIRS = (
    ("bce", "cox_truncated"),
    ("bce", "cond_logit"),
    ("cond_logit", "cox_truncated"),
    ("cox_truncated_bce", "cox_truncated"),
    ("cox_truncated_bce_w4", "cox_truncated"),
)
SESOI = 0.005


def quantiles(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    lo90, hi90 = np.quantile(d, [0.05, 0.95])
    return dict(
        se=float(d.std(ddof=1)),
        ci_low=float(lo),
        ci_high=float(hi),
        ci90_low=float(lo90),
        ci90_high=float(hi90),
    )


def cohort_rows(cohort: str) -> tuple[list[dict], list[dict]]:
    root = BASE / "objective_controls" / cohort
    frames, files = {}, []
    for c in CONDITIONS:
        for s in SEEDS:
            path = root / c / f"seed{s}" / "pred_test.csv"
            frames[(c, s)] = M.read_predictions(path)
            files.append(
                dict(
                    path=str(path.relative_to(REPO)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    seed = int(hashlib.sha256(f"20261008|objective|{cohort}|all".encode()).hexdigest()[:8], 16)
    n_matches = frames[("bce", SEEDS[0])]["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    point, draws = {}, {}
    for key, frame in frames.items():
        dec = Decomposition(frame)
        assert dec.n_matches == n_matches, key
        point[key] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
        draws[key] = dec.evaluate(counts)
        rs = RiskSetAUC(frame)
        point[key]["within_match"], draws[key]["within_match"] = (
            float(rs.evaluate(np.ones(n_matches))),
            rs.evaluate(counts),
        )
    pm = lambda cond, m: float(np.mean([point[(cond, s)][m] for s in SEEDS]))
    dm = lambda cond, m: np.mean([draws[(cond, s)][m] for s in SEEDS], axis=0)
    rows = []
    for c in CONDITIONS:
        for m in METRICS:
            rows.append(
                dict(
                    cohort=cohort,
                    contrast=f"level:{c}",
                    metric=m,
                    estimate=pm(c, m),
                    se=float("nan"),
                )
            )
    for plus, minus in PAIRS:
        for m in METRICS:
            rows.append(
                dict(
                    cohort=cohort,
                    contrast=f"{plus}_minus_{minus}",
                    metric=m,
                    estimate=pm(plus, m) - pm(minus, m),
                    **quantiles(dm(plus, m) - dm(minus, m)),
                )
            )
    m = "original_within_phase"
    gap_point = pm("bce", m) - pm("cox_truncated", m)
    gap_draws = dm("bce", m) - dm("cox_truncated", m)
    for v, w in WEIGHTS.items():
        rows.append(
            dict(
                cohort=cohort,
                contrast=f"closure_share:{v}",
                metric=m,
                weight=w,
                estimate=(pm(v, m) - pm("cox_truncated", m)) / gap_point,
                **quantiles((dm(v, m) - dm("cox_truncated", m)) / gap_draws),
            )
        )
    rows.append(
        dict(
            cohort=cohort,
            contrast="closure_share:cond_logit_as_reference",
            metric=m,
            estimate=(pm("bce", m) - pm("cond_logit", m)) / gap_point,
            **quantiles((dm("bce", m) - dm("cond_logit", m)) / gap_draws),
        )
    )
    # Phase-minus-match difference: within-phase difference minus within-match difference with equal phase weights.
    for plus, minus in DID_PAIRS:
        dp = dm(plus, "original_within_phase") - dm(minus, "original_within_phase")
        dq = dm(plus, "within_match_phase_mean") - dm(minus, "within_match_phase_mean")
        est = (pm(plus, "original_within_phase") - pm(minus, "original_within_phase")) - (
            pm(plus, "within_match_phase_mean") - pm(minus, "within_match_phase_mean")
        )
        rows.append(
            dict(
                cohort=cohort,
                contrast=f"{plus}_minus_{minus}",
                metric="phase_minus_match",
                estimate=est,
                **quantiles(dp - dq),
            )
        )
    return rows, files


def pool(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (contrast, metric), g in per[~per.contrast.str.startswith("level:")].groupby(
        ["contrast", "metric"]
    ):
        for scope in ("nine_new", "all_11"):
            h = g[~g.cohort.isin(HYPOTHESIS_COHORTS)] if scope == "nine_new" else g
            h = h[np.isfinite(h["se"])]
            if len(h) < 2:
                continue
            dl = M.dersimonian_laird(h["estimate"].to_numpy(), h["se"].to_numpy())
            same = int((np.sign(h["estimate"]) == np.sign(dl["theta"])).sum())
            out.append(dict(scope=scope, contrast=contrast, metric=metric, same_sign=same, **dl))
    return pd.DataFrame(out)


def interval(r, key):
    lo, hi = (float(x) for x in str(r[key]).strip("()[]").split(","))
    return lo, hi


def held(kind: str, lo95, hi95, lo90, hi90) -> bool:
    if kind == "positive":
        return lo95 > 0
    if kind == "equivalent":
        return lo90 > -SESOI and hi90 < SESOI
    if kind == "includes_zero":
        return lo95 <= 0 <= hi95
    raise ValueError(kind)


def predictions(per: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p = per.set_index(["cohort", "contrast", "metric"])
    tests = [
        (1, "bce_minus_cond_logit", "original_within_phase", "positive"),
        (2, "bce_minus_cond_logit", "within_match", "equivalent"),
        (3, "cond_logit_minus_cox_truncated", "original_within_phase", "equivalent"),
        (3, "cond_logit_minus_cox_truncated", "within_match", "equivalent"),
        (4, "bce_minus_cox_truncated_bce_w4", "original_within_phase", "includes_zero"),
    ] + [(5, f"{v}_minus_cox_truncated", "within_match", "equivalent") for v in WEIGHTS]
    for cohort in HYPOTHESIS_COHORTS:
        for k, contrast, metric, kind in tests:
            r = p.loc[(cohort, contrast, metric)]
            rows.append(
                dict(
                    scope=cohort,
                    prediction=k,
                    contrast=contrast,
                    metric=metric,
                    kind=kind,
                    estimate=float(r["estimate"]),
                    ci95_low=float(r["ci_low"]),
                    ci95_high=float(r["ci_high"]),
                    ci90_low=float(r["ci90_low"]),
                    ci90_high=float(r["ci90_high"]),
                    held=held(kind, r["ci_low"], r["ci_high"], r["ci90_low"], r["ci90_high"]),
                )
            )
        shares = [
            float(p.loc[(cohort, f"closure_share:{v}", "original_within_phase")]["estimate"])
            for v in WEIGHTS
        ]
        rows.append(
            dict(
                scope=cohort,
                prediction=4,
                contrast="closure_share monotone in w",
                metric="original_within_phase",
                kind="monotone",
                estimate=float("nan"),
                held=bool(np.all(np.diff(shares) > 0)),
                note=" < ".join(f"{w}:{s:.2f}" for w, s in zip(WEIGHTS.values(), shares)),
            )
        )
    q = pooled[pooled.scope == "nine_new"].set_index(["contrast", "metric"])
    for k, contrast, metric, kind in [
        (6, "bce_minus_cond_logit", "original_within_phase", "positive"),
        (6, "bce_minus_cond_logit", "within_match", "equivalent"),
        (6, "cond_logit_minus_cox_truncated", "original_within_phase", "equivalent"),
        (6, "cond_logit_minus_cox_truncated", "within_match", "equivalent"),
        (7, "bce_minus_cox_truncated_bce_w4", "original_within_phase", "includes_zero"),
    ]:
        if (contrast, metric) not in q.index:
            continue
        r = q.loc[(contrast, metric)]
        lo95, hi95 = interval(r, "ci95")
        lo90, hi90 = interval(r, "ci90")
        rows.append(
            dict(
                scope="nine_new_pooled",
                prediction=k,
                contrast=contrast,
                metric=metric,
                kind=kind,
                estimate=float(r["theta"]),
                ci95_low=lo95,
                ci95_high=hi95,
                ci90_low=lo90,
                ci90_high=hi90,
                held=held(kind, lo95, hi95, lo90, hi90),
                k=int(r["k"]),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cohorts",
        nargs="*",
        default=None,
        help="default: the 11 primary cohorts with results on disk",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = (
        args.cohorts
        or pd.read_csv(BASE / "cohort_counts.csv")
        .query("analysis_role == 'main'")["cohort"]
        .tolist()
    )
    rows, files, done = [], [], []
    for cohort in cohorts:
        if not all(
            (BASE / "objective_controls" / cohort / c / f"seed{s}" / "pred_test.csv").exists()
            for c in CONDITIONS
            for s in SEEDS
        ):
            print("SKIP (incomplete)", cohort, flush=True)
            continue
        print("START", cohort, flush=True)
        r, f = cohort_rows(cohort)
        rows += r
        files += f
        done.append(cohort)
        pd.DataFrame(rows).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
    per = pd.DataFrame(rows)
    pooled = pool(per)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    pred = (
        predictions(per, pooled) if all(c in done for c in HYPOTHESIS_COHORTS) else pd.DataFrame()
    )
    pred.to_csv(OUT / "predictions.csv", index=False)
    plan = (
        dict(
            plan=str(PLAN.relative_to(REPO)),
            plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        )
        if PLAN.exists()
        else dict(plan=None, plan_sha256=None)
    )  # the plan file is not in the public release
    (OUT / "complete.json").write_text(
        json.dumps(
            dict(
                cohorts=done,
                conditions=CONDITIONS,
                pairs=PAIRS,
                n_boot=N_BOOT,
                seeds=SEEDS,
                **plan,
                files=files,
            ),
            indent=1,
        )
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if len(pred):
            print(pred.to_string(index=False))


if __name__ == "__main__":
    main()
