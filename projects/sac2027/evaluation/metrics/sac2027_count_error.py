#!/usr/bin/env python3
"""Post hoc checks of the RQ1 explanation on stored objective-control predictions (nothing is retrained).

1. Boundary-level elimination counts.  For every scored boundary (match-phase) the predicted number of next-phase
   eliminations, sum_i p_hat, is compared with the observed number, sum_i Y.  The mean absolute count error over
   boundaries asks whether a loss learns each match-phase's risk level: Cox-type losses get no signal for it, BCE does.
   p_hat is each run's own validation calibration (logistic slope on the score plus phase intercepts,
   ``tools/sac2027_objective_controls.py``), so every condition is calibrated per phase on validation data.
2. Human teams only.  Within-phase and within-match AUC of BCE and next-phase Cox restricted to rows of human teams
   (``team_ai_frac == 0``, the definition of the human-team evaluation), to check that BCE's within-phase advantage is
   not carried by AI teams.

Intervals: paired match bootstrap with the resampling seed of ``tools/sac2027_objective_all_cohorts.py`` (2,000 draws;
human-only metrics use the same draws restricted to matches with human rows), seed-averaged like
``tools/sac2027_condlogit_eval.py``; DerSimonian-Laird pooling over the nine cohorts not used to form the explanation
and over all 11.
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

import hashlib
import json
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pubg_lab.survival.ptr.data.data as D  # noqa: E402
import pubg_lab.survival.ptr.models.hgb as H  # noqa: E402
import pubg_lab.survival.ptr.evaluation.metrics as M  # noqa: E402
from pubg_lab.survival.ptr.evaluation.supplement import RiskSetAUC  # noqa: E402
from projects.sac2027.evaluation.risk_models.sac2027_match_offset import (
    BASE,
    N_BOOT,
    SEEDS,
    Decomposition,
)  # noqa: E402

OUT = BASE / "control_evaluation/count_error"
HYPOTHESIS_COHORTS = ("erangel_tpp_squad", "taego_tpp_squad")
CONDITIONS = ("bce", "cox_truncated", "cond_logit", "cox_truncated_bce", "cox_truncated_bce_w4")
COUNT_PAIRS = (
    ("bce", "cox_truncated"),
    ("bce", "cond_logit"),
    ("cond_logit", "cox_truncated"),
    ("cox_truncated_bce", "cox_truncated"),
    ("cox_truncated_bce_w4", "cox_truncated"),
)
HUMAN_PAIRS = (("bce", "cox_truncated"),)
HUMAN_METRICS = ("original_within_phase", "within_match")


class CountError:
    """Mean absolute boundary-level count error |sum p_hat - sum Y| as a function of per-match bootstrap weights."""

    def __init__(self, frame: pd.DataFrame) -> None:
        match_ids = np.unique(frame["match_id"].to_numpy())
        g = (
            frame.groupby(["match_id", "phase"], sort=True)
            .agg(p=("p_hazard", "sum"), y=("y", "sum"))
            .reset_index()
        )
        codes = np.searchsorted(match_ids, g["match_id"].to_numpy())
        self.n_matches = len(match_ids)
        self.abs_sum = np.bincount(
            codes, weights=np.abs(g["p"] - g["y"]).to_numpy(), minlength=self.n_matches
        )
        self.signed_sum = np.bincount(
            codes, weights=(g["p"] - g["y"]).to_numpy(), minlength=self.n_matches
        )
        self.n_bound = np.bincount(codes, minlength=self.n_matches).astype(np.float64)

    def evaluate(self, w: np.ndarray) -> dict:
        w = np.asarray(w, dtype=np.float64)
        den = w @ self.n_bound
        return dict(count_mae=(w @ self.abs_sum) / den, count_bias=(w @ self.signed_sum) / den)


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


def ai_meta(cohort: str) -> pd.DataFrame:
    rows = H.build_rows(D.load_dump(BASE / "dumps/phase_v2" / cohort / "test.pt"), "HGB-cut", "v2")
    meta = rows.keys.copy()
    meta["ai_meta"] = rows.X[:, rows.feature_names.index("team_ai_frac")]
    return meta


def cohort_rows(cohort: str) -> tuple[list[dict], list[dict]]:
    root = BASE / "objective_controls" / cohort
    meta = ai_meta(cohort)
    frames, files = {}, []
    for c in CONDITIONS:
        for s in SEEDS:
            path = root / c / f"seed{s}" / "pred_test.csv"
            f = M.read_predictions(path).merge(
                meta, on=M.KEY_COLUMNS, validate="one_to_one", how="left"
            )
            assert f["ai_meta"].notna().all(), (cohort, c, s)
            frames[(c, s)] = f
            files.append(
                dict(
                    path=str(path.relative_to(REPO)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    seed = int(hashlib.sha256(f"20261008|objective|{cohort}|all".encode()).hexdigest()[:8], 16)
    full_ids = np.unique(frames[("bce", SEEDS[0])]["match_id"].to_numpy())
    counts = M.resample_counts(len(full_ids), N_BOOT, seed)
    point, draws = {}, {}
    for key, f in frames.items():
        ce = CountError(f)
        assert ce.n_matches == len(full_ids), key
        point[key] = {m: float(v) for m, v in ce.evaluate(np.ones(len(full_ids))).items()}
        draws[key] = ce.evaluate(counts)
        if key[0] in {c for pair in HUMAN_PAIRS for c in pair}:
            h = f.loc[f["ai_meta"] == 0, M.PRED_COLUMNS].reset_index(drop=True)
            idx = np.searchsorted(full_ids, np.unique(h["match_id"].to_numpy()))
            hc, ones = counts[:, idx], np.ones(len(idx))
            dec, rs = Decomposition(h), RiskSetAUC(h)
            point[key]["human_original_within_phase"] = float(
                dec.evaluate(ones)["original_within_phase"][0]
            )
            draws[key]["human_original_within_phase"] = dec.evaluate(hc)["original_within_phase"]
            point[key]["human_within_match"], draws[key]["human_within_match"] = (
                float(rs.evaluate(ones)),
                rs.evaluate(hc),
            )
    pm = lambda c, m: float(np.mean([point[(c, s)][m] for s in SEEDS]))
    dm = lambda c, m: np.mean([draws[(c, s)][m] for s in SEEDS], axis=0)
    rows = []
    for c in CONDITIONS:
        for m in ("count_mae", "count_bias"):
            rows.append(
                dict(
                    cohort=cohort,
                    contrast=f"level:{c}",
                    metric=m,
                    estimate=pm(c, m),
                    se=float("nan"),
                )
            )
    for plus, minus in COUNT_PAIRS:
        rows.append(
            dict(
                cohort=cohort,
                contrast=f"{plus}_minus_{minus}",
                metric="count_mae",
                estimate=pm(plus, "count_mae") - pm(minus, "count_mae"),
                **quantiles(dm(plus, "count_mae") - dm(minus, "count_mae")),
            )
        )
    for plus, minus in HUMAN_PAIRS:
        for m in HUMAN_METRICS:
            k = f"human_{m}"
            rows.append(
                dict(
                    cohort=cohort,
                    contrast=f"{plus}_minus_{minus}",
                    metric=k,
                    estimate=pm(plus, k) - pm(minus, k),
                    **quantiles(dm(plus, k) - dm(minus, k)),
                )
            )
    f0 = frames[("bce", SEEDS[0])]
    rows.append(
        dict(
            cohort=cohort,
            contrast="info",
            metric="human_row_share",
            estimate=float((f0["ai_meta"] == 0).mean()),
            se=float("nan"),
        )
    )
    rows.append(
        dict(
            cohort=cohort,
            contrast="info",
            metric="mean_observed_eliminations_per_boundary",
            estimate=float(f0.groupby(["match_id", "phase"])["y"].sum().mean()),
            se=float("nan"),
        )
    )
    return rows, files


def pool(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (contrast, metric), g in per[per.se.notna()].groupby(["contrast", "metric"]):
        for scope in ("nine_new", "all_11"):
            h = g[~g.cohort.isin(HYPOTHESIS_COHORTS)] if scope == "nine_new" else g
            if len(h) < 2:
                continue
            dl = M.dersimonian_laird(h["estimate"].to_numpy(), h["se"].to_numpy())
            out.append(
                dict(
                    scope=scope,
                    contrast=contrast,
                    metric=metric,
                    same_sign=int((np.sign(h["estimate"]) == np.sign(dl["theta"])).sum()),
                    **dl,
                )
            )
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = (
        pd.read_csv(BASE / "cohort_counts.csv").query("analysis_role == 'main'")["cohort"].tolist()
    )
    rows, files = [], []
    for c in cohorts:
        r, f = cohort_rows(c)
        rows += r
        files += f
        print("DONE", c, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_cohort.csv", index=False)
    pooled = pool(per)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    (OUT / "complete.json").write_text(
        json.dumps(
            dict(cohorts=cohorts, conditions=CONDITIONS, seeds=SEEDS, n_boot=N_BOOT, files=files),
            indent=1,
        )
    )
    with pd.option_context(
        "display.width", 220, "display.max_columns", 20, "display.max_rows", 200
    ):
        print(
            per[per.cohort.isin(HYPOTHESIS_COHORTS)][
                ["cohort", "contrast", "metric", "estimate", "ci_low", "ci_high"]
            ].to_string(index=False)
        )
        print(
            pooled[["scope", "contrast", "metric", "same_sign", "theta", "ci95", "ci90"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
