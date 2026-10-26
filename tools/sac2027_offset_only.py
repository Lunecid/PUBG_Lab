#!/usr/bin/env python3
"""Offset-only control: frozen next-phase Cox scores plus a match-phase term learned with cross-entropy.

Plan recorded before running: reports/sac2027_patch_v14/OFFSET_ONLY_PLAN_KO_2026-10-09.md.  For each seed of the
next-phase Cox objective control, the Cox score s is split into its match-phase mean m_g and the centred score
r = s - m_g.  A logistic regression fitted on validation (patch 41.1) maps

    logit p = a * r + beta_phase + gamma' z_g,

where z_g holds only quantities shared by all teams of the match-phase and observed before the cutoff: the mean of the
151 reference inputs over the teams at risk (standardized with training-patch means and SDs), the log number of teams
at risk, and m_g.  With a > 0 the ranking inside every match-phase is the Cox ranking, so within-match AUC cannot
change; any within-phase gain comes from the match-level term.  Test predictions are scored on the resampled matches of
the objective suite (tools/sac2027_match_offset.py), next to next-phase Cox and BCE.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from model_stgcn.ptr import data as D  # noqa: E402
from model_stgcn.ptr import hgb as H  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402
from model_stgcn.ptr.supplement import RiskSetAUC  # noqa: E402
from tools.sac2027_match_offset import BASE, DUMP, N_BOOT, SEEDS, Decomposition  # noqa: E402

OUT = BASE / "control_evaluation/offset_only"
COHORTS = ("erangel_tpp_squad", "taego_tpp_squad")
GRID_C = (0.001, 0.01, 0.1, 1.0)


def match_phase_inputs(cohort: str) -> dict[str, pd.DataFrame]:
    """Per split: keys, y, and the match-phase means of the standardized reference inputs plus log teams at risk."""
    rows = {split: H.build_rows(D.load_dump(DUMP / cohort / f"{split}.pt"), "HGB-cut", "v2") for split in ("train", "val", "test")}
    mu = rows["train"].X.mean(axis=0)
    sd = rows["train"].X.std(axis=0)
    sd[sd == 0] = 1.0
    out = {}
    for split in ("val", "test"):
        r = rows[split]
        keys = r.keys[M.KEY_COLUMNS].reset_index(drop=True)
        z = pd.DataFrame((r.X - mu) / sd, columns=[f"z{i}" for i in range(r.X.shape[1])])
        z[["match_id", "phase"]] = keys[["match_id", "phase"]]
        g = z.groupby(["match_id", "phase"])
        means = g.transform("mean")
        means["log_at_risk"] = np.log(g["z0"].transform("size").to_numpy())
        out[split] = pd.concat([keys, means], axis=1)
    return out


def design(frame: pd.DataFrame, inputs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Columns: r, m_g, phase indicators, match-phase input means; rows aligned to ``frame``."""
    f = frame.merge(inputs, on=M.KEY_COLUMNS, validate="one_to_one", how="left")
    assert f.notna().all().all(), "missing inputs for some prediction rows"
    s = -f["rank_score"].to_numpy()  # the Cox score (logit of the raw hazard output)
    m_g = pd.Series(s).groupby([f["match_id"], f["phase"]]).transform("mean").to_numpy()
    phases = pd.get_dummies(f["phase"].astype(int), prefix="k").reindex(columns=[f"k_{k}" for k in range(1, 8)], fill_value=0)
    zcols = [c for c in inputs.columns if c.startswith("z") or c == "log_at_risk"]
    X = np.column_stack([s - m_g, m_g, phases.to_numpy(dtype=float), f[zcols].to_numpy(dtype=float)])
    return X, f["y"].to_numpy(), f


def fit(Xv: np.ndarray, yv: np.ndarray, groups: np.ndarray) -> tuple[LogisticRegression, dict]:
    """L2 logistic regression; C by match-grouped 5-fold validation log loss; columns after the phase block standardized."""
    scale_cols = np.r_[0:2, 9:Xv.shape[1]]
    mu, sd = Xv[:, scale_cols].mean(axis=0), Xv[:, scale_cols].std(axis=0)
    sd[sd == 0] = 1.0

    def scaled(X):
        X = X.copy()
        X[:, scale_cols] = (X[:, scale_cols] - mu) / sd
        return X

    Xs = scaled(Xv)
    cv = {}
    for C in GRID_C:
        losses = []
        for tr, te in GroupKFold(n_splits=5).split(Xs, yv, groups):
            lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Xs[tr], yv[tr])
            losses.append(log_loss(yv[te], lr.predict_proba(Xs[te])[:, 1], labels=[0, 1]))
        cv[C] = float(np.mean(losses))
    best = min(cv, key=cv.get)
    model = LogisticRegression(C=best, max_iter=2000, solver="lbfgs").fit(Xs, yv)
    a = float(model.coef_[0][0] / sd[0])
    assert a > 0, "the centred Cox score must keep a positive weight, or the within-match ranking would flip"
    model.scaled = scaled  # type: ignore[attr-defined]
    return model, dict(C=best, cv_log_loss=cv, weight_on_centred_cox_score=a)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arm_rows, contrast_rows, fits = [], [], {}
    for cohort in COHORTS:
        inputs = match_phase_inputs(cohort)
        frames = {}
        for seed in SEEDS:
            src = BASE / "objective_controls" / cohort / "cox_truncated" / f"seed{seed}"
            val = M.read_predictions(src / "pred_val_raw.csv")
            test = M.read_predictions(src / "pred_test_raw.csv")
            Xv, yv, fv = design(val, inputs["val"])
            Xt, _, ft = design(test, inputs["test"])
            model, info = fit(Xv, yv, fv["match_id"].to_numpy())
            p = model.predict_proba(model.scaled(Xt))[:, 1]
            pred = test.copy()
            key = pred[M.KEY_COLUMNS].merge(ft[M.KEY_COLUMNS].assign(p=p), on=M.KEY_COLUMNS, validate="one_to_one")
            pred["p_hazard"] = key["p"].to_numpy()
            pred["rank_score"] = -np.log(pred["p_hazard"] / (1 - pred["p_hazard"]))
            dest = OUT / cohort / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            M.write_predictions(pred, dest / "pred_test.csv")
            fits[f"{cohort}/seed{seed}"] = info
            frames[("offset_only", seed)] = pred
            for cond in ("cox_truncated", "bce"):
                frames[(cond, seed)] = M.read_predictions(BASE / "objective_controls" / cohort / cond / f"seed{seed}" / "pred_test.csv")
        seed_ = int(hashlib.sha256(f"20261008|objective|{cohort}|all".encode()).hexdigest()[:8], 16)
        n_matches = frames[("bce", SEEDS[0])]["match_id"].nunique()
        counts = M.resample_counts(n_matches, N_BOOT, seed_)
        point, draws = {}, {}
        for k, frame in frames.items():
            dec = Decomposition(frame)
            point[k] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
            draws[k] = dec.evaluate(counts)
            rs = RiskSetAUC(frame)
            point[k]["within_match"], draws[k]["within_match"] = rs.evaluate(np.ones(n_matches)), rs.evaluate(counts)
        metrics = [m for m in point[("bce", SEEDS[0])] if m != "same_match_pair_share"]
        for cond in ("offset_only", "cox_truncated", "bce"):
            arm_rows.append(dict(cohort=cohort, condition=cond, **{m: float(np.mean([point[(cond, s)][m] for s in SEEDS])) for m in metrics}))
        mean = lambda cond, src, m: np.mean([src[(cond, s)][m] for s in SEEDS], axis=0)
        for plus, minus in (("offset_only", "cox_truncated"), ("bce", "offset_only"), ("bce", "cox_truncated")):
            for m in metrics:
                est = mean(plus, point, m) - mean(minus, point, m)
                d = mean(plus, draws, m) - mean(minus, draws, m)
                d = d[np.isfinite(d)]
                contrast_rows.append(dict(cohort=cohort, contrast=f"{plus}_minus_{minus}", metric=m, estimate=float(est),
                                          se=float(d.std(ddof=1)), ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                                          ci90_low=float(np.quantile(d, .05)), ci90_high=float(np.quantile(d, .95))))
        m = "original_within_phase"
        est = (mean("offset_only", point, m) - mean("cox_truncated", point, m)) / (mean("bce", point, m) - mean("cox_truncated", point, m))
        r = (mean("offset_only", draws, m) - mean("cox_truncated", draws, m)) / (mean("bce", draws, m) - mean("cox_truncated", draws, m))
        r = r[np.isfinite(r)]
        contrast_rows.append(dict(cohort=cohort, contrast="closure_share_offset_only", metric=m, estimate=float(est), se=float(r.std(ddof=1)),
                                  ci_low=float(np.quantile(r, .025)), ci_high=float(np.quantile(r, .975)),
                                  ci90_low=float(np.quantile(r, .05)), ci90_high=float(np.quantile(r, .95))))
        pd.DataFrame(arm_rows).to_csv(OUT / "arm_metrics.csv", index=False)
        pd.DataFrame(contrast_rows).to_csv(OUT / "contrasts.csv", index=False)
        print("DONE", cohort, flush=True)
    (OUT / "fits.json").write_text(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
