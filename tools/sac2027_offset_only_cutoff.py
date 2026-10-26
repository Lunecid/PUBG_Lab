#!/usr/bin/env python3
"""Post hoc sensitivity of the offset-only control to its information set (not part of the recorded plan).

The recorded offset-only control (tools/sac2027_offset_only.py) summarizes each match-phase over the rows of the
retrospective risk set R (teams with final elimination time after the prediction time b): the log number of rows, the
mean Cox score, and the mean reference inputs.  R leaves out teams eliminated between the feature cutoff a and b, so
these summaries can carry information from after the cutoff.  Two variants remove it:

* ``no_size``: the recorded design without the log number of teams at risk;
* ``cutoff_only``: the raw next-phase Cox score s, phase indicators, and only the reference inputs that are constant
  within every match-phase (match-level quantities fixed at the cutoff, such as the alive-team count and the zone);
  no risk-set means, no risk-set size;
* ``cutoff_means``: the raw Cox score, phase indicators, the means of all reference inputs over the teams observed
  alive at the cutoff (the attention keys of the neural designs), and the log of their number;
* ``raw_risk_set``: the same form as ``cutoff_means`` (raw score, no separate mean-score term) but with the recorded
  risk-set summaries, so recorded -> raw_risk_set isolates the form and raw_risk_set -> cutoff_means the information.

With a positive weight on the score, both keep the Cox ranking inside every match-phase.  Fits use validation data
(patch 41.1) as in the recorded control; test contrasts use the same match resamples.
Usage: python tools/sac2027_offset_only_cutoff.py
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
from tools.sac2027_offset_only import COHORTS, GRID_C, match_phase_inputs  # noqa: E402

OUT = BASE / "control_evaluation/offset_only_cutoff"
VARIANTS = ("no_size", "cutoff_only", "cutoff_means", "raw_risk_set")


def constant_columns(cohort: str) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Reference inputs constant within every validation match-phase, and their per-match-phase values per split."""
    names, frames = None, {}
    for split in ("val", "test"):
        r = H.build_rows(D.load_dump(DUMP / cohort / f"{split}.pt"), "HGB-cut", "v2")
        names = r.feature_names
        f = pd.DataFrame(r.X, columns=names)
        f[["match_id", "phase"]] = r.keys[M.KEY_COLUMNS].reset_index(drop=True)[["match_id", "phase"]]
        frames[split] = f
    spread = frames["val"].groupby(["match_id", "phase"])[names].std().fillna(0.0).max()
    keep = [n for n in names if spread[n] < 1e-9 and frames["val"][n].std() > 0]
    out = {s: f.groupby(["match_id", "phase"])[keep].first().reset_index() for s, f in frames.items()}
    return keep, out


def cutoff_set_inputs(cohort: str) -> dict[str, pd.DataFrame]:
    """Per split and match-phase: means of the HGB-cut inputs over the teams causally alive at the cutoff, and log count."""
    out = {}
    for split in ("val", "test"):
        dump = D.load_dump(DUMP / cohort / f"{split}.pt")
        event = D.uses_event_alive(dump, "v2")
        recs = []
        for m in dump["matches"]:
            X = np.asarray(m["X"], dtype=np.float64)
            share = np.asarray(m["team_ai_frac"], dtype=np.float64)
            key = D.key_mask(m, "v2", event).numpy()
            for p in range(X.shape[0]):
                alive = np.nonzero(key[p])[0]
                if len(alive) == 0:
                    continue
                feats = np.concatenate([X[p, alive], share[alive][:, None]], axis=1)
                recs.append((str(m["match_id"]), p + M.PHASE_OFFSET, np.log(len(alive)), *np.nanmean(feats, axis=0)))
        n = len(recs[0]) - 3
        out[split] = pd.DataFrame(recs, columns=["match_id", "phase", "log_alive_cut", *[f"c{i}" for i in range(n)]])
    return out


def design(frame: pd.DataFrame, variant: str, recorded: pd.DataFrame, constant: pd.DataFrame, cutset: pd.DataFrame) -> tuple:
    """Columns: score (centred for no_size, raw for cutoff_only), [m_g], 7 phase indicators, match-phase covariates."""
    s0 = -frame["rank_score"].to_numpy()
    if variant == "no_size":
        f = frame.merge(recorded, on=M.KEY_COLUMNS, validate="one_to_one", how="left")
        zcols = [c for c in recorded.columns if c.startswith("z")]
    elif variant == "raw_risk_set":
        f = frame.merge(recorded, on=M.KEY_COLUMNS, validate="one_to_one", how="left")
        zcols = [c for c in recorded.columns if c.startswith("z") or c == "log_at_risk"]
    elif variant == "cutoff_only":
        f = frame.merge(constant, on=["match_id", "phase"], validate="many_to_one", how="left")
        zcols = [c for c in constant.columns if c not in ("match_id", "phase")]
    else:
        f = frame.merge(cutset, on=["match_id", "phase"], validate="many_to_one", how="left")
        zcols = [c for c in cutset.columns if c not in ("match_id", "phase")]
        f[zcols] = f[zcols].fillna(f[zcols].mean())
    assert f[zcols].notna().all().all(), "missing covariates for some prediction rows"
    s = -f["rank_score"].to_numpy()
    assert np.allclose(s, s0)
    phases = pd.get_dummies(f["phase"].astype(int), prefix="k").reindex(columns=[f"k_{k}" for k in range(1, 8)], fill_value=0)
    if variant == "no_size":
        m_g = pd.Series(s).groupby([f["match_id"], f["phase"]]).transform("mean").to_numpy()
        cols, dense = [s - m_g, m_g], 2
    else:
        cols, dense = [s], 1
    X = np.column_stack(cols + [phases.to_numpy(dtype=float), f[zcols].to_numpy(dtype=float)])
    scale = np.r_[0:dense, dense + 7:X.shape[1]]
    return X, f["y"].to_numpy(), f, scale


def fit(Xv: np.ndarray, yv: np.ndarray, groups: np.ndarray, scale: np.ndarray):
    mu, sd = Xv[:, scale].mean(axis=0), Xv[:, scale].std(axis=0)
    sd[sd == 0] = 1.0

    def scaled(X):
        X = X.copy()
        X[:, scale] = (X[:, scale] - mu) / sd
        return X

    Xs = scaled(Xv)
    cv = {}
    for C in GRID_C:
        losses = []
        for tr, te in GroupKFold(n_splits=5).split(Xs, yv, groups):
            lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(Xs[tr], yv[tr])
            losses.append(log_loss(yv[te], lr.predict_proba(Xs[te])[:, 1], labels=[0, 1]))
        cv[C] = float(np.mean(losses))
    best = min(cv, key=cv.get)
    model = LogisticRegression(C=best, max_iter=3000, solver="lbfgs").fit(Xs, yv)
    a = float(model.coef_[0][0] / sd[0])
    assert a > 0, "the Cox score must keep a positive weight, or the within-match ranking would flip"
    return model, scaled, dict(C=best, cv_log_loss=cv, weight_on_cox_score=a)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arm_rows, contrast_rows, fits, kept = [], [], {}, {}
    for cohort in COHORTS:
        recorded = match_phase_inputs(cohort)
        keep, constant = constant_columns(cohort)
        cutset = cutoff_set_inputs(cohort)
        kept[cohort] = keep
        frames = {}
        for seed in SEEDS:
            src = BASE / "objective_controls" / cohort / "cox_truncated" / f"seed{seed}"
            val = M.read_predictions(src / "pred_val_raw.csv")
            test = M.read_predictions(src / "pred_test_raw.csv")
            for variant in VARIANTS:
                Xv, yv, fv, scale = design(val, variant, recorded["val"], constant["val"], cutset["val"])
                Xt, _, ft, _ = design(test, variant, recorded["test"], constant["test"], cutset["test"])
                model, scaled, info = fit(Xv, yv, fv["match_id"].to_numpy(), scale)
                p = model.predict_proba(scaled(Xt))[:, 1]
                pred = test.copy()
                key = pred[M.KEY_COLUMNS].merge(ft[M.KEY_COLUMNS].assign(p=p), on=M.KEY_COLUMNS, validate="one_to_one")
                pred["p_hazard"] = key["p"].to_numpy()
                pred["rank_score"] = -np.log(pred["p_hazard"] / (1 - pred["p_hazard"]))
                fits[f"{cohort}/{variant}/seed{seed}"] = info
                frames[(variant, seed)] = pred
            frames[("offset_only", seed)] = M.read_predictions(BASE / "control_evaluation/offset_only" / cohort / f"seed{seed}" / "pred_test.csv")
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
        conds = (*VARIANTS, "offset_only", "cox_truncated", "bce")
        for cond in conds:
            arm_rows.append(dict(cohort=cohort, condition=cond, **{m: float(np.mean([point[(cond, s)][m] for s in SEEDS])) for m in metrics}))
        mean = lambda cond, src, m: np.mean([src[(cond, s)][m] for s in SEEDS], axis=0)
        pairs = [(v, "cox_truncated") for v in (*VARIANTS, "offset_only")] + [("bce", v) for v in (*VARIANTS, "offset_only")]
        for plus, minus in pairs:
            for m in metrics:
                est = mean(plus, point, m) - mean(minus, point, m)
                d = mean(plus, draws, m) - mean(minus, draws, m)
                d = d[np.isfinite(d)]
                contrast_rows.append(dict(cohort=cohort, contrast=f"{plus}_minus_{minus}", metric=m, estimate=float(est),
                                          se=float(d.std(ddof=1)), ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                                          ci90_low=float(np.quantile(d, .05)), ci90_high=float(np.quantile(d, .95))))
        m = "original_within_phase"
        for v in (*VARIANTS, "offset_only"):
            est = (mean(v, point, m) - mean("cox_truncated", point, m)) / (mean("bce", point, m) - mean("cox_truncated", point, m))
            r = (mean(v, draws, m) - mean("cox_truncated", draws, m)) / (mean("bce", draws, m) - mean("cox_truncated", draws, m))
            r = r[np.isfinite(r)]
            contrast_rows.append(dict(cohort=cohort, contrast=f"closure_share_{v}", metric=m, estimate=float(est), se=float(r.std(ddof=1)),
                                      ci_low=float(np.quantile(r, .025)), ci_high=float(np.quantile(r, .975)),
                                      ci90_low=float(np.quantile(r, .05)), ci90_high=float(np.quantile(r, .95))))
        pd.DataFrame(arm_rows).to_csv(OUT / "arm_metrics.csv", index=False)
        pd.DataFrame(contrast_rows).to_csv(OUT / "contrasts.csv", index=False)
        print("DONE", cohort, flush=True)
    (OUT / "fits.json").write_text(json.dumps(fits, indent=1))
    (OUT / "constant_columns.json").write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    main()
