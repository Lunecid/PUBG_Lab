#!/usr/bin/env python3
"""Review follow-ups on the patch-split test set (stored seed-42 predictions; only the logistic baseline is new).

1. Same-target simple baseline: full-feature regularized logistic (``tools/sac2027_logistic_baseline.py``) against the
   reference GBDT, PTR-Former, and phase-adjusted KRAFTON, with paired match bootstrap and DerSimonian-Laird pooling.
2. Retrospective review utility: top-3 capture, the share of next-phase eliminations among the three highest-risk teams of
   each (match, phase) risk set (boundaries without an elimination excluded; ties broken at random in expectation), and
   calibration slope (y on logit p) and calibration-in-the-large (intercept with logit p as offset).  A remaining-members rule
   (fewer members = higher risk) and phase-only risk serve as simple references.
3. Mode sensitivity: AUC and within-match AUC on normal ("official") and ranked ("competitive") matches separately.
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
import statsmodels.api as sm  # noqa: E402

from model_stgcn.ptr import data as D  # noqa: E402
from model_stgcn.ptr import hgb as H  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402
from model_stgcn.ptr.supplement import RiskSetAUC  # noqa: E402

BASE = REPO / "reports/sac2027_patch_v14"
DUMP = BASE / "dumps/phase_v2"
OUT = BASE / "control_evaluation/extra_v16"
ARMS = {"GBDT": "HGB-cut", "PTR": "M11R", "KRAFTON": "KRAFTON-tf", "KRAFTON+phase": "KRAFTON-tf-h0",
        "Sagong": "SAGONG-xgb", "Union": "HGB-union", "Phase-only": "PHASE-only"}
METRICS = ("auc", "auc_within_phase", "nll", "brier")
CONTRASTS = (("Logistic", "GBDT"), ("Logistic", "PTR"), ("Logistic", "KRAFTON+phase"))
TOPK_CONTRASTS = (("GBDT", "Members"), ("PTR", "GBDT"), ("Union", "GBDT"), ("Logistic", "GBDT"), ("GBDT", "KRAFTON+phase"))
MODE_ARMS = ("GBDT", "PTR", "KRAFTON+phase")
MODE_CONTRASTS = (("PTR", "KRAFTON+phase"), ("GBDT", "KRAFTON+phase"), ("PTR", "GBDT"))
K = 3
N_BOOT = 2000
MIN_MODE_MATCHES = 20


def seed_of(*parts: str) -> int:
    return int(hashlib.sha256("|".join(("20261008", "extra_v16") + parts).encode()).hexdigest()[:8], 16)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(frame: pd.DataFrame) -> dict:
    y, z = frame["y"].to_numpy(float), logit(frame["p_hazard"].to_numpy())
    slope = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit().params[0]
    return dict(calibration_slope=float(slope), calibration_in_the_large=float(citl))


class TopK:
    """Expected top-k capture per match (random tie-breaking) as a function of per-match bootstrap weights."""

    def __init__(self, frame: pd.DataFrame, risk: np.ndarray, k: int = K) -> None:
        match_ids, match_codes = np.unique(frame["match_id"].to_numpy(), return_inverse=True)
        y = frame["y"].to_numpy(float)
        groups = frame.groupby(["match_id", "phase"], sort=True).ngroup().to_numpy()
        captured = np.zeros(len(match_ids))
        events = np.zeros(len(match_ids))
        order = np.argsort(groups, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(groups))]
        for g in range(len(bounds) - 1):
            idx = order[bounds[g]:bounds[g + 1]]
            yg, sg = y[idx], risk[idx]
            if yg.sum() == 0:
                continue
            m = match_codes[idx[0]]
            events[m] += yg.sum()
            if len(idx) <= k:
                captured[m] += yg.sum()
                continue
            s_sorted = np.sort(sg)[::-1]
            cut = s_sorted[k - 1]
            above = sg > cut
            tie = sg == cut
            slots = k - above.sum()
            captured[m] += yg[above].sum() + slots / tie.sum() * yg[tie].sum()
        self.captured, self.events = captured, events

    def evaluate(self, w: np.ndarray) -> np.ndarray | float:
        w = np.asarray(w, dtype=np.float64)
        num, den = w @ self.captured, w @ self.events
        out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        return float(out) if out.ndim == 0 else out


def load(cohort: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rows = H.build_rows(D.load_dump(DUMP / cohort / "test.pt"), "HGB-cut", "v2")
    meta = rows.keys.copy()
    meta["y"] = rows.y
    meta["members"] = rows.X[:, rows.feature_names.index("alive_players_end_norm")]
    meta["ai"] = rows.X[:, rows.feature_names.index("team_ai_frac")]
    paths = {name: BASE / "checkpoints/ptr_v2" / cohort / arm / "seed42" / "pred_test.csv" for name, arm in ARMS.items()}
    paths["Logistic"] = BASE / "baseline_logistic" / cohort / "pred_test.csv"
    frames = {}
    for name, path in paths.items():
        f = M.read_predictions(path)
        j = f.merge(meta, on=M.KEY_COLUMNS, validate="one_to_one", how="outer", indicator=True, suffixes=("", "_m"))
        assert bool(j["_merge"].eq("both").all()) and np.array_equal(j["y"], j["y_m"]), (cohort, name)
        frames[name] = j.drop(columns=["_merge", "y_m"])
    return frames, meta


def contrast_rows(cohort, scope, point, draws, pairs, metrics):
    out = []
    for plus, minus in pairs:
        for m in metrics:
            d = np.asarray(draws[plus][m]) - np.asarray(draws[minus][m])
            d = d[np.isfinite(d)]
            out.append(dict(cohort=cohort, scope=scope, contrast=f"{plus}_minus_{minus}", metric=m,
                            estimate=float(point[plus][m] - point[minus][m]), se=float(d.std(ddof=1)),
                            ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975))))
    return out


def analyze(cohort: str) -> tuple[list, list, list]:
    frames, _ = load(cohort)
    names = list(frames)
    preds = {n: frames[n][M.PRED_COLUMNS] for n in names}
    n_matches = preds["GBDT"]["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed_of(cohort, "all"))
    _, point, samples = M.bootstrap_arms(preds, metrics=METRICS, n_boot=N_BOOT, seed=seed_of(cohort, "all"))
    arm_rows, contrasts, mode_rows = [], [], []
    for n in names:
        rs = RiskSetAUC(preds[n])
        point[n]["auc_within_match"] = rs.evaluate(np.ones(n_matches))
        samples[n]["auc_within_match"] = rs.evaluate(counts)
        tk = TopK(frames[n], frames[n]["p_hazard"].to_numpy())
        point[n]["top3_capture"] = tk.evaluate(np.ones(n_matches))
        samples[n]["top3_capture"] = tk.evaluate(counts)
        point[n].update(calibration(frames[n]))
    members = TopK(frames["GBDT"], -frames["GBDT"]["members"].to_numpy())
    point["Members"] = {"top3_capture": members.evaluate(np.ones(n_matches))}
    samples["Members"] = {"top3_capture": members.evaluate(counts)}
    for n in point:
        arm_rows.append(dict(cohort=cohort, arm=n, n_matches=n_matches, **point[n]))
    contrasts += contrast_rows(cohort, "all", point, samples, CONTRASTS, (*METRICS, "auc_within_match"))
    contrasts += contrast_rows(cohort, "all", point, samples, TOPK_CONTRASTS, ("top3_capture",))
    for mode in ("official", "competitive"):
        sub = {n: preds[n].loc[preds[n]["match_type"] == mode].reset_index(drop=True) for n in MODE_ARMS}
        nm = sub["GBDT"]["match_id"].nunique()
        if nm < MIN_MODE_MATCHES:
            continue
        c = M.resample_counts(nm, N_BOOT, seed_of(cohort, mode))
        _, p_mode, s_mode = M.bootstrap_arms(sub, metrics=("auc",), n_boot=N_BOOT, seed=seed_of(cohort, mode))
        for n in MODE_ARMS:
            rs = RiskSetAUC(sub[n])
            p_mode[n]["auc_within_match"] = rs.evaluate(np.ones(nm))
            s_mode[n]["auc_within_match"] = rs.evaluate(c)
            mode_rows.append(dict(cohort=cohort, mode=mode, arm=n, n_matches=nm, n_rows=len(sub[n]), **p_mode[n]))
        contrasts += contrast_rows(cohort, mode, p_mode, s_mode, MODE_CONTRASTS, ("auc", "auc_within_match"))
    return arm_rows, contrasts, mode_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(BASE / "cohort_counts.csv")
    main_cohorts = roles.loc[roles["analysis_role"] == "main", "cohort"].tolist()
    arms, contrasts, modes = [], [], []
    for cohort in main_cohorts:
        print("START", cohort, flush=True)
        a, c, m = analyze(cohort)
        arms += a
        contrasts += c
        modes += m
        pd.DataFrame(arms).to_csv(OUT / "arm_metrics.csv", index=False)
        pd.DataFrame(contrasts).to_csv(OUT / "contrasts_per_cohort.csv", index=False)
        pd.DataFrame(modes).to_csv(OUT / "mode_arm_metrics.csv", index=False)
    per = pd.DataFrame(contrasts)
    pooled = []
    for (scope, contrast, metric), g in per.groupby(["scope", "contrast", "metric"]):
        dl = M.dersimonian_laird(g["estimate"].to_numpy(), g["se"].to_numpy())
        pooled.append(dict(scope=scope, contrast=contrast, metric=metric,
                           same_sign=int((np.sign(g["estimate"]) == np.sign(dl["theta"])).sum()), **dl))
    pd.DataFrame(pooled).to_csv(OUT / "pooled.csv", index=False)
    means = pd.DataFrame(arms).drop(columns=["cohort", "n_matches"]).groupby("arm").mean(numeric_only=True)
    means.to_csv(OUT / "arm_means.csv")
    (OUT / "complete.json").write_text(json.dumps(dict(cohorts=main_cohorts, arms=ARMS, k=K, n_boot=N_BOOT,
                                                       min_mode_matches=MIN_MODE_MATCHES), indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
