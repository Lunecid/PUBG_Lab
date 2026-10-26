#!/usr/bin/env python3
"""Match-offset decomposition of within-phase AUC for the matched objective controls (stored test predictions only).

Within one phase, every (eliminated, surviving) pair either shares a match (same-match pairs, A_W: the within-match-phase AUC)
or comes from two matches (different-match pairs, A_B).  For each (cohort, condition, seed) prediction file of the objective
suite this script reports, on the logit of the stored calibrated probability:

* ``original``: within-phase AUC, its different-match part A_B, and the share of same-match pairs;
* ``centred``: the same after subtracting each model's mean logit per (match, phase), i.e. without any match-level offset;
* ``match_mean``: A_B when every team keeps only its (match, phase) mean logit, i.e. the match-level component alone.

A Cox-type loss is unchanged when a constant is added to all scores of one risk set, so it receives no gradient for a
match-phase offset, whereas cross-entropy does.  If that drives the BCE-over-truncated-Cox within-phase gap, the gap should
vanish for ``centred`` scores and sit in ``match_mean``.  No model is retrained.  Bootstrap draws reuse the objective suite's
seed (``tools/sac2027_evaluate_controls.py``), so the intervals share resampled matches with Table 6.
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

from model_stgcn.ptr import data as D  # noqa: E402
from model_stgcn.ptr import hgb as H  # noqa: E402
from model_stgcn.ptr import metrics as M  # noqa: E402
from model_stgcn.ptr.metrics import MIN_PHASE_ROWS, SortedBinary  # noqa: E402
from model_stgcn.ptr.supplement import RiskSetAUC  # noqa: E402

BASE = REPO / "reports/sac2027_patch_v14"
DUMP = BASE / "dumps/phase_v2"
OUT = BASE / "control_evaluation/objective_match_offset"
SEEDS = (42, 13, 2026)
CONDITIONS = ("weighted_full", "weighted_truncated", "cox_full", "cox_truncated", "bce", "cox_truncated_bce")
PAIRS = (("bce", "cox_truncated"), ("bce", "cox_full"), ("bce", "weighted_truncated"), ("cox_truncated", "cox_full"),
         ("bce", "cox_truncated_bce"), ("cox_truncated_bce", "cox_truncated"))  # last two: COXBCE_PLAN_KO_2026-10-09.md
COHORTS = ("erangel_tpp_squad", "taego_tpp_squad")
N_BOOT = 2000
EPS = 1e-6
SCORES = ("original", "centred", "match_mean")
SCALED = SCORES + ("scaled",)  # + location and spread removed per (match, phase); opt-in


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Decomposition:
    """Within-phase AUC split into same-match and different-match pairs, as functions of per-match bootstrap weights."""

    def __init__(self, frame: pd.DataFrame, names: tuple[str, ...] = SCORES) -> None:
        self.names = names
        self.match_ids, self.match_codes = np.unique(frame["match_id"].to_numpy(), return_inverse=True)
        self.n_matches = len(self.match_ids)
        y = frame["y"].to_numpy().astype(np.float64)
        phase = frame["phase"].to_numpy()
        s = logit(frame["p_hazard"].to_numpy())
        groups = frame.groupby(["match_id", "phase"], sort=True).ngroup().to_numpy()
        mean = pd.Series(s).groupby(groups).transform("mean").to_numpy()
        scores = {"original": s, "centred": s - mean, "match_mean": mean}
        if "scaled" in names:  # (s - match-phase mean) / match-phase SD; a constant set keeps 0
            sd = pd.Series(s).groupby(groups).transform(lambda v: v.std(ddof=0)).to_numpy()
            scores["scaled"] = np.divide(s - mean, sd, out=np.zeros_like(s), where=sd > 0)
        self.parts = []
        for k in np.unique(phase):  # eligible phases as in WithinPhaseAUC: enough rows and both classes, unweighted
            idx = np.flatnonzero(phase == k)
            yk = y[idx]
            if idx.size < MIN_PHASE_ROWS or yk.min() == yk.max():
                continue
            mk = self.match_codes[idx]
            n = np.bincount(mk, minlength=self.n_matches).astype(np.float64)
            pos = np.bincount(mk, weights=yk, minlength=self.n_matches)
            same_pairs = pos * (n - pos)  # same-match pairs of this phase, per match
            per_score = {}
            for name in self.names:
                sk = scores[name][idx]
                rank = pd.Series(sk).groupby(mk).rank(method="average").to_numpy()
                u = np.bincount(mk, weights=rank * yk, minlength=self.n_matches) - pos * (pos + 1.0) / 2.0
                per_score[name] = (SortedBinary(yk, sk), u)
            self.parts.append((int(k), mk, yk, same_pairs, per_score))

    def evaluate(self, match_w: np.ndarray) -> dict[str, np.ndarray]:
        w = np.atleast_2d(np.asarray(match_w, dtype=np.float64))
        w2 = w**2
        out: dict[str, np.ndarray] = {}
        share = []
        for name in self.names:
            within, between = [], []
            for _, mk, yk, same_pairs, per_score in self.parts:
                sb, u = per_score[name]
                rw = w[:, mk]
                pos, neg = (rw * yk).sum(1), (rw * (1.0 - yk)).sum(1)
                auc = np.array([sb.auc(r) for r in rw])
                all_pairs = pos * neg
                same = w2 @ same_pairs
                with np.errstate(divide="ignore", invalid="ignore"):
                    between.append((auc * all_pairs - w2 @ u) / (all_pairs - same))
                within.append(auc)
                if name == "original":
                    share.append(same / all_pairs)
            out[f"{name}_within_phase"] = np.nanmean(np.vstack(within), axis=0)
            out[f"{name}_between_match"] = np.nanmean(np.vstack(between), axis=0)
        out["same_match_pair_share"] = np.nanmean(np.vstack(share), axis=0)
        # Within-match AUC with the phase weights of within-phase AUC (unweighted mean over the same eligible phases) and
        # as an unweighted mean over match-phase risk sets.  A resampled match counts once per copy (weight w, as in
        # RiskSetAUC), whereas the pooled within-match AUC weighs each risk set by its number of pairs.
        per_phase, set_num, set_den = [], 0.0, 0.0
        for _, mk, yk, same_pairs, per_score in self.parts:
            u = per_score["original"][1] if "original" in per_score else None
            if u is None:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                per_phase.append((w @ u) / (w @ same_pairs))
                ok = same_pairs > 0
                a = np.where(ok, u / np.where(ok, same_pairs, 1.0), 0.0)
            set_num = set_num + w @ a
            set_den = set_den + w @ ok.astype(np.float64)
        if per_phase:
            out["within_match_phase_mean"] = np.nanmean(np.vstack(per_phase), axis=0)
            out["within_match_set_mean"] = set_num / set_den
        return out


def load_meta(cohort: str) -> pd.DataFrame:
    rows = H.build_rows(D.load_dump(DUMP / cohort / "test.pt"), "HGB-cut", "v2")
    meta = rows.keys.copy()
    meta["ai"] = rows.X[:, rows.feature_names.index("team_ai_frac")]
    meta["y"] = rows.y
    return meta


def analyze(cohort: str, population: str, meta: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    frames, provenance = {}, []
    for cond in CONDITIONS:
        for seed in SEEDS:
            path = BASE / "objective_controls" / cohort / cond / f"seed{seed}" / "pred_test.csv"
            frame = M.read_predictions(path)
            joined = frame.merge(meta, on=M.KEY_COLUMNS, validate="one_to_one", how="outer", indicator=True,
                                 suffixes=("", "_check"))
            assert bool(joined["_merge"].eq("both").all()) and np.array_equal(joined["y"], joined["y_check"])
            if population == "human":
                frame = joined.loc[joined["ai"] == 0, M.PRED_COLUMNS].reset_index(drop=True)
            frames[(cond, seed)] = frame
            provenance.append(dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    seed = int(hashlib.sha256(f"20261008|objective|{cohort}|{population}".encode()).hexdigest()[:8], 16)
    n_matches = next(iter(frames.values()))["match_id"].nunique()
    counts = M.resample_counts(n_matches, N_BOOT, seed)
    point, draws = {}, {}
    for key, frame in frames.items():
        dec = Decomposition(frame)
        assert dec.n_matches == n_matches
        point[key] = {m: float(v[0]) for m, v in dec.evaluate(np.ones(n_matches)).items()}
        draws[key] = dec.evaluate(counts)
        rs = RiskSetAUC(frame)
        point[key]["within_match"] = rs.evaluate(np.ones(n_matches))
        draws[key]["within_match"] = rs.evaluate(counts)
    metrics = list(point[next(iter(point))])
    arms, contrasts = [], []
    for cond in CONDITIONS:
        keys = [(cond, s) for s in SEEDS]
        arms.append(dict(cohort=cohort, population=population, condition=cond, n_matches=n_matches,
                         **{m: float(np.mean([point[k][m] for k in keys])) for m in metrics}))
    for plus, minus in PAIRS:
        for m in metrics:
            est = np.mean([point[(plus, s)][m] for s in SEEDS]) - np.mean([point[(minus, s)][m] for s in SEEDS])
            d = np.mean([draws[(plus, s)][m] for s in SEEDS], axis=0) - np.mean([draws[(minus, s)][m] for s in SEEDS], axis=0)
            d = d[np.isfinite(d)]
            lo, hi = np.quantile(d, [0.025, 0.975])
            lo90, hi90 = np.quantile(d, [0.05, 0.95])  # equivalence intervals (Sec. 5.2)
            contrasts.append(dict(cohort=cohort, population=population, contrast=f"{plus}_minus_{minus}", metric=m,
                                  estimate=float(est), se=float(d.std(ddof=1)), ci_low=float(lo), ci_high=float(hi),
                                  ci90_low=float(lo90), ci90_high=float(hi90)))
    # Share of the BCE - next-phase Cox within-phase gap closed by the combined objective, on the same resampled matches
    m = "original_within_phase"
    seed_mean = lambda cond, src: np.mean([src[(cond, s)][m] for s in SEEDS], axis=0)
    est = (seed_mean("cox_truncated_bce", point) - seed_mean("cox_truncated", point)) / (
        seed_mean("bce", point) - seed_mean("cox_truncated", point))
    ratio = (seed_mean("cox_truncated_bce", draws) - seed_mean("cox_truncated", draws)) / (
        seed_mean("bce", draws) - seed_mean("cox_truncated", draws))
    ratio = ratio[np.isfinite(ratio)]
    lo, hi = np.quantile(ratio, [0.025, 0.975])
    lo90, hi90 = np.quantile(ratio, [0.05, 0.95])
    contrasts.append(dict(cohort=cohort, population=population, contrast="closure_share_cox_truncated_bce", metric=m,
                          estimate=float(est), se=float(ratio.std(ddof=1)), ci_low=float(lo), ci_high=float(hi),
                          ci90_low=float(lo90), ci90_high=float(hi90)))
    # Seed-matched point estimates (no resampling), to show how much the seed alone moves each contrast
    for plus, minus in PAIRS:
        for metric in metrics:
            for s in SEEDS:
                per_seed.append(dict(cohort=cohort, population=population, contrast=f"{plus}_minus_{minus}", metric=metric,
                                     seed=s, estimate=float(point[(plus, s)][metric] - point[(minus, s)][metric])))
    return arms, contrasts, provenance


per_seed: list[dict] = []


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arm_rows, contrast_rows, provenance = [], [], []
    for cohort in COHORTS:
        meta = load_meta(cohort)
        for population in ("all", "human"):
            print("START", cohort, population, flush=True)
            arms, contrasts, files = analyze(cohort, population, meta)
            arm_rows += arms
            contrast_rows += contrasts
            provenance += files
            pd.DataFrame(arm_rows).to_csv(OUT / "arm_metrics.csv", index=False)
            pd.DataFrame(contrast_rows).to_csv(OUT / "contrasts.csv", index=False)
            pd.DataFrame(per_seed).to_csv(OUT / "per_seed.csv", index=False)
            print("DONE", cohort, population, flush=True)
    (OUT / "complete.json").write_text(json.dumps(dict(cohorts=COHORTS, conditions=CONDITIONS, seeds=SEEDS, n_boot=N_BOOT,
                                                       files=provenance), indent=1))


if __name__ == "__main__":
    main()
