"""Supplementary analyses E1-E3 of the BigComp 2027 results (post hoc, exploratory: no Holm family, no pre-registered claim).

Everything works from the prediction files of the run matrix (``<runs>/<cohort>/<arm>/seed42/pred_test.csv``, labels v2) and from the
per-cohort contrast table of the aggregator (``contrasts_per_cohort.csv``); nothing is retrained here (the E1 arms are trained by
``run_matrix --norank``).  The bootstrap is the aggregator's: 2,000 resamples of whole TEST MATCHES with the seed
``aggregate.task_seed(cohort, 'all', 0)``, one resample shared by all arms of a cohort, so a contrast computed here for arms the
aggregator also knows reproduces its per-cohort numbers.  Pooling over cohorts is DerSimonian-Laird on the per-cohort estimates with
their bootstrap SEs (``metrics.dersimonian_laird``).

E1 -- rank-loss control.  The PTR arms train with BCE + 0.1 * pairwise rank loss, the KRAFTON-on-our-features arms (K2 / K3a / K3) with BCE
only, so the architecture step K3 -> M01 also adds a loss.  ``M01-norank`` / ``M11R-norank`` (``arms.NORANK_ARMS``, rank weight 0, everything
else identical) remove that confound.  These contrasts are decomposition steps of the aggregator now (``aggregate.DECOMP_STEPS``: (e) ``m01norank_vs_k3``,
(f) ``m11rnorank_vs_m01norank`` and the separate group ``m01_vs_m01norank`` / ``m11r_vs_m11rnorank``, the rank auxiliary loss); E1 recomputes them
(to - from) with the same bootstrap resamples and adds what the aggregator does not have: the equal-footing ``spearman_from_hazard`` and the E2 metric,
the arm table with the rank-score source, and the run facts of the four PTR arms.  The coefficients are the aggregator's (``aggregate.FORMULAS``): one
definition.  The two original steps ``M01 - K3`` and ``M11R - M01`` (with the rank loss) are recomputed next to them.  Metrics: pooled AUC,
``auc_within_phase`` (unweighted mean over cut phases of the per-phase AUC; the aggregator's), Spearman (the arm's own ranking: the rank head of a PTR arm,
``-hazard logit`` for an arm without a trained rank head), the same Spearman read from ``p_hazard`` for every arm (``spearman_from_hazard``: an
equal-footing comparison of the hazard heads), and the E2 metric below.  ``supplement_summary.json['aggregator']`` records that every E1 contrast
that the aggregator also has reproduces its per-cohort estimates and bootstrap SEs (a mismatch means other runs, resamples or seed and raises).

E2 -- within match-phase AUC (``auc_within_match_phase``), a supplementary metric: for every (match, cut phase) risk set that holds both
labels, the AUC of ``p_hazard`` for ``y`` (ties 1/2); pooled over the risk sets weighted by their number of positive-negative pairs
(``P_s * N_s``), i.e. a stratified concordance ``sum_s U_s / sum_s P_s N_s`` with ``U_s`` the Mann-Whitney count of risk set ``s``.  It ranks
teams only against teams of the same match AND the same cut phase, so neither the phase baseline nor the difficulty of a match enters.  Risk
sets with a single label carry no pair and are excluded; their share of the risk sets and of the rows is reported per cohort.  Under a match
bootstrap a risk set is duplicated with its match and its AUC does not change, so the per-resample statistic is exactly
``counts @ U / counts @ D`` over the matches (``U_m``, ``D_m`` = sums over the match's risk sets); it equals recomputing the per-risk-set
AUCs on an explicitly duplicated frame (tested).  DL-pooled for every row of ``tab_decomp.tex`` (the single-factor steps (a)-(g) and the rank auxiliary loss group)
and for the RQ2 contrasts of ``tab_rq2_external.tex``.

E3 -- pooling sensitivity.  For the same contrasts and for P1-P5 (``contrasts_pooled.csv``) the DL pooling is repeated with REML ``tau^2``
and the Hartung-Knapp-Sidik-Jonkman (HKSJ) interval, ``theta +- t_{k-1} * sqrt(q / sum w)`` with ``q = sum w (y - theta)^2 / (k - 1)``
(t with k - 1 df for the interval and the p-value).  The ``hksjt_*`` columns are the variant with ``max(1, q)`` (never narrower than the Wald
interval); ``tau^2`` is the global maximiser of the restricted likelihood (:func:`reml_tau2`).  Compared with DL: the sign of the pooled estimate, whether the 95% interval excludes 0, and the verdict category (gain /
equivalent / inconclusive, ``aggregate.verdict``: Holm over P1-P5 for the primary tests, the unadjusted one-sided p otherwise, SESOI as in
the aggregator; a metric without SESOI has no verdict).

Outputs (``--out-dir``): ``e1_arm_metrics_per_cohort.csv``, ``e1_contrasts_per_cohort.csv``, ``e1_contrasts_pooled_dl.csv``, ``e1_run_info.csv``;
``e2_arm_wmp_per_cohort.csv`` (point value and bootstrap interval per cohort and arm), ``e2_contrasts_per_cohort.csv``, ``e2_contrasts_pooled_dl.csv``,
``e2_excluded_risk_sets.csv`` / ``e2_excluded_by_phase.csv`` (the single-label risk sets and rows); ``e3_pooling_sensitivity.csv``;
``supplement_summary.json``.

Reads the v2 aggregator outputs (it runs AFTER ``model_stgcn.ptr.aggregate`` in the orchestration): ``--results-dir/contrasts_per_cohort.csv`` and
``summary.json``.  ``--n-boot`` and ``--seed`` default to the aggregator's (so that the resamples are the same), and ``summary.json['runs_dir']`` must be
``--runs-dir``.  The defaults are the v2 roots of :mod:`model_stgcn.ptr.paths`; ``--dump-dir`` is compared with ``--runs-dir`` and ``--results-dir`` and a
v1 root next to a v2 dump is refused (without it the two directories are compared with each other).  Arms that have no prediction file in a cohort are skipped (named in
``supplement_summary.json['missing_arms']``), the contrasts that need them are not computed.

    python -m model_stgcn.ptr.supplement --runs-dir model_stgcn/checkpoints/bigcomp2027_ptr_v2 \\
        --results-dir reports/bigcomp2027_results_v2 --out-dir reports/bigcomp2027_results_v2/supplement
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[5]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import concurrent.futures as cf
import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

import pubg_lab.survival.ptr.evaluation.aggregate as A
import pubg_lab.survival.ptr.evaluation.metrics as M
import pubg_lab.survival.ptr.data.paths as P
from pubg_lab.survival.ptr.common.util import git_info, write_json

logger = logging.getLogger("pubg_lab.survival.ptr.evaluation.supplement")

SEED = 0  # base seed of the bootstrap: the aggregator's default (the per-cohort offset is ``aggregate.task_seed``)
N_BOOT = 2000
MAIN_SEED = A.MAIN_SEED
WMP = "auc_within_match_phase"  # the E2 metric
SPEARMAN_HAZ = "spearman_from_hazard"  # Spearman of p_hazard (rank_col='p_hazard'), the equal-footing version for the E1 arms
STANDARD_METRICS: tuple[str, ...] = (
    A.METRICS
)  # auc, ap, nll, brier, spearman, top5_auc, cindex, auc_within_phase
E_METRICS: tuple[str, ...] = (*STANDARD_METRICS, SPEARMAN_HAZ, WMP)
KEY_METRICS: tuple[str, ...] = ("auc", "auc_within_phase", "spearman", WMP)  # what E1 / E3 report
DIRECTION = {**A.DIRECTION, SPEARMAN_HAZ: 1, WMP: 1}
SESOI = dict(A.SESOI)  # the within-phase metrics have none: no verdict (as in the aggregator)
ILLUSTRATIVE_SESOI = A.SESOI[
    "auc"
]  # E3 only: the AUC SESOI applied to the within-phase AUCs, flagged as illustrative

# ---- E1
K3 = "KRAFTON-tf-ourfeat-hazard"
E1_ARMS: tuple[str, ...] = (K3, "M01-norank", "M11R-norank", "M01", "M11R")
# E1 name -> (label, aggregator formula): every contrast is "to - from" and its coefficients ARE the aggregator's (one definition)
E1_SPEC: dict[str, tuple[str, str]] = {
    "k3_to_m01": ("M01 - K3 (original step)", "m01_vs_k3"),
    "m01_to_m11r": ("M11R - M01 (original step)", "m11r_vs_m01"),
    "k3_to_m01norank": ("(e) M01-norank - K3", "m01norank_vs_k3"),
    "m01norank_to_m11rnorank": ("(f) M11R-norank - M01-norank", "m11rnorank_vs_m01norank"),
    "rankloss_m01": ("rank loss, M01: M01 - M01-norank", "m01_vs_m01norank"),
    "rankloss_m11r": ("rank loss, M11R: M11R - M11R-norank", "m11r_vs_m11rnorank"),
}  # fmt: skip
E1_AGG_NAME: dict[str, str] = {k: v[1] for k, v in E1_SPEC.items()}
E1_FORMULAS: dict[str, tuple[str, dict[str, float]]] = {
    k: (label, {a: float(c) for a, c in A.FORMULAS[agg].coefs.items()})
    for k, (label, agg) in E1_SPEC.items()
}  # name -> (label, {arm: coefficient})
E1_ORIGINAL: tuple[str, ...] = (
    "k3_to_m01",
    "m01_to_m11r",
)  # the two with the rank loss: not decomposition steps any more
AGG_TOLERANCE = (
    1e-8  # the E1 contrasts must reproduce the aggregator's per-cohort estimates and SEs to this
)

# ---- E2: the arms of the per-cohort table and the contrasts that are pooled
E2_ARMS: tuple[str, ...] = (
    "HGB-cut", "M11R", "M01", "KRAFTON-tf-ourfeat", "KRAFTON-tf-ourfeat-h0", "KRAFTON-tf-ourfeat-bce", K3, "KRAFTON-tf", "KRAFTON-tf-h0", "SAGONG-hgb",
    "SAGONG-xgb", "EHM-lite", "M01-norank", "M11R-norank",
)  # fmt: skip
E2_EXTRA_ARMS: tuple[str, ...] = (
    "HGB-kraftonfeat",
    "HGB-union",
)  # needed by the decomposition steps (input vs KRAFTON, extended representation)
E2_DECOMP: tuple[str, ...] = tuple(
    name for _, _, _, name in A.DECOMP_STEPS
)  # every row of tab_decomp.tex (steps + rank auxiliary loss), in table order
E2_RQ2: tuple[str, ...] = tuple(
    name for _, _, name in A.RQ2_ROWS
)  # the bundled external contrasts of tab_rq2_external.tex
E2_FORMULAS: tuple[str, ...] = tuple(dict.fromkeys((*E2_DECOMP, *E2_RQ2)))


# --------------------------------------------------------------------------- E2: the stratified within-risk-set AUC
class RiskSetAUC:
    """Within match-phase AUC of one prediction frame as a function of per-match weights (bootstrap multiplicities).

    ``score_col`` (default ``p_hazard``, the risk the hazard AUC reads) is compared with ``y`` inside every (match, cut phase) risk set that
    has both labels: ``U_s`` = positive-over-negative pairs plus half the ties (Mann-Whitney, average ranks), ``D_s = P_s * N_s``.  A match
    weighs all its risk sets alike, so ``evaluate(w) = sum_m w_m U_m / sum_m w_m D_m``.  Risk sets without both labels are excluded and counted
    (``n_sets``, ``n_sets_used``, ``n_rows``, ``n_rows_used``; ``by_phase()``)."""

    def __init__(self, df: pd.DataFrame, score_col: str = "p_hazard") -> None:
        df = M.read_predictions(df)
        y = df["y"].to_numpy()
        score = df[score_col].to_numpy()
        if not (np.isfinite(y).all() and np.isfinite(score).all()):
            raise ValueError("the prediction frame has non-finite labels or scores")
        self.match_ids, match_codes = np.unique(
            df["match_id"].to_numpy(), return_inverse=True
        )  # the aggregator's order of the matches
        self.n_matches = len(self.match_ids)
        codes = M._group_codes(df)  # (match_id, phase) risk sets
        n_sets = int(codes.max()) + 1 if len(df) else 0
        n = np.bincount(codes, minlength=n_sets).astype(np.float64)
        pos = np.bincount(codes, weights=y, minlength=n_sets)
        neg = n - pos
        rank = pd.Series(score).groupby(codes).rank(method="average").to_numpy()
        u = np.bincount(codes, weights=rank * y, minlength=n_sets) - pos * (pos + 1.0) / 2.0
        first = np.zeros(n_sets, dtype=np.int64)
        first[codes[::-1]] = np.arange(len(df))[::-1]  # first row of every risk set
        self._set_match = match_codes[first]
        self._set_phase = df["phase"].to_numpy()[first]
        ok = (pos > 0) & (neg > 0)
        self._ok, self._n, self._pos, self._neg, self._u = ok, n, pos, neg, u
        self.n_rows, self.n_sets = int(len(df)), int(n_sets)
        self.n_sets_used, self.n_rows_used = int(ok.sum()), int(n[ok].sum())
        self.pairs = float((pos * neg)[ok].sum())
        self._num = np.bincount(self._set_match[ok], weights=u[ok], minlength=self.n_matches)
        self._den = np.bincount(
            self._set_match[ok], weights=(pos * neg)[ok], minlength=self.n_matches
        )

    def evaluate(self, match_w: np.ndarray) -> np.ndarray | float:
        """The metric under match weights ``[n_matches]`` (a float) or a resample matrix ``[n_boot, n_matches]`` (an array)."""
        w = np.asarray(match_w, dtype=np.float64)
        num, den = w @ self._num, w @ self._den
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return float(out) if out.ndim == 0 else out

    def set_auc(self) -> pd.DataFrame:
        """One row per risk set that has both labels: ``phase``, ``n``, ``pos``, ``auc`` (for checks and diagnostics)."""
        ok = self._ok
        return pd.DataFrame({"phase": self._set_phase[ok], "n": self._n[ok].astype(int), "pos": self._pos[ok].astype(int),
                             "auc": self._u[ok] / (self._pos[ok] * self._neg[ok])})  # fmt: skip

    def exclusions(self) -> dict:
        """Counts of the risk sets and rows left out because a risk set has one label only (no positive / no negative)."""
        no_pos, no_neg = self._pos == 0, self._neg == 0
        return {
            "n_risk_sets": self.n_sets, "n_risk_sets_used": self.n_sets_used,
            "n_risk_sets_no_positive": int(no_pos.sum()), "n_risk_sets_no_negative": int(no_neg.sum()),
            "share_risk_sets_excluded": 1.0 - self.n_sets_used / self.n_sets if self.n_sets else float("nan"),
            "n_rows": self.n_rows, "n_rows_used": self.n_rows_used,
            "share_rows_excluded": 1.0 - self.n_rows_used / self.n_rows if self.n_rows else float("nan"),
            "n_pairs": self.pairs,
        }  # fmt: skip

    def by_phase(self) -> pd.DataFrame:
        """Risk sets / rows and the excluded shares per cut phase."""
        rows = []
        for k in np.unique(self._set_phase):
            sel = self._set_phase == k
            used = sel & self._ok
            rows.append({"phase": int(k), "n_risk_sets": int(sel.sum()), "n_risk_sets_used": int(used.sum()), "n_rows": int(self._n[sel].sum()),
                         "n_rows_used": int(self._n[used].sum()), "n_pairs": float((self._pos * self._neg)[used].sum())})  # fmt: skip
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- per-cohort computation
def pred_path(
    runs_dir: str | Path, cohort: str, arm: str, split: str = "test", seed: int = MAIN_SEED
) -> Path:
    return Path(runs_dir) / cohort / arm / f"seed{seed}" / f"pred_{split}.csv"


def load_frames(
    runs_dir: str | Path, cohort: str, arms: Sequence[str], *, missing: str = "raise"
) -> dict[str, pd.DataFrame]:
    """Test prediction frames of ``arms`` (aligned: identical rows and labels, checked).  A missing file raises ``FileNotFoundError``
    (``missing='raise'``) or leaves the arm out (``missing='skip'``; the caller names what is absent)."""
    frames: dict[str, pd.DataFrame] = {}
    for arm in arms:
        path = pred_path(runs_dir, cohort, arm)
        if not path.is_file():
            if missing == "skip":
                continue
            raise FileNotFoundError(f"{path} not found ({cohort} / {arm})")
        frames[arm] = M.read_predictions(path)
    if not frames:
        raise FileNotFoundError(
            f"none of {list(arms)} has a prediction file for {cohort} in {runs_dir}"
        )
    present = [a for a in arms if a in frames]
    first = frames[present[0]]
    for arm in present[1:]:
        M.align_predictions(first, frames[arm])
    return frames


def cohort_seed(cohort: str, seed: int = SEED) -> int:
    return A.task_seed(cohort, A.STRATUM_ALL, seed)


def _summ(point: float, draws: np.ndarray) -> dict:
    return M.summarise_draws(point, draws, 0.95)


def e_cohort(task: dict) -> dict:
    """One cohort: every arm's point metrics, the paired bootstrap of the contrasts ``task['formulas']`` for the standard metrics (when
    ``task['standard']``), :data:`SPEARMAN_HAZ` and the E2 metric :data:`WMP` (all on the same resamples), and the risk-set exclusions."""
    cohort, formulas = task["cohort"], task["formulas"]
    frames = load_frames(
        task["runs_dir"], cohort, list(task["arms"]), missing=task.get("missing", "raise")
    )
    arms = [
        a for a in task["arms"] if a in frames
    ]  # the arms that exist (all of them unless missing='skip')
    absent = [a for a in task["arms"] if a not in frames]
    ref = frames[arms[0]]
    n_matches = int(ref["match_id"].nunique())
    seed = cohort_seed(cohort, task["seed"])
    counts = M.resample_counts(
        n_matches, task["n_boot"], seed
    )  # the very resamples of ``bootstrap_arms`` for this seed
    ones = np.ones(n_matches)
    point: dict[str, dict[str, float]] = {a: {} for a in arms}
    draws: dict[str, dict[str, np.ndarray]] = {a: {} for a in arms}
    if task["standard"] and len(arms) >= 2:
        _, p_std, s_std = M.bootstrap_arms(
            frames, rank_col=A.RANK_COL, metrics=STANDARD_METRICS, n_boot=task["n_boot"], seed=seed
        )
        _, p_hz, s_hz = M.bootstrap_arms(
            frames, rank_col="p_hazard", metrics=("spearman",), n_boot=task["n_boot"], seed=seed
        )
        for a in arms:
            point[a].update(p_std[a])
            draws[a].update(s_std[a])
            point[a][SPEARMAN_HAZ], draws[a][SPEARMAN_HAZ] = (
                p_hz[a]["spearman"],
                s_hz[a]["spearman"],
            )
    else:
        for a in arms:
            suite = M.MetricSuite(frames[a], A.RANK_COL)
            point[a].update(suite.evaluate(ones, ("auc", "auc_within_phase", "spearman")))
    engines = {a: RiskSetAUC(frames[a]) for a in arms}
    assert all(e.n_matches == n_matches for e in engines.values())
    for a in arms:
        point[a][WMP] = engines[a].evaluate(ones)
        draws[a][WMP] = engines[a].evaluate(counts)
    arm_rows = []
    for a in arms:
        row = {"cohort": cohort, "arm": a, "n_rows": int(len(ref)), "n_matches": n_matches}
        for m, v in point[a].items():
            row[m] = float(v)
        s = _summ(point[a][WMP], draws[a][WMP])
        row.update(
            {f"{WMP}_se": s["se"], f"{WMP}_ci_low": s["ci_low"], f"{WMP}_ci_high": s["ci_high"]}
        )
        arm_rows.append(row)
    metrics = [m for m in E_METRICS if all(m in draws[a] for a in arms)]
    contrast_rows = []
    for name, coefs in formulas.items():
        if not all(a in point for a in coefs):
            continue
        for m in metrics:
            est = sum(c * point[a][m] for a, c in coefs.items())
            dr = sum(c * draws[a][m] for a, c in coefs.items())
            s = _summ(est, dr)
            contrast_rows.append({"cohort": cohort, "contrast": name, "metric": m, "stratum": A.STRATUM_ALL, "estimate": s["estimate"], "se": s["se"],
                                  "ci_low": s["ci_low"], "ci_high": s["ci_high"], "n_valid": s["n_valid"], "n_rows": int(len(ref)),
                                  "n_matches": n_matches})  # fmt: skip
    ex = engines[arms[0]].exclusions()
    for a in arms[1:]:  # the exclusions depend on the labels only: the same for every arm
        e2 = engines[a].exclusions()
        assert {k: e2[k] for k in ex if k != "n_pairs"} == {
            k: ex[k] for k in ex if k != "n_pairs"
        }, f"{cohort}: exclusions differ between arms"
    phase = engines[arms[0]].by_phase()
    phase.insert(0, "cohort", cohort)
    return {"cohort": cohort, "arm_rows": arm_rows, "contrast_rows": contrast_rows, "exclusions": {"cohort": cohort, **ex}, "by_phase": phase,
            "n_matches": n_matches, "missing_arms": absent}  # fmt: skip


def _map(fn, tasks: Sequence[dict], workers: int) -> list:
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(fn, tasks))


# --------------------------------------------------------------------------- pooling: DerSimonian-Laird and REML + HKSJ
def _clean(y, se) -> tuple[np.ndarray, np.ndarray]:
    y, se = np.asarray(y, dtype=np.float64), np.asarray(se, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)  # the filter of ``metrics.dersimonian_laird``
    return y[ok], se[ok]


def reml_tau2(y, se) -> float:
    """REML estimate of the between-study variance: the GLOBAL maximiser of the restricted log-likelihood over ``tau2 >= 0``.

    The candidates are the boundary 0 and every root of the score ``sum w - sum w^2 / sum w - sum w^2 (y - mu)^2`` (``w = 1 / (se^2 + tau2)``) at
    which the likelihood turns from rising to falling (found on a log grid, refined by Brent); with few studies of unequal precision the
    restricted likelihood can have a boundary maximum AND an interior one, and the higher of them is returned (a single root search from one
    side could report the boundary)."""
    y, se = _clean(y, se)
    v = se**2
    if y.size < 2:
        return 0.0

    def score(t2: float) -> float:  # derivative of -2 * restricted log-likelihood
        w = 1.0 / (v + t2)
        sw = w.sum()
        mu = (w * y).sum() / sw
        return float(sw - (w**2).sum() / sw - (w**2 * (y - mu) ** 2).sum())

    scale = max(float(np.var(y, ddof=1)), float(v.mean()), np.finfo(float).tiny)
    grid = np.concatenate([[0.0], np.geomspace(1e-9 * scale, 1e4 * scale, 600)])
    values = np.array([score(t) for t in grid])
    candidates = [0.0]
    for i in range(len(grid) - 1):
        if (
            values[i] < 0.0 <= values[i + 1]
        ):  # -2 loglik falls then rises: a local maximum of the likelihood in between
            candidates.append(
                float(
                    optimize.brentq(
                        score, grid[i], grid[i + 1], xtol=1e-15 * scale, rtol=1e-13, maxiter=500
                    )
                )
            )
    if values[-1] < 0.0:
        raise RuntimeError("REML: the likelihood is still rising at the end of the search range")
    ll = [reml_loglik(y, se, t) for t in candidates]
    return float(candidates[int(np.argmax(ll))])


def reml_loglik(y, se, tau2: float) -> float:
    """Restricted log-likelihood (up to a constant) at ``tau2`` (a check for :func:`reml_tau2`)."""
    y, se = _clean(y, se)
    w = 1.0 / (se**2 + tau2)
    mu = (w * y).sum() / w.sum()
    return float(-0.5 * (np.log(se**2 + tau2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum()))


def _verdict(
    theta: float,
    ci90: tuple[float, float],
    p: float,
    metric: str,
    direction: int,
    sesoi: float | None = None,
) -> str:
    return A.verdict(theta, ci90, p, SESOI.get(metric) if sesoi is None else sesoi, direction)


def _degenerate_pool(y) -> dict:
    """No cohort has a positive standard error: both arms rank every row identically by construction (a per-phase monotone
    baseline such as H0 leaves the within-phase and within match-phase rankings unchanged), so the contrast is its common
    value with no sampling variance.  Reported as that value with a zero-width interval and no p-value."""
    y = np.asarray(y, dtype=np.float64)
    fin = y[np.isfinite(y)]
    theta, nan = (float(fin.mean()) if fin.size else float("nan")), float("nan")
    return {
        "k": 0, "theta": theta, "se": 0.0, "ci95_low": theta, "ci95_high": theta, "ci90_low": theta, "ci90_high": theta,
        "p_two_sided": nan, "p_one_sided": nan, "tau2": 0.0, "Q": nan, "I2": nan, "same_sign": 0,
    }  # fmt: skip


def pool_dl(y, se, direction: int = 1) -> dict:
    """DerSimonian-Laird pooling with the aggregator's conventions (one-sided p for an improvement in ``direction``)."""
    if _clean(y, se)[0].size == 0:
        return _degenerate_pool(y)
    dl = M.dersimonian_laird(*_clean(y, se))
    return {
        "k": dl["k"], "theta": dl["theta"], "se": dl["se"], "ci95_low": dl["ci95"][0], "ci95_high": dl["ci95"][1], "ci90_low": dl["ci90"][0],
        "ci90_high": dl["ci90"][1], "p_two_sided": dl["p"], "p_one_sided": float(stats.norm.sf(direction * dl["z"])), "tau2": dl["tau2"],
        "Q": dl["Q"], "I2": dl["I2"], "same_sign": int(np.sum(np.sign(_clean(y, se)[0]) == np.sign(dl["theta"]))),
    }  # fmt: skip


def pool_hksj(y, se, direction: int = 1, *, truncate: bool = False) -> dict:
    """REML random-effects mean with the Hartung-Knapp-Sidik-Jonkman interval: ``se_hk = sqrt(q / sum w)``, ``q = sum w (y - theta)^2 / (k - 1)``
    (``max(1, q)`` with ``truncate``), t with ``k - 1`` df.  ``reml_wald_*`` is the same REML estimate with the normal interval."""
    y, se = _clean(y, se)
    k = int(y.size)
    tau2 = reml_tau2(y, se)
    w = 1.0 / (se**2 + tau2)
    theta = float((w * y).sum() / w.sum())
    q = float((w * (y - theta) ** 2).sum() / (k - 1)) if k > 1 else float("nan")
    scale = max(1.0, q) if truncate else q
    se_hk = float(np.sqrt(scale / w.sum()))
    df = k - 1
    t = theta / se_hk
    t95, t90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
    se_wald = float(np.sqrt(1.0 / w.sum()))
    z95 = stats.norm.ppf(0.975)
    return {
        "k": k, "theta": theta, "se": se_hk, "df": df, "tau2": tau2, "q_hk": q, "t": t, "ci95_low": theta - t95 * se_hk, "ci95_high": theta + t95 * se_hk,
        "ci90_low": theta - t90 * se_hk, "ci90_high": theta + t90 * se_hk, "p_two_sided": float(2 * stats.t.sf(abs(t), df)),
        "p_one_sided": float(stats.t.sf(direction * t, df)), "reml_wald_ci95_low": theta - z95 * se_wald, "reml_wald_ci95_high": theta + z95 * se_wald,
        "same_sign": int(np.sum(np.sign(y) == np.sign(theta))),
    }  # fmt: skip


# --------------------------------------------------------------------------- pooled tables
POOLED_E_COLUMNS = [
    "set", "contrast", "label", "metric", "k", "theta", "se", "ci95_low", "ci95_high", "same_sign", "tau2", "I2", "Q", "p_one_sided", "p_two_sided",
]  # fmt: skip


def pool_contrast_table(
    per_cohort: pd.DataFrame, labels: Mapping[str, str] | None = None, set_name: str = ""
) -> pd.DataFrame:
    """DL pooling of a per-cohort contrast table (aggregator columns) per (contrast, metric); rows keep the order of first appearance."""
    rows = []
    for (name, metric), g in per_cohort.groupby(["contrast", "metric"], sort=False):
        d = DIRECTION[metric]
        r = pool_dl(g["estimate"].to_numpy(), g["se"].to_numpy(), d)
        rows.append({"set": set_name, "contrast": name, "label": (labels or {}).get(name, name), "metric": metric,
                     **{k: r[k] for k in ("k", "theta", "se", "ci95_low", "ci95_high", "same_sign", "tau2", "I2", "Q", "p_one_sided", "p_two_sided")}})  # fmt: skip
    return pd.DataFrame(rows, columns=POOLED_E_COLUMNS)


SENS_COLUMNS = [
    "set", "contrast", "label", "metric", "k", "sesoi", "direction",
    "dl_theta", "dl_ci95_low", "dl_ci95_high", "dl_se", "dl_tau2", "dl_I2", "dl_p_one_sided", "dl_p_holm", "dl_verdict",
    "hksj_theta", "hksj_ci95_low", "hksj_ci95_high", "hksj_se", "hksj_tau2", "hksj_q", "hksj_p_one_sided", "hksj_p_holm", "hksj_verdict",
    "hksjt_ci95_low", "hksjt_ci95_high", "hksjt_p_one_sided", "hksjt_p_holm", "hksjt_verdict", "reml_wald_ci95_low", "reml_wald_ci95_high",
    "sign_dl", "sign_hksj", "sign_changed", "excl0_dl", "excl0_hksj", "excl0_changed", "excl0_hksjt", "excl0_changed_trunc",
    "verdict_changed", "verdict_changed_trunc", "verdict_illustrative_dl", "verdict_illustrative_hksj", "verdict_illustrative_changed",
]  # fmt: skip


def sensitivity_table(per_cohort: pd.DataFrame, sets: Mapping[tuple[str, str], str], labels: Mapping[str, str],
                      primary: Mapping[tuple[str, str], str] | None = None) -> pd.DataFrame:  # fmt: skip
    """DL against REML + HKSJ for every ``(contrast, metric)`` of ``per_cohort`` (columns ``cohort, contrast, metric, estimate, se``).

    ``sets``: ``(contrast, metric) -> set name`` (the rows to analyse, in table order); ``primary``: ``(contrast, metric) -> 'P1'..'P5'``,
    the five tests whose verdict uses the Holm-adjusted one-sided p, recomputed WITHIN each method (the family is P1-P5)."""
    primary = dict(primary or {})
    rows = []
    for (name, metric), set_name in sets.items():
        g = per_cohort[(per_cohort["contrast"] == name) & (per_cohort["metric"] == metric)]
        if g.empty:
            continue
        d = DIRECTION[metric]
        y, se = g["estimate"].to_numpy(), g["se"].to_numpy()
        if (
            _clean(y, se)[0].size == 0
        ):  # identical rankings by construction: nothing to pool or test (see _degenerate_pool)
            continue
        dl, hk, hkt = pool_dl(y, se, d), pool_hksj(y, se, d), pool_hksj(y, se, d, truncate=True)
        rows.append({"set": set_name, "contrast": name, "label": labels.get(name, name), "metric": metric, "k": dl["k"],
                     "sesoi": SESOI.get(metric, float("nan")), "direction": d, "_dl": dl, "_hk": hk, "_hkt": hkt, "_pid": primary.get((name, metric), "")})  # fmt: skip
    holm = {}
    for method in (
        "_dl",
        "_hk",
        "_hkt",
    ):  # Holm within each method over the primary tests (a missing test counts as p = 1, as in the aggregator)
        holm[method] = A.holm({r["_pid"]: r[method]["p_one_sided"] for r in rows if r["_pid"]})
    out = []
    for r in rows:
        pid = r["_pid"]
        rec = {k: r[k] for k in ("set", "contrast", "label", "metric", "k", "sesoi", "direction")}
        for tag, key in (("dl", "_dl"), ("hksj", "_hk"), ("hksjt", "_hkt")):
            x = r[key]
            p_used = holm[key][pid] if pid else x["p_one_sided"]
            ci90 = (x["ci90_low"], x["ci90_high"])
            rec.update({f"{tag}_ci95_low": x["ci95_low"], f"{tag}_ci95_high": x["ci95_high"], f"{tag}_p_one_sided": x["p_one_sided"],
                        f"{tag}_p_holm": holm[key][pid] if pid else float("nan"),
                        f"{tag}_verdict": _verdict(x["theta"], ci90, p_used, r["metric"], r["direction"])})  # fmt: skip
            if tag != "hksjt":
                rec.update(
                    {f"{tag}_theta": x["theta"], f"{tag}_se": x["se"], f"{tag}_tau2": x["tau2"]}
                )
            if r["metric"] in (
                WMP,
                "auc_within_phase",
            ):  # no SESOI: the AUC SESOI, flagged as illustrative
                rec[f"verdict_illustrative_{tag}"] = _verdict(
                    x["theta"], ci90, p_used, r["metric"], r["direction"], ILLUSTRATIVE_SESOI
                )
        rec.update({"dl_I2": r["_dl"]["I2"], "hksj_q": r["_hk"]["q_hk"], "reml_wald_ci95_low": r["_hk"]["reml_wald_ci95_low"],
                    "reml_wald_ci95_high": r["_hk"]["reml_wald_ci95_high"]})  # fmt: skip
        rec["sign_dl"], rec["excl0_dl"] = (
            int(np.sign(r["_dl"]["theta"])),
            bool(rec["dl_ci95_low"] > 0 or rec["dl_ci95_high"] < 0),
        )
        rec["sign_hksj"], rec["excl0_hksj"] = (
            int(np.sign(r["_hk"]["theta"])),
            bool(rec["hksj_ci95_low"] > 0 or rec["hksj_ci95_high"] < 0),
        )
        rec["excl0_hksjt"] = bool(rec["hksjt_ci95_low"] > 0 or rec["hksjt_ci95_high"] < 0)
        rec["sign_changed"] = rec["sign_dl"] != rec["sign_hksj"]
        rec["excl0_changed"] = rec["excl0_dl"] != rec["excl0_hksj"]
        rec["excl0_changed_trunc"] = rec["excl0_dl"] != rec["excl0_hksjt"]
        rec["verdict_changed"] = rec["dl_verdict"] != rec["hksj_verdict"]
        rec["verdict_changed_trunc"] = rec["dl_verdict"] != rec["hksjt_verdict"]
        rec["verdict_illustrative_changed"] = (
            rec["verdict_illustrative_dl"] != rec["verdict_illustrative_hksj"]
            if "verdict_illustrative_dl" in rec
            else False
        )
        out.append(rec)
    return pd.DataFrame(out).reindex(columns=SENS_COLUMNS)


# --------------------------------------------------------------------------- the stages
def e12_run(
    runs_dir: str | Path,
    cohorts: Sequence[str],
    *,
    n_boot: int = N_BOOT,
    workers: int = 1,
    seed: int = SEED,
) -> dict:
    """E1 and E2 over the cohorts: per-cohort arm tables, per-cohort contrasts, the exclusions."""
    runs_dir = str(runs_dir)
    e1_formulas = {k: v[1] for k, v in E1_FORMULAS.items()}
    e2_formulas = {name: dict(A.FORMULAS[name].coefs) for name in E2_FORMULAS}
    e2_arms = tuple(
        dict.fromkeys((*E2_ARMS, *E2_EXTRA_ARMS, *(a for c in e2_formulas.values() for a in c)))
    )
    tasks = [{"cohort": c, "arms": E1_ARMS, "formulas": e1_formulas, "standard": True, "runs_dir": runs_dir, "n_boot": n_boot, "seed": seed,
              "missing": "skip"} for c in cohorts]  # fmt: skip
    e1 = _map(e_cohort, tasks, workers)
    tasks = [{"cohort": c, "arms": e2_arms, "formulas": e2_formulas, "standard": False, "runs_dir": runs_dir, "n_boot": n_boot, "seed": seed,
              "missing": "skip"} for c in cohorts]  # fmt: skip
    e2 = _map(e_cohort, tasks, workers)
    return {"e1": e1, "e2": e2, "e2_arms": e2_arms}


def run_info_table(
    runs_dir: str | Path,
    cohorts: Sequence[str],
    arms: Sequence[str] = ("M01", "M01-norank", "M11R", "M11R-norank"),
) -> pd.DataFrame:
    rows = []
    for c in cohorts:
        for a in arms:
            p = Path(runs_dir) / c / a / f"seed{MAIN_SEED}" / "metrics.json"
            if not p.is_file():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"cohort": c, "arm": a, "rank_weight": (m.get("train_config") or {}).get("rank_weight"), "epochs_run": m.get("epochs_run"),
                         "best_epoch": m.get("best_epoch"), "best_val_nll": m.get("best_val_nll"), "train_sec": m.get("train_sec"), "n_params": m.get("n_params"),
                         "rank_score_source": m.get("rank_score_source", "rank_head"), "config_hash": m.get("config_hash")})  # fmt: skip
    return pd.DataFrame(rows)


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(frames)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise_exclusions(exclusions: pd.DataFrame) -> dict:
    """The pooled shares over all cohorts (risk sets and rows)."""
    if exclusions.empty:
        return {}
    return {
        "n_risk_sets": int(exclusions["n_risk_sets"].sum()), "n_risk_sets_used": int(exclusions["n_risk_sets_used"].sum()),
        "share_risk_sets_excluded": float(1 - exclusions["n_risk_sets_used"].sum() / exclusions["n_risk_sets"].sum()),
        "n_rows": int(exclusions["n_rows"].sum()), "n_rows_used": int(exclusions["n_rows_used"].sum()),
        "share_rows_excluded": float(1 - exclusions["n_rows_used"].sum() / exclusions["n_rows"].sum()),
        "share_risk_sets_excluded_min_max": [float(exclusions["share_risk_sets_excluded"].min()), float(exclusions["share_risk_sets_excluded"].max())],
        "share_rows_excluded_min_max": [float(exclusions["share_rows_excluded"].min()), float(exclusions["share_rows_excluded"].max())],
    }  # fmt: skip


def build_e3(results_dir: str | Path, e1_pc: pd.DataFrame, e2_pc: pd.DataFrame) -> pd.DataFrame:
    """The pooling-sensitivity table: P1-P5, the E2 contrasts (decomposition steps and RQ2 contrasts) and the E1 contrasts, DL against REML + HKSJ."""
    pc = pd.read_csv(Path(results_dir) / "contrasts_per_cohort.csv")
    pc = pc[pc["stratum"] == A.STRATUM_ALL]
    labels = {n: f.label for n, f in A.FORMULAS.items()} | {k: v[0] for k, v in E1_FORMULAS.items()}
    # existing metrics of the E2 contrasts come from the aggregator's table, the E2 metric from the new per-cohort contrasts
    base_cols = ["cohort", "contrast", "metric", "estimate", "se"]
    frames = [pc[base_cols], e2_pc[e2_pc["metric"] == WMP][base_cols], e1_pc[base_cols]]
    allpc = pd.concat(frames, ignore_index=True)
    sets: dict[tuple[str, str], str] = {}
    primary: dict[tuple[str, str], str] = {}
    for pid, key in A.PRIMARY.items():
        sets[key] = f"primary {pid}"
        primary[key] = pid
    for name in E2_DECOMP:
        step = next(s for s in A.DECOMP_STEPS if s[3] == name)[1]
        for m in KEY_METRICS:
            sets.setdefault((name, m), f"decomposition: {step}")
    for name in E2_RQ2:
        for m in KEY_METRICS:
            sets.setdefault((name, m), "RQ2 contrast")
    for name in E1_ORIGINAL:  # (e), (f) and the rank auxiliary loss are decomposition steps already (same contrasts, aggregator names)
        for m in KEY_METRICS:
            sets[(name, m)] = "E1 original step (with the rank loss)"
    return sensitivity_table(allpc, sets, labels, primary)


def read_aggregator_summary(results_dir: str | Path) -> dict | None:
    """``summary.json`` of the aggregator run in ``results_dir`` (None if there is none)."""
    path = Path(results_dir) / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_runs_dir(results_dir: str | Path, runs_dir: str | Path) -> dict | None:
    """The aggregator output must be the one of ``runs_dir``: ``summary.json['runs_dir']`` resolves to it (ValueError otherwise).  Returns the summary."""
    info = read_aggregator_summary(results_dir)
    if info is not None and info.get("runs_dir"):
        a, b = Path(info["runs_dir"]).resolve(), Path(runs_dir).resolve()
        if a != b:
            raise ValueError(
                f"{results_dir} was aggregated from {a}, not from {b}: its contrasts_per_cohort.csv is not the table of these runs"
            )
    return info


def check_against_aggregator(
    results_dir: str | Path, e1_pc: pd.DataFrame, tol: float = AGG_TOLERANCE
) -> dict:
    """The E1 contrasts that the aggregator also has (``E1_AGG_NAME``) must reproduce its per-cohort estimates AND bootstrap SEs for the standard
    metrics: same arms, same rows, same resamples.  Raises ``ValueError`` on a mismatch (other runs, ``n_boot`` or seed).  Returns the comparison."""
    path = Path(results_dir) / "contrasts_per_cohort.csv"
    if not path.is_file() or e1_pc.empty:
        return {
            "n_compared": 0,
            "max_abs_diff_estimate": None,
            "max_abs_diff_se": None,
            "note": "no aggregator table or no E1 contrasts",
        }
    pc = pd.read_csv(path)
    pc = pc[pc["stratum"] == A.STRATUM_ALL]
    n, d_est, d_se = 0, 0.0, 0.0
    for e1_name, agg_name in E1_AGG_NAME.items():
        a, b = pc[pc["contrast"] == agg_name], e1_pc[e1_pc["contrast"] == e1_name]
        for metric in STANDARD_METRICS:
            x = a[a["metric"] == metric].set_index("cohort")
            y = b[b["metric"] == metric].set_index("cohort")
            common = x.index.intersection(y.index)
            for col, kind in (("estimate", "est"), ("se", "se")):
                if len(common):
                    diff = float(
                        np.nanmax(
                            np.abs(
                                x.loc[common, col].to_numpy(float)
                                - y.loc[common, col].to_numpy(float)
                            )
                        )
                    )
                    if kind == "est":
                        d_est = max(d_est, diff)
                    else:
                        d_se = max(d_se, diff)
            n += len(common)
    out = {
        "n_compared": int(n),
        "max_abs_diff_estimate": d_est if n else None,
        "max_abs_diff_se": d_se if n else None,
        "tolerance": tol,
    }
    if n and (d_est > tol or d_se > tol):
        raise ValueError(
            f"the E1 contrasts differ from the aggregator's in {path} (max |estimate| {d_est:.3g}, max |se| {d_se:.3g} > {tol:g}): "
            "other runs, another --n-boot / --seed than the aggregator's, or the aggregator is stale"
        )
    return out


def run(runs_dir: str | Path, results_dir: str | Path, out_dir: str | Path, *, n_boot: int = N_BOOT, workers: int = 1, stage: str = "all",
        cohorts: Sequence[str] | None = None, seed: int = SEED) -> dict:  # fmt: skip
    out = Path(out_dir)
    results_dir, runs_dir = Path(results_dir), Path(runs_dir)
    agg_summary = check_runs_dir(results_dir, runs_dir)  # before anything is written
    out.mkdir(parents=True, exist_ok=True)
    if cohorts is None:
        cohorts = sorted(
            p.name
            for p in runs_dir.iterdir()
            if p.is_dir()
            and p.name != "logs"
            and (p / "M11R" / f"seed{MAIN_SEED}" / "pred_test.csv").is_file()
        )
        if agg_summary and agg_summary.get("cohorts"):
            cohorts = [
                c for c in cohorts if c in set(agg_summary["cohorts"])
            ]  # the cohorts the aggregator analysed
    cohorts = list(cohorts)
    res = e12_run(runs_dir, cohorts, n_boot=n_boot, workers=workers, seed=seed)
    e1, e2 = res["e1"], res["e2"]
    e1_arm, e1_pc = (
        _concat(pd.DataFrame(r["arm_rows"]) for r in e1),
        _concat(pd.DataFrame(r["contrast_rows"]) for r in e1),
    )
    e2_arm, e2_pc = (
        _concat(pd.DataFrame(r["arm_rows"]) for r in e2),
        _concat(pd.DataFrame(r["contrast_rows"]) for r in e2),
    )
    excl = pd.DataFrame([r["exclusions"] for r in e2])
    by_phase = _concat(r["by_phase"] for r in e2)
    labels = {n: A.FORMULAS[n].label for n in E2_FORMULAS} | {
        k: v[0] for k, v in E1_FORMULAS.items()
    }
    e1_pooled = pool_contrast_table(e1_pc, labels, "E1")
    e2_pooled = pool_contrast_table(e2_pc, labels, "E2")
    aggregator = check_against_aggregator(results_dir, e1_pc)
    e3 = build_e3(results_dir, e1_pc, e2_pc)
    info = run_info_table(runs_dir, cohorts)
    files = {
        "e1_arm_metrics_per_cohort.csv": e1_arm, "e1_contrasts_per_cohort.csv": e1_pc, "e1_contrasts_pooled_dl.csv": e1_pooled, "e1_run_info.csv": info,
        "e2_arm_wmp_per_cohort.csv": e2_arm, "e2_contrasts_per_cohort.csv": e2_pc, "e2_contrasts_pooled_dl.csv": e2_pooled,
        "e2_excluded_risk_sets.csv": excl, "e2_excluded_by_phase.csv": by_phase, "e3_pooling_sensitivity.csv": e3,
    }  # fmt: skip
    for name, frame in files.items():
        frame.to_csv(out / name, index=False, float_format="%.10g")
    changed = (
        e3[e3[["sign_changed", "excl0_changed", "verdict_changed"]].any(axis=1)] if len(e3) else e3
    )
    missing_arms = {
        "e1": {r["cohort"]: r["missing_arms"] for r in e1 if r["missing_arms"]},
        "e2": {r["cohort"]: r["missing_arms"] for r in e2 if r["missing_arms"]},
    }
    summary = {
        "cohorts": cohorts, "n_boot": int(n_boot), "seed": int(seed), "e2_exclusions_pooled": summarise_exclusions(excl),
        "e3_rows": int(len(e3)), "e3_sign_changed": int(e3["sign_changed"].sum()), "e3_excl0_changed": int(e3["excl0_changed"].sum()),
        "e3_verdict_changed": int(e3["verdict_changed"].sum()), "e3_excl0_changed_trunc": int(e3["excl0_changed_trunc"].sum()),
        "e3_verdict_changed_trunc": int(e3["verdict_changed_trunc"].sum()),
        "e3_changed": changed[["set", "contrast", "metric", "dl_theta", "hksj_theta", "dl_verdict", "hksj_verdict", "excl0_dl", "excl0_hksj"]].to_dict("records"),
        "aggregator": {
            "results_dir": str(results_dir), "runs_dir": str(runs_dir), "summary_found": agg_summary is not None,
            "n_boot": (agg_summary or {}).get("n_boot"), "seed": (agg_summary or {}).get("seed"), "generated_utc": (agg_summary or {}).get("generated_utc"),
            "same_bootstrap": agg_summary is not None and (agg_summary.get("n_boot"), agg_summary.get("seed")) == (int(n_boot), int(seed)),
            "e1_reproduces_aggregator": aggregator,
        },
        "missing_arms": missing_arms,
        "git": git_info(),
    }  # fmt: skip
    write_json(out / "supplement_summary.json", summary)
    return {"summary": summary, "e1_arm": e1_arm, "e1_pc": e1_pc, "e1_pooled": e1_pooled, "e2_arm": e2_arm, "e2_pc": e2_pc, "e2_pooled": e2_pooled,
            "excl": excl, "by_phase": by_phase, "e3": e3, "info": info}  # fmt: skip


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m model_stgcn.ptr.supplement", description=__doc__.split("\n")[0]
    )
    ap.add_argument(
        "--runs-dir", default=str(P.RUNS_DIR), help="run directory (default: the v2 root)"
    )
    ap.add_argument(
        "--results-dir",
        default=str(P.RESULTS_DIR),
        help="the aggregator's output (contrasts_per_cohort.csv, summary.json); default: the v2 root",
    )
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/supplement")
    ap.add_argument(
        "--dump-dir",
        default=None,
        help="dump directory whose meta.version is compared with --runs-dir / --results-dir (without it those two are compared with each other)",
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help=f"default: the aggregator's (summary.json), else {N_BOOT}",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed of the bootstrap; default: the aggregator's (summary.json), else 0 (its default)",
    )
    ap.add_argument("--workers", type=int, default=3, help="parallel cohorts (spawned processes)")
    ap.add_argument(
        "--cohorts",
        default=None,
        help="comma-separated cohorts (default: every cohort with an M11R run that the aggregator analysed)",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    try:
        if args.dump_dir is not None:
            P.check_same_version(
                args.dump_dir,
                {"runs (--runs-dir)": args.runs_dir, "results (--results-dir)": args.results_dir},
            )
        else:  # without a dump the two directories must agree with each other
            rv, ov = P.artifact_version(args.runs_dir)[0], P.artifact_version(args.results_dir)[0]
            if rv and ov and rv != ov:
                raise P.VersionMixError(
                    f"--runs-dir {args.runs_dir} is {rv} but --results-dir {args.results_dir} is {ov}"
                )
    except P.VersionMixError as exc:
        raise SystemExit(f"refusing to run: {exc}") from exc
    out_dir = args.out_dir or str(Path(args.results_dir) / "supplement")
    agg = read_aggregator_summary(args.results_dir) or {}
    n_boot = args.n_boot if args.n_boot is not None else int(agg.get("n_boot") or N_BOOT)
    seed = args.seed if args.seed is not None else int(agg.get("seed") or SEED)
    if agg and (agg.get("n_boot"), agg.get("seed")) != (n_boot, seed):
        logger.warning(
            "bootstrap differs from the aggregator's (n_boot %s, seed %s): the E1 contrasts will not reproduce its SEs",
            agg.get("n_boot"),
            agg.get("seed"),
        )
    cohorts = [c for c in args.cohorts.split(",") if c] if args.cohorts else None
    res = run(
        args.runs_dir,
        args.results_dir,
        out_dir,
        n_boot=n_boot,
        workers=args.workers,
        cohorts=cohorts,
        seed=seed,
    )
    s = res["summary"]
    print(f"{len(s['cohorts'])} cohorts, {s['n_boot']} resamples; E3: {s['e3_rows']} rows, sign changed {s['e3_sign_changed']}, CI-excludes-0 changed "
          f"{s['e3_excl0_changed']}, verdict changed {s['e3_verdict_changed']}")  # fmt: skip
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
