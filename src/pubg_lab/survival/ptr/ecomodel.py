"""Ecological hazard model (EHM): an explicit discrete-time local-extinction model of team death (exploratory).

In the spirit of the incidence function model of metapopulation ecology, a team is a local population that goes extinct
between two cuts with a probability that depends on the state of its habitat patch.  Every cut phase ``k`` is one
habitat-loss step of the shrinking zone, so the model is a logistic regression of the next-phase death on the cut phase
(one base rate per step) and a SMALL set of habitat covariates, fitted per cohort on train and scored on val / test.  It is
the interpretable counterpart of the learned arms: how much of their discrimination does a handful of ecologically
motivated terms recover?

Arms (all on exactly the rows of ``HGB-cut``: ``hgb.build_rows(dump, 'HGB-cut', 'v2')``, labels v2 with winner censoring)::

    PHASE-only   intercept + cut-phase dummies (the base rate of every habitat-loss step; nothing else)
    EHM          PHASE-only + the six habitat terms below
    EHM-int      EHM + outside depth x (cut phase - 4): "the edge gradient steepens as the habitat shrinks"
    EHM-lite     PHASE-only + required speed, local crowding, log alive teams, team members: EHM without the two edge terms
                 (outside depth, inside cushion), which are collinear with required speed (r ~ 0.9); same fitting, SEs and rows

Habitat terms (dump columns, ``meta['feature_names']``; the dump divides them by fixed constants, which the z-scoring removes)::

    outside_depth    max(0, -d)   d = hab_signed_boundary_distance_end_m (positive inside the blue zone, negative outside)
    inside_cushion   max(0,  d)
    required_speed   hab_required_speed_to_safe_mps        speed needed to reach the target circle (distance / phase duration)
    local_crowding   hab_local_enemy_count_200m            enemy teams within 200 m at the team's last snapshot
    log_alive_teams  log(100 * hab_alive_teams)            log of the teams alive entering the next phase (equals the number
                                                           of risk-set teams of that match and cut in 96% of the cells)
    team_members     alive_players_end_norm                event-based alive members of the team at the phase end (count / 4);
                                                           constant in solo cohorts, where it is dropped (recorded)

The six terms are z-scored on the TRAIN rows of the cohort (so a coefficient is the log-odds change per SD); the phase dummies
(reference: the first phase) are not scaled.  The fit is an unpenalised logistic regression (Newton / IRLS with step halving);
only if it does not converge, or a coefficient diverges (separation), the same model is refitted with a tiny L2 penalty
(lambda = 1e-4 on the summed log-likelihood, intercept unpenalised) and ``penalty`` / ``fallback_reason`` are recorded.  Standard
errors: model based (inverse information) and cluster robust by match (the primary ones: rows of one match are not
independent), with the Stata-style small-sample factor.  A pooled coefficient is the DerSimonian-Laird random-effects mean of the
per-cohort coefficients with those standard errors (``metrics.dersimonian_laird``).

Outputs: per cohort and arm ``<runs>/<cohort>/<arm>/seed42/{pred_val.csv, pred_test.csv, metrics.json}`` in the standard format
(``rank_score = -p_hazard`` like the other tabular arms; ``kind`` ecological) and, from :func:`report`, the tables in ``--out-dir``::

    auc_comparison.csv           pooled and within-phase test AUC of the ecological arms next to HGB-cut, M11R, K3a, KRAFTON-tf, SAGONG-hgb
    share_of_discrimination.csv  (AUC_arm - 0.5) / (AUC_HGB-cut - 0.5) per cohort and pooled, with match-bootstrap intervals
    coefficients_per_cohort.csv  standardised coefficients with both standard errors
    coefficients_pooled.csv      per covariate: sign agreement, DL pooled coefficient and 95% interval
    collinearity.csv             VIF of every habitat term and its correlations (outside depth and required speed are r ~ 0.9)
    vif.csv                      VIF among the habitat terms IN each arm (EHM, EHM-int, EHM-lite)
    auc_difference_ehm_minus_lite.csv
                                 EHM - EHM-lite test AUC (pooled, within phase) per cohort and pooled, paired match bootstrap
    single_term_coefficients.csv, single_term_pooled.csv
                                 PHASE-only + ONE habitat term at a time (the phase-adjusted marginal coefficient, per cohort / pooled)
    fit_info.csv, summary.json

Reading the joint coefficients needs the diagnostics: outside depth and required speed are nearly collinear (a team outside the blue
zone is far from the target circle), so in the joint model they form a suppression pair (opposite signs that largely cancel); the
single-term fits give each term's marginal effect.

    python -m model_stgcn.ptr.ecomodel --dump-dir model_stgcn/dumps/bigcomp2027_phase_v1 \\
        --runs-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 --out-dir reports/bigcomp2027_results/ecological_model

RQ1 (``--stage rq1``, tables and forest figure in ``<out-dir>/rq1/``): the association of exposure, movement burden, local competition and
remaining team size with next-phase elimination, per cohort (M-main, and M-phase with covariate x phase-group interactions), pooled
over cohorts, with the test that a slope is the same in the early, mid and late phases; see the RQ1 section of this module.

RQ1 in physical units (``--stage rq1 --rq1-units physical``, supplementary experiment E4, tables in ``<out-dir>/rq1_physical/``): the same M-main
and M-phase models with the covariates in PHYSICAL units instead of per-cohort SD (outside depth and inside cushion per 100 m, migration deficit
per 1 m/s, local crowding per 1 enemy team within 200 m, remaining members per 1 teammate, log alive teams unchanged), DL-pooled
coefficients with sign counts and I^2, and the average marginal effect on the probability scale per covariate and phase group; see the last
section of this module.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pubg_lab.survival.ptr import data as D
from pubg_lab.survival.ptr import hgb as H
from pubg_lab.survival.ptr import metrics as M
from pubg_lab.survival.ptr.cohorts import discover_cohorts
from pubg_lab.survival.ptr.util import (
    code_fingerprint,
    config_hash,
    dump_fingerprint,
    git_info,
    write_json,
)

logger = logging.getLogger("model_stgcn.ptr.ecomodel")

PHASE_ONLY, EHM, EHM_INT, EHM_LITE = "PHASE-only", "EHM", "EHM-int", "EHM-lite"
ARMS: tuple[str, ...] = (PHASE_ONLY, EHM, EHM_INT, EHM_LITE)
SEED = 42  # the arms are deterministic; the run directory is seed42 like the other arms
PHASE_CENTER = 4  # the interaction is outside depth x (cut phase - 4): the main effect is the slope at the middle of the 7 cuts
ALIVE_TEAMS_SCALE = 100.0  # the dump stores hab_alive_teams / 100 (stdb/queries.py)
INTERACTION = "outside_depth:phase"
L2_LAMBDA = 1e-4  # the tiny ridge of the fallback fit (summed log-likelihood scale)
COEF_LIMIT = (
    25.0  # |coefficient| beyond this means quasi-separation: the unpenalised fit is rejected
)
CONSTANT_SD = 1e-12
COMPARISON_ARMS: tuple[str, ...] = (
    "HGB-cut",
    "M11R",
    "KRAFTON-tf-ourfeat-bce",
    "KRAFTON-tf",
    "SAGONG-hgb",
)  # K3a = KRAFTON-tf-ourfeat-bce


@dataclass(frozen=True)
class Covariate:
    term: str
    column: str  # the dump column it is built from
    meaning: str


COVARIATES: tuple[Covariate, ...] = (
    Covariate("outside_depth", "hab_signed_boundary_distance_end_m", "max(0, -d): depth outside the blue-zone boundary at the cut"),
    Covariate("inside_cushion", "hab_signed_boundary_distance_end_m", "max(0, d): distance inside the boundary at the cut"),
    Covariate("required_speed", "hab_required_speed_to_safe_mps", "speed needed to reach the target circle"),
    Covariate("local_crowding", "hab_local_enemy_count_200m", "enemy teams within 200 m at the last snapshot"),
    Covariate("log_alive_teams", "hab_alive_teams", "log of the alive teams entering the next phase (competitor density)"),
    Covariate("team_members", "alive_players_end_norm", "event-based alive members of the team at the phase end (count / 4)"),
)  # fmt: skip
TERMS: tuple[str, ...] = tuple(c.term for c in COVARIATES)
SOURCE_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(c.column for c in COVARIATES))
LITE_TERMS: tuple[str, ...] = (
    "required_speed",
    "local_crowding",
    "log_alive_teams",
    "team_members",
)  # EHM-lite: no edge terms


# --------------------------------------------------------------------------- inputs: the rows of HGB-cut
@dataclass
class Inputs:
    """What the model reads from a split's HGB-cut rows."""

    phase: np.ndarray  # int64 cut phase k = 1..7
    raw: pd.DataFrame  # the six habitat terms, float64, columns :data:`TERMS`
    match: np.ndarray  # int64 match code (the clusters of the robust standard errors)

    def __len__(self) -> int:
        return int(len(self.phase))


def column_values(rows: H.Rows, name: str) -> np.ndarray:
    """One feature of ``rows.X`` by its dump name (float64)."""
    names = list(rows.feature_names)
    if name not in names:
        raise ValueError(f"the dump has no feature '{name}' (needed by the ecological model); known habitat columns: "
                         f"{[n for n in names if n.startswith('hab_')][:8]} ...")  # fmt: skip
    return np.asarray(rows.X[:, names.index(name)], dtype=np.float64)


TERM_SOURCES: dict[
    str, str
] = {  # habitat term -> the dump column it is read from (the edge terms and log_alive_teams are derived below)
    "required_speed": "hab_required_speed_to_safe_mps",
    "local_crowding": "hab_local_enemy_count_200m",
    "migration_deficit": "hab_migration_deficit_mps",
    "team_members": "alive_players_end_norm",
}


def _term_frame(rows: H.Rows, terms: Sequence[str]) -> pd.DataFrame:
    """The raw habitat terms ``terms`` of a split's rows; only the dump columns that these terms need are read."""
    out: dict[str, np.ndarray] = {}
    d = None
    for t in terms:
        if t in ("outside_depth", "inside_cushion"):
            d = column_values(rows, "hab_signed_boundary_distance_end_m") if d is None else d
            out[t] = np.maximum(0.0, -d) if t == "outside_depth" else np.maximum(0.0, d)
        elif t == "log_alive_teams":
            out[t] = np.log(
                np.maximum(ALIVE_TEAMS_SCALE * column_values(rows, "hab_alive_teams"), 1.0)
            )
        else:
            out[t] = column_values(rows, TERM_SOURCES[t])
    return pd.DataFrame(out)


def _inputs(rows: H.Rows, terms: Sequence[str]) -> Inputs:
    raw = _term_frame(rows, terms)
    if not np.isfinite(raw.to_numpy()).all():
        bad = [c for c in raw.columns if not np.isfinite(raw[c].to_numpy()).all()]
        raise ValueError(
            f"non-finite habitat terms {bad}: the dump rule is NULL / NaN / inf -> 0, so this should not happen"
        )
    return Inputs(
        phase=rows.keys["phase"].to_numpy(dtype=np.int64),
        raw=raw,
        match=pd.factorize(rows.keys["match_id"])[0].astype(np.int64),
    )


def make_inputs(rows: H.Rows) -> Inputs:
    """The cut phase, the six raw habitat terms and the match clusters of a split (rows = ``hgb.build_rows(dump, 'HGB-cut')``)."""
    return _inputs(rows, TERMS)


# --------------------------------------------------------------------------- logistic regression
@dataclass
class LogitFit:
    beta: np.ndarray
    cov_model: np.ndarray  # inverse information at the solution
    cov_robust: np.ndarray | None  # clustered by match (None without groups)
    loglik: float  # unpenalised log-likelihood at the solution
    n_iter: int
    converged: bool  # the iterations converged AND every coefficient lies inside the bound (:data:`COEF_LIMIT`)
    penalty: str  # 'none' | 'l2'
    l2: float
    n: int
    fallback_reason: str = ""
    newton_converged: bool = (
        True  # the iterations alone converged (False: they ran out, or the Hessian was singular)
    )

    @property
    def se_model(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_model))

    @property
    def se_robust(self) -> np.ndarray:
        return (
            np.sqrt(np.diag(self.cov_robust))
            if self.cov_robust is not None
            else np.full(len(self.beta), np.nan)
        )


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * eta))  # stable for any |eta|


def _loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    l2: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-9,
    coef_scale: np.ndarray | None = None,
) -> LogitFit:
    """Logistic regression by Newton / IRLS with step halving; column 0 must be the intercept (never penalised).

    ``l2 > 0`` adds ``l2 * |beta|^2 / 2`` (all columns but the intercept) to the negative log-likelihood.  ``groups`` (cluster
    codes) adds the cluster-robust covariance with the factor ``G / (G - 1) * (n - 1) / (n - K)``.  ``converged`` is False if the
    iterations ran out, the Hessian was singular or a coefficient left every bound (see :data:`COEF_LIMIT`): quasi-separation.
    The bound applies to ``|beta_j| * coef_scale_j`` (default scale 1: the coefficient itself); pass the standard deviation of the design
    columns when some of them have a small spread, where a large coefficient is only a unit effect, not separation."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, k = X.shape
    pen = np.full(k, float(l2))
    pen[0] = 0.0
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta = np.zeros(k)
    beta[0] = np.log(p0 / (1.0 - p0))

    def objective(b: np.ndarray) -> float:
        return _loglik(X, y, b) - 0.5 * float(pen @ (b * b))

    obj = objective(beta)
    converged, n_iter = False, 0
    for n_iter in range(1, max_iter + 1):
        p = _sigmoid(X @ beta)
        grad = X.T @ (y - p) - pen * beta
        hess = (X * (p * (1.0 - p))[:, None]).T @ X + np.diag(pen)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while True:
            cand = beta + t * step
            cand_obj = objective(cand)
            if cand_obj >= obj - 1e-12 or t < 1e-12:
                break
            t *= 0.5
        gain = cand_obj - obj
        beta, obj = cand, cand_obj
        if float(np.max(np.abs(t * step))) < tol * (1.0 + float(np.max(np.abs(beta)))) and abs(
            gain
        ) < 1e-9 * (1.0 + abs(obj)):
            converged = True
            break
    scale = np.ones(k) if coef_scale is None else np.asarray(coef_scale, dtype=np.float64)
    bounded = bool(np.isfinite(beta).all() and float(np.max(np.abs(beta) * scale)) <= COEF_LIMIT)
    p = _sigmoid(X @ beta)
    hess = (X * (p * (1.0 - p))[:, None]).T @ X + np.diag(pen)
    try:
        bread = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        bread, converged = np.linalg.pinv(hess), False
    newton_ok = bool(converged)
    converged = newton_ok and bounded
    cov_robust = None
    if groups is not None:
        codes = pd.factorize(np.asarray(groups))[0]
        n_groups = int(codes.max()) + 1
        scores = np.zeros((n_groups, k))
        np.add.at(scores, codes, X * (y - p)[:, None])
        meat = scores.T @ scores
        factor = (n_groups / max(n_groups - 1, 1)) * ((n - 1) / max(n - k, 1))
        cov_robust = factor * bread @ meat @ bread
    return LogitFit(
        beta=beta, cov_model=bread, cov_robust=cov_robust, loglik=_loglik(X, y, beta), n_iter=n_iter, converged=bool(converged),
        penalty="l2" if l2 > 0 else "none", l2=float(l2), n=int(n), newton_converged=newton_ok,
    )  # fmt: skip


def fit_logit_auto(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    coef_scale: np.ndarray | None = None,
) -> LogitFit:
    """Unpenalised first; a tiny L2 (:data:`L2_LAMBDA`) only if that does not converge, with the reason recorded.

    ``coef_scale``: see :func:`fit_logit`."""
    fit = fit_logit(X, y, groups=groups, coef_scale=coef_scale)
    if fit.converged:
        return fit
    scale = 1.0 if coef_scale is None else np.asarray(coef_scale, dtype=np.float64)
    largest = (
        float(np.max(np.abs(fit.beta) * scale)) if np.isfinite(fit.beta).all() else float("inf")
    )
    if fit.newton_converged:
        reason = f"unpenalised fit left the coefficient bound after {fit.n_iter} iterations (largest |coefficient| {largest:.1f} > {COEF_LIMIT:g})"
    else:
        reason = f"unpenalised fit did not converge after {fit.n_iter} iterations (largest |coefficient| {largest:.1f})"
    fit2 = fit_logit(X, y, groups=groups, l2=L2_LAMBDA, coef_scale=coef_scale)
    fit2.fallback_reason = reason
    return fit2


# --------------------------------------------------------------------------- the arms
@dataclass
class ArmModel:
    arm: str
    phases: list[
        int
    ]  # the cut phases of the train rows; the first one is the reference level of the dummies
    terms: list[
        str
    ]  # design columns: intercept, phase_<k>, the habitat terms in use, the interaction
    used: list[str]  # habitat terms in the model (constant ones are dropped)
    dropped: dict[str, str]  # term -> reason
    mean: dict[str, float]  # train mean of every raw habitat term (used or not)
    sd: dict[str, float]
    fit: LogitFit

    def coefficients(self) -> list[dict]:
        se_r, se_m = self.fit.se_robust, self.fit.se_model
        return [
            {"term": t, "beta": float(b), "se_robust": float(sr), "se_model": float(sm)}
            for t, b, sr, sm in zip(self.terms, self.fit.beta, se_r, se_m)
        ]  # fmt: skip


def _nearest_phase(phase: np.ndarray, phases: Sequence[int]) -> tuple[np.ndarray, int]:
    known = np.asarray(phases)
    unseen = ~np.isin(phase, known)
    out = phase.copy()
    if unseen.any():
        out[unseen] = known[np.abs(phase[unseen, None] - known[None, :]).argmin(axis=1)]
    return out, int(unseen.sum())


def design_matrix(model: ArmModel, inputs: Inputs) -> np.ndarray:
    """``[1 | phase dummies | z-scored habitat terms | outside depth x (k - 4)]`` of a split under the train coding of ``model``."""
    n = len(inputs)
    phase, _ = _nearest_phase(inputs.phase, model.phases)
    cols = [np.ones(n)]
    cols += [(phase == k).astype(np.float64) for k in model.phases[1:]]
    z = {t: (inputs.raw[t].to_numpy() - model.mean[t]) / model.sd[t] for t in model.used}
    cols += [z[t] for t in model.used]
    if INTERACTION in model.terms:
        cols.append(z["outside_depth"] * (inputs.phase - PHASE_CENTER))
    X = np.column_stack(cols)
    if X.shape[1] != len(model.terms):
        raise AssertionError(f"design has {X.shape[1]} columns for {len(model.terms)} terms")
    return X


def fit_arm(arm: str, train: Inputs, y: np.ndarray) -> ArmModel:
    """Fit one arm on the train rows (the z-scoring statistics come from these rows only)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm '{arm}'; known: {ARMS}")
    mean = {t: float(train.raw[t].mean()) for t in TERMS}
    sd = {t: float(train.raw[t].std(ddof=0)) for t in TERMS}
    phases = sorted(int(k) for k in np.unique(train.phase))
    used, dropped = [], {}
    if arm != PHASE_ONLY:
        for t in LITE_TERMS if arm == EHM_LITE else TERMS:
            if sd[t] > CONSTANT_SD:
                used.append(t)
            else:
                dropped[t] = "constant in the train rows (zero variance)"
    terms = ["intercept"] + [f"phase_{k}" for k in phases[1:]] + used
    if arm == EHM_INT:
        if "outside_depth" in used:
            terms.append(INTERACTION)
        else:
            dropped[INTERACTION] = "outside depth is not in the model"
    model = ArmModel(arm, phases, terms, used, dropped, mean, sd, fit=None)  # type: ignore[arg-type]
    model.fit = fit_logit_auto(design_matrix(model, train), y, groups=train.match)
    return model


def predict(model: ArmModel, inputs: Inputs) -> np.ndarray:
    """``p_hazard`` of a split; rows with a cut phase unseen in train take the nearest train phase's base rate."""
    return _sigmoid(design_matrix(model, inputs) @ model.fit.beta)


def n_unseen_phase_rows(model: ArmModel, inputs: Inputs) -> int:
    """Rows whose cut phase is not a train phase (they are coded as the nearest train phase; none in the real cohorts)."""
    return _nearest_phase(inputs.phase, model.phases)[1]


def vif(train: Inputs, terms: Sequence[str]) -> dict[str, float]:
    """Variance inflation factors of ``terms`` among themselves on the train rows: the diagonal of the inverse correlation matrix
    (1 for a single term; ``{}`` for none)."""
    terms = list(terms)
    if not terms:
        return {}
    if len(terms) == 1:
        return {terms[0]: 1.0}
    Z = np.column_stack(
        [(train.raw[t].to_numpy() - train.raw[t].mean()) / train.raw[t].std(ddof=0) for t in terms]
    )
    return {
        t: float(v) for t, v in zip(terms, np.diag(np.linalg.inv(np.corrcoef(Z, rowvar=False))))
    }


def diagnostics(train: Inputs, y: np.ndarray) -> dict:
    """Collinearity of the habitat terms and the phase-adjusted marginal coefficient of each term (train rows).

    ``corr`` is the correlation matrix of the terms, ``vif`` its inverse diagonal (variance inflation factors), ``single_term`` the
    standardised coefficient of each term in ``PHASE-only + that term`` (unpenalised unless it fails, cluster-robust SE)."""
    used = [t for t in TERMS if float(train.raw[t].std(ddof=0)) > CONSTANT_SD]
    Z = np.column_stack(
        [(train.raw[t].to_numpy() - train.raw[t].mean()) / train.raw[t].std(ddof=0) for t in used]
    )
    corr = np.corrcoef(Z, rowvar=False)
    phases = sorted(int(k) for k in np.unique(train.phase))
    base = [np.ones(len(train))] + [(train.phase == k).astype(np.float64) for k in phases[1:]]
    single = {}
    for j, t in enumerate(used):
        fit = fit_logit_auto(np.column_stack(base + [Z[:, j]]), y, groups=train.match)
        single[t] = {"beta": float(fit.beta[-1]), "se_robust": float(fit.se_robust[-1]), "se_model": float(fit.se_model[-1]),
                     "penalty": fit.penalty}  # fmt: skip
    return {
        "terms": used,
        "corr": {a: {b: float(corr[i, j]) for j, b in enumerate(used)} for i, a in enumerate(used)},
        "vif": vif(train, used),
        "single_term": single,
    }


# --------------------------------------------------------------------------- one cohort -> standard run directories
def _ecomodel_hash() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()


def run_dir(runs_dir: str | Path, cohort: str, arm: str, seed: int = SEED) -> Path:
    return Path(runs_dir) / cohort / arm / f"seed{seed}"


def run_cohort(
    dump_dir: str | Path,
    cohort: str,
    runs_dir: str | Path,
    *,
    labels: str = D.DEFAULT_LABELS,
    match_types: tuple[str, ...] | None = D.DEFAULT_MATCH_TYPES,
    arms: Sequence[str] = ARMS,
) -> dict[str, dict]:
    """Fit the arms on the train rows of a cohort, write predictions and ``metrics.json`` of val and test for each."""
    t0 = time.perf_counter()
    splits = D.load_cohort(dump_dir, cohort, labels=labels, match_types=match_types)
    rows = {
        s: H.build_rows(splits[s], "HGB-cut", labels) for s in splits
    }  # the very rows of HGB-cut
    train_y = rows["train"].y
    if len(np.unique(train_y)) < 2:
        raise ValueError(f"{cohort}: the train rows hold a single hazard class")
    inputs = {s: make_inputs(r) for s, r in rows.items()}
    out: dict[str, dict] = {}
    for arm in arms:
        model = fit_arm(arm, inputs["train"], train_y)
        directory = run_dir(runs_dir, cohort, arm)
        result: dict = {
            "variant": arm, "arm": arm, "kind": "ecological", "cohort": cohort, "seed": SEED, "labels": labels,
            "label_fields": dict(zip(("y", "target"), D.LABEL_FIELDS[labels])),
            "config_hash": config_hash(
                {"kind": "ecological", "arm": arm, "cohort": cohort, "seed": SEED, "labels": labels, "terms": list(TERMS),
                 "phase_center": PHASE_CENTER, "dump": dump_fingerprint(dump_dir, cohort), "code": code_fingerprint(),
                 "ecomodel": _ecomodel_hash()}
            ),  # fmt: skip
            "n_matches": {s: len(splits[s]["matches"]) for s in splits},
            "n_train_rows": int(len(train_y)),
            "n_features": len(model.terms),
            "covariates": [{"term": c.term, "column": c.column, "meaning": c.meaning} for c in COVARIATES],
            "model": {
                "terms": model.terms, "used_covariates": model.used, "dropped": model.dropped, "phases": model.phases,
                "reference_phase": model.phases[0], "phase_center": PHASE_CENTER,
                "standardisation": {t: {"mean": model.mean[t], "sd": model.sd[t]} for t in TERMS},
                "penalty": model.fit.penalty, "l2_lambda": model.fit.l2, "converged": model.fit.converged,
                "n_iter": model.fit.n_iter, "fallback_reason": model.fit.fallback_reason, "loglik_train": model.fit.loglik,
                "standard_errors": "se_robust = clustered by match (Stata factor G/(G-1)*(n-1)/(n-K)); se_model = inverse information",
            },
            "coefficients": model.coefficients(),
            "git": git_info(),
        }  # fmt: skip
        if model.used:
            result["vif"] = vif(
                inputs["train"], model.used
            )  # among the habitat terms that are IN this arm
        if arm == EHM:
            result["diagnostics"] = diagnostics(inputs["train"], train_y)
        for split in ("val", "test"):
            df = H._prediction_frame(rows[split], predict(model, inputs[split]))
            M.write_predictions(df, directory / f"pred_{split}.csv")
            result[split] = M.all_metrics(df)
            result[split]["within_phase"] = M.within_phase_auc(df)
        result["n_unseen_phase_rows"] = {
            s: n_unseen_phase_rows(model, inputs[s]) for s in ("val", "test")
        }
        result["elapsed_sec"] = time.perf_counter() - t0
        write_json(directory / "metrics.json", result)
        out[arm] = result
        logger.info("%s %-10s test AUC %.4f (within phase %.4f)  penalty %s  %d terms", cohort, arm, result["test"]["hazard"]["auc"],
                    result["test"]["within_phase"]["auc_within_phase"], model.fit.penalty, len(model.terms))  # fmt: skip
    return out


# --------------------------------------------------------------------------- report: AUCs, share of discrimination, coefficients
def pooled_dl(beta: Sequence[float], se: Sequence[float]) -> dict:
    """Random-effects (DerSimonian-Laird) pooled coefficient and the sign agreement of the cohorts."""
    b = np.asarray(beta, dtype=np.float64)
    s = np.asarray(se, dtype=np.float64)
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return {"k": 0}
    dl = M.dersimonian_laird(b[ok], s[ok])
    sign = np.sign(dl["theta"])
    return {
        "k": dl["k"], "n_positive": int((b[ok] > 0).sum()), "n_negative": int((b[ok] < 0).sum()),
        "same_sign_as_pooled": int((np.sign(b[ok]) == sign).sum()), "theta": dl["theta"], "se": dl["se"],
        "ci95_low": dl["ci95"][0], "ci95_high": dl["ci95"][1], "z": dl["z"], "p": dl["p"], "tau2": dl["tau2"], "I2": dl["I2"],
    }  # fmt: skip


def share_of_discrimination(auc_arm: float, auc_ref: float) -> float:
    """``(AUC_arm - 0.5) / (AUC_ref - 0.5)``; NaN when the reference does not discriminate."""
    den = auc_ref - 0.5
    return float("nan") if not np.isfinite(den) or den <= 0 else float((auc_arm - 0.5) / den)


def _arm_frame(runs_dir: Path, cohort: str, arm: str, seed: int = SEED) -> pd.DataFrame | None:
    path = run_dir(runs_dir, cohort, arm, seed) / "pred_test.csv"
    return M.read_predictions(path) if path.is_file() else None


def auc_comparison(
    runs_dir: str | Path, cohorts: Sequence[str], arms: Sequence[str] = (*ARMS, *COMPARISON_ARMS)
) -> pd.DataFrame:
    """Test AUC (pooled, within phase), AP, NLL and Brier of every arm that exists, per cohort, from the prediction files."""
    rows = []
    for cohort in cohorts:
        for arm in arms:
            df = _arm_frame(Path(runs_dir), cohort, arm)
            if df is None:
                continue
            h = M.hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy())
            w = M.within_phase_auc(df)
            rows.append({"cohort": cohort, "arm": arm, "n_rows": h["n"], "auc": h["auc"], "auc_within_phase": w["auc_within_phase"],
                         "n_phases_within": w["n_phases"], "ap": h["ap"], "nll": h["nll"], "brier": h["brier"]})  # fmt: skip
    return pd.DataFrame(rows)


def _cohort_seed(cohort: str) -> int:
    return 20260 + zlib.crc32(cohort.encode()) % 100_000


def _scored(
    runs_dir: Path,
    cohorts: Sequence[str],
    arms: Sequence[str],
    metrics: Sequence[str],
    n_boot: int,
    frames_in_memory: Mapping[tuple[str, str], pd.DataFrame] | None = None,
) -> tuple[list[str], dict, dict]:
    """``(cohorts that have every arm, points, draws)`` of ``metrics`` for ``arms``: ``points[(cohort, arm, metric)]`` on the full test
    rows and, with ``n_boot > 0``, ``draws[...]`` as ``[n_boot]`` arrays from ONE match-level resampling per cohort shared by the arms.
    ``frames_in_memory[(cohort, arm)]`` supplies prediction frames that are not (or not yet) run directories."""
    metrics = tuple(metrics)
    points: dict[tuple[str, str, str], float] = {}
    draws: dict[tuple[str, str, str], np.ndarray] = {}
    used: list[str] = []
    given = frames_in_memory or {}
    for cohort in cohorts:
        frames = {
            a: given.get((cohort, a))
            if given.get((cohort, a)) is not None
            else _arm_frame(runs_dir, cohort, a)
            for a in arms
        }
        if any(f is None for f in frames.values()):
            continue
        used.append(cohort)
        if n_boot > 0:
            _, point, samples = M.bootstrap_arms(
                frames, metrics=metrics, n_boot=n_boot, seed=_cohort_seed(cohort)
            )
        else:
            suites = {a: M.MetricSuite(f) for a, f in frames.items()}
            point = {a: s.evaluate(np.ones(s.n_matches), metrics) for a, s in suites.items()}
            samples = None
        for a in arms:
            for m in metrics:
                points[(cohort, a, m)] = float(point[a][m])
                if samples is not None:
                    draws[(cohort, a, m)] = samples[a][m]
    return used, points, draws


def auc_difference_table(
    runs_dir: str | Path,
    cohorts: Sequence[str],
    *,
    a: str = EHM,
    b: str = EHM_LITE,
    n_boot: int = 2000,
    frames_in_memory: Mapping[tuple[str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """``AUC(a) - AUC(b)`` (pooled and within phase) per cohort and pooled, with a paired match-level bootstrap interval; ``pooled`` is the
    difference of the cross-cohort mean AUCs, its interval from the cohort-wise resamples (same resamples as :func:`share_table`)."""
    metrics = ("auc", "auc_within_phase")
    used, points, draws = _scored(
        Path(runs_dir), cohorts, (a, b), metrics, n_boot, frames_in_memory
    )
    rows = []
    for m in metrics:
        for cohort in (*used, "pooled") if used else ():
            if cohort == "pooled":
                auc_a = float(np.mean([points[(c, a, m)] for c in used]))
                auc_b = float(np.mean([points[(c, b, m)] for c in used]))
                d_a = np.mean([draws[(c, a, m)] for c in used], axis=0) if draws else None
                d_b = np.mean([draws[(c, b, m)] for c in used], axis=0) if draws else None
            else:
                auc_a, auc_b = points[(cohort, a, m)], points[(cohort, b, m)]
                d_a, d_b = draws.get((cohort, a, m)), draws.get((cohort, b, m))
            lo = hi = float("nan")
            if d_a is not None and d_b is not None:
                diff = (d_a - d_b)[np.isfinite(d_a - d_b)]
                if diff.size:
                    lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            rows.append({"cohort": cohort, "metric": m, "difference": auc_a - auc_b, "ci_low": lo, "ci_high": hi, "auc_a": auc_a,
                         "auc_b": auc_b, "a": a, "b": b})  # fmt: skip
    return pd.DataFrame(rows)


def vif_table(
    runs_dir: str | Path, cohorts: Sequence[str], arms: Sequence[str] = (EHM, EHM_INT, EHM_LITE)
) -> pd.DataFrame:
    """VIF among the habitat terms of each arm, per cohort (``metrics.json`` ``vif``; an EHM run from before that key falls back to the
    ``diagnostics`` block, which holds the same numbers)."""
    rows = []
    for cohort in cohorts:
        for arm in arms:
            res = read_run(runs_dir, cohort, arm)
            if res is None:
                continue
            values = res.get("vif") or (res.get("diagnostics") or {}).get("vif") or {}
            rows += [
                {"cohort": cohort, "arm": arm, "term": t, "vif": float(v)}
                for t, v in values.items()
            ]
    return pd.DataFrame(rows, columns=["cohort", "arm", "term", "vif"])


def share_table(
    runs_dir: str | Path,
    cohorts: Sequence[str],
    *,
    arms: Sequence[str] = ARMS,
    reference: str = "HGB-cut",
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Share of discrimination of the ecological arms relative to ``reference`` for the pooled and the within-phase AUC.

    Per cohort with a paired match-level bootstrap interval (the arms and the reference are scored on the same resamples); ``pooled``
    is the ratio of the cross-cohort MEANS of the AUCs, with an interval from the cohort-wise resamples (cohorts are independent, so
    the mean over cohorts of one resample index is one draw of the pooled statistic)."""
    metrics = ("auc", "auc_within_phase")
    used_cohorts, points, draws = _scored(
        Path(runs_dir), cohorts, (*arms, reference), metrics, n_boot
    )
    rows = []
    for m in metrics:
        for a in arms:
            for cohort in (*used_cohorts, "pooled"):
                if cohort == "pooled":
                    if not used_cohorts:
                        continue
                    arm_auc = float(np.mean([points[(c, a, m)] for c in used_cohorts]))
                    ref_auc = float(np.mean([points[(c, reference, m)] for c in used_cohorts]))
                    d_arm = (
                        np.mean([draws[(c, a, m)] for c in used_cohorts], axis=0) if draws else None
                    )
                    d_ref = (
                        np.mean([draws[(c, reference, m)] for c in used_cohorts], axis=0)
                        if draws
                        else None
                    )
                else:
                    arm_auc, ref_auc = points[(cohort, a, m)], points[(cohort, reference, m)]
                    d_arm, d_ref = draws.get((cohort, a, m)), draws.get((cohort, reference, m))
                lo = hi = float("nan")
                if d_arm is not None and d_ref is not None:
                    num, den = d_arm - 0.5, d_ref - 0.5
                    ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                    ratio = ratio[np.isfinite(ratio)]
                    if ratio.size:
                        lo, hi = float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))
                rows.append({"cohort": cohort, "arm": a, "metric": m, "auc_arm": arm_auc, "auc_reference": ref_auc,
                             "share": share_of_discrimination(arm_auc, ref_auc), "ci_low": lo, "ci_high": hi, "reference": reference})  # fmt: skip
    return pd.DataFrame(rows)


def read_run(runs_dir: str | Path, cohort: str, arm: str) -> dict | None:
    import json

    path = run_dir(runs_dir, cohort, arm) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def coefficient_table(
    runs_dir: str | Path, cohorts: Sequence[str], arms: Sequence[str] = (EHM, EHM_INT, EHM_LITE)
) -> pd.DataFrame:
    """The standardised coefficients (log-odds per SD) of every cohort and arm, from the ``metrics.json`` of the runs."""
    rows = []
    for cohort in cohorts:
        for arm in arms:
            res = read_run(runs_dir, cohort, arm)
            if res is None:
                continue
            for c in res["coefficients"]:
                se = c["se_robust"]
                z = c["beta"] / se if se and np.isfinite(se) and se > 0 else float("nan")
                rows.append({"cohort": cohort, "arm": arm, "term": c["term"], "beta": c["beta"], "se_robust": se, "se_model": c["se_model"],
                             "z_robust": z, "kind": "phase" if c["term"].startswith("phase_") else "intercept" if c["term"] == "intercept" else "habitat"})  # fmt: skip
    return pd.DataFrame(rows)


def pooled_coefficients(coefs: pd.DataFrame, *, se_column: str = "se_robust") -> pd.DataFrame:
    """Per arm and habitat term: sign agreement over the cohorts that have it and the DL pooled coefficient with its 95% interval."""
    rows = []
    sub = coefs[coefs["kind"] == "habitat"]
    for (arm, term), g in sub.groupby(["arm", "term"], sort=False):
        pooled = pooled_dl(g["beta"].to_numpy(), g[se_column].to_numpy())
        rows.append(
            {"arm": arm, "term": term, "n_cohorts": int(len(g)), "se_type": se_column, **pooled}
        )
    return pd.DataFrame(rows)


def diagnostic_tables(
    runs_dir: str | Path, cohorts: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(collinearity, single-term coefficients per cohort, single-term pooled)`` from the EHM ``metrics.json`` of the cohorts."""
    coll, single = [], []
    for cohort in cohorts:
        res = read_run(runs_dir, cohort, EHM)
        diag = None if res is None else res.get("diagnostics")
        if not diag:
            continue
        for t in diag["terms"]:
            coll.append({"cohort": cohort, "term": t, "vif": diag["vif"][t], "corr_outside_depth": diag["corr"][t].get("outside_depth", float("nan")),
                         "corr_required_speed": diag["corr"][t].get("required_speed", float("nan"))})  # fmt: skip
            c = diag["single_term"][t]
            single.append(
                {
                    "cohort": cohort,
                    "term": t,
                    "beta": c["beta"],
                    "se_robust": c["se_robust"],
                    "se_model": c["se_model"],
                }
            )
    single = pd.DataFrame(single)
    pooled = []
    if len(single):
        for t, g in single.groupby("term", sort=False):
            pooled.append(
                {
                    "term": t,
                    "n_cohorts": int(len(g)),
                    **pooled_dl(g["beta"].to_numpy(), g["se_robust"].to_numpy()),
                }
            )
    return pd.DataFrame(coll), single, pd.DataFrame(pooled)


def fit_info_table(
    runs_dir: str | Path, cohorts: Sequence[str], arms: Sequence[str] = ARMS
) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        for arm in arms:
            res = read_run(runs_dir, cohort, arm)
            if res is None:
                continue
            m = res["model"]
            rows.append({"cohort": cohort, "arm": arm, "penalty": m["penalty"], "l2_lambda": m["l2_lambda"], "converged": m["converged"],
                         "n_iter": m["n_iter"], "loglik_train": m["loglik_train"], "n_train_rows": res["n_train_rows"],
                         "n_terms": len(m["terms"]), "dropped": ";".join(f"{k}" for k in m["dropped"]),
                         "fallback_reason": m["fallback_reason"]})  # fmt: skip
    return pd.DataFrame(rows)


def report(
    runs_dir: str | Path,
    out_dir: str | Path,
    cohorts: Sequence[str] | None = None,
    *,
    n_boot: int = 2000,
) -> dict:
    """Write the tables of this module's docstring to ``out_dir`` and return the key numbers (also ``summary.json``)."""
    runs_dir, out = Path(runs_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if cohorts is None:
        cohorts = sorted(
            p.name
            for p in runs_dir.iterdir()
            if p.is_dir() and (p / EHM / f"seed{SEED}" / "metrics.json").is_file()
        )
    cohorts = list(cohorts)
    auc = auc_comparison(runs_dir, cohorts)
    share = share_table(runs_dir, cohorts, n_boot=n_boot)
    coefs = coefficient_table(runs_dir, cohorts)
    pooled = pooled_coefficients(coefs)
    pooled_model = pooled_coefficients(coefs, se_column="se_model")
    info = fit_info_table(runs_dir, cohorts)
    coll, single, single_pooled = diagnostic_tables(runs_dir, cohorts)
    vifs = vif_table(runs_dir, cohorts)
    diff = auc_difference_table(runs_dir, cohorts, n_boot=n_boot)
    for name, frame in (("auc_comparison", auc), ("share_of_discrimination", share), ("coefficients_per_cohort", coefs),
                        ("coefficients_pooled", pooled), ("coefficients_pooled_model_se", pooled_model), ("collinearity", coll),
                        ("single_term_coefficients", single), ("single_term_pooled", single_pooled), ("vif", vifs),
                        ("auc_difference_ehm_minus_lite", diff), ("fit_info", info)):  # fmt: skip
        frame.to_csv(out / f"{name}.csv", index=False)
    r_out_speed = (
        coll[(coll["term"] == "required_speed")]["corr_outside_depth"]
        if len(coll)
        else pd.Series(dtype=float)
    )
    mean = auc.groupby("arm")[["auc", "auc_within_phase"]].mean() if len(auc) else pd.DataFrame()
    n_arm = auc.groupby("arm").size().to_dict() if len(auc) else {}
    shares = (
        share[share["cohort"] == "pooled"].set_index(["arm", "metric"])
        if len(share)
        else pd.DataFrame()
    )
    summary = {
        "cohorts": cohorts,
        "n_boot": int(n_boot),
        "mean_auc": {a: {"auc": float(r.auc), "auc_within_phase": float(r.auc_within_phase), "n_cohorts": int(n_arm[a])} for a, r in mean.iterrows()},
        "share_of_discrimination_pooled": {
            f"{a}/{m}": {"share": float(r.share), "ci": [float(r.ci_low), float(r.ci_high)]} for (a, m), r in shares.iterrows()
        },
        "coefficients_pooled": pooled.to_dict("records"),
        "ehm_minus_lite": {
            r.metric: {"difference": float(r.difference), "ci": [float(r.ci_low), float(r.ci_high)]} for r in diff[diff["cohort"] == "pooled"].itertuples()
        } if len(diff) else {},
        "max_vif": {arm: float(g.vif.max()) for arm, g in vifs.groupby("arm")} if len(vifs) else {},
        "single_term_pooled": single_pooled.to_dict("records"),
        "corr_outside_depth_required_speed": (
            {"min": float(r_out_speed.min()), "median": float(r_out_speed.median()), "max": float(r_out_speed.max())} if len(r_out_speed) else None
        ),
        "penalty_fallbacks": [f"{r.cohort}/{r.arm}" for r in info.itertuples() if r.penalty != "none"] if len(info) else [],
        "dropped_terms": {f"{r.cohort}/{r.arm}": r.dropped for r in info.itertuples() if isinstance(r.dropped, str) and r.dropped} if len(info) else {},
        "git": git_info(),
    }  # fmt: skip
    write_json(out / "summary.json", summary)
    return summary


# =========================================================================== RQ1: exposure, movement burden, competition, team size
# RQ1 (exploratory, descriptive): how are spatial exposure, movement burden, local competition density and remaining team size
# associated with next-phase elimination risk, and how consistent is each association across phases and cohorts?
#
# Required speed is NOT used (collinear with exposure); the movement burden is the migration deficit (required minus observed speed,
# r ~ 0.4 with exposure).  Covariates, z-scored on the TRAIN rows of the cohort (rows = HGB-cut rows, labels v2)::
#
#     exposure           outside_depth = max(0, -d), inside_cushion = max(0, d), d = hab_signed_boundary_distance_end_m
#     movement burden    migration_deficit  = hab_migration_deficit_mps
#     local competition  local_crowding     = hab_local_enemy_count_200m
#     control            log_alive_teams    = log(100 * hab_alive_teams)
#     remaining size     team_members       = alive_players_end_norm  (constant in solo cohorts: dropped there)
#
# M-main = phase dummies + the covariates; M-phase = M-main + covariate x phase-group interactions (early = cut phases 1-2 is the
# reference, mid = 3-5, late = 6-7).  The slope of a group is main + interaction with its standard error from the (match-clustered)
# covariance matrix.  Pooling over cohorts: DerSimonian-Laird on the per-cohort coefficients; the test that a covariate's slope is the
# same in the three groups is a multivariate random-effects Wald test on the two interaction coefficients (:func:`mv_random_effects`).
#
# Two scales for the phase-group slopes, both from the one M-phase fit.  ``train_sd`` (the specified standardisation): one SD per
# covariate, from all train rows of the cohort -- the slope is the log-odds per train SD and it is large in a phase group where the
# covariate hardly varies (late zones are small: the spread of the exposure terms is 5-50 times below the all-phase spread), so the
# slopes of different groups cannot be compared by size, and their interval is wide there for the same reason.  ``group_sd``: the same
# slope times SD(covariate in the train rows of the group) / SD(all train rows), i.e. the log-odds per SD of the covariate within that
# phase group (a rescaling of the coefficient, with the matching rescaling of its covariance).  Signs and the test of equal train-SD
# slopes (= equal slopes per unit of the covariate) do not depend on the choice; the test of equal group-SD slopes asks whether the
# standardised effect is the same in the three groups.
RQ1_COVARIATES: tuple[Covariate, ...] = (
    Covariate("outside_depth", "hab_signed_boundary_distance_end_m", "exposure: max(0, -d), depth outside the blue-zone boundary at the cut"),
    Covariate("inside_cushion", "hab_signed_boundary_distance_end_m", "exposure: max(0, d), distance inside the boundary at the cut"),
    Covariate("migration_deficit", "hab_migration_deficit_mps", "movement burden: required minus observed speed (negative = moves faster than needed)"),
    Covariate("local_crowding", "hab_local_enemy_count_200m", "local competition: enemy teams within 200 m at the last snapshot"),
    Covariate("log_alive_teams", "hab_alive_teams", "control: log of the alive teams entering the next phase"),
    Covariate("team_members", "alive_players_end_norm", "remaining team size: event-based alive members at the phase end (count / 4)"),
)  # fmt: skip
RQ1_TERMS: tuple[str, ...] = tuple(c.term for c in RQ1_COVARIATES)
RQ1_CONSTRUCT: dict[str, str] = {
    "outside_depth": "exposure", "inside_cushion": "exposure", "migration_deficit": "movement burden", "local_crowding": "local competition",
    "log_alive_teams": "control", "team_members": "team size",
}  # fmt: skip
RQ1_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("early", (1, 2)),
    ("mid", (3, 4, 5)),
    ("late", (6, 7)),
)
GROUP_NAMES: tuple[str, ...] = tuple(g for g, _ in RQ1_GROUPS)
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("mid-early", "mid", "early"),
    ("late-early", "late", "early"),
    ("late-mid", "late", "mid"),
)
RQ1_ARMS: tuple[str, str] = ("M-main", "M-phase")
SCALES: tuple[str, str] = ("train_sd", "group_sd")
PHYSICAL = "physical"  # the scale label of the E4 fits: the slope per physical unit (RQ1_UNITS), not per SD
GROUP_SD_MIN_RATIO = 1e-9  # a group whose covariate spread is below this share of the train spread has no group-SD slope (NaN)


def phase_group_index(phase: np.ndarray) -> np.ndarray:
    """0 / 1 / 2 for the early / mid / late cut phases (:data:`RQ1_GROUPS`); phases below the first / above the last range join the
    nearest group."""
    bounds = [max(phases) for _, phases in RQ1_GROUPS[:-1]]
    return np.searchsorted(bounds, np.asarray(phase), side="left")


def make_rq1_inputs(rows: H.Rows) -> Inputs:
    """The cut phase, the six RQ1 terms and the match clusters of a split (rows = ``hgb.build_rows(dump, 'HGB-cut')``)."""
    return _inputs(rows, RQ1_TERMS)


@dataclass
class Rq1Model:
    kind: str  # 'main' | 'phase'
    phases: list[int]  # train cut phases; the first one is the reference level of the dummies
    terms: list[
        str
    ]  # design columns: intercept, phase_<k>, the terms in use, then (kind 'phase') <term>:mid and <term>:late
    used: list[str]  # RQ1 terms in the model (a constant one is dropped)
    dropped: dict[str, str]
    mean: dict[str, float]  # train mean / SD of every RQ1 term (all train rows)
    sd: dict[str, float]
    group_ratio: dict[
        str, dict[str, float]
    ]  # term -> group -> SD of the term within the group's train rows / train SD (NaN: no spread)
    fit: LogitFit
    scale: dict[str, float] | None = (
        None  # None: the terms are z-scored (divided by ``sd``); else the divisor of each centred term (physical units, E4)
    )


def design_rq1(model: Rq1Model, inputs: Inputs) -> np.ndarray:
    """``[1 | phase dummies | z-scored terms | (phase kind) z x mid | z x late]`` under the train coding of ``model``."""
    n = len(inputs)
    phase, _ = _nearest_phase(inputs.phase, model.phases)
    cols = [np.ones(n)] + [(phase == k).astype(np.float64) for k in model.phases[1:]]
    div = (
        model.scale if model.scale is not None else model.sd
    )  # SD of the train rows (z-score) or the physical unit width
    z = {t: (inputs.raw[t].to_numpy() - model.mean[t]) / div[t] for t in model.used}
    cols += [z[t] for t in model.used]
    if model.kind == "phase":
        group = phase_group_index(inputs.phase)
        cols += [z[t] * (group == 1) for t in model.used] + [
            z[t] * (group == 2) for t in model.used
        ]
    X = np.column_stack(cols)
    if X.shape[1] != len(model.terms):
        raise AssertionError(f"design has {X.shape[1]} columns for {len(model.terms)} terms")
    return X


def _group_ratio(x: np.ndarray, in_group: np.ndarray, sd_all: float) -> float:
    if not in_group.any():
        return float("nan")
    ratio = float(np.std(x[in_group])) / sd_all
    return ratio if ratio > GROUP_SD_MIN_RATIO else float("nan")


def fit_rq1(
    kind: str, train: Inputs, y: np.ndarray, units: Mapping[str, float] | None = None
) -> Rq1Model:
    """Fit M-main (``kind='main'``) or M-phase (``'phase'``) on the train rows: same fitting and standard errors as the EHM arms.

    The quasi-separation bound (:data:`COEF_LIMIT`) is applied to the coefficients per SD of their design column, so a large slope
    on a column with a small spread (a late-phase interaction) does not send the fit to the ridge fallback.

    ``units`` (E4): ``{term: width of one physical unit in dump units}`` (:data:`RQ1_UNITS`); the terms are then centred at the train mean and
    divided by that width instead of their SD, so every coefficient is the log-odds per physical unit (a covariate that is constant in
    the train rows is still dropped: that test uses the empirical SD).  Predictions and the fit do not change: it is a reparametrisation."""
    if kind not in ("main", "phase"):
        raise ValueError(f"kind must be 'main' or 'phase', got {kind!r}")
    mean = {t: float(train.raw[t].mean()) for t in RQ1_TERMS}
    sd = {t: float(train.raw[t].std(ddof=0)) for t in RQ1_TERMS}
    used = [t for t in RQ1_TERMS if sd[t] > CONSTANT_SD]
    dropped = {t: "constant in the train rows (zero variance)" for t in RQ1_TERMS if t not in used}
    group = phase_group_index(train.phase)
    group_ratio = {
        t: {
            g: _group_ratio(train.raw[t].to_numpy(), group == i, sd[t])
            for i, g in enumerate(GROUP_NAMES)
        }
        for t in used
    }
    phases = sorted(int(k) for k in np.unique(train.phase))
    terms = ["intercept"] + [f"phase_{k}" for k in phases[1:]] + used
    if kind == "phase":
        terms += [f"{t}:mid" for t in used] + [f"{t}:late" for t in used]
    scale = None if units is None else {t: float(units[t]) for t in RQ1_TERMS}
    model = Rq1Model(
        kind, phases, terms, used, dropped, mean, sd, group_ratio, fit=None, scale=scale
    )  # type: ignore[arg-type]
    X = design_rq1(model, train)
    model.fit = fit_logit_auto(X, y, groups=train.match, coef_scale=X.std(axis=0))
    return model


def predict_rq1(model: Rq1Model, inputs: Inputs) -> np.ndarray:
    return _sigmoid(design_rq1(model, inputs) @ model.fit.beta)


def vif_design(model: Rq1Model, train: Inputs) -> dict[str, float]:
    """VIF of the RQ1 terms in the design of an RQ1 model *including the phase dummies*: the diagonal of the inverse correlation matrix
    of all design columns but the intercept (``vif`` leaves the dummies out and only compares the terms with each other).  A term that
    follows the phase, like the log of the alive teams, is inflated here."""
    X = design_rq1(model, train)[:, 1:]
    names = model.terms[1:]
    if X.shape[1] == 1:
        return {names[0]: 1.0}
    diag = np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))
    return {t: float(v) for t, v in zip(names, diag) if t in model.used}


def _unit(n: int, weights: Mapping[int, float]) -> np.ndarray:
    v = np.zeros(n)
    for i, w in weights.items():
        v[i] += w
    return v


def _std_error(variance: float) -> float:
    return float("nan") if np.isnan(variance) else float(np.sqrt(max(variance, 0.0)))


def _combination(fit: LogitFit, L: np.ndarray) -> tuple[float, float, float]:
    """``(estimate, robust SE, model SE)`` of ``L' beta`` from the full covariance matrices."""
    se_r = _std_error(float(L @ fit.cov_robust @ L)) if fit.cov_robust is not None else float("nan")
    return float(L @ fit.beta), se_r, _std_error(float(L @ fit.cov_model @ L))


_ESTIMATE_COLUMNS = [
    "cohort",
    "analysis",
    "covariate",
    "group",
    "scale",
    "estimate",
    "se_robust",
    "se_model",
]
_CONTRAST_COLUMNS = [
    "cohort",
    "covariate",
    "scale",
    "contrast",
    "estimate",
    "se_robust",
    "se_model",
]
_INTERACTION_COLUMNS = [
    "cohort",
    "covariate",
    "scale",
    "d_mid",
    "d_late",
    "s11",
    "s12",
    "s22",
    "wald",
    "p_wald",
]


def rq1_tables(model: Rq1Model, cohort: str, scale: str = "train_sd") -> dict[str, pd.DataFrame]:
    """Per-cohort estimates of a fitted RQ1 model on one scale (see the RQ1 section header).

    ``estimates``: M-main -> the coefficient of every term (group ``all``); M-phase -> the slope of every term in every phase group
    (main for early, main + interaction for mid / late) with robust and model-based standard errors.  ``contrasts`` (M-phase): the
    slope differences mid - early, late - early, late - mid.  ``interactions`` (M-phase): the differences mid - early and late - early
    (the two interaction coefficients on the train-SD scale), their 2 x 2 robust covariance and the Wald test (chi-square, 2 df) that
    the three slopes are equal.  ``scale='group_sd'`` (M-phase only) multiplies the slope of each group by its SD ratio
    (:attr:`Rq1Model.group_ratio`); every combination is formed on the coefficients and their covariance matrix."""
    from scipy import stats

    if scale not in (*SCALES, PHYSICAL):
        raise ValueError(f"scale must be one of {(*SCALES, PHYSICAL)}, got {scale!r}")
    if (scale == PHYSICAL) != (model.scale is not None):
        raise ValueError(
            f"the {PHYSICAL!r} scale belongs to a model fitted with physical units (fit_rq1(..., units=...)), and only to it"
        )
    if model.kind == "main" and scale == "group_sd":
        raise ValueError(
            f"M-main has a single slope per covariate: the {scale!r} scale needs the M-phase model"
        )
    ix = {t: i for i, t in enumerate(model.terms)}
    fit = model.fit
    n = len(fit.beta)
    est, con, inter = [], [], []
    if model.kind == "main":
        for t in model.used:
            e, sr, sm = _combination(fit, _unit(n, {ix[t]: 1.0}))
            est.append(
                {
                    "cohort": cohort,
                    "analysis": "M-main",
                    "covariate": t,
                    "group": "all",
                    "scale": scale,
                    "estimate": e,
                    "se_robust": sr,
                    "se_model": sm,
                }
            )
    else:
        for t in model.used:
            s = {g: model.group_ratio[t][g] if scale == "group_sd" else 1.0 for g in GROUP_NAMES}
            vec = {
                "early": _unit(n, {ix[t]: s["early"]}),
                "mid": _unit(n, {ix[t]: s["mid"], ix[f"{t}:mid"]: s["mid"]}),
                "late": _unit(n, {ix[t]: s["late"], ix[f"{t}:late"]: s["late"]}),
            }
            for g in GROUP_NAMES:
                e, sr, sm = _combination(fit, vec[g])
                est.append(
                    {
                        "cohort": cohort,
                        "analysis": "M-phase",
                        "covariate": t,
                        "group": g,
                        "scale": scale,
                        "estimate": e,
                        "se_robust": sr,
                        "se_model": sm,
                    }
                )
            for name, hi, lo in CONTRASTS:
                e, sr, sm = _combination(fit, vec[hi] - vec[lo])
                con.append(
                    {
                        "cohort": cohort,
                        "covariate": t,
                        "scale": scale,
                        "contrast": name,
                        "estimate": e,
                        "se_robust": sr,
                        "se_model": sm,
                    }
                )
            Lm, Ll = vec["mid"] - vec["early"], vec["late"] - vec["early"]
            d = np.array([Lm @ fit.beta, Ll @ fit.beta])
            V = fit.cov_robust
            S = np.array([[Lm @ V @ Lm, Lm @ V @ Ll], [Ll @ V @ Lm, Ll @ V @ Ll]])
            wald = float("nan")
            if np.isfinite(d).all() and np.isfinite(S).all():
                try:
                    wald = float(d @ np.linalg.solve(S, d))
                except np.linalg.LinAlgError:  # a singular 2 x 2 block: no test
                    pass
            inter.append({"cohort": cohort, "covariate": t, "scale": scale, "d_mid": float(d[0]), "d_late": float(d[1]), "s11": float(S[0, 0]),
                          "s12": float(S[0, 1]), "s22": float(S[1, 1]), "wald": wald,
                          "p_wald": float(stats.chi2.sf(wald, 2)) if np.isfinite(wald) else float("nan")})  # fmt: skip
    return {
        "estimates": pd.DataFrame(est, columns=_ESTIMATE_COLUMNS),
        "contrasts": pd.DataFrame(con, columns=_CONTRAST_COLUMNS),
        "interactions": pd.DataFrame(inter, columns=_INTERACTION_COLUMNS),
    }


def group_ratio_table(model: Rq1Model, cohort: str) -> pd.DataFrame:
    """Per covariate and phase group: the SD of the covariate within the group's train rows over its SD in all train rows (the factor
    that turns a train-SD slope into a group-SD slope)."""
    rows = [
        {"cohort": cohort, "covariate": t, "group": g, "sd_ratio": model.group_ratio[t][g]}
        for t in model.used
        for g in GROUP_NAMES
    ]
    return pd.DataFrame(rows, columns=["cohort", "covariate", "group", "sd_ratio"])


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """``pd.concat`` of the non-empty frames (an empty one carries no rows, only column dtypes); the columns of the first frame if all
    are empty."""
    kept = [f for f in frames if len(f)]
    return pd.concat(kept, ignore_index=True) if kept else frames[0].iloc[0:0].copy()


def mv_random_effects(d: np.ndarray, S: np.ndarray) -> dict:
    """Multivariate random-effects meta-analysis of ``k`` cohort estimates ``d`` (``k x p``) with within-cohort covariances ``S``
    (``k x p x p``) and the tests that the pooled vector is zero.

    The between-cohort covariance ``T`` is the unweighted method of moments, ``(sum (d - mean)(d - mean)' - (1 - 1/k) sum S) / (k - 1)``
    clipped to positive semi-definite; the pooled vector uses the weights ``(S_c + T)^-1`` and the Wald statistic ``theta' V^-1 theta``.
    Three references for "the pooled vector is zero":

    * ``p_chi2``: the Wald statistic against chi-square(p) -- nominal, but ``T`` is estimated, and with ``k`` around ten cohorts it
      over-rejects (simulated size 12-14% at the 5% level);
    * ``p_f`` (the headline): ``wald * (k - p) / (p * (k - 1))`` against F(p, k - p).  When ``T`` dominates the sampling error the Wald
      statistic is Hotelling's ``T^2`` and this reference is exact; when ``T = 0`` it is conservative (needs ``k > p``);
    * ``p_hotelling``: Hotelling's ``T^2`` on the unweighted cohort vectors against F(p, k - p), the assumption-light check (needs ``k > p``).

    A cohort with a non-finite estimate or covariance is dropped."""
    from scipy import stats

    d = np.asarray(d, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    ok = np.isfinite(d).all(axis=1) & np.isfinite(S).all(axis=(1, 2))
    d, S = d[ok], S[ok]
    k, p = d.shape
    nan = float("nan")
    out = {"k": int(k), "p": int(p), "df": int(p), "theta": np.full(p, nan), "cov": np.full((p, p), nan), "se": np.full(p, nan),
           "T": np.full((p, p), nan), "wald": nan, "p_chi2": nan, "p_f": nan, "hotelling_t2": nan, "p_hotelling": nan}  # fmt: skip
    if k < 2:
        return out
    dbar = d.mean(axis=0)
    dev = d - dbar
    T = (dev.T @ dev - (1.0 - 1.0 / k) * S.sum(axis=0)) / (k - 1)
    T = (T + T.T) / 2.0
    vals, vecs = np.linalg.eigh(T)
    T = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
    W = np.linalg.inv(S + T[None, :, :])
    Wsum = W.sum(axis=0)
    V = np.linalg.inv(Wsum)
    theta = V @ np.einsum("kij,kj->i", W, d)
    wald = float(theta @ Wsum @ theta)
    out.update(
        theta=theta,
        cov=V,
        se=np.sqrt(np.diag(V)),
        T=T,
        wald=wald,
        p_chi2=float(stats.chi2.sf(wald, p)),
    )
    if k > p:
        out["p_f"] = float(stats.f.sf(wald * (k - p) / (p * (k - 1)), p, k - p))
        try:
            t2 = float(k * dbar @ np.linalg.solve(dev.T @ dev / (k - 1), dbar))
            f = t2 * (k - p) / (p * (k - 1))
            out.update(hotelling_t2=t2, p_hotelling=float(stats.f.sf(f, p, k - p)))
        except np.linalg.LinAlgError:
            pass
    return out


def _scales_of(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(frame["scale"])) if "scale" in frame.columns else ["train_sd"]


def _scale_part(frame: pd.DataFrame, scale: str) -> pd.DataFrame:
    return frame[frame["scale"] == scale] if "scale" in frame.columns else frame


def _pooled_row(
    analysis: str, covariate: str, group: str, scale: str, g: pd.DataFrame
) -> dict | None:
    pooled = pooled_dl(g["estimate"].to_numpy(), g["se_robust"].to_numpy())
    if not pooled.get("k"):
        return None
    return {"analysis": analysis, "covariate": covariate, "construct": RQ1_CONSTRUCT[covariate], "group": group, "scale": scale, "k": pooled["k"],
            "n_positive": pooled["n_positive"], "n_negative": pooled["n_negative"], "same_sign_as_pooled": pooled["same_sign_as_pooled"],
            "sign_agreement": f"{pooled['same_sign_as_pooled']}/{pooled['k']}", "theta": pooled["theta"], "se": pooled["se"],
            "ci95_low": pooled["ci95_low"], "ci95_high": pooled["ci95_high"], "tau2": pooled["tau2"], "I2": pooled["I2"], "p": pooled["p"]}  # fmt: skip


_TIDY_COLUMNS = ["analysis", "covariate", "construct", "group", "scale", "k", "n_positive", "n_negative", "same_sign_as_pooled", "sign_agreement",
                 "theta", "se", "ci95_low", "ci95_high", "tau2", "I2", "p"]  # fmt: skip


def rq1_pool_main(estimates: pd.DataFrame) -> pd.DataFrame:
    """M-main: per covariate the DL pooled standardised coefficient (``theta`` with its 95% interval), ``tau2``, ``I2`` and the sign
    agreement ``n/k`` over the cohorts that have the covariate."""
    sub = estimates[estimates["analysis"] == "M-main"]
    rows = [
        r
        for t in RQ1_TERMS
        if len(g := sub[sub["covariate"] == t])
        and (r := _pooled_row("M-main", t, "all", "train_sd", g)) is not None
    ]
    return pd.DataFrame(rows, columns=_TIDY_COLUMNS)


def rq1_pool_groups(estimates: pd.DataFrame) -> pd.DataFrame:
    """M-phase: per scale, covariate and phase group the DL pooled slope with its 95% interval and the sign agreement."""
    sub = estimates[estimates["analysis"] == "M-phase"]
    rows = []
    for scale in _scales_of(sub):
        part = _scale_part(sub, scale)
        for t in RQ1_TERMS:
            for g in GROUP_NAMES:
                cell = part[(part["covariate"] == t) & (part["group"] == g)]
                if len(cell) and (r := _pooled_row("M-phase", t, g, scale, cell)) is not None:
                    rows.append(r)
    return pd.DataFrame(rows, columns=_TIDY_COLUMNS)


def rq1_equality(interactions: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """Per scale and covariate: the test that the slope is the same in the early, mid and late phases, pooled over the cohorts.

    ``wald``: the multivariate random-effects Wald statistic of the two group differences (:func:`mv_random_effects`) with
    ``p_equal`` (the F reference F(2, k - 2), the headline), ``p_chi2`` (the nominal chi-square(2) reference, liberal at k near ten) and
    ``p_hotelling`` (Hotelling's T^2 check); ``diff_*``: the DL pooled slope differences with 95% intervals; ``tau_*``: the
    between-cohort SD of the two differences and their correlation; ``n_cohorts_p05``: cohorts whose own Wald test has p < 0.05."""
    rows = []
    for scale in _scales_of(interactions):
        inter, con = _scale_part(interactions, scale), _scale_part(contrasts, scale)
        for t in RQ1_TERMS:
            x = inter[inter["covariate"] == t]
            if x.empty:
                continue
            S = np.array([[[a, b], [b, c]] for a, b, c in zip(x["s11"], x["s12"], x["s22"])])
            mv = mv_random_effects(x[["d_mid", "d_late"]].to_numpy(), S)
            row = {"scale": scale, "covariate": t, "construct": RQ1_CONSTRUCT[t], "k": mv["k"]}
            for name, _, _ in CONTRASTS:
                c = con[(con["covariate"] == t) & (con["contrast"] == name)]
                pooled = pooled_dl(c["estimate"].to_numpy(), c["se_robust"].to_numpy())
                key = "diff_" + name.replace("-", "_")
                row.update(
                    {
                        key: pooled.get("theta", np.nan),
                        key + "_low": pooled.get("ci95_low", np.nan),
                        key + "_high": pooled.get("ci95_high", np.nan),
                    }
                )
            tau = np.sqrt(np.clip(np.diag(mv["T"]), 0.0, None))
            corr = float(mv["T"][0, 1] / (tau[0] * tau[1])) if np.all(tau > 0) else float("nan")
            row.update({"wald": mv["wald"], "df": mv["df"], "p_equal": mv["p_f"], "p_chi2": mv["p_chi2"], "hotelling_t2": mv["hotelling_t2"],
                        "p_hotelling": mv["p_hotelling"], "tau_mid": float(tau[0]), "tau_late": float(tau[1]), "tau_corr": corr,
                        "n_cohorts_p05": int((x["p_wald"] < 0.05).sum())})  # fmt: skip
            rows.append(row)
    return pd.DataFrame(rows)


def rq1_tidy(main: pd.DataFrame, groups: pd.DataFrame, equality: pd.DataFrame) -> pd.DataFrame:
    """The pooled results in one long table: per covariate the M-main row (group ``all``) and the M-phase rows of each scale, the latter
    carrying the equality test of their covariate and scale."""
    eq = equality.set_index(["scale", "covariate"]) if len(equality) else pd.DataFrame()
    frames = []
    for t in RQ1_TERMS:
        frames.append(main[main["covariate"] == t])
        for scale in _scales_of(groups):
            part = _scale_part(groups, scale)
            frames.append(part[part["covariate"] == t])
    tidy = _concat(frames) if frames else pd.DataFrame(columns=_TIDY_COLUMNS)
    for col, src in (
        ("equality_wald", "wald"),
        ("equality_p", "p_equal"),
        ("equality_p_chi2", "p_chi2"),
        ("equality_p_hotelling", "p_hotelling"),
    ):
        tidy[col] = [
            float(eq.loc[(sc, c), src])
            if (a == "M-phase" and (sc, c) in eq.index)
            else float("nan")
            for a, sc, c in zip(tidy["analysis"], tidy["scale"], tidy["covariate"])
        ]
    return tidy


# ---- the forest figure
RQ1_ROW_LABELS = {
    "all": "All phases",
    "early": "Early (1-2)",
    "mid": "Mid (3-5)",
    "late": "Late (6-7)",
}
RQ1_ROW_STYLE = {  # group -> (colour, marker)
    "all": ("#0b0b0b", "D"), "early": ("#2a78d6", "o"), "mid": ("#d9822b", "s"), "late": ("#7a3b99", "^"),
}  # fmt: skip
RQ1_COVARIATE_LABELS = {
    "outside_depth": "Outside depth", "inside_cushion": "Inside cushion", "migration_deficit": "Migration deficit",
    "local_crowding": "Enemy teams within 200 m", "log_alive_teams": "Log alive teams", "team_members": "Team members alive",
}  # fmt: skip
RQ1_X_LABELS = {
    "group_sd": "Log-odds per SD of the covariate (within the phase group)\nDL pooled over cohorts, 95% CI",
    "train_sd": "Log-odds per train SD of the covariate (all phases)\nDL pooled over cohorts, 95% CI",
}  # fmt: skip


def rq1_forest_data(tidy: pd.DataFrame, scale: str = "group_sd") -> pd.DataFrame:
    """The numbers behind a forest: per covariate the row ``all`` (M-main, train SD) and the rows ``early`` / ``mid`` / ``late`` (M-phase,
    on ``scale``) with a row position ``y`` that grows downwards and leaves a gap between covariates."""
    rows, y0 = [], 0.0
    for t in RQ1_TERMS:
        part = tidy[tidy["covariate"] == t]
        picks = []
        for g in ("all", *GROUP_NAMES):
            if g == "all":
                hit = part[(part["analysis"] == "M-main") & (part["group"] == "all")]
            else:
                hit = part[
                    (part["analysis"] == "M-phase")
                    & (part["group"] == g)
                    & (part["scale"] == scale)
                ]
            if len(hit):
                picks.append((g, hit.iloc[0]))
        for j, (g, r) in enumerate(picks):
            rows.append({"figure_scale": scale, "covariate": t, "construct": RQ1_CONSTRUCT[t], "group": g, "y": y0 + j, "theta": float(r["theta"]),
                         "ci95_low": float(r["ci95_low"]), "ci95_high": float(r["ci95_high"]), "k": int(r["k"]),
                         "same_sign_as_pooled": int(r["same_sign_as_pooled"]), "sign_agreement": r["sign_agreement"], "analysis": r["analysis"],
                         "row_label": RQ1_ROW_LABELS[g]})  # fmt: skip
        if picks:
            y0 += len(picks) + 1
    return pd.DataFrame(rows)


def build_rq1_forest(data: pd.DataFrame, scale: str = "group_sd"):
    """One compact forest: rows = covariates x (all phases, early, mid, late), the pooled standardised coefficient with its 95%
    interval, the sign agreement n/k at the right edge; a bold header above each covariate."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from pubg_lab.survival.ptr import aggregate as A

    fig = Figure(figsize=(3.5, 4.9), dpi=300, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.215, 0.125, 0.625, 0.815])
    ymax = float(data["y"].max()) if len(data) else 1.0
    lo = min([0.0, *data["ci95_low"].tolist()]) if len(data) else -1.0
    hi = max([0.0, *data["ci95_high"].tolist()]) if len(data) else 1.0
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(ymax + 0.8, -1.35)
    ax.axvline(0.0, color=A.MUTED, lw=0.7, zorder=1)
    ax.xaxis.grid(True, color=A.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for r in data.itertuples(index=False):
        color, marker = RQ1_ROW_STYLE[r.group]
        ax.hlines(r.y, r.ci95_low, r.ci95_high, color=color, lw=1.3, zorder=2)
        ax.plot(
            [r.theta],
            [r.y],
            marker=marker,
            ls="none",
            ms=3.6 if r.group != "all" else 4.0,
            color=color,
            mec="white",
            mew=0.4,
            zorder=3,
        )
        ax.text(
            1.02,
            r.y,
            r.sign_agreement,
            transform=ax.get_yaxis_transform(),
            fontsize=4.6,
            color=A.INK2,
            va="center",
            ha="left",
        )
    ax.set_yticks(data["y"].tolist())
    ax.set_yticklabels(data["row_label"].tolist(), fontsize=4.8, color=A.INK2)
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    first_header = None
    for t in dict.fromkeys(data["covariate"]):
        y_head = float(data[data["covariate"] == t]["y"].min()) - 0.95
        first_header = y_head if first_header is None else first_header
        head = ax.text(-0.29, y_head, RQ1_COVARIATE_LABELS[t], transform=ax.get_yaxis_transform(), fontsize=5.4, fontweight="bold", color=A.INK,
                       ha="left", va="center", clip_on=False)  # fmt: skip
        x_end = float(to_axes.transform((head.get_window_extent(renderer).x1, 0.0))[0])
        ax.text(
            x_end + 0.02,
            y_head,
            RQ1_CONSTRUCT[t],
            transform=ax.get_yaxis_transform(),
            fontsize=4.6,
            color=A.MUTED,
            ha="left",
            va="center",
            clip_on=False,
        )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(A.BASELINE)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(
        axis="x", labelsize=5.2, length=2, width=0.6, color=A.BASELINE, labelcolor=A.INK2, pad=1.5
    )
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel(RQ1_X_LABELS[scale], fontsize=5.4, color=A.INK2, labelpad=2, linespacing=1.3)
    ax.text(
        1.02,
        first_header if first_header is not None else 0.0,
        "n/k",
        transform=ax.get_yaxis_transform(),
        fontsize=4.6,
        color=A.INK2,
        ha="left",
        va="center",
    )
    handles = [
        Line2D(
            [],
            [],
            marker=RQ1_ROW_STYLE[g][1],
            color=RQ1_ROW_STYLE[g][0],
            lw=1.3,
            ms=3.6,
            label=RQ1_ROW_LABELS[g],
        )
        for g in ("all", *GROUP_NAMES)
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=5.0,
        bbox_to_anchor=(0.52, 0.0),
        handlelength=1.6,
        columnspacing=1.0,
    )
    for t in leg.get_texts():
        t.set_color(A.INK2)
    fig.text(
        0.02,
        0.985,
        "RQ1: association with next-phase elimination (exploratory)",
        fontsize=5.8,
        color=A.INK,
        fontweight="bold",
        ha="left",
        va="top",
    )
    return fig


def draw_rq1_forest(data: pd.DataFrame, stem: str | Path, scale: str = "group_sd") -> None:
    """Write ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` (TrueType fonts)."""
    from matplotlib import rc_context

    stem = Path(stem)
    fig = build_rq1_forest(data, scale)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


# ---- one cohort and the whole analysis
def _auc_rows(cohort: str, arm: str, frame: pd.DataFrame) -> dict:
    h = M.hazard_metrics(frame["y"].to_numpy(), frame["p_hazard"].to_numpy())
    w = M.within_phase_auc(frame)
    return {"cohort": cohort, "arm": arm, "n_rows": h["n"], "auc": h["auc"], "auc_within_phase": w["auc_within_phase"], "n_phases_within": w["n_phases"],
            "ap": h["ap"], "nll": h["nll"], "brier": h["brier"]}  # fmt: skip


def rq1_cohort(
    dump_dir: str | Path,
    cohort: str,
    runs_dir: str | Path | None = None,
    *,
    labels: str = D.DEFAULT_LABELS,
    match_types: tuple[str, ...] | None = D.DEFAULT_MATCH_TYPES,
) -> dict:
    """Fit M-main and M-phase on the train rows of a cohort and collect everything the pooled analysis needs: the per-cohort tables
    (train-SD scale for both models, group-SD scale for M-phase), the VIF of M-main, the fit information and the test AUC
    (``frames`` = the test prediction frames, in memory)."""
    splits = D.load_cohort(dump_dir, cohort, labels=labels, match_types=match_types)
    rows = {
        s: H.build_rows(splits[s], "HGB-cut", labels) for s in splits
    }  # the very rows of HGB-cut
    y = rows["train"].y
    if len(np.unique(y)) < 2:
        raise ValueError(f"{cohort}: the train rows hold a single hazard class")
    inputs = {s: make_rq1_inputs(r) for s, r in rows.items()}
    models = {
        "M-main": fit_rq1("main", inputs["train"], y),
        "M-phase": fit_rq1("phase", inputs["train"], y),
    }
    tables = [
        rq1_tables(models["M-main"], cohort),
        rq1_tables(models["M-phase"], cohort),
        rq1_tables(models["M-phase"], cohort, scale="group_sd"),
    ]
    frames = {
        a: H._prediction_frame(rows["test"], predict_rq1(m, inputs["test"]))
        for a, m in models.items()
    }
    auc = [_auc_rows(cohort, a, f) for a, f in frames.items()]
    if runs_dir is not None:
        ref = auc_comparison(runs_dir, [cohort], arms=(EHM_LITE, EHM))
        auc += ref.to_dict("records")
    fit_info = [
        {"cohort": cohort, "model": a, "penalty": m.fit.penalty, "l2_lambda": m.fit.l2, "converged": m.fit.converged, "n_iter": m.fit.n_iter,
         "loglik_train": m.fit.loglik, "n_train_rows": int(len(y)), "n_terms": len(m.terms), "dropped": ";".join(m.dropped),
         "fallback_reason": m.fit.fallback_reason}
        for a, m in models.items()
    ]  # fmt: skip
    return {
        "cohort": cohort,
        "estimates": _concat([t["estimates"] for t in tables]),
        "contrasts": _concat([t["contrasts"] for t in tables]),
        "interactions": _concat([t["interactions"] for t in tables]),
        "group_ratio": group_ratio_table(models["M-phase"], cohort),
        "vif": vif(inputs["train"], models["M-main"].used),
        "vif_design": vif_design(models["M-main"], inputs["train"]),
        "fit_info": fit_info,
        "auc": auc,
        "frames": frames,
    }


def _vif_max(vif_rows: pd.DataFrame) -> pd.DataFrame:
    """Per cohort the largest VIF among the terms (``max_vif``, ``term``) and in the design with the phase dummies
    (``max_vif_with_phase``, ``term_with_phase``)."""
    cols = ["cohort", "max_vif", "term", "max_vif_with_phase", "term_with_phase"]
    if vif_rows.empty:
        return pd.DataFrame(columns=cols)
    a = vif_rows.loc[vif_rows.groupby("cohort")["vif"].idxmax()].set_index("cohort")
    b = vif_rows.loc[vif_rows.groupby("cohort")["vif_with_phase"].idxmax()].set_index("cohort")
    out = pd.DataFrame(
        {
            "max_vif": a["vif"],
            "term": a["term"],
            "max_vif_with_phase": b["vif_with_phase"],
            "term_with_phase": b["term"],
        }
    )
    return out.reset_index()[cols]


def rq1_run(
    dump_dir: str | Path,
    cohorts: Sequence[str],
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    labels: str = D.DEFAULT_LABELS,
    n_boot: int = 2000,
) -> dict:
    """RQ1 over the cohorts: fit, pool, test, write the tables and the forest figures to ``out_dir``, return the key numbers.

    Figures: ``rq1_forest`` (slopes per SD within the phase group: comparable across groups) and ``rq1_forest_train_sd`` (slopes per
    train SD: the stated standardisation; the late-phase slopes of the exposure terms are inflated by their small late-phase spread)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohorts = list(cohorts)
    results = [rq1_cohort(dump_dir, c, runs_dir, labels=labels) for c in cohorts]
    estimates = _concat([r["estimates"] for r in results])
    contrasts = _concat([r["contrasts"] for r in results])
    interactions = _concat([r["interactions"] for r in results])
    main, groups = rq1_pool_main(estimates), rq1_pool_groups(estimates)
    equality = rq1_equality(interactions, contrasts)
    tidy = rq1_tidy(main, groups, equality)
    forests = {sc: rq1_forest_data(tidy, sc) for sc in SCALES}
    vif_rows = pd.DataFrame(
        [{"cohort": r["cohort"], "term": t, "vif": v, "vif_with_phase": r["vif_design"][t]} for r in results for t, v in r["vif"].items()],
        columns=["cohort", "term", "vif", "vif_with_phase"],
    )  # fmt: skip
    vmax = _vif_max(vif_rows)
    auc = pd.DataFrame([row for r in results for row in r["auc"]])
    order = {a: i for i, a in enumerate(["M-main", "M-phase", "EHM-lite", "EHM"])}
    auc = auc.sort_values(
        ["cohort", "arm"],
        key=lambda col: col.map(order) if col.name == "arm" else col,
        kind="mergesort",
    )
    frames = {(r["cohort"], a): f for r in results for a, f in r["frames"].items()}
    diff = auc_difference_table(
        runs_dir, cohorts, a="M-main", b=EHM_LITE, n_boot=n_boot, frames_in_memory=frames
    )
    fit_info = pd.DataFrame([row for r in results for row in r["fit_info"]])
    ratio = _concat([r["group_ratio"] for r in results])
    for name, frame in (("rq1_tidy", tidy), ("rq1_cohort_estimates", estimates), ("rq1_cohort_contrasts", contrasts), ("rq1_interactions_per_cohort", interactions),
                        ("rq1_slope_equality", equality), ("rq1_group_sd_ratio", ratio), ("rq1_vif", vif_rows), ("rq1_vif_max", vmax), ("rq1_auc", auc),
                        ("rq1_auc_difference", diff), ("rq1_fit_info", fit_info), ("rq1_forest_data", _concat(list(forests.values())))):  # fmt: skip
        frame.to_csv(out / f"{name}.csv", index=False)
    draw_rq1_forest(forests["group_sd"], out / "rq1_forest", "group_sd")
    draw_rq1_forest(forests["train_sd"], out / "rq1_forest_train_sd", "train_sd")
    mean_auc = auc.groupby("arm")[["auc", "auc_within_phase"]].mean()
    summary = {
        "cohorts": cohorts,
        "n_boot": int(n_boot),
        "terms": list(RQ1_TERMS),
        "phase_groups": {g: list(ph) for g, ph in RQ1_GROUPS},
        "scales": list(SCALES),
        "main_pooled": main.to_dict("records"),
        "group_pooled": groups.to_dict("records"),
        "slope_equality": equality.to_dict("records"),
        "max_vif_per_cohort": {r.cohort: {"max_vif": float(r.max_vif), "term": r.term, "max_vif_with_phase": float(r.max_vif_with_phase),
                                          "term_with_phase": r.term_with_phase} for r in vmax.itertuples()},  # fmt: skip
        "auc_mean": {a: {"auc": float(v.auc), "auc_within_phase": float(v.auc_within_phase)} for a, v in mean_auc.iterrows()},
        "auc_main_minus_ehm_lite": {r.metric: {"difference": float(r.difference), "ci": [float(r.ci_low), float(r.ci_high)]} for r in diff[diff["cohort"] == "pooled"].itertuples()},
        "penalty_fallbacks": [f"{r.cohort}/{r.model}" for r in fit_info.itertuples() if r.penalty != "none"],
        "dropped_terms": {f"{r.cohort}/{r.model}": r.dropped for r in fit_info.itertuples() if isinstance(r.dropped, str) and r.dropped},
        "git": git_info(),
    }  # fmt: skip
    write_json(out / "rq1_summary.json", summary)
    return summary


# =========================================================================== RQ1 in physical units (supplementary experiment E4)
# The RQ1 models of the section above z-score every covariate on the train rows of its cohort, so a coefficient is "log-odds per SD of THIS
# cohort" and the cohorts' coefficients are on different scales (the spread of the exposure terms is 2 m-scale in a small map and several km in
# a large one).  E4 refits M-main and M-phase with the covariates in PHYSICAL units, the same for every cohort:
#
#     outside depth, inside cushion   per 100 m
#     migration deficit               per 1 m/s
#     local crowding                  per 1 enemy team within 200 m
#     remaining members               per 1 teammate
#     log alive teams                 unchanged (the natural log of the alive teams: per 1 log unit)
#
# The dump stores every habitat column divided by a fixed constant (``stdb/queries.py``: distances / 8160 m, speeds / 40 m/s, enemy counts / 20,
# alive players / 4; ``hab_alive_teams`` / 100 is undone by the log term already), so a physical unit is a fixed number of dump units
# (:data:`RQ1_UNITS`); the count variables must come out as integers after the conversion, which is checked.  The covariates are centred at the train
# mean (a shift that the phase dummies absorb: slopes, standard errors and predictions are unaffected) and divided by the unit instead of the SD, so
# the fit is a reparametrisation of the per-SD fit: ``beta_physical = beta_per_SD * unit / SD`` coefficient by coefficient (checked per cohort against a
# fresh per-SD fit).  What changes is the pooling: the DerSimonian-Laird weights, ``tau^2``, ``I^2`` and the pooled coefficient are now on one common scale.
#
# Average marginal effect (AME) on the probability scale: for covariate ``j`` and phase group ``g``, ``AME = mean_i [p_i (1 - p_i)] * slope_jg`` over the
# train rows ``i`` of the group (the estimation sample; ``slope_jg`` = the main coefficient for the early group, plus the interaction for mid / late in
# M-phase; the single coefficient in M-main), per physical unit of ``j``; its standard error is the delta method with the match-clustered covariance
# matrix of the fit (gradient ``mean_i [p_i (1 - p_i) L + slope_jg p_i (1 - p_i)(1 - 2 p_i) x_i]``, ``L`` the coefficient selector).  The discrete variant
# ``mean_i [p(x_i + 1 unit) - p(x_i)]`` is reported next to it.  Pooling over cohorts: DerSimonian-Laird on the per-cohort AMEs with those standard errors.
ARENA_SIZE_M = 8160.0  # stdb/rq1_feature_etl.ARENA_SIZE_M: the dump divides the distances by it
SPEED_SCALE_MPS = 40.0  # ... the speeds by 40 m/s
ENEMY_COUNT_SCALE = 20.0  # ... the enemy counts by 20
ALIVE_PLAYERS_SCALE = 4.0  # ... the alive players by 4
RQ1_UNITS: dict[str, tuple[float, str]] = {  # term -> (width of ONE physical unit in dump units, unit label)
    "outside_depth": (100.0 / ARENA_SIZE_M, "per 100 m"),
    "inside_cushion": (100.0 / ARENA_SIZE_M, "per 100 m"),
    "migration_deficit": (1.0 / SPEED_SCALE_MPS, "per 1 m/s"),
    "local_crowding": (1.0 / ENEMY_COUNT_SCALE, "per 1 enemy team within 200 m"),
    "log_alive_teams": (1.0, "per 1 log unit (unchanged)"),
    "team_members": (1.0 / ALIVE_PLAYERS_SCALE, "per 1 teammate"),
}  # fmt: skip
INTEGER_TERMS: dict[str, float] = {
    "local_crowding": ENEMY_COUNT_SCALE,
    "team_members": ALIVE_PLAYERS_SCALE,
}  # dump value * factor must be an integer
INTEGER_TOL = 1e-3


def rq1_units() -> dict[str, float]:
    """``{term: unit width in dump units}`` of :data:`RQ1_UNITS` (what :func:`fit_rq1` takes)."""
    return {t: RQ1_UNITS[t][0] for t in RQ1_TERMS}


def check_units(inputs: Inputs) -> dict[str, float]:
    """The dump scaling constants of :data:`RQ1_UNITS` hold: the count variables are integers after the conversion (largest deviation per term
    is returned; a deviation above :data:`INTEGER_TOL` raises), and the alive-team log decodes to whole teams."""
    dev = {}
    for t, factor in INTEGER_TERMS.items():
        if t in inputs.raw.columns:
            v = inputs.raw[t].to_numpy() * factor
            dev[t] = float(np.abs(v - np.round(v)).max())
    dev["log_alive_teams"] = float(
        np.abs(
            np.exp(inputs.raw["log_alive_teams"].to_numpy())
            - np.round(np.exp(inputs.raw["log_alive_teams"].to_numpy()))
        ).max()
    )
    bad = {t: d for t, d in dev.items() if d > INTEGER_TOL}
    if bad:
        raise ValueError(
            f"the dump scaling constants of RQ1_UNITS do not hold (non-integer counts after the conversion): {bad}"
        )
    return dev


_AME_COLUMNS = [
    "cohort",
    "analysis",
    "covariate",
    "group",
    "unit",
    "n_rows",
    "mean_p",
    "group_sd",
    "slope",
    "ame",
    "se",
    "ame_disc",
    "se_disc",
]


def ame_table(model: Rq1Model, inputs: Inputs, cohort: str, analysis: str) -> pd.DataFrame:
    """Average marginal effects on the probability scale per covariate and phase group over the rows ``inputs`` (see the E4 section header).

    ``ame`` / ``se``: derivative form with the delta-method standard error (match-clustered covariance of the fit); ``ame_disc`` / ``se_disc``: the
    effect of ``+1`` physical unit averaged over the rows.  Every figure is per physical unit when ``model`` was fitted with units.  ``group_sd``: the SD of
    the covariate within the rows of the group, in the same units: where it is far below 1, one unit is outside the support of the data (a late-phase
    exposure term: the zone is small), the derivative and the +1-unit form disagree, and the per-unit figure should not be read as a realistic change."""
    X = design_rq1(model, inputs)
    fit = model.fit
    if fit.cov_robust is None:
        raise ValueError("the AME standard errors need the match-clustered covariance of the fit")
    beta, V = fit.beta, fit.cov_robust
    eta = X @ beta
    p = _sigmoid(eta)
    dens = p * (1.0 - p)
    group = phase_group_index(inputs.phase)
    ix = {t: i for i, t in enumerate(model.terms)}
    unit = {t: RQ1_UNITS[t][1] for t in RQ1_TERMS}
    div = model.scale if model.scale is not None else model.sd
    rows = []
    for t in model.used:
        for gi, g in enumerate(GROUP_NAMES):
            sel = group == gi
            if not sel.any():
                continue
            L = np.zeros(len(beta))
            L[ix[t]] = 1.0
            if model.kind == "phase" and g != "early":
                L[ix[f"{t}:{g}"]] = 1.0
            slope = float(L @ beta)
            Xg, pg, dg = X[sel], p[sel], dens[sel]
            grad = L * dg.mean() + slope * (Xg * (dg * (1.0 - 2.0 * pg))[:, None]).mean(axis=0)
            p_up = _sigmoid(
                eta[sel] + slope
            )  # +1 unit of the covariate moves the linear predictor of a row of this group by the group's slope
            grad_d = ((p_up * (1.0 - p_up))[:, None] * (Xg + L[None, :]) - dg[:, None] * Xg).mean(
                axis=0
            )
            rows.append({
                "cohort": cohort, "analysis": analysis, "covariate": t, "group": g, "unit": unit[t], "n_rows": int(sel.sum()), "mean_p": float(pg.mean()),
                "group_sd": float(inputs.raw[t].to_numpy()[sel].std() / div[t]), "slope": slope, "ame": float(dg.mean() * slope), "se": _std_error(float(grad @ V @ grad)), "ame_disc": float((p_up - pg).mean()),
                "se_disc": _std_error(float(grad_d @ V @ grad_d)),
            })  # fmt: skip
    return pd.DataFrame(rows, columns=_AME_COLUMNS)


def ame_pooled(ame: pd.DataFrame) -> pd.DataFrame:
    """DerSimonian-Laird pooled AME per analysis, covariate and phase group (derivative form, and the discrete +1 unit form as ``*_disc``), with the
    sign agreement of the cohorts, ``I^2`` and the unweighted mean over cohorts; also in percentage points (``*_pp``)."""
    rows = []
    for (analysis, t, g), part in ame.groupby(["analysis", "covariate", "group"], sort=False):
        d = pooled_dl(part["ame"].to_numpy(), part["se"].to_numpy())
        dd = pooled_dl(part["ame_disc"].to_numpy(), part["se_disc"].to_numpy())
        if not d.get("k"):
            continue
        rows.append({
            "analysis": analysis, "covariate": t, "construct": RQ1_CONSTRUCT[t], "group": g, "unit": part["unit"].iloc[0], "k": d["k"],
            "n_positive": d["n_positive"], "n_negative": d["n_negative"], "sign_agreement": f"{d['same_sign_as_pooled']}/{d['k']}",
            "theta": d["theta"], "se": d["se"], "ci95_low": d["ci95_low"], "ci95_high": d["ci95_high"], "tau2": d["tau2"], "I2": d["I2"], "p": d["p"],
            "theta_pp": 100 * d["theta"], "ci95_low_pp": 100 * d["ci95_low"], "ci95_high_pp": 100 * d["ci95_high"],
            "theta_disc": dd["theta"], "ci95_low_disc": dd["ci95_low"], "ci95_high_disc": dd["ci95_high"], "I2_disc": dd["I2"],
            "theta_disc_pp": 100 * dd["theta"], "mean_cohort_ame": float(part["ame"].mean()), "mean_p": float(part["mean_p"].mean()),
            "median_group_sd": float(part["group_sd"].median()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def _units_rows(cohort: str, model: Rq1Model) -> list[dict]:
    return [
        {"cohort": cohort, "covariate": t, "construct": RQ1_CONSTRUCT[t], "unit": RQ1_UNITS[t][1], "unit_width_dump": RQ1_UNITS[t][0],
         "train_mean_physical": model.mean[t] / RQ1_UNITS[t][0], "train_sd_physical": model.sd[t] / RQ1_UNITS[t][0], "train_sd_dump": model.sd[t]}
        for t in model.used
    ]  # fmt: skip


def _equivalence_rows(cohort: str, analysis: str, phys: Rq1Model, sd_model: Rq1Model) -> list[dict]:
    """``beta_physical * SD / unit = beta_per_SD`` for every covariate coefficient (and the SEs): the fits are reparametrisations of one another."""
    rows = []
    cov = [
        (i, t, base)
        for i, t in enumerate(phys.terms)
        for base in [t.split(":")[0]]
        if base in phys.used
    ]
    for i, term, base in cov:
        k = phys.sd[base] / phys.scale[base]
        for kind, a, b in (
            ("beta", phys.fit.beta[i] * k, sd_model.fit.beta[i]),
            ("se_robust", phys.fit.se_robust[i] * k, sd_model.fit.se_robust[i]),
        ):
            rows.append({"cohort": cohort, "analysis": analysis, "term": term, "quantity": kind, "converted_from_physical": float(a), "per_sd_fit": float(b),
                         "abs_diff": float(abs(a - b)), "rel_diff": float(abs(a - b) / max(abs(b), 1e-12))})  # fmt: skip
    for i, term in enumerate(
        phys.terms
    ):  # intercept and phase dummies are untouched by the rescaling (same centring)
        if term == "intercept" or term.startswith("phase_"):
            a, b = phys.fit.beta[i], sd_model.fit.beta[i]
            rows.append({"cohort": cohort, "analysis": analysis, "term": term, "quantity": "beta", "converted_from_physical": float(a), "per_sd_fit": float(b),
                         "abs_diff": float(abs(a - b)), "rel_diff": float(abs(a - b) / max(abs(b), 1e-12))})  # fmt: skip
    return rows


def rq1_physical_cohort(dump_dir: str | Path, cohort: str, *, labels: str = D.DEFAULT_LABELS,
                        match_types: tuple[str, ...] | None = D.DEFAULT_MATCH_TYPES) -> dict:  # fmt: skip
    """E4 for one cohort: fit M-main and M-phase in physical units on the train rows of HGB-cut (labels v2), the per-cohort estimates, the AMEs, the
    per-SD reference fits for the equivalence check, the fit information."""
    splits = D.load_cohort(dump_dir, cohort, labels=labels, match_types=match_types)
    rows = H.build_rows(
        splits["train"], "HGB-cut", labels
    )  # the very rows of HGB-cut (only the estimation sample is needed)
    y = rows.y
    if len(np.unique(y)) < 2:
        raise ValueError(f"{cohort}: the train rows hold a single hazard class")
    inputs = make_rq1_inputs(rows)
    deviation = check_units(inputs)
    units = rq1_units()
    models = {
        "M-main": fit_rq1("main", inputs, y, units),
        "M-phase": fit_rq1("phase", inputs, y, units),
    }
    reference = {"M-main": fit_rq1("main", inputs, y), "M-phase": fit_rq1("phase", inputs, y)}
    tables = [rq1_tables(m, cohort, scale=PHYSICAL) for m in models.values()]
    ame = _concat([ame_table(m, inputs, cohort, a) for a, m in models.items()])
    equivalence = [r for a in models for r in _equivalence_rows(cohort, a, models[a], reference[a])]
    fit_info = [
        {"cohort": cohort, "model": a, "penalty": m.fit.penalty, "l2_lambda": m.fit.l2, "converged": m.fit.converged, "n_iter": m.fit.n_iter,
         "loglik_train": m.fit.loglik, "n_train_rows": int(len(y)), "n_terms": len(m.terms), "dropped": ";".join(m.dropped), "fallback_reason": m.fit.fallback_reason,
         "loglik_per_sd_fit": reference[a].fit.loglik}
        for a, m in models.items()
    ]  # fmt: skip
    return {
        "cohort": cohort, "estimates": _concat([t["estimates"] for t in tables]), "contrasts": _concat([t["contrasts"] for t in tables]), "ame": ame,
        "units": _units_rows(cohort, models["M-main"]), "equivalence": equivalence, "fit_info": fit_info, "integer_deviation": deviation,
        "n_train_rows": int(len(y)), "mean_p": float(y.mean()),
    }  # fmt: skip


def rq1_physical_run(dump_dir: str | Path, cohorts: Sequence[str], out_dir: str | Path, *, labels: str = D.DEFAULT_LABELS,
                     sd_dir: str | Path | None = None) -> dict:  # fmt: skip
    """E4 over the cohorts: fit, pool (DL), AMEs, write the tables to ``out_dir`` (``rq1_physical_*.csv``, ``rq1_physical_summary.json``).

    ``sd_dir``: the per-SD tables of ``--stage rq1`` (``rq1_tidy.csv``); when present the sign counts of the two scalings are compared (they must agree:
    a positive rescaling cannot flip a sign)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohorts = list(cohorts)
    results = []
    for c in cohorts:
        t0 = time.perf_counter()
        results.append(rq1_physical_cohort(dump_dir, c, labels=labels))
        logger.info("%s: E4 physical-unit fits done in %.1fs", c, time.perf_counter() - t0)
    estimates = _concat([r["estimates"] for r in results])
    contrasts = _concat([r["contrasts"] for r in results])
    ame = _concat([r["ame"] for r in results])
    main, groups = rq1_pool_main(estimates), rq1_pool_groups(estimates)
    main["scale"] = PHYSICAL
    unit_of = {t: RQ1_UNITS[t][1] for t in RQ1_TERMS}
    tidy = _concat([main, groups])
    tidy["unit"] = tidy["covariate"].map(unit_of)
    for col, src in (
        ("odds_ratio", "theta"),
        ("or_ci95_low", "ci95_low"),
        ("or_ci95_high", "ci95_high"),
    ):
        tidy[col] = np.exp(tidy[src])
    order = {t: i for i, t in enumerate(RQ1_TERMS)}
    gorder = {"all": 0, "early": 1, "mid": 2, "late": 3}
    tidy = tidy.assign(_c=tidy["covariate"].map(order), _g=tidy["group"].map(gorder)).sort_values(
        ["analysis", "_c", "_g"], kind="mergesort", ascending=[True, True, True]
    )
    tidy = tidy.drop(columns=["_c", "_g"]).reset_index(drop=True)
    ame_pool = ame_pooled(ame)
    equivalence = pd.DataFrame([row for r in results for row in r["equivalence"]])
    units = pd.DataFrame([row for r in results for row in r["units"]])
    fit_info = pd.DataFrame([row for r in results for row in r["fit_info"]])
    for name, frame in (("rq1_physical_tidy", tidy), ("rq1_physical_cohort_estimates", estimates), ("rq1_physical_cohort_contrasts", contrasts),
                        ("rq1_physical_ame_per_cohort", ame), ("rq1_physical_ame_pooled", ame_pool), ("rq1_physical_units", units),
                        ("rq1_physical_check_vs_sd", equivalence), ("rq1_physical_fit_info", fit_info)):  # fmt: skip
        frame.to_csv(out / f"{name}.csv", index=False)
    sign_same = None
    if sd_dir is not None and (Path(sd_dir) / "rq1_tidy.csv").is_file():
        sd = pd.read_csv(Path(sd_dir) / "rq1_tidy.csv")
        sd = sd[(sd["analysis"] == "M-main") | (sd["scale"] == "train_sd")]
        key = ["analysis", "covariate", "group"]
        merged = tidy.merge(
            sd[key + ["k", "n_positive", "n_negative", "theta", "I2"]], on=key, suffixes=("", "_sd")
        )
        sign_same = bool(
            (
                (merged["k"] == merged["k_sd"])
                & (merged["n_positive"] == merged["n_positive_sd"])
                & (merged["n_negative"] == merged["n_negative_sd"])
            ).all()
        )
        merged[key + ["k", "n_positive", "n_negative", "theta", "I2", "theta_sd", "I2_sd"]].to_csv(
            out / "rq1_physical_vs_sd_pooled.csv", index=False
        )
    rel = equivalence["rel_diff"]
    summary = {
        "cohorts": cohorts, "units": {t: u for t, (_, u) in RQ1_UNITS.items()}, "dump_scaling": {"distance_m": ARENA_SIZE_M, "speed_mps": SPEED_SCALE_MPS,
                                                                                                "enemy_count": ENEMY_COUNT_SCALE, "alive_players": ALIVE_PLAYERS_SCALE},
        "main_pooled": main.to_dict("records"), "group_pooled": groups.to_dict("records"), "ame_pooled": ame_pool.to_dict("records"),
        "max_rel_diff_vs_per_sd_fit": float(rel.max()), "max_abs_diff_vs_per_sd_fit": float(equivalence["abs_diff"].max()),
        "sign_counts_equal_to_per_sd_tables": sign_same,
        "max_integer_deviation": {c["cohort"]: c["integer_deviation"] for c in results},
        "penalty_fallbacks": [f"{r.cohort}/{r.model}" for r in fit_info.itertuples() if r.penalty != "none"],
        "dropped_terms": {f"{r.cohort}/{r.model}": r.dropped for r in fit_info.itertuples() if isinstance(r.dropped, str) and r.dropped},
        "git": git_info(),
    }  # fmt: skip
    write_json(out / "rq1_physical_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- command line
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m model_stgcn.ptr.ecomodel", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument(
        "--runs-dir",
        required=True,
        help="where <cohort>/<arm>/seed42/ is written (and the existing arms are read from)",
    )
    ap.add_argument("--out-dir", required=True, help="the report tables")
    ap.add_argument(
        "--cohorts",
        default=None,
        help="comma separated (default: every complete cohort of the dump directory)",
    )
    ap.add_argument("--stage", default="all", choices=("fit", "report", "all", "rq1"),
                    help="rq1: the RQ1 analysis (exposure, movement burden, competition, team size), written to <out-dir>/rq1/")  # fmt: skip
    ap.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="arms to fit (the other run directories are left untouched); default all",
    )
    ap.add_argument("--rq1-units", default="sd", choices=("sd", "physical"),
                    help="--stage rq1: 'sd' = the per-cohort-SD analysis (<out-dir>/rq1/); 'physical' = E4, covariates in physical units + AMEs (<out-dir>/rq1_physical/)")  # fmt: skip
    ap.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="match-level resamples of the share-of-discrimination intervals (0: none)",
    )
    ap.add_argument("--labels", default=D.DEFAULT_LABELS, choices=D.LABEL_CHOICES)
    args = ap.parse_args(argv)
    args.arms = tuple(a for a in args.arms.split(",") if a)
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown or not args.arms:
        ap.error(f"--arms must name arms of {ARMS}, got {unknown or 'none'}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cohorts = (
        [c for c in args.cohorts.split(",") if c]
        if args.cohorts
        else discover_cohorts(args.dump_dir)
    )
    if args.stage == "rq1" and args.rq1_units == "physical":
        summary = rq1_physical_run(
            args.dump_dir,
            cohorts,
            Path(args.out_dir) / "rq1_physical",
            labels=args.labels,
            sd_dir=Path(args.out_dir) / "rq1",
        )
        print(
            f"RQ1 in physical units, {len(summary['cohorts'])} cohorts (DL pooled log-odds per unit; M-main):"
        )
        for r in summary["main_pooled"]:
            print(
                f"  {r['covariate']:18s} {RQ1_UNITS[r['covariate']][1]:32s} {r['theta']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  {r['sign_agreement']}  I2 {r['I2']:.2f}"
            )
        print(
            f"max relative difference to the per-SD fits {summary['max_rel_diff_vs_per_sd_fit']:.2e}; tables in {Path(args.out_dir) / 'rq1_physical'}"
        )
        return 0
    if args.stage == "rq1":
        summary = rq1_run(
            args.dump_dir,
            cohorts,
            args.runs_dir,
            Path(args.out_dir) / "rq1",
            labels=args.labels,
            n_boot=args.n_boot,
        )
        print(
            f"RQ1, {len(summary['cohorts'])} cohorts (DL pooled standardised coefficients; equality p = multivariate random-effects Wald):"
        )
        for r in summary["main_pooled"]:
            print(
                f"  {r['covariate']:18s} {r['theta']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  {r['sign_agreement']}"
            )
        print(f"tables and figure in {Path(args.out_dir) / 'rq1'}")
        return 0
    if args.stage in ("fit", "all"):
        for cohort in cohorts:
            run_cohort(args.dump_dir, cohort, args.runs_dir, labels=args.labels, arms=args.arms)
    if args.stage in ("report", "all"):
        summary = report(args.runs_dir, args.out_dir, cohorts, n_boot=args.n_boot)
        print(f"{len(summary['cohorts'])} cohorts; mean test AUC:")
        for arm, v in summary["mean_auc"].items():
            print(
                f"  {arm:24s} {v['auc']:.4f}  within phase {v['auc_within_phase']:.4f}  ({v['n_cohorts']} cohorts)"
            )
        print(f"tables in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
