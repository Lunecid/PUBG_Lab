"""Metrics, paired match-level bootstrap and DerSimonian-Laird pooling for PTR prediction files.

Prediction CSV (one row per target team at a cut phase)::

    match_id, team_id, phase, y, p_hazard, rank_score, final_rank_norm, top5, match_type

The first seven columns are the agreed format; ``top5`` (final top-5 finisher, 0/1) is appended because
the top-5 AUC cannot be recovered from ``final_rank_norm`` alone (it needs the match's team count), and
``match_type`` (official / competitive, one value per match) because results are also reported per match
type (design doc v2.3): ``all_metrics`` adds a ``by_match_type`` block and the bootstrap helpers accept
``match_type=``.  ``y`` / the risk set are those of the label choice the run used (recorded in its
``metrics.json``, ``labels``); files from runs with different label choices cover different rows and the
paired utilities refuse to compare them.
``phase`` is the cut phase ``k = p + 1`` (1-based game phase, the ``cut_phase`` of the label table and
the "k" of the design doc), where ``p`` is the 0-based phase slot of the dump (:data:`PHASE_OFFSET`).

Direction convention (design doc v2.2, see :mod:`model_stgcn.ptr.loss`): ``final_rank_norm = rank / n_teams``
with rank 1 = winner, so a SMALLER value is a better placement.  ``rank_score`` is survival-like (higher =
better survival, the output of the rank head trained with ``log(1 + exp(-(s_i - s_j)))`` for i better than j;
the HGB baselines write ``rank_score = -p_hazard``).  ``p_hazard`` is a risk (higher = goes out earlier).  All
rank metrics are evaluated on a *risk* score, ``risk = -rank_score`` or ``risk = p_hazard`` (:data:`RISK_SIGN`),
so every rank metric reads "higher is better" and a perfect ranking gives +1 / 1.

Hazard metrics (pooled over target rows): AUC, AP, NLL (probabilities clipped to [1e-7, 1-1e-7], as in
``feature_ablation._bce``), Brier.  A single-class sample gives NaN for AUC/AP.

Rank metrics (driven by ``rank_score``; ``rank_col='p_hazard'`` gives the same metrics from the hazard
probability, which is how the hazard head's own ranking is read):

* ``spearman``  mean over (match, phase) groups of Spearman(risk, final_rank_norm), i.e. the correlation of
  the score with *worse placement*; groups need >= 2 rows and non-constant score and rank;
* ``top5_auc``  pooled AUC of the risk for "will NOT finish top 5" (== AUC of ``rank_score`` for top 5);
* ``cindex``    mean over (match, phase) groups of the concordance of pairs with different
  ``final_rank_norm`` (higher risk <-> worse placement, i.e. the better team has the higher ``rank_score``;
  risk ties count 1/2).

Exploratory hazard metric ``auc_within_phase`` (:data:`EXPLORATORY_METRICS`, complementary to the pooled AUC, which stays the
pre-specified one): the UNWEIGHTED MEAN OVER CUT PHASES of the per-phase hazard AUC (every eligible phase counts once, whatever its
number of rows; this one definition is used everywhere: arms, contrasts, strata, tables).  A pooled AUC also ranks rows of different phases
against each other, so a model that only knows "later phases die more" scores above chance without telling teams of one phase
apart; the within-phase mean removes that phase baseline.  Which phases enter is decided once, on the data of the evaluated
frame: a phase needs at least :data:`MIN_PHASE_ROWS` rows and both classes (:class:`WithinPhaseAUC`), so the set is the same for
every resample and for every arm scored on the same rows (a paired contrast is a difference of means over the same phases).  Under
bootstrap weights an eligible phase whose resample holds a single class has no AUC in that draw and is skipped in that
draw's mean (NaN if none is left).  It is computed by the bootstrap helpers when asked for (``metrics=ALL_METRIC_NAMES``); the
default metric set, :func:`all_metrics` and ``metrics.json`` do not change.

The bootstrap resamples *matches* with replacement (multinomial counts) and evaluates every metric as a
count-weighted statistic, which equals recomputing it on an explicitly duplicated frame (tested).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

PRED_COLUMNS: list[str] = [
    "match_id", "team_id", "phase", "y", "p_hazard", "rank_score", "final_rank_norm", "top5", "match_type",
]  # fmt: skip
KEY_COLUMNS = ["match_id", "team_id", "phase"]
PHASE_OFFSET = (
    1  # CSV ``phase`` = dump phase slot p + 1 (= phase_idx = the cut k of the design doc)
)
N_BOOT = 2000
CLIP = 1e-7
# rank metrics are computed on a risk score: rank_score is survival-like (higher = better), p_hazard is a risk
RISK_SIGN = {"rank_score": -1.0, "p_hazard": 1.0}
HAZARD_METRICS = ("auc", "ap", "nll", "brier")
RANK_METRICS = ("spearman", "top5_auc", "cindex")
METRIC_NAMES = HAZARD_METRICS + RANK_METRICS
EXPLORATORY_METRICS = (
    "auc_within_phase",
)  # computed on request by the bootstrap helpers; not part of all_metrics / metrics.json
ALL_METRIC_NAMES = METRIC_NAMES + EXPLORATORY_METRICS
MIN_PHASE_ROWS = (
    30  # a cut phase needs this many rows (and both classes) to enter the within-phase AUC
)


# --------------------------------------------------------------------------- prediction frames
def read_predictions(src) -> pd.DataFrame:
    """CSV path or DataFrame -> canonical frame (typed, rows sorted by match_id, team_id, phase)."""
    if isinstance(src, (str, Path)):
        df = pd.read_csv(src, dtype={"match_id": str, "match_type": str}, keep_default_na=True)
    else:
        df = src.copy()
    missing = [c for c in PRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prediction frame is missing columns {missing}")
    df = df[PRED_COLUMNS].copy()
    if df["match_type"].isna().any() or (df["match_type"].astype(str).str.strip() == "").any():
        raise ValueError("match_type is missing for some rows of the prediction frame")
    df["match_type"] = df["match_type"].astype(str)
    df["match_id"] = df["match_id"].astype(str)
    if (df.groupby("match_id")["match_type"].nunique() > 1).any():
        raise ValueError("match_type must be a single value per match (one match has several)")
    df["team_id"] = df["team_id"].astype(np.int64)
    df["phase"] = df["phase"].astype(np.int64)
    for c in ("y", "p_hazard", "rank_score", "final_rank_norm", "top5"):
        df[c] = df[c].astype(np.float64)
    return df.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def write_predictions(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    read_predictions(df).to_csv(path, index=False, float_format="%.9g")


def align_predictions(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both files must hold identical (match_id, team_id, phase) rows with identical labels."""
    a, b = read_predictions(a), read_predictions(b)
    same_rows = len(a) == len(b) and all(
        np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in KEY_COLUMNS
    )
    if not same_rows:
        raise ValueError(
            f"prediction files do not cover identical rows ({len(a)} vs {len(b)} rows, or different keys)"
        )
    for c in ("y", "top5"):
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
            raise ValueError(f"label column '{c}' differs between the prediction files")
    if not np.allclose(a["final_rank_norm"], b["final_rank_norm"], rtol=0, atol=1e-6):
        raise ValueError("label column 'final_rank_norm' differs between the prediction files")
    if not np.array_equal(a["match_type"].to_numpy(), b["match_type"].to_numpy()):
        raise ValueError("match_type differs between the prediction files")
    return a, b


# --------------------------------------------------------------------------- weighted ranking core
class SortedBinary:
    """AUC / AP of one score against one binary label under non-negative row weights.

    Sorting is done once; every evaluation is O(n) with tie groups handled exactly like sklearn
    (AUC: ties count 1/2; AP: one threshold per distinct score).
    """

    def __init__(self, y: np.ndarray, score: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        self.n = y.size
        self.order = np.argsort(score, kind="mergesort")
        s = score[self.order]
        self.y = y[self.order]
        self.starts = (
            np.flatnonzero(np.r_[True, s[1:] != s[:-1]]) if self.n else np.zeros(0, np.int64)
        )

    def _groups(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ws = np.asarray(w, dtype=np.float64)[self.order]
        pos = ws * self.y
        neg = ws - pos
        return np.add.reduceat(pos, self.starts), np.add.reduceat(neg, self.starts)

    def auc(self, w: np.ndarray) -> float:
        if not self.n:
            return float("nan")
        gp, gn = self._groups(w)
        p_tot, n_tot = gp.sum(), gn.sum()
        if p_tot <= 0 or n_tot <= 0:
            return float("nan")
        below = np.cumsum(gn) - gn  # negatives strictly below each tie group
        return float((gp * (below + 0.5 * gn)).sum() / (p_tot * n_tot))

    def ap(self, w: np.ndarray) -> float:
        if not self.n:
            return float("nan")
        gp, gn = self._groups(w)
        p_tot, n_tot = gp.sum(), gn.sum()
        if p_tot <= 0 or n_tot <= 0:
            return float("nan")
        gp, gn = gp[::-1], gn[::-1]  # descending score
        tp, fp = np.cumsum(gp), np.cumsum(gn)
        den = tp + fp
        prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
        return float((gp * prec).sum() / p_tot)


class WithinPhaseAUC:
    """Per-phase hazard AUCs of one frame under row weights and their mean (the ``auc_within_phase`` metric).

    The eligible phases are fixed at construction from the unweighted data: at least ``min_rows`` rows and both classes
    (``excluded`` says why a phase is left out).  :meth:`per_phase` / :meth:`mean` take row weights (bootstrap multiplicities);
    a phase whose weighted sample has a single class is NaN in :meth:`per_phase` and skipped by :meth:`mean`."""

    def __init__(
        self, y: np.ndarray, p: np.ndarray, phase: np.ndarray, min_rows: int = MIN_PHASE_ROWS
    ) -> None:
        y, p, phase = (
            np.asarray(y, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.asarray(phase),
        )
        self.min_rows = int(min_rows)
        self.phases: list[int] = []
        self.excluded: dict[int, str] = {}
        self._parts: list[tuple[np.ndarray, SortedBinary]] = []
        for k in np.unique(phase):
            idx = np.flatnonzero(phase == k)
            yk = y[idx]
            if idx.size < self.min_rows:
                self.excluded[int(k)] = f"{idx.size} rows (fewer than {self.min_rows})"
            elif yk.min() == yk.max():
                self.excluded[int(k)] = "a single class"
            else:
                self.phases.append(int(k))
                self._parts.append((idx, SortedBinary(yk, p[idx])))

    def per_phase(self, w: np.ndarray) -> dict[int, float]:
        w = np.asarray(w, dtype=np.float64)
        return {k: sb.auc(w[idx]) for k, (idx, sb) in zip(self.phases, self._parts)}

    def mean(self, w: np.ndarray) -> float:
        vals = [v for v in self.per_phase(w).values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")


def _group_codes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby(["match_id", "phase"], sort=True).ngroup().to_numpy()


def _group_spearman(
    score: np.ndarray, rank: np.ndarray, codes: np.ndarray, n_groups: int
) -> np.ndarray:
    """Per-group Spearman (average ranks); NaN for groups of < 2 rows or constant inputs."""
    ranks = []
    for v in (score, rank):
        ranks.append(pd.Series(v).groupby(codes).rank(method="average").to_numpy())
    a, b = ranks
    n = np.bincount(codes, minlength=n_groups).astype(np.float64)
    sa, sb = np.bincount(codes, a, n_groups), np.bincount(codes, b, n_groups)
    saa, sbb = np.bincount(codes, a * a, n_groups), np.bincount(codes, b * b, n_groups)
    sab = np.bincount(codes, a * b, n_groups)
    cov = n * sab - sa * sb
    va, vb = n * saa - sa * sa, n * sbb - sb * sb
    out = np.full(n_groups, np.nan)
    ok = (n >= 2) & (va > 1e-12) & (vb > 1e-12)
    out[ok] = cov[ok] / np.sqrt(va[ok] * vb[ok])
    return np.clip(out, -1.0, 1.0)


def _group_cindex(
    score: np.ndarray, rank: np.ndarray, codes: np.ndarray, n_groups: int
) -> np.ndarray:
    """Per-group concordance of (higher risk, worse placement) pairs; NaN without comparable pairs."""
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=n_groups))]
    out = np.full(n_groups, np.nan)
    for g in range(n_groups):
        idx = order[bounds[g] : bounds[g + 1]]
        if idx.size < 2:
            continue
        s, f = score[idx], rank[idx]
        comparable = (f[:, None] - f[None, :]) > 1e-9  # row i is the worse placement
        n_pairs = comparable.sum()
        if not n_pairs:
            continue
        ds = s[:, None] - s[None, :]
        out[g] = ((ds > 0) & comparable).sum() / n_pairs + 0.5 * (
            (ds == 0) & comparable
        ).sum() / n_pairs
    return out


class MetricSuite:
    """All metrics of one prediction frame as functions of per-match weights (bootstrap counts)."""

    def __init__(self, df: pd.DataFrame, rank_col: str = "rank_score") -> None:
        df = read_predictions(df)
        self.n_rows = len(df)
        self._within: WithinPhaseAUC | None = (
            None  # built on first use (only the exploratory metric needs it)
        )
        self._within_args = (df["y"].to_numpy(), df["p_hazard"].to_numpy(), df["phase"].to_numpy())
        self.match_ids, self.match_codes = np.unique(df["match_id"].to_numpy(), return_inverse=True)
        self.n_matches = len(self.match_ids)
        y = df["y"].to_numpy()
        p = df["p_hazard"].to_numpy()
        pc = np.clip(p, CLIP, 1 - CLIP)
        self.nll_row = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        self.brier_row = (p - y) ** 2
        self.haz = SortedBinary(y, p)
        if rank_col not in RISK_SIGN:
            raise ValueError(f"rank_col must be one of {sorted(RISK_SIGN)}, got '{rank_col}'")
        score = (
            RISK_SIGN[rank_col] * df[rank_col].to_numpy()
        )  # risk: higher = expected to place worse
        self.top5 = SortedBinary(1.0 - df["top5"].to_numpy(), score)  # positive = outside top 5
        codes = _group_codes(df)
        n_groups = int(codes.max()) + 1 if len(df) else 0
        first = np.zeros(n_groups, dtype=np.int64)
        first[codes[::-1]] = np.arange(len(df))[::-1]  # first row of every group
        self.group_match = self.match_codes[first]
        rank = df["final_rank_norm"].to_numpy()
        self.group_spearman = _group_spearman(score, rank, codes, n_groups)
        self.group_cindex = _group_cindex(score, rank, codes, n_groups)

    @property
    def within_phase(self) -> WithinPhaseAUC:
        """The per-phase AUC engine of this frame (eligible phases fixed from its rows)."""
        if self._within is None:
            self._within = WithinPhaseAUC(*self._within_args)
        return self._within

    @staticmethod
    def _group_mean(values: np.ndarray, weights: np.ndarray) -> float:
        ok = np.isfinite(values) & (weights > 0)
        total = weights[ok].sum()
        return float((values[ok] * weights[ok]).sum() / total) if total > 0 else float("nan")

    def evaluate(
        self, match_w: np.ndarray, names: tuple[str, ...] = METRIC_NAMES
    ) -> dict[str, float]:
        mw = np.asarray(match_w, dtype=np.float64)
        w = mw[self.match_codes]
        total = w.sum()
        gw = mw[self.group_match]
        fns = {
            "auc": lambda: self.haz.auc(w),
            "ap": lambda: self.haz.ap(w),
            "nll": lambda: float((w * self.nll_row).sum() / total) if total > 0 else float("nan"),
            "brier": lambda: float((w * self.brier_row).sum() / total)
            if total > 0
            else float("nan"),
            "spearman": lambda: self._group_mean(self.group_spearman, gw),
            "top5_auc": lambda: self.top5.auc(w),
            "cindex": lambda: self._group_mean(self.group_cindex, gw),
            "auc_within_phase": lambda: self.within_phase.mean(w),
        }
        return {name: fns[name]() for name in names}


# --------------------------------------------------------------------------- plain metrics
def hazard_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = int(y.size)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "pos_rate": nan, "auc": nan, "ap": nan, "nll": nan, "brier": nan}
    pc = np.clip(p, CLIP, 1 - CLIP)
    sb = SortedBinary(y, p)
    ones = np.ones(n)
    return {
        "n": n,
        "pos_rate": float(y.mean()),
        "auc": sb.auc(ones),
        "ap": sb.ap(ones),
        "nll": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "brier": float(((p - y) ** 2).mean()),
    }


def rank_metrics(df: pd.DataFrame, score_col: str = "rank_score") -> dict[str, float]:
    suite = MetricSuite(df, rank_col=score_col)
    res = suite.evaluate(np.ones(suite.n_matches))
    return {
        "spearman": res["spearman"],
        "top5_auc": res["top5_auc"],
        "cindex": res["cindex"],
        "n_groups": int(len(suite.group_match)),
        "n_groups_spearman": int(np.isfinite(suite.group_spearman).sum()),
        "n_groups_cindex": int(np.isfinite(suite.group_cindex).sum()),
    }


def within_phase_auc(df: pd.DataFrame, *, min_rows: int = MIN_PHASE_ROWS) -> dict:
    """Mean over cut phases of the per-phase hazard AUC of a prediction frame (see the module docstring): ``{"auc_within_phase",
    "phases": {k: AUC}, "n_phases", "excluded": {k: reason}}``.  Phases with fewer than ``min_rows`` rows or a single class are left
    out; NaN when none is left."""
    df = read_predictions(df)
    wp = WithinPhaseAUC(
        df["y"].to_numpy(), df["p_hazard"].to_numpy(), df["phase"].to_numpy(), min_rows
    )
    per = wp.per_phase(np.ones(len(df)))
    return {
        "auc_within_phase": wp.mean(np.ones(len(df))),
        "phases": per,
        "n_phases": len(per),
        "excluded": dict(wp.excluded),
    }


def _metrics_block(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_matches": int(df["match_id"].nunique()),
        "hazard": hazard_metrics(df["y"].to_numpy(), df["p_hazard"].to_numpy()),
        "rank": rank_metrics(df, "rank_score"),
        "rank_from_hazard": rank_metrics(df, "p_hazard"),
    }


def all_metrics(df: pd.DataFrame, by_match_type: bool = True) -> dict:
    """Hazard and rank metrics of a prediction frame; ``by_match_type`` adds the same block per match type."""
    df = read_predictions(df)
    out = _metrics_block(df)
    if by_match_type:
        out["by_match_type"] = {
            mt: _metrics_block(g.reset_index(drop=True))
            for mt, g in df.groupby("match_type", sort=True)
        }
    return out


# --------------------------------------------------------------------------- bootstrap
def resample_counts(n_matches: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n_matches]`` multiplicities of a with-replacement resample of whole matches."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_matches, np.full(n_matches, 1.0 / n_matches), size=n_boot)


def _check_metrics(metrics: tuple[str, ...]) -> tuple[str, ...]:
    metrics = tuple(metrics)
    unknown = [m for m in metrics if m not in ALL_METRIC_NAMES]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; known: {ALL_METRIC_NAMES}")
    return metrics


def _as_types(match_type) -> tuple[str, ...] | None:
    if match_type is None:
        return None
    return (match_type,) if isinstance(match_type, str) else tuple(match_type)


def _bootstrap_models(
    preds: Mapping,
    rank_col: str,
    metrics: tuple[str, ...],
    n_boot: int,
    seed: int,
    match_type=None,
) -> tuple[dict, dict, dict]:
    """Score several prediction files on the *same* resampled matches.

    Returns ``(info, point, samples)``: ``point[model][metric]`` on the full data and
    ``samples[model][metric]`` as ``[n_boot]`` arrays.  All files must hold identical rows / labels;
    ``match_type`` (a name or a list) restricts every file to the matches of that stratum before resampling.
    """
    if len(preds) < 2:
        raise ValueError("need at least two prediction files")
    names = list(preds)
    frames = {names[0]: read_predictions(preds[names[0]])}
    for name in names[1:]:
        _, frames[name] = align_predictions(frames[names[0]], preds[name])
    types = _as_types(match_type)
    if types is not None:
        keep = frames[names[0]]["match_type"].isin(types).to_numpy()
        if not keep.any():
            have = sorted(frames[names[0]]["match_type"].unique())
            raise ValueError(f"match_type {list(types)} selects no rows (the files hold {have})")
        frames = {n: f[keep].reset_index(drop=True) for n, f in frames.items()}
    suites = {n: MetricSuite(f, rank_col) for n, f in frames.items()}
    n_matches = suites[names[0]].n_matches
    ones = np.ones(n_matches)
    point = {n: su.evaluate(ones, metrics) for n, su in suites.items()}
    counts = resample_counts(n_matches, n_boot, seed)
    samples = {n: {m: np.empty(n_boot) for m in metrics} for n in names}
    for k in range(n_boot):
        for n, su in suites.items():
            res = su.evaluate(counts[k], metrics)
            for m in metrics:
                samples[n][m][k] = res[m]
    info = {
        "n_rows": suites[names[0]].n_rows,
        "n_matches": n_matches,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "rank_col": rank_col,
        "match_type": None if types is None else list(types),
    }
    return info, point, samples


def bootstrap_arms(
    preds: Mapping,
    *,
    rank_col: str = "rank_score",
    metrics: tuple[str, ...] = METRIC_NAMES,
    n_boot: int = N_BOOT,
    seed: int = 0,
    match_type=None,
) -> tuple[dict, dict, dict]:
    """Score several prediction files on the *same* resampled matches (the engine of every paired helper).

    ``preds``: ``{arm: CSV path or frame}`` with identical rows / labels (checked).  Returns ``(info, point, samples)``:
    ``point[arm][metric]`` on the full data and ``samples[arm][metric]`` as ``[n_boot]`` arrays of resampled values.
    Any linear contrast, ratio or pooling of arms can then be formed from the samples with one shared resampling.
    """
    return _bootstrap_models(preds, rank_col, _check_metrics(metrics), n_boot, seed, match_type)


def _summarise(point: float, draws: np.ndarray, ci: float) -> dict:
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)
    d = draws[np.isfinite(draws)]
    return {
        "estimate": float(point),
        "se": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
        "ci_low": float(np.percentile(d, lo)) if d.size else float("nan"),
        "ci_high": float(np.percentile(d, hi)) if d.size else float("nan"),
        "n_valid": int(d.size),
    }


summarise_draws = (
    _summarise  # public name: {estimate, se, ci_low, ci_high, n_valid} of a bootstrap distribution
)


def paired_contrasts(
    preds: Mapping,
    contrasts: Mapping[str, Mapping[str, float]],
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    rank_col: str = "rank_score",
    metrics: tuple[str, ...] = METRIC_NAMES,
    ci: float = 0.95,
    return_samples: bool = False,
    match_type=None,
) -> dict:
    """Linear contrasts ``sum_k c_k * metric(model_k)`` with a match-level paired bootstrap.

    ``preds``: ``{model name: CSV path or frame}``; every file must hold identical rows and labels
    (checked), and all models are scored on the *same* resampled matches, so shared match-level noise
    cancels.  ``contrasts``: ``{contrast name: {model name: coefficient}}``.  Per contrast and metric:
    ``estimate`` (on the full data), bootstrap ``se``, percentile interval ``[ci_low, ci_high]`` and
    ``n_valid`` (resamples with a defined value; NaN draws, e.g. single-class AUC, are ignored).
    Differences are raw ``A - B`` (for NLL / Brier a negative value favours A).
    """
    metrics = _check_metrics(metrics)
    for cname, coefs in contrasts.items():
        unknown = [m for m in coefs if m not in preds]
        if unknown:
            raise ValueError(f"contrast '{cname}' uses unknown models {unknown}")
    info, point, samples = _bootstrap_models(preds, rank_col, metrics, n_boot, seed, match_type)
    out = {**info, "ci": ci, "models": point, "contrasts": {}}
    draws_out = {}
    for cname, coefs in contrasts.items():
        out["contrasts"][cname] = {}
        draws_out[cname] = {}
        for m in metrics:
            est = sum(c * point[model][m] for model, c in coefs.items())
            draws = sum(c * samples[model][m] for model, c in coefs.items())
            out["contrasts"][cname][m] = _summarise(est, draws, ci)
            draws_out[cname][m] = draws
    if return_samples:
        out["samples"] = draws_out
    return out


def paired_bootstrap(
    pred_a,
    pred_b,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    rank_col: str = "rank_score",
    metrics: tuple[str, ...] = METRIC_NAMES,
    ci: float = 0.95,
    return_samples: bool = False,
    match_type=None,
) -> dict:
    """Match-level paired bootstrap of ``metric(A) - metric(B)`` on identical rows.

    Special case of :func:`paired_contrasts` with ``{A: +1, B: -1}``; per metric the result holds
    ``a``, ``b``, ``diff`` (= ``estimate``), ``se``, ``ci_low``, ``ci_high``, ``n_valid``.
    """
    res = paired_contrasts(
        {"a": pred_a, "b": pred_b},
        {"diff": {"a": 1.0, "b": -1.0}},
        n_boot=n_boot,
        seed=seed,
        rank_col=rank_col,
        metrics=metrics,
        ci=ci,
        return_samples=return_samples,
        match_type=match_type,
    )
    out_metrics = {}
    for m in res["contrasts"]["diff"]:
        c = res["contrasts"]["diff"][m]
        out_metrics[m] = {
            "a": res["models"]["a"][m],
            "b": res["models"]["b"][m],
            "diff": c["estimate"],
            **c,
        }
    out = {
        k: res[k] for k in ("n_rows", "n_matches", "n_boot", "seed", "ci", "rank_col", "match_type")
    }
    out["metrics"] = out_metrics
    if return_samples:
        out["samples"] = res["samples"]["diff"]
    return out


# ---- factorial 2x2 helpers (design doc section 4.1) ----------------------------------------
# For a factor with simple effects (A - B) and (C - D) at the two levels of the other factor:
#     main effect  = 1/2 [(A - B) + (C - D)]            interaction = (A - B) - (C - D)
# Phase-axis main effect (P1):   A=M10, B=M00, C=M11, D=M01
# Team-axis  main effect (P2/P3): A=M01, B=M00, C=M11, D=M10
# Interaction as written in 4.1, (M11 - M10) - (M01 - M00):  A=M11, B=M10, C=M01, D=M00
# ``factorial_2x2`` wires those up from the four arms so callers cannot mix the labels.
FACTORIAL_MAIN = {"A": 0.5, "B": -0.5, "C": 0.5, "D": -0.5}
FACTORIAL_INTERACTION = {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0}


def factorial_main_effect(a, b, c, d, **kw) -> dict:
    """Paired factorial main effect ``1/2 [(A - B) + (C - D)]`` with a match-level paired bootstrap.

    Returns ``{"n_rows", ..., "metrics": {metric: {estimate, se, ci_low, ci_high, n_valid}}}``.
    """
    res = paired_contrasts({"A": a, "B": b, "C": c, "D": d}, {"main": FACTORIAL_MAIN}, **kw)
    return {
        **{k: v for k, v in res.items() if k != "contrasts"},
        "metrics": res["contrasts"]["main"],
    }


def factorial_interaction(a, b, c, d, **kw) -> dict:
    """2x2 interaction ``(A - B) - (C - D)``; pass A=M11, B=M10, C=M01, D=M00 for the 4.1 definition."""
    res = paired_contrasts(
        {"A": a, "B": b, "C": c, "D": d}, {"interaction": FACTORIAL_INTERACTION}, **kw
    )
    return {
        **{k: v for k, v in res.items() if k != "contrasts"},
        "metrics": res["contrasts"]["interaction"],
    }


def factorial_2x2(m00, m10, m01, m11, **kw) -> dict:
    """All 2x2 contrasts of the phase x team design from one shared bootstrap.

    ``phase_main`` = 1/2[(M10-M00)+(M11-M01)] (P1), ``team_main`` = 1/2[(M01-M00)+(M11-M10)] (P2, P3),
    ``interaction`` = (M11-M10)-(M01-M00), plus the four simple effects.
    """
    h = 0.5
    contrasts = {
        "phase_main": {"M10": h, "M00": -h, "M11": h, "M01": -h},
        "team_main": {"M01": h, "M00": -h, "M11": h, "M10": -h},
        "interaction": {"M11": 1.0, "M10": -1.0, "M01": -1.0, "M00": 1.0},
        "phase_given_team_off": {"M10": 1.0, "M00": -1.0},
        "phase_given_team_on": {"M11": 1.0, "M01": -1.0},
        "team_given_phase_off": {"M01": 1.0, "M00": -1.0},
        "team_given_phase_on": {"M11": 1.0, "M10": -1.0},
    }
    return paired_contrasts({"M00": m00, "M10": m10, "M01": m01, "M11": m11}, contrasts, **kw)


def recovery_ratio(a, b, c, *, n_boot: int = N_BOOT, seed: int = 0, rank_col: str = "rank_score",
                   metrics: tuple[str, ...] = ("auc",), ci: float = 0.95,
                   return_samples: bool = False, match_type=None) -> dict:  # fmt: skip
    """Recovery ratio ``kappa = (a - b) / (c - b)`` with a match-level paired bootstrap CI.

    Design doc RQ4: ``a = M11RZ-noHab``, ``b = M11R-noHab``, ``c = M11R``: the share of the information
    carried by the hand-crafted zone / habitat features that the zone object recovers.  Per metric:
    ``kappa`` (full data), percentile ``ci_low`` / ``ci_high`` over resamples with a non-zero
    denominator (``n_valid``), the ``numerator`` and ``denominator`` contrasts (same fields as
    :func:`paired_contrasts`) and ``den_nonpositive_frac`` (share of resamples whose denominator is
    <= 0, where the ratio is not interpretable).

    TODO(aggregator): the design rule reports kappa across cohorts only when the *pooled* denominator
    CI lower bound reaches the SESOI.  Pool numerator and denominator with
    :func:`dersimonian_laird` from each cohort's ``numerator`` / ``denominator`` ``estimate`` and
    ``se``, apply that gate, and if needed bootstrap the pooled ratio from the per-cohort resamples
    (``return_samples=True`` returns the numerator / denominator draws).  Per-cohort kappa is
    descriptive only; RQ4 itself is decided by P5.
    """
    metrics = _check_metrics(metrics)
    info, point, samples = _bootstrap_models(
        {"a": a, "b": b, "c": c}, rank_col, metrics, n_boot, seed, match_type
    )
    out = {**info, "ci": ci, "metrics": {}}
    draws_out = {}
    for m in metrics:
        num_pt, den_pt = point["a"][m] - point["b"][m], point["c"][m] - point["b"][m]
        num = samples["a"][m] - samples["b"][m]
        den = samples["c"][m] - samples["b"][m]
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.where(den != 0, num / den, np.nan)
        kappa_pt = num_pt / den_pt if den_pt != 0 else float("nan")
        summary = _summarise(kappa_pt, kappa, ci)
        out["metrics"][m] = {
            "kappa": summary["estimate"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "n_valid": summary["n_valid"],
            "numerator": _summarise(num_pt, num, ci),
            "denominator": _summarise(den_pt, den, ci),
            "den_nonpositive_frac": float(np.mean(den[np.isfinite(den)] <= 0)) if np.isfinite(den).any() else float("nan"),
        }  # fmt: skip
        draws_out[m] = {"kappa": kappa, "numerator": num, "denominator": den}
    if return_samples:
        out["samples"] = draws_out
    return out


# --------------------------------------------------------------------------- random-effects pooling
def dersimonian_laird(effects, ses) -> dict:
    """DerSimonian-Laird random-effects pooling of per-cohort effects with standard errors.

    ``tau2 = max(0, (Q - (k-1)) / (S1 - S2/S1))`` with ``S1 = sum w``, ``S2 = sum w^2``,
    ``w = 1/se^2``; pooled estimate with weights ``1/(se^2 + tau2)``; normal-theory 95% / 90% intervals and
    the z test (two-sided ``p``; ``p_one_sided`` for H0: effect <= 0, the improvement test of design doc v2.2;
    Holm adjustment across the five primary contrasts is left to the aggregator).
    Studies with a non-finite effect or standard error are dropped (``k`` counts the rest).
    """
    y = np.asarray(effects, dtype=np.float64)
    se = np.asarray(ses, dtype=np.float64)
    if y.shape != se.shape or y.ndim != 1:
        raise ValueError("effects and ses must be 1-D arrays of the same length")
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[ok], se[ok]
    k = int(y.size)
    if k == 0:
        raise ValueError("no study with a finite effect and a positive standard error")
    w = 1.0 / se**2
    fixed = float((w * y).sum() / w.sum())
    q = float((w * (y - fixed) ** 2).sum())
    df = k - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if df > 0 and c > 0 else 0.0
    ws = 1.0 / (se**2 + tau2)
    theta = float((ws * y).sum() / ws.sum())
    se_re = float(np.sqrt(1.0 / ws.sum()))
    z95, z90 = stats.norm.ppf(0.975), stats.norm.ppf(0.95)
    z = theta / se_re
    return {
        "k": k,
        "theta": theta,
        "se": se_re,
        "ci95": (theta - z95 * se_re, theta + z95 * se_re),
        "ci90": (theta - z90 * se_re, theta + z90 * se_re),
        "z": z,
        "p": float(2 * stats.norm.sf(abs(z))),
        "p_one_sided": float(
            stats.norm.sf(z)
        ),  # H0: effect <= 0 against effect > 0 (the verdict rule)
        "tau2": float(tau2),
        "Q": q,
        "df": df,
        "I2": float(max(0.0, (q - df) / q)) if q > 0 and df > 0 else 0.0,
        "p_Q": float(stats.chi2.sf(q, df)) if df > 0 else float("nan"),
        "fixed_effect": fixed,
    }
