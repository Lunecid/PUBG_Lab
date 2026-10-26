"""Exploratory analyses derived from the existing prediction files and contrast tables (no retraining, no change to the
main tables).  Everything here is labelled exploratory; outputs go next to the main results, in ``--out-dir``.

    python -m model_stgcn.ptr.derived --runs-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 \\
        --out-dir reports/bigcomp2027_results [--what phase teamsize calibration] [--workers 2]

``phase``        the contrasts by phase group.  The rows of the cut phases k = 1-2 (early), 3-4 (mid) and 5-7 (late) of every
                 prediction file form three strata next to the match-type strata of :mod:`model_stgcn.ptr.aggregate`
                 (``aggregate(..., phase_strata="only")``): all arms scored on the same resampled matches *within* the
                 stratum (paired match bootstrap), every contrast of the registry (P1-P5, the secondary, the external ones and
                 M11R - HGB-cut) per cohort, then DerSimonian-Laird pooling.  ``contrasts_by_phase.csv`` and
                 ``figures/phase_strata.{pdf,png}`` (Delta AUC of P2, P4, P5, M11R - HGB-cut, M11R - SAGONG-xgb and
                 M11R - KRAFTON-tf by phase group).  The three strata share the matches (a match has rows in several phase
                 groups), so the groups are not independent samples; no test between groups is made.
``teamsize``     the per-cohort contrasts pooled within the squad, duo and solo cohorts (DerSimonian-Laird per subgroup, own
                 tau^2) with the between-subgroup Q test (``contrasts_by_teamsize.csv``).  Input: ``contrasts_per_cohort.csv``
                 of an earlier aggregation, so no bootstrap is repeated.
``calibration``  per cohort, on the test split: ECE (10 equal-mass bins), Brier score and the calibration slope and intercept
                 (logistic recalibration of the outcome on logit(p)) of M00, M11, M11R, M11RZ, HGB-cut, SAGONG-xgb and
                 KRAFTON-tf after Platt scaling (a, b fitted on the validation split of the same cohort and model; the raw
                 probabilities are reported too); unweighted means across cohorts; reliability diagrams of two cohorts
                 (``calibration.csv``, ``figures/calibration.{pdf,png}``, ``figures/calibration_bins.csv``).
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
import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import pubg_lab.survival.ptr.evaluation.aggregate as A
import pubg_lab.survival.ptr.evaluation.metrics as M
from pubg_lab.survival.ptr.common.util import REPO_ROOT

logger = logging.getLogger("pubg_lab.survival.ptr.experiments.derived")

# ---------------------------------------------------------------------------------------------------- phase strata figure
PHASE_PANELS: tuple[tuple[str, str, str], ...] = (  # (contrast, metric, title)
    ("team_main", "auc", "P2 team axis"),
    ("rel_vs_shuf", "auc", "P4 relation vs. shuffle"),
    ("zone_nohab", "auc", "P5 zone object (noHab)"),
    ("m11r_vs_hgbcut", "auc", "M11R − HGB-cut"),
    ("m11r_vs_sagong_xgb", "auc", "M11R − SAGONG-xgb"),
    ("m11r_vs_krafton_tf", "auc", "M11R − KRAFTON-tf"),
)
PHASE_TEXT_OFFSETS = (21.0, 11.5)  # points above the panel: title, subtitle


def _tick_label(group: str) -> str:
    ph = A.PHASE_GROUPS[group]
    return f"{group}\nk = {ph[0]}–{ph[-1]}" if len(ph) > 1 else f"{group}\nk = {ph[0]}"


def build_phase_figure(by_phase: pd.DataFrame):
    """2 x 3 small multiples of the pooled Delta AUC by phase group (95% DerSimonian-Laird interval) with the per-cohort
    estimates as grey dots; a panel whose contrast was not computed says so."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    groups = list(A.PHASE_GROUPS)
    fig = Figure(figsize=(7.2, 4.75), dpi=300, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3, gridspec_kw={"wspace": 0.42, "hspace": 0.95})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.165)
    for ax, (name, metric, title) in zip(axes.ravel(), PHASE_PANELS):
        sub = by_phase[(by_phase["contrast"] == name) & (by_phase["metric"] == metric)]
        sesoi = A.SESOI[metric]
        pooled = {g: sub[(sub["level"] == "pooled") & (sub["phase_group"] == g)] for g in groups}
        cohorts = {
            g: sub[(sub["level"] == "cohort") & (sub["phase_group"] == g)].sort_values("cohort")
            for g in groups
        }
        have = [g for g in groups if len(pooled[g])]
        ax.axhspan(-sesoi, sesoi, color=A.BAND, lw=0, zorder=0)
        ax.axhline(0.0, color=A.MUTED, lw=0.7, zorder=1)
        ax.yaxis.grid(True, color=A.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        lows, highs = [-sesoi, 0.0], [sesoi, 0.0]
        for x, g in enumerate(groups):
            c = cohorts[g]
            if len(c):
                offs = np.linspace(-0.26, 0.26, len(c)) if len(c) > 1 else np.zeros(1)
                ax.plot(
                    x + offs,
                    c["estimate"],
                    "o",
                    ms=2.7,
                    color=A.MUTED,
                    mec="white",
                    mew=0.35,
                    zorder=2,
                )
                lows.append(float(c["estimate"].min()))
                highs.append(float(c["estimate"].max()))
            if len(pooled[g]):
                r = pooled[g].iloc[0]
                ax.vlines(x, r["ci95_low"], r["ci95_high"], color=A.ACCENT, lw=1.7, zorder=3)
                ax.plot(
                    [x],
                    [r["estimate"]],
                    "D",
                    ms=5.0,
                    color=A.ACCENT,
                    mec="white",
                    mew=0.6,
                    zorder=4,
                )
                lows.append(float(r["ci95_low"]))
                highs.append(float(r["ci95_high"]))
        span = max(highs) - min(lows)
        ax.set_ylim(min(lows) - 0.08 * span, max(highs) + 0.08 * span)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([_tick_label(g) for g in groups])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(A.BASELINE)
            ax.spines[side].set_linewidth(0.7)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(
            axis="both",
            labelsize=5.8,
            length=2,
            width=0.6,
            color=A.BASELINE,
            labelcolor=A.INK2,
            pad=1.5,
        )
        ax.set_ylabel(f"Δ {A.METRIC_LABEL[metric]}", fontsize=6.2, color=A.INK2, labelpad=1.5)
        if have:
            ks = sorted({int(pooled[g].iloc[0]["k"]) for g in have})
            total = int(pooled[have[0]].iloc[0]["k_total"])
            sub_text = f"k = {ks[0]}" if len(ks) == 1 else f"k = {ks[0]}–{ks[-1]}"
            sub_text += f" of {total} cohorts"
        else:
            sub_text = "not computed"
            ax.text(
                0.5,
                0.5,
                "not run",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.5,
                color=A.INK2,
            )
        for text, off, size_pt, weight, color in (
            (title, PHASE_TEXT_OFFSETS[0], 6.8, "bold", A.INK),
            (sub_text, PHASE_TEXT_OFFSETS[1], 6.0, "normal", A.INK2),
        ):
            ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, off), textcoords="offset points", fontsize=size_pt,
                        color=color, fontweight=weight, ha="left", va="bottom")  # fmt: skip
    handles = [
        Line2D([], [], marker="o", color=A.MUTED, lw=0, ms=3.2, label="Cohort estimate"),
        Line2D(
            [],
            [],
            marker="D",
            color=A.ACCENT,
            lw=1.7,
            ms=5.0,
            label="Pooled (DerSimonian–Laird, 95% interval)",
        ),
        Patch(facecolor=A.BAND, label="±SESOI"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.3,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.8,
    )
    for t in leg.get_texts():
        t.set_color(A.INK2)
    fig.text(
        0.085,
        0.965,
        "Contrasts by phase group (exploratory)",
        fontsize=7.4,
        color=A.INK,
        fontweight="bold",
        ha="left",
        va="center",
    )
    return fig


def draw_phase_strata(by_phase: pd.DataFrame, stem: Path) -> None:
    """Write ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` (TrueType fonts)."""
    from matplotlib import rc_context

    fig = build_phase_figure(by_phase)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


def phase_strata(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    n_boot: int = A.N_BOOT,
    seed: int = 0,
    workers: int = 2,
    cohorts: Sequence[str] | None = None,
    skip_misaligned: bool = False,
    figures: bool = True,
) -> pd.DataFrame:
    """Phase-group strata of every contrast: ``contrasts_by_phase.csv`` (+ the figure) in ``out_dir``; nothing else is written."""
    res = A.aggregate(
        runs_dir, out_dir, n_boot=n_boot, seed=seed, workers=workers, cohorts=cohorts, skip_misaligned=skip_misaligned,
        figures=figures, phase_strata="only",
    )  # fmt: skip
    return res.by_phase


# ---------------------------------------------------------------------------------------------------- team-size subgroups
TEAM_SIZES: tuple[str, ...] = ("squad", "duo", "solo")
TEAMSIZE_COLUMNS = [
    "contrast", "metric", "stratum", "subgroup", "k", "theta", "se", "ci95_low", "ci95_high", "ci90_low", "ci90_high", "tau2", "Q", "I2",
    "p_one_sided", "p_two_sided", "verdict", "q_between", "df_between", "p_between", "n_subgroups", "exploratory", "cohorts",
]  # fmt: skip


def team_size_of(cohort: str) -> str:
    """``squad`` / ``duo`` / ``solo`` from the cohort name (``<map>_<tpp|fpp>_<size>``), else ``unknown``."""
    last = cohort.rsplit("_", 1)[-1]
    return last if last in TEAM_SIZES else "unknown"


def subgroup_heterogeneity(thetas: Sequence[float], ses: Sequence[float]) -> dict:
    """Between-subgroup Q test on subgroup estimates: ``Q = sum_g (theta_g - theta_bar)^2 / se_g^2`` with the inverse-variance mean
    ``theta_bar``, ``df = G - 1`` and a chi-square p.  NaN for fewer than two subgroups."""
    th, se = np.asarray(thetas, dtype=np.float64), np.asarray(ses, dtype=np.float64)
    g = len(th)
    if g < 2:
        return {"q": float("nan"), "df": max(g - 1, 0), "p": float("nan")}
    w = 1.0 / se**2
    bar = float((w * th).sum() / w.sum())
    q = float((w * (th - bar) ** 2).sum())
    return {"q": q, "df": g - 1, "p": float(stats.chi2.sf(q, g - 1))}


def teamsize_subgroups(
    contrasts: pd.DataFrame, sizes: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Pool the per-cohort contrasts within the team-size subgroups (squad, duo, solo; DerSimonian-Laird with a tau^2 of its
    own per subgroup) and test the heterogeneity between the subgroups.

    ``contrasts``: per-cohort rows (``cohort, contrast, metric, stratum, estimate, se``), e.g. ``contrasts_per_cohort.csv``.
    ``sizes``: optional ``{cohort: subgroup}`` override of :func:`team_size_of`.  Per (contrast, metric, stratum): one row per
    subgroup with data plus ``subgroup = 'all'`` (the usual pooled estimate over every cohort); ``q_between`` / ``df_between`` /
    ``p_between`` (repeated on every row of that triple) test whether the subgroup estimates differ.  A subgroup of one cohort
    (solo) carries that cohort's own SE (``tau2`` 0), so its estimate is as uncertain as the cohort's.  Rows follow the
    one-sided p / unadjusted verdict conventions of ``contrasts_pooled.csv`` and are exploratory."""
    sizes = dict(sizes or {})
    rows = []
    order = [*TEAM_SIZES, "unknown"]
    fo = list(A.FORMULAS)
    keyed = {}
    for r in contrasts.itertuples(index=False):
        keyed.setdefault((r.contrast, r.metric, r.stratum), []).append(r)
    for (name, metric, stratum), items in sorted(
        keyed.items(),
        key=lambda kv: (
            fo.index(kv[0][0]) if kv[0][0] in fo else len(fo),
            A.METRICS.index(kv[0][1]),
            kv[0][2],
        ),
    ):
        ok = [r for r in items if np.isfinite(r.estimate) and np.isfinite(r.se) and r.se > 0]
        if not ok:
            continue
        groups: dict[str, list] = {}
        for r in ok:
            groups.setdefault(sizes.get(r.cohort, team_size_of(r.cohort)), []).append(r)
        direction, sesoi = A.DIRECTION[metric], A.SESOI.get(metric)
        block = []
        for label, members in [
            ("all", ok),
            *[(g, groups[g]) for g in order if g in groups],
            *[(g, v) for g, v in groups.items() if g not in order],
        ]:
            dl = M.dersimonian_laird([m.estimate for m in members], [m.se for m in members])
            p_one = float(stats.norm.sf(direction * dl["z"]))
            block.append({
                "contrast": name, "metric": metric, "stratum": stratum, "subgroup": label, "k": dl["k"], "theta": dl["theta"],
                "se": dl["se"], "ci95_low": dl["ci95"][0], "ci95_high": dl["ci95"][1], "ci90_low": dl["ci90"][0],
                "ci90_high": dl["ci90"][1], "tau2": dl["tau2"], "Q": dl["Q"], "I2": dl["I2"], "p_one_sided": p_one,
                "p_two_sided": dl["p"], "verdict": A.verdict(dl["theta"], dl["ci90"], p_one, sesoi, direction),
                "exploratory": True, "cohorts": ";".join(sorted(m.cohort for m in members)),
            })  # fmt: skip
        subs = [b for b in block if b["subgroup"] != "all"]
        het = subgroup_heterogeneity([b["theta"] for b in subs], [b["se"] for b in subs])
        for b in block:
            b.update(
                q_between=het["q"], df_between=het["df"], p_between=het["p"], n_subgroups=len(subs)
            )
        rows += block
    return pd.DataFrame(rows, columns=TEAMSIZE_COLUMNS)


def teamsize_from_csv(
    contrasts_csv: str | Path, out_csv: str | Path, sizes: Mapping[str, str] | None = None
) -> pd.DataFrame:
    path = Path(contrasts_csv)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run the aggregator once (it writes contrasts_per_cohort.csv) or pass --contrasts-csv"
        )
    df = teamsize_subgroups(pd.read_csv(path), sizes)
    A._write_csv(df, Path(out_csv))
    return df


# ---------------------------------------------------------------------------------------------------- calibration
CALIBRATION_MODELS: tuple[str, ...] = (
    "M00",
    "M11",
    "M11R",
    "M11RZ",
    "HGB-cut",
    "SAGONG-xgb",
    "KRAFTON-tf",
)
RELIABILITY_COHORTS: tuple[str, ...] = ("erangel_tpp_squad", "taego_tpp_squad")
N_BINS = 10
CLIP = 1e-7
CAL_COLUMNS = [
    "level", "cohort", "model", "variant", "n_cohorts", "n_rows", "n_matches", "pos_rate", "mean_pred", "mean_obs", "ece10", "brier",
    "cal_slope", "cal_intercept", "platt_a", "platt_b", "n_val_rows", "ece10_sd", "brier_sd", "cal_slope_sd", "cal_intercept_sd", "exploratory",
]  # fmt: skip
CAL_BIN_COLUMNS = ["cohort", "model", "variant", "bin", "n", "mean_pred", "obs_rate"]


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


def fit_logistic(x, y, *, max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Maximum-likelihood ``P(y = 1) = sigmoid(a + b x)`` by Newton-Raphson with step halving; returns ``(a, b)``.

    Raises ``ValueError`` for a single outcome class; a constant ``x`` gives ``b = nan`` (``a`` is then the log-odds)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if y.size == 0 or y.min() == y.max():
        raise ValueError("logistic recalibration needs both outcome classes")
    base = float(np.log(y.mean() / (1.0 - y.mean())))
    if np.ptp(x) < 1e-12:
        return base, float("nan")
    X = np.column_stack([np.ones_like(x), x])

    def loglik(beta):
        eta = X @ beta
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    beta = np.array([base, 0.0])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        cur, t = loglik(beta), 1.0
        while t > 1e-10 and loglik(beta + t * step) < cur - 1e-12:
            t /= 2.0
        beta = beta + t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return float(beta[0]), float(beta[1])


def platt_scale(p_fit, y_fit):
    """Platt scaling fitted on ``(p_fit, y_fit)`` (logistic regression of y on logit p): returns ``(a, b, apply)`` with
    ``apply(p) = sigmoid(a + b logit p)``."""
    a, b = fit_logistic(logit(p_fit), y_fit)
    return a, b, (lambda p: 1.0 / (1.0 + np.exp(-(a + b * logit(p)))))


def equal_mass_bins(p, y, bins: int = N_BINS) -> pd.DataFrame:
    """``bins`` bins of (almost) equal size after sorting by ``p`` (ties broken by order): size, mean prediction, observed rate."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for b, idx in enumerate(np.array_split(order, bins), start=1):
        if idx.size:
            rows.append(
                {
                    "bin": b,
                    "n": int(idx.size),
                    "mean_pred": float(p[idx].mean()),
                    "obs_rate": float(y[idx].mean()),
                }
            )
    return pd.DataFrame(rows, columns=["bin", "n", "mean_pred", "obs_rate"])


def calibration_metrics(p, y, bins: int = N_BINS) -> dict:
    """ECE over ``bins`` equal-mass bins, Brier score, the calibration slope and intercept (logistic recalibration of the outcome
    on logit p, both parameters free) and the means of prediction and outcome."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    table = equal_mass_bins(p, y, bins)
    ece = float(
        (table["n"] * (table["obs_rate"] - table["mean_pred"]).abs()).sum() / table["n"].sum()
    )
    try:
        a, b = fit_logistic(logit(p), y)
    except ValueError:
        a = b = float("nan")
    return {"n_rows": int(y.size), "pos_rate": float(y.mean()), "mean_pred": float(p.mean()), "mean_obs": float(y.mean()), "ece10": ece,
            "brier": float(np.mean((p - y) ** 2)), "cal_slope": b, "cal_intercept": a}  # fmt: skip


def _read_pair(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    val, test = run_dir / "pred_val.csv", run_dir / "pred_test.csv"
    if not (val.is_file() and test.is_file()) or not (run_dir / "metrics.json").is_file():
        return None
    return M.read_predictions(val), M.read_predictions(test)


def calibration_tables(
    runs_dir: str | Path,
    models: Sequence[str] = CALIBRATION_MODELS,
    cohorts: Sequence[str] | None = None,
    seed: int = A.MAIN_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration of every model on every cohort's test split (``calibration.csv`` rows and the reliability bins).

    Per cohort and model, for the raw ``p_hazard`` and for the Platt-scaled one (``variant`` ``raw`` | ``platt``; Platt fitted on
    the validation split of that run); then the unweighted means across the cohorts that have the model (``level = mean``,
    with the SD) and across the cohorts that have all the models (``level = mean_common``)."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"runs directory not found: {root}")
    rows, bins = [], []
    present = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "logs")
    for cohort in present if cohorts is None else [c for c in present if c in set(cohorts)]:
        for model in models:
            pair = _read_pair(root / cohort / model / f"seed{seed}")
            if pair is None:
                continue
            val, test = pair
            y_val, y_test = val["y"].to_numpy(), test["y"].to_numpy()
            try:
                a, b, apply = platt_scale(val["p_hazard"].to_numpy(), y_val)
            except ValueError:
                logger.warning(
                    "%s %s: the validation split has a single outcome class; Platt scaling skipped",
                    cohort,
                    model,
                )
                a = b = float("nan")
                apply = None
            for variant, p in (
                ("raw", test["p_hazard"].to_numpy()),
                ("platt", None if apply is None else apply(test["p_hazard"].to_numpy())),
            ):
                if p is None:
                    continue
                cm = calibration_metrics(p, y_test)
                rows.append({"level": "cohort", "cohort": cohort, "model": model, "variant": variant, "n_cohorts": 1, "n_matches": int(test["match_id"].nunique()),
                             "platt_a": a, "platt_b": b, "n_val_rows": int(len(val)), "exploratory": True, **cm})  # fmt: skip
                for r in equal_mass_bins(p, y_test).itertuples(index=False):
                    bins.append(
                        {
                            "cohort": cohort,
                            "model": model,
                            "variant": variant,
                            "bin": r.bin,
                            "n": r.n,
                            "mean_pred": r.mean_pred,
                            "obs_rate": r.obs_rate,
                        }
                    )
    per_cohort = pd.DataFrame(rows, columns=CAL_COLUMNS)
    means = []
    metrics_cols = (
        "ece10",
        "brier",
        "cal_slope",
        "cal_intercept",
        "mean_pred",
        "mean_obs",
        "pos_rate",
        "n_rows",
        "n_matches",
    )
    for level, restrict in (("mean", False), ("mean_common", True)):
        for variant in ("raw", "platt"):
            sub = per_cohort[per_cohort["variant"] == variant]
            if sub.empty:
                continue
            common = None
            if restrict:
                have = sub.groupby("cohort")["model"].nunique()
                common = set(have[have == len(models)].index) if len(models) else set()
                if not common:
                    continue
            for model in models:
                part = sub[sub["model"] == model]
                if common is not None:
                    part = part[part["cohort"].isin(common)]
                if part.empty:
                    continue
                row = {"level": level, "cohort": "", "model": model, "variant": variant, "n_cohorts": int(len(part)), "exploratory": True,
                       "platt_a": float(part["platt_a"].mean()), "platt_b": float(part["platt_b"].mean()), "n_val_rows": int(part["n_val_rows"].sum())}  # fmt: skip
                for col in metrics_cols:
                    row[col] = float(part[col].mean())
                for col in ("ece10", "brier", "cal_slope", "cal_intercept"):
                    row[col + "_sd"] = (
                        float(part[col].std(ddof=1)) if len(part) > 1 else float("nan")
                    )
                means.append(row)
    table = (
        pd.concat([pd.DataFrame(means, columns=CAL_COLUMNS), per_cohort], ignore_index=True)
        if means
        else per_cohort
    )
    mo = {m: i for i, m in enumerate(models)}
    lv = {"mean": 0, "mean_common": 1, "cohort": 2}
    order = sorted(
        range(len(table)),
        key=lambda i: (
            lv[table["level"].iloc[i]],
            table["cohort"].iloc[i],
            mo[table["model"].iloc[i]],
            table["variant"].iloc[i],
        ),
    )
    return table.iloc[order].reset_index(drop=True), pd.DataFrame(bins, columns=CAL_BIN_COLUMNS)


def build_calibration_figure(
    table: pd.DataFrame,
    bins: pd.DataFrame,
    cohorts: Sequence[str] = RELIABILITY_COHORTS,
    models: Sequence[str] = CALIBRATION_MODELS,
):
    """Reliability diagrams (equal-mass bins): one row per cohort, one panel per model; the Platt-scaled curve in the accent colour, the
    raw one in grey, the identity dashed.  ECE (Platt) above every panel."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    cohorts = [c for c in cohorts if (bins["cohort"] == c).any()]
    nrow, ncol = max(len(cohorts), 1), len(models)
    fig = Figure(figsize=(7.2, 1.35 * nrow + 1.15), dpi=300, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(
        nrow,
        ncol,
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.16, "hspace": 0.72},
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.99,
        top=1 - 0.52 / (1.35 * nrow + 1.15),
        bottom=0.64 / (1.35 * nrow + 1.15) + 0.04,
    )
    hi = float(bins[["mean_pred", "obs_rate"]].max().max()) if len(bins) else 1.0
    lim = min(1.0, np.ceil(hi * 10 + 0.5) / 10)
    for i, cohort in enumerate(cohorts):
        for j, model in enumerate(models):
            ax = axes[i, j]
            ax.plot([0, lim], [0, lim], ls="--", lw=0.6, color=A.BASELINE, zorder=1)
            for variant, color, lw, zorder in (
                ("raw", A.MUTED, 0.8, 2),
                ("platt", A.ACCENT, 1.3, 3),
            ):
                b = bins[
                    (bins["cohort"] == cohort)
                    & (bins["model"] == model)
                    & (bins["variant"] == variant)
                ]
                if len(b):
                    ax.plot(
                        b["mean_pred"],
                        b["obs_rate"],
                        "-o",
                        ms=2.1,
                        lw=lw,
                        color=color,
                        mec="white",
                        mew=0.25,
                        zorder=zorder,
                    )
            row = table[
                (table["level"] == "cohort")
                & (table["cohort"] == cohort)
                & (table["model"] == model)
                & (table["variant"] == "platt")
            ]
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal", adjustable="box")
            ticks = [0.0, round(lim / 2, 2), lim]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
            ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(A.BASELINE)
                ax.spines[side].set_linewidth(0.6)
            ax.tick_params(
                axis="both",
                labelsize=5.2,
                length=1.8,
                width=0.5,
                color=A.BASELINE,
                labelcolor=A.INK2,
                pad=1.2,
            )
            ax.grid(True, color=A.GRID, lw=0.4)
            ax.set_axisbelow(True)
            if i == 0:
                ax.annotate(
                    model,
                    xy=(0.0, 1.0),
                    xycoords="axes fraction",
                    xytext=(0.0, 12.5),
                    textcoords="offset points",
                    fontsize=6.0,
                    color=A.INK,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )
            ece = "n/a" if row.empty else f"ECE {float(row.iloc[0]['ece10']):.3f}"
            ax.annotate(ece if len(row) or i else "not run", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, 3.5), textcoords="offset points",
                        fontsize=5.4, color=A.INK2, ha="left", va="bottom") if (len(row) or i == 0 or True) else None  # fmt: skip
            if j == 0:
                ax.set_ylabel("observed", fontsize=5.6, color=A.INK2, labelpad=1.5)
            if i == nrow - 1:
                ax.set_xlabel("predicted", fontsize=5.6, color=A.INK2, labelpad=1.5)
        fig.text(0.012, axes[i, 0].get_position().y0 + axes[i, 0].get_position().height / 2, A.cohort_label(cohort), rotation=90,
                 fontsize=6.4, color=A.INK, fontweight="bold", ha="left", va="center")  # fmt: skip
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=A.ACCENT,
            lw=1.3,
            ms=3.0,
            label="Platt-scaled on the validation split",
        ),
        Line2D([], [], marker="o", color=A.MUTED, lw=0.8, ms=3.0, label="raw p_hazard"),
        Line2D([], [], ls="--", color=A.BASELINE, lw=0.8, label="perfect calibration"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.0,
        bbox_to_anchor=(0.55, 0.0),
        handlelength=2.0,
    )
    for t in leg.get_texts():
        t.set_color(A.INK2)
    return fig


def draw_calibration(
    table: pd.DataFrame,
    bins: pd.DataFrame,
    stem: Path,
    cohorts: Sequence[str] = RELIABILITY_COHORTS,
) -> None:
    from matplotlib import rc_context

    fig = build_calibration_figure(table, bins, cohorts)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


def calibration(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    models: Sequence[str] = CALIBRATION_MODELS,
    cohorts: Sequence[str] | None = None,
    figures: bool = True,
) -> pd.DataFrame:
    out = Path(out_dir)
    table, bins = calibration_tables(runs_dir, models, cohorts)
    if table.empty:
        raise ValueError(
            f"no run with pred_val.csv and pred_test.csv of {list(models)} under {runs_dir}"
        )
    A._write_csv(table, out / "calibration.csv")
    if figures:
        A._write_csv(bins, out / "figures" / "calibration_bins.csv")
        draw_calibration(table, bins, out / "figures" / "calibration")
    return table


# ---------------------------------------------------------------------------------------------------- command line
def format_calibration(table: pd.DataFrame) -> str:
    sel = table[(table["level"] == "mean") & (table["variant"] == "platt")]
    raw = table[(table["level"] == "mean") & (table["variant"] == "raw")].set_index("model")
    lines = [
        f"{'model':<12} {'cohorts':>7} {'ECE(platt)':>11} {'Brier':>8} {'slope':>7} {'intercept':>10} {'ECE(raw)':>9} {'slope(raw)':>10}"
    ]
    for r in sel.itertuples(index=False):
        rr = raw.loc[r.model] if r.model in raw.index else None
        lines.append(f"{r.model:<12} {int(r.n_cohorts):>7} {r.ece10:>11.4f} {r.brier:>8.4f} {r.cal_slope:>7.3f} {r.cal_intercept:>10.3f} "
                     f"{(rr.ece10 if rr is not None else float('nan')):>9.4f} {(rr.cal_slope if rr is not None else float('nan')):>10.3f}")  # fmt: skip
    return "\n".join(lines)


def format_phase(by_phase: pd.DataFrame) -> str:
    lines = [f"{'contrast':<22} {'early':>26} {'mid':>26} {'late':>26}"]
    for name, metric, _ in PHASE_PANELS:
        cells = []
        for g in A.PHASE_GROUPS:
            r = by_phase[
                (by_phase["contrast"] == name)
                & (by_phase["metric"] == metric)
                & (by_phase["phase_group"] == g)
                & (by_phase["level"] == "pooled")
            ]
            cells.append(
                "not computed"
                if r.empty
                else f"{r.iloc[0]['estimate']:+.4f} [{r.iloc[0]['ci95_low']:+.4f}, {r.iloc[0]['ci95_high']:+.4f}]"
            )
        lines.append(f"{name:<22} " + " ".join(f"{c:>26}" for c in cells))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Exploratory derived analyses: phase groups, team-size subgroups, calibration."
    )
    ap.add_argument(
        "--runs-dir", default=None, help="run_matrix output (needed for phase and calibration)"
    )
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "bigcomp2027_results"))
    ap.add_argument(
        "--what",
        nargs="+",
        default=["phase", "teamsize", "calibration"],
        choices=["phase", "teamsize", "calibration"],
    )
    ap.add_argument(
        "--workers", type=int, default=2, help="processes of the phase-strata bootstrap (default 2)"
    )
    ap.add_argument("--n-boot", type=int, default=A.N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cohorts", default=None, help="comma-separated cohorts (phase and calibration)"
    )
    ap.add_argument(
        "--contrasts-csv",
        default=None,
        help="per-cohort contrasts for the team-size subgroups (default: <out-dir>/contrasts_per_cohort.csv)",
    )
    ap.add_argument("--skip-misaligned", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("fontTools", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(args.out_dir)
    cohorts = [c for c in args.cohorts.split(",") if c] if args.cohorts else None
    if ("phase" in args.what or "calibration" in args.what) and not args.runs_dir:
        raise SystemExit("--runs-dir is required for --what phase / calibration")
    if "teamsize" in args.what:
        csv = Path(args.contrasts_csv) if args.contrasts_csv else out / "contrasts_per_cohort.csv"
        df = teamsize_from_csv(csv, out / "contrasts_by_teamsize.csv")
        print(f"contrasts_by_teamsize.csv: {len(df)} rows from {csv}")
    if "phase" in args.what:
        by_phase = phase_strata(args.runs_dir, out, n_boot=args.n_boot, seed=args.seed, workers=args.workers, cohorts=cohorts,
                                skip_misaligned=args.skip_misaligned, figures=not args.no_figures)  # fmt: skip
        print("contrasts_by_phase.csv: pooled delta AUC by phase group")
        print(format_phase(by_phase))
    if "calibration" in args.what:
        table = calibration(args.runs_dir, out, cohorts=cohorts, figures=not args.no_figures)
        print("calibration.csv: unweighted means across cohorts, test split")
        print(format_calibration(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
