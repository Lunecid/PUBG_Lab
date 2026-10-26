"""Learning curve of design doc 4.4 (exploratory): test AUC against the number of training matches.

    python -m model_stgcn.ptr.learning_curve --lc-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1_lc \\
        --main-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 [--steps-dir ..._lc_steps] \\
        --out-dir reports/bigcomp2027_results

Reads ``metrics.json`` of the runs of :mod:`model_stgcn.ptr.run_matrix` (``--lc``): subsampled runs
``<cohort>/<arm>/sub<N>_ss<s>_seed<s>/`` and full-size runs ``<cohort>/<arm>/seed<s>/``.  The full-size points come
from the learning-curve directory *and* the main matrix (``--main-dir``: seeds 13 / 42 / 2026 of M00 and M11R, seed 42
of the others); a run found in several directories is taken from the first one listed.  ``--steps-dir`` is the
steps-matched variant (epochs scaled so that the number of optimiser steps stays near the full-size run); it shares the
full-size points with the fixed-epoch curve.

Outputs: ``learning_curve.csv`` (tidy, one row per run: every test metric), ``learning_curve_summary.csv`` (mean, SD
over seeds per cohort / model / size / protocol) and ``figures/learning_curve.{pdf,png}`` (one panel per cohort,
log x axis, mean +- SD over the seeds; ``learning_curve_steps.*`` for the steps-matched protocol).  The SD at full size
and for HGB is small because only the seed changes there; in the subsampled runs the seed also changes the matches.
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
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pubg_lab.survival.ptr.common.util import REPO_ROOT

logger = logging.getLogger("pubg_lab.survival.ptr.experiments.learning_curve")

MODELS: tuple[str, ...] = ("HGB-cut", "M00", "M10", "M01", "M11R")
TEST_METRICS = (
    ("auc", "hazard", "auc"), ("ap", "hazard", "ap"), ("nll", "hazard", "nll"), ("brier", "hazard", "brier"),
    ("spearman", "rank", "spearman"), ("top5_auc", "rank", "top5_auc"), ("cindex", "rank", "cindex"),
)  # fmt: skip
FULL_PATTERN = re.compile(r"seed(\d+)")
SUB_PATTERN = re.compile(r"sub(\d+)_ss(-?\d+)_seed(\d+)")
TIDY_COLUMNS = [
    "protocol", "exploratory", "cohort", "arm", "n_train", "n_full", "full_size", "seed", "subsample_seed", "epochs_run", "best_epoch",
    "max_epochs", "test_auc", "test_ap", "test_nll", "test_brier", "test_spearman", "test_top5_auc", "test_cindex",
    "labels", "source", "run_dir",
]  # fmt: skip
SUMMARY_COLUMNS = [
    "protocol",
    "exploratory",
    "cohort",
    "arm",
    "n_train",
    "full_size",
    "n_seeds",
    "auc_mean",
    "auc_sd",
    "auc_min",
    "auc_max",
]
# colours: the validated categorical order (blue, orange, aqua, yellow) for the four neural models, a neutral dashed
# line for the HGB reference; the headline model M11R takes the first slot
COLORS = {
    "M11R": "#2a78d6",
    "M00": "#eb6834",
    "M10": "#1baf7a",
    "M01": "#eda100",
    "HGB-cut": "#52514e",
}
MARKERS = {"M11R": "o", "M00": "s", "M10": "^", "M01": "v", "HGB-cut": "D"}
INK, INK2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(
    sources: Sequence[tuple[str, str | Path]], cohorts: Iterable[str] | None = None
) -> pd.DataFrame:
    """Tidy table of every finished run below the ``(protocol, directory)`` sources, in the order given.

    A full-size run (``seed<s>``) is listed once per cohort / model / seed under the *first* source that has it, and
    is repeated for every protocol (``all``-protocol sources: a source labelled ``main`` or ``fixed`` is shared with
    the steps-matched curve, see :func:`curve_tables`).  ``n_train`` is the number of training matches the model saw
    (the subsample size, or the full train size read from the neural runs of the cohort)."""
    wanted = None if cohorts is None else set(cohorts)
    rows: list[dict] = []
    for protocol, root in sources:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"learning-curve directory not found: {root}")
        for meta_path in sorted(root.glob("*/*/*/metrics.json")):
            run_dir = meta_path.parent
            arm, cohort = run_dir.parent.name, run_dir.parent.parent.name
            if wanted is not None and cohort not in wanted:
                continue
            sub, full = SUB_PATTERN.fullmatch(run_dir.name), FULL_PATTERN.fullmatch(run_dir.name)
            if not sub and not full:
                continue
            meta = _read(meta_path)
            if meta is None or "test" not in meta:
                logger.warning("skipping unreadable / unfinished run %s", run_dir)
                continue
            seed = int(sub.group(3) if sub else full.group(1))
            sub_info = (meta.get("data") or {}).get("train_subsample") or meta.get(
                "train_subsample"
            )
            n_matches = (meta.get("n_matches") or {}).get("train")
            if n_matches is None and sub_info:
                n_matches = sub_info["n_used"]
            row = {
                "protocol": protocol, "exploratory": True, "cohort": cohort, "arm": arm, "n_train": n_matches,
                "n_full": sub_info["n_full"] if sub_info else n_matches, "full_size": sub is None, "seed": seed,
                "subsample_seed": int(sub.group(2)) if sub else np.nan,
                "epochs_run": meta.get("epochs_run", np.nan), "best_epoch": meta.get("best_epoch", np.nan),
                "max_epochs": (meta.get("train_config") or {}).get("max_epochs", np.nan),
                "labels": meta.get("labels", ""), "source": str(root), "run_dir": str(run_dir),
            }  # fmt: skip
            for name, block, key in TEST_METRICS:
                row[f"test_{name}"] = (meta["test"].get(block) or {}).get(key, np.nan)
            rows.append(row)
    df = pd.DataFrame(rows, columns=TIDY_COLUMNS)
    if df.empty:
        return df
    # HGB runs of the main matrix carry no match count: a full-size run takes the train size of the cohort's neural runs
    full_n = df[df["full_size"] & df["n_train"].notna()].groupby("cohort")["n_train"].max()
    miss = df["n_train"].isna() & df["full_size"]
    df.loc[miss, "n_train"] = df.loc[miss, "cohort"].map(full_n)
    df.loc[df["n_full"].isna(), "n_full"] = df.loc[df["n_full"].isna(), "cohort"].map(full_n)
    unknown = df["n_train"].isna()
    if unknown.any():
        raise ValueError(
            f"cannot tell the number of training matches of {int(unknown.sum())} runs, e.g. {df[unknown].iloc[0]['run_dir']}"
        )
    df["n_train"] = df["n_train"].astype(int)
    df["n_full"] = df["n_full"].astype(int)
    return df


def curve_tables(
    sources: Sequence[tuple[str, str | Path]], shared: Sequence[str | Path] = (), cohorts=None
) -> pd.DataFrame:
    """Tidy runs of every protocol.  ``sources`` are ``(protocol, dir)`` pairs of subsampled runs; ``shared`` are
    directories whose *full-size* runs belong to every protocol (the fixed-epoch learning-curve dir, the main matrix).
    A full-size run found in several directories is kept once (first directory wins)."""
    protocols = [p for p, _ in sources]
    frames = []
    for protocol, root in sources:
        sub = collect([(protocol, root)], cohorts)
        frames.append(sub[~sub["full_size"] & sub["arm"].isin(MODELS)])
    lc_cohorts = set(pd.concat(frames)["cohort"]) if any(len(f) for f in frames) else set()
    full_dirs = [
        *(root for _, root in sources[:1]),
        *shared,
    ]  # the first source is the fixed-epoch learning-curve dir
    full = collect([("full", d) for d in full_dirs], cohorts if cohorts is not None else lc_cohorts)
    full = full[full["full_size"] & full["arm"].isin(MODELS)]
    full = full.drop_duplicates(["cohort", "arm", "seed"], keep="first")
    for protocol in protocols:
        part = full.copy()
        part["protocol"] = protocol
        frames.append(part)
    df = (
        pd.concat([f for f in frames if len(f)], ignore_index=True)
        if any(len(f) for f in frames)
        else pd.DataFrame(columns=TIDY_COLUMNS)
    )
    if df.empty:
        return df
    order = {m: i for i, m in enumerate(MODELS)}
    df["_m"] = df["arm"].map(lambda a: order.get(a, len(order)))
    df = (
        df.sort_values(["protocol", "cohort", "_m", "n_train", "seed"], kind="mergesort")
        .drop(columns="_m")
        .reset_index(drop=True)
    )
    return df[TIDY_COLUMNS]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD (ddof = 1) of the test AUC over the seeds per protocol / cohort / model / size."""
    rows = []
    for (protocol, cohort, arm, n), g in df.groupby(
        ["protocol", "cohort", "arm", "n_train"], sort=False
    ):
        v = g["test_auc"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        rows.append({
            "protocol": protocol, "exploratory": True, "cohort": cohort, "arm": arm, "n_train": int(n), "full_size": bool(g["full_size"].all()),
            "n_seeds": int(v.size), "auc_mean": float(v.mean()), "auc_sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "auc_min": float(v.min()), "auc_max": float(v.max()),
        })  # fmt: skip
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def cohort_label(name: str) -> str:
    tokens = {"tpp": "TPP", "fpp": "FPP", "solo": "solo", "duo": "duo", "squad": "squad"}
    return " ".join(tokens.get(t, t.capitalize()) for t in name.split("_"))


def draw(summary: pd.DataFrame, stem: Path, protocol: str, title: str) -> None:
    """One panel per cohort: mean +- SD of the test AUC over the seeds against the training matches (log x)."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    sub = summary[summary["protocol"] == protocol]
    cohorts = sorted(sub["cohort"].unique())
    fig = Figure(figsize=(7.2, 3.3), dpi=300, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(
        1, max(len(cohorts), 1), sharey=True, squeeze=False, gridspec_kw={"wspace": 0.07}
    )[0]
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.255)
    for ax, cohort in zip(axes, cohorts):
        g = sub[sub["cohort"] == cohort]
        ticks = sorted(g["n_train"].unique())
        ax.set_xscale("log")
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for arm in MODELS:
            s = g[g["arm"] == arm].sort_values("n_train")
            if s.empty:
                continue
            style = "--" if arm == "HGB-cut" else "-"
            lw = 1.5 if arm in ("M11R", "HGB-cut") else 0.9
            ax.errorbar(
                s["n_train"], s["auc_mean"], yerr=s["auc_sd"].fillna(0.0), color=COLORS[arm], ls=style, lw=lw, marker=MARKERS[arm],
                ms=3.4, mec="white", mew=0.5, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3 if arm in ("M11R", "HGB-cut") else 2,
            )  # fmt: skip
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.7)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(
            axis="both",
            labelsize=6.2,
            length=2,
            width=0.6,
            color=BASELINE,
            labelcolor=INK2,
            pad=1.5,
        )
        ax.set_xlabel("training matches", fontsize=6.6, color=INK2, labelpad=2)
        ax.text(
            0.0,
            1.03,
            cohort_label(cohort),
            transform=ax.transAxes,
            fontsize=7,
            color=INK,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        lo, hi = g["auc_mean"].min(), g["auc_mean"].max()
        pad = 0.06 * (hi - lo + 1e-9)
        ax.set_ylim(lo - 0.10 * (hi - lo + 1e-9) - pad, hi + 0.10 * (hi - lo + 1e-9) + pad)
        last = g[g["n_train"] == g["n_train"].max()]
        for arm in ("M11R", "HGB-cut"):  # direct labels on the two series the question is about
            r = last[last["arm"] == arm]
            if len(r):
                ax.annotate(arm, (float(r["n_train"].iloc[0]), float(r["auc_mean"].iloc[0])), xytext=(4, 0), textcoords="offset points",
                            fontsize=6.0, color=INK2, va="center", ha="left", annotation_clip=False)  # fmt: skip
        ax.set_xlim(min(ticks) / 1.25, max(ticks) * 1.9)  # room on the right for the direct labels
    axes[0].set_ylabel("test AUC", fontsize=6.8, color=INK2, labelpad=2)
    fig.text(
        0.085, 0.935, title, fontsize=7.4, color=INK, fontweight="bold", ha="left", va="center"
    )
    fig.text(
        0.985, 0.935, "exploratory (design 4.4)", fontsize=6.4, color=INK2, ha="right", va="center"
    )
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[a],
            ls="--" if a == "HGB-cut" else "-",
            lw=1.5 if a in ("M11R", "HGB-cut") else 0.9,
            marker=MARKERS[a],
            ms=3.4,
            label=a,
        )  # fmt: skip
        for a in MODELS
        if (sub["arm"] == a).any()
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=6.4,
        bbox_to_anchor=(0.54, 0.0),
        handlelength=2.2,
    )
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(
        0.54,
        0.075,
        "mean ± SD over seeds; the seed changes the subsample and the initialisation",
        fontsize=5.8,
        color=MUTED,
        ha="center",
        va="center",
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


def run(
    lc_dir: str | Path,
    out_dir: str | Path,
    *,
    main_dir: str | Path | None = None,
    steps_dir: str | Path | None = None,
    cohorts: Iterable[str] | None = None,
    figures: bool = True,
) -> dict:
    """Write the tidy CSV, the summary CSV and the figure(s); returns the tables."""
    out = Path(out_dir)
    sources = [("fixed_epochs", lc_dir)] + ([("steps_matched", steps_dir)] if steps_dir else [])
    df = curve_tables(sources, shared=[main_dir] if main_dir else [], cohorts=cohorts)
    if df.empty:
        raise ValueError(f"no finished learning-curve runs under {lc_dir}")
    summary = summarise(df)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "learning_curve.csv", index=False, float_format="%.10g")
    summary.to_csv(out / "learning_curve_summary.csv", index=False, float_format="%.10g")
    if figures:
        draw(
            summary,
            out / "figures" / "learning_curve",
            "fixed_epochs",
            "Learning curve: 12 epochs at every size",
        )
        if steps_dir and (summary["protocol"] == "steps_matched").any():
            draw(
                summary,
                out / "figures" / "learning_curve_steps",
                "steps_matched",
                "Learning curve: epochs scaled to the full-size number of steps",
            )
    return {"runs": df, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Learning curve (design doc 4.4): tidy CSV and figure."
    )
    ap.add_argument("--lc-dir", required=True, help="run_matrix --lc output (fixed epochs)")
    ap.add_argument(
        "--main-dir", default=None, help="main matrix output: its full-size runs complete the curve"
    )
    ap.add_argument("--steps-dir", default=None, help="run_matrix --lc --lc-steps-matched output")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "bigcomp2027_results"))
    ap.add_argument("--cohorts", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("fontTools", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    res = run(args.lc_dir, args.out_dir, main_dir=args.main_dir, steps_dir=args.steps_dir,
              cohorts=[c for c in args.cohorts.split(",") if c] if args.cohorts else None, figures=not args.no_figures)  # fmt: skip
    s = res["summary"]
    print(f"{len(res['runs'])} runs, {s['cohort'].nunique()} cohorts, outputs in {args.out_dir}")
    for (protocol, cohort), g in s.groupby(["protocol", "cohort"]):
        piv = g.pivot_table(index="arm", columns="n_train", values="auc_mean")
        print(f"\n{protocol} / {cohort}: mean test AUC")
        print(piv.reindex([m for m in MODELS if m in piv.index]).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
