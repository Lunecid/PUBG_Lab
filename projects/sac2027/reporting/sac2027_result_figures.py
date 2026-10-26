#!/usr/bin/env python3
"""Rebuild the v14 contrast figure from only the eleven primary patch cohorts."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
from pathlib import Path
import sys

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
import projects.battle_royale.reporting.make_result_figures_v2 as F

BASE = REPO / "reports/sac2027_patch_v14"
OUT = REPO / "paper/sac2027/v14_review/figures"


def plot(phase, paths):
    # Reuse the publication's drawing primitives, but choose axes from the new
    # results; the old fixed limits must not crop a changed patch result.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    W, H = 17.8, 5.8
    bottom = 2.2
    height = H - 0.55 - bottom
    fig = F.new_figure(W, H)
    axa = F.add_axes_cm(fig, 1.0, bottom, 3.35, height)
    axs = F.add_axes_cm(fig, 5.78, bottom, 3.35, height)
    axk = F.add_axes_cm(fig, 10.5, bottom, 7.25, height)
    handles, labels = F.draw_panel_a(axa, phase)
    la = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.12 / W, (bottom - 0.66) / H),
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.08,
        borderpad=0,
        borderaxespad=0,
        fontsize=7,
    )
    for ax, key, start_color, end_color, key_steps in (
        (axs, "sagong", F.SAGONG, F.OURS, F.SAGONG_KEY_STEPS),
        (axk, "krafton", F.KRAFTON_C, F.PTR, F.KRAFTON_KEY_STEPS),
    ):
        d = paths[key]
        levels = np.r_[d["start"], d["start"] + np.cumsum([s["md"] for s in d["steps"]]), d["end"]]
        lo = np.floor(levels.min() / 0.02) * 0.02
        hi = np.ceil(levels.max() / 0.02) * 0.02 + 0.02
        F.draw_waterfall(
            ax,
            d,
            (lo, hi),
            np.arange(lo, hi + 1e-8, 0.02).tolist(),
            start_color,
            end_color,
            35,
            key_steps,
        )
        ax.set_ylabel(ax.get_ylabel().replace("12-cohort", "11-cohort"))
    handles = [
        Patch(facecolor=F.STRONG_BLUE, linewidth=0),
        Patch(facecolor=F.LIGHT_BLUE, linewidth=0),
        Patch(facecolor=F.SAGONG, edgecolor="white", hatch="////", linewidth=0),
        Line2D([0], [0], color=F.INK, lw=0.8, marker="_", markersize=4.5, markeredgewidth=0.8),
    ]
    lk = axk.legend(
        handles,
        ["Key step", "Other step", "Negative step", "95% CI"],
        loc="lower left",
        bbox_to_anchor=(0.48, 0.04),
        frameon=False,
        handlelength=1.1,
        handleheight=0.8,
        handletextpad=0.5,
        labelspacing=0.15,
        borderpad=0,
        borderaxespad=0,
        fontsize=7,
    )
    for x, letter, title in (
        (0.06, "(a)", "Per-phase calibration"),
        (4.58, "(b)", "Decomposition, Sagong path"),
        (9.36, "(c)", "Decomposition, KRAFTON path"),
    ):
        F.panel_title(fig, x, H - 0.30, letter, title)
    fig._legends_to_check = [la, lk]
    return fig


def main():
    counts = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = sorted(counts.loc[counts.analysis_role == "main", "cohort"])
    assert len(cohorts) == 11
    F.RUNS = BASE / "checkpoints/ptr_v2"
    primary = BASE / "primary_results_v2"
    F.SRC.update(
        contrasts_pooled=primary / "contrasts_pooled.csv",
        contrasts_per_cohort=primary / "contrasts_per_cohort.csv",
        arm_per_cohort=primary / "arm_metrics_per_cohort.csv",
        e2_pooled=primary / "supplement/e2_contrasts_pooled_dl.csv",
    )

    def values(apc, arm, metric):
        g = apc[
            (apc.arm == arm) & (apc.seed == 42) & (apc.stratum == "all") & (apc.metric == metric)
        ]
        assert not g.cohort.duplicated().any() and sorted(g.cohort) == cohorts
        return g.set_index("cohort").value.sort_index()

    F.arm_cohort_values = values
    apc = F.read_csv("arm_per_cohort")
    mean = lambda arm: float(values(apc, arm, "auc").mean())
    cp = F.read_csv("contrasts_pooled")
    cp = cp[(cp.metric == "auc") & (cp.stratum == "all")].set_index("contrast")
    e2 = F.read_csv("e2_pooled").set_index("contrast")
    paths = {}
    steps = []
    for label, contrast, to, frm in F.KRAFTON_STEPS:
        r = cp.loc[contrast]
        assert int(r["k"]) == len(cohorts)
        w = e2.loc[contrast] if contrast in e2.index else None
        steps.append(
            dict(
                label=label,
                contrast=contrast,
                to=to,
                **{"from": frm},
                theta=float(r.theta),
                ci95_low=float(r.ci95_low),
                ci95_high=float(r.ci95_high),
                k=int(r["k"]),
                same_sign=int(r.same_sign),
                mean_difference=mean(to) - mean(frm),
                wmp_theta=float(w.theta) if w is not None else np.nan,
                wmp_ci95_low=float(w.ci95_low) if w is not None else np.nan,
                wmp_ci95_high=float(w.ci95_high) if w is not None else np.nan,
                **F.mean_step(apc, to, frm),
            )
        )
    start, end = mean(F.KRAFTON_START[1]), mean(F.KRAFTON_END[1])
    total = sum(s["theta"] for s in steps)
    paths["krafton"] = dict(
        start_label=F.KRAFTON_START[0],
        start=start,
        end_label=F.KRAFTON_END[0],
        end=end,
        steps=steps,
        sum_steps=total,
        end_minus_start=end - start,
        mismatch=total - (end - start),
    )
    pooled = F.sagong_steps(cohorts)
    steps = []
    for (label, to, frm), key in zip(F.SAGONG_STEPS, ("hab", "rest")):
        steps.append(
            dict(
                label=label,
                contrast=to + " - " + frm,
                to=to,
                **{"from": frm},
                **pooled[key],
                mean_difference=mean(to) - mean(frm),
                **F.mean_step(apc, to, frm),
            )
        )
    start, end = mean(F.SAGONG_START[1]), mean(F.SAGONG_END[1])
    total = sum(s["theta"] for s in steps)
    paths["sagong"] = dict(
        start_label=F.SAGONG_START[0],
        start=start,
        end_label=F.SAGONG_END[0],
        end=end,
        steps=steps,
        sum_steps=total,
        end_minus_start=end - start,
        mismatch=total - (end - start),
        pooled_total=pooled["total"]["theta"],
    )
    for path in paths.values():
        assert abs(sum(s["md"] for s in path["steps"]) - (path["end"] - path["start"])) < 1e-10
    phase = F.per_phase_table(cohorts)
    phase.to_csv(OUT / "fig_contrast_panel_a_per_phase.csv", index=False)
    F.waterfall_frame(paths).to_csv(OUT / "fig_contrast_panels_bc_steps.csv", index=False)
    F.setup_style()
    fig = plot(phase, paths)
    notes = F.save_and_check(fig, OUT / "fig_contrast", 17.8, 9.0, True)
    (OUT / "fig_contrast_qa.txt").write_text("\n".join(notes) + "\n")
    print("Rebuilt contrast figure from 11 primary patch cohorts.", notes, flush=True)


if __name__ == "__main__":
    main()
