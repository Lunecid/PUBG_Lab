#!/usr/bin/env python
"""Figures of the exploratory PTR-Former interpretability analysis (reads the CSVs written by interp.py).

fig_relation_bias.{pdf,png}   learned relation biases, raw and effective, M11R / M11RZ vs the shuffled control
fig_attention_lift.{pdf,png}  team-axis attention lift by pair class, M11R vs M11 (seed 42)
fig_zone_bias_curve.{pdf,png} learned team<->zone bias vs current / next zone d/r (M11RZ seed 42) + zone-token attention
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUT = Path("/Users/lunecid/project/PUBG/reports/bigcomp2027_results/interpretability")
CH = ["prox", "combat_out", "combat_in", "copress"]
COL = {"M11R": "#0072B2", "M11RZ": "#009E73", "M11R-shuf": "#8c8c8c", "M11": "#D55E00"}
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.frameon": False, "savefig.dpi": 220, "pdf.fonttype": 42,
})  # fmt: skip


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_relation_bias() -> None:
    df = pd.read_csv(OUT / "relation_bias_long.csv")
    ws = pd.read_csv(OUT / "relation_bias_edge_weight_stats.csv")
    wm = ws.groupby("channel")[["w_median", "w_p90"]].mean()
    panels = [("b_raw", "(a) raw b  (per block x head x run)", "linear", None),
              ("eff_med", "(b) effective bias = b x median edge weight", "symlog", "w_median"),
              ("eff_p90", "(c) effective bias = b x p90 edge weight", "symlog", "w_p90")]  # fmt: skip
    arms = ["M11R", "M11RZ", "M11R-shuf"]
    offs = {"M11R": -0.27, "M11RZ": 0.0, "M11R-shuf": 0.27}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))
    rng = np.random.default_rng(0)
    for ax, (col, title, scale, wcol) in zip(axes, panels):
        for arm in arms:
            for ci, ch in enumerate(CH):
                v = df[(df["arm"] == arm) & (df["channel"] == ch)][col].to_numpy()
                x0 = ci + offs[arm]
                ax.boxplot([v], positions=[x0], widths=0.2, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="none", edgecolor=COL[arm], linewidth=1.1),
                           medianprops=dict(color=COL[arm], linewidth=1.8), whiskerprops=dict(color=COL[arm]),
                           capprops=dict(color=COL[arm]))  # fmt: skip
                ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), v, s=5, color=COL[arm], alpha=0.35, linewidths=0, zorder=3)
        ax.axhline(0, color="k", lw=0.6, ls="--", zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels([c if wcol is None else f"{c}\nw={wm.loc[c, wcol]:.3g}" for c in CH])
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylabel("logit bias added per edge (symlog, linear below 0.01)")
        else:
            ax.set_ylabel("b  (logit units per unit edge weight)")
        ax.set_title(title, loc="left")
    handles = [Line2D([0], [0], color=COL[a], lw=2, label=l) for a, l in
               [("M11R", "M11R (real edges)"), ("M11RZ", "M11RZ (real edges)"), ("M11R-shuf", "M11R-shuf (shuffled edges)")]]  # fmt: skip
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    n = df.groupby("arm")["cohort"].apply(lambda s: s.nunique()).to_dict()
    nr = df.groupby("arm").apply(lambda g: g[["cohort", "seed"]].drop_duplicates().shape[0]).to_dict()
    fig.suptitle(f"Learned relation biases of PTR-Former checkpoints (EXPLORATORY, descriptive). Points = block x head x run; "
                 f"{nr['M11R']} runs/arm over {n['M11R']} cohorts; x-label w = mean over cohorts of the test-dump edge weight", fontsize=9, y=1.02)  # fmt: skip
    fig.tight_layout()
    save(fig, "fig_relation_bias")


def fig_attention_lift() -> None:
    df = pd.read_csv(OUT / "attention_lift_by_cohort.csv")
    h = df[(df["head"] == "mean") & (df["subset"] == "all_alive") & df["arm"].isin(["M11R", "M11"])].copy()
    both = h.groupby(["cohort", "arm", "class"], as_index=False)["mean_lift"].mean()
    both["block"] = "both"
    h["block"] = h["block"].map(lambda b: f"L{b}")
    h = pd.concat([h[["cohort", "arm", "class", "block", "mean_lift"]], both], ignore_index=True)
    classes = ["combat", "prox_only", "prox_copress", "none"]  # copress-only pairs do not exist: copress edges always co-occur with prox
    labels = ["combat\n(either dir.)", "prox-only", "copress\n(+prox)", "no edge"]
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, blk, title in zip(axes, ["L0", "L1", "both"], ["block 1 (L0)", "block 2 (L1)", "mean of the two blocks"]):
        for arm, dx in (("M11R", -0.19), ("M11", 0.19)):
            for ci, cl in enumerate(classes):
                v = h[(h["block"] == blk) & (h["arm"] == arm) & (h["class"] == cl)]["mean_lift"].dropna().to_numpy()
                if len(v) == 0:
                    continue
                ax.scatter(ci + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color=COL[arm], alpha=0.5, linewidths=0, zorder=3)
                ax.hlines(v.mean(), ci + dx - 0.15, ci + dx + 0.15, color=COL[arm], lw=2.4, zorder=4)
        ax.axhline(1.0, color="k", lw=0.7, ls="--", zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 6.0)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("attention lift (1 = uniform)")
    ncoh = h["cohort"].nunique()
    axes[0].legend(handles=[Line2D([0], [0], color=COL["M11R"], lw=2.4, label="M11R (relation bias)"),
                            Line2D([0], [0], color=COL["M11"], lw=2.4, label="M11 (no relation bias)")], loc="upper right", fontsize=8)  # fmt: skip
    fig.suptitle("Team-axis attention lift by pair class: EXPLORATORY, descriptive (seed 42, <=300 test matches per cohort, mean over heads)\n"
                 f"lift = mean attention per class key / (1 / alive keys); dots = {ncoh} cohorts, bar = cohort mean; combat = edge in either direction; "
                 "no copress-only pairs exist (copress edges always co-occur with prox)", fontsize=8.5, y=1.07)  # fmt: skip
    fig.tight_layout()
    save(fig, "fig_attention_lift")


def fig_zone_bias_curve() -> None:
    cm = pd.read_csv(OUT / "zone_bias_curve_mean.csv")
    z = pd.read_csv(OUT / "attention_zone_token.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    cohorts = sorted(c for c in cm["cohort"].unique() if c != "mean_over_cohorts")
    for ax, which, title in zip(axes[:2], ["cur", "next"], ["(a) vs current-zone d/r (next-zone input at its median)",
                                                             "(b) vs next-zone d/r (current-zone input at its median)"]):  # fmt: skip
        for c in cohorts:
            g = cm[(cm["cohort"] == c) & (cm["variant"] == "median") & (cm["which"] == which)].sort_values("d")
            ax.plot(g["d"], g["bias_mean"], color=COL["M11RZ"], lw=0.8, alpha=0.45)
        m = cm[(cm["cohort"] == "mean_over_cohorts") & (cm["variant"] == "median") & (cm["which"] == which)].sort_values("d")
        p = cm[(cm["cohort"] == "mean_over_cohorts") & (cm["variant"] == "pdp") & (cm["which"] == which)].sort_values("d")
        ax.plot(m["d"], m["bias_mean"], color="k", lw=2.2, label="mean over cohorts (other inputs at medians)")
        ax.plot(p["d"], p["bias_mean"], color="#D55E00", lw=1.6, ls="--", label="mean over cohorts, partial dependence")
        ax.axvline(1.0, color="0.5", lw=0.7, ls=":")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlim(0, 2)
        ax.set_xlabel("d / r  (1 = zone boundary)")
        ax.set_ylabel("team<->zone logit bias, mean over heads and blocks")
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[0].legend(handles=[Line2D([0], [0], color=COL["M11RZ"], lw=1, label=f"one cohort (n={len(cohorts)})")] + axes[0].get_legend_handles_labels()[0],
                   fontsize=7.5, loc="upper left")  # fmt: skip
    # (c) zone-token attention lift, inside vs outside the next zone, per block
    ax = axes[2]
    zz = z[(z["head"] == "mean") & (z["subset"] == "all_alive") & z["group"].isin(["next_inside", "next_outside"])]
    pos = {("0", "next_inside"): 0.0, ("0", "next_outside"): 1.0, ("1", "next_inside"): 2.6, ("1", "next_outside"): 3.6}
    for blk in (0, 1):
        pv = zz[zz["block"] == blk].pivot(index="cohort", columns="group", values="mean_lift").dropna()
        x_in, x_out = pos[(str(blk), "next_inside")], pos[(str(blk), "next_outside")]
        for _, r in pv.iterrows():
            ax.plot([x_in, x_out], [r["next_inside"], r["next_outside"]], color=COL["M11RZ"], lw=0.8, alpha=0.5, marker="o", ms=2.5)
        ax.plot([x_in, x_out], [pv["next_inside"].mean(), pv["next_outside"].mean()], color="k", lw=2.2, marker="o", ms=5, zorder=5)
    ax.axhline(1.0, color="0.5", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(["inside\nnext zone", "outside\nnext zone", "inside\nnext zone", "outside\nnext zone"])
    ax.text(0.5, 1.02, "block 1 (L0)", ha="center", transform=ax.get_xaxis_transform(), fontsize=8.5)
    ax.text(3.1, 1.02, "block 2 (L1)", ha="center", transform=ax.get_xaxis_transform(), fontsize=8.5)
    ax.set_ylabel("attention lift on the zone token (1 = uniform)")
    ax.set_title("(c) zone-token attention vs next-zone membership", loc="left", fontsize=8.5, pad=24)
    fig.suptitle("M11RZ seed 42: learned team<->zone bias and zone-token attention (EXPLORATORY, descriptive; d/r from the team centre to the zone circle centre / radius)",
                 fontsize=9, y=1.03)  # fmt: skip
    fig.tight_layout()
    save(fig, "fig_zone_bias_curve")


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["rel", "att", "zone"]
    if "rel" in which:
        fig_relation_bias()
    if "att" in which:
        fig_attention_lift()
    if "zone" in which:
        fig_zone_bias_curve()
