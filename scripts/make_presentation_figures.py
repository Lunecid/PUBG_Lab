"""Presentation/paper figures from verified V3 results (see docs/PRESENTATION_DATA_2026-06-21.md).
Outputs PNG (150 dpi) to docs/presentation_figures/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/presentation_figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12})
BLUE, RED, GREEN, GRAY, PURP = "#2563eb", "#dc2626", "#16a34a", "#9ca3af", "#7c3aed"


def fig_resolution():
    labels = ["HGB\n(tabular)", "phase_tcn", "phase_rgcn", "mr_tcn\n(snapshot)", "mr_rgcn\n(snapshot+graph)"]
    vals = [0.7864, 0.7872, 0.7926, 0.8349, 0.8467]
    cols = [GRAY, GRAY, GRAY, BLUE, BLUE]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.793, ls="--", c=RED, lw=1.2)
    ax.text(0.05, 0.797, "phase-aggregation ceiling ≈0.79", color=RED, fontsize=10)
    ax.annotate("", xy=(4, 0.8467), xytext=(2, 0.7926),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(2.55, 0.818, "+0.054\nsnapshot breaks\nthe ceiling", color=GREEN, fontsize=10, fontweight="bold")
    ax.set_ylim(0.74, 0.86); ax.set_ylabel("Next-phase hazard AUC")
    ax.set_title("Multi-resolution snapshots break the phase-aggregation ceiling\n(V3 full-DB squad, shared test split)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig1_resolution_ladder.png", dpi=150); plt.close(fig)


def fig_graph_null():
    labels = ["no_graph", "observed\n(real topology)", "edge_shuffle\n(random null)"]
    vals = [0.834, 0.845, 0.847]
    cols = [GRAY, BLUE, RED]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0008, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0.825, 0.855); ax.set_ylabel("Next-phase hazard AUC")
    ax.annotate("random ≥ observed\n(3/3 seeds)", xy=(2, 0.847), xytext=(1.1, 0.851),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), color=RED, fontsize=10, fontweight="bold")
    ax.text(0.5, 0.838, "graph helps (+0.011)\n= mean-field SMOOTHING,\nnot topology", fontsize=10, color=PURP,
            ha="center", bbox=dict(boxstyle="round", fc="#f3e8ff", ec=PURP))
    ax.set_title("Dynamic-graph gain is smoothing, not relational topology\n(null-model controls, all row-normalized)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig2_graph_null.png", dpi=150); plt.close(fig)


def fig_rank_lockin():
    phases = [1, 2, 3, 4, 5, 6, 7]
    spr = [0.771, 0.723, 0.727, 0.766, 0.800, 0.813, 0.785]
    lo = [0.762, 0.712, 0.717, 0.754, 0.788, 0.797, 0.758]
    hi = [0.780, 0.734, 0.739, 0.777, 0.812, 0.826, 0.809]
    mins = [9.5, 13.1, 16.1, 19.1, 21.7, 23.6, 25.2]  # phase end minute
    err = [np.array(spr) - np.array(lo), np.array(hi) - np.array(spr)]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(phases, spr, yerr=err, marker="o", color=BLUE, lw=2, capsize=4, label="rank Spearman (±95% CI)")
    ax.axvspan(4, 6, color=GREEN, alpha=0.12)
    ax.text(5, 0.70, "decision window\nphase 4–6\n(≈16–24 min)", ha="center", color=GREEN, fontsize=10, fontweight="bold")
    ax.annotate("+0.085 sig.\n(3→6)", xy=(6, 0.813), xytext=(4.2, 0.835),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5), color=GREEN, fontsize=10, fontweight="bold")
    ax.text(6.05, 0.778, "plateau 5→6 (ns)\ndrop at 7", fontsize=9, color=GRAY)
    ax.set_xlabel("cut phase"); ax.set_ylabel("final-rank alignment (Spearman)")
    ax.set_ylim(0.68, 0.85); ax.set_xticks(phases); ax.grid(alpha=0.3)
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(phases); secax.set_xticklabels([f"{m:.0f}m" for m in mins]); secax.set_xlabel("game time (phase end)")
    ax.set_title("Final standing locks in during the mid-late zone collapse (rank lock-in)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig3_rank_lockin.png", dpi=150); plt.close(fig)


def fig_hazard_ci():
    phases = [1, 2, 3, 4, 5, 6, 7]
    v3 = [0.889, 0.826, 0.792, 0.811, 0.798, 0.794, 0.745]
    v3lo = [0.884, 0.819, 0.784, 0.802, 0.786, 0.780, 0.715]
    v3hi = [0.895, 0.835, 0.801, 0.820, 0.808, 0.808, 0.772]
    hgb = [0.825, 0.748, 0.725, 0.756, 0.742, 0.720, 0.655]
    e3 = [np.array(v3) - np.array(v3lo), np.array(v3hi) - np.array(v3)]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(phases, v3, yerr=e3, marker="o", color=BLUE, lw=2, capsize=4, label="V3 MR-GNN (±95% CI)")
    ax.plot(phases, hgb, marker="s", color=RED, lw=2, ls="--", label="HGB tabular baseline")
    for p, a, b in zip(phases, v3, hgb):
        ax.annotate(f"+{a-b:.3f}", (p, (a + b) / 2), fontsize=8, color=GREEN, ha="center")
    ax.axhline(0.5, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("cut phase"); ax.set_ylabel("Next-phase hazard AUC")
    ax.set_xticks(phases); ax.set_ylim(0.6, 0.92); ax.grid(alpha=0.3)
    ax.set_title("V3 significantly beats strong tabular baseline at every cut phase\n(all ΔAUC 95% CIs exclude 0)")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig4_hazard_vs_baseline_ci.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_resolution(); fig_graph_null(); fig_rank_lockin(); fig_hazard_ci()
    print("wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p)
