"""Result figures (academic tone, revised spec):
  fig5 — next-phase ELIMINATION hazard AUC across game phases (all encoders + range band)
  fig6 — FINAL-RANK (placement) Spearman across game phases (all encoders + range band)
Tufte/Cleveland-McGill discipline: absolute scale, small effects small, range band,
guided readouts (spread bracket, lock-in band, direct GAT label), disclosed truncation.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#374151",
    }
)
OUT = Path("docs/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

NEUTRAL, MUTED, GRID = "#374151", "#6b7280", "#cccccc"
RANK_ACC, RANK_FILL = "#d97706", "#fef3c7"
ranges = [
    (1.5, 9.5),
    (9.7, 13.1),
    (13.3, 16.1),
    (16.3, 19.1),
    (19.3, 21.7),
    (21.9, 23.6),
    (24.0, 25.2),
]
mids = np.array([(a + b) / 2 for a, b in ranges])
zone_R = [5506, 1965, 1099, 659, 364, 188, 98]

STYLE = {  # encoder: (color, linestyle, marker)
    "No graph": ("#7f7f7f", (0, (4, 2)), "o"),
    "GCN": ("#1f77b4", "-", "o"),
    "GraphSAGE": ("#ff7f0e", "-", "s"),
    "GAT": ("#d62728", "-", "^"),
    "R-GCN": ("#2ca02c", "-", "D"),
}
HAZARD = {
    "No graph": [0.8813, 0.8167, 0.7791, 0.7985, 0.7900, 0.7823, 0.7412],
    "GCN": [0.8896, 0.8306, 0.7949, 0.8104, 0.7983, 0.7907, 0.7516],
    "GraphSAGE": [0.8871, 0.8269, 0.7944, 0.8082, 0.7984, 0.7861, 0.7477],
    "GAT": [0.8925, 0.8345, 0.7997, 0.8151, 0.8048, 0.7916, 0.7414],
    "R-GCN": [0.8891, 0.8289, 0.7952, 0.8128, 0.7998, 0.7890, 0.7407],
}
RANK = {
    "No graph": [0.6661, 0.6253, 0.6551, 0.7010, 0.7372, 0.7418, 0.6889],
    "GCN": [0.7146, 0.6628, 0.6795, 0.7155, 0.7496, 0.7606, 0.7352],
    "GraphSAGE": [0.7039, 0.6587, 0.6765, 0.7181, 0.7502, 0.7641, 0.7382],
    "GAT": [0.7716, 0.7222, 0.7267, 0.7616, 0.7933, 0.8006, 0.7627],
    "R-GCN": [0.7133, 0.6653, 0.6812, 0.7192, 0.7576, 0.7543, 0.7037],
}


def base_ax(ylim, ylab):
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    for (a, b), R in zip(ranges, zone_R):
        ax.axvspan(a, b, color="#f3f6fb", alpha=1.0, lw=0, zorder=0)
    for i, ((a, b), R) in enumerate(zip(ranges, zone_R)):
        if i % 2 == 1:
            ax.axvspan(a, b, color="#e8eef7", alpha=1.0, lw=0, zorder=0)
        ax.text(
            (a + b) / 2,
            ylim[1] + 0.012 * (ylim[1] - ylim[0]),
            f"P{i + 1}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NEUTRAL,
            clip_on=False,
        )
        ax.text(
            (a + b) / 2,
            ylim[1] + 0.001 * (ylim[1] - ylim[0]),
            f"{R:,} m",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
            clip_on=False,
        )
    ax.set_ylim(*ylim)
    ax.set_xlim(1.0, 26.5)
    ax.set_ylabel(ylab, fontsize=11.5)
    ax.set_xlabel("Game time (min) — play zone contracts", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    return fig, ax


def draw_lines(ax):
    arr = np.array([HAZARD[k] for k in STYLE])  # placeholder, replaced per fig
    return arr


# ---------- fig5: elimination ----------
fig, ax = base_ax((0.725, 0.92), "Next-phase hazard AUC")
arr = np.array([HAZARD[k] for k in STYLE])
lo, hi = arr.min(0), arr.max(0)
ax.fill_between(mids, lo, hi, color="#9e9e9e", alpha=0.14, lw=0, zorder=1)
for k, (c, ls, mk) in STYLE.items():
    ax.plot(
        mids,
        HAZARD[k],
        marker=mk,
        linestyle=ls,
        color=c,
        lw=1.7,
        ms=5,
        mec="white",
        mew=0.5,
        zorder=3,
        label=k if k != "No graph" else None,
    )
# guided readout: spread bracket at P3, no-graph direct label, cluster note
xb = mids[2]
ax.annotate(
    "",
    xy=(xb + 0.35, hi[2]),
    xytext=(xb + 0.35, lo[2]),
    arrowprops=dict(arrowstyle="<->", color=NEUTRAL, lw=1.0),
)
ax.text(
    xb + 0.6,
    (hi[2] + lo[2]) / 2,
    "encoder\nspread ≈ 0.02",
    fontsize=8.5,
    color=NEUTRAL,
    va="center",
)
ax.plot([mids[1]], [HAZARD["No graph"][1]], "o", color="#7f7f7f", ms=6, zorder=4)
ax.annotate(
    "No graph (lower envelope)",
    xy=(mids[1], HAZARD["No graph"][1]),
    xytext=(mids[1] - 0.3, 0.778),
    fontsize=8.5,
    color="#7f7f7f",
    ha="center",
    arrowprops=dict(arrowstyle="-", color="#7f7f7f", lw=0.8),
)
ax.text(
    mids[6] + 0.5,
    HAZARD["GAT"][6] + 0.004,
    "graph encoders\n(GCN·SAGE·GAT·R-GCN)",
    fontsize=8,
    color=NEUTRAL,
    va="center",
)
ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=9,
    handlelength=2.0,
    ncol=2,
    title="graph encoders",
    title_fontsize=9,
)
fig.text(
    0.012,
    0.012,
    "Lines = per-phase hazard AUC; band = encoder min–max range. "
    "y-axis truncated (full span 0.725–0.92) to show small differences honestly. Source: V3 full-DB squad.",
    fontsize=7.5,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT / "fig5_hazard_by_phase.png", dpi=160, bbox_inches="tight")
print("wrote fig5")
plt.close(fig)

# ---------- fig6: ranking ----------
fig, ax = base_ax((0.61, 0.83), "Final-rank Spearman ρ")
ax.axvspan(13.3, 23.6, color=RANK_FILL, alpha=0.55, lw=0, zorder=0.5)  # P3–P6 lock-in band
ax.text(
    18.4,
    0.815,
    "survivor narrowing (P3-6)",
    ha="center",
    fontsize=10.5,
    color=RANK_ACC,
    fontweight="bold",
)
arr = np.array([RANK[k] for k in STYLE])
lo, hi = arr.min(0), arr.max(0)
ax.fill_between(mids, lo, hi, color="#9e9e9e", alpha=0.14, lw=0, zorder=1)
for k, (c, ls, mk) in STYLE.items():
    ax.plot(
        mids,
        RANK[k],
        marker=mk,
        linestyle=ls,
        color=c,
        lw=1.7,
        ms=5,
        mec="white",
        mew=0.5,
        zorder=3,
        label=k if k not in ("GAT", "No graph") else None,
    )
# direct labels for the two informative lines
ax.text(
    mids[6] + 0.45,
    RANK["GAT"][6],
    "GAT (best ranker)",
    color="#d62728",
    fontsize=9,
    va="center",
    fontweight="bold",
)
ax.text(
    mids[6] + 0.45,
    RANK["No graph"][6] - 0.004,
    "No graph",
    color="#7f7f7f",
    fontsize=9,
    va="center",
)
# early/late spread brackets
for xi, lab in [(0, "wide early spread\n≈ 0.11"), (5, "≈ 0.06")]:
    x = mids[xi]
    ax.annotate(
        "",
        xy=(x - 0.35, hi[xi]),
        xytext=(x - 0.35, lo[xi]),
        arrowprops=dict(arrowstyle="<->", color=NEUTRAL, lw=1.0),
    )
    ax.text(
        x - 0.55, (hi[xi] + lo[xi]) / 2, lab, fontsize=8, color=NEUTRAL, va="center", ha="right"
    )
ax.legend(
    loc="lower right",
    frameon=False,
    fontsize=9,
    handlelength=2.0,
    title="other encoders",
    title_fontsize=9,
)
fig.text(
    0.012,
    0.012,
    "Lines = per-phase rank Spearman; band = encoder min–max range; amber = survivor-narrowing window. "
    "Spread is wider than for elimination (graph + rank decoder help ordering). Source: V3 full-DB squad.",
    fontsize=7.5,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig(OUT / "fig6_rank_by_phase.png", dpi=160, bbox_inches="tight")
print("wrote fig6")
plt.close(fig)
