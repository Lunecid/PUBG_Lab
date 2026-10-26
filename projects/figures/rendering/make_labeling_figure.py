"""Figure: how the two prediction targets are labeled (leakage-safe, cut-phase).
Schematic grounded in stdb.team_phase_labels:
  hazard_next  = at cut_phase p (team alive at end of p), is the team eliminated
                 during phase p+1?  (1/0)            -> ELIMINATION target
  final_rank   = the team's final placement (1 = winner), broadcast to all p
                 -> RANKING target (final_top3/5/10, winner derived from it)
  target_defined (at-risk mask) = only cells with team alive at end of p and a
                 next phase are scored.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
OUT = Path("docs/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

ALIVE, DEAD, OBS, ACC = "#2ca02c", "#d62728", "#dbeafe", "#374151"
phases = [1, 2, 3, 4, 5, 6, 7]

fig, ax = plt.subplots(figsize=(11.5, 5.6))
ax.set_xlim(0.0, 9.6)
ax.set_ylim(-0.2, 4.4)
ax.axis("off")

# phase header
for p in phases:
    ax.text(
        p, 4.15, f"Phase {p}", ha="center", va="center", fontsize=10, color=ACC, fontweight="bold"
    )
ax.text(8.6, 4.15, "Match end", ha="center", va="center", fontsize=10, color=ACC, fontweight="bold")

# two example teams: alive_at_phase_end state per phase
teams = [
    (
        "Team A",
        3.0,
        [True, True, True, True, False, False, False],
        8,
        False,
    ),  # eliminated in phase 5
    ("Team B", 1.4, [True, True, True, True, True, True, True], 1, True),  # winner
]


def cell(x, y, alive):
    c = ALIVE if alive else DEAD
    ax.add_patch(
        Rectangle(
            (x - 0.34, y - 0.22),
            0.68,
            0.44,
            facecolor=c,
            alpha=0.85,
            edgecolor="white",
            lw=1.2,
            zorder=2,
        )
    )
    ax.text(
        x,
        y,
        "alive" if alive else "elim.",
        ha="center",
        va="center",
        color="white",
        fontsize=8.5,
        zorder=3,
    )


for name, y, alive_end, rank, winner in teams:
    ax.text(0.05, y, name, ha="left", va="center", fontsize=11, fontweight="bold")
    for p, a in zip(phases, alive_end):
        cell(p, y, a)
    # hazard_next labels (only where target_defined: alive at end of p AND a next phase exists)
    for i, p in enumerate(phases[:-1]):
        if alive_end[i]:  # team alive at end of cut_phase p
            hz = 0 if alive_end[i + 1] else 1
            col = DEAD if hz else ALIVE
            ax.text(
                p + 0.5,
                y - 0.42,
                f"{hz}",
                ha="center",
                va="center",
                fontsize=10,
                color=col,
                fontweight="bold",
            )
    # final rank (broadcast)
    tag = f"#{rank}  (winner)" if winner else f"#{rank}"
    ax.text(
        8.6,
        y,
        tag,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=ACC,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#d97706", lw=1),
    )

ax.text(
    1.0,
    2.18,
    "hazard_next  (between phases):",
    ha="left",
    fontsize=8.5,
    color="#555555",
    style="italic",
)

# illustrate the cut-phase causal mechanic on Team A at cut_phase 4
ax.add_patch(
    Rectangle(
        (0.55, 2.55), 3.9, 0.9, facecolor=OBS, alpha=0.6, edgecolor="#60a5fa", lw=1.2, zorder=1
    )
)
ax.text(
    2.5, 3.62, "observed  ≤  cut_phase 4  (model input)", ha="center", fontsize=9, color="#1d4ed8"
)
ax.add_patch(
    FancyArrowPatch(
        (4.45, 3.0),
        (5.0, 3.0),
        arrowstyle="-|>",
        mutation_scale=16,
        color="#1d4ed8",
        lw=2,
        zorder=4,
    )
)
ax.text(5.0, 2.62, "predict\nphase 5", ha="center", va="top", fontsize=8.5, color="#1d4ed8")

# legend / key
ax.text(0.05, 0.55, "Targets:", fontsize=10.5, fontweight="bold")
ax.text(
    0.05,
    0.18,
    "ELIMINATION  =  hazard_next  ∈ {0,1}  :  alive at end of cut_phase p, eliminated during phase p+1 ?",
    fontsize=9.5,
    color=ACC,
)
ax.text(
    0.05,
    -0.12,
    "RANKING  =  final_rank (1 = winner), normalized; final_top3/5/10 derived — broadcast to every cut_phase.",
    fontsize=9.5,
    color=ACC,
)
ax.text(
    5.1,
    0.55,
    "At-risk mask (target_defined): only cells with the team alive at end of p\n"
    "and a next phase exist are scored — dead teams are not predicted.",
    fontsize=8.5,
    color="#666666",
)

ax.set_title(
    "Leakage-safe labeling: features observed through cut_phase p, predict phase p+1",
    fontsize=13,
    pad=12,
)
fig.tight_layout()
fig.savefig(OUT / "fig0_labeling.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "fig0_labeling.png")
