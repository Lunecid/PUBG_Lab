# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = REPO_ROOT
out = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "paper/sac2027/v14_review/figures"
d = pd.read_csv(base / "reports/sac2027_patch_v14/rq1_controls/tidy_ai_adjusted.csv")
d = d[d.scale == "group_sd"]
terms = ["outside_depth", "inside_cushion", "migration_deficit", "local_crowding", "team_members"]
names = [
    "Outside depth",
    "Inside depth",
    "Movement residual",
    "Local competition",
    "Remaining members",
]  # 2026-10-08: names of the RQ1 variables in the text
plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(figsize=(3.42, 2.85))
colors = ["#2166ac", "#c65e18", "#6d4793"]
for j, (phase, color, label) in enumerate(
    zip(["early", "mid", "late"], colors, ["Early (1–2)", "Middle (3–5)", "Late (6–7)"])
):
    part = d[d.group == phase].set_index("covariate").loc[terms]
    y = np.arange(5) * 1.05 + (j - 1) * 0.23
    ax.errorbar(
        part.theta,
        y,
        xerr=[part.theta - part.ci95_low, part.ci95_high - part.theta],
        fmt=["o", "s", "^"][j],
        color=color,
        ms=3,
        capsize=1.8,
        lw=0.85,
        label=label,
    )
ax.set_yticks(np.arange(5) * 1.05, names)
ax.invert_yaxis()
ax.axvline(0, color="#9ca3af", lw=0.65, ls="--")
ax.set_xlim(-0.85, 0.65)
ax.set_xlabel("Log-odds change per within-band SD (95% CI)")
ax.tick_params(axis="y", length=0)
ax.set_xticks([-0.8, -0.4, 0, 0.4])
ax.grid(axis="x", color="#e7e9ed", lw=0.4)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.30, 1.18),
    ncol=3,
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.4,
    fontsize=6.4,
)
fig.subplots_adjust(left=0.33, right=0.985, bottom=0.19, top=0.86)
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig_rq1_effects.{ext}", dpi=220)
print("saved")
