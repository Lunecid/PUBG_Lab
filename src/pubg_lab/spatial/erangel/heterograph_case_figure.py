"""Reproducible four-scene evidence figure, plotted from saved observer frames."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
import math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_case_40_2_2026-10-15"
D = json.loads((OUT / "replay_observer.json").read_text())
im = np.asarray(Image.open(OUT / "map.jpg"))
pc = {"P1": "#41cbe5", "P2": "#b69aff", "P3": "#72dda0", "P4": "#f397b4"}
colors = {"recorded": "#41cbe5", "derived": "#f8bb5e", "unknown": "#8b98a9"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": "#e7eef7",
        "axes.labelcolor": "#cad8e9",
        "xtick.color": "#cad8e9",
        "ytick.color": "#cad8e9",
        "axes.edgecolor": "#3e4c61",
    }
)
fig = plt.figure(figsize=(18, 10), facecolor="#0c1421")
gs = fig.add_gridspec(
    2,
    4,
    height_ratios=[1, 1],
    left=0.025,
    right=0.985,
    top=0.86,
    bottom=0.08,
    wspace=0.1,
    hspace=0.18,
)
titles = [
    "04:00 | Four dispersed players",
    "12:35 | P1 vehicle evidence",
    "14:10 | Two players converge",
    "22:02 | Incoming damage evidence",
]
for j, index in enumerate(D["scene_indices"]):
    f = D["frames"][index]
    ax = fig.add_subplot(gs[0, j])
    ax.set_facecolor("#142033")
    ax.imshow(im, extent=(0, 8160, 8160, 0))
    xy = np.array([p["xy_m"] for p in f["players"] if p["position_valid"]])
    lo = xy.min(0)
    hi = xy.max(0)
    span = max(float((hi - lo).max()) + 600, 1000)
    center = (lo + hi) / 2
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] + span / 2, center[1] - span / 2)
    for p in f["players"]:
        if not p["position_valid"]:
            continue
        x, y = p["xy_m"]
        ax.add_patch(
            Rectangle(
                (int(x // 100) * 100, int(y // 100) * 100),
                100,
                100,
                facecolor=pc[p["id"]],
                edgecolor=pc[p["id"]],
                alpha=0.22,
            )
        )
        ax.scatter(x, y, c=pc[p["id"]], s=35, edgecolors="#0c1421", zorder=5)
        ax.annotate(
            p["id"],
            (x, y),
            xytext=(8, 12 if p["id"] == "P1" else -14),
            textcoords="offset points",
            color=pc[p["id"]],
            weight="bold",
            fontsize=10,
            bbox=dict(facecolor="#0c1421", edgecolor="none", alpha=0.8, pad=1),
        )
    for z in (f["game"] or {}).get("zones", []):
        ax.add_patch(
            Circle(
                z["xy_m"],
                z["radius_m"],
                fill=False,
                color="white" if z["id"] == "safe_zone" else "#79afff",
                lw=1,
                ls="--",
                alpha=0.6,
            )
        )
    ax.scatter(*f["team"]["center_m"], marker="D", c=colors["derived"], s=32, zorder=4)
    ax.add_patch(
        Circle(
            f["team"]["center_m"],
            f["team"]["radius_m"],
            fill=False,
            color=colors["derived"],
            ls=":",
            lw=1,
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[j], loc="left", color="#edf5ff", fontsize=12, pad=12)
    ax.set_xlabel(
        f"{len(f['team']['valid_members'])} valid members | max separation {f['team']['max_pair_distance_m']:,.0f} m\nView width {span / 1000:.1f} km; positions sampled at different times",
        fontsize=9,
    )
    ax = fig.add_subplot(gs[1, j])
    ax.set_facecolor("#111e2f")
    loc = {}
    counts = {}
    for n in f["nodes"]:
        typ = n["type"]
        i = counts.get(typ, 0)
        counts[typ] = i + 1
        if typ == "player":
            loc[n["id"]] = (3.1, 1.1 + int(n["id"][1]) * 1.35)
        elif typ == "team":
            loc[n["id"]] = (5.0, 4.5)
        elif typ == "equipment":
            loc[n["id"]] = (0.8, 1.2 + i * 0.65)
        elif typ == "vehicle":
            loc[n["id"]] = (0.8, 6.0)
        elif typ == "contact":
            loc[n["id"]] = (0.8, 7.2 + i * 0.6)
        elif typ == "cell":
            loc[n["id"]] = (7.0, 1.6 + i * 1.3)
        elif typ == "region":
            loc[n["id"]] = (9.0, 1.6 + i * 1.3)
        elif typ == "zone":
            loc[n["id"]] = (7.0 + i * 2, 8.0)
        elif typ == "game":
            loc[n["id"]] = (5.0, 8.0)
    for e in f["edges"]:
        a, b = loc[e["source"]], loc[e["target"]]
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=colors[e["kind"]],
            alpha=0.3,
            lw=0.8,
            ls="-" if e["kind"] == "recorded" else "--",
            zorder=1,
        )
    for n in f["nodes"]:
        x, y = loc[n["id"]]
        typ = n["type"]
        label = (
            n["id"]
            if typ in ("player", "cell", "region")
            else n["label"]
            if typ == "equipment"
            else {
                "team": "TEAM",
                "game": "GAME",
                "zone": n["id"].replace("_zone", ""),
                "vehicle": "vehicle",
                "contact": "hit evidence",
            }[typ]
        )
        if typ == "equipment":
            label = (
                label.replace("Head_F_01_Lv2", "Helmet L2")
                .replace("Armor_D_01_Lv2", "Vest L2")
                .replace("Armor_C_01_Lv3", "Vest L3")
            )
        ax.scatter(
            x,
            y,
            s=150 if typ == "team" else 65,
            marker="D" if typ == "team" else "s" if typ in ("cell", "region", "equipment") else "o",
            facecolor="#182a41",
            edgecolor=colors[n["kind"]],
            lw=1.4,
            zorder=3,
        )
        ax.text(x, y + 0.29, label, ha="center", va="top", fontsize=7, color=colors[n["kind"]])
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(9, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{len(f['nodes'])} nodes / {len(f['edges'])} typed edges", fontsize=10)
fig.text(
    0.025, 0.953, "One real match, four evolving heterogeneous graphs", fontsize=24, weight="bold"
)
fig.text(
    0.025,
    0.916,
    "Erangel / 2026-03-31 / human observer P1 / training split / patch 40.2 inferred by date",
    fontsize=12,
    color="#adbed2",
)
fig.text(
    0.025,
    0.035,
    "Cyan: recorded observable-information proxy   |   Amber: derived state / geometry prior   |   Grey: unknown or dead/stale",
    fontsize=11,
)
fig.text(
    0.025,
    0.012,
    "Team center is not a traversable destination. No opponent positions or teammate private loadout enter these graphs. Historical map geometry is unverified for 40.2.",
    fontsize=9,
    color="#adbed2",
)
fig.savefig(OUT / "four_scenes.png", dpi=160, facecolor=fig.get_facecolor())
fig.savefig(OUT / "four_scenes.pdf", facecolor=fig.get_facecolor())
print("Saved four_scenes.png and PDF")
