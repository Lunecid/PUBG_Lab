"""Generate a paper figure showing a phase snapshot as a dynamic graph."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MATCH_PATH = ROOT / "webapp/public/data/matches/3faf93bf-bd7b-4ff3-9e13-2083267f3556.json"
MAP_PATH = ROOT / "webapp/public/maps/baltic_main.jpg"
OUT_PATH = ROOT / "paper/figures/dynamic_heterogeneous_graph_example.png"
PHASE = 6

COLOR_PROX = "#2c6fb7"
COLOR_COMBAT = "#cf2f2f"
COLOR_PRESSURE = "#7b3fb2"
COLOR_ZONE = "#111111"
NODE_FACE = "#ffffff"
NODE_EDGE = "#111111"


def recover_phase_starts(frames: list[dict]) -> tuple[list[int], list[tuple[int, int, dict]]]:
    phase_by_frame: list[int] = []
    phase_starts: list[tuple[int, int, dict]] = []
    phase = 0
    last_target = None
    seen_target = False

    for frame_index, frame in enumerate(frames):
        zone = frame.get("zone", {})
        target = (
            round(float(zone.get("px") or 0), 1),
            round(float(zone.get("py") or 0), 1),
            round(float(zone.get("pr") or 0), 1),
        )
        if target[2] > 0 and not seen_target:
            seen_target = True
            phase = 1
            last_target = target
            phase_starts.append((phase, frame_index, frame))
        elif target[2] > 0 and target != last_target:
            phase += 1
            last_target = target
            phase_starts.append((phase, frame_index, frame))
        phase_by_frame.append(phase)

    return phase_by_frame, phase_starts


def phase_interval(frames: list[dict], phase_starts: list[tuple[int, int, dict]], phase: int) -> tuple[int, int]:
    start = next(frame_index for p, frame_index, _ in phase_starts if p == phase)
    later = [frame_index for p, frame_index, _ in phase_starts if p == phase + 1]
    end = later[0] - 1 if later else len(frames) - 1
    return start, end


def draw_arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str,
    lw: float,
    alpha: float = 0.9,
    rad: float = 0.0,
    arrow: bool = True,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    style = "-|>" if arrow else "-"
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=9,
        shrinkB=9,
        zorder=zorder,
    )
    ax.add_patch(patch)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(MATCH_PATH.read_text())
    frames = data["frames"]
    nodes = {int(node["id"]): node for node in data["nodes"]}
    arena = float(data.get("arena_m", 8160))
    _, phase_starts = recover_phase_starts(frames)
    start_idx, end_idx = phase_interval(frames, phase_starts, PHASE)
    frame = frames[(start_idx + end_idx) // 2]
    t0, t1 = float(frames[start_idx]["t"]), float(frames[end_idx]["t"])
    alive_ids = [int(team_id) for team_id in frame["alive"] if str(team_id) in frame["cx"]]
    positions = {
        team_id: (float(frame["cx"][str(team_id)]), float(frame["cy"][str(team_id)]))
        for team_id in alive_ids
    }

    combat_weight: dict[tuple[int, int], float] = defaultdict(float)
    for src, dst, event_t, damage in data["events"]:
        src_i, dst_i = int(src), int(dst)
        if t0 <= float(event_t) <= t1 and src_i in positions and dst_i in positions:
            combat_weight[(src_i, dst_i)] += float(damage)
    combat_edges = sorted(combat_weight.items(), key=lambda item: item[1], reverse=True)[:5]

    proximity_edges: list[tuple[int, int, float]] = []
    for i, src in enumerate(alive_ids):
        for dst in alive_ids[i + 1 :]:
            dist = math.hypot(positions[src][0] - positions[dst][0], positions[src][1] - positions[dst][1])
            if dist <= 170:
                proximity_edges.append((src, dst, dist))

    zone = frame["zone"]
    zx, zy = float(zone["px"]), float(zone["py"])
    safe_r = float(zone["pr"])
    pressure_teams = []
    for team_id, (x, y) in positions.items():
        boundary_distance = math.hypot(x - zx, y - zy) - safe_r
        if boundary_distance >= -80:
            pressure_teams.append(team_id)
    co_pressure_edges: list[tuple[int, int]] = []
    for i, src in enumerate(pressure_teams):
        for dst in pressure_teams[i + 1 :]:
            dist = math.hypot(positions[src][0] - positions[dst][0], positions[src][1] - positions[dst][1])
            if dist <= 220:
                co_pressure_edges.append((src, dst))

    image = Image.open(MAP_PATH).convert("RGB")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    # Compact two-column figure size for a short ACM-style paper.
    fig = plt.figure(figsize=(7.2, 2.75), dpi=300)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.04)
    map_ax = fig.add_subplot(grid[0, 0])
    graph_ax = fig.add_subplot(grid[0, 1])

    map_ax.imshow(image, extent=[0, arena, arena, 0], alpha=0.88)
    map_ax.set_xlim(4150, 4900)
    map_ax.set_ylim(5000, 4100)
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    for spine in map_ax.spines.values():
        spine.set_visible(False)
    map_ax.text(
        0.02,
        0.98,
        "(a)",
        transform=map_ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": (1, 1, 1, 0.88), "ec": "none"},
        zorder=10,
    )

    map_ax.add_patch(
        Circle(
            (zx, zy),
            safe_r,
            fill=False,
            edgecolor=COLOR_ZONE,
            linewidth=1.5,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )

    for src, dst, _ in proximity_edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        map_ax.plot([x0, x1], [y0, y1], color=COLOR_PROX, lw=1.0, alpha=0.55, zorder=2)

    for (src, dst), weight in combat_edges:
        draw_arrow(
            map_ax,
            positions[src],
            positions[dst],
            COLOR_COMBAT,
            0.8 + min(2.2, weight / 160),
            0.75,
            0.12,
            True,
            zorder=4,
        )

    for team_id, (x, y) in positions.items():
        label = f"T{nodes[team_id]['team']}"
        label_size = 3.5 if len(label) > 3 else 3.9
        map_ax.scatter([x], [y], s=52, c=NODE_FACE, edgecolors=NODE_EDGE, linewidths=0.65, zorder=6)
        map_ax.text(x, y, label, fontsize=label_size, ha="center", va="center", zorder=7)

    graph_ax.set_axis_off()
    graph_ax.set_xlim(0, 1)
    graph_ax.set_ylim(0, 1)
    graph_ax.text(
        0.02,
        0.98,
        "(b)",
        transform=graph_ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": (1, 1, 1, 0.88), "ec": "none"},
        zorder=10,
    )

    graph_pos = {
        0: (0.43, 0.70),
        1: (0.17, 0.39),
        2: (0.76, 0.48),
        3: (0.30, 0.28),
        4: (0.62, 0.26),
        5: (0.82, 0.76),
    }

    for src, dst, _ in proximity_edges:
        draw_arrow(graph_ax, graph_pos[src], graph_pos[dst], COLOR_PROX, 1.3, 0.52, 0.0, False, zorder=1)
    for src, dst in co_pressure_edges:
        draw_arrow(
            graph_ax,
            graph_pos[src],
            graph_pos[dst],
            COLOR_PRESSURE,
            1.5,
            0.68,
            -0.14,
            False,
            linestyle=(0, (3, 2)),
            zorder=2,
        )
    for (src, dst), weight in combat_edges:
        draw_arrow(
            graph_ax,
            graph_pos[src],
            graph_pos[dst],
            COLOR_COMBAT,
            1.0 + min(2.3, weight / 170),
            0.78,
            0.18,
            True,
            zorder=3,
        )

    for team_id, (gx, gy) in graph_pos.items():
        graph_ax.scatter([gx], [gy], s=260, c=NODE_FACE, edgecolors=NODE_EDGE, linewidths=0.9, zorder=5)
        graph_ax.text(gx, gy, f"T{nodes[team_id]['team']}", fontsize=6.4, ha="center", va="center", zorder=6)

    fig.add_artist(
        FancyArrowPatch(
            (0.487, 0.55),
            (0.527, 0.55),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#555555",
            alpha=0.75,
            zorder=20,
        )
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NODE_FACE,
            markeredgecolor=NODE_EDGE,
            markersize=5.0,
            label="team node",
        ),
        Line2D([0], [0], color=COLOR_PROX, lw=1.5, label="proximity"),
        Line2D([0], [0], color=COLOR_COMBAT, lw=1.8, label="combat (width=damage)"),
        Line2D([0], [0], color=COLOR_PRESSURE, lw=1.5, linestyle=(0, (3, 2)), label="co-pressure"),
        Line2D([0], [0], color=COLOR_ZONE, lw=1.2, linestyle=(0, (4, 3)), label="safe-zone"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=5.0,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        columnspacing=0.9,
        handlelength=1.7,
    )
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.04)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
