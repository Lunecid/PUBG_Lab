"""Generate a paper figure showing PUBG telemetry as team-phase features."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MATCH_PATH = ROOT / "webapp/public/data/matches/3faf93bf-bd7b-4ff3-9e13-2083267f3556.json"
MAP_PATH = ROOT / "webapp/public/maps/baltic_main.jpg"
OUT_PATH = ROOT / "paper/figures/team_phase_feature_example.png"
NODE_ID = 4


def recover_phases(frames: list[dict]) -> tuple[list[int], list[tuple[int, int, dict]]]:
    """Recover coarse PUBG phases from target safe-zone changes."""
    phase_by_frame: list[int] = []
    phase_starts: list[tuple[int, int, dict]] = [(0, 0, frames[0])] if frames else []
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


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(MATCH_PATH.read_text())
    frames = data["frames"]
    node = data["nodes"][NODE_ID]
    arena = float(data.get("arena_m", 8160))
    phase_by_frame, phase_starts = recover_phases(frames)

    points = []
    key = str(NODE_ID)
    for frame_index, frame in enumerate(frames):
        if key in frame["cx"] and key in frame["cy"]:
            points.append(
                {
                    "frame_index": frame_index,
                    "phase": phase_by_frame[frame_index],
                    "t": float(frame["t"]),
                    "x": float(frame["cx"][key]),
                    "y": float(frame["cy"][key]),
                    "zone": frame.get("zone", {}),
                }
            )
    if len(points) < 2:
        raise RuntimeError("not enough trajectory points")

    coords = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    phases = np.array([p["phase"] for p in points], dtype=int)

    selected_phases = (0, 1, 2, 5)
    selected_phase_set = set(selected_phases)
    selected_indices = [idx for idx, phase in enumerate(phases) if int(phase) in selected_phase_set]
    landing_t = float(data.get("landing_offset_s", 90))
    landing_idx = next((idx for idx, point in enumerate(points) if point["t"] >= landing_t), 0)
    image = Image.open(MAP_PATH).convert("RGB")
    width, height = image.size

    def to_px(x_m: float, y_m: float) -> tuple[float, float]:
        return x_m / arena * width, y_m / arena * height

    phase_colors = {
        0: "#9aa0a6",
        1: "#2c6fb7",
        2: "#d98b2b",
        3: "#1f9d8a",
        4: "#8a5bb5",
        5: "#4f9d3a",
        6: "#4f9d3a",
    }
    default_phase_color = "#5b6c7a"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 7.0), dpi=300)
    ax.imshow(image)
    ax.set_xlim(1950, 4050)
    ax.set_ylim(3750, 1200)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # A light darkening layer keeps labels legible while preserving the map.
    ax.imshow(np.zeros((height, width, 3)), extent=[0, width, height, 0], alpha=0.13)

    for phase, _, frame in phase_starts:
        if phase not in selected_phase_set:
            continue
        zone = frame["zone"]
        px, py, pr = [float(zone.get(name, 0) or 0) for name in ("px", "py", "pr")]
        if px and py and pr:
            cx, cy = to_px(px, py)
            radius = pr / arena * width
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius,
                    fill=False,
                    edgecolor=phase_colors.get(phase, default_phase_color),
                    linewidth=1.25,
                    linestyle=(0, (4, 3)),
                    alpha=0.78,
                    zorder=2,
                )
            )

    xy = np.array([to_px(x, y) for x, y in coords])
    selected_segments = []
    selected_segment_colors = []
    for idx in range(len(xy) - 1):
        phase = int(phases[idx])
        if phase in selected_phase_set and int(phases[idx + 1]) == phase:
            selected_segments.append([xy[idx], xy[idx + 1]])
            selected_segment_colors.append(phase_colors.get(phase, default_phase_color))
    line_collection = LineCollection(selected_segments, colors=selected_segment_colors)
    line_collection.set_linewidth(5.1)
    line_collection.set_alpha(0.97)
    line_collection.set_zorder(6)
    ax.add_collection(line_collection)

    ax.scatter(
        xy[selected_indices, 0],
        xy[selected_indices, 1],
        s=15,
        c=[phase_colors.get(int(phases[idx]), default_phase_color) for idx in selected_indices],
        edgecolors="white",
        linewidths=0.4,
        zorder=7,
    )
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=70, c="#9aa0a6", edgecolors="black", linewidths=1.0, zorder=9)
    ax.scatter(
        [xy[landing_idx, 0]],
        [xy[landing_idx, 1]],
        s=120,
        marker="D",
        c="#ffd23f",
        edgecolors="black",
        linewidths=1.1,
        zorder=9,
    )
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=112, c="#e74c3c", edgecolors="black", linewidths=1.1, zorder=9)

    def label_point(text: str, anchor: np.ndarray, dx: float, dy: float, color: str = "white") -> None:
        ax.annotate(
            text,
            xy=anchor,
            xytext=(anchor[0] + dx, anchor[1] + dy),
            fontsize=8.6,
            color=color,
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.95},
            bbox={"boxstyle": "round,pad=0.18", "fc": (0, 0, 0, 0.58), "ec": "none"},
            zorder=10,
        )

    phase_offsets = {
        0: (-260, -120),
        1: (170, -165),
        2: (-215, -120),
        5: (150, -120),
    }
    for phase, (dx, dy) in phase_offsets.items():
        phase_indices = np.where(phases == phase)[0]
        if len(phase_indices) == 0:
            continue
        anchor = xy[int(phase_indices[len(phase_indices) // 2])]
        label_point(f"Phase {phase}", anchor, dx, dy, phase_colors.get(phase, "white"))

    label_point("landing", xy[landing_idx], -180, 95, "#ffd23f")
    label_point("eliminated", xy[-1], -160, -90)

    handles = [
        Line2D([0], [0], color=phase_colors[phase], lw=3, label=f"Phase {phase}")
        for phase in selected_phases
    ] + [
        Line2D([0], [0], color="#222222", lw=1.3, linestyle=(0, (4, 3)), label="safe-zone boundary"),
        Line2D([0], [0], color="#ffd23f", marker="D", lw=0, markersize=6, label="landing point"),
        Line2D([0], [0], color="#e74c3c", marker="o", lw=0, markersize=6, label="eliminated"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.5,
        ncol=1,
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="#444444",
        borderpad=0.55,
        handlelength=2.0,
    )
    legend.set_zorder(12)

    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.03)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
