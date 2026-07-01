"""Build an interactive ST-aware Caldera 3D scene.

This is the research-scene layer, not just a map image.  It combines:

- reconstructed Caldera semantic mesh triangles with real x/y/z geometry;
- observed player trajectories with x/y/z/t samples;
- nearest reconstructed terrain elevation diagnostics for trajectory points;
- a Plotly animation slider that shows cumulative paths and current positions.

The default trajectory source is the previously generated Plotly snapshot HTML,
so this script can run without reconnecting to PostgreSQL.  If a fresher DB
extract is needed later, the trajectory source can be replaced by JSON/DB ETL
without changing the 3D scene contract.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from array import array
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "reports" / "caldera_3d" / "semantic_basemap" / "caldera_semantic_triangles.pkl"
DEFAULT_TRAJECTORY_HTML = ROOT / "reports" / "caldera_3d" / "caldera_spatial_snapshot.html"
DEFAULT_OUTPUT = ROOT / "reports" / "caldera_3d" / "caldera_st_3d_scene.html"

CATEGORIES = [
    "terrain",
    "coast_beach",
    "water",
    "vegetation",
    "road",
    "runway",
    "building",
    "prop",
]

DISPLAY_BUDGETS = {
    "terrain": 80_000,
    "coast_beach": 14_000,
    "water": 8_000,
    "vegetation": 22_000,
    "road": 32_000,
    "runway": 18_000,
    "building": 70_000,
    "prop": 8_000,
}

COLORS = {
    "terrain": "#48683c",
    "coast_beach": "#c7a96b",
    "water": "#16758a",
    "vegetation": "#1c5f2f",
    "road": "#c7c9bd",
    "runway": "#5b6260",
    "building": "#252b2d",
    "prop": "#6c695c",
}

OPACITY = {
    "terrain": 0.90,
    "coast_beach": 0.62,
    "water": 0.62,
    "vegetation": 0.46,
    "road": 0.72,
    "runway": 0.82,
    "building": 0.62,
    "prop": 0.36,
}


def array_to_tris(flat: array):
    import numpy as np

    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))


def filter_tris_by_xy_area(tris, max_xy_area: float):
    import numpy as np

    if max_xy_area <= 0 or len(tris) == 0:
        return tris
    ax = tris[:, 1, 0] - tris[:, 0, 0]
    ay = tris[:, 1, 1] - tris[:, 0, 1]
    bx = tris[:, 2, 0] - tris[:, 0, 0]
    by = tris[:, 2, 1] - tris[:, 0, 1]
    xy_area = np.abs(ax * by - ay * bx) * 0.5
    return tris[xy_area <= max_xy_area]


def deterministic_sample(tris, max_count: int):
    import numpy as np

    if max_count <= 0 or len(tris) <= max_count:
        return tris
    idx = np.linspace(0, len(tris) - 1, max_count, dtype=np.int64)
    return tris[idx]


def compute_bounds(triangle_by_category: dict[str, Any], trim_percentile: float) -> dict[str, list[float]]:
    import numpy as np

    parts = []
    for category in CATEGORIES:
        tris = triangle_by_category.get(category)
        if tris is not None and len(tris):
            parts.append(tris.reshape((-1, 3)))
    if not parts:
        return {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 1.0]}
    points = np.concatenate(parts, axis=0)
    trim = max(0.0, min(10.0, trim_percentile))
    low_q = trim
    high_q = 100.0 - trim
    return {
        "x": [float(v) for v in np.percentile(points[:, 0], [low_q, high_q])],
        "y": [float(v) for v in np.percentile(points[:, 1], [low_q, high_q])],
        "z": [float(v) for v in np.percentile(points[:, 2], [low_q, high_q])],
    }


def filter_by_bounds(tris, bounds: dict[str, list[float]]):
    import numpy as np

    if len(tris) == 0:
        return tris
    centroid = tris.mean(axis=1)
    mask = (
        (centroid[:, 0] >= bounds["x"][0])
        & (centroid[:, 0] <= bounds["x"][1])
        & (centroid[:, 1] >= bounds["y"][0])
        & (centroid[:, 1] <= bounds["y"][1])
    )
    return tris[mask]


def load_scene_triangles(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    payload = pickle.load(args.cache.open("rb"))
    raw = {
        category: filter_tris_by_xy_area(array_to_tris(payload["triangles"][category]), args.max_xy_area)
        for category in CATEGORIES
    }
    bounds = compute_bounds(raw, args.bounds_trim_percentile)
    display = {}
    for category in CATEGORIES:
        bounded = filter_by_bounds(raw[category], bounds)
        display[category] = deterministic_sample(bounded, getattr(args, f"{category}_display_triangles"))
    return payload, display, bounds


def extract_plotly_data_array(html_path: Path) -> list[dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    new_plot_idx = text.rfind("Plotly.newPlot")
    if new_plot_idx < 0:
        raise RuntimeError(f"Could not find Plotly.newPlot in {html_path}")
    start = text.find("[", new_plot_idx)
    if start < 0:
        raise RuntimeError(f"Could not find Plotly data array in {html_path}")
    in_string = False
    escaped = False
    depth = 0
    end = None
    for idx, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
    if end is None:
        raise RuntimeError(f"Could not parse Plotly data array in {html_path}")
    return json.loads(text[start:end])


def time_from_hover(value: str) -> float | None:
    match = re.search(r"(?:^|<br>)t=([0-9.]+)s", value)
    if not match:
        match = re.search(r"current_t=([0-9.]+)s", value)
    return float(match.group(1)) if match else None


def parse_match_agent(name: str) -> tuple[str, str]:
    value = name.replace("Observed path ", "")
    if " / " not in value:
        return ("unknown_match", value)
    match_id, agent_id = value.split(" / ", 1)
    return (match_id.strip(), agent_id.strip())


def load_trajectories(html_path: Path) -> list[dict[str, Any]]:
    traces = extract_plotly_data_array(html_path)
    rows: list[dict[str, Any]] = []
    for trace in traces:
        name = trace.get("name", "")
        if not name.startswith("Observed path "):
            continue
        match_id, agent_id = parse_match_agent(name)
        xs = trace.get("x", [])
        ys = trace.get("y", [])
        zs = trace.get("z", [])
        hover = trace.get("hovertext", [])
        color = trace.get("line", {}).get("color", "#facc15")
        for idx, (x, y, z) in enumerate(zip(xs, ys, zs)):
            sample_t = time_from_hover(hover[idx]) if idx < len(hover) else None
            rows.append(
                {
                    "match_id": match_id,
                    "agent_id": agent_id,
                    "sample_t": float(sample_t if sample_t is not None else idx),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "color": color,
                    "source_name": name,
                    "hover": hover[idx] if idx < len(hover) else "",
                }
            )
    rows.sort(key=lambda row: (row["match_id"], row["agent_id"], row["sample_t"]))
    return rows


def group_trajectories(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["match_id"], row["agent_id"]), []).append(row)
    return grouped


def attach_terrain_diagnostics(rows: list[dict[str, Any]], terrain_tris) -> dict[str, float | None]:
    import numpy as np

    if not rows or len(terrain_tris) == 0:
        return {"min_height_above_terrain": None, "median_height_above_terrain": None, "max_height_above_terrain": None}

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {"min_height_above_terrain": None, "median_height_above_terrain": None, "max_height_above_terrain": None}

    centroids = terrain_tris.mean(axis=1)
    tree = cKDTree(centroids[:, :2])
    points = np.asarray([(row["x"], row["y"]) for row in rows], dtype=np.float32)
    _, nearest = tree.query(points, k=1)
    terrain_z = centroids[nearest, 2]
    dz = np.asarray([row["z"] for row in rows], dtype=np.float32) - terrain_z
    for row, z0, delta in zip(rows, terrain_z, dz):
        row["terrain_z"] = float(z0)
        row["height_above_terrain"] = float(delta)
    return {
        "min_height_above_terrain": float(np.min(dz)),
        "median_height_above_terrain": float(np.median(dz)),
        "max_height_above_terrain": float(np.max(dz)),
    }


def make_mesh_trace(category: str, tris, z_scale: float):
    import numpy as np
    import plotly.graph_objects as go

    if len(tris) == 0:
        return None
    verts = tris.reshape((-1, 3)).astype(float)
    n = len(tris)
    base = np.arange(n, dtype=np.int64) * 3
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2] * z_scale,
        i=base,
        j=base + 1,
        k=base + 2,
        name=category,
        color=COLORS[category],
        opacity=OPACITY[category],
        flatshading=True,
        lighting={"ambient": 0.68, "diffuse": 0.72, "roughness": 0.9, "specular": 0.05},
        hoverinfo="skip",
        showlegend=True,
    )


def row_hover(row: dict[str, Any]) -> str:
    terrain = row.get("terrain_z")
    dz = row.get("height_above_terrain")
    terrain_line = ""
    if terrain is not None and dz is not None:
        terrain_line = f"<br>terrain_z={terrain:.1f}m<br>agent_z-terrain_z={dz:.1f}m"
    return (
        f"match={row['match_id']}<br>"
        f"agent={row['agent_id']}<br>"
        f"t={row['sample_t']:.2f}s<br>"
        f"x={row['x']:.1f}m, y={row['y']:.1f}m, z={row['z']:.1f}m"
        f"{terrain_line}"
    )


def make_static_trajectory_traces(grouped: dict[tuple[str, str], list[dict[str, Any]]], z_scale: float):
    import plotly.graph_objects as go

    traces = []
    for (match_id, agent_id), rows in grouped.items():
        color = rows[0].get("color", "#facc15")
        traces.append(
            go.Scatter3d(
                x=[row["x"] for row in rows],
                y=[row["y"] for row in rows],
                z=[row["z"] * z_scale for row in rows],
                mode="lines",
                name=f"{match_id}/{agent_id}",
                legendgroup=f"{match_id}/{agent_id}",
                line={"color": color, "width": 5},
                opacity=0.38,
                hovertext=[row_hover(row) for row in rows],
                hoverinfo="text",
                showlegend=False,
            )
        )
    return traces


def make_empty_dynamic_traces(grouped: dict[tuple[str, str], list[dict[str, Any]]]):
    import plotly.graph_objects as go

    traces = []
    for (match_id, agent_id), rows in grouped.items():
        color = rows[0].get("color", "#facc15")
        traces.append(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                name=f"active {match_id}/{agent_id}",
                legendgroup=f"{match_id}/{agent_id}",
                line={"color": color, "width": 8},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    traces.append(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            name="current agent positions",
            marker={"size": 8, "color": [], "line": {"color": "white", "width": 2}},
            hovertext=[],
            hoverinfo="text",
            showlegend=True,
        )
    )
    return traces


def frame_times(rows: list[dict[str, Any]], frame_count: int) -> list[float]:
    import numpy as np

    if not rows:
        return []
    times = np.asarray(sorted({row["sample_t"] for row in rows}), dtype=float)
    if len(times) <= frame_count:
        return [float(v) for v in times]
    quantiles = np.linspace(0.0, 1.0, frame_count)
    return [float(np.quantile(times, q)) for q in quantiles]


def make_frames(grouped: dict[tuple[str, str], list[dict[str, Any]]], times: list[float], trace_start: int, z_scale: float):
    import plotly.graph_objects as go

    frames = []
    group_items = list(grouped.items())
    marker_trace_idx = trace_start + len(group_items)
    frame_trace_indices = list(range(trace_start, marker_trace_idx + 1))

    for frame_t in times:
        frame_data = []
        marker_x = []
        marker_y = []
        marker_z = []
        marker_color = []
        marker_hover = []
        for (_key, rows) in group_items:
            past = [row for row in rows if row["sample_t"] <= frame_t]
            if past:
                frame_data.append(
                    go.Scatter3d(
                        x=[row["x"] for row in past],
                        y=[row["y"] for row in past],
                        z=[row["z"] * z_scale for row in past],
                        hovertext=[row_hover(row) for row in past],
                        hoverinfo="text",
                    )
                )
                current = past[-1]
                marker_x.append(current["x"])
                marker_y.append(current["y"])
                marker_z.append(current["z"] * z_scale)
                marker_color.append(current.get("color", "#facc15"))
                marker_hover.append(row_hover(current))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[], hovertext=[]))
        frame_data.append(
            go.Scatter3d(
                x=marker_x,
                y=marker_y,
                z=marker_z,
                marker={"size": 9, "color": marker_color, "line": {"color": "white", "width": 2}},
                hovertext=marker_hover,
                hoverinfo="text",
            )
        )
        frames.append(go.Frame(name=f"{frame_t:.2f}", data=frame_data, traces=frame_trace_indices))
    return frames


def build_figure(args: argparse.Namespace, display_tris: dict[str, Any], bounds: dict[str, list[float]], rows: list[dict[str, Any]]):
    import plotly.graph_objects as go

    fig = go.Figure()
    mesh_trace_count = 0
    for category in CATEGORIES:
        trace = make_mesh_trace(category, display_tris[category], args.z_scale)
        if trace is not None:
            fig.add_trace(trace)
            mesh_trace_count += 1

    grouped = group_trajectories(rows)
    for trace in make_static_trajectory_traces(grouped, args.z_scale):
        fig.add_trace(trace)

    dynamic_trace_start = len(fig.data)
    for trace in make_empty_dynamic_traces(grouped):
        fig.add_trace(trace)

    times = frame_times(rows, args.frames)
    fig.frames = make_frames(grouped, times, dynamic_trace_start, args.z_scale)

    sliders = []
    if times:
        sliders = [
            {
                "active": 0,
                "currentvalue": {"prefix": "t = ", "suffix": " s"},
                "pad": {"t": 28},
                "steps": [
                    {
                        "label": f"{frame_t:.0f}",
                        "method": "animate",
                        "args": [
                            [f"{frame_t:.2f}"],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                        ],
                    }
                    for frame_t in times
                ],
            }
        ]

    fig.update_layout(
        title=(
            "Caldera ST-aware 3D movement scene: reconstructed terrain/geometry "
            "+ physical x/y/z/t trajectories"
        ),
        scene={
            "xaxis": {"title": "x (m)", "showbackground": False, "gridcolor": "#314042"},
            "yaxis": {"title": "y (m)", "showbackground": False, "gridcolor": "#314042"},
            "zaxis": {"title": f"z (m) x {args.z_scale:g}", "showbackground": False, "gridcolor": "#314042"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": args.z_aspect},
            "camera": {"eye": {"x": 1.55, "y": -1.72, "z": 1.05}, "projection": {"type": "perspective"}},
        },
        paper_bgcolor="#101819",
        plot_bgcolor="#101819",
        font={"color": "#e8eee7"},
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        legend={"orientation": "h", "y": 0.01, "x": 0.01, "bgcolor": "rgba(16,24,25,0.65)"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.10,
                "buttons": [
                    {
                        "label": "Play ST trajectory",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": args.frame_duration_ms, "redraw": True}, "fromcurrent": True}],
                    }
                ],
            }
        ],
        sliders=sliders,
    )
    return fig, {"mesh_traces": mesh_trace_count, "trajectory_agents": len(grouped), "frame_count": len(times)}


def write_manifest(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def add_category_args(parser: argparse.ArgumentParser) -> None:
    for category, default in DISPLAY_BUDGETS.items():
        parser.add_argument(f"--{category.replace('_', '-')}-display-triangles", type=int, default=default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--trajectory-html", type=Path, default=DEFAULT_TRAJECTORY_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-xy-area", type=float, default=30_000.0)
    parser.add_argument("--bounds-trim-percentile", type=float, default=0.05)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--z-aspect", type=float, default=0.22)
    parser.add_argument("--frames", type=int, default=28)
    parser.add_argument("--frame-duration-ms", type=int, default=360)
    parser.add_argument("--include-plotlyjs", default="inline", choices=["inline", "cdn", "directory"])
    add_category_args(parser)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.trajectory_html = args.trajectory_html.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    payload, display_tris, bounds = load_scene_triangles(args)
    rows = load_trajectories(args.trajectory_html)
    dz_stats = attach_terrain_diagnostics(rows, display_tris["terrain"])
    fig, summary = build_figure(args, display_tris, bounds, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    include_plotlyjs: bool | str
    include_plotlyjs = True if args.include_plotlyjs == "inline" else args.include_plotlyjs
    fig.write_html(str(args.output), include_plotlyjs=include_plotlyjs, full_html=True)

    manifest = {
        "description": "Interactive Caldera ST-aware 3D scene from semantic mesh geometry and observed trajectories.",
        "output": str(args.output.relative_to(ROOT)),
        "semantic_cache": str(args.cache.relative_to(ROOT)),
        "trajectory_source": str(args.trajectory_html.relative_to(ROOT)),
        "source_root_stage": payload.get("root_stage"),
        "render_bounds": bounds,
        "display_triangles": {category: int(len(display_tris[category])) for category in CATEGORIES},
        "source_selected_triangles": payload.get("selected_triangles"),
        "trajectory_points": len(rows),
        "trajectory_matches": sorted({row["match_id"] for row in rows}),
        "trajectory_agents": len(group_trajectories(rows)),
        "time_range_s": [
            min((row["sample_t"] for row in rows), default=None),
            max((row["sample_t"] for row in rows), default=None),
        ],
        "height_above_terrain_stats_m": dz_stats,
        "figure_summary": summary,
        "notes": [
            "Mesh z and trajectory z are both rendered in meters with optional z-scale.",
            "height_above_terrain is approximate: nearest displayed terrain triangle centroid.",
            "This is a research scene layer, not a game-engine collision/navmesh reconstruction.",
        ],
    }
    write_manifest(args, manifest)
    print(f"html_written={args.output}")
    print(f"manifest_written={args.output.with_suffix('.manifest.json')}")
    print(json.dumps({k: manifest[k] for k in ("display_triangles", "trajectory_points", "trajectory_agents", "time_range_s", "height_above_terrain_stats_m")}, indent=2))


if __name__ == "__main__":
    main()
