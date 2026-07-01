"""Build a sampled 3D Caldera spatial scene from the common BR DB layer.

The output is an interactive Plotly HTML prototype for paper-figure scouting:
trajectory line samples, anonymous endpoint samples, and environment
bbox/centroid candidates are rendered as separate traces.

Example:
    PG_PASSWORD=... python3 scripts/visualize_caldera_3d.py \
        --match-limit 2 --players-per-match 8 --endpoint-limit 3000

Paper snapshot:
    PG_PASSWORD=... python3 scripts/visualize_caldera_3d.py \
        --figure-mode paper-snapshot --camera-preset oblique
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stdb.conn import connect

SOURCE_ID = "activision_caldera"
ENVIRONMENT_ID = "caldera_map_source_v1"
DEFAULT_OUTPUT = Path("reports/caldera_3d/caldera_spatial_scene.html")
PAPER_OUTPUT = Path("reports/caldera_3d/caldera_spatial_snapshot.html")


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required to write the Caldera 3D HTML scene.\n"
            "Install it with one of:\n"
            "  python3 -m pip install plotly\n"
            "  .venv/bin/python -m pip install plotly\n"
            "Then rerun this script. Use --dry-run to validate DB sampling "
            "without Plotly."
        ) from exc
    return go


def fetch_trajectory_rows(conn, args: argparse.Namespace) -> list[dict[str, Any]]:
    if (
        args.match_limit == 0
        or args.players_per_match == 0
        or args.trajectory_point_limit == 0
    ):
        return []

    sql = """
        WITH selected_matches AS (
            SELECT br_match_id
            FROM br_ground.trajectory_samples
            WHERE br_match_id LIKE %s
              AND pos_x_m IS NOT NULL
              AND pos_y_m IS NOT NULL
              AND pos_z_m IS NOT NULL
            GROUP BY br_match_id
            ORDER BY br_match_id
            LIMIT %s
        ),
        selected_agents AS (
            SELECT br_match_id, br_agent_id
            FROM (
                SELECT
                    a.br_match_id,
                    a.br_agent_id,
                    row_number() OVER (
                        PARTITION BY a.br_match_id
                        ORDER BY a.br_agent_id
                    ) AS agent_rank
                FROM br_ground.agents a
                JOIN selected_matches m
                  ON m.br_match_id = a.br_match_id
                WHERE a.agent_kind = 'player'
            ) ranked_agents
            WHERE agent_rank <= %s
        ),
        ranked_samples AS (
            SELECT
                t.br_match_id,
                t.br_agent_id,
                t.sample_t,
                t.pos_x_m,
                t.pos_y_m,
                t.pos_z_m,
                t.alive_state,
                row_number() OVER (
                    PARTITION BY t.br_match_id, t.br_agent_id
                    ORDER BY t.sample_t
                ) AS sample_rank
            FROM br_ground.trajectory_samples t
            JOIN selected_agents a
              ON a.br_match_id = t.br_match_id
             AND a.br_agent_id = t.br_agent_id
            WHERE t.pos_x_m IS NOT NULL
              AND t.pos_y_m IS NOT NULL
              AND t.pos_z_m IS NOT NULL
        )
        SELECT
            br_match_id,
            br_agent_id,
            sample_t,
            pos_x_m,
            pos_y_m,
            pos_z_m,
            alive_state
        FROM ranked_samples
        WHERE (sample_rank - 1) %% %s = 0
        ORDER BY br_match_id, br_agent_id, sample_t
        LIMIT %s
    """
    params = (
        args.match_id_like,
        args.match_limit,
        args.players_per_match,
        args.trajectory_sample_stride,
        args.trajectory_point_limit,
    )
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def fetch_endpoint_rows(conn, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.endpoint_limit == 0:
        return []

    sql = """
        SELECT
            endpoint_idx,
            sample_t,
            pos_x_m,
            pos_y_m,
            pos_z_m,
            color_r,
            color_g,
            color_b
        FROM br_ground.endpoint_samples
        WHERE source_id = %s
          AND pos_x_m IS NOT NULL
          AND pos_y_m IS NOT NULL
          AND pos_z_m IS NOT NULL
        ORDER BY endpoint_idx
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (args.source_id, args.endpoint_limit))
        return list(cur.fetchall())


def fetch_environment_rows(conn, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.environment_limit == 0:
        return []

    sql = """
        SELECT
            prim_path,
            prim_name,
            COALESCE(prim_type, 'unknown') AS prim_type,
            bbox_min_x_m,
            bbox_min_y_m,
            bbox_min_z_m,
            bbox_max_x_m,
            bbox_max_y_m,
            bbox_max_z_m,
            COALESCE(ST_X(centroid), (bbox_min_x_m + bbox_max_x_m) / 2.0)
                AS centroid_x_m,
            COALESCE(ST_Y(centroid), (bbox_min_y_m + bbox_max_y_m) / 2.0)
                AS centroid_y_m,
            COALESCE(ST_Z(centroid), (bbox_min_z_m + bbox_max_z_m) / 2.0)
                AS centroid_z_m
        FROM br_ground.environment_prims
        WHERE source_id = %s
          AND environment_id = %s
          AND bbox_min_x_m IS NOT NULL
          AND bbox_min_y_m IS NOT NULL
          AND bbox_min_z_m IS NOT NULL
          AND bbox_max_x_m IS NOT NULL
          AND bbox_max_y_m IS NOT NULL
          AND bbox_max_z_m IS NOT NULL
        ORDER BY
            COALESCE(
                ABS((bbox_max_x_m - bbox_min_x_m)
                  * (bbox_max_y_m - bbox_min_y_m)),
                0.0
            ) DESC,
            COALESCE(ABS(bbox_max_z_m - bbox_min_z_m), 0.0) DESC,
            prim_path
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (args.source_id, args.environment_id, args.environment_limit))
        return list(cur.fetchall())


def fetch_scene(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    with connect() as conn:
        return {
            "trajectories": fetch_trajectory_rows(conn, args),
            "endpoints": fetch_endpoint_rows(conn, args),
            "environment": fetch_environment_rows(conn, args),
        }


def short_id(value: str) -> str:
    if value.startswith("caldera:"):
        return value[len("caldera:") :]
    return value


def endpoint_color(row: dict[str, Any]) -> str:
    values = [row.get("color_r"), row.get("color_g"), row.get("color_b")]
    if any(value is None for value in values):
        return "rgba(234, 179, 8, 0.55)"
    scale = 255.0 if max(values) <= 1.0 else 1.0
    rgb = [max(0, min(255, int(round(value * scale)))) for value in values]
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.50)"


def hover_line(row: dict[str, Any]) -> str:
    alive = row.get("alive_state") or "unknown"
    return (
        f"match={row['br_match_id']}<br>"
        f"agent={row['br_agent_id']}<br>"
        f"t={row['sample_t']:.2f}s<br>"
        f"x={row['pos_x_m']:.1f}m, y={row['pos_y_m']:.1f}m, "
        f"z={row['pos_z_m']:.1f}m<br>"
        f"state={alive}"
    )


def group_trajectories(
    rows: list[dict[str, Any]],
) -> OrderedDict[tuple[str, str], list[dict[str, Any]]]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = (row["br_match_id"], row["br_agent_id"])
        grouped.setdefault(key, []).append(row)
    return grouped


def resolve_snapshot_t(rows: list[dict[str, Any]], args: argparse.Namespace) -> float | None:
    if args.snapshot_t is not None:
        return args.snapshot_t
    times = sorted(row["sample_t"] for row in rows if row.get("sample_t") is not None)
    if not times:
        return None
    idx = min(len(times) - 1, max(0, int(round((len(times) - 1) * args.snapshot_percentile))))
    return float(times[idx])


def split_by_snapshot(
    rows: list[dict[str, Any]], snapshot_t: float | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if snapshot_t is None:
        return rows, [], rows[-1] if rows else None
    past = [row for row in rows if row["sample_t"] <= snapshot_t]
    future = [row for row in rows if row["sample_t"] > snapshot_t]
    if past:
        current = past[-1]
    elif rows:
        current = min(rows, key=lambda row: abs(row["sample_t"] - snapshot_t))
    else:
        current = None
    return past, future, current


def scene_bounds(scene: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for row in scene["trajectories"]:
        xs.append(float(row["pos_x_m"]))
        ys.append(float(row["pos_y_m"]))
        zs.append(float(row["pos_z_m"]))
    for row in scene["endpoints"]:
        xs.append(float(row["pos_x_m"]))
        ys.append(float(row["pos_y_m"]))
        zs.append(float(row["pos_z_m"]))
    for row in scene["environment"]:
        for key in ("bbox_min_x_m", "bbox_max_x_m"):
            xs.append(float(row[key]))
        for key in ("bbox_min_y_m", "bbox_max_y_m"):
            ys.append(float(row[key]))
        for key in ("bbox_min_z_m", "bbox_max_z_m"):
            zs.append(float(row[key]))

    if not xs:
        return {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}

    pad_x = max(100.0, (max(xs) - min(xs)) * 0.06)
    pad_y = max(100.0, (max(ys) - min(ys)) * 0.06)
    return {
        "xmin": min(xs) - pad_x,
        "xmax": max(xs) + pad_x,
        "ymin": min(ys) - pad_y,
        "ymax": max(ys) + pad_y,
        "zmin": min(zs),
        "zmax": max(zs),
    }


def camera_for_preset(name: str, projection: str) -> dict[str, Any]:
    cameras = {
        "overview": {"eye": {"x": 1.45, "y": -1.55, "z": 1.05}},
        "oblique": {"eye": {"x": 1.85, "y": -1.25, "z": 0.72}},
        "topdown": {"eye": {"x": 0.0, "y": 0.0, "z": 2.35}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}},
        "low": {"eye": {"x": 1.75, "y": -0.85, "z": 0.35}},
    }
    camera = cameras[name].copy()
    camera["projection"] = {"type": projection}
    return camera


def bbox_edges(row: dict[str, Any]) -> list[tuple[float | None, float | None, float | None]]:
    x0, y0, z0 = row["bbox_min_x_m"], row["bbox_min_y_m"], row["bbox_min_z_m"]
    x1, y1, z1 = row["bbox_max_x_m"], row["bbox_max_y_m"], row["bbox_max_z_m"]
    corners = {
        "000": (x0, y0, z0),
        "100": (x1, y0, z0),
        "110": (x1, y1, z0),
        "010": (x0, y1, z0),
        "001": (x0, y0, z1),
        "101": (x1, y0, z1),
        "111": (x1, y1, z1),
        "011": (x0, y1, z1),
    }
    pairs = [
        ("000", "100"),
        ("100", "110"),
        ("110", "010"),
        ("010", "000"),
        ("001", "101"),
        ("101", "111"),
        ("111", "011"),
        ("011", "001"),
        ("000", "001"),
        ("100", "101"),
        ("110", "111"),
        ("010", "011"),
    ]
    points: list[tuple[float | None, float | None, float | None]] = []
    for start, end in pairs:
        points.extend([corners[start], corners[end], (None, None, None)])
    return points


def build_explore_figure(scene: dict[str, list[dict[str, Any]]], args: argparse.Namespace):
    go = import_plotly()
    fig = go.Figure()

    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
    ]
    for idx, ((match_id, agent_id), rows) in enumerate(
        group_trajectories(scene["trajectories"]).items()
    ):
        line_name = f"{short_id(match_id)} / {short_id(agent_id).split(':')[-1]}"
        fig.add_trace(
            go.Scatter3d(
                name=f"Trajectory {line_name}",
                x=[row["pos_x_m"] for row in rows],
                y=[row["pos_y_m"] for row in rows],
                z=[row["pos_z_m"] for row in rows],
                mode="lines",
                line={"color": palette[idx % len(palette)], "width": 4},
                hovertext=[hover_line(row) for row in rows],
                hoverinfo="text",
                legendgroup="trajectories",
            )
        )

    endpoint_rows = scene["endpoints"]
    if endpoint_rows:
        fig.add_trace(
            go.Scatter3d(
                name="Endpoint samples",
                x=[row["pos_x_m"] for row in endpoint_rows],
                y=[row["pos_y_m"] for row in endpoint_rows],
                z=[row["pos_z_m"] for row in endpoint_rows],
                mode="markers",
                marker={
                    "size": 2,
                    "color": [endpoint_color(row) for row in endpoint_rows],
                    "opacity": 0.5,
                },
                hovertext=[
                    "endpoint_idx={endpoint_idx}<br>t={sample_t}<br>"
                    "x={pos_x_m:.1f}m, y={pos_y_m:.1f}m, z={pos_z_m:.1f}m".format(
                        **row
                    )
                    for row in endpoint_rows
                ],
                hoverinfo="text",
            )
        )

    environment_rows = scene["environment"]
    if environment_rows:
        bx: list[float | None] = []
        by: list[float | None] = []
        bz: list[float | None] = []
        for row in environment_rows:
            for x, y, z in bbox_edges(row):
                bx.append(x)
                by.append(y)
                bz.append(z)
        fig.add_trace(
            go.Scatter3d(
                name="Environment bbox candidates",
                x=bx,
                y=by,
                z=bz,
                mode="lines",
                line={"color": "rgba(31, 41, 55, 0.30)", "width": 2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                name="Environment centroids",
                x=[row["centroid_x_m"] for row in environment_rows],
                y=[row["centroid_y_m"] for row in environment_rows],
                z=[row["centroid_z_m"] for row in environment_rows],
                mode="markers",
                marker={"size": 4, "color": "#f97316", "opacity": 0.78},
                hovertext=[
                    f"{row['prim_type']}<br>{row['prim_path']}<br>"
                    f"x={row['centroid_x_m']:.1f}m, "
                    f"y={row['centroid_y_m']:.1f}m, "
                    f"z={row['centroid_z_m']:.1f}m"
                    for row in environment_rows
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=(
            "Caldera spatial scene: trajectories, endpoints, and environment catalog"
        ),
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z/elevation (m)",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
    )
    return fig


def add_paper_ground_plane(fig, go, bounds: dict[str, float]) -> None:
    z = min(0.0, bounds["zmin"])
    fig.add_trace(
        go.Mesh3d(
            name="Map ground plane",
            x=[bounds["xmin"], bounds["xmax"], bounds["xmax"], bounds["xmin"]],
            y=[bounds["ymin"], bounds["ymin"], bounds["ymax"], bounds["ymax"]],
            z=[z, z, z, z],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#d9ded7",
            opacity=0.28,
            hoverinfo="skip",
            showscale=False,
        )
    )


def build_paper_snapshot_figure(scene: dict[str, list[dict[str, Any]]], args: argparse.Namespace):
    go = import_plotly()
    fig = go.Figure()
    bounds = scene_bounds(scene)
    snapshot_t = resolve_snapshot_t(scene["trajectories"], args)

    add_paper_ground_plane(fig, go, bounds)

    environment_rows = scene["environment"]
    if environment_rows:
        bx: list[float | None] = []
        by: list[float | None] = []
        bz: list[float | None] = []
        for row in environment_rows:
            for x, y, z in bbox_edges(row):
                bx.append(x)
                by.append(y)
                bz.append(z)
        fig.add_trace(
            go.Scatter3d(
                name="Clay environment scaffold",
                x=bx,
                y=by,
                z=bz,
                mode="lines",
                line={"color": "rgba(146, 152, 160, 0.24)", "width": 2},
                hoverinfo="skip",
            )
        )

    if args.show_endpoints and scene["endpoints"]:
        fig.add_trace(
            go.Scatter3d(
                name="Endpoint field",
                x=[row["pos_x_m"] for row in scene["endpoints"]],
                y=[row["pos_y_m"] for row in scene["endpoints"]],
                z=[row["pos_z_m"] for row in scene["endpoints"]],
                mode="markers",
                marker={"size": 2, "color": "rgba(124, 58, 237, 0.18)", "opacity": 0.5},
                hoverinfo="skip",
            )
        )

    palette = ["#06b6d4", "#facc15", "#22c55e", "#f97316", "#a855f7", "#ef4444", "#38bdf8", "#84cc16"]
    current_rows: list[dict[str, Any]] = []
    current_colors: list[str] = []
    current_text: list[str] = []

    for idx, ((match_id, agent_id), rows) in enumerate(group_trajectories(scene["trajectories"]).items()):
        color = palette[idx % len(palette)]
        past, future, current = split_by_snapshot(rows, snapshot_t)
        line_name = f"{short_id(match_id)} / {short_id(agent_id).split(':')[-1]}"
        if past:
            fig.add_trace(
                go.Scatter3d(
                    name=f"Observed path {line_name}",
                    x=[row["pos_x_m"] for row in past],
                    y=[row["pos_y_m"] for row in past],
                    z=[row["pos_z_m"] for row in past],
                    mode="lines",
                    line={"color": color, "width": 5},
                    opacity=0.88,
                    hovertext=[hover_line(row) for row in past],
                    hoverinfo="text",
                    legendgroup="snapshot-paths",
                    showlegend=idx < 6,
                )
            )
        if args.show_future_path and future:
            fig.add_trace(
                go.Scatter3d(
                    name=f"Future path {line_name}",
                    x=[row["pos_x_m"] for row in future],
                    y=[row["pos_y_m"] for row in future],
                    z=[row["pos_z_m"] for row in future],
                    mode="lines",
                    line={"color": "rgba(90, 97, 110, 0.18)", "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if current is not None:
            current_rows.append(current)
            current_colors.append(color)
            snapshot_prefix = f"snapshot_t={snapshot_t:.2f}s<br>" if snapshot_t is not None else ""
            current_text.append(
                snapshot_prefix
                + f"match={match_id}<br>agent={agent_id}<br>"
                + f"current_t={current['sample_t']:.2f}s<br>"
                + f"x={current['pos_x_m']:.1f}m, y={current['pos_y_m']:.1f}m, z={current['pos_z_m']:.1f}m"
            )

    if current_rows:
        fig.add_trace(
            go.Scatter3d(
                name="Player positions at snapshot",
                x=[row["pos_x_m"] for row in current_rows],
                y=[row["pos_y_m"] for row in current_rows],
                z=[row["pos_z_m"] for row in current_rows],
                mode="markers",
                marker={
                    "size": 8,
                    "color": current_colors,
                    "opacity": 0.96,
                    "line": {"color": "white", "width": 2},
                    "symbol": "circle",
                },
                hovertext=current_text,
                hoverinfo="text",
            )
        )

    title = "Caldera movement snapshot"
    if snapshot_t is not None:
        title += f" at t={snapshot_t:.1f}s"

    axis_style = {
        "visible": not args.hide_axes,
        "showbackground": False,
        "showgrid": False,
        "zeroline": False,
        "title": "",
    }
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#eef1ee",
        plot_bgcolor="#eef1ee",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={
            "xaxis": axis_style,
            "yaxis": axis_style,
            "zaxis": axis_style,
            "aspectmode": "data",
            "camera": camera_for_preset(args.camera_preset, args.projection),
        },
        legend={"itemsizing": "constant", "orientation": "h", "y": 0.02, "x": 0.02},
    )
    return fig


def build_figure(scene: dict[str, list[dict[str, Any]]], args: argparse.Namespace):
    if args.figure_mode == "paper-snapshot":
        return build_paper_snapshot_figure(scene, args)
    return build_explore_figure(scene, args)


def print_scene_summary(scene: dict[str, list[dict[str, Any]]]) -> None:
    trajectory_agents = len(group_trajectories(scene["trajectories"]))
    trajectory_matches = len({row["br_match_id"] for row in scene["trajectories"]})
    print(f"trajectory_matches={trajectory_matches}")
    print(f"trajectory_agents={trajectory_agents}")
    print(f"trajectory_points={len(scene['trajectories'])}")
    print(f"endpoint_points={len(scene['endpoints'])}")
    print(f"environment_prims={len(scene['environment'])}")


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--source-id", default=SOURCE_ID)
    parser.add_argument("--environment-id", default=ENVIRONMENT_ID)
    parser.add_argument(
        "--match-id-like",
        default="caldera:%",
        help="SQL LIKE pattern for Caldera trajectory matches.",
    )
    parser.add_argument("--match-limit", type=nonnegative_int, default=2)
    parser.add_argument("--players-per-match", type=nonnegative_int, default=8)
    parser.add_argument("--endpoint-limit", type=nonnegative_int, default=3000)
    parser.add_argument("--environment-limit", type=nonnegative_int, default=800)
    parser.add_argument(
        "--trajectory-sample-stride",
        type=positive_int,
        default=12,
        help="Keep every Nth trajectory sample per selected player.",
    )
    parser.add_argument(
        "--trajectory-point-limit",
        type=nonnegative_int,
        default=30000,
        help="Hard cap after match/player/stride filtering.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=("inline", "cdn"),
        default="inline",
        help="Use inline for a self-contained HTML file, or cdn for smaller output.",
    )
    parser.add_argument(
        "--figure-mode",
        choices=("explore", "paper-snapshot"),
        default="explore",
        help="Use explore for all sampled data or paper-snapshot for a clay-map snapshot figure.",
    )
    parser.add_argument(
        "--snapshot-t",
        type=float,
        help="Game seconds for the paper snapshot. Defaults to --snapshot-percentile over sampled rows.",
    )
    parser.add_argument(
        "--snapshot-percentile",
        type=unit_interval,
        default=0.45,
        help="Sample-time percentile used when --snapshot-t is omitted.",
    )
    parser.add_argument(
        "--camera-preset",
        choices=("overview", "oblique", "topdown", "low"),
        default="oblique",
        help="Camera preset for paper-snapshot mode.",
    )
    parser.add_argument(
        "--projection",
        choices=("perspective", "orthographic"),
        default="perspective",
        help="3D camera projection.",
    )
    parser.add_argument(
        "--show-future-path",
        action="store_true",
        help="In paper-snapshot mode, include faint path segments after the snapshot time.",
    )
    parser.add_argument(
        "--hide-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide 3D axes in paper-snapshot mode.",
    )
    parser.add_argument(
        "--show-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show endpoint field in paper-snapshot mode.",
    )
    parser.add_argument(
        "--static-output",
        type=Path,
        help="Optional static image path, such as reports/caldera_3d/snapshot.png. Requires kaleido.",
    )
    parser.add_argument("--static-width", type=positive_int, default=1920)
    parser.add_argument("--static-height", type=positive_int, default=1080)
    parser.add_argument("--static-scale", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and report sample counts without importing Plotly or writing HTML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figure_mode == "paper-snapshot" and args.output == DEFAULT_OUTPUT:
        args.output = PAPER_OUTPUT
    try:
        scene = fetch_scene(args)
    except Exception as exc:
        raise SystemExit(f"Failed to read Caldera DB samples: {exc}") from None
    print_scene_summary(scene)
    if args.figure_mode == "paper-snapshot":
        snapshot_t = resolve_snapshot_t(scene["trajectories"], args)
        print(f"snapshot_t={snapshot_t}")
    if args.dry_run:
        print("dry_run=true html_written=false")
        return

    fig = build_figure(scene, args)
    output = resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    include_plotlyjs = True if args.include_plotlyjs == "inline" else "cdn"
    fig.write_html(str(output), include_plotlyjs=include_plotlyjs, full_html=True)
    print(f"html_written={output}")
    if args.static_output:
        static_output = resolve_output(args.static_output)
        static_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(
                str(static_output),
                width=args.static_width,
                height=args.static_height,
                scale=args.static_scale,
            )
        except Exception as exc:
            raise SystemExit(
                "Static image export failed. Plotly static export requires kaleido.\n"
                "Install it with: python3 -m pip install kaleido\n"
                f"Original error: {exc}"
            ) from None
        print(f"static_written={static_output}")


if __name__ == "__main__":
    main()
