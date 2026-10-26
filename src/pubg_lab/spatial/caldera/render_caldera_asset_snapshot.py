"""Render a Caldera asset-mesh snapshot with player trajectories.

This is the paper-figure path for Caldera: use actual OpenUSD mesh vertices
from the downloaded assets, simplify them into a clay-style static view, then
overlay Caldera breadcrumb agents from the DB at a snapshot time.

The script is intentionally split-runtime:

- extraction uses `.venv/bin/python`, because that environment has `pxr`;
- rendering uses system `python3`, because it has matplotlib and psycopg2.

Example:

    PG_PASSWORD=... python3 scripts/render_caldera_asset_snapshot.py

If the mesh cache is missing, the script invokes itself with the project venv
to create it first.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = REPO_ROOT
DEFAULT_USD = (
    ROOT
    / "data"
    / "br_l0"
    / "caldera"
    / "repo"
    / "map_source"
    / "prefabs"
    / "br"
    / "wz_vg"
    / "mp_wz_island"
    / "season_4"
    / "map_capital.usd"
)
DEFAULT_CACHE = ROOT / "reports" / "caldera_3d" / "map_capital_mesh_cache.pkl"
DEFAULT_OUTPUT = ROOT / "reports" / "caldera_3d" / "map_capital_agent_snapshot.png"


def parse_range(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    return (min(parts), max(parts))


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


def rgb_hex(value: str) -> tuple[int, int, int]:
    if not value.startswith("#") or len(value) != 7:
        raise argparse.ArgumentTypeError("color must be #RRGGBB")
    try:
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be #RRGGBB") from exc


def extract_mesh_cache(args: argparse.Namespace) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {args.usd_path}")

    scale = float(stage.GetMetadata("metersPerUnit") or 0.0254)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    triangles: list[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    ] = []
    mesh_count = 0
    skipped = 0

    clip_x = parse_range(args.clip_x)
    clip_y = parse_range(args.clip_y)
    clip_z = parse_range(args.clip_z)

    def in_clip(points: list[tuple[float, float, float]]) -> bool:
        if not points:
            return False
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        if clip_x and (max(xs) < clip_x[0] or min(xs) > clip_x[1]):
            return False
        if clip_y and (max(ys) < clip_y[0] or min(ys) > clip_y[1]):
            return False
        if clip_z and (max(zs) < clip_z[0] or min(zs) > clip_z[1]):
            return False
        return True

    def triangle_in_clip(
        tri: tuple[
            tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
        ],
    ) -> bool:
        if not (clip_x or clip_y or clip_z):
            return True
        centroid = (
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
            (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
        )
        if clip_x and not (clip_x[0] <= centroid[0] <= clip_x[1]):
            return False
        if clip_y and not (clip_y[0] <= centroid[1] <= clip_y[1]):
            return False
        if clip_z and not (clip_z[0] <= centroid[2] <= clip_z[1]):
            return False
        return True

    for prim in stage.Traverse():
        if args.max_meshes and mesh_count >= args.max_meshes:
            break
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = prim.GetPath().pathString
        include_filters = args.include_path_contains or []
        exclude_filters = args.exclude_path_contains or []
        if include_filters and not any(token in prim_path for token in include_filters):
            continue
        if exclude_filters and any(token in prim_path for token in exclude_filters):
            continue
        mesh_count += 1
        if args.mesh_stride > 1 and mesh_count % args.mesh_stride != 0:
            continue

        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not local_points or not counts or not indices:
            skipped += 1
            continue

        matrix = xform_cache.GetLocalToWorldTransform(prim)
        world_points = []
        for point in local_points:
            wp = matrix.Transform(point)
            world_points.append((float(wp[0]) * scale, float(wp[1]) * scale, float(wp[2]) * scale))
        if not in_clip(world_points):
            skipped += 1
            continue

        cursor = 0
        mesh_added = 0
        for face_idx, count in enumerate(counts):
            face_indices = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            if args.face_stride > 1 and face_idx % args.face_stride != 0:
                continue
            if args.max_faces_per_mesh and mesh_added >= args.max_faces_per_mesh:
                continue
            for i in range(1, count - 1):
                try:
                    tri = (
                        world_points[face_indices[0]],
                        world_points[face_indices[i]],
                        world_points[face_indices[i + 1]],
                    )
                except IndexError:
                    continue
                if not triangle_in_clip(tri):
                    continue
                triangles.append(tri)
                mesh_added += 1
                if args.max_faces and len(triangles) >= args.max_faces:
                    break
            if args.max_faces and len(triangles) >= args.max_faces:
                break
        if args.max_faces and len(triangles) >= args.max_faces:
            break

    if not triangles:
        raise RuntimeError("No mesh triangles extracted; relax clip or sampling options.")

    xs = [coord for tri in triangles for coord in (tri[0][0], tri[1][0], tri[2][0])]
    ys = [coord for tri in triangles for coord in (tri[0][1], tri[1][1], tri[2][1])]
    zs = [coord for tri in triangles for coord in (tri[0][2], tri[1][2], tri[2][2])]
    payload = {
        "usd_path": str(args.usd_path),
        "stage_meters_per_unit": scale,
        "triangles": triangles,
        "bounds": {
            "x": clip_x if clip_x else (min(xs), max(xs)),
            "y": clip_y if clip_y else (min(ys), max(ys)),
            "z": clip_z if clip_z else (min(zs), max(zs)),
        },
        "mesh_count_seen": mesh_count,
        "skipped_meshes": skipped,
        "triangle_count": len(triangles),
        "sampling": {
            "max_meshes": args.max_meshes,
            "mesh_stride": args.mesh_stride,
            "face_stride": args.face_stride,
            "max_faces": args.max_faces,
            "max_faces_per_mesh": args.max_faces_per_mesh,
            "clip_x": args.clip_x,
            "clip_y": args.clip_y,
            "clip_z": args.clip_z,
            "include_path_contains": args.include_path_contains,
            "exclude_path_contains": args.exclude_path_contains,
        },
    }
    args.mesh_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.mesh_cache.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"mesh_cache_written={args.mesh_cache}")
    print(f"triangles={len(triangles)} meshes_seen={mesh_count} skipped={skipped}")
    print(f"bounds={payload['bounds']}")


def ensure_mesh_cache(args: argparse.Namespace) -> None:
    if args.mesh_cache.exists() and not args.force_extract:
        return
    venv_python = ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(f"Missing OpenUSD venv python: {venv_python}")
    cmd = [
        str(venv_python),
        str(Path(__file__).resolve()),
        "--extract-only",
        "--usd-path",
        str(args.usd_path),
        "--mesh-cache",
        str(args.mesh_cache),
        "--max-meshes",
        str(args.max_meshes),
        "--mesh-stride",
        str(args.mesh_stride),
        "--face-stride",
        str(args.face_stride),
        "--max-faces",
        str(args.max_faces),
        "--max-faces-per-mesh",
        str(args.max_faces_per_mesh),
    ]
    for flag_name in ("clip_x", "clip_y", "clip_z"):
        value = getattr(args, flag_name)
        if value:
            cmd.append(f"--{flag_name.replace('_', '-')}={value}")
    for value in args.include_path_contains or []:
        cmd.extend(["--include-path-contains", value])
    for value in args.exclude_path_contains or []:
        cmd.extend(["--exclude-path-contains", value])
    subprocess.run(cmd, check=True)


def load_mesh_cache(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def connect_db():
    import psycopg2

    password = os.environ.get("PG_PASSWORD")
    if not password:
        raise SystemExit("PG_PASSWORD is required to read Caldera trajectory rows.")
    return psycopg2.connect(
        host="localhost", dbname="pubg_survival", user="postgres", password=password
    )


def fetch_agent_rows(
    args: argparse.Namespace, bounds: dict[str, tuple[float, float]]
) -> list[dict[str, Any]]:
    from psycopg2.extras import RealDictCursor

    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    pad = args.agent_padding_m
    x0 -= pad
    x1 += pad
    y0 -= pad
    y1 += pad

    extra_filters = []
    params: list[Any] = [x0, x1, y0, y1]
    if args.agent_z_min is not None:
        extra_filters.append("pos_z_m >= %s")
        params.append(args.agent_z_min)
    if args.agent_z_max is not None:
        extra_filters.append("pos_z_m <= %s")
        params.append(args.agent_z_max)
    if args.sample_t_min is not None:
        extra_filters.append("sample_t >= %s")
        params.append(args.sample_t_min)
    if args.sample_t_max is not None:
        extra_filters.append("sample_t <= %s")
        params.append(args.sample_t_max)
    extra_sql = ""
    if extra_filters:
        extra_sql = " AND " + " AND ".join(extra_filters)

    with connect_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if args.match_id or args.agent_id:
                identity_filters = []
                identity_params: list[Any] = []
                if args.match_id:
                    identity_filters.append("t.br_match_id = %s")
                    identity_params.append(args.match_id)
                if args.agent_id:
                    identity_filters.append("t.br_agent_id = %s")
                    identity_params.append(args.agent_id)
                identity_sql = " AND " + " AND ".join(identity_filters)
                cur.execute(
                    f"""
                    WITH ranked AS (
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
                            ) AS rn
                        FROM br_ground.trajectory_samples t
                        WHERE t.source_artifact_id = 'caldera_minimal_layers_breadcrumbs_usd'
                          AND t.pos_x_m BETWEEN %s AND %s
                          AND t.pos_y_m BETWEEN %s AND %s
                          {extra_sql.replace("pos_z_m", "t.pos_z_m").replace("sample_t", "t.sample_t")}
                          {identity_sql}
                    )
                    SELECT br_match_id, br_agent_id, sample_t, pos_x_m, pos_y_m, pos_z_m, alive_state
                    FROM ranked
                    WHERE (rn - 1) %% %s = 0
                    ORDER BY br_match_id, br_agent_id, sample_t
                    """,
                    tuple(params + identity_params + [args.trajectory_stride]),
                )
                return list(cur.fetchall())

            cur.execute(
                f"""
                WITH region_samples AS (
                    SELECT br_match_id, br_agent_id, count(*) AS n
                    FROM br_ground.trajectory_samples
                    WHERE source_artifact_id = 'caldera_minimal_layers_breadcrumbs_usd'
                      AND pos_x_m BETWEEN %s AND %s
                      AND pos_y_m BETWEEN %s AND %s
                      {extra_sql}
                    GROUP BY br_match_id, br_agent_id
                ),
                selected_match AS (
                    SELECT br_match_id
                    FROM region_samples
                    GROUP BY br_match_id
                    ORDER BY count(*) DESC, sum(n) DESC, br_match_id
                    LIMIT 1
                ),
                selected_agents AS (
                    SELECT r.br_match_id, r.br_agent_id
                    FROM region_samples r
                    JOIN selected_match m ON m.br_match_id = r.br_match_id
                    ORDER BY r.n DESC, r.br_agent_id
                    LIMIT %s
                ),
                ranked AS (
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
                        ) AS rn
                    FROM br_ground.trajectory_samples t
                    JOIN selected_agents a
                      ON a.br_match_id = t.br_match_id
                     AND a.br_agent_id = t.br_agent_id
                    WHERE t.pos_x_m BETWEEN %s AND %s
                      AND t.pos_y_m BETWEEN %s AND %s
                      {extra_sql.replace("pos_z_m", "t.pos_z_m").replace("sample_t", "t.sample_t")}
                )
                SELECT br_match_id, br_agent_id, sample_t, pos_x_m, pos_y_m, pos_z_m, alive_state
                FROM ranked
                WHERE (rn - 1) %% %s = 0
                ORDER BY br_match_id, br_agent_id, sample_t
                """,
                tuple(params + [args.players] + params + [args.trajectory_stride]),
            )
            return list(cur.fetchall())


def snapshot_time(
    rows: list[dict[str, Any]], percentile: float, explicit: float | None
) -> float | None:
    if explicit is not None:
        return explicit
    times = sorted(float(row["sample_t"]) for row in rows)
    if not times:
        return None
    idx = max(0, min(len(times) - 1, int(round((len(times) - 1) * percentile))))
    return times[idx]


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["br_agent_id"]), []).append(row)
    return grouped


def _vec_sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(a: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_vec_dot(a, a))
    if length <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _camera_basis(
    elev: float, azim: float
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    elev_r = math.radians(elev)
    azim_r = math.radians(azim)
    view = _vec_norm(
        (math.cos(elev_r) * math.cos(azim_r), math.cos(elev_r) * math.sin(azim_r), math.sin(elev_r))
    )
    world_up = (0.0, 0.0, 1.0)
    right = _vec_norm(_vec_cross(view, world_up))
    if _vec_dot(right, right) <= 1e-9:
        right = (1.0, 0.0, 0.0)
    up = _vec_norm(_vec_cross(right, view))
    return view, right, up


def _project_point(
    p: tuple[float, float, float],
    center: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    view: tuple[float, float, float],
) -> tuple[float, float, float]:
    rel = _vec_sub(p, center)
    return (_vec_dot(rel, right), _vec_dot(rel, up), _vec_dot(rel, view))


def render_snapshot_raster(
    args: argparse.Namespace,
    cache: dict[str, Any],
    rows: list[dict[str, Any]],
    snap_t: float | None,
) -> None:
    from PIL import Image, ImageDraw

    triangles = cache["triangles"]
    bounds = cache["bounds"]
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5)
    view, right, up = _camera_basis(args.elev, args.azim)

    projected_bounds: list[tuple[float, float]] = []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                u, v, _ = _project_point((x, y, z), center, right, up, view)
                projected_bounds.append((u, v))
    min_u = min(p[0] for p in projected_bounds)
    max_u = max(p[0] for p in projected_bounds)
    min_v = min(p[1] for p in projected_bounds)
    max_v = max(p[1] for p in projected_bounds)
    span_u = max(1e-6, max_u - min_u)
    span_v = max(1e-6, max_v - min_v)
    margin = args.raster_margin_px
    scale = (
        min((args.width - 2 * margin) / span_u, (args.height - 2 * margin) / span_v)
        * args.raster_zoom
    )
    cx = args.width * 0.5
    cy = args.height * 0.52

    def to_screen(p: tuple[float, float, float]) -> tuple[float, float, float]:
        u, v, depth = _project_point(p, center, right, up, view)
        return (cx + u * scale, cy - v * scale, depth)

    bg = tuple(int(args.background_color[i : i + 2], 16) for i in (1, 3, 5))
    image = Image.new("RGB", (args.width, args.height), bg)
    draw = ImageDraw.Draw(image, "RGBA")

    light = _vec_norm((-0.45, -0.35, 0.82))
    prepared = []
    for tri in triangles:
        p0, p1, p2 = tri
        normal = _vec_norm(_vec_cross(_vec_sub(p1, p0), _vec_sub(p2, p0)))
        area_hint = math.sqrt(
            _vec_dot(
                _vec_cross(_vec_sub(p1, p0), _vec_sub(p2, p0)),
                _vec_cross(_vec_sub(p1, p0), _vec_sub(p2, p0)),
            )
        )
        if area_hint <= args.min_triangle_area:
            continue
        if args.max_triangle_area and area_hint >= args.max_triangle_area:
            continue
        shade = 0.44 + 0.42 * abs(_vec_dot(normal, light)) + 0.14 * max(0.0, normal[2])
        shade = max(0.34, min(0.94, shade))
        base = args.mesh_tint
        rgb = tuple(max(0, min(255, int(channel * shade))) for channel in base)
        pts = [to_screen(p0), to_screen(p1), to_screen(p2)]
        if all(
            (pt[0] < -100 or pt[0] > args.width + 100 or pt[1] < -100 or pt[1] > args.height + 100)
            for pt in pts
        ):
            continue
        prepared.append((sum(pt[2] for pt in pts) / 3.0, rgb, pts))

    prepared.sort(key=lambda item: item[0])
    for _, rgb, pts in prepared:
        xy = [(pt[0], pt[1]) for pt in pts]
        draw.polygon(xy, fill=(rgb[0], rgb[1], rgb[2], int(255 * args.mesh_alpha)))
        if args.raster_edge_alpha > 0:
            draw.line(
                xy + [xy[0]],
                fill=(255, 255, 255, int(255 * args.raster_edge_alpha)),
                width=max(1, int(args.raster_edge_width)),
            )

    if args.mesh_point_alpha > 0 and args.mesh_point_stride > 0:
        point_fill = args.mesh_point_color + (int(255 * args.mesh_point_alpha),)
        point_radius = max(0, int(args.mesh_point_size))
        point_batch = []
        for tri_idx, tri in enumerate(triangles):
            if tri_idx % args.mesh_point_stride != 0:
                continue
            for vertex in tri:
                x, y, _ = to_screen(vertex)
                if -10 <= x <= args.width + 10 and -10 <= y <= args.height + 10:
                    if point_radius <= 0:
                        point_batch.append((int(x), int(y)))
                    else:
                        draw.ellipse(
                            (
                                x - point_radius,
                                y - point_radius,
                                x + point_radius,
                                y + point_radius,
                            ),
                            fill=point_fill,
                        )
        if point_batch:
            draw.point(point_batch, fill=point_fill)

    palette = [
        "#06b6d4",
        "#facc15",
        "#22c55e",
        "#f97316",
        "#a855f7",
        "#ef4444",
        "#38bdf8",
        "#84cc16",
    ]
    grouped = group_rows(rows)
    current_points = []
    for idx, (agent_id, agent_rows) in enumerate(grouped.items()):
        color = palette[idx % len(palette)]
        rgba = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) + (int(255 * args.path_alpha),)
        if snap_t is None:
            past = agent_rows
        else:
            past = [row for row in agent_rows if float(row["sample_t"]) <= snap_t]
        if len(past) < 2:
            continue
        line_pts = [
            to_screen(
                (
                    float(row["pos_x_m"]),
                    float(row["pos_y_m"]),
                    float(row["pos_z_m"]) + args.agent_z_offset_m,
                )
            )
            for row in past
        ]
        xy = [(p[0], p[1]) for p in line_pts]
        if len(xy) >= 2:
            draw.line(xy, fill=rgba, width=max(1, int(args.path_width)))
        current_points.append((xy[-1][0], xy[-1][1], rgba))

    for x, y, rgba in current_points:
        r = args.agent_size * 0.5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba, outline=(255, 255, 255, 245), width=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"image_written={args.output}")
    print(
        f"triangles={len(triangles)} drawn_triangles={len(prepared)} agents={len(grouped)} rows={len(rows)} snapshot_t={snap_t}"
    )
    print(f"mesh_bounds={bounds}")


def render_snapshot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cache = load_mesh_cache(args.mesh_cache)
    triangles = cache["triangles"]
    bounds = cache["bounds"]
    rows = fetch_agent_rows(args, bounds)
    snap_t = snapshot_time(rows, args.snapshot_percentile, args.snapshot_t)

    if args.renderer == "raster":
        render_snapshot_raster(args, cache, rows, snap_t)
        return

    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#eef1ee")
    fig.patch.set_facecolor("#eef1ee")

    mesh = Poly3DCollection(
        triangles,
        facecolor=(0.82, 0.84, 0.82, args.mesh_alpha),
        edgecolor=(1.0, 1.0, 1.0, args.edge_alpha),
        linewidths=args.edge_width,
    )
    ax.add_collection3d(mesh)

    palette = [
        "#06b6d4",
        "#facc15",
        "#22c55e",
        "#f97316",
        "#a855f7",
        "#ef4444",
        "#38bdf8",
        "#84cc16",
    ]
    current_points = []
    grouped = group_rows(rows)
    for idx, (agent_id, agent_rows) in enumerate(grouped.items()):
        color = palette[idx % len(palette)]
        if snap_t is None:
            past = agent_rows
        else:
            past = [row for row in agent_rows if float(row["sample_t"]) <= snap_t]
        if len(past) < 2:
            continue
        xs = [float(row["pos_x_m"]) for row in past]
        ys = [float(row["pos_y_m"]) for row in past]
        zs = [float(row["pos_z_m"]) for row in past]
        ax.plot(xs, ys, zs, color=color, linewidth=args.path_width, alpha=0.92)
        current_points.append((xs[-1], ys[-1], zs[-1], color))

    if current_points:
        ax.scatter(
            [p[0] for p in current_points],
            [p[1] for p in current_points],
            [p[2] for p in current_points],
            c=[p[3] for p in current_points],
            s=args.agent_size,
            depthshade=True,
            edgecolors="white",
            linewidths=0.8,
        )

    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    z_pad = max(8.0, (z1 - z0) * 0.08)
    ax.set_zlim(z0 - z_pad, z1 + z_pad)
    ax.view_init(elev=args.elev, azim=args.azim)
    try:
        ax.set_box_aspect((x1 - x0, y1 - y0, max(1.0, z1 - z0) * args.z_aspect))
    except Exception:
        pass
    ax.set_axis_off()
    title = f"Caldera asset-mesh movement snapshot"
    if snap_t is not None:
        title += f"  t={snap_t:.1f}s"
    ax.set_title(title, fontsize=11, pad=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(
        args.output,
        dpi=args.dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"image_written={args.output}")
    print(f"triangles={len(triangles)} agents={len(grouped)} rows={len(rows)} snapshot_t={snap_t}")
    print(f"mesh_bounds={bounds}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--usd-path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--mesh-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--max-meshes", type=nonnegative_int, default=1800)
    parser.add_argument("--mesh-stride", type=positive_int, default=4)
    parser.add_argument("--face-stride", type=positive_int, default=6)
    parser.add_argument("--max-faces", type=nonnegative_int, default=16000)
    parser.add_argument("--max-faces-per-mesh", type=nonnegative_int, default=80)
    parser.add_argument("--clip-x")
    parser.add_argument("--clip-y")
    parser.add_argument("--clip-z")
    parser.add_argument("--include-path-contains", action="append")
    parser.add_argument("--exclude-path-contains", action="append")
    parser.add_argument("--players", type=positive_int, default=14)
    parser.add_argument(
        "--match-id", help="Exact br_ground match id to render, e.g. caldera:match_31"
    )
    parser.add_argument(
        "--agent-id", help="Exact br_ground agent id to render, e.g. caldera:match_31:player_25"
    )
    parser.add_argument("--trajectory-stride", type=positive_int, default=6)
    parser.add_argument("--agent-padding-m", type=float, default=80.0)
    parser.add_argument("--agent-z-min", type=float)
    parser.add_argument("--agent-z-max", type=float)
    parser.add_argument("--sample-t-min", type=float)
    parser.add_argument("--sample-t-max", type=float)
    parser.add_argument("--snapshot-t", type=float)
    parser.add_argument("--snapshot-percentile", type=unit_interval, default=0.50)
    parser.add_argument("--width", type=positive_int, default=1920)
    parser.add_argument("--height", type=positive_int, default=1080)
    parser.add_argument("--dpi", type=positive_int, default=140)
    parser.add_argument("--elev", type=float, default=52.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--z-aspect", type=float, default=1.7)
    parser.add_argument("--mesh-alpha", type=float, default=0.78)
    parser.add_argument("--edge-alpha", type=float, default=0.16)
    parser.add_argument("--edge-width", type=float, default=0.12)
    parser.add_argument("--path-width", type=float, default=1.35)
    parser.add_argument("--path-alpha", type=unit_interval, default=0.92)
    parser.add_argument("--agent-size", type=float, default=44.0)
    parser.add_argument("--renderer", choices=["raster", "matplotlib"], default="raster")
    parser.add_argument("--background-color", default="#eef1ee")
    parser.add_argument("--raster-margin-px", type=float, default=70.0)
    parser.add_argument("--raster-zoom", type=float, default=1.0)
    parser.add_argument("--raster-edge-alpha", type=float, default=0.10)
    parser.add_argument("--raster-edge-width", type=float, default=1.0)
    parser.add_argument("--mesh-point-alpha", type=unit_interval, default=0.0)
    parser.add_argument("--mesh-point-stride", type=positive_int, default=8)
    parser.add_argument("--mesh-point-size", type=float, default=0.0)
    parser.add_argument("--mesh-point-color", type=rgb_hex, default=(68, 76, 72))
    parser.add_argument("--min-triangle-area", type=float, default=0.0001)
    parser.add_argument("--max-triangle-area", type=float, default=0.0)
    parser.add_argument("--agent-z-offset-m", type=float, default=2.5)
    parser.add_argument("--mesh-tint", type=rgb_hex, default=(205, 210, 205))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.usd_path = args.usd_path.resolve()
    args.mesh_cache = args.mesh_cache.resolve()
    args.output = args.output.resolve()
    if args.extract_only:
        extract_mesh_cache(args)
        return
    ensure_mesh_cache(args)
    render_snapshot(args)


if __name__ == "__main__":
    main()
