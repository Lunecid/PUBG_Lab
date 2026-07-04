"""Render a terrain-aware Caldera trajectory experience snapshot.

The figure combines a local oblique view of the reconstructed semantic mesh
with trajectory diagnostics:

- agent altitude and nearest terrain elevation;
- height above nearest terrain;
- nearest building distance;
- rough semantic context along the path.

It intentionally uses only the semantic triangle cache and the DB trajectory
rows so it can be rerun for any exact Caldera match/player pair.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stdb.conn import connect


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
    "terrain": 120_000,
    "coast_beach": 16_000,
    "water": 14_000,
    "vegetation": 40_000,
    "road": 56_000,
    "runway": 18_000,
    "building": 100_000,
    "prop": 8_000,
}

BASE_COLORS = {
    "coast_beach": (187, 162, 98, 165),
    "water": (30, 105, 124, 190),
    "vegetation": (28, 96, 44, 150),
    "road": (183, 187, 177, 205),
    "runway": (75, 82, 80, 225),
    "building": (39, 44, 44, 215),
    "prop": (113, 109, 96, 125),
}

CONTEXT_COLORS = {
    "terrain": (112, 145, 93),
    "road": (208, 211, 197),
    "runway": (115, 124, 121),
    "building": (238, 178, 71),
    "vegetation": (47, 148, 72),
    "water": (46, 151, 178),
    "coast_beach": (220, 193, 118),
}

PANEL_BG = (16, 23, 24)
PANEL_EDGE = (54, 66, 66)
TEXT = (235, 243, 240)
MUTED = (171, 190, 184)
GRID = (48, 60, 60)
CYAN = (23, 201, 224)
ORANGE = (238, 178, 71)
RED = (235, 88, 74)
GREEN = (112, 190, 106)


def parse_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    lo, hi = (float(part.strip()) for part in parts)
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be greater than lower bound")
    return (lo, hi)


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))


def sample_evenly(values: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0 or len(values) <= max_count:
        return values
    idx = np.linspace(0, len(values) - 1, max_count, dtype=np.int64)
    return values[idx]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle(rect, radius=8, fill=PANEL_BG, outline=PANEL_EDGE, width=1)
    draw.text((rect[0] + 18, rect[1] + 14), title, fill=TEXT, font=font)


def fetch_agent_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sample_t, pos_x_m, pos_y_m, pos_z_m, alive_state
                  FROM br_ground.trajectory_samples
                 WHERE br_match_id = %s
                   AND br_agent_id = %s
                   AND source_artifact_id = %s
                   AND sample_t >= %s
                   AND pos_x_m IS NOT NULL
                   AND pos_y_m IS NOT NULL
                   AND pos_z_m IS NOT NULL
                 ORDER BY sample_t
                """,
                (args.match_id, args.agent_id, args.source_artifact_id, args.sample_t_min),
            )
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    if args.snapshot_t is not None:
        rows = [row for row in rows if float(row["sample_t"]) <= args.snapshot_t]
    return rows


def compute_bounds(
    rows: list[dict[str, Any]],
    explicit_x: tuple[float, float] | None,
    explicit_y: tuple[float, float] | None,
    explicit_z: tuple[float, float] | None,
    xy_margin: float,
    z_margin: float,
) -> dict[str, tuple[float, float]]:
    pts = np.asarray([(float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["pos_z_m"])) for row in rows], dtype=np.float32)
    bounds = {
        "x": (float(np.min(pts[:, 0]) - xy_margin), float(np.max(pts[:, 0]) + xy_margin)),
        "y": (float(np.min(pts[:, 1]) - xy_margin), float(np.max(pts[:, 1]) + xy_margin)),
        "z": (float(np.min(pts[:, 2]) - z_margin), float(np.max(pts[:, 2]) + z_margin)),
    }
    if explicit_x is not None:
        bounds["x"] = explicit_x
    if explicit_y is not None:
        bounds["y"] = explicit_y
    if explicit_z is not None:
        bounds["z"] = explicit_z
    return bounds


def localize_triangles(raw: dict[str, np.ndarray], bounds: dict[str, tuple[float, float]], margin: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for category, tris in raw.items():
        if not len(tris):
            out[category] = tris
            continue
        centroids = tris.mean(axis=1)
        mask = (
            (centroids[:, 0] >= bounds["x"][0] - margin)
            & (centroids[:, 0] <= bounds["x"][1] + margin)
            & (centroids[:, 1] >= bounds["y"][0] - margin)
            & (centroids[:, 1] <= bounds["y"][1] + margin)
        )
        out[category] = sample_evenly(tris[mask], DISPLAY_BUDGETS[category])
    return out


def category_tree(tris: np.ndarray) -> tuple[cKDTree | None, np.ndarray]:
    if not len(tris):
        return None, np.empty((0, 3), dtype=np.float32)
    centroids = tris.mean(axis=1).astype(np.float32)
    return cKDTree(centroids[:, :2]), centroids


def query_nearest(tree: cKDTree | None, centroids: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tree is None or len(centroids) == 0:
        n = len(points_xy)
        return np.full(n, np.inf, dtype=np.float32), np.full(n, -1, dtype=np.int64)
    dist, idx = tree.query(points_xy, k=1)
    return dist.astype(np.float32), idx.astype(np.int64)


def attach_spatial_diagnostics(rows: list[dict[str, Any]], local_tris: dict[str, np.ndarray]) -> dict[str, Any]:
    points_xy = np.asarray([(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in rows], dtype=np.float32)
    trees: dict[str, tuple[cKDTree | None, np.ndarray]] = {category: category_tree(local_tris[category]) for category in CATEGORIES}
    terrain_dist, terrain_idx = query_nearest(*trees["terrain"], points_xy)
    building_dist, _ = query_nearest(*trees["building"], points_xy)

    terrain_centroids = trees["terrain"][1]
    agent_z = np.asarray([float(row["pos_z_m"]) for row in rows], dtype=np.float32)
    terrain_z = np.where(terrain_idx >= 0, terrain_centroids[terrain_idx, 2], np.nan).astype(np.float32)
    height_above = agent_z - terrain_z

    distances: dict[str, np.ndarray] = {}
    for category in CATEGORIES:
        distances[category], _ = query_nearest(*trees[category], points_xy)

    contexts = []
    for idx, row in enumerate(rows):
        if distances["building"][idx] <= 25:
            context = "building"
        elif distances["road"][idx] <= 9:
            context = "road"
        elif distances["runway"][idx] <= 14:
            context = "runway"
        elif distances["vegetation"][idx] <= 10:
            context = "vegetation"
        elif distances["water"][idx] <= 14:
            context = "water"
        elif distances["coast_beach"][idx] <= 14:
            context = "coast_beach"
        else:
            context = "terrain"
        row["terrain_z"] = float(terrain_z[idx])
        row["height_above_terrain_m"] = float(height_above[idx])
        row["nearest_terrain_xy_m"] = float(terrain_dist[idx])
        row["nearest_building_xy_m"] = float(building_dist[idx])
        row["semantic_context"] = context
        contexts.append(context)

    finite_building = building_dist[np.isfinite(building_dist)]
    finite_height = height_above[np.isfinite(height_above)]
    return {
        "agent_z": agent_z,
        "terrain_z": terrain_z,
        "height_above_terrain": height_above,
        "building_dist": building_dist,
        "context_counts": dict(Counter(contexts)),
        "near_building_ratio_25m": float(np.mean(building_dist <= 25)) if len(building_dist) else None,
        "near_road_ratio_9m": float(np.mean(distances["road"] <= 9)) if len(rows) else None,
        "height_above_terrain_stats_m": {
            "min": float(np.nanmin(finite_height)) if len(finite_height) else None,
            "median": float(np.nanmedian(finite_height)) if len(finite_height) else None,
            "p90": float(np.nanpercentile(finite_height, 90)) if len(finite_height) else None,
            "max": float(np.nanmax(finite_height)) if len(finite_height) else None,
        },
        "agent_altitude_stats_m": {
            "min": float(np.nanmin(agent_z)),
            "median": float(np.nanmedian(agent_z)),
            "max": float(np.nanmax(agent_z)),
        },
        "building_distance_stats_m": {
            "min": float(np.nanmin(finite_building)) if len(finite_building) else None,
            "median": float(np.nanmedian(finite_building)) if len(finite_building) else None,
            "p25": float(np.nanpercentile(finite_building, 25)) if len(finite_building) else None,
        },
    }


def terrain_color(z: float, z_bounds: tuple[float, float]) -> tuple[int, int, int, int]:
    z0, z1 = z_bounds
    t = max(0.0, min(1.0, (float(z) - z0) / max(1e-6, z1 - z0)))
    stops = [
        (0.00, np.array((45, 77, 51), dtype=np.float32)),
        (0.45, np.array((96, 126, 75), dtype=np.float32)),
        (0.72, np.array((146, 139, 95), dtype=np.float32)),
        (1.00, np.array((210, 204, 166), dtype=np.float32)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            rgb = (c0 * (1.0 - u) + c1 * u).astype(np.uint8)
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]), 235)
    rgb = stops[-1][1].astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), 235)


def altitude_color(value: float, lo: float, hi: float) -> tuple[int, int, int, int]:
    t = max(0.0, min(1.0, (float(value) - lo) / max(1e-6, hi - lo)))
    if t < 0.55:
        u = t / 0.55
        a = np.array((18, 201, 224), dtype=np.float32)
        b = np.array((128, 220, 112), dtype=np.float32)
    else:
        u = (t - 0.55) / 0.45
        a = np.array((128, 220, 112), dtype=np.float32)
        b = np.array((238, 178, 71), dtype=np.float32)
    rgb = (a * (1.0 - u) + b * u).astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)


def make_iso_projector(
    bounds: dict[str, tuple[float, float]],
    rect: tuple[int, int, int, int],
    z_exag: float,
) -> Callable[[np.ndarray], np.ndarray]:
    theta = math.radians(31)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    pitch = 0.50
    cx = (bounds["x"][0] + bounds["x"][1]) * 0.5
    cy = (bounds["y"][0] + bounds["y"][1]) * 0.5
    z0 = bounds["z"][0]

    def raw_project(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        x = pts[:, 0] - cx
        y = pts[:, 1] - cy
        z = pts[:, 2] - z0
        u = x * cos_t - y * sin_t
        v = (x * sin_t + y * cos_t) * pitch - z * z_exag
        return np.column_stack([u, v])

    corners = np.asarray(
        [
            (x, y, z)
            for x in bounds["x"]
            for y in bounds["y"]
            for z in (bounds["z"][0], bounds["z"][1])
        ],
        dtype=np.float32,
    )
    uv = raw_project(corners)
    u0, v0 = uv.min(axis=0)
    u1, v1 = uv.max(axis=0)
    left, top, right, bottom = rect
    scale = min((right - left - 44) / max(1e-6, u1 - u0), (bottom - top - 72) / max(1e-6, v1 - v0))
    offset_x = (left + right) * 0.5 - (u0 + u1) * 0.5 * scale
    offset_y = (top + bottom) * 0.5 - (v0 + v1) * 0.5 * scale + 16

    def project(points: np.ndarray) -> np.ndarray:
        uv2 = raw_project(points)
        out = np.empty_like(uv2)
        out[:, 0] = offset_x + uv2[:, 0] * scale
        out[:, 1] = offset_y + uv2[:, 1] * scale
        return np.round(out).astype(np.int32)

    return project


def draw_oblique_scene(
    image: Image.Image,
    rows: list[dict[str, Any]],
    local_tris: dict[str, np.ndarray],
    bounds: dict[str, tuple[float, float]],
    rect: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw_panel(draw, rect, "Local 3D terrain window: trajectory colored by altitude", font)
    scene_rect = (rect[0] + 18, rect[1] + 58, rect[2] - 18, rect[3] - 18)
    draw.rectangle(scene_rect, fill=(12, 20, 20, 255))

    projector = make_iso_projector(bounds, scene_rect, z_exag=1.72)
    path_xy = np.asarray([(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in rows], dtype=np.float32)
    path_tree = cKDTree(path_xy)

    draw_items = []
    for category in CATEGORIES:
        tris = local_tris[category]
        if not len(tris):
            continue
        centroids = tris.mean(axis=1)
        depth = (centroids[:, 0] * 0.20 + centroids[:, 1] * 0.75 + centroids[:, 2] * 0.10).astype(float)
        if category == "building":
            dist_to_path, _ = path_tree.query(centroids[:, :2], k=1)
        else:
            dist_to_path = np.full(len(tris), np.inf, dtype=np.float32)
        for tri, centroid, z_depth, near_dist in zip(tris, centroids, depth, dist_to_path):
            draw_items.append((float(z_depth), category, tri, float(centroid[2]), float(near_dist)))
    draw_items.sort(key=lambda item: item[0])

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(layer, "RGBA")
    for _, category, tri, z, near_dist in draw_items:
        pts = projector(tri)
        poly = [tuple(map(int, point)) for point in pts]
        if category == "terrain":
            fill = terrain_color(z, bounds["z"])
        elif category == "building" and near_dist <= 35:
            fill = (189, 137, 63, 230)
        else:
            fill = BASE_COLORS[category]
        ldraw.polygon(poly, fill=fill)
    image.alpha_composite(layer)
    draw = ImageDraw.Draw(image, "RGBA")

    path3 = np.asarray(
        [(float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["pos_z_m"]) + 4.0) for row in rows],
        dtype=np.float32,
    )
    projected = projector(path3)
    agent_z = np.asarray([float(row["pos_z_m"]) for row in rows], dtype=np.float32)
    z_lo, z_hi = float(np.nanpercentile(agent_z, 3)), float(np.nanpercentile(agent_z, 97))
    for p0, p1 in zip(projected[:-1], projected[1:]):
        draw.line([tuple(p0), tuple(p1)], fill=(3, 8, 9, 230), width=12)
    for idx, (p0, p1) in enumerate(zip(projected[:-1], projected[1:])):
        color = altitude_color(float(agent_z[idx + 1]), z_lo, z_hi)
        draw.line([tuple(p0), tuple(p1)], fill=color, width=7)

    def marker(index: int, label: str, value: str, offset: tuple[int, int]) -> None:
        x, y = map(int, projected[index])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(4, 12, 13, 245), outline=TEXT, width=2)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ORANGE if "building" in label.lower() else CYAN)
        tx, ty = x + offset[0], y + offset[1]
        w1, h1 = text_size(draw, label, small)
        w2, h2 = text_size(draw, value, small)
        box = (tx - 8, ty - 7, tx + max(w1, w2) + 8, ty + h1 + h2 + 9)
        draw.rounded_rectangle(box, radius=5, fill=(8, 14, 15, 225), outline=(74, 88, 88, 220), width=1)
        draw.line((x, y, tx, ty + 8), fill=(214, 224, 218, 160), width=1)
        draw.text((tx, ty), label, fill=TEXT, font=small)
        draw.text((tx, ty + h1 + 2), value, fill=MUTED, font=small)

    highest = int(np.nanargmax(agent_z))
    closest_building = int(np.nanargmin([float(row["nearest_building_xy_m"]) for row in rows]))
    marker(0, "terrain contact", f"t={float(rows[0]['sample_t']):.1f}s z={float(rows[0]['pos_z_m']):.1f}m", (18, -52))
    if highest != 0:
        marker(highest, "highest path", f"z={float(rows[highest]['pos_z_m']):.1f}m", (16, -48))
    marker(closest_building, "building pressure", f"d={float(rows[closest_building]['nearest_building_xy_m']):.1f}m", (22, 18))
    marker(len(rows) - 1, "final position", f"z={float(rows[-1]['pos_z_m']):.1f}m", (20, -44))

    legend_x = scene_rect[0] + 16
    legend_y = scene_rect[3] - 74
    for idx, (label, color) in enumerate(
        [
            ("low altitude", (18, 201, 224)),
            ("higher altitude", (238, 178, 71)),
            ("near-path buildings", (189, 137, 63)),
        ]
    ):
        y = legend_y + idx * 22
        draw.rounded_rectangle((legend_x, y, legend_x + 28, y + 12), radius=3, fill=color + (245,))
        draw.text((legend_x + 38, y - 2), label, fill=MUTED, font=small)


def normalize_points(times: np.ndarray, values: np.ndarray, rect: tuple[int, int, int, int], y_min: float, y_max: float) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    t0, t1 = float(np.min(times)), float(np.max(times))
    xs = left + (times - t0) / max(1e-6, t1 - t0) * (right - left)
    ys = bottom - (values - y_min) / max(1e-6, y_max - y_min) * (bottom - top)
    ys = np.clip(ys, top, bottom)
    return [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]


def draw_series_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    times: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
    y_limits: tuple[float, float] | None = None,
    threshold: tuple[float, str, tuple[int, int, int]] | None = None,
) -> None:
    draw_panel(draw, rect, title, font)
    plot = (rect[0] + 58, rect[1] + 58, rect[2] - 22, rect[3] - 42)
    values = np.concatenate([item[1][np.isfinite(item[1])] for item in series if np.isfinite(item[1]).any()])
    if y_limits is None:
        y0 = float(np.nanmin(values))
        y1 = float(np.nanmax(values))
        pad = max(1.0, (y1 - y0) * 0.10)
        y0 -= pad
        y1 += pad
    else:
        y0, y1 = y_limits
    for frac in np.linspace(0.0, 1.0, 5):
        y = int(round(plot[3] - frac * (plot[3] - plot[1])))
        draw.line((plot[0], y, plot[2], y), fill=GRID, width=1)
        label = f"{y0 + frac * (y1 - y0):.0f}"
        draw.text((rect[0] + 18, y - 7), label, fill=MUTED, font=small)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=PANEL_EDGE, width=1)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=PANEL_EDGE, width=1)

    if threshold is not None:
        value, label, color = threshold
        y = int(round(plot[3] - (value - y0) / max(1e-6, y1 - y0) * (plot[3] - plot[1])))
        draw.line((plot[0], y, plot[2], y), fill=color + (180,), width=2)
        draw.text((plot[0] + 8, y - 18), label, fill=color, font=small)

    for idx, (label, vals, color) in enumerate(series):
        pts = normalize_points(times, vals, plot, y0, y1)
        if len(pts) >= 2:
            draw.line(pts, fill=color + (245,), width=3)
        lx = plot[0] + 12 + idx * 180
        ly = rect[3] - 29
        draw.line((lx, ly + 7, lx + 26, ly + 7), fill=color + (245,), width=4)
        draw.text((lx + 34, ly), label, fill=MUTED, font=small)

    draw.text((plot[0], rect[3] - 28), f"t={float(times[0]):.1f}s", fill=MUTED, font=small)
    end_label = f"t={float(times[-1]):.1f}s"
    w, _ = text_size(draw, end_label, small)
    draw.text((plot[2] - w, rect[3] - 28), end_label, fill=MUTED, font=small)


def draw_context_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    rows: list[dict[str, Any]],
    times: np.ndarray,
    diagnostics: dict[str, Any],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    draw_panel(draw, rect, "Semantic context along path", font)
    strip = (rect[0] + 28, rect[1] + 74, rect[2] - 28, rect[1] + 118)
    t0, t1 = float(times[0]), float(times[-1])
    for idx, row in enumerate(rows):
        ta = float(row["sample_t"])
        tb = float(rows[min(idx + 1, len(rows) - 1)]["sample_t"]) if idx < len(rows) - 1 else t1
        x0 = int(round(strip[0] + (ta - t0) / max(1e-6, t1 - t0) * (strip[2] - strip[0])))
        x1 = int(round(strip[0] + (tb - t0) / max(1e-6, t1 - t0) * (strip[2] - strip[0])))
        color = CONTEXT_COLORS.get(row["semantic_context"], CONTEXT_COLORS["terrain"])
        draw.rectangle((x0, strip[1], max(x1, x0 + 1), strip[3]), fill=color + (235,))
    draw.rectangle(strip, outline=PANEL_EDGE, width=1)

    counts = Counter(row["semantic_context"] for row in rows)
    total = max(1, len(rows))
    legend_x = rect[0] + 28
    legend_y = strip[3] + 22
    for idx, (label, color) in enumerate(CONTEXT_COLORS.items()):
        x = legend_x + (idx % 4) * 210
        y = legend_y + (idx // 4) * 28
        draw.rounded_rectangle((x, y, x + 24, y + 14), radius=3, fill=color + (240,))
        draw.text((x + 34, y - 2), f"{label} {counts[label] / total * 100:.0f}%", fill=MUTED, font=small)

    h = diagnostics["height_above_terrain_stats_m"]
    b = diagnostics["building_distance_stats_m"]
    a = diagnostics["agent_altitude_stats_m"]
    stats = [
        ("altitude", f"{a['min']:.1f}-{a['max']:.1f}m, med {a['median']:.1f}m"),
        ("above terrain", f"med {h['median']:.1f}m, p90 {h['p90']:.1f}m"),
        ("building distance", f"min {b['min']:.1f}m, p25 {b['p25']:.1f}m"),
        ("near building", f"{diagnostics['near_building_ratio_25m'] * 100:.1f}% <=25m"),
        ("near road", f"{diagnostics['near_road_ratio_9m'] * 100:.1f}% <=9m"),
    ]
    stat_y = rect[3] - 94
    for idx, (label, value) in enumerate(stats):
        x = rect[0] + 28 + idx * 490
        draw.rounded_rectangle((x, stat_y, x + 440, rect[3] - 26), radius=6, fill=(11, 17, 18), outline=(45, 58, 58), width=1)
        draw.text((x + 16, stat_y + 13), label, fill=TEXT, font=small)
        draw.text((x + 16, stat_y + 39), value, fill=MUTED, font=small)


def diagnostics_for_manifest(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_counts": diagnostics["context_counts"],
        "near_building_ratio_25m": diagnostics["near_building_ratio_25m"],
        "near_road_ratio_9m": diagnostics["near_road_ratio_9m"],
        "height_above_terrain_stats_m": diagnostics["height_above_terrain_stats_m"],
        "agent_altitude_stats_m": diagnostics["agent_altitude_stats_m"],
        "building_distance_stats_m": diagnostics["building_distance_stats_m"],
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    payload = pickle.load(args.cache.open("rb"))
    raw = {category: array_to_tris(payload["triangles"][category]) for category in CATEGORIES}
    rows = fetch_agent_rows(args)
    if not rows:
        raise SystemExit("No trajectory rows matched the requested match/player/time filters.")

    bounds = compute_bounds(rows, args.bounds_x, args.bounds_y, args.bounds_z, args.xy_margin, args.z_margin)
    local_tris = localize_triangles(raw, bounds, args.mesh_margin)
    diagnostics = attach_spatial_diagnostics(rows, local_tris)

    image = Image.new("RGBA", (args.width, args.height), (9, 14, 15, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(32, bold=True)
    panel_font = load_font(20, bold=True)
    small = load_font(16)
    tiny = load_font(14)

    title = "Caldera player terrain experience snapshot"
    subtitle = (
        f"{args.match_id} / {args.agent_id.split(':')[-1]} | "
        f"points={len(rows)} | t={float(rows[0]['sample_t']):.1f}-{float(rows[-1]['sample_t']):.1f}s | "
        f"local bounds x={bounds['x'][0]:.0f}..{bounds['x'][1]:.0f}, y={bounds['y'][0]:.0f}..{bounds['y'][1]:.0f}"
    )
    draw.text((36, 24), title, fill=TEXT, font=title_font)
    draw.text((36, 62), subtitle, fill=MUTED, font=small)

    scene_rect = (34, 96, 1664, 1228)
    altitude_rect = (1700, 96, 2566, 430)
    dz_rect = (1700, 462, 2566, 796)
    building_rect = (1700, 828, 2566, 1162)
    context_rect = (34, 1262, 2566, 1654)

    draw_oblique_scene(image, rows, local_tris, bounds, scene_rect, panel_font, tiny)
    draw = ImageDraw.Draw(image, "RGBA")

    times = np.asarray([float(row["sample_t"]) for row in rows], dtype=np.float32)
    agent_z = diagnostics["agent_z"]
    terrain_z = diagnostics["terrain_z"]
    height_above = diagnostics["height_above_terrain"]
    building_dist = np.clip(diagnostics["building_dist"], 0, 140)

    altitude_limits = (
        float(min(np.nanmin(agent_z), np.nanmin(terrain_z)) - 8),
        float(max(np.nanmax(agent_z), np.nanmax(terrain_z)) + 8),
    )
    draw_series_panel(
        draw,
        altitude_rect,
        "Altitude vs nearest terrain",
        times,
        [("agent z", agent_z, CYAN), ("terrain z", terrain_z, GREEN)],
        panel_font,
        tiny,
        y_limits=altitude_limits,
    )
    draw_series_panel(
        draw,
        dz_rect,
        "Height above terrain",
        times,
        [("agent z - terrain z", height_above, ORANGE)],
        panel_font,
        tiny,
        y_limits=(float(np.nanmin(height_above) - 2), float(max(20.0, np.nanpercentile(height_above, 98) + 4))),
        threshold=(10.0, "10m exposure", RED),
    )
    draw_series_panel(
        draw,
        building_rect,
        "Nearest building distance",
        times,
        [("distance", building_dist, ORANGE)],
        panel_font,
        tiny,
        y_limits=(0.0, 140.0),
        threshold=(25.0, "25m pressure", RED),
    )
    draw_context_panel(draw, context_rect, rows, times, diagnostics, panel_font, tiny)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(args.output)

    manifest = {
        "output": str(args.output.relative_to(ROOT)),
        "cache": str(args.cache.relative_to(ROOT)),
        "match_id": args.match_id,
        "agent_id": args.agent_id,
        "source_artifact_id": args.source_artifact_id,
        "rows": len(rows),
        "time_range_s": [float(rows[0]["sample_t"]), float(rows[-1]["sample_t"])],
        "bounds": bounds,
        "display_triangles": {category: int(len(local_tris[category])) for category in CATEGORIES},
        "diagnostics": diagnostics_for_manifest(diagnostics),
        "notes": [
            "Terrain elevation is approximated by the nearest local semantic terrain triangle centroid.",
            "Building influence is approximated by XY distance to the nearest local semantic building triangle centroid.",
            "Semantic context is priority-based proximity, not an official surface label.",
        ],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--source-artifact-id", default="caldera_minimal_layers_breadcrumbs_usd")
    parser.add_argument("--sample-t-min", type=float, default=55.0)
    parser.add_argument("--snapshot-t", type=float)
    parser.add_argument("--bounds-x", type=parse_range)
    parser.add_argument("--bounds-y", type=parse_range)
    parser.add_argument("--bounds-z", type=parse_range)
    parser.add_argument("--xy-margin", type=float, default=105.0)
    parser.add_argument("--z-margin", type=float, default=18.0)
    parser.add_argument("--mesh-margin", type=float, default=70.0)
    parser.add_argument("--width", type=int, default=2600)
    parser.add_argument("--height", type=int, default=1700)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = render(args)
    print(f"image_written={args.output}")
    print(
        json.dumps(
            {
                "rows": manifest["rows"],
                "time_range_s": manifest["time_range_s"],
                "bounds": manifest["bounds"],
                "diagnostics": manifest["diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
