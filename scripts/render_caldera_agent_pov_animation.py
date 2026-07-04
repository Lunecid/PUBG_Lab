"""Render a first-person-ish Caldera trajectory animation from semantic mesh data.

This is not a game replay renderer.  It uses only the reconstructed Caldera
semantic triangle cache and DB trajectory samples, then projects nearby map
geometry from the player's estimated eye position and movement heading.  The
result is a compact visual proxy for what terrain/building pressure may have
felt like from the player's point of view.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
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

BUDGETS = {
    "terrain": 28_000,
    "coast_beach": 4_000,
    "water": 3_500,
    "vegetation": 12_000,
    "road": 10_000,
    "runway": 3_000,
    "building": 22_000,
    "prop": 2_500,
}

BASE_COLORS = {
    "terrain": (86, 119, 72),
    "coast_beach": (188, 165, 104),
    "water": (28, 104, 126),
    "vegetation": (27, 89, 43),
    "road": (177, 181, 170),
    "runway": (73, 80, 79),
    "building": (64, 67, 64),
    "prop": (112, 106, 94),
}

TEXT = (232, 241, 238)
MUTED = (174, 194, 188)
CYAN = (19, 205, 225)
ORANGE = (240, 173, 65)
RED = (225, 72, 72)


def parse_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    lo, hi = (float(part.strip()) for part in parts)
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be greater than lower bound")
    return lo, hi


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    tris = np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))
    sane = np.isfinite(tris).all(axis=(1, 2)) & (np.abs(tris) < 10_000).all(axis=(1, 2))
    return tris[sane]


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
                   AND (%s IS NULL OR sample_t <= %s)
                   AND pos_x_m IS NOT NULL
                   AND pos_y_m IS NOT NULL
                   AND pos_z_m IS NOT NULL
                 ORDER BY sample_t
                """,
                (
                    args.match_id,
                    args.agent_id,
                    args.source_artifact_id,
                    args.sample_t_min,
                    args.sample_t_max,
                    args.sample_t_max,
                ),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def localize_tris(raw: dict[str, np.ndarray], rows: list[dict[str, Any]], margin: float, args: argparse.Namespace) -> dict[str, np.ndarray]:
    pts = np.asarray([(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in rows], dtype=np.float32)
    zs = np.asarray([float(row["pos_z_m"]) for row in rows], dtype=np.float32)
    x0, y0 = np.min(pts, axis=0) - margin
    x1, y1 = np.max(pts, axis=0) + margin
    z0 = float(np.min(zs) - 80.0)
    z1 = float(np.max(zs) + 220.0)
    local = {}
    for category, tris in raw.items():
        if not len(tris):
            local[category] = tris
            continue
        centroids = tris.mean(axis=1)
        finite = np.isfinite(centroids).all(axis=1)
        tris = tris[finite]
        centroids = centroids[finite]
        if not len(tris):
            local[category] = tris
            continue
        mask = (
            (centroids[:, 0] >= x0)
            & (centroids[:, 0] <= x1)
            & (centroids[:, 1] >= y0)
            & (centroids[:, 1] <= y1)
            & (centroids[:, 2] >= z0)
            & (centroids[:, 2] <= z1)
        )
        local[category] = sample_evenly(tris[mask], getattr(args, f"{category}_triangles"))
    return local


def category_tree(tris: np.ndarray) -> tuple[cKDTree | None, np.ndarray]:
    if not len(tris):
        return None, np.empty((0, 3), dtype=np.float32)
    centroids = tris.mean(axis=1).astype(np.float32)
    return cKDTree(centroids[:, :2]), centroids


def query_tree(tree: cKDTree | None, centroids: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tree is None or len(centroids) == 0:
        n = len(points_xy)
        return np.full(n, np.inf, dtype=np.float32), np.full(n, -1, dtype=np.int64)
    dist, idx = tree.query(points_xy, k=1)
    return dist.astype(np.float32), idx.astype(np.int64)


def attach_diagnostics(rows: list[dict[str, Any]], local: dict[str, np.ndarray]) -> dict[str, Any]:
    pts = np.asarray([(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in rows], dtype=np.float32)
    terrain_tree, terrain_centroids = category_tree(local["terrain"])
    building_tree, building_centroids = category_tree(local["building"])
    road_tree, road_centroids = category_tree(local["road"])
    terrain_dist, terrain_idx = query_tree(terrain_tree, terrain_centroids, pts)
    building_dist, _ = query_tree(building_tree, building_centroids, pts)
    road_dist, _ = query_tree(road_tree, road_centroids, pts)
    terrain_z = np.where(terrain_idx >= 0, terrain_centroids[terrain_idx, 2], np.nan).astype(np.float32)
    agent_z = np.asarray([float(row["pos_z_m"]) for row in rows], dtype=np.float32)
    height_above = agent_z - terrain_z
    for idx, row in enumerate(rows):
        row["terrain_z"] = float(terrain_z[idx])
        row["height_above_terrain_m"] = float(height_above[idx])
        row["nearest_building_xy_m"] = float(building_dist[idx])
        row["nearest_road_xy_m"] = float(road_dist[idx])
        row["nearest_terrain_xy_m"] = float(terrain_dist[idx])
    return {
        "terrain_z": terrain_z,
        "height_above_terrain": height_above,
        "building_dist": building_dist,
        "road_dist": road_dist,
    }


def frame_indices(n: int, count: int) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted(set(int(round(v)) for v in np.linspace(0, n - 1, count)))


def heading_at(rows: list[dict[str, Any]], idx: int, fallback: np.ndarray | None = None) -> np.ndarray:
    n = len(rows)
    lo = max(0, idx - 4)
    hi = min(n - 1, idx + 5)
    p0 = np.array([float(rows[lo]["pos_x_m"]), float(rows[lo]["pos_y_m"])], dtype=np.float32)
    p1 = np.array([float(rows[hi]["pos_x_m"]), float(rows[hi]["pos_y_m"])], dtype=np.float32)
    v = p1 - p0
    norm = float(np.linalg.norm(v))
    if norm < 1e-4:
        return fallback if fallback is not None else np.array([0.0, 1.0], dtype=np.float32)
    return v / norm


def terrain_rgb(z: float, z_bounds: tuple[float, float]) -> tuple[int, int, int]:
    lo, hi = z_bounds
    t = max(0.0, min(1.0, (float(z) - lo) / max(1e-6, hi - lo)))
    low = np.array((47, 79, 55), dtype=np.float32)
    mid = np.array((111, 143, 83), dtype=np.float32)
    high = np.array((190, 179, 129), dtype=np.float32)
    if t < 0.55:
        u = t / 0.55
        rgb = low * (1 - u) + mid * u
    else:
        u = (t - 0.55) / 0.45
        rgb = mid * (1 - u) + high * u
    return tuple(int(v) for v in rgb)


def shade(color: tuple[int, int, int], dist: float, max_dist: float, boost: float = 1.0) -> tuple[int, int, int, int]:
    fog = max(0.0, min(1.0, dist / max_dist))
    rgb = np.array(color, dtype=np.float32)
    sky = np.array((165, 190, 188), dtype=np.float32)
    rgb = rgb * (1.0 - fog * 0.58) + sky * (fog * 0.58)
    rgb *= boost
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = int(255 - fog * 75)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)


def camera_basis(heading: np.ndarray, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitch = math.radians(pitch_deg)
    xy = heading.astype(np.float32)
    xy = xy / max(1e-6, float(np.linalg.norm(xy)))
    forward = np.array([math.cos(pitch) * xy[0], math.cos(pitch) * xy[1], math.sin(pitch)], dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / max(1e-6, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up = up / max(1e-6, float(np.linalg.norm(up)))
    return forward, right, up


def project(points: np.ndarray, camera: np.ndarray, forward: np.ndarray, right: np.ndarray, up: np.ndarray, focal: float, width: int, height: int, near: float) -> tuple[np.ndarray, np.ndarray]:
    rel = points.astype(np.float32) - camera
    z = rel @ forward
    x = rel @ right
    y = rel @ up
    valid = z > near
    sx = width * 0.5 + focal * x / np.maximum(z, near)
    sy = height * 0.53 - focal * y / np.maximum(z, near)
    return np.column_stack([sx, sy]).astype(np.float32), valid


def draw_sky_and_haze(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    for y in range(height):
        t = y / max(1, height - 1)
        if y < height * 0.55:
            u = y / (height * 0.55)
            top = np.array((128, 179, 200), dtype=np.float32)
            bottom = np.array((205, 215, 205), dtype=np.float32)
            rgb = top * (1 - u) + bottom * u
        else:
            u = (y - height * 0.55) / (height * 0.45)
            top = np.array((68, 84, 72), dtype=np.float32)
            bottom = np.array((37, 48, 39), dtype=np.float32)
            rgb = top * (1 - u) + bottom * u
        draw.line((0, y, width, y), fill=tuple(int(v) for v in rgb))
    horizon = int(height * 0.55)
    draw.rectangle((0, horizon - 3, width, horizon + 5), fill=(210, 221, 210, 55))


def clay_color(category: str, z: float, z_bounds: tuple[float, float]) -> tuple[int, int, int]:
    if category == "terrain":
        lo, hi = z_bounds
        t = max(0.0, min(1.0, (float(z) - lo) / max(1e-6, hi - lo)))
        low = np.array((194, 210, 204), dtype=np.float32)
        high = np.array((238, 243, 239), dtype=np.float32)
        rgb = low * (1.0 - t) + high * t
        return tuple(int(v) for v in rgb)
    if category == "building":
        return (232, 238, 237)
    if category in {"road", "runway"}:
        return (204, 212, 209)
    if category == "prop":
        return (218, 219, 211)
    return (210, 224, 220)


def draw_hud(draw: ImageDraw.ImageDraw, row: dict[str, Any], width: int, height: int, mini_path: list[tuple[float, float]], current_idx: int, font: ImageFont.ImageFont, small: ImageFont.ImageFont) -> None:
    t = float(row["sample_t"])
    z = float(row["pos_z_m"])
    dz = float(row["height_above_terrain_m"])
    bdist = float(row["nearest_building_xy_m"])
    rdist = float(row["nearest_road_xy_m"])

    draw.rounded_rectangle((18, 18, 244, 106), radius=8, fill=(8, 13, 14, 175), outline=(235, 244, 239, 70), width=1)
    draw.text((34, 30), f"t {t:6.1f}s", fill=(242, 248, 246), font=font)
    draw.text((34, 61), f"z {z:5.1f}m  dz {dz:4.1f}m", fill=MUTED, font=small)
    draw.text((34, 82), f"building {bdist:4.1f}m  road {rdist:4.1f}m", fill=MUTED, font=small)

    draw.rectangle((width - 258, 22, width - 26, 82), fill=(9, 14, 15, 155), outline=(235, 244, 239, 65), width=1)
    pressure = max(0.0, min(1.0, 1.0 - bdist / 25.0))
    bar_w = int(190 * pressure)
    draw.text((width - 244, 33), "STRUCTURE PRESSURE", fill=MUTED, font=small)
    draw.rounded_rectangle((width - 244, 58, width - 54, 70), radius=5, fill=(55, 60, 55, 180))
    draw.rounded_rectangle((width - 244, 58, width - 244 + bar_w, 70), radius=5, fill=ORANGE + (235,))

    cx, cy = width // 2, height // 2
    draw.line((cx - 14, cy, cx - 4, cy), fill=(231, 242, 239, 160), width=1)
    draw.line((cx + 4, cy, cx + 14, cy), fill=(231, 242, 239, 160), width=1)
    draw.line((cx, cy - 14, cx, cy - 4), fill=(231, 242, 239, 160), width=1)
    draw.line((cx, cy + 4, cx, cy + 14), fill=(231, 242, 239, 160), width=1)

    map_rect = (24, 122, 164, 262)
    draw.ellipse(map_rect, fill=(10, 15, 16, 140), outline=(235, 244, 239, 70), width=1)
    xs = [p[0] for p in mini_path]
    ys = [p[1] for p in mini_path]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1.0)
    mx0, my0, mx1, my1 = map_rect
    mini = []
    for x, y in mini_path:
        sx = (mx0 + mx1) * 0.5 + (x - (x0 + x1) * 0.5) / span * (mx1 - mx0) * 0.78
        sy = (my0 + my1) * 0.5 - (y - (y0 + y1) * 0.5) / span * (my1 - my0) * 0.78
        mini.append((sx, sy))
    if len(mini) > 1:
        draw.line(mini, fill=CYAN + (205,), width=2)
    px, py = mini[current_idx]
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=ORANGE + (240,), outline=(255, 255, 255, 230), width=1)


def draw_player_silhouette(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    # Generic first-person framing only; not a reconstructed game asset.
    hand = [
        (int(width * 0.62), int(height * 0.78)),
        (int(width * 0.76), int(height * 0.70)),
        (int(width * 0.82), int(height * 0.76)),
        (int(width * 0.70), int(height * 0.91)),
    ]
    draw.polygon(hand, fill=(68, 55, 48, 230), outline=(18, 17, 16, 230))
    weapon = [
        (int(width * 0.64), int(height * 0.70)),
        (int(width * 0.94), int(height * 0.62)),
        (int(width * 0.98), int(height * 0.68)),
        (int(width * 0.72), int(height * 0.79)),
    ]
    draw.polygon(weapon, fill=(47, 45, 41, 235), outline=(16, 17, 16, 240))
    draw.rounded_rectangle((int(width * 0.73), int(height * 0.58), int(width * 0.82), int(height * 0.70)), radius=8, fill=(30, 32, 32, 240), outline=(83, 86, 82, 230), width=2)
    draw.ellipse((int(width * 0.745), int(height * 0.59), int(width * 0.805), int(height * 0.68)), fill=(13, 17, 17, 235), outline=(94, 101, 96, 220), width=2)
    draw.rectangle((int(width * 0.88), int(height * 0.63), int(width * 0.99), int(height * 0.66)), fill=(34, 35, 32, 240))


def draw_pressure_vignette(image: Image.Image, pressure: float) -> None:
    if pressure <= 0.02:
        return
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    alpha = int(105 * min(1.0, pressure))
    for i in range(26):
        a = int(alpha * (1.0 - i / 26) ** 1.5)
        draw.rectangle((i, i, width - i, height - i), outline=(170, 20, 32, a), width=2)
    image.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(radius=3)))


def render_frame(
    rows: list[dict[str, Any]],
    frame_row_idx: int,
    local: dict[str, np.ndarray],
    z_bounds: tuple[float, float],
    args: argparse.Namespace,
    mini_idx_lookup: dict[int, int],
    mini_path: list[tuple[float, float]],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
    fallback_heading: np.ndarray | None,
) -> tuple[Image.Image, np.ndarray]:
    width, height = args.width, args.height
    row = rows[frame_row_idx]
    heading = heading_at(rows, frame_row_idx, fallback_heading)
    forward, right, up = camera_basis(heading, args.pitch_deg)
    eye_z = max(float(row["pos_z_m"]), float(row["terrain_z"])) + args.eye_height
    camera = np.array([float(row["pos_x_m"]), float(row["pos_y_m"]), eye_z], dtype=np.float32)
    focal = width / (2.0 * math.tan(math.radians(args.fov_deg) * 0.5))

    image = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_sky_and_haze(draw, width, height)

    draw_items = []
    for category, tris in local.items():
        if not len(tris):
            continue
        centroids = tris.mean(axis=1)
        rel = centroids.astype(np.float64) - camera.astype(np.float64)
        with np.errstate(all="ignore"):
            depth = rel @ forward.astype(np.float64)
            lateral = np.abs(rel @ right.astype(np.float64))
            vertical = np.abs(rel @ up.astype(np.float64))
            dist_xy = np.linalg.norm(rel[:, :2], axis=1)
        finite = np.isfinite(depth) & np.isfinite(lateral) & np.isfinite(vertical) & np.isfinite(dist_xy)
        cone = (depth > args.near_clip) & (depth < args.max_distance) & (lateral < depth * 1.20 + 40) & (vertical < depth * 0.85 + 80)
        cone &= finite
        for tri, d, xy_dist in zip(tris[cone], depth[cone], dist_xy[cone]):
            draw_items.append((float(d), float(xy_dist), category, tri))
    draw_items.sort(key=lambda item: item[0], reverse=True)

    for depth, xy_dist, category, tri in draw_items:
        projected, valid = project(tri, camera, forward, right, up, focal, width, height, args.near_clip)
        if not valid.all():
            continue
        if np.max(projected[:, 0]) < -80 or np.min(projected[:, 0]) > width + 80:
            continue
        if np.max(projected[:, 1]) < -120 or np.min(projected[:, 1]) > height + 140:
            continue
        poly = [tuple(int(round(v)) for v in point) for point in projected]
        mean_z = float(np.mean(tri[:, 2]))
        if args.clay_mode:
            fill = shade(clay_color(category, mean_z, z_bounds), depth, args.max_distance, 1.0)
        elif category == "terrain":
            fill = shade(terrain_rgb(mean_z, z_bounds), depth, args.max_distance, 1.0)
        elif category == "building":
            pressure_boost = 1.12 if xy_dist < 45 else 0.92
            fill = shade((132, 96, 54), depth, args.max_distance, pressure_boost)
        else:
            fill = shade(BASE_COLORS[category], depth, args.max_distance, 1.0)
        draw.polygon(poly, fill=fill)
        if args.wireframe or (category in {"building", "road", "runway"} and depth < 85):
            if args.clay_mode:
                edge = (25, 40, 42, 145 if depth < 100 else 70)
            else:
                edge = (15, 18, 17, 135) if category == "building" else (226, 230, 216, 85)
            draw.line(poly + [poly[0]], fill=edge, width=1)

    # Draw the recent/future breadcrumb as a subtle path on the terrain.
    lo = max(0, frame_row_idx - 10)
    hi = min(len(rows), frame_row_idx + 18)
    path = np.asarray(
        [
            (float(item["pos_x_m"]), float(item["pos_y_m"]), max(float(item["pos_z_m"]), float(item["terrain_z"])) + 0.35)
            for item in rows[lo:hi]
        ],
        dtype=np.float32,
    )
    if len(path) > 1:
        projected, valid = project(path, camera, forward, right, up, focal, width, height, args.near_clip)
        pts = [tuple(int(round(v)) for v in point) for point, ok in zip(projected, valid) if ok]
        if len(pts) > 1:
            draw.line(pts, fill=(7, 14, 15, 180), width=7)
            draw.line(pts, fill=CYAN + (220,), width=4)

    pressure = max(0.0, min(1.0, 1.0 - float(row["nearest_building_xy_m"]) / 25.0))
    draw_pressure_vignette(image, pressure)
    if not args.no_player_silhouette:
        draw_player_silhouette(draw, width, height)
    draw_hud(draw, row, width, height, mini_path, mini_idx_lookup[frame_row_idx], font, small)
    return image.convert("RGB"), heading


def make_contact_sheet(frames: list[Image.Image], output: Path, labels: list[str], columns: int = 3) -> None:
    if not frames:
        return
    thumb_w = 520
    thumb_h = int(frames[0].height * thumb_w / frames[0].width)
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 34)), (8, 13, 14))
    draw = ImageDraw.Draw(sheet)
    font = load_font(15)
    for idx, frame in enumerate(frames):
        x = (idx % columns) * thumb_w
        y = (idx // columns) * (thumb_h + 34)
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y + 28))
        draw.text((x + 10, y + 7), labels[idx], fill=TEXT, font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def render(args: argparse.Namespace) -> dict[str, Any]:
    payload = pickle.load(args.cache.open("rb"))
    raw = {category: array_to_tris(payload["triangles"][category]) for category in CATEGORIES}
    rows = fetch_agent_rows(args)
    if len(rows) < 2:
        raise SystemExit("Need at least two trajectory rows for POV animation.")
    local = localize_tris(raw, rows, args.mesh_margin, args)
    diagnostics = attach_diagnostics(rows, local)
    terrain_z = diagnostics["terrain_z"]
    z_bounds = (
        float(np.nanpercentile(terrain_z, 1)) - 5.0,
        float(np.nanpercentile(terrain_z, 99)) + 15.0,
    )

    indices = frame_indices(len(rows), args.frame_count)
    mini_path = [(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in rows]
    mini_lookup = {idx: min(range(len(rows)), key=lambda ridx: abs(ridx - idx)) for idx in indices}
    # Convert lookup value to path index in the full minimap path.
    mini_lookup = {idx: idx for idx in indices}
    font = load_font(22, bold=True)
    small = load_font(14)

    frames: list[Image.Image] = []
    labels: list[str] = []
    heading = None
    for idx in indices:
        frame, heading = render_frame(rows, idx, local, z_bounds, args, mini_lookup, mini_path, font, small, heading)
        frames.append(frame)
        labels.append(f"t={float(rows[idx]['sample_t']):.1f}s  z={float(rows[idx]['pos_z_m']):.1f}m")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / max(1, args.fps)))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    if args.contact_sheet:
        sheet_indices = frame_indices(len(frames), min(args.contact_sheet_frames, len(frames)))
        make_contact_sheet(
            [frames[i] for i in sheet_indices],
            args.contact_sheet,
            [labels[i] for i in sheet_indices],
            columns=args.contact_sheet_columns,
        )

    finite_height = diagnostics["height_above_terrain"][np.isfinite(diagnostics["height_above_terrain"])]
    finite_building = diagnostics["building_dist"][np.isfinite(diagnostics["building_dist"])]
    manifest = {
        "output": str(args.output.relative_to(ROOT)),
        "contact_sheet": str(args.contact_sheet.relative_to(ROOT)) if args.contact_sheet else None,
        "cache": str(args.cache.relative_to(ROOT)),
        "match_id": args.match_id,
        "agent_id": args.agent_id,
        "rows": len(rows),
        "frames": len(frames),
        "time_range_s": [float(rows[0]["sample_t"]), float(rows[-1]["sample_t"])],
        "display_triangles": {category: int(len(local[category])) for category in CATEGORIES},
        "diagnostics": {
            "height_above_terrain_median": float(np.nanmedian(finite_height)) if len(finite_height) else None,
            "height_above_terrain_p90": float(np.nanpercentile(finite_height, 90)) if len(finite_height) else None,
            "building_distance_median": float(np.nanmedian(finite_building)) if len(finite_building) else None,
            "near_building_ratio_25m": float(np.mean(diagnostics["building_dist"] <= 25)),
        },
        "notes": [
            "This is a first-person proxy, not a Warzone/game-engine render.",
            "Camera heading is estimated from the trajectory direction.",
            "The lower-right first-person silhouette is a generic visual frame, not reconstructed asset data.",
        ],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--source-artifact-id", default="caldera_minimal_layers_breadcrumbs_usd")
    parser.add_argument("--sample-t-min", type=float, default=1150.0)
    parser.add_argument("--sample-t-max", type=float)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov-deg", type=float, default=78.0)
    parser.add_argument("--pitch-deg", type=float, default=-4.0)
    parser.add_argument("--eye-height", type=float, default=1.65)
    parser.add_argument("--near-clip", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=180.0)
    parser.add_argument("--mesh-margin", type=float, default=220.0)
    parser.add_argument("--clay-mode", action="store_true")
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument("--no-player-silhouette", action="store_true")
    for category, default in BUDGETS.items():
        parser.add_argument(f"--{category.replace('_', '-')}-triangles", type=int, default=default)
    parser.add_argument("--contact-sheet-frames", type=int, default=6)
    parser.add_argument("--contact-sheet-columns", type=int, default=3)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    if args.contact_sheet:
        args.contact_sheet = args.contact_sheet.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = render(args)
    print(f"animation_written={args.output}")
    if args.contact_sheet:
        print(f"contact_sheet_written={args.contact_sheet}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
