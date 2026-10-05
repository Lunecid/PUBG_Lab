"""Render clean player-POV proxy panels for Caldera RQ3.

The raw mesh POV renderer is useful for debugging reconstruction, but it
shows individual triangle artifacts.  This script turns the same Caldera mesh
and observed trajectory samples into a cleaner manuscript visual:

- terrain is rendered as a smoothed local surface;
- roads and runways are drawn as semantic surface bands;
- building mesh points are aggregated into solid massing blocks.

The result is still a proxy view from the observed player position and
estimated heading, not a textured Warzone/game-engine screenshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "terrain",
    "road",
    "runway",
    "building",
]

SKY_TOP = np.array((126, 176, 199), dtype=np.float32)
SKY_HORIZON = np.array((211, 220, 206), dtype=np.float32)
GROUND_FAR = np.array((64, 82, 63), dtype=np.float32)


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    tris = np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))
    sane = np.isfinite(tris).all(axis=(1, 2)) & (np.abs(tris) < 10_000).all(axis=(1, 2))
    return tris[sane]


def load_mesh(cache: Path) -> dict[str, np.ndarray]:
    payload = pickle.load(cache.open("rb"))
    return {category: array_to_tris(payload["triangles"][category]) for category in CATEGORIES}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def closest_rows(rows: list[dict[str, Any]], targets: list[float]) -> list[dict[str, Any]]:
    times = np.asarray([as_float(row, "t") for row in rows], dtype=np.float64)
    return [rows[int(np.argmin(np.abs(times - target)))] for target in targets]


def heading_vector(row: dict[str, Any]) -> np.ndarray:
    heading = math.radians(as_float(row, "heading_deg", 0.0))
    vec = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return vec / norm


def camera_basis(heading_xy: np.ndarray, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitch = math.radians(pitch_deg)
    xy = heading_xy / max(1e-6, float(np.linalg.norm(heading_xy)))
    forward = np.array([math.cos(pitch) * xy[0], math.cos(pitch) * xy[1], math.sin(pitch)], dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / max(1e-6, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up = up / max(1e-6, float(np.linalg.norm(up)))
    return forward, right, up


def project(
    points: np.ndarray,
    camera: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    focal: float,
    width: int,
    height: int,
    near: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = points.astype(np.float32) - camera
    z = rel @ forward
    x = rel @ right
    y = rel @ up
    valid = z > near
    sx = width * 0.5 + focal * x / np.maximum(z, near)
    sy = height * 0.54 - focal * y / np.maximum(z, near)
    return np.column_stack([sx, sy]).astype(np.float32), valid, z


def draw_sky(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    horizon = int(height * 0.55)
    for y in range(height):
        if y < horizon:
            t = y / max(1, horizon)
            rgb = SKY_TOP * (1 - t) + SKY_HORIZON * t
        else:
            t = (y - horizon) / max(1, height - horizon)
            rgb = SKY_HORIZON * (1 - t) + GROUND_FAR * t
        draw.line((0, y, width, y), fill=tuple(int(v) for v in rgb))


def fog_color(color: tuple[int, int, int], depth: float, max_depth: float, shade: float = 1.0) -> tuple[int, int, int]:
    fog = max(0.0, min(1.0, depth / max_depth))
    base = np.array(color, dtype=np.float32) * shade
    rgb = base * (1.0 - 0.45 * fog) + SKY_HORIZON * (0.45 * fog)
    return tuple(int(v) for v in np.clip(rgb, 0, 255))


class TerrainSampler:
    def __init__(self, terrain_tris: np.ndarray) -> None:
        centroids = terrain_tris.mean(axis=1).astype(np.float32)
        self.xy = centroids[:, :2]
        self.z = centroids[:, 2]
        self.tree = cKDTree(self.xy)

    def sample(self, xy: np.ndarray, k: int = 10) -> np.ndarray:
        k = max(1, min(k, len(self.xy)))
        dist, idx = self.tree.query(xy, k=k)
        if k == 1:
            return self.z[idx]
        weights = 1.0 / np.maximum(dist, 0.75) ** 2
        far = dist > 22.0
        weights[far] *= 0.15
        weights_sum = np.maximum(weights.sum(axis=1), 1e-6)
        return (self.z[idx] * weights).sum(axis=1) / weights_sum


def make_local_grid(
    row: dict[str, Any],
    terrain: TerrainSampler,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heading = heading_vector(row)
    forward, right, up = camera_basis(heading, args.pitch_deg)
    terrain_z = as_float(row, "terrain_z_m", as_float(row, "z"))
    eye_z = max(as_float(row, "z"), terrain_z) + args.eye_height
    camera = np.array([as_float(row, "x"), as_float(row, "y"), eye_z], dtype=np.float32)

    u_vals = np.linspace(args.near_clip + 0.5, args.max_distance, args.grid_forward, dtype=np.float32)
    max_side = args.max_distance * math.tan(math.radians(args.fov_deg) * 0.58)
    v_vals = np.linspace(-max_side, max_side, args.grid_lateral, dtype=np.float32)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    forward_xy = heading.astype(np.float32)
    right_xy = np.array([right[0], right[1]], dtype=np.float32)
    xy = camera[:2] + uu[..., None] * forward_xy + vv[..., None] * right_xy
    z = terrain.sample(xy.reshape(-1, 2)).reshape(uu.shape)
    points = np.dstack([xy, z + args.surface_lift]).astype(np.float32)
    return points, uu, vv, camera, forward, right, up


def draw_surface(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    uu: np.ndarray,
    semantic: np.ndarray | None,
    camera: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    args: argparse.Namespace,
) -> None:
    width, height = args.width, args.height
    focal = width / (2.0 * math.tan(math.radians(args.fov_deg) * 0.5))
    z_grid = points[:, :, 2]
    z_lo, z_hi = np.nanpercentile(z_grid, [5, 95])

    cells = []
    for i in range(points.shape[0] - 1):
        for j in range(points.shape[1] - 1):
            quad = np.asarray([points[i, j], points[i + 1, j], points[i + 1, j + 1], points[i, j + 1]], dtype=np.float32)
            projected, valid, depth = project(quad, camera, forward, right, up, focal, width, height, args.near_clip)
            if not valid.all():
                continue
            if np.max(projected[:, 0]) < -80 or np.min(projected[:, 0]) > width + 80:
                continue
            if np.max(projected[:, 1]) < -80 or np.min(projected[:, 1]) > height + 120:
                continue
            mean_depth = float(np.mean(depth))
            if semantic is None:
                dz_u = float(z_grid[min(i + 1, z_grid.shape[0] - 1), j] - z_grid[i, j])
                dz_v = float(z_grid[i, min(j + 1, z_grid.shape[1] - 1)] - z_grid[i, j])
                slope = min(1.0, math.sqrt(dz_u * dz_u + dz_v * dz_v) / 4.0)
                elev = max(0.0, min(1.0, (float(np.mean(quad[:, 2])) - z_lo) / max(1e-6, z_hi - z_lo)))
                low = np.array((51, 81, 55), dtype=np.float32)
                high = np.array((146, 161, 105), dtype=np.float32)
                rough = np.array((118, 127, 90), dtype=np.float32)
                rgb = low * (1 - elev) + high * elev
                rgb = rgb * (1 - 0.45 * slope) + rough * (0.45 * slope)
                color = fog_color(tuple(int(v) for v in rgb), mean_depth, args.max_distance)
            else:
                sem = int(round(float(np.mean(semantic[i : i + 2, j : j + 2]))))
                if sem == 2:
                    color = fog_color((86, 88, 84), mean_depth, args.max_distance, 1.0)
                elif sem == 1:
                    color = fog_color((101, 105, 98), mean_depth, args.max_distance, 1.0)
                else:
                    continue
            cells.append((mean_depth, projected, color))

    cells.sort(key=lambda item: item[0], reverse=True)
    for _, projected, color in cells:
        poly = [tuple(int(round(v)) for v in point) for point in projected]
        draw.polygon(poly, fill=color)


def semantic_grid(
    points: np.ndarray,
    road_centroids: np.ndarray,
    runway_centroids: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    xy = points[:, :, :2].reshape(-1, 2)
    semantic = np.zeros(len(xy), dtype=np.int8)
    if len(road_centroids):
        tree = cKDTree(road_centroids[:, :2])
        dist, _ = tree.query(xy, k=1)
        semantic[dist <= args.road_width_m] = 1
    if len(runway_centroids):
        tree = cKDTree(runway_centroids[:, :2])
        dist, _ = tree.query(xy, k=1)
        semantic[dist <= args.runway_width_m] = 2
    return semantic.reshape(points.shape[:2])


def local_building_blocks(
    row: dict[str, Any],
    building_centroids: np.ndarray,
    terrain: TerrainSampler,
    camera: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not len(building_centroids):
        return []
    building_centroids = building_centroids[np.isfinite(building_centroids).all(axis=1)]
    if not len(building_centroids):
        return []
    heading = heading_vector(row)
    right_xy = np.array([right[0], right[1]], dtype=np.float32)
    rel_xy = building_centroids[:, :2].astype(np.float64) - camera[:2].astype(np.float64)
    u = rel_xy @ heading.astype(np.float64)
    v = rel_xy @ right_xy.astype(np.float64)
    max_side = np.maximum(26.0, u * math.tan(math.radians(args.fov_deg) * 0.56) + 30.0)
    mask = (u > -8.0) & (u < args.building_draw_distance_m) & (np.abs(v) <= max_side)
    mask &= np.isfinite(u) & np.isfinite(v)
    pts = building_centroids[mask]
    u = u[mask]
    v = v[mask]
    if not len(pts):
        return []

    cell = args.building_cell_m
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (uu, vv) in enumerate(zip(u, v)):
        buckets[(int(math.floor(uu / cell)), int(math.floor(vv / cell)))] .append(idx)

    blocks: list[dict[str, Any]] = []
    for (bu, bv), idxs in buckets.items():
        if len(idxs) < args.building_min_points:
            continue
        u0 = max(args.near_clip + 1.0, bu * cell)
        u1 = (bu + 1) * cell
        v0 = bv * cell
        v1 = (bv + 1) * cell
        if u1 <= u0 + 1.0:
            continue
        center_xy = camera[:2] + ((u0 + u1) * 0.5) * heading + ((v0 + v1) * 0.5) * right_xy
        base_z = float(terrain.sample(center_xy.reshape(1, 2))[0])
        z_values = pts[idxs, 2]
        top_z = float(np.nanpercentile(z_values, 92)) + args.building_top_lift
        top_z = max(top_z, base_z + args.building_min_height_m)
        top_z = min(top_z, base_z + args.building_max_height_m)
        blocks.append(
            {
                "u0": u0,
                "u1": u1,
                "v0": v0,
                "v1": v1,
                "base_z": base_z,
                "top_z": top_z,
                "count": len(idxs),
            }
        )
    return blocks


def block_corners(block: dict[str, Any], camera: np.ndarray, heading: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    corners = []
    for z in (block["base_z"], block["top_z"]):
        for u, v in (
            (block["u0"], block["v0"]),
            (block["u1"], block["v0"]),
            (block["u1"], block["v1"]),
            (block["u0"], block["v1"]),
        ):
            xy = camera[:2] + u * heading + v * right_xy
            corners.append([xy[0], xy[1], z])
    return np.asarray(corners, dtype=np.float32)


def draw_blocks(
    draw: ImageDraw.ImageDraw,
    blocks: list[dict[str, Any]],
    row: dict[str, Any],
    camera: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if not blocks:
        return
    width, height = args.width, args.height
    focal = width / (2.0 * math.tan(math.radians(args.fov_deg) * 0.5))
    heading = heading_vector(row)
    right_xy = np.array([right[0], right[1]], dtype=np.float32)
    faces = []
    face_defs = [
        ([0, 1, 2, 3], (112, 100, 82), 0.86),  # base footprint, usually hidden
        ([4, 5, 6, 7], (155, 142, 114), 1.04),  # roof
        ([0, 1, 5, 4], (137, 117, 88), 0.92),
        ([1, 2, 6, 5], (126, 108, 82), 0.82),
        ([2, 3, 7, 6], (119, 101, 78), 0.78),
        ([3, 0, 4, 7], (145, 124, 92), 0.95),
    ]
    for block in blocks:
        corners = block_corners(block, camera, heading, right_xy)
        for idxs, base_color, shade in face_defs[1:]:
            face = corners[idxs]
            projected, valid, depth = project(face, camera, forward, right, up, focal, width, height, args.near_clip)
            if not valid.all():
                continue
            if np.max(projected[:, 0]) < -120 or np.min(projected[:, 0]) > width + 120:
                continue
            if np.max(projected[:, 1]) < -120 or np.min(projected[:, 1]) > height + 160:
                continue
            faces.append((float(np.mean(depth)), projected, base_color, shade))
    faces.sort(key=lambda item: item[0], reverse=True)
    for depth, projected, color, shade in faces:
        fill = fog_color(color, depth, args.max_distance, shade)
        outline = fog_color((54, 56, 50), depth, args.max_distance, 0.95)
        poly = [tuple(int(round(v)) for v in point) for point in projected]
        draw.polygon(poly, fill=fill)
        draw.line(poly + [poly[0]], fill=outline, width=1)


def render_panel(
    scene_name: str,
    row: dict[str, Any],
    mesh: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    terrain: TerrainSampler,
    args: argparse.Namespace,
) -> Image.Image:
    image = Image.new("RGB", (args.width, args.height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw_sky(draw, args.width, args.height)

    points, uu, _vv, camera, forward, right, up = make_local_grid(row, terrain, args)
    draw_surface(draw, points, uu, None, camera, forward, right, up, args)

    if args.always_show_roads or scene_name == "route_corridor":
        semantic = semantic_grid(points, centroids["road"], centroids["runway"], args)
        road_points = points.copy()
        road_points[:, :, 2] += args.road_lift
        draw_surface(draw, road_points, uu, semantic, camera, forward, right, up, args)

    if args.always_show_buildings or scene_name == "building_affordance":
        blocks = local_building_blocks(row, centroids["building"], terrain, camera, forward, right, args)
        draw_blocks(draw, blocks, row, camera, forward, right, up, args)
    return image


def make_montage(frames: list[Image.Image], output: Path, columns: int = 2) -> None:
    if not frames:
        return
    rows = math.ceil(len(frames) / columns)
    width = frames[0].width
    height = frames[0].height
    sheet = Image.new("RGB", (columns * width, rows * height), (8, 13, 14))
    for idx, frame in enumerate(frames):
        x = (idx % columns) * width
        y = (idx // columns) * height
        sheet.paste(frame, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def render(args: argparse.Namespace) -> dict[str, Any]:
    mesh = load_mesh(args.cache)
    centroids = {category: mesh[category].mean(axis=1).astype(np.float32) if len(mesh[category]) else np.empty((0, 3), dtype=np.float32) for category in CATEGORIES}
    terrain = TerrainSampler(mesh["terrain"])

    rows = load_rows(args.features_csv)
    target_rows = closest_rows(rows, args.frame_times)
    names = args.frame_names
    if len(names) != len(target_rows):
        raise SystemExit("--frame-names must have the same length as --frame-times")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    stills: list[str] = []
    for idx, (name, row) in enumerate(zip(names, target_rows), start=1):
        frame = render_panel(name, row, mesh, centroids, terrain, args)
        path = args.output_dir / f"{idx:02d}_{name}_clean_pov_t{as_float(row, 't'):.1f}.png"
        frame.save(path)
        frames.append(frame)
        stills.append(str(path.relative_to(ROOT)))

    make_montage(frames, args.montage, args.columns)
    manifest = {
        "montage": str(args.montage.relative_to(ROOT)),
        "stills": stills,
        "cache": str(args.cache.relative_to(ROOT)),
        "features_csv": str(args.features_csv.relative_to(ROOT)),
        "frame_times_requested_s": args.frame_times,
        "frame_times_used_s": [as_float(row, "t") for row in target_rows],
        "frame_names": names,
        "render_type": "clean first-person proxy with smoothed terrain, semantic road/runway bands, and building massing blocks",
        "boundary": "Not a textured game-engine screenshot; not supervised prediction evidence.",
    }
    args.montage.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "reports/caldera_3d/single_agent/match_31_player_25_local_raw_mesh.pkl")
    parser.add_argument("--features-csv", type=Path, default=ROOT / "reports/caldera_3d/single_agent/world_build/match_31_player_25_terrain_experience_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/caldera_3d/rq3_clean_pov_proxy/stills")
    parser.add_argument("--montage", type=Path, default=ROOT / "reports/caldera_3d/rq3_clean_pov_proxy/caldera_rq3_clean_pov_proxy_montage.png")
    parser.add_argument("--frame-times", type=parse_csv_floats, default=parse_csv_floats("1187.232,1199.184,1235.040,1625.472"))
    parser.add_argument("--frame-names", type=parse_csv_strings, default=parse_csv_strings("rough_terrain,relative_elevation,building_affordance,route_corridor"))
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--fov-deg", type=float, default=78.0)
    parser.add_argument("--pitch-deg", type=float, default=-4.0)
    parser.add_argument("--eye-height", type=float, default=1.65)
    parser.add_argument("--near-clip", type=float, default=1.2)
    parser.add_argument("--max-distance", type=float, default=145.0)
    parser.add_argument("--grid-forward", type=int, default=86)
    parser.add_argument("--grid-lateral", type=int, default=100)
    parser.add_argument("--surface-lift", type=float, default=0.0)
    parser.add_argument("--road-lift", type=float, default=0.08)
    parser.add_argument("--road-width-m", type=float, default=8.5)
    parser.add_argument("--runway-width-m", type=float, default=15.0)
    parser.add_argument("--building-cell-m", type=float, default=10.0)
    parser.add_argument("--building-min-points", type=int, default=10)
    parser.add_argument("--building-draw-distance-m", type=float, default=95.0)
    parser.add_argument("--building-min-height-m", type=float, default=5.5)
    parser.add_argument("--building-max-height-m", type=float, default=36.0)
    parser.add_argument("--building-top-lift", type=float, default=2.5)
    parser.add_argument("--always-show-buildings", action="store_true")
    parser.add_argument("--always-show-roads", action="store_true")
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.features_csv = args.features_csv.resolve()
    args.output_dir = args.output_dir.resolve()
    args.montage = args.montage.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = render(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
