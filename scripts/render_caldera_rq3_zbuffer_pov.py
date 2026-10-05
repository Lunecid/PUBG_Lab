"""Render Caldera player-POV panels with a small CPU z-buffer rasterizer.

This keeps the render data-grounded in the reconstructed Caldera mesh while
avoiding the broken triangle-wire/debug look of the earlier painter renderer.
It intentionally draws no HUD, labels, path, crosshair, or explanatory text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

CATEGORY_ORDER = ["terrain", "road", "runway", "building", "prop"]
DEFAULT_BUDGETS = {
    "terrain": 160_000,
    "road": 55_000,
    "runway": 25_000,
    "building": 260_000,
    "prop": 20_000,
}


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    tris = np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))
    sane = np.isfinite(tris).all(axis=(1, 2)) & (np.abs(tris) < 10_000).all(axis=(1, 2))
    return tris[sane]


def load_mesh(cache: Path) -> dict[str, np.ndarray]:
    payload = pickle.load(cache.open("rb"))
    return {category: array_to_tris(payload["triangles"].get(category, array("f"))) for category in CATEGORY_ORDER}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def draw_sky(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    top = np.array((128, 179, 200), dtype=np.float32)
    horizon = np.array((211, 221, 211), dtype=np.float32)
    ground = np.array((56, 72, 58), dtype=np.float32)
    horizon_y = int(height * 0.55)
    for y in range(height):
        if y < horizon_y:
            t = y / max(1, horizon_y)
            rgb = top * (1.0 - t) + horizon * t
        else:
            t = (y - horizon_y) / max(1, height - horizon_y)
            rgb = horizon * (1.0 - t) + ground * t
        image[y, :, :] = np.clip(rgb, 0, 255).astype(np.uint8)
    return image


def project(points: np.ndarray, camera: np.ndarray, forward: np.ndarray, right: np.ndarray, up: np.ndarray, focal: float, width: int, height: int, near: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = points.astype(np.float32) - camera
    z = rel @ forward
    x = rel @ right
    y = rel @ up
    valid = z > near
    sx = width * 0.5 + focal * x / np.maximum(z, near)
    sy = height * 0.54 - focal * y / np.maximum(z, near)
    return np.column_stack([sx, sy]).astype(np.float32), valid, z.astype(np.float32)


def fog_color(base: np.ndarray, depth: np.ndarray, max_depth: float) -> np.ndarray:
    fog = np.clip(depth / max_depth, 0.0, 1.0)[:, None]
    sky = np.array((205, 218, 211), dtype=np.float32)
    rgb = base.astype(np.float32) * (1.0 - 0.50 * fog) + sky * (0.50 * fog)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def category_base_colors(category: str, mean_z: np.ndarray, z_bounds: tuple[float, float]) -> np.ndarray:
    n = len(mean_z)
    if category == "terrain":
        lo, hi = z_bounds
        t = np.clip((mean_z - lo) / max(1e-6, hi - lo), 0.0, 1.0)[:, None]
        low = np.array((48, 78, 52), dtype=np.float32)
        high = np.array((142, 158, 103), dtype=np.float32)
        return np.clip(low * (1.0 - t) + high * t, 0, 255).astype(np.uint8)
    if category == "road":
        return np.tile(np.array((98, 102, 96), dtype=np.uint8), (n, 1))
    if category == "runway":
        return np.tile(np.array((70, 76, 76), dtype=np.uint8), (n, 1))
    if category == "building":
        return np.tile(np.array((140, 120, 88), dtype=np.uint8), (n, 1))
    return np.tile(np.array((118, 109, 91), dtype=np.uint8), (n, 1))


def select_triangles(
    tris: np.ndarray,
    category: str,
    camera: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if not len(tris):
        return tris
    centroids = tris.mean(axis=1)
    rel = centroids.astype(np.float32) - camera
    depth = rel @ forward
    lateral = np.abs(rel @ right)
    vertical = np.abs(rel @ up)
    side = depth * math.tan(math.radians(args.fov_deg) * 0.62) + args.frustum_pad_m
    mask = (
        (depth > args.near_clip)
        & (depth < args.max_distance)
        & (lateral < side)
        & (vertical < depth * 0.95 + args.vertical_pad_m)
        & np.isfinite(depth)
        & np.isfinite(lateral)
        & np.isfinite(vertical)
    )
    selected = tris[mask]
    max_count = getattr(args, f"{category}_triangles")
    if len(selected) > max_count:
        idx = np.linspace(0, len(selected) - 1, max_count, dtype=np.int64)
        selected = selected[idx]
    return selected


def prepare_projected_triangles(
    mesh: dict[str, np.ndarray],
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heading = heading_vector(row)
    forward, right, up = camera_basis(heading, args.pitch_deg)
    terrain_z = as_float(row, "terrain_z_m", as_float(row, "z"))
    eye_z = max(as_float(row, "z"), terrain_z) + args.eye_height
    camera = np.array([as_float(row, "x"), as_float(row, "y"), eye_z], dtype=np.float32)
    focal = args.width / (2.0 * math.tan(math.radians(args.fov_deg) * 0.5))

    terrain_all = mesh["terrain"].mean(axis=1)[:, 2] if len(mesh["terrain"]) else np.asarray([0.0])
    z_bounds = (float(np.nanpercentile(terrain_all, 5)), float(np.nanpercentile(terrain_all, 98)))

    screen_tris: list[np.ndarray] = []
    depth_tris: list[np.ndarray] = []
    color_tris: list[np.ndarray] = []
    for category in CATEGORY_ORDER:
        tris = select_triangles(mesh[category], category, camera, forward, right, up, args)
        if not len(tris):
            continue
        projected, valid, depth = project(tris.reshape(-1, 3), camera, forward, right, up, focal, args.width, args.height, args.near_clip)
        projected = projected.reshape((-1, 3, 2))
        depth = depth.reshape((-1, 3))
        valid = valid.reshape((-1, 3)).all(axis=1)
        on_screen = (
            (np.max(projected[:, :, 0], axis=1) >= -args.screen_pad_px)
            & (np.min(projected[:, :, 0], axis=1) <= args.width + args.screen_pad_px)
            & (np.max(projected[:, :, 1], axis=1) >= -args.screen_pad_px)
            & (np.min(projected[:, :, 1], axis=1) <= args.height + args.screen_pad_px)
        )
        keep = valid & on_screen
        if not np.any(keep):
            continue
        projected = projected[keep]
        depth = depth[keep]
        kept_tris = tris[keep]
        mean_z = kept_tris.mean(axis=(1, 2)) if category == "terrain" else kept_tris.mean(axis=1)[:, 2]
        base = category_base_colors(category, mean_z.astype(np.float32), z_bounds)
        mean_depth = depth.mean(axis=1)
        colors = fog_color(base, mean_depth, args.max_distance)
        screen_tris.append(projected.astype(np.float32))
        depth_tris.append(depth.astype(np.float32))
        color_tris.append(colors.astype(np.uint8))
    if not screen_tris:
        return np.empty((0, 3, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(screen_tris, axis=0), np.concatenate(depth_tris, axis=0), np.concatenate(color_tris, axis=0)


@njit(cache=True)
def rasterize(screen: np.ndarray, depth: np.ndarray, colors: np.ndarray, image: np.ndarray, zbuf: np.ndarray) -> None:
    h, w = zbuf.shape
    for i in range(screen.shape[0]):
        x0 = screen[i, 0, 0]
        y0 = screen[i, 0, 1]
        x1 = screen[i, 1, 0]
        y1 = screen[i, 1, 1]
        x2 = screen[i, 2, 0]
        y2 = screen[i, 2, 1]
        z0 = depth[i, 0]
        z1 = depth[i, 1]
        z2 = depth[i, 2]
        minx = max(0, int(math.floor(min(x0, x1, x2))))
        maxx = min(w - 1, int(math.ceil(max(x0, x1, x2))))
        miny = max(0, int(math.floor(min(y0, y1, y2))))
        maxy = min(h - 1, int(math.ceil(max(y0, y1, y2))))
        if maxx < minx or maxy < miny:
            continue
        if (maxx - minx) * (maxy - miny) > w * h * 2:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-4:
            continue
        for py in range(miny, maxy + 1):
            yy = py + 0.5
            for px in range(minx, maxx + 1):
                xx = px + 0.5
                a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
                b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
                c = 1.0 - a - b
                if a >= -0.0001 and b >= -0.0001 and c >= -0.0001:
                    zz = a * z0 + b * z1 + c * z2
                    if zz > 0.0 and zz < zbuf[py, px]:
                        zbuf[py, px] = zz
                        image[py, px, 0] = colors[i, 0]
                        image[py, px, 1] = colors[i, 1]
                        image[py, px, 2] = colors[i, 2]


def render_panel(mesh: dict[str, np.ndarray], row: dict[str, Any], args: argparse.Namespace) -> Image.Image:
    image = draw_sky(args.width, args.height)
    zbuf = np.full((args.height, args.width), np.inf, dtype=np.float32)
    screen, depth, colors = prepare_projected_triangles(mesh, row, args)
    if len(screen):
        rasterize(screen, depth, colors, image, zbuf)
    return Image.fromarray(image, mode="RGB")


def make_montage(frames: list[Image.Image], output: Path, columns: int) -> None:
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
    rows = load_rows(args.features_csv)
    target_rows = closest_rows(rows, args.frame_times)
    if len(args.frame_names) != len(target_rows):
        raise SystemExit("--frame-names must have the same length as --frame-times")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    stills = []
    for idx, (name, row) in enumerate(zip(args.frame_names, target_rows), start=1):
        frame = render_panel(mesh, row, args)
        path = args.output_dir / f"{idx:02d}_{name}_zbuffer_pov_t{as_float(row, 't'):.1f}.png"
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
        "frame_names": args.frame_names,
        "render_type": "CPU z-buffer player-POV render from reconstructed Caldera mesh",
        "boundary": "Not a textured game-engine screenshot; no HUD/path/text overlays; not supervised prediction evidence.",
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/caldera_3d/rq3_zbuffer_pov/stills")
    parser.add_argument("--montage", type=Path, default=ROOT / "reports/caldera_3d/rq3_zbuffer_pov/caldera_rq3_zbuffer_pov_montage.png")
    parser.add_argument("--frame-times", type=parse_csv_floats, default=parse_csv_floats("1187.232,1199.184,1235.040,1625.472"))
    parser.add_argument("--frame-names", type=parse_csv_strings, default=parse_csv_strings("rough_terrain,relative_elevation,building_affordance,route_corridor"))
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=506)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--fov-deg", type=float, default=78.0)
    parser.add_argument("--pitch-deg", type=float, default=-4.0)
    parser.add_argument("--eye-height", type=float, default=1.65)
    parser.add_argument("--near-clip", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=145.0)
    parser.add_argument("--frustum-pad-m", type=float, default=24.0)
    parser.add_argument("--vertical-pad-m", type=float, default=50.0)
    parser.add_argument("--screen-pad-px", type=float, default=160.0)
    for category, default in DEFAULT_BUDGETS.items():
        parser.add_argument(f"--{category}-triangles", type=int, default=default)
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
