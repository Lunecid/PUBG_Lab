"""Render one Caldera agent trajectory over a semantic top-down mesh cache.

This is a light-weight snapshot renderer that avoids OpenCV.  It expects a
semantic triangle cache produced by ``render_caldera_semantic_basemap.py`` and
draws one exact DB agent over the reconstructed Caldera spatial layer.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import os
import pickle
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pubg_lab.data.stdb.connection.conn import connect


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
    "terrain": 110_000,
    "coast_beach": 24_000,
    "water": 16_000,
    "vegetation": 30_000,
    "road": 42_000,
    "runway": 18_000,
    "building": 80_000,
    "prop": 8_000,
}

COLORS = {
    "coast_beach": (194, 169, 105, 220),
    "water": (31, 105, 124, 225),
    "vegetation": (27, 92, 42, 155),
    "road": (188, 191, 181, 220),
    "runway": (72, 79, 77, 230),
    "building": (34, 39, 39, 200),
    "prop": (104, 101, 91, 150),
}


def parse_rgb(value: str) -> tuple[int, int, int]:
    if not value.startswith("#") or len(value) != 7:
        raise argparse.ArgumentTypeError("color must be #RRGGBB")
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def parse_rgba(value: str) -> tuple[int, int, int, int]:
    r, g, b = parse_rgb(value)
    return (r, g, b, 245)


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


def deterministic_sample(tris: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0 or len(tris) <= max_count:
        return tris
    idx = np.linspace(0, len(tris) - 1, max_count, dtype=np.int64)
    return tris[idx]


def compute_bounds(
    triangles: dict[str, np.ndarray], trim_percentile: float, square: bool
) -> dict[str, tuple[float, float]]:
    parts = [tris.reshape((-1, 3)) for tris in triangles.values() if len(tris)]
    if not parts:
        return {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.0)}
    pts = np.concatenate(parts, axis=0)
    trim = max(0.0, min(10.0, float(trim_percentile)))
    low, high = trim, 100.0 - trim
    x0, x1 = np.percentile(pts[:, 0], [low, high])
    y0, y1 = np.percentile(pts[:, 1], [low, high])
    z0, z1 = np.percentile(pts[:, 2], [low, high])
    if square:
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        half = max(x1 - x0, y1 - y0) * 0.5
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
    return {"x": (float(x0), float(x1)), "y": (float(y0), float(y1)), "z": (float(z0), float(z1))}


def project_xy(
    points: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    width: int,
    height: int,
    padding: int,
) -> np.ndarray:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    sx = (width - 2 * padding) / max(1e-6, x1 - x0)
    sy = (height - 2 * padding) / max(1e-6, y1 - y0)
    scale = min(sx, sy)
    x_margin = (width - scale * (x1 - x0)) * 0.5
    y_margin = (height - scale * (y1 - y0)) * 0.5
    out = np.empty((len(points), 2), dtype=np.int32)
    out[:, 0] = np.round(x_margin + (points[:, 0] - x0) * scale).astype(np.int32)
    out[:, 1] = np.round(height - (y_margin + (points[:, 1] - y0) * scale)).astype(np.int32)
    return out


def terrain_color(z: float, z_bounds: tuple[float, float]) -> tuple[int, int, int, int]:
    z0, z1 = z_bounds
    t = (float(z) - z0) / max(1e-6, z1 - z0)
    t = max(0.0, min(1.0, t))
    low = np.array((55, 83, 48), dtype=np.float32)
    high = np.array((144, 151, 111), dtype=np.float32)
    rgb = (low * (1.0 - t) + high * t).astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), 245)


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
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def draw_title(
    draw: ImageDraw.ImageDraw,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    bounds: dict[str, tuple[float, float]],
) -> None:
    if not args.title:
        return
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
        small = ImageFont.truetype("Arial.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    pad = 22
    line1 = args.title
    if rows:
        line2 = (
            f"{args.match_id} / {args.agent_id.split(':')[-1]} | "
            f"points={len(rows)} | t={float(rows[0]['sample_t']):.1f}-{float(rows[-1]['sample_t']):.1f}s | "
            f"bounds x={bounds['x'][0]:.0f}..{bounds['x'][1]:.0f}, y={bounds['y'][0]:.0f}..{bounds['y'][1]:.0f}"
        )
    else:
        line2 = "no trajectory rows"
    draw.rectangle((0, 0, args.width, 76), fill=(14, 20, 20, 210))
    draw.text((pad, 14), line1, fill=(238, 244, 239, 255), font=font)
    draw.text((pad, 45), line2, fill=(198, 213, 207, 255), font=small)


def render(args: argparse.Namespace) -> dict[str, Any]:
    payload = pickle.load(args.cache.open("rb"))
    raw = {category: array_to_tris(payload["triangles"][category]) for category in CATEGORIES}
    display = {
        category: deterministic_sample(raw[category], getattr(args, f"{category}_triangles"))
        for category in CATEGORIES
    }
    bounds = compute_bounds(display, args.bounds_trim_percentile, args.square_bounds)
    for axis, override in (("x", args.bounds_x), ("y", args.bounds_y), ("z", args.bounds_z)):
        if override is not None:
            bounds[axis] = override

    image = Image.new("RGB", (args.width, args.height), args.background)
    overlay = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for category in CATEGORIES:
        tris = display[category]
        if not len(tris):
            continue
        centroids = tris.mean(axis=1)
        mask = (
            (centroids[:, 0] >= bounds["x"][0])
            & (centroids[:, 0] <= bounds["x"][1])
            & (centroids[:, 1] >= bounds["y"][0])
            & (centroids[:, 1] <= bounds["y"][1])
        )
        tris = tris[mask]
        for tri in tris:
            pts = project_xy(tri[:, :2], bounds, args.width, args.height, args.padding)
            poly = [tuple(map(int, point)) for point in pts]
            if category == "terrain":
                fill = terrain_color(float(tri[:, 2].mean()), bounds["z"])
            else:
                fill = COLORS[category]
            draw.polygon(poly, fill=fill)

    rows = fetch_agent_rows(args)
    snapshot_t = (
        args.snapshot_t
        if args.snapshot_t is not None
        else (float(rows[-1]["sample_t"]) if rows else None)
    )
    visible = [row for row in rows if snapshot_t is None or float(row["sample_t"]) <= snapshot_t]
    if visible:
        path = np.asarray(
            [(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in visible], dtype=np.float32
        )
        projected = project_xy(path, bounds, args.width, args.height, args.padding)
        line = [tuple(map(int, point)) for point in projected]
        if len(line) >= 2:
            draw.line(line, fill=args.path_shadow, width=args.path_width + 4, joint="curve")
            draw.line(line, fill=args.path_color, width=args.path_width, joint="curve")
        x, y = line[-1]
        r = args.agent_radius
        draw.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill=(10, 15, 15, 235))
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=args.path_color,
            outline=(255, 255, 255, 245),
            width=2,
        )

    draw_title(draw, args, visible, bounds)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    manifest = {
        "output": str(args.output.relative_to(ROOT)),
        "cache": str(args.cache.relative_to(ROOT)),
        "match_id": args.match_id,
        "agent_id": args.agent_id,
        "trajectory_rows": len(rows),
        "visible_rows": len(visible),
        "snapshot_t": snapshot_t,
        "bounds": bounds,
        "display_triangles": {category: int(len(display[category])) for category in CATEGORIES},
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def add_category_args(parser: argparse.ArgumentParser) -> None:
    for category, default in DISPLAY_BUDGETS.items():
        parser.add_argument(f"--{category.replace('_', '-')}-triangles", type=int, default=default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--source-artifact-id", default="caldera_minimal_layers_breadcrumbs_usd")
    parser.add_argument("--sample-t-min", type=float, default=0.0)
    parser.add_argument("--snapshot-t", type=float)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=2400)
    parser.add_argument("--padding", type=int, default=80)
    parser.add_argument("--bounds-trim-percentile", type=float, default=0.05)
    parser.add_argument("--bounds-x", type=parse_range)
    parser.add_argument("--bounds-y", type=parse_range)
    parser.add_argument("--bounds-z", type=parse_range)
    parser.add_argument("--square-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--background", type=parse_rgb, default=(16, 24, 25))
    parser.add_argument("--path-color", type=parse_rgba, default=(21, 195, 220, 245))
    parser.add_argument("--path-shadow", type=parse_rgba, default=(7, 17, 19, 210))
    parser.add_argument("--path-width", type=int, default=7)
    parser.add_argument("--agent-radius", type=int, default=15)
    parser.add_argument(
        "--title", default="Caldera single-agent trajectory over semantic spatial layer"
    )
    add_category_args(parser)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = render(args)
    print(f"image_written={args.output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
