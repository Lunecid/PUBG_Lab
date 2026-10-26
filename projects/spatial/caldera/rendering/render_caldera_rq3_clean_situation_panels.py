"""Render clean Caldera RQ3 situation panels from geometry features.

This script intentionally avoids text-heavy boards.  It renders small
geometry-grounded situation cuts that can be referenced from a paper table:
rough terrain, relative elevation, building / compound context, and route
corridor context.

The output figure contains no explanatory text; the feature names and meanings
belong in the manuscript table/caption.
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

import csv
import json
import math
import pickle
from array import array
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = REPO_ROOT
OUT_DIR = ROOT / "reports" / "caldera_3d" / "rq3_situation_renders"
MESH_CACHE = (
    ROOT / "reports" / "caldera_3d" / "single_agent" / "match_31_player_25_local_raw_mesh.pkl"
)
FEATURE_CSV = (
    ROOT
    / "reports"
    / "caldera_3d"
    / "single_agent"
    / "world_build"
    / "match_31_player_25_terrain_experience_features.csv"
)
MONTAGE_OUT = OUT_DIR / "caldera_rq3_clean_situation_montage.png"
TABLE_OUT = OUT_DIR / "caldera_rq3_clean_situation_feature_table.md"
MANIFEST_OUT = OUT_DIR / "caldera_rq3_clean_situation_manifest.json"

CATEGORIES = ["terrain", "road", "runway", "building", "prop"]
CATEGORY_COLORS = {
    "terrain": (138, 154, 139, 255),
    "road": (178, 181, 176, 255),
    "runway": (126, 132, 130, 255),
    "building": (207, 212, 211, 255),
    "prop": (160, 157, 145, 220),
}
LINE_COLOR = (23, 190, 215, 255)
MARKER_COLOR = (255, 112, 47, 255)
MARKER_OUTLINE = (255, 245, 235, 245)
BG = (231, 235, 235)
PANEL_BG = (222, 228, 228)

TRIANGLE_BUDGETS = {
    "terrain": 32_000,
    "road": 11_000,
    "runway": 8_000,
    "building": 44_000,
    "prop": 5_000,
}


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))


def sample_evenly(tris: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0 or len(tris) <= budget:
        return tris
    idx = np.linspace(0, len(tris) - 1, budget, dtype=np.int64)
    return tris[idx]


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def load_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(FEATURE_CSV.open(encoding="utf-8")))


def select_situations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": "rough_terrain",
            "caption_name": "rough terrain transition",
            "row": max(rows, key=lambda row: f(row, "terrain_slope_deg", -1.0)),
            "radius": 88.0,
            "feature_family": "Terrain shape",
            "features": ["terrain_slope_deg", "local_relief_r50_m", "terrain_ruggedness_r50_m"],
        },
        {
            "id": "relative_elevation",
            "caption_name": "relative local elevation",
            "row": max(rows, key=lambda row: f(row, "relative_elevation_r50_m", -1.0)),
            "radius": 94.0,
            "feature_family": "Local high-ground context",
            "features": ["relative_elevation_r50_m", "high_ground_local_r50", "forward_grade"],
        },
        {
            "id": "building_affordance",
            "caption_name": "compound / building affordance",
            "row": min(rows, key=lambda row: f(row, "building_dist_m", 1e9)),
            "radius": 82.0,
            "feature_family": "Compound / building geometry",
            "features": ["building_dist_m", "building_tri_density_r25_ha", "cover_building_r25"],
        },
        {
            "id": "route_corridor",
            "caption_name": "road / runway corridor contact",
            "row": min(rows, key=lambda row: f(row, "road_or_runway_dist_m", 1e9)),
            "radius": 96.0,
            "feature_family": "Route topology",
            "features": [
                "road_dist_m",
                "runway_dist_m",
                "road_or_runway_dist_m",
                "on_corridor_proxy",
            ],
        },
    ]


def load_mesh() -> dict[str, np.ndarray]:
    payload = pickle.load(MESH_CACHE.open("rb"))
    return {category: array_to_tris(payload["triangles"][category]) for category in CATEGORIES}


def filter_local(tris: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    if len(tris) == 0:
        return tris
    centroids = tris.mean(axis=1)
    mask = (
        (centroids[:, 0] >= cx - radius)
        & (centroids[:, 0] <= cx + radius)
        & (centroids[:, 1] >= cy - radius)
        & (centroids[:, 1] <= cy + radius)
    )
    return tris[mask]


def project(points: np.ndarray, cx: float, cy: float, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    z = points[:, 2]
    u = dx * math.cos(theta) - dy * math.sin(theta)
    depth = dx * math.sin(theta) + dy * math.cos(theta)
    v = -z * 2.25 - depth * 0.54
    return np.column_stack([u, v, depth])


def render_panel(
    mesh: dict[str, np.ndarray],
    trajectory_rows: list[dict[str, str]],
    situation: dict[str, Any],
    width: int = 1100,
    height: int = 740,
) -> Image.Image:
    row = situation["row"]
    cx = f(row, "x")
    cy = f(row, "y")
    radius = situation["radius"]
    angle = 32.0

    local: dict[str, np.ndarray] = {}
    projected_parts = []
    for category in CATEGORIES:
        tris = sample_evenly(
            filter_local(mesh[category], cx, cy, radius), TRIANGLE_BUDGETS[category]
        )
        local[category] = tris
        if len(tris):
            projected_parts.append(project(tris.reshape((-1, 3)), cx, cy, angle)[:, :2])
    if projected_parts:
        all_projected = np.concatenate(projected_parts, axis=0)
        min_xy = all_projected.min(axis=0)
        max_xy = all_projected.max(axis=0)
    else:
        min_xy = np.array([-radius, -radius])
        max_xy = np.array([radius, radius])

    margin = 52
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    offset = np.array([(width - span[0] * scale) * 0.5, (height - span[1] * scale) * 0.5])

    def to_px(points_uv: np.ndarray) -> np.ndarray:
        return np.rint((points_uv - min_xy) * scale + offset).astype(np.int32)

    image = Image.new("RGB", (width, height), PANEL_BG)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    draw.rectangle((0, 0, width, height), fill=(231, 235, 235, 255))

    triangles_to_draw: list[tuple[float, str, np.ndarray]] = []
    for category, tris in local.items():
        if not len(tris):
            continue
        proj = project(tris.reshape((-1, 3)), cx, cy, angle).reshape((-1, 3, 3))
        depths = proj[:, :, 2].mean(axis=1)
        for depth, tri_proj in zip(depths, proj):
            triangles_to_draw.append((float(depth), category, tri_proj[:, :2]))
    triangles_to_draw.sort(key=lambda item: item[0])

    for _, category, tri_uv in triangles_to_draw:
        pts = [tuple(map(int, p)) for p in to_px(tri_uv)]
        color = CATEGORY_COLORS[category]
        draw.polygon(pts, fill=color)

    path_points = []
    for r in trajectory_rows:
        x = f(r, "x")
        y = f(r, "y")
        if cx - radius <= x <= cx + radius and cy - radius <= y <= cy + radius:
            path_points.append((x, y, f(r, "z") + 1.4))
    if len(path_points) >= 2:
        path = np.asarray(path_points, dtype=np.float32)
        pts = to_px(project(path, cx, cy, angle)[:, :2])
        line = [tuple(map(int, point)) for point in pts]
        draw.line(line, fill=(24, 42, 44, 180), width=9, joint="curve")
        draw.line(line, fill=LINE_COLOR, width=5, joint="curve")

    marker = np.asarray([[f(row, "x"), f(row, "y"), f(row, "z") + 6.0]], dtype=np.float32)
    mx, my = to_px(project(marker, cx, cy, angle)[:, :2])[0]
    draw.ellipse(
        (mx - 15, my - 15, mx + 15, my + 15), fill=MARKER_COLOR, outline=MARKER_OUTLINE, width=4
    )

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return image


def write_feature_table(situations: list[dict[str, Any]]) -> None:
    lines = [
        "# Caldera RQ3 Clean Situation Feature Table",
        "",
        "| Panel order | Situation | Feature family | Designed features | Example values |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for idx, situation in enumerate(situations, start=1):
        row = situation["row"]
        values = (
            f"t={f(row, 't'):.3f}s; "
            f"slope={f(row, 'terrain_slope_deg'):.2f}deg; "
            f"relief50={f(row, 'local_relief_r50_m'):.2f}m; "
            f"building_dist={f(row, 'building_dist_m'):.2f}m; "
            f"road/runway_dist={f(row, 'road_or_runway_dist_m'):.2f}m"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    situation["caption_name"],
                    situation["feature_family"],
                    ", ".join(f"`{name}`" for name in situation["features"]),
                    values,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary: the panels are geometry-grounded situation examples. They are not Caldera survival or rank prediction results.",
        ]
    )
    TABLE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    mesh = load_mesh()
    situations = select_situations(rows)
    panels = []
    outputs = []
    for situation in situations:
        panel = render_panel(mesh, rows, situation)
        output = OUT_DIR / f"caldera_rq3_situation_{situation['id']}.png"
        panel.save(output)
        panels.append(panel)
        outputs.append(output)

    gap = 24
    w, h = panels[0].size
    montage = Image.new("RGB", (w * 2 + gap, h * 2 + gap), BG)
    for idx, panel in enumerate(panels):
        x = (idx % 2) * (w + gap)
        y = (idx // 2) * (h + gap)
        montage.paste(panel, (x, y))
    montage.save(MONTAGE_OUT)
    write_feature_table(situations)

    manifest = {
        "purpose": "Clean text-free Caldera RQ3 situation renders from geometry-derived feature exemplars.",
        "mesh_cache": str(MESH_CACHE.relative_to(ROOT)),
        "feature_csv": str(FEATURE_CSV.relative_to(ROOT)),
        "montage": str(MONTAGE_OUT.relative_to(ROOT)),
        "feature_table": str(TABLE_OUT.relative_to(ROOT)),
        "panel_outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "situations": [
            {
                "id": situation["id"],
                "caption_name": situation["caption_name"],
                "feature_family": situation["feature_family"],
                "features": situation["features"],
                "t": f(situation["row"], "t"),
                "x": f(situation["row"], "x"),
                "y": f(situation["row"], "y"),
                "z": f(situation["row"], "z"),
            }
            for situation in situations
        ],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"montage={MONTAGE_OUT}")
    print(f"table={TABLE_OUT}")
    print(f"manifest={MANIFEST_OUT}")


if __name__ == "__main__":
    main()
