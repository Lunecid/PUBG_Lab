"""Assemble the Caldera O6 construct-validation artifact package.

This creates and extends the RQ3 paper artifact requested by
``CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md``:

1. one composite figure with a semantic basemap, trajectory overlay,
   local terrain-experience view, and feature callout panel;
2. one descriptive construct-feature table.
3. one PUBG-to-Warzone observability-gap bridge plus situation cards.

The script intentionally uses existing Caldera caches and extracted features.
It does not query the database and does not rebuild the full world.
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = REPO_ROOT
OUT_DIR = ROOT / "reports" / "caldera_3d" / "construct_validation"

SEMANTIC_CACHE = (
    ROOT / "reports" / "caldera_3d" / "single_agent" / "match_31_player_25_semantic_triangles.pkl"
)
TOPDOWN_IMAGE = (
    ROOT / "reports" / "caldera_3d" / "single_agent" / "match_31_player_25_topdown_snapshot.png"
)
EXPERIENCE_IMAGE = (
    ROOT / "reports" / "caldera_3d" / "single_agent" / "match_31_player_25_experience_snapshot.png"
)
FEATURE_CSV = (
    ROOT
    / "reports"
    / "caldera_3d"
    / "single_agent"
    / "world_build"
    / "match_31_player_25_terrain_experience_features.csv"
)
FEATURE_SUMMARY = FEATURE_CSV.with_suffix(".summary.json")

BASEMAP_OUT = OUT_DIR / "caldera_o6_semantic_basemap_pil.png"
CALLOUT_OUT = OUT_DIR / "caldera_o6_feature_callout.png"
COMPOSITE_OUT = OUT_DIR / "caldera_o6_construct_validation_composite.png"
TABLE_CSV_OUT = OUT_DIR / "caldera_o6_construct_validation_descriptive_table.csv"
TABLE_MD_OUT = OUT_DIR / "caldera_o6_construct_validation_descriptive_table.md"
BRIDGE_CSV_OUT = OUT_DIR / "caldera_o6_pubg_warzone_observability_bridge.csv"
BRIDGE_MD_OUT = OUT_DIR / "caldera_o6_pubg_warzone_observability_bridge.md"
SITUATION_MD_OUT = OUT_DIR / "caldera_o6_pubg_gap_situation_cards.md"
BRIDGE_BOARD_OUT = OUT_DIR / "caldera_o6_pubg_warzone_observability_bridge_board.png"
MANIFEST_OUT = OUT_DIR / "caldera_o6_construct_validation_manifest.json"

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
    "terrain": 70_000,
    "coast_beach": 18_000,
    "water": 10_000,
    "vegetation": 22_000,
    "road": 30_000,
    "runway": 9_000,
    "building": 45_000,
    "prop": 5_000,
}

COLORS = {
    "coast_beach": (190, 166, 102),
    "water": (31, 105, 124),
    "vegetation": (28, 90, 43),
    "road": (184, 188, 178),
    "runway": (76, 82, 80),
    "building": (39, 45, 45),
    "prop": (105, 102, 91),
}

BG = (17, 23, 24)
PANEL_BG = (20, 27, 28)
PANEL_EDGE = (72, 84, 84)
TEXT = (236, 244, 241)
MUTED = (174, 191, 187)
ACCENT = (24, 194, 219)
ORANGE = (239, 178, 69)
GREEN = (111, 188, 100)
RED = (232, 91, 78)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))


def sample_evenly(tris: np.ndarray, max_count: int) -> np.ndarray:
    if len(tris) <= max_count:
        return tris
    idx = np.linspace(0, len(tris) - 1, max_count, dtype=np.int64)
    return tris[idx]


def square_bounds(
    points: np.ndarray, trim_percentile: float = 0.2
) -> dict[str, tuple[float, float]]:
    low = trim_percentile
    high = 100.0 - trim_percentile
    x0, x1 = np.percentile(points[:, 0], [low, high])
    y0, y1 = np.percentile(points[:, 1], [low, high])
    z0, z1 = np.percentile(points[:, 2], [1.0, 99.0])
    span = max(float(x1 - x0), float(y1 - y0))
    cx = float((x0 + x1) * 0.5)
    cy = float((y0 + y1) * 0.5)
    return {
        "x": (cx - span * 0.5, cx + span * 0.5),
        "y": (cy - span * 0.5, cy + span * 0.5),
        "z": (float(z0), float(max(z1, z0 + 1.0))),
    }


def project_xy(
    points: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    width: int,
    height: int,
    padding: int,
) -> np.ndarray:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    scale = min(
        (width - 2 * padding) / max(1e-6, x1 - x0), (height - 2 * padding) / max(1e-6, y1 - y0)
    )
    draw_w = (x1 - x0) * scale
    draw_h = (y1 - y0) * scale
    left = (width - draw_w) * 0.5
    top = (height - draw_h) * 0.5
    px = left + (points[:, 0] - x0) * scale
    py = top + (y1 - points[:, 1]) * scale
    return np.rint(np.column_stack([px, py])).astype(np.int32)


def terrain_color(z: float, z_bounds: tuple[float, float]) -> tuple[int, int, int]:
    z0, z1 = z_bounds
    t = max(0.0, min(1.0, (z - z0) / max(1e-6, z1 - z0)))
    stops = [
        (0.0, np.array((43, 74, 53), dtype=np.float32)),
        (0.45, np.array((92, 112, 67), dtype=np.float32)),
        (0.72, np.array((128, 113, 76), dtype=np.float32)),
        (1.0, np.array((164, 157, 121), dtype=np.float32)),
    ]
    for (left_t, left), (right_t, right) in zip(stops, stops[1:]):
        if t <= right_t:
            u = (t - left_t) / max(1e-6, right_t - left_t)
            rgb = (left * (1.0 - u) + right * u).astype(np.uint8)
            return int(rgb[0]), int(rgb[1]), int(rgb[2])
    rgb = stops[-1][1].astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def render_semantic_basemap() -> dict[str, Any]:
    payload = pickle.load(SEMANTIC_CACHE.open("rb"))
    triangles = {
        category: sample_evenly(
            array_to_tris(payload["triangles"][category]), DISPLAY_BUDGETS[category]
        )
        for category in CATEGORIES
    }
    all_points = np.concatenate(
        [tris.reshape((-1, 3)) for tris in triangles.values() if len(tris)], axis=0
    )
    bounds = square_bounds(all_points)

    width = 1600
    height = 1600
    padding = 58
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image, "RGBA")

    for category in CATEGORIES:
        tris = triangles[category]
        if len(tris) == 0:
            continue
        centroids = tris.mean(axis=1)
        keep = (
            (centroids[:, 0] >= bounds["x"][0])
            & (centroids[:, 0] <= bounds["x"][1])
            & (centroids[:, 1] >= bounds["y"][0])
            & (centroids[:, 1] <= bounds["y"][1])
        )
        tris = tris[keep]
        projected = project_xy(
            tris.reshape((-1, 3))[:, :2], bounds, width, height, padding
        ).reshape((-1, 3, 2))
        for tri, pts in zip(tris, projected):
            if category == "terrain":
                color = (*terrain_color(float(tri[:, 2].mean()), bounds["z"]), 235)
            elif category == "building":
                color = (*COLORS[category], 210)
            elif category in {"road", "runway"}:
                color = (*COLORS[category], 225)
            elif category == "water":
                color = (*COLORS[category], 205)
            else:
                color = (*COLORS[category], 155)
            draw.polygon([tuple(map(int, point)) for point in pts], fill=color)

    font = load_font(34, bold=True)
    small = load_font(20)
    draw.rectangle((0, 0, width, 88), fill=(12, 18, 19, 222))
    draw.text((30, 18), "A. Caldera semantic basemap", fill=TEXT, font=font)
    draw.text(
        (30, 55),
        "OpenUSD geometry classified by terrain, road, runway, building, vegetation, water",
        fill=MUTED,
        font=small,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(BASEMAP_OUT)
    return {
        "output": str(BASEMAP_OUT.relative_to(ROOT)),
        "source_cache": str(SEMANTIC_CACHE.relative_to(ROOT)),
        "bounds": {key: [float(value[0]), float(value[1])] for key, value in bounds.items()},
        "display_triangles": {key: int(len(value)) for key, value in triangles.items()},
    }


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def summarize_column(rows: list[dict[str, str]], column: str) -> dict[str, float | None]:
    values = [value for row in rows if (value := parse_float(row.get(column, ""))) is not None]
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def metric_line(summary: dict[str, float | None], digits: int = 2) -> str:
    return (
        f"min {fmt(summary['min'], digits)}, p25 {fmt(summary['p25'], digits)}, "
        f"median {fmt(summary['median'], digits)}, p75 {fmt(summary['p75'], digits)}, "
        f"max {fmt(summary['max'], digits)}"
    )


def read_feature_rows() -> list[dict[str, str]]:
    with FEATURE_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_feature_summary() -> dict[str, Any]:
    rows = read_feature_rows()
    summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    tactical_counts = Counter(row["tactical_state"] for row in rows)
    return {
        "rows": rows,
        "summary": summary,
        "tactical_counts": tactical_counts,
        "height_above_terrain_m": summarize_column(rows, "height_above_terrain_m"),
        "terrain_slope_deg": summarize_column(rows, "terrain_slope_deg"),
        "local_relief_r50_m": summarize_column(rows, "local_relief_r50_m"),
        "terrain_ruggedness_r50_m": summarize_column(rows, "terrain_ruggedness_r50_m"),
        "relative_elevation_r50_m": summarize_column(rows, "relative_elevation_r50_m"),
        "road_or_runway_dist_m": summarize_column(rows, "road_or_runway_dist_m"),
        "building_dist_m": summarize_column(rows, "building_dist_m"),
        "building_tri_density_r25_ha": summarize_column(rows, "building_tri_density_r25_ha"),
        "speed_mps": summarize_column(rows, "speed_mps"),
        "cover_building_r25_ratio": sum(parse_bool(row["cover_building_r25"]) for row in rows)
        / max(1, len(rows)),
        "on_corridor_proxy_ratio": sum(parse_bool(row["on_corridor_proxy"]) for row in rows)
        / max(1, len(rows)),
        "high_ground_local_r50_ratio": sum(parse_bool(row["high_ground_local_r50"]) for row in rows)
        / max(1, len(rows)),
    }


def build_descriptive_table(features: dict[str, Any]) -> list[dict[str, str]]:
    summary = features["summary"]
    policy = summary["motion_gap_policy"]
    tactical_counts = ", ".join(
        f"{key}={value}" for key, value in sorted(features["tactical_counts"].items())
    )
    return [
        {
            "Feature family": "Observation window",
            "Construct proxy": "trajectory segment projected into local Caldera mesh",
            "Descriptive values": (
                f"rows={summary['feature_rows']}; t={summary['time_range_s'][0]:.3f}-{summary['time_range_s'][1]:.3f}s; "
                f"life_segments={summary['life_segments']}"
            ),
            "Claim role": "defines the case-study unit; not a survival/rank label",
        },
        {
            "Feature family": "Life-segmentation guard",
            "Construct proxy": "motion gap and redeploy discontinuity policy",
            "Descriptive values": (
                f"motion_gap_rows={summary['motion_gap_rows']}; redeploy_gap_candidate_rows={summary['redeploy_gap_candidate_rows']}; "
                f"rule=dt>{policy['life_gap_s']:.0f}s and XY>{policy['life_displacement_m']:.0f}m"
            ),
            "Claim role": "prevents separated combat episodes from being treated as one continuous route",
        },
        {
            "Feature family": "Vertical consistency",
            "Construct proxy": "height_above_terrain_m",
            "Descriptive values": metric_line(features["height_above_terrain_m"], 2),
            "Claim role": "shows trajectory can be aligned against mesh-derived ground elevation",
        },
        {
            "Feature family": "Local terrain shape",
            "Construct proxy": "terrain_slope_deg; local_relief_r50_m; ruggedness_r50_m",
            "Descriptive values": (
                f"slope median {fmt(features['terrain_slope_deg']['median'])}, max {fmt(features['terrain_slope_deg']['max'])}; "
                f"relief50 {metric_line(features['local_relief_r50_m'], 2)}; "
                f"rugged50 median {fmt(features['terrain_ruggedness_r50_m']['median'])}"
            ),
            "Claim role": "captures terrain affordance that PUBG pos_z-only proxies cannot support robustly",
        },
        {
            "Feature family": "Built environment",
            "Construct proxy": "building_dist_m; building mesh density; cover_building_r25",
            "Descriptive values": (
                f"building distance {metric_line(features['building_dist_m'], 2)}; "
                f"density25 median {fmt(features['building_tri_density_r25_ha']['median'], 1)} tri/ha; "
                f"cover_r25={features['cover_building_r25_ratio']:.2%}"
            ),
            "Claim role": "measures compound/building exposure from actual geometry categories",
        },
        {
            "Feature family": "Route context",
            "Construct proxy": "road_or_runway_dist_m; on_corridor_proxy",
            "Descriptive values": (
                f"road/runway distance {metric_line(features['road_or_runway_dist_m'], 2)}; "
                f"corridor_proxy={features['on_corridor_proxy_ratio']:.2%}"
            ),
            "Claim role": "links route movement to road/runway corridor affordances",
        },
        {
            "Feature family": "Tactical state",
            "Construct proxy": "terrain/building/corridor rules under battle_royale_ontology_v1",
            "Descriptive values": tactical_counts,
            "Claim role": "demonstrates descriptive construct labeling, not supervised outcome prediction",
        },
    ]


def write_tables(table_rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Feature family", "Construct proxy", "Descriptive values", "Claim role"]
    with TABLE_CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    lines = [
        "# Caldera O6 Construct-Validation Descriptive Table",
        "",
        "| Feature family | Construct proxy | Descriptive values | Claim role |",
        "| --- | --- | --- | --- |",
    ]
    for row in table_rows:
        values = [row[field].replace("|", "\\|") for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    TABLE_MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def row_float(row: dict[str, str], key: str) -> float:
    value = parse_float(row.get(key, ""))
    return value if value is not None else math.nan


def select_extreme(rows: list[dict[str, str]], key: str, *, reverse: bool) -> dict[str, str]:
    values = [row for row in rows if math.isfinite(row_float(row, key))]
    if not values:
        return rows[0]
    return sorted(values, key=lambda row: row_float(row, key), reverse=reverse)[0]


def build_observability_bridge(
    features: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = features["rows"]
    rough = select_extreme(rows, "terrain_slope_deg", reverse=True)
    relief = select_extreme(rows, "local_relief_r50_m", reverse=True)
    building = select_extreme(rows, "building_dist_m", reverse=False)
    corridor = select_extreme(rows, "road_or_runway_dist_m", reverse=False)
    density = select_extreme(rows, "building_tri_density_r25_ha", reverse=True)
    relative_high = select_extreme(rows, "relative_elevation_r50_m", reverse=True)

    bridge_rows = [
        {
            "Construct / situation": "terrain surface and vertical context",
            "PUBG official telemetry gap": "Player or team z exists, but the official telemetry stream does not provide the terrain mesh below the actor; z alone cannot separate altitude from terrain surface, slope, or local relief.",
            "Warzone / Caldera representation": "OpenUSD terrain triangles plus trajectory samples produce terrain_z_m, height_above_terrain_m, relative_elevation_r50_m, and local relief.",
            "Current evidence": f"height_above_terrain_m median {fmt(features['height_above_terrain_m']['median'])}m, p75 {fmt(features['height_above_terrain_m']['p75'])}m, max {fmt(features['height_above_terrain_m']['max'])}m; example t={rough['t']}s.",
            "Status": "supported descriptive construct",
        },
        {
            "Construct / situation": "rough terrain mobility",
            "PUBG official telemetry gap": "Official PUBG telemetry can show movement over coordinates, but not whether the route crosses a steep or rugged local surface without external terrain geometry.",
            "Warzone / Caldera representation": "Mesh-derived terrain_slope_deg, local_relief_r50_m, and ruggedness features around each trajectory sample.",
            "Current evidence": f"max slope {float(rough['terrain_slope_deg']):.2f}deg at t={rough['t']}s; max relief50 {float(relief['local_relief_r50_m']):.2f}m at t={relief['t']}s; rough_mobility rows=18.",
            "Status": "supported descriptive construct",
        },
        {
            "Construct / situation": "compound and building affordance",
            "PUBG official telemetry gap": "Official telemetry alone does not expose building meshes, wall mass, doorway topology, or compound geometry needed to interpret cover-like spatial affordance.",
            "Warzone / Caldera representation": "Semantic building triangles support nearest-building distance, building mesh density, and cover_building_r25 proxies.",
            "Current evidence": f"building_dist_m median {fmt(features['building_dist_m']['median'])}m; closest sample {float(building['building_dist_m']):.2f}m at t={building['t']}s; cover_building_r25={features['cover_building_r25_ratio']:.2%}.",
            "Status": "supported descriptive construct",
        },
        {
            "Construct / situation": "route corridor and road/runway context",
            "PUBG official telemetry gap": "Coordinates reveal where a team moved, but official telemetry does not label whether the path is on a road, runway, bridge-like corridor, or open terrain.",
            "Warzone / Caldera representation": "Road and runway mesh categories support road_or_runway_dist_m and on_corridor_proxy.",
            "Current evidence": f"road_or_runway_dist_m median {fmt(features['road_or_runway_dist_m']['median'])}m; nearest corridor sample {float(corridor['road_or_runway_dist_m']):.2f}m at t={corridor['t']}s; on_corridor_proxy={features['on_corridor_proxy_ratio']:.2%}.",
            "Status": "supported descriptive construct",
        },
        {
            "Construct / situation": "local tactical-state labeling",
            "PUBG official telemetry gap": "Telemetry events can indicate combat and movement, but they do not directly assign geometry-grounded states such as compound_movement or rough_mobility.",
            "Warzone / Caldera representation": "Rules over mesh-derived terrain, building, and corridor features produce descriptive tactical_state_v0 labels.",
            "Current evidence": ", ".join(
                f"{key}={value}" for key, value in sorted(features["tactical_counts"].items())
            ),
            "Status": "supported descriptive construct, not outcome supervision",
        },
        {
            "Construct / situation": "visibility, occlusion, and line of sight",
            "PUBG official telemetry gap": "Official telemetry lacks occluding terrain/building mesh needed to ray-cast whether two agents are mutually visible.",
            "Warzone / Caldera representation": "The same geometry cache and same-clock trajectories can support future mesh ray-cast LOS or occlusion edges.",
            "Current evidence": "Not computed in this package; reserved as LOS stretch and marked pending in the bridge artifact.",
            "Status": "pending extension, not claimed",
        },
    ]

    situation_rows = [
        {
            "Situation card": "rough terrain transition",
            "Example": f"t={rough['t']}s, slope={float(rough['terrain_slope_deg']):.2f}deg, relief50={float(rough['local_relief_r50_m']):.2f}m, state={rough['tactical_state']}",
            "Why PUBG official telemetry is insufficient": "The route may look like ordinary coordinate movement unless local mesh slope and relief are available.",
            "What Warzone / Caldera shows": "The same trajectory point can be grounded in a steep, rugged local terrain surface.",
        },
        {
            "Situation card": "compound movement beside built mass",
            "Example": f"t={building['t']}s, building_dist={float(building['building_dist_m']):.2f}m, density25={float(building['building_tri_density_r25_ha']):.1f} tri/ha, cover_r25={building['cover_building_r25']}",
            "Why PUBG official telemetry is insufficient": "Actor position alone does not expose nearby building mesh density or compound geometry.",
            "What Warzone / Caldera shows": "The path can be described as moving through or beside dense built geometry.",
        },
        {
            "Situation card": "road/runway corridor contact",
            "Example": f"t={corridor['t']}s, road_or_runway_dist={float(corridor['road_or_runway_dist_m']):.2f}m, corridor_proxy={corridor['on_corridor_proxy']}",
            "Why PUBG official telemetry is insufficient": "Official telemetry does not directly attach road/runway semantics to coordinates.",
            "What Warzone / Caldera shows": "The path can be tied to explicit route-surface categories.",
        },
        {
            "Situation card": "dense building-mesh pocket",
            "Example": f"t={density['t']}s, building_density25={float(density['building_tri_density_r25_ha']):.1f} tri/ha, building_dist={float(density['building_dist_m']):.2f}m",
            "Why PUBG official telemetry is insufficient": "A coordinate sample cannot quantify nearby built mass without world geometry.",
            "What Warzone / Caldera shows": "The local environment can be measured as a high-density built pocket.",
        },
        {
            "Situation card": "relative local elevation",
            "Example": f"t={relative_high['t']}s, relative_elevation_r50={float(relative_high['relative_elevation_r50_m']):.2f}m, relief50={float(relative_high['local_relief_r50_m']):.2f}m",
            "Why PUBG official telemetry is insufficient": "Raw z does not say whether the actor is high or low relative to nearby terrain.",
            "What Warzone / Caldera shows": "The sample can be placed into a local elevation neighborhood.",
        },
    ]
    return bridge_rows, situation_rows


def write_bridge_tables(
    bridge_rows: list[dict[str, str]], situation_rows: list[dict[str, str]]
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge_fields = [
        "Construct / situation",
        "PUBG official telemetry gap",
        "Warzone / Caldera representation",
        "Current evidence",
        "Status",
    ]
    with BRIDGE_CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bridge_fields)
        writer.writeheader()
        writer.writerows(bridge_rows)

    lines = [
        "# PUBG-to-Warzone Observability Gap Bridge",
        "",
        "This table is the RQ3 claim surface: it states which Battle Royale",
        "situations are not directly observable from official PUBG telemetry alone",
        "and how Warzone / Caldera geometry can represent them.",
        "",
        "| Construct / situation | PUBG official telemetry gap | Warzone / Caldera representation | Current evidence | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in bridge_rows:
        values = [row[field].replace("|", "\\|") for field in bridge_fields]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    BRIDGE_MD_OUT.write_text("\n".join(lines), encoding="utf-8")

    situation_fields = [
        "Situation card",
        "Example",
        "Why PUBG official telemetry is insufficient",
        "What Warzone / Caldera shows",
    ]
    card_lines = [
        "# PUBG Gap Situation Cards From Caldera O6 Features",
        "",
        "These cards are examples for prose and figure captions. They should be",
        "described as construct evidence, not as survival/rank prediction evidence.",
        "",
    ]
    for idx, row in enumerate(situation_rows, start=1):
        card_lines.extend(
            [
                f"## {idx}. {row['Situation card'].title()}",
                "",
                f"- Example: {row['Example']}",
                f"- PUBG telemetry gap: {row['Why PUBG official telemetry is insufficient']}",
                f"- Warzone / Caldera representation: {row['What Warzone / Caldera shows']}",
                "",
            ]
        )
    SITUATION_MD_OUT.write_text("\n".join(card_lines), encoding="utf-8")


def wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 5,
) -> int:
    x, y = xy
    line_height = text_size(draw, "Ag", font)[1] + line_gap
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def render_bridge_board(bridge_rows: list[dict[str, str]]) -> None:
    width, height = 3400, 2380
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    title = load_font(58, bold=True)
    subtitle = load_font(27)
    header = load_font(25, bold=True)
    body = load_font(22)
    small = load_font(19)

    margin = 58
    draw.text(
        (margin, 36),
        "RQ3 Bridge: PUBG Telemetry Gap -> Warzone Geometry Representation",
        fill=TEXT,
        font=title,
    )
    draw.text(
        (margin, 104),
        "The point of Warzone / Caldera is not a second supervised dataset; it shows Battle Royale situations that PUBG official telemetry cannot directly express.",
        fill=MUTED,
        font=subtitle,
    )

    columns = [
        ("Situation", 0.16),
        ("PUBG official telemetry gap", 0.29),
        ("Warzone / Caldera representation", 0.29),
        ("Current evidence / status", 0.26),
    ]
    table_x = margin
    table_y = 175
    table_w = width - margin * 2
    row_h = 300
    header_h = 70
    x_positions = [table_x]
    for _, frac in columns[:-1]:
        x_positions.append(x_positions[-1] + int(table_w * frac))
    col_widths = [int(table_w * frac) for _, frac in columns]
    col_widths[-1] = table_x + table_w - x_positions[-1]

    draw.rounded_rectangle(
        (table_x, table_y, table_x + table_w, table_y + header_h),
        radius=10,
        fill=(11, 17, 18),
        outline=PANEL_EDGE,
    )
    for idx, (label_text, _) in enumerate(columns):
        draw.text((x_positions[idx] + 18, table_y + 21), label_text, fill=TEXT, font=header)

    y = table_y + header_h + 12
    status_colors = {
        "supported descriptive construct": ACCENT,
        "supported descriptive construct, not outcome supervision": GREEN,
        "pending extension, not claimed": ORANGE,
    }
    for row in bridge_rows:
        draw.rounded_rectangle(
            (table_x, y, table_x + table_w, y + row_h),
            radius=10,
            fill=PANEL_BG,
            outline=PANEL_EDGE,
            width=1,
        )
        for idx in range(1, len(x_positions)):
            draw.line(
                (x_positions[idx], y + 16, x_positions[idx], y + row_h - 16),
                fill=(54, 65, 66),
                width=1,
            )

        draw_wrapped_text(
            draw,
            (x_positions[0] + 18, y + 26),
            row["Construct / situation"],
            header,
            TEXT,
            col_widths[0] - 36,
            6,
        )
        draw_wrapped_text(
            draw,
            (x_positions[1] + 18, y + 26),
            row["PUBG official telemetry gap"],
            body,
            MUTED,
            col_widths[1] - 36,
            5,
        )
        draw_wrapped_text(
            draw,
            (x_positions[2] + 18, y + 26),
            row["Warzone / Caldera representation"],
            body,
            MUTED,
            col_widths[2] - 36,
            5,
        )
        evidence_y = draw_wrapped_text(
            draw,
            (x_positions[3] + 18, y + 26),
            row["Current evidence"],
            body,
            TEXT,
            col_widths[3] - 36,
            5,
        )
        status_color = status_colors.get(row["Status"], ACCENT)
        draw.rounded_rectangle(
            (
                x_positions[3] + 18,
                evidence_y + 12,
                x_positions[3] + col_widths[3] - 18,
                evidence_y + 46,
            ),
            radius=8,
            fill=(32, 41, 42),
        )
        draw.text(
            (x_positions[3] + 34, evidence_y + 18), row["Status"], fill=status_color, font=small
        )
        y += row_h + 18

    image.save(BRIDGE_BOARD_OUT)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    value: float,
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=6, fill=(42, 51, 52), outline=(70, 82, 82), width=1)
    fill_x = x0 + int((x1 - x0) * max(0.0, min(1.0, value)))
    draw.rounded_rectangle((x0, y0, fill_x, y1), radius=6, fill=color)


def render_feature_callout(features: dict[str, Any]) -> dict[str, Any]:
    summary = features["summary"]
    width, height = 1600, 1120
    image = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(image, "RGBA")
    title = load_font(48, bold=True)
    section = load_font(31, bold=True)
    label = load_font(24, bold=True)
    small = load_font(23)
    tiny = load_font(19)

    draw.text((52, 42), "D. O6 construct feature callout", fill=TEXT, font=title)
    draw.text(
        (52, 98),
        "Descriptive terrain, building, corridor, and guard features; no survival/rank labels",
        fill=MUTED,
        font=small,
    )

    cards = [
        ("Rows", f"{summary['feature_rows']}", ACCENT),
        ("Life segments", f"{summary['life_segments']}", GREEN),
        ("Redeploy gaps", f"{summary['redeploy_gap_candidate_rows']}", RED),
        ("Motion gaps", f"{summary['motion_gap_rows']}", ORANGE),
    ]
    card_w = 350
    for idx, (name, value, color) in enumerate(cards):
        x = 52 + idx * (card_w + 26)
        y = 160
        draw.rounded_rectangle(
            (x, y, x + card_w, y + 146), radius=10, fill=(28, 36, 37), outline=PANEL_EDGE, width=1
        )
        draw.text((x + 24, y + 20), name, fill=MUTED, font=small)
        draw.text((x + 24, y + 58), value, fill=color, font=load_font(50, bold=True))

    y = 358
    draw.text((52, y), "Geometry-derived distributions", fill=TEXT, font=section)
    metrics = [
        ("Height above terrain", features["height_above_terrain_m"], "m", ACCENT),
        ("Local relief r50", features["local_relief_r50_m"], "m", GREEN),
        ("Building distance", features["building_dist_m"], "m", ORANGE),
        ("Road/runway distance", features["road_or_runway_dist_m"], "m", RED),
    ]
    y += 58
    for name, stats, unit, color in metrics:
        max_value = max(abs(stats["max"] or 0.0), 1.0)
        median = max(0.0, float(stats["median"] or 0.0)) / max_value
        p75 = max(0.0, float(stats["p75"] or 0.0)) / max_value
        draw.text((70, y), name, fill=TEXT, font=label)
        draw.text(
            (70, y + 31),
            f"median {fmt(stats['median'])} {unit} | p75 {fmt(stats['p75'])} {unit} | max {fmt(stats['max'])} {unit}",
            fill=MUTED,
            font=tiny,
        )
        draw_bar(
            draw, (640, y + 8, 1455, y + 32), p75, tuple(max(0, channel - 38) for channel in color)
        )
        draw_bar(draw, (640, y + 42, 1455, y + 66), median, color)
        y += 88

    y += 24
    draw.text((52, y), "Tactical state distribution", fill=TEXT, font=section)
    total = max(1, sum(features["tactical_counts"].values()))
    y += 60
    state_colors = [ACCENT, GREEN, ORANGE, RED]
    for idx, (state, count) in enumerate(
        sorted(features["tactical_counts"].items(), key=lambda item: (-item[1], item[0]))
    ):
        ratio = count / total
        color = state_colors[idx % len(state_colors)]
        draw.text((70, y), state, fill=TEXT, font=label)
        draw.text((430, y), f"{count} ({ratio:.1%})", fill=MUTED, font=small)
        draw_bar(draw, (640, y + 4, 1455, y + 33), ratio, color)
        y += 62

    y += 18
    guard = summary["motion_gap_policy"]
    draw.rounded_rectangle(
        (52, height - 128, width - 52, height - 42),
        radius=9,
        fill=(27, 35, 36),
        outline=PANEL_EDGE,
        width=1,
    )
    draw.text(
        (78, height - 104),
        f"Life guard: new segment if dt > {guard['life_gap_s']:.0f}s and XY displacement > {guard['life_displacement_m']:.0f}m",
        fill=TEXT,
        font=small,
    )
    draw.text(
        (78, height - 73),
        "Interpretation: construct validation for O6 spatial affordance, not Caldera outcome prediction.",
        fill=MUTED,
        font=tiny,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(CALLOUT_OUT)
    return {
        "output": str(CALLOUT_OUT.relative_to(ROOT)),
        "source_summary": str(FEATURE_SUMMARY.relative_to(ROOT)),
    }


def paste_panel(
    canvas: Image.Image,
    source_path: Path,
    rect: tuple[int, int, int, int],
    label_text: str,
    subtitle: str,
    *,
    crop_to_fill: bool = False,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=10, fill=PANEL_BG, outline=PANEL_EDGE, width=2)
    header_h = 86
    draw.rounded_rectangle((x0, y0, x1, y0 + header_h), radius=10, fill=(12, 18, 19, 232))
    draw.rectangle((x0, y0 + header_h - 12, x1, y0 + header_h), fill=(12, 18, 19, 232))
    draw.text((x0 + 24, y0 + 17), label_text, fill=TEXT, font=load_font(31, bold=True))
    draw.text((x0 + 24, y0 + 53), subtitle, fill=MUTED, font=load_font(20))

    inner = (x0 + 18, y0 + header_h + 18, x1 - 18, y1 - 18)
    target_w = inner[2] - inner[0]
    target_h = inner[3] - inner[1]
    image = Image.open(source_path).convert("RGB")
    if crop_to_fill:
        fitted = ImageOps.fit(
            image, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )
    else:
        fitted = ImageOps.contain(image, (target_w, target_h), method=Image.Resampling.LANCZOS)
    px = inner[0] + (target_w - fitted.width) // 2
    py = inner[1] + (target_h - fitted.height) // 2
    canvas.paste(fitted, (px, py))


def assemble_composite() -> None:
    width, height = 3200, 2350
    margin = 64
    title_h = 134
    gap = 38
    panel_w = (width - margin * 2 - gap) // 2
    panel_h = (height - margin - title_h - gap - margin) // 2
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text(
        (margin, 42),
        "Caldera / Warzone O6 Construct Validation",
        fill=TEXT,
        font=load_font(58, bold=True),
    )
    draw.text(
        (margin, 104),
        "Geometry-rich evidence for terrain, topology, route, and built-environment representation; not supervised survival/rank prediction.",
        fill=MUTED,
        font=load_font(27),
    )

    top_y = title_h
    rect_a = (margin, top_y, margin + panel_w, top_y + panel_h)
    rect_b = (margin + panel_w + gap, top_y, width - margin, top_y + panel_h)
    rect_c = (margin, top_y + panel_h + gap, margin + panel_w, height - margin)
    rect_d = (margin + panel_w + gap, top_y + panel_h + gap, width - margin, height - margin)

    paste_panel(
        canvas,
        BASEMAP_OUT,
        rect_a,
        "A. Semantic basemap",
        "OpenUSD mesh categories establish geometry-rich spatial substrate",
        crop_to_fill=False,
    )
    paste_panel(
        canvas,
        TOPDOWN_IMAGE,
        rect_b,
        "B. Trajectory overlay",
        "Same agent path projected onto terrain, roads, runway, buildings",
        crop_to_fill=False,
    )
    paste_panel(
        canvas,
        EXPERIENCE_IMAGE,
        rect_c,
        "C. Terrain-experience zoom",
        "Local route context with altitude, terrain, and proximity diagnostics",
        crop_to_fill=True,
    )
    paste_panel(
        canvas,
        CALLOUT_OUT,
        rect_d,
        "D. Feature callout",
        "Descriptive O6 construct features with life-segmentation guard",
        crop_to_fill=False,
    )

    canvas.save(COMPOSITE_OUT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basemap_manifest = render_semantic_basemap()
    features = compute_feature_summary()
    table_rows = build_descriptive_table(features)
    write_tables(table_rows)
    bridge_rows, situation_rows = build_observability_bridge(features)
    write_bridge_tables(bridge_rows, situation_rows)
    render_bridge_board(bridge_rows)
    callout_manifest = render_feature_callout(features)
    assemble_composite()

    manifest = {
        "purpose": "Caldera O6 PUBG-to-Warzone observability-gap bridge and geometry-rich construct-validation artifact package for RQ3.",
        "created_outputs": {
            "composite_figure": str(COMPOSITE_OUT.relative_to(ROOT)),
            "semantic_basemap": str(BASEMAP_OUT.relative_to(ROOT)),
            "feature_callout": str(CALLOUT_OUT.relative_to(ROOT)),
            "descriptive_table_csv": str(TABLE_CSV_OUT.relative_to(ROOT)),
            "descriptive_table_md": str(TABLE_MD_OUT.relative_to(ROOT)),
            "pubg_warzone_observability_bridge_csv": str(BRIDGE_CSV_OUT.relative_to(ROOT)),
            "pubg_warzone_observability_bridge_md": str(BRIDGE_MD_OUT.relative_to(ROOT)),
            "pubg_gap_situation_cards_md": str(SITUATION_MD_OUT.relative_to(ROOT)),
            "pubg_warzone_observability_bridge_board": str(BRIDGE_BOARD_OUT.relative_to(ROOT)),
        },
        "source_inputs": {
            "semantic_cache": str(SEMANTIC_CACHE.relative_to(ROOT)),
            "topdown_image": str(TOPDOWN_IMAGE.relative_to(ROOT)),
            "experience_image": str(EXPERIENCE_IMAGE.relative_to(ROOT)),
            "feature_csv": str(FEATURE_CSV.relative_to(ROOT)),
            "feature_summary": str(FEATURE_SUMMARY.relative_to(ROOT)),
        },
        "basemap_manifest": basemap_manifest,
        "callout_manifest": callout_manifest,
        "claim_boundary": [
            "Allowed: Warzone / Caldera shows how geometry-rich data can represent terrain, route, building, and local tactical-state constructs that official PUBG telemetry cannot directly observe.",
            "Allowed: Caldera complements PUBG because PUBG is label-rich for supervised survival/rank prediction while Caldera is geometry-rich for construct validation.",
            "Avoid: Caldera validates supervised survival/rank prediction.",
            "Avoid: The current artifact proves LOS or occlusion effects.",
            "Avoid: The render is a full game-engine reconstruction.",
        ],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"composite={COMPOSITE_OUT}")
    print(f"table_csv={TABLE_CSV_OUT}")
    print(f"table_md={TABLE_MD_OUT}")
    print(f"bridge_md={BRIDGE_MD_OUT}")
    print(f"situation_cards={SITUATION_MD_OUT}")
    print(f"bridge_board={BRIDGE_BOARD_OUT}")
    print(f"manifest={MANIFEST_OUT}")


if __name__ == "__main__":
    main()
