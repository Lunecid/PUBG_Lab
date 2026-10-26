"""Extract terrain-experience features for one Caldera trajectory segment.

This is a local prototype for the terrain-experience feature family. It reads a
Caldera local raw mesh cache plus a trajectory JSON and writes sample-level CSV
and JSON feature rows.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import csv
import json
import math
import pickle
from array import array
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


CATEGORIES = ["terrain", "road", "runway", "building"]
TACTICAL_STATE_SCHEMA = "battle_royale_ontology_v1"


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    tris = np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))
    return tris[np.isfinite(tris).all(axis=(1, 2))]


def parse_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    lo, hi = (float(part.strip()) for part in parts)
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be greater than lower bound")
    return lo, hi


def crop_tris(tris: np.ndarray, bounds: dict[str, tuple[float, float]] | None) -> np.ndarray:
    if not bounds or len(tris) == 0:
        return tris
    centroids = tris.mean(axis=1)
    keep = np.ones(len(tris), dtype=bool)
    axis_idx = {"x": 0, "y": 1, "z": 2}
    for axis, (lo, hi) in bounds.items():
        keep &= (centroids[:, axis_idx[axis]] >= lo) & (centroids[:, axis_idx[axis]] <= hi)
    return tris[keep]


def trajectory_bounds(
    rows: list[dict[str, Any]], xy_margin: float, z_margin: float
) -> dict[str, tuple[float, float]]:
    points = np.asarray([(row["x"], row["y"], row["z"]) for row in rows], dtype=np.float32)
    return {
        "x": (float(points[:, 0].min() - xy_margin), float(points[:, 0].max() + xy_margin)),
        "y": (float(points[:, 1].min() - xy_margin), float(points[:, 1].max() + xy_margin)),
        "z": (float(points[:, 2].min() - z_margin), float(points[:, 2].max() + z_margin)),
    }


def slope_from_triangles(tris: np.ndarray) -> np.ndarray:
    if len(tris) == 0:
        return np.empty((0,), dtype=np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    valid = norm > 1e-8
    slopes = np.full(len(tris), np.nan, dtype=np.float32)
    cos_theta = np.zeros(len(tris), dtype=np.float32)
    cos_theta[valid] = np.abs(normals[valid, 2]) / norm[valid]
    slopes[valid] = np.degrees(np.arccos(np.clip(cos_theta[valid], 0.0, 1.0)))
    return slopes


def barycentric_z(point_xy: np.ndarray, tri: np.ndarray) -> tuple[float, bool]:
    a, b, c = tri
    px, py = point_xy
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(den)) < 1e-8:
        return float(tri[:, 2].mean()), False
    w1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / den
    w2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / den
    w3 = 1.0 - w1 - w2
    inside = (w1 >= -0.03) and (w2 >= -0.03) and (w3 >= -0.03)
    z = w1 * a[2] + w2 * b[2] + w3 * c[2]
    return float(z), bool(inside)


class TerrainIndex:
    def __init__(self, tris: np.ndarray):
        if len(tris) == 0:
            raise ValueError("terrain triangle set is empty")
        self.tris = tris
        self.centroids = tris.mean(axis=1).astype(np.float32)
        self.centroid_tree = cKDTree(self.centroids[:, :2])
        self.slopes = slope_from_triangles(tris)
        vertices = tris.reshape((-1, 3)).astype(np.float32)
        self.vertex_xy = vertices[:, :2]
        self.vertex_z = vertices[:, 2]
        self.vertex_tree = cKDTree(self.vertex_xy)

    def surface_at(self, point_xy: np.ndarray, k: int = 32) -> dict[str, float | bool]:
        k = min(k, len(self.tris))
        dist, idx = self.centroid_tree.query(point_xy, k=k)
        idxs = np.atleast_1d(idx)
        dists = np.atleast_1d(dist)
        best = int(idxs[0])
        best_z = float(self.centroids[best, 2])
        best_inside = False
        best_dist = float(dists[0])
        for tri_idx, tri_dist in zip(idxs, dists):
            z, inside = barycentric_z(point_xy, self.tris[int(tri_idx)])
            if inside:
                return {
                    "terrain_z_m": z,
                    "terrain_slope_deg": float(self.slopes[int(tri_idx)]),
                    "terrain_lookup_inside_tri": True,
                    "terrain_lookup_xy_dist_m": float(tri_dist),
                }
            if float(tri_dist) < best_dist:
                best = int(tri_idx)
                best_z = z
                best_dist = float(tri_dist)
        return {
            "terrain_z_m": best_z,
            "terrain_slope_deg": float(self.slopes[best]),
            "terrain_lookup_inside_tri": best_inside,
            "terrain_lookup_xy_dist_m": best_dist,
        }

    def relief(self, point_xy: np.ndarray, radius: float) -> tuple[float, float, float]:
        idxs = self.vertex_tree.query_ball_point(point_xy, r=radius)
        if len(idxs) < 2:
            return math.nan, math.nan, math.nan
        values = self.vertex_z[idxs]
        return float(values.max() - values.min()), float(values.mean()), float(values.std())


class CategoryIndex:
    def __init__(self, tris: np.ndarray):
        self.tris = tris
        self.centroids = (
            tris.mean(axis=1).astype(np.float32)
            if len(tris)
            else np.empty((0, 3), dtype=np.float32)
        )
        self.tree = cKDTree(self.centroids[:, :2]) if len(self.centroids) else None

    def distance(self, point_xy: np.ndarray) -> float:
        if self.tree is None:
            return math.inf
        dist, _ = self.tree.query(point_xy, k=1)
        return float(dist)

    def density_per_ha(self, point_xy: np.ndarray, radius: float) -> float:
        if self.tree is None:
            return 0.0
        count = len(self.tree.query_ball_point(point_xy, r=radius))
        area_m2 = math.pi * radius * radius
        return float(count / area_m2 * 10_000.0)


def compute_motion(
    rows: list[dict[str, Any]],
    *,
    max_motion_dt_s: float,
    life_gap_s: float,
    life_displacement_m: float,
) -> None:
    prev_heading = math.nan
    life_segment_id = 0
    for idx, row in enumerate(rows):
        if idx == 0:
            row["dt_s"] = math.nan
            row["speed_mps"] = math.nan
            row["heading_deg"] = math.nan
            row["turn_rate_deg_s"] = math.nan
            row["horizontal_step_m"] = math.nan
            row["vertical_step_m"] = math.nan
            row["motion_gap_break"] = False
            row["redeploy_gap_candidate"] = False
            row["life_segment_id"] = life_segment_id
            continue
        prev = rows[idx - 1]
        dt = row["t"] - prev["t"]
        dx = row["x"] - prev["x"]
        dy = row["y"] - prev["y"]
        dz = row["z"] - prev["z"]
        dist_xy = math.hypot(dx, dy)
        is_motion_gap = dt <= 0 or dt > max_motion_dt_s
        is_redeploy_gap = dt > life_gap_s and dist_xy > life_displacement_m
        if is_redeploy_gap:
            life_segment_id += 1
            prev_heading = math.nan

        row["dt_s"] = dt
        row["horizontal_step_m"] = dist_xy
        row["vertical_step_m"] = dz
        row["motion_gap_break"] = bool(is_motion_gap)
        row["redeploy_gap_candidate"] = bool(is_redeploy_gap)
        row["life_segment_id"] = life_segment_id

        if is_motion_gap:
            row["speed_mps"] = math.nan
            row["heading_deg"] = math.nan
            row["turn_rate_deg_s"] = math.nan
            prev_heading = math.nan
        else:
            row["speed_mps"] = math.sqrt(dx * dx + dy * dy + dz * dz) / dt
            heading = math.degrees(math.atan2(dy, dx))
            row["heading_deg"] = heading
            delta = (heading - prev_heading + 180.0) % 360.0 - 180.0
            row["turn_rate_deg_s"] = math.nan if math.isnan(prev_heading) else abs(delta) / dt
            prev_heading = heading


def tactical_state(row: dict[str, Any]) -> str:
    if row["height_above_terrain_m"] > 8.0:
        return "airborne_or_unresolved"
    if row["terrain_slope_deg"] >= 25.0 or row["local_relief_r50_m"] >= 12.0:
        return "rough_mobility"
    if row["relative_elevation_r50_m"] >= 4.0 and (
        row["building_dist_m"] <= 25.0 or row["building_tri_density_r25_ha"] > 25.0
    ):
        return "covered_high_ground"
    if row["relative_elevation_r50_m"] >= 4.0:
        return "open_high_ground"
    if row["relative_elevation_r50_m"] <= -4.0:
        return "open_low_ground"
    if row["building_dist_m"] <= 25.0 or row["building_tri_density_r25_ha"] > 25.0:
        return "compound_movement"
    if row["road_or_runway_dist_m"] <= 12.0 and row["terrain_slope_deg"] < 12.0:
        return "corridor_movement"
    return "neutral_or_unclassified"


def sanitize(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.floating):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.integer):
        return int(value)
    return value


def numeric_summary(
    rows: list[dict[str, Any]], columns: list[str]
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for column in columns:
        values = []
        for row in rows:
            value = row.get(column)
            if isinstance(value, (float, int, np.floating, np.integer)) and math.isfinite(
                float(value)
            ):
                values.append(float(value))
        if not values:
            summary[column] = {
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
                "mean": None,
            }
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[column] = {
            "min": float(np.min(arr)),
            "p25": float(np.quantile(arr, 0.25)),
            "median": float(np.median(arr)),
            "p75": float(np.quantile(arr, 0.75)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-cache", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--xy-margin", type=float, default=180.0)
    parser.add_argument("--z-margin", type=float, default=80.0)
    parser.add_argument("--crop-bounds-x", type=parse_range)
    parser.add_argument("--crop-bounds-y", type=parse_range)
    parser.add_argument("--crop-bounds-z", type=parse_range)
    parser.add_argument("--max-motion-dt-s", type=float, default=30.0)
    parser.add_argument("--life-gap-s", type=float, default=30.0)
    parser.add_argument("--life-displacement-m", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
    if len(rows) < 2:
        raise SystemExit("Need at least two trajectory rows.")
    compute_motion(
        rows,
        max_motion_dt_s=args.max_motion_dt_s,
        life_gap_s=args.life_gap_s,
        life_displacement_m=args.life_displacement_m,
    )

    explicit_bounds = {
        axis: getattr(args, f"crop_bounds_{axis}")
        for axis in ("x", "y", "z")
        if getattr(args, f"crop_bounds_{axis}") is not None
    }
    bounds = explicit_bounds or trajectory_bounds(rows, args.xy_margin, args.z_margin)

    with args.mesh_cache.open("rb") as handle:
        payload = pickle.load(handle)

    tris_by_category = {
        category: crop_tris(array_to_tris(payload["triangles"].get(category, array("f"))), bounds)
        for category in CATEGORIES
    }
    terrain = TerrainIndex(tris_by_category["terrain"])
    indexes = {
        category: CategoryIndex(tris_by_category[category])
        for category in ("road", "runway", "building")
    }

    feature_rows: list[dict[str, Any]] = []
    previous_terrain_z = math.nan
    previous_row: dict[str, Any] | None = None
    for row in rows:
        point_xy = np.asarray([row["x"], row["y"]], dtype=np.float32)
        terrain_values = terrain.surface_at(point_xy)
        relief25, mean25, rugged25 = terrain.relief(point_xy, 25.0)
        relief50, mean50, rugged50 = terrain.relief(point_xy, 50.0)
        relief100, mean100, rugged100 = terrain.relief(point_xy, 100.0)
        terrain_z = float(terrain_values["terrain_z_m"])
        height_above = row["z"] - terrain_z
        if (
            previous_row is not None
            and not row.get("motion_gap_break", False)
            and not math.isnan(previous_terrain_z)
        ):
            step = max(float(row.get("horizontal_step_m", math.nan)), 1e-6)
            forward_grade = (terrain_z - previous_terrain_z) / step
        else:
            forward_grade = math.nan

        road_dist = indexes["road"].distance(point_xy)
        runway_dist = indexes["runway"].distance(point_xy)
        building_dist = indexes["building"].distance(point_xy)
        out = {
            "t": row["t"],
            "x": row["x"],
            "y": row["y"],
            "z": row["z"],
            "alive_state": row.get("alive_state"),
            "dt_s": row.get("dt_s", math.nan),
            "horizontal_step_m": row.get("horizontal_step_m", math.nan),
            "vertical_step_m": row.get("vertical_step_m", math.nan),
            "life_segment_id": row.get("life_segment_id", 0),
            "motion_gap_break": row.get("motion_gap_break", False),
            "redeploy_gap_candidate": row.get("redeploy_gap_candidate", False),
            "speed_mps": row.get("speed_mps", math.nan),
            "heading_deg": row.get("heading_deg", math.nan),
            "turn_rate_deg_s": row.get("turn_rate_deg_s", math.nan),
            "terrain_z_m": terrain_z,
            "height_above_terrain_m": height_above,
            "terrain_slope_deg": float(terrain_values["terrain_slope_deg"]),
            "terrain_lookup_inside_tri": bool(terrain_values["terrain_lookup_inside_tri"]),
            "terrain_lookup_xy_dist_m": float(terrain_values["terrain_lookup_xy_dist_m"]),
            "local_relief_r25_m": relief25,
            "local_relief_r50_m": relief50,
            "local_relief_r100_m": relief100,
            "terrain_ruggedness_r25_m": rugged25,
            "terrain_ruggedness_r50_m": rugged50,
            "terrain_ruggedness_r100_m": rugged100,
            "relative_elevation_r25_m": terrain_z - mean25 if not math.isnan(mean25) else math.nan,
            "relative_elevation_r50_m": terrain_z - mean50 if not math.isnan(mean50) else math.nan,
            "relative_elevation_r100_m": terrain_z - mean100
            if not math.isnan(mean100)
            else math.nan,
            "forward_grade": forward_grade,
            "road_dist_m": road_dist,
            "runway_dist_m": runway_dist,
            "road_or_runway_dist_m": min(road_dist, runway_dist),
            "building_dist_m": building_dist,
            "building_tri_density_r25_ha": indexes["building"].density_per_ha(point_xy, 25.0),
            "building_tri_density_r50_ha": indexes["building"].density_per_ha(point_xy, 50.0),
            "cover_building_r10": building_dist <= 10.0,
            "cover_building_r25": building_dist <= 25.0,
            "on_corridor_proxy": min(road_dist, runway_dist) <= 12.0,
            "high_ground_local_r50": (terrain_z - mean50) >= 4.0
            if not math.isnan(mean50)
            else False,
            "high_ground_local_r100": (terrain_z - mean100) >= 6.0
            if not math.isnan(mean100)
            else False,
        }
        out["tactical_state"] = tactical_state(out)
        out["tactical_state_v0"] = out["tactical_state"]
        feature_rows.append(out)
        previous_terrain_z = terrain_z
        previous_row = row

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(feature_rows[0].keys())
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows:
            writer.writerow({key: sanitize(value) for key, value in row.items()})

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                [{key: sanitize(value) for key, value in row.items()} for row in feature_rows],
                indent=2,
            ),
            encoding="utf-8",
        )

    state_counts: dict[str, int] = {}
    for row in feature_rows:
        state = row["tactical_state"]
        state_counts[state] = state_counts.get(state, 0) + 1
    motion_gap_rows = sum(1 for row in feature_rows if row["motion_gap_break"])
    redeploy_gap_rows = sum(1 for row in feature_rows if row["redeploy_gap_candidate"])
    terrain_inside_ratio = (
        sum(1 for row in feature_rows if row["terrain_lookup_inside_tri"]) / len(feature_rows)
        if feature_rows
        else math.nan
    )
    summary_path = args.output_summary or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mesh_cache": str(args.mesh_cache.resolve()),
        "trajectory_json": str(args.trajectory_json.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "output_json": str(args.output_json.resolve()) if args.output_json is not None else None,
        "feature_rows": len(feature_rows),
        "time_range_s": [float(feature_rows[0]["t"]), float(feature_rows[-1]["t"])],
        "crop_bounds": {axis: [float(lo), float(hi)] for axis, (lo, hi) in bounds.items()},
        "triangles": {key: int(len(value)) for key, value in tris_by_category.items()},
        "tactical_state_schema": TACTICAL_STATE_SCHEMA,
        "tactical_state_counts": state_counts,
        "motion_gap_rows": motion_gap_rows,
        "redeploy_gap_candidate_rows": redeploy_gap_rows,
        "life_segments": int(max(row["life_segment_id"] for row in feature_rows) + 1),
        "terrain_lookup_inside_tri_ratio": terrain_inside_ratio,
        "motion_gap_policy": {
            "max_motion_dt_s": args.max_motion_dt_s,
            "life_gap_s": args.life_gap_s,
            "life_displacement_m": args.life_displacement_m,
        },
        "numeric_summary": numeric_summary(
            feature_rows,
            [
                "dt_s",
                "horizontal_step_m",
                "vertical_step_m",
                "height_above_terrain_m",
                "terrain_slope_deg",
                "local_relief_r25_m",
                "local_relief_r50_m",
                "local_relief_r100_m",
                "terrain_ruggedness_r25_m",
                "terrain_ruggedness_r50_m",
                "terrain_ruggedness_r100_m",
                "relative_elevation_r50_m",
                "relative_elevation_r100_m",
                "road_or_runway_dist_m",
                "building_dist_m",
                "building_tri_density_r25_ha",
                "forward_grade",
                "speed_mps",
                "turn_rate_deg_s",
            ],
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"feature_rows={len(feature_rows)}")
    print(f"output_csv={args.output_csv.resolve()}")
    if args.output_json is not None:
        print(f"output_json={args.output_json.resolve()}")
    print(f"output_summary={summary_path.resolve()}")
    print(f"crop_bounds={bounds}")
    print(
        "triangles="
        + json.dumps(
            {key: int(len(value)) for key, value in tris_by_category.items()}, sort_keys=True
        )
    )
    print("tactical_state_counts=" + json.dumps(state_counts, sort_keys=True))
    print(f"motion_gap_rows={motion_gap_rows}")
    print(f"redeploy_gap_candidate_rows={redeploy_gap_rows}")


if __name__ == "__main__":
    main()
