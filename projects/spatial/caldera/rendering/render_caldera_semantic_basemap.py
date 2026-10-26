"""Render a semantic top-down Caldera basemap from local OpenUSD assets.

The Activision Caldera release does not include a single paper-ready minimap
texture in the downloaded repository, and the local macOS USD viewport renderer
is not stable for this full scene.  This script therefore uses the strongest
local signal we do have: real USD mesh geometry plus path-level asset semantics.

It builds a reproducible 2D basemap by extracting sampled triangles from the
root Caldera USD stage, classifying them into terrain, coast, water, roads,
runway, buildings, vegetation, and props, then rasterizing those layers into a
satellite-like map image.

Extraction uses `.venv/bin/python` because that environment contains `pxr`.
Rendering uses the normal project/system Python because it has numpy/Pillow/cv2.
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
import datetime as dt
import json
import math
import pickle
import random
import re
import subprocess
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = REPO_ROOT
REPO = ROOT / "data" / "br_l0" / "caldera" / "repo"
ROOT_STAGE = REPO / "map_source" / "mp_wz_island.usd"
GEO_ROOT = "/world/mp_wz_island/mp_wz_island_paths/mp_wz_island_geo"
DEFAULT_OUT_DIR = ROOT / "reports" / "caldera_3d" / "semantic_basemap"

TERRAIN_BRANCHES = [
    "st_b",
    "st_d",
    "st_c",
    "st_f",
    "st_e",
    "st_h",
    "st_g",
    "st_j",
    "st_i",
    "st_l",
    "st_k",
    "st_n",
    "st_p",
    "st_o",
]

DISTRICT_BRANCHES = [
    "map_phosphate_mine",
    "map_tile_p",
    "map_tile_d",
    "map_tile_c",
    "map_tile_e",
    "map_tile_f",
    "map_tile_h",
    "map_tile_j",
    "map_tile_k",
    "map_tile_l",
    "map_tile_o",
    "map_ruins",
    "map_vista",
    "map_village",
    "map_tile_i",
    "map_tile_b",
    "map_airstrip",
    "map_docks",
    "map_tile_g",
    "map_beachhead",
    "map_subpen",
    "map_tile_n",
    "map_agricultural_center",
    "map_airfield",
    "map_arsenal",
    "map_capital",
    "map_caldera",
]

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

DEFAULT_LIMITS = {
    "terrain": 360_000,
    "coast_beach": 80_000,
    "water": 35_000,
    "vegetation": 95_000,
    "road": 150_000,
    "runway": 40_000,
    "building": 230_000,
    "prop": 45_000,
}

DEFAULT_FACE_STRIDES = {
    "terrain": 7,
    "coast_beach": 5,
    "water": 3,
    "vegetation": 24,
    "road": 4,
    "runway": 2,
    "building": 8,
    "prop": 32,
}

DEFAULT_MAX_FACES_PER_MESH = {
    "terrain": 0,
    "coast_beach": 120,
    "water": 120,
    "vegetation": 24,
    "road": 160,
    "runway": 240,
    "building": 90,
    "prop": 18,
}

EXCLUDE_SUBSTRINGS = [
    "/loot/",
    "/audio/",
    "/lighting",
    "/objectives",
    "/ambient",
    "/br_loot_",
    "client_trigger",
    "trigger_",
    "volume_",
    "collision",
    "outofbounds",
    "vehicle_spawns",
    "lightgrid",
    "physicsbroadphase",
]

ROAD_TOKENS = {
    "road",
    "roads",
    "street",
    "streets",
    "asphalt",
    "highway",
    "lane",
    "lanes",
    "sidewalk",
    "curb",
    "bridge",
    "bridges",
    "rail",
    "rails",
    "railroad",
    "tram",
}

RUNWAY_TOKENS = {"runway", "runways", "airstrip"}

WATER_TOKENS = {"water", "ocean", "sea", "lagoon", "river", "pond", "stream"}
WATER_FALSE_POSITIVE_TOKENS = {
    "tank",
    "tanks",
    "tower",
    "pipe",
    "pipes",
    "pump",
    "barrel",
    "barrels",
}

COAST_TOKENS = {"beach", "beaches", "coast", "coastal", "shore", "shoreline", "riverbed", "sand"}

VEGETATION_TOKENS = {
    "foliage",
    "tree",
    "trees",
    "grass",
    "plant",
    "plants",
    "palm",
    "palms",
    "bush",
    "bushes",
    "shrub",
    "shrubs",
    "clutter",
    "alocasia",
    "broadleaf",
    "jungle",
    "vine",
    "vines",
}

BUILDING_TOKENS = {
    "building",
    "buildings",
    "house",
    "houses",
    "roof",
    "roofs",
    "city",
    "hotel",
    "hospital",
    "bank",
    "station",
    "arsenal",
    "dock",
    "docks",
    "subpen",
    "factory",
    "warehouse",
    "hangar",
    "bunker",
    "capital",
    "villa",
    "hut",
    "barracks",
    "garage",
    "tower",
    "wall",
    "walls",
    "window",
    "door",
    "stairs",
    "stair",
    "floor",
    "platform",
    "scaffold",
    "container",
    "containers",
}

TERRAIN_TOKENS = {
    "terrain",
    "super",
    "height",
    "ground",
    "soil",
    "mud",
    "dirt",
    "rock",
    "rocks",
    "cliff",
    "cliffs",
    "worldspawn",
    "brush",
    "patch",
}


@dataclass
class Bounds:
    min_x: float = math.inf
    min_y: float = math.inf
    min_z: float = math.inf
    max_x: float = -math.inf
    max_y: float = -math.inf
    max_z: float = -math.inf

    @property
    def valid(self) -> bool:
        return self.min_x < math.inf

    def update(self, point: tuple[float, float, float]) -> None:
        self.min_x = min(self.min_x, point[0])
        self.min_y = min(self.min_y, point[1])
        self.min_z = min(self.min_z, point[2])
        self.max_x = max(self.max_x, point[0])
        self.max_y = max(self.max_y, point[1])
        self.max_z = max(self.max_z, point[2])

    def as_dict(self) -> dict[str, list[float] | None]:
        if not self.valid:
            return {"x": None, "y": None, "z": None}
        return {
            "x": [self.min_x, self.max_x],
            "y": [self.min_y, self.max_y],
            "z": [self.min_z, self.max_z],
        }


@dataclass(frozen=True)
class SourceSpec:
    group: str
    root_path: str

    @property
    def label(self) -> str:
        return self.root_path.rstrip("/").split("/")[-1]


def tokenize_path(path: str) -> set[str]:
    return {part for part in re.split(r"[^a-z]+", path.lower()) if part}


def has_any(tokens: set[str], options: Iterable[str]) -> bool:
    return bool(tokens.intersection(options))


def classify_path(path: str, group: str, source_label: str) -> str | None:
    lower = path.lower()
    if any(token in lower for token in EXCLUDE_SUBSTRINGS):
        return None

    tokens = tokenize_path(lower)
    source_tokens = tokenize_path(source_label)

    if has_any(tokens, RUNWAY_TOKENS):
        return "runway"
    if has_any(tokens, ROAD_TOKENS):
        return "road"
    if has_any(tokens, WATER_TOKENS) and not has_any(tokens, WATER_FALSE_POSITIVE_TOKENS):
        return "water"
    if has_any(tokens, COAST_TOKENS):
        return "coast_beach"
    if has_any(tokens, VEGETATION_TOKENS):
        return "vegetation"
    if has_any(tokens, BUILDING_TOKENS) or has_any(source_tokens, BUILDING_TOKENS):
        return "building"
    if group == "district" and has_any(tokens, {"worldspawn", "brush", "patch", "geo"}):
        return "building"
    if group == "terrain" and has_any(tokens, TERRAIN_TOKENS):
        return "terrain"
    if group == "terrain":
        return "prop"
    if group == "district":
        return "building"
    return "prop"


def discover_sources(groups: set[str]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    if "terrain" in groups:
        for branch in TERRAIN_BRANCHES:
            sources.append(SourceSpec("terrain", f"{GEO_ROOT}/st_main/{branch}"))
    if "districts" in groups:
        for branch in DISTRICT_BRANCHES:
            sources.append(SourceSpec("district", f"{GEO_ROOT}/{branch}"))
    return sources


def ensure_pxr_runtime() -> None:
    try:
        import pxr  # noqa: F401

        return
    except Exception:
        pass

    venv_python = ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit("OpenUSD pxr bindings are required, and .venv/bin/python was not found.")
    cmd = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def set_variant_sets(root_prim, new_state: str) -> int:
    from pxr import Usd

    if new_state == "none":
        return 0
    changed = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.HasVariantSets():
            continue
        variant_sets = prim.GetVariantSets()
        for variant_name in variant_sets.GetNames():
            if "terrain" not in variant_name.lower() and "district" not in variant_name.lower():
                continue
            variant_set = variant_sets.GetVariantSet(variant_name)
            if new_state not in variant_set.GetVariantNames():
                continue
            before = variant_sets.GetVariantSelection(variant_name)
            if before != new_state:
                variant_set.SetVariantSelection(new_state)
                changed += 1
    return changed


def triangle_area(
    tri: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> float:
    ax = tri[1][0] - tri[0][0]
    ay = tri[1][1] - tri[0][1]
    az = tri[1][2] - tri[0][2]
    bx = tri[2][0] - tri[0][0]
    by = tri[2][1] - tri[0][1]
    bz = tri[2][2] - tri[0][2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def parse_category_options(
    args: argparse.Namespace,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    limits = {category: int(getattr(args, f"{category}_triangles")) for category in CATEGORIES}
    strides = {category: int(getattr(args, f"{category}_face_stride")) for category in CATEGORIES}
    max_faces = {
        category: int(getattr(args, f"{category}_max_faces_per_mesh")) for category in CATEGORIES
    }
    return limits, strides, max_faces


def add_triangle_to_reservoir(
    reservoirs: dict[str, array],
    seen: dict[str, int],
    selected: dict[str, int],
    limits: dict[str, int],
    category: str,
    tri: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    rng: random.Random,
) -> None:
    limit = limits[category]
    if limit <= 0:
        return
    flat = [float(coord) for point in tri for coord in point]
    seen[category] += 1
    if selected[category] < limit:
        reservoirs[category].extend(flat)
        selected[category] += 1
        return

    replacement = rng.randrange(seen[category])
    if replacement >= limit:
        return
    start = replacement * 9
    reservoirs[category][start : start + 9] = array("f", flat)


def extract_semantic_cache(args: argparse.Namespace) -> None:
    ensure_pxr_runtime()
    from pxr import Usd, UsdGeom

    limits, strides, max_faces_per_mesh = parse_category_options(args)
    groups = {part.strip() for part in args.groups.split(",") if part.strip()}
    sources = discover_sources(groups)
    if args.limit_sources:
        sources = sources[: args.limit_sources]
    if not sources:
        raise SystemExit(f"No sources discovered for groups={sorted(groups)}")

    stage = Usd.Stage.Open(str(ROOT_STAGE))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {ROOT_STAGE}")

    scale = float(stage.GetMetadata("metersPerUnit") or 0.0254)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rng = random.Random(args.seed)

    reservoirs = {category: array("f") for category in CATEGORIES}
    seen = {category: 0 for category in CATEGORIES}
    selected = {category: 0 for category in CATEGORIES}
    mesh_seen = {category: 0 for category in CATEGORIES}
    mesh_used = {category: 0 for category in CATEGORIES}
    category_bounds = {category: Bounds() for category in CATEGORIES}
    total_bounds = Bounds()
    source_manifest = []

    total_meshes_seen = 0
    skipped_by_path = 0
    skipped_by_purpose = 0
    skipped_empty = 0

    for source_idx, source in enumerate(sources, start=1):
        root_prim = stage.GetPrimAtPath(source.root_path)
        source_info = {
            "group": source.group,
            "root_path": source.root_path,
            "missing": False,
            "variant_sets_changed": 0,
            "mesh_seen": 0,
            "mesh_used": 0,
            "candidate_triangles_seen_by_category": {category: 0 for category in CATEGORIES},
            "selected_triangles_by_category": {category: 0 for category in CATEGORIES},
        }
        if not root_prim or not root_prim.IsValid():
            source_info["missing"] = True
            source_manifest.append(source_info)
            print(f"[{source_idx}/{len(sources)}] missing {source.root_path}", flush=True)
            continue

        source_info["variant_sets_changed"] = set_variant_sets(root_prim, args.variant)
        print(f"[{source_idx}/{len(sources)}] scanning {source.label}", flush=True)

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            total_meshes_seen += 1
            source_info["mesh_seen"] += 1
            prim_path = prim.GetPath().pathString
            category = classify_path(prim_path, source.group, source.label)
            if category is None:
                skipped_by_path += 1
                continue
            if limits[category] <= 0:
                continue

            imageable = UsdGeom.Imageable(prim)
            purpose = str(imageable.ComputePurpose())
            if purpose == "render" and args.exclude_render_purpose:
                skipped_by_purpose += 1
                continue
            if purpose == "guide" and args.exclude_guide_purpose:
                skipped_by_purpose += 1
                continue

            mesh = UsdGeom.Mesh(prim)
            local_points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not local_points or not counts or not indices:
                skipped_empty += 1
                continue

            matrix = xform_cache.GetLocalToWorldTransform(prim)
            world_points = []
            for point in local_points:
                wp = matrix.Transform(point)
                world_points.append(
                    (float(wp[0]) * scale, float(wp[1]) * scale, float(wp[2]) * scale)
                )

            cursor = 0
            mesh_added = 0
            mesh_had_triangle = False
            face_stride = max(1, strides[category])
            mesh_cap = max_faces_per_mesh[category]
            for face_idx, count in enumerate(counts):
                face_indices = indices[cursor : cursor + count]
                cursor += count
                if count < 3:
                    continue
                if face_stride > 1 and face_idx % face_stride != 0:
                    continue
                if mesh_cap and mesh_added >= mesh_cap:
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
                    area = triangle_area(tri)
                    if area < args.min_triangle_area:
                        continue
                    if args.max_triangle_area and area > args.max_triangle_area:
                        continue

                    for point in tri:
                        total_bounds.update(point)
                        category_bounds[category].update(point)
                    add_triangle_to_reservoir(
                        reservoirs, seen, selected, limits, category, tri, rng
                    )
                    source_info["candidate_triangles_seen_by_category"][category] += 1
                    mesh_added += 1
                    mesh_had_triangle = True
                    if mesh_cap and mesh_added >= mesh_cap:
                        break
                if mesh_cap and mesh_added >= mesh_cap:
                    break

            if mesh_had_triangle:
                source_info["mesh_used"] += 1
                mesh_seen[category] += 1
                mesh_used[category] += 1

        for category in CATEGORIES:
            source_info["selected_triangles_by_category"][category] = selected[category]
        source_manifest.append(source_info)
        compact_counts = {
            category: source_info["candidate_triangles_seen_by_category"][category]
            for category in CATEGORIES
            if source_info["candidate_triangles_seen_by_category"][category]
        }
        print(f"    candidates={compact_counts}", flush=True)

    payload = {
        "description": "Reservoir-sampled semantic triangles extracted from Caldera OpenUSD map assets.",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root_stage": str(ROOT_STAGE.relative_to(ROOT)),
        "scale_meters_per_unit": scale,
        "bounds": total_bounds.as_dict(),
        "category_bounds": {
            category: category_bounds[category].as_dict() for category in CATEGORIES
        },
        "categories": CATEGORIES,
        "limits": limits,
        "face_strides": strides,
        "max_faces_per_mesh": max_faces_per_mesh,
        "selected_triangles": selected,
        "candidate_triangles_seen": seen,
        "mesh_seen_by_category": mesh_seen,
        "mesh_used_by_category": mesh_used,
        "total_meshes_seen": total_meshes_seen,
        "skipped_by_path": skipped_by_path,
        "skipped_by_purpose": skipped_by_purpose,
        "skipped_empty": skipped_empty,
        "sources": source_manifest,
        "triangles": reservoirs,
    }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {key: value for key, value in payload.items() if key != "triangles"}
    manifest_path = args.cache.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"cache_written={args.cache}", flush=True)
    print(f"manifest_written={manifest_path}", flush=True)
    print(
        json.dumps({"selected": selected, "seen": seen, "bounds": payload["bounds"]}, indent=2),
        flush=True,
    )


def invoke_extractor(args: argparse.Namespace) -> None:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit("Cannot extract USD geometry because .venv/bin/python was not found.")
    cmd = [
        str(venv_python),
        str(Path(__file__).resolve()),
        "--extract-only",
        "--cache",
        str(args.cache),
        "--out-dir",
        str(args.out_dir),
        "--groups",
        args.groups,
        "--variant",
        args.variant,
        "--seed",
        str(args.seed),
        "--min-triangle-area",
        str(args.min_triangle_area),
        "--max-triangle-area",
        str(args.max_triangle_area),
    ]
    if args.limit_sources:
        cmd.extend(["--limit-sources", str(args.limit_sources)])
    if args.exclude_render_purpose:
        cmd.append("--exclude-render-purpose")
    if args.exclude_guide_purpose:
        cmd.append("--exclude-guide-purpose")
    for category in CATEGORIES:
        cmd.extend(
            [
                f"--{category.replace('_', '-')}-triangles",
                str(getattr(args, f"{category}_triangles")),
            ]
        )
        cmd.extend(
            [
                f"--{category.replace('_', '-')}-face-stride",
                str(getattr(args, f"{category}_face_stride")),
            ]
        )
        cmd.extend(
            [
                f"--{category.replace('_', '-')}-max-faces-per-mesh",
                str(getattr(args, f"{category}_max_faces_per_mesh")),
            ]
        )
    subprocess.run(cmd, check=True)


def load_payload(cache_path: Path) -> dict:
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def parse_color(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        raise argparse.ArgumentTypeError("color must be #RRGGBB")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def color_ramp(t: float) -> tuple[int, int, int]:
    stops = [
        (0.00, (40, 74, 56)),
        (0.24, (67, 100, 54)),
        (0.50, (97, 106, 61)),
        (0.72, (118, 94, 63)),
        (1.00, (147, 143, 122)),
    ]
    t = max(0.0, min(1.0, t))
    for idx in range(len(stops) - 1):
        left_t, left_color = stops[idx]
        right_t, right_color = stops[idx + 1]
        if left_t <= t <= right_t:
            local = (t - left_t) / max(1e-9, right_t - left_t)
            return tuple(
                int(round(left_color[channel] * (1 - local) + right_color[channel] * local))
                for channel in range(3)
            )  # type: ignore[return-value]
    return stops[-1][1]


def array_to_tris(flat: array):
    import numpy as np

    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))


def filter_tris_by_xy_area(tris, max_xy_area: float):
    import numpy as np

    if max_xy_area <= 0 or len(tris) == 0:
        return tris
    ax = tris[:, 1, 0] - tris[:, 0, 0]
    ay = tris[:, 1, 1] - tris[:, 0, 1]
    bx = tris[:, 2, 0] - tris[:, 0, 0]
    by = tris[:, 2, 1] - tris[:, 0, 1]
    xy_area = np.abs(ax * by - ay * bx) * 0.5
    return tris[xy_area <= max_xy_area]


def compute_render_bounds(
    triangles: dict[str, object], payload: dict, args: argparse.Namespace
) -> dict[str, list[float]]:
    import numpy as np

    if args.bounds_mode == "cache":
        return payload["bounds"]

    parts = []
    for category in CATEGORIES:
        tri = triangles[category]
        if len(tri):
            parts.append(tri.reshape((-1, 3)))
    if not parts:
        return payload["bounds"]

    points = np.concatenate(parts, axis=0)
    trim = float(args.bounds_trim_percentile)
    if args.bounds_mode == "robust" and trim > 0:
        low_q = trim
        high_q = 100.0 - trim
        min_x, max_x = np.percentile(points[:, 0], [low_q, high_q])
        min_y, max_y = np.percentile(points[:, 1], [low_q, high_q])
    else:
        min_x = float(points[:, 0].min())
        max_x = float(points[:, 0].max())
        min_y = float(points[:, 1].min())
        max_y = float(points[:, 1].max())

    if args.square_bounds:
        span_x = max_x - min_x
        span_y = max_y - min_y
        span = max(span_x, span_y)
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0
        min_x = mid_x - span / 2.0
        max_x = mid_x + span / 2.0
        min_y = mid_y - span / 2.0
        max_y = mid_y + span / 2.0

    return {
        "x": [float(min_x), float(max_x)],
        "y": [float(min_y), float(max_y)],
        "z": payload["bounds"].get("z"),
    }


def projected_points(tris, bounds: dict, width: int, height: int, padding: int):
    import numpy as np

    min_x, max_x = bounds["x"]
    min_y, max_y = bounds["y"]
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
    draw_w = span_x * scale
    draw_h = span_y * scale
    left = (width - draw_w) / 2.0
    top = (height - draw_h) / 2.0

    px = left + (tris[:, :, 0] - min_x) * scale
    py = top + (max_y - tris[:, :, 1]) * scale
    pts = np.stack([px, py], axis=2)
    return np.rint(pts).astype(np.int32)


def project_xy(points, bounds: dict, width: int, height: int, padding: int):
    import numpy as np

    min_x, max_x = bounds["x"]
    min_y, max_y = bounds["y"]
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
    draw_w = span_x * scale
    draw_h = span_y * scale
    left = (width - draw_w) / 2.0
    top = (height - draw_h) / 2.0

    arr = np.asarray(points, dtype=np.float32)
    px = left + (arr[:, 0] - min_x) * scale
    py = top + (max_y - arr[:, 1]) * scale
    return np.rint(np.stack([px, py], axis=1)).astype(np.int32)


def fill_polygons(
    canvas, pts, color: tuple[int, int, int], alpha: float, mask_accum=None, chunk_size: int = 5000
) -> None:
    import cv2
    import numpy as np

    if len(pts) == 0:
        return
    color_arr = np.array(color, dtype=np.float32)
    for start in range(0, len(pts), chunk_size):
        chunk = pts[start : start + chunk_size]
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, chunk, 255)
        selected = mask > 0
        if not selected.any():
            continue
        if mask_accum is not None:
            mask_accum[selected] = 255
        if alpha >= 0.999:
            canvas[selected] = color_arr.astype(np.uint8)
        else:
            blended = canvas[selected].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
            canvas[selected] = np.clip(blended, 0, 255).astype(np.uint8)


def apply_hillshade(canvas, elevation, land_mask) -> None:
    import cv2
    import numpy as np

    if not (land_mask > 0).any():
        return
    blurred = cv2.GaussianBlur(elevation.astype(np.float32), (0, 0), sigmaX=5.5, sigmaY=5.5)
    grad_y, grad_x = np.gradient(blurred / 255.0)
    shade = 0.96 - 0.72 * grad_x + 0.45 * grad_y
    shade = cv2.GaussianBlur(shade, (0, 0), sigmaX=2.0, sigmaY=2.0)
    shade = np.clip(shade, 0.74, 1.08)
    selected = land_mask > 0
    shaded = canvas[selected].astype(np.float32) * shade[selected, None]
    canvas[selected] = np.clip(shaded, 0, 255).astype(np.uint8)


def draw_contours(canvas, mask, color: tuple[int, int, int], thickness: int) -> None:
    import cv2

    if not (mask > 0).any():
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness=thickness, lineType=cv2.LINE_AA)


def fetch_trajectory_rows(args: argparse.Namespace, render_bounds: dict) -> list[dict[str, Any]]:
    from psycopg2.extras import RealDictCursor
    from pubg_lab.data.stdb.connection.conn import connect

    x0, x1 = render_bounds["x"]
    y0, y1 = render_bounds["y"]
    extra_filters = []
    params: list[Any] = [args.source_artifact_id, x0, x1, y0, y1]
    if args.sample_t_min is not None:
        extra_filters.append("sample_t >= %s")
        params.append(args.sample_t_min)
    if args.sample_t_max is not None:
        extra_filters.append("sample_t <= %s")
        params.append(args.sample_t_max)
    extra_sql = ""
    if extra_filters:
        extra_sql = " AND " + " AND ".join(extra_filters)

    with connect() as conn:
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
                        WHERE t.source_artifact_id = %s
                          AND t.pos_x_m BETWEEN %s AND %s
                          AND t.pos_y_m BETWEEN %s AND %s
                          AND t.pos_x_m IS NOT NULL
                          AND t.pos_y_m IS NOT NULL
                          {extra_sql.replace("sample_t", "t.sample_t")}
                          {identity_sql}
                    )
                    SELECT br_match_id, br_agent_id, sample_t, pos_x_m, pos_y_m, pos_z_m, alive_state
                    FROM ranked
                    WHERE (rn - 1) %% %s = 0
                    ORDER BY br_match_id, br_agent_id, sample_t
                    LIMIT %s
                    """,
                    tuple(
                        params
                        + identity_params
                        + [args.trajectory_stride, args.trajectory_point_limit]
                    ),
                )
                return list(cur.fetchall())

            cur.execute(
                f"""
                WITH region_samples AS (
                    SELECT br_match_id, br_agent_id, count(*) AS n
                    FROM br_ground.trajectory_samples
                    WHERE source_artifact_id = %s
                      AND pos_x_m BETWEEN %s AND %s
                      AND pos_y_m BETWEEN %s AND %s
                      AND pos_x_m IS NOT NULL
                      AND pos_y_m IS NOT NULL
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
                    WHERE t.source_artifact_id = %s
                      AND t.pos_x_m BETWEEN %s AND %s
                      AND t.pos_y_m BETWEEN %s AND %s
                      {extra_sql.replace("sample_t", "t.sample_t")}
                )
                SELECT br_match_id, br_agent_id, sample_t, pos_x_m, pos_y_m, pos_z_m, alive_state
                FROM ranked
                WHERE (rn - 1) %% %s = 0
                ORDER BY br_match_id, br_agent_id, sample_t
                LIMIT %s
                """,
                tuple(
                    params
                    + [args.players]
                    + [args.source_artifact_id, x0, x1, y0, y1]
                    + [value for value in params[5:]]
                    + [args.trajectory_stride, args.trajectory_point_limit]
                ),
            )
            return list(cur.fetchall())


def choose_snapshot_t(rows: list[dict[str, Any]], args: argparse.Namespace) -> float | None:
    if args.snapshot_t is not None:
        return float(args.snapshot_t)
    if not rows:
        return None
    times = sorted(float(row["sample_t"]) for row in rows)
    idx = max(0, min(len(times) - 1, int(round((len(times) - 1) * args.snapshot_percentile))))
    return times[idx]


def draw_trajectory_overlay(
    canvas, rows: list[dict[str, Any]], render_bounds: dict, args: argparse.Namespace
) -> dict[str, Any]:
    import cv2

    snapshot_t = choose_snapshot_t(rows, args)
    if snapshot_t is None:
        return {"enabled": True, "rows": 0, "agents": 0, "snapshot_t": None}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["br_agent_id"]), []).append(row)

    palette = [
        (255, 196, 36),
        (27, 174, 234),
        (239, 71, 111),
        (78, 205, 196),
        (255, 127, 17),
        (131, 90, 241),
        (61, 220, 151),
        (246, 81, 29),
        (180, 225, 50),
        (255, 255, 255),
    ]

    agents_drawn = 0
    points_drawn = 0
    for idx, (agent_id, agent_rows) in enumerate(sorted(grouped.items())):
        visible_rows = [
            row
            for row in sorted(agent_rows, key=lambda item: float(item["sample_t"]))
            if float(row["sample_t"]) <= snapshot_t
        ]
        if not visible_rows:
            continue
        points = [(float(row["pos_x_m"]), float(row["pos_y_m"])) for row in visible_rows]
        projected = project_xy(points, render_bounds, args.width, args.height, args.padding)
        color = palette[idx % len(palette)]
        if len(projected) >= 2:
            cv2.polylines(
                canvas,
                [projected],
                isClosed=False,
                color=color,
                thickness=args.trajectory_thickness,
                lineType=cv2.LINE_AA,
            )
        current = tuple(int(value) for value in projected[-1])
        cv2.circle(
            canvas, current, args.agent_radius + 2, (12, 15, 15), thickness=-1, lineType=cv2.LINE_AA
        )
        cv2.circle(canvas, current, args.agent_radius, color, thickness=-1, lineType=cv2.LINE_AA)
        agents_drawn += 1
        points_drawn += len(projected)

    return {
        "enabled": True,
        "rows": len(rows),
        "agents": agents_drawn,
        "points": points_drawn,
        "snapshot_t": snapshot_t,
    }


def render_semantic_basemap(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    payload = load_payload(args.cache)
    if not payload["bounds"]["x"] or not payload["bounds"]["y"]:
        raise RuntimeError(f"Cache has no drawable bounds: {args.cache}")

    width = args.width
    height = args.height
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.array(parse_color(args.background), dtype=np.uint8)

    land_mask = np.zeros((height, width), dtype=np.uint8)
    road_mask = np.zeros((height, width), dtype=np.uint8)
    building_mask = np.zeros((height, width), dtype=np.uint8)
    elevation = np.zeros((height, width), dtype=np.float32)

    triangles = {
        category: filter_tris_by_xy_area(
            array_to_tris(payload["triangles"][category]), args.render_max_xy_area
        )
        for category in CATEGORIES
    }
    render_bounds = compute_render_bounds(triangles, payload, args)
    projected = {
        category: projected_points(triangles[category], render_bounds, width, height, args.padding)
        for category in CATEGORIES
    }

    terrain = triangles["terrain"]
    if len(terrain):
        z = terrain[:, :, 2].mean(axis=1)
        low, high = np.percentile(z, [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        bins = np.linspace(low, high, args.terrain_bins + 1)
        z_norm = np.clip((z - low) / (high - low), 0.0, 1.0)
        for bin_idx in range(args.terrain_bins):
            in_bin = (z >= bins[bin_idx]) & (
                z <= bins[bin_idx + 1]
                if bin_idx == args.terrain_bins - 1
                else z < bins[bin_idx + 1]
            )
            if not in_bin.any():
                continue
            color = color_ramp((bin_idx + 0.5) / args.terrain_bins)
            pts = projected["terrain"][in_bin]
            fill_polygons(canvas, pts, color, 1.0, land_mask, args.chunk_size)
            elev_value = int(round(40 + 210 * float(np.median(z_norm[in_bin]))))
            elev_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(elev_mask, pts, 255)
            elevation[elev_mask > 0] = elev_value

    if args.fill_land_holes and (land_mask > 0).any():
        kernel = np.ones((args.land_close_kernel, args.land_close_kernel), dtype=np.uint8)
        closed_land = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed_land = cv2.dilate(closed_land, kernel, iterations=1)
        base_only = (closed_land > 0) & (land_mask == 0)
        canvas[base_only] = np.array((58, 76, 54), dtype=np.uint8)
        if (elevation > 0).any():
            elevation[base_only] = float(np.median(elevation[elevation > 0]))
        land_mask[closed_land > 0] = 255

    apply_hillshade(canvas, elevation, land_mask)

    fill_polygons(
        canvas, projected["coast_beach"], (188, 166, 105), 0.34, land_mask, args.chunk_size
    )
    fill_polygons(canvas, projected["water"], (31, 105, 124), 0.58, None, args.chunk_size)
    fill_polygons(canvas, projected["vegetation"], (26, 84, 42), 0.42, None, args.chunk_size)
    fill_polygons(canvas, projected["road"], (178, 181, 170), 0.62, road_mask, args.chunk_size)
    fill_polygons(canvas, projected["runway"], (73, 80, 78), 0.92, road_mask, args.chunk_size)
    fill_polygons(canvas, projected["building"], (41, 46, 45), 0.88, building_mask, args.chunk_size)
    fill_polygons(canvas, projected["prop"], (102, 101, 88), 0.45, None, args.chunk_size)

    if args.contours:
        draw_contours(canvas, land_mask, (230, 232, 219), args.coastline_thickness)

    if args.internal_edges:
        road_edges = cv2.Canny(road_mask, 60, 160)
        building_edges = cv2.Canny(building_mask, 70, 180)
        canvas[road_edges > 0] = (210, 213, 202)
        canvas[building_edges > 0] = (20, 24, 24)

    trajectory_info = {"enabled": False}
    if args.overlay_trajectories:
        rows = fetch_trajectory_rows(args, render_bounds)
        trajectory_info = draw_trajectory_overlay(canvas, rows, render_bounds, args)

    if args.title:
        img = Image.fromarray(canvas)
    else:
        img = Image.fromarray(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)

    render_manifest = {
        "output": str(args.output.relative_to(ROOT)),
        "cache": str(args.cache.relative_to(ROOT)),
        "width": width,
        "height": height,
        "source_cache_bounds": payload["bounds"],
        "render_bounds": render_bounds,
        "bounds_mode": args.bounds_mode,
        "selected_triangles": payload["selected_triangles"],
        "candidate_triangles_seen": payload["candidate_triangles_seen"],
        "trajectory_overlay": trajectory_info,
        "colors": {
            "terrain": "elevation ramp",
            "coast_beach": "#bca669",
            "water": "#1f697c",
            "vegetation": "#1a542a",
            "road": "#b2b5aa",
            "runway": "#49504e",
            "building": "#292e2d",
            "prop": "#666558",
        },
        "note": "Semantic basemap reconstructed from geometry/path categories, not an official minimap texture.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(render_manifest, indent=2), encoding="utf-8"
    )
    print(f"rendered={args.output}", flush=True)
    print(json.dumps(render_manifest["selected_triangles"], indent=2), flush=True)


def add_category_args(parser: argparse.ArgumentParser) -> None:
    for category in CATEGORIES:
        dashed = category.replace("_", "-")
        parser.add_argument(f"--{dashed}-triangles", type=int, default=DEFAULT_LIMITS[category])
        parser.add_argument(
            f"--{dashed}-face-stride", type=int, default=DEFAULT_FACE_STRIDES[category]
        )
        parser.add_argument(
            f"--{dashed}-max-faces-per-mesh", type=int, default=DEFAULT_MAX_FACES_PER_MESH[category]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_OUT_DIR / "caldera_semantic_triangles.pkl"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUT_DIR / "caldera_semantic_basemap.png"
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--groups", default="terrain,districts")
    parser.add_argument("--variant", default="full", choices=["full", "proxy", "none"])
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--min-triangle-area", type=float, default=0.00004)
    parser.add_argument("--max-triangle-area", type=float, default=0.0)
    parser.add_argument("--exclude-render-purpose", action="store_true")
    parser.add_argument("--exclude-guide-purpose", action="store_true")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=2400)
    parser.add_argument("--padding", type=int, default=80)
    parser.add_argument(
        "--bounds-mode", choices=["selected", "robust", "cache"], default="selected"
    )
    parser.add_argument("--bounds-trim-percentile", type=float, default=0.2)
    parser.add_argument("--square-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-max-xy-area", type=float, default=30_000.0)
    parser.add_argument("--terrain-bins", type=int, default=28)
    parser.add_argument("--fill-land-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--land-close-kernel", type=int, default=13)
    parser.add_argument("--chunk-size", type=int, default=4500)
    parser.add_argument("--background", default="#11191a")
    parser.add_argument("--contours", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coastline-thickness", type=int, default=3)
    parser.add_argument("--internal-edges", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlay-trajectories", action="store_true")
    parser.add_argument("--source-artifact-id", default="caldera_minimal_layers_breadcrumbs_usd")
    parser.add_argument("--players", type=int, default=12)
    parser.add_argument(
        "--match-id", help="Exact br_ground match id to overlay, e.g. caldera:match_31"
    )
    parser.add_argument(
        "--agent-id", help="Exact br_ground agent id to overlay, e.g. caldera:match_31:player_25"
    )
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--trajectory-point-limit", type=int, default=5000)
    parser.add_argument("--snapshot-t", type=float, default=None)
    parser.add_argument("--snapshot-percentile", type=float, default=0.62)
    parser.add_argument("--sample-t-min", type=float, default=None)
    parser.add_argument("--sample-t-max", type=float, default=None)
    parser.add_argument("--trajectory-thickness", type=int, default=3)
    parser.add_argument("--agent-radius", type=int, default=9)
    parser.add_argument("--title", default="")
    add_category_args(parser)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.extract_only:
        extract_semantic_cache(args)
        return
    if args.force_extract or not args.cache.exists():
        invoke_extractor(args)
    render_semantic_basemap(args)


if __name__ == "__main__":
    main()
