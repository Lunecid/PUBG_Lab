"""Extract a local Caldera mesh window directly from the OpenUSD source stage.

Unlike the semantic basemap cache, this keeps local raw triangles from the
source USD scene for a bounded area.  The output is intended for local clay /
wireframe renders around one trajectory segment.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import math
import pickle
import re
import subprocess
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = REPO_ROOT
REPO = ROOT / "data" / "br_l0" / "caldera" / "repo"
ROOT_STAGE = REPO / "map_source" / "mp_wz_island.usd"
GEO_ROOT = "/world/mp_wz_island/mp_wz_island_paths/mp_wz_island_geo"

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

CATEGORIES = ["terrain", "coast_beach", "water", "vegetation", "road", "runway", "building", "prop"]

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

    def update(self, point: tuple[float, float, float]) -> None:
        self.min_x = min(self.min_x, point[0])
        self.min_y = min(self.min_y, point[1])
        self.min_z = min(self.min_z, point[2])
        self.max_x = max(self.max_x, point[0])
        self.max_y = max(self.max_y, point[1])
        self.max_z = max(self.max_z, point[2])

    def as_dict(self) -> dict[str, list[float] | None]:
        if self.min_x == math.inf:
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


def ensure_pxr_runtime() -> None:
    try:
        import pxr  # noqa: F401

        return
    except Exception:
        pass

    venv_python = ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit("OpenUSD pxr bindings are required, and .venv/bin/python was not found.")
    raise SystemExit(
        subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], check=False
        ).returncode
    )


def parse_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    lo, hi = (float(part.strip()) for part in parts)
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be greater than lower bound")
    return lo, hi


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
        sources.extend(
            SourceSpec("terrain", f"{GEO_ROOT}/st_main/{branch}") for branch in TERRAIN_BRANCHES
        )
    if "districts" in groups:
        sources.extend(
            SourceSpec("district", f"{GEO_ROOT}/{branch}") for branch in DISTRICT_BRANCHES
        )
    return sources


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
            if variant_sets.GetVariantSelection(variant_name) != new_state:
                variant_set.SetVariantSelection(new_state)
                changed += 1
    return changed


def overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def bbox_overlaps(
    bbox, scale: float, bx: tuple[float, float], by: tuple[float, float], bz: tuple[float, float]
) -> bool:
    box = bbox.ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    x = (float(mn[0]) * scale, float(mx[0]) * scale)
    y = (float(mn[1]) * scale, float(mx[1]) * scale)
    z = (float(mn[2]) * scale, float(mx[2]) * scale)
    return overlap(x, bx) and overlap(y, by) and overlap(z, bz)


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


def max_edge_length(
    tri: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> float:
    out = 0.0
    for i, j in ((0, 1), (1, 2), (2, 0)):
        dx = tri[i][0] - tri[j][0]
        dy = tri[i][1] - tri[j][1]
        dz = tri[i][2] - tri[j][2]
        out = max(out, math.sqrt(dx * dx + dy * dy + dz * dz))
    return out


def centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def in_bounds(
    point: tuple[float, float, float],
    bx: tuple[float, float],
    by: tuple[float, float],
    bz: tuple[float, float],
) -> bool:
    return bx[0] <= point[0] <= bx[1] and by[0] <= point[1] <= by[1] and bz[0] <= point[2] <= bz[1]


def append_triangle(
    store: array,
    tri: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> None:
    for point in tri:
        store.extend((float(point[0]), float(point[1]), float(point[2])))


def extract(args: argparse.Namespace) -> dict:
    ensure_pxr_runtime()
    from pxr import Usd, UsdGeom

    groups = {part.strip() for part in args.groups.split(",") if part.strip()}
    wanted = {part.strip() for part in args.categories.split(",") if part.strip()}
    sources = discover_sources(groups)
    stage = Usd.Stage.Open(str(ROOT_STAGE))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {ROOT_STAGE}")

    scale = float(stage.GetMetadata("metersPerUnit") or 0.0254)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    triangles = {category: array("f") for category in CATEGORIES}
    triangle_path_ids = {category: array("I") for category in CATEGORIES}
    source_paths: list[str] = []
    path_to_id: dict[str, int] = {}
    bounds = {category: Bounds() for category in CATEGORIES}
    counts = {category: 0 for category in CATEGORIES}
    meshes_used = {category: 0 for category in CATEGORIES}
    meshes_seen = {category: 0 for category in CATEGORIES}
    skipped_bbox = 0
    skipped_category = 0
    skipped_purpose = 0
    variant_changes = 0

    for source_idx, source in enumerate(sources, 1):
        root_prim = stage.GetPrimAtPath(source.root_path)
        if not root_prim or not root_prim.IsValid():
            continue
        variant_changes += set_variant_sets(root_prim, args.variant)
        print(f"[{source_idx}/{len(sources)}] scanning {source.label}", flush=True)
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            prim_path = prim.GetPath().pathString
            category = classify_path(prim_path, source.group, source.label)
            if category not in wanted:
                skipped_category += 1
                continue
            meshes_seen[category] += 1
            try:
                if not bbox_overlaps(
                    bbox_cache.ComputeWorldBound(prim),
                    scale,
                    args.bounds_x,
                    args.bounds_y,
                    args.bounds_z,
                ):
                    skipped_bbox += 1
                    continue
            except Exception:
                pass

            imageable = UsdGeom.Imageable(prim)
            purpose = str(imageable.ComputePurpose())
            if purpose == "guide" or (purpose == "render" and args.exclude_render_purpose):
                skipped_purpose += 1
                continue

            mesh = UsdGeom.Mesh(prim)
            local_points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not local_points or not face_counts or not face_indices:
                continue

            matrix = xform_cache.GetLocalToWorldTransform(prim)
            world_points = []
            for point in local_points:
                wp = matrix.Transform(point)
                world_points.append(
                    (float(wp[0]) * scale, float(wp[1]) * scale, float(wp[2]) * scale)
                )

            path_id = path_to_id.get(prim_path)
            if path_id is None:
                path_id = len(source_paths)
                path_to_id[prim_path] = path_id
                source_paths.append(prim_path)

            cursor = 0
            mesh_added = 0
            stride = (
                args.terrain_face_stride if category == "terrain" else args.nonterrain_face_stride
            )
            limit = args.max_faces_per_mesh
            for face_idx, face_count in enumerate(face_counts):
                indices = face_indices[cursor : cursor + face_count]
                cursor += face_count
                if face_count < 3:
                    continue
                if stride > 1 and face_idx % stride != 0:
                    continue
                if limit and mesh_added >= limit:
                    continue
                face_points = [
                    world_points[index] for index in indices if 0 <= index < len(world_points)
                ]
                if len(face_points) < 3 or not in_bounds(
                    centroid(face_points), args.bounds_x, args.bounds_y, args.bounds_z
                ):
                    continue
                for i in range(1, face_count - 1):
                    try:
                        tri = (
                            world_points[indices[0]],
                            world_points[indices[i]],
                            world_points[indices[i + 1]],
                        )
                    except Exception:
                        continue
                    area = triangle_area(tri)
                    if area < args.min_triangle_area:
                        continue
                    if args.max_triangle_area and area > args.max_triangle_area:
                        continue
                    if args.max_edge_length and max_edge_length(tri) > args.max_edge_length:
                        continue
                    append_triangle(triangles[category], tri)
                    triangle_path_ids[category].append(path_id)
                    for point in tri:
                        bounds[category].update(point)
                    counts[category] += 1
                    mesh_added += 1
                    if limit and mesh_added >= limit:
                        break
            if mesh_added:
                meshes_used[category] += 1

    payload = {
        "description": "Local Caldera mesh window extracted directly from OpenUSD source geometry.",
        "root_stage": str(ROOT_STAGE.relative_to(ROOT)),
        "scale_meters_per_unit": scale,
        "requested_bounds": {
            "x": list(args.bounds_x),
            "y": list(args.bounds_y),
            "z": list(args.bounds_z),
        },
        "categories": CATEGORIES,
        "selected_categories": sorted(wanted),
        "triangles": triangles,
        "triangle_path_ids": triangle_path_ids,
        "source_paths": source_paths,
        "triangle_counts": counts,
        "mesh_seen_by_category": meshes_seen,
        "mesh_used_by_category": meshes_used,
        "category_bounds": {category: bounds[category].as_dict() for category in CATEGORIES},
        "variant": args.variant,
        "variant_sets_changed": variant_changes,
        "skipped_bbox": skipped_bbox,
        "skipped_category": skipped_category,
        "skipped_purpose": skipped_purpose,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"triangles", "triangle_path_ids"}
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounds-x", type=parse_range, required=True)
    parser.add_argument("--bounds-y", type=parse_range, required=True)
    parser.add_argument("--bounds-z", type=parse_range, required=True)
    parser.add_argument("--groups", default="terrain,districts")
    parser.add_argument("--categories", default="terrain,building,road,runway,prop")
    parser.add_argument("--variant", default="full", choices=["full", "proxy", "none"])
    parser.add_argument("--terrain-face-stride", type=int, default=1)
    parser.add_argument("--nonterrain-face-stride", type=int, default=1)
    parser.add_argument("--max-faces-per-mesh", type=int, default=0)
    parser.add_argument("--min-triangle-area", type=float, default=0.0001)
    parser.add_argument("--max-triangle-area", type=float, default=1500.0)
    parser.add_argument("--max-edge-length", type=float, default=180.0)
    parser.add_argument("--exclude-render-purpose", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = extract(args)
    print(f"cache_written={args.output}")
    print(f"manifest_written={args.output.with_suffix('.manifest.json')}")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("requested_bounds", "triangle_counts", "mesh_used_by_category")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
