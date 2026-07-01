"""Build a tile-based Caldera map mesh from OpenUSD assets.

This is the geometry-first path.  It materializes a reproducible, simplified
3D map model before any agent trajectory figure is drawn.

The full Caldera dataset is too large to flatten naively: the release README
describes roughly 17.5M primitives and more than 2B points.  This script
therefore creates an LOD mesh package:

- terrain tiles from superterrrain/season_4/*.terrain.usd;
- district/POI wrappers from season_4/map_*.usd;
- non-visual gameplay/audio/loot/trigger layers filtered out;
- a streamed OBJ plus a manifest with source bounds and face counts.

Use the project's `.venv/bin/python`; it contains the `pxr` OpenUSD bindings.
If invoked from a Python without `pxr`, the script reinvokes itself through the
project venv when available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "br_l0" / "caldera" / "repo"
MAP_ROOT = REPO / "map_source" / "prefabs" / "br" / "wz_vg" / "mp_wz_island"
SEASON4 = MAP_ROOT / "season_4"
SUPERTERRAIN = MAP_ROOT / "superterrrain" / "season_4"
ROOT_STAGE = REPO / "map_source" / "mp_wz_island.usd"
DEFAULT_OUT_DIR = ROOT / "reports" / "caldera_3d" / "map_mesh_lod0"
GEO_ROOT = "/world/mp_wz_island/mp_wz_island_paths/mp_wz_island_geo"

TERRAIN_BRANCHES = ["st_b", "st_d", "st_c", "st_f", "st_e", "st_h", "st_g", "st_j", "st_i", "st_l", "st_k", "st_n", "st_p", "st_o"]
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

DEFAULT_EXCLUDES = [
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
]


@dataclass
class Bounds:
    min_x: float = math.inf
    min_y: float = math.inf
    min_z: float = math.inf
    max_x: float = -math.inf
    max_y: float = -math.inf
    max_z: float = -math.inf

    def update(self, p: tuple[float, float, float]) -> None:
        self.min_x = min(self.min_x, p[0])
        self.min_y = min(self.min_y, p[1])
        self.min_z = min(self.min_z, p[2])
        self.max_x = max(self.max_x, p[0])
        self.max_y = max(self.max_y, p[1])
        self.max_z = max(self.max_z, p[2])

    def merge(self, other: "Bounds") -> None:
        if not other.valid:
            return
        self.update((other.min_x, other.min_y, other.min_z))
        self.update((other.max_x, other.max_y, other.max_z))

    @property
    def valid(self) -> bool:
        return self.min_x < math.inf

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
    stage_path: Path
    root_path: str

    @property
    def label(self) -> str:
        tail = self.root_path.rstrip("/").split("/")[-1]
        return f"{self.group}_{tail}"


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


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def discover_sources(groups: set[str]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    if "terrain" in groups:
        for branch in TERRAIN_BRANCHES:
            sources.append(SourceSpec("terrain", ROOT_STAGE, f"{GEO_ROOT}/st_main/{branch}"))
    if "districts" in groups:
        for branch in DISTRICT_BRANCHES:
            sources.append(SourceSpec("district", ROOT_STAGE, f"{GEO_ROOT}/{branch}"))
    return sources


def set_variant_sets(root_prim, new_state: str) -> int:
    from pxr import Usd

    changed = 0
    if new_state == "none":
        return changed
    for prim in Usd.PrimRange(root_prim):
        if not prim.HasVariantSets():
            continue
        variant_sets = prim.GetVariantSets()
        for variant_name in variant_sets.GetNames():
            if "terrain" not in variant_name.lower() and "district" not in variant_name.lower():
                continue
            variant_set = variant_sets.GetVariantSet(variant_name)
            if new_state in variant_set.GetVariantNames():
                before = variant_sets.GetVariantSelection(variant_name)
                if before != new_state:
                    variant_set.SetVariantSelection(new_state)
                    changed += 1
    return changed


def should_skip_path(path: str, excludes: Iterable[str], includes: Iterable[str]) -> bool:
    include_tokens = list(includes)
    if include_tokens and not any(token in path for token in include_tokens):
        return True
    return any(token in path for token in excludes)


def vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def area2(tri: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> float:
    cross = vec_cross(vec_sub(tri[1], tri[0]), vec_sub(tri[2], tri[0]))
    return math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)


def write_obj_triangle(obj, vertex_offset: int, tri) -> int:
    for p in tri:
        obj.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    obj.write(f"f {vertex_offset + 1} {vertex_offset + 2} {vertex_offset + 3}\n")
    return vertex_offset + 3


def extract_source(
    spec: SourceSpec,
    args: argparse.Namespace,
    obj,
    vertex_offset: int,
    stage_cache: dict[Path, object],
) -> tuple[dict, int]:
    from pxr import Usd, UsdGeom

    stage = stage_cache.get(spec.stage_path)
    if stage is None:
        stage = Usd.Stage.Open(str(spec.stage_path))
        if stage is None:
            raise RuntimeError(f"Could not open USD stage: {spec.stage_path}")
        stage_cache[spec.stage_path] = stage

    root_prim = stage.GetPrimAtPath(spec.root_path)
    if not root_prim or not root_prim.IsValid():
        return (
            {
                "group": spec.group,
                "stage": str(spec.stage_path.relative_to(ROOT)),
                "root_path": spec.root_path,
                "missing": True,
                "faces_written": 0,
                "meshes_used": 0,
                "meshes_seen": 0,
                "bounds": Bounds().as_dict(),
            },
            vertex_offset,
        )

    changed_variants = set_variant_sets(root_prim, args.variant)
    scale = float(stage.GetMetadata("metersPerUnit") or 0.0254)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    source_bounds = Bounds()
    meshes_seen = 0
    meshes_used = 0
    faces_written = 0
    faces_seen = 0
    skipped_by_path = 0
    skipped_by_purpose = 0

    obj.write(f"\no {safe_name(spec.label)}\n")

    for prim in Usd.PrimRange(root_prim):
        if args.max_faces_per_source and faces_written >= args.max_faces_per_source:
            break
        if not prim.IsA(UsdGeom.Mesh):
            continue
        meshes_seen += 1
        prim_path = prim.GetPath().pathString
        if should_skip_path(prim_path, args.exclude_path_contains, args.include_path_contains):
            skipped_by_path += 1
            continue

        imageable = UsdGeom.Imageable(prim)
        purpose = str(imageable.ComputePurpose())
        if purpose == "guide":
            skipped_by_purpose += 1
            continue
        if purpose == "render" and not args.include_render_purpose:
            skipped_by_purpose += 1
            continue

        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not local_points or not counts or not indices:
            continue

        matrix = xform_cache.GetLocalToWorldTransform(prim)
        world_points = []
        for point in local_points:
            wp = matrix.Transform(point)
            world_points.append((float(wp[0]) * scale, float(wp[1]) * scale, float(wp[2]) * scale))

        cursor = 0
        mesh_used = False
        mesh_faces_written = 0
        max_faces_per_mesh = args.terrain_max_faces_per_mesh if spec.group == "terrain" else args.district_max_faces_per_mesh
        if args.max_faces_per_mesh >= 0:
            max_faces_per_mesh = args.max_faces_per_mesh
        face_stride = args.terrain_face_stride if spec.group == "terrain" else args.district_face_stride
        if args.max_faces_per_source and not max_faces_per_mesh:
            remaining_budget = max(1, args.max_faces_per_source - faces_written)
            face_stride = max(face_stride, math.ceil(len(counts) / remaining_budget))
        for face_idx, count in enumerate(counts):
            face_indices = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            faces_seen += 1
            if face_stride > 1 and face_idx % face_stride != 0:
                continue
            if max_faces_per_mesh and mesh_faces_written >= max_faces_per_mesh:
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
                a2 = area2(tri)
                if a2 < args.min_triangle_area:
                    continue
                if spec.group != "terrain" and args.max_triangle_area and a2 > args.max_triangle_area:
                    continue
                vertex_offset = write_obj_triangle(obj, vertex_offset, tri)
                for p in tri:
                    source_bounds.update(p)
                faces_written += 1
                mesh_faces_written += 1
                mesh_used = True
                if args.max_faces_per_source and faces_written >= args.max_faces_per_source:
                    break
                if max_faces_per_mesh and mesh_faces_written >= max_faces_per_mesh:
                    break
            if args.max_faces_per_source and faces_written >= args.max_faces_per_source:
                break
        if mesh_used:
            meshes_used += 1

    return (
        {
            "group": spec.group,
            "stage": str(spec.stage_path.relative_to(ROOT)),
            "root_path": spec.root_path,
            "scale_meters_per_unit": scale,
            "variant": args.variant,
            "variant_sets_changed": changed_variants,
            "meshes_seen": meshes_seen,
            "meshes_used": meshes_used,
            "faces_seen": faces_seen,
            "faces_written": faces_written,
            "skipped_by_path": skipped_by_path,
            "skipped_by_purpose": skipped_by_purpose,
            "bounds": source_bounds.as_dict(),
        },
        vertex_offset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--groups", default="terrain,districts", help="Comma-separated: terrain,districts")
    parser.add_argument("--variant", default="full", choices=["full", "proxy", "none"])
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--max-faces-per-source", type=int, default=25000)
    parser.add_argument("--max-faces-per-mesh", type=int, default=-1, help="Override both terrain/district mesh caps when >= 0")
    parser.add_argument("--terrain-max-faces-per-mesh", type=int, default=0)
    parser.add_argument("--district-max-faces-per-mesh", type=int, default=120)
    parser.add_argument("--terrain-face-stride", type=int, default=8)
    parser.add_argument("--district-face-stride", type=int, default=18)
    parser.add_argument("--min-triangle-area", type=float, default=0.00001)
    parser.add_argument("--max-triangle-area", type=float, default=0.0)
    parser.add_argument("--include-render-purpose", action="store_true")
    parser.add_argument("--include-path-contains", action="append", default=[])
    parser.add_argument("--exclude-path-contains", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    ensure_pxr_runtime()
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.exclude_path_contains = [*DEFAULT_EXCLUDES, *args.exclude_path_contains]

    groups = {part.strip() for part in args.groups.split(",") if part.strip()}
    sources = discover_sources(groups)
    if args.limit_sources:
        sources = sources[: args.limit_sources]
    if not sources:
        raise SystemExit(f"No sources discovered for groups={sorted(groups)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = args.out_dir / "caldera_map_lod0.obj"
    manifest_path = args.out_dir / "manifest.json"

    total_bounds = Bounds()
    manifest = {
        "description": "Caldera map mesh LOD built from OpenUSD visual assets.",
        "repo": str(REPO.relative_to(ROOT)),
        "obj": str(obj_path.relative_to(ROOT)),
        "options": {
            "groups": sorted(groups),
            "variant": args.variant,
            "max_faces_per_source": args.max_faces_per_source,
            "max_faces_per_mesh": args.max_faces_per_mesh,
            "terrain_max_faces_per_mesh": args.terrain_max_faces_per_mesh,
            "district_max_faces_per_mesh": args.district_max_faces_per_mesh,
            "terrain_face_stride": args.terrain_face_stride,
            "district_face_stride": args.district_face_stride,
            "include_render_purpose": args.include_render_purpose,
            "exclude_path_contains": args.exclude_path_contains,
            "include_path_contains": args.include_path_contains,
        },
        "sources": [],
    }

    vertex_offset = 0
    stage_cache: dict[Path, object] = {}
    with obj_path.open("w", encoding="utf-8") as obj:
        obj.write("# Caldera map LOD0 mesh generated from OpenUSD assets\n")
        obj.write("# Coordinates are in meters in the source USD world frame.\n")
        for idx, spec in enumerate(sources, start=1):
            print(f"[{idx}/{len(sources)}] extracting {spec.group}: {spec.root_path}", flush=True)
            source_info, vertex_offset = extract_source(spec, args, obj, vertex_offset, stage_cache)
            manifest["sources"].append(source_info)
            b = source_info["bounds"]
            if b["x"] and b["y"] and b["z"]:
                source_bounds = Bounds(b["x"][0], b["y"][0], b["z"][0], b["x"][1], b["y"][1], b["z"][1])
                total_bounds.merge(source_bounds)
            print(
                f"    faces={source_info['faces_written']} meshes={source_info['meshes_used']}/"
                f"{source_info['meshes_seen']} bounds={source_info['bounds']}",
                flush=True,
            )

    manifest["summary"] = {
        "source_count": len(sources),
        "total_faces_written": sum(item["faces_written"] for item in manifest["sources"]),
        "total_meshes_used": sum(item["meshes_used"] for item in manifest["sources"]),
        "bounds": total_bounds.as_dict(),
        "obj_size_bytes": obj_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"obj_written={obj_path}")
    print(f"manifest_written={manifest_path}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
