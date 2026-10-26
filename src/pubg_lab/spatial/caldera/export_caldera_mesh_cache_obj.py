"""Export a Caldera local mesh cache to OBJ/MTL for Blender rendering."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT

import argparse
import json
import pickle
from array import array
from pathlib import Path

import numpy as np


ROOT = REPO_ROOT
CATEGORIES = ["terrain", "coast_beach", "water", "vegetation", "road", "runway", "building", "prop"]
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

DEFAULT_BUDGETS = {
    "terrain": 240_000,
    "coast_beach": 40_000,
    "water": 30_000,
    "vegetation": 80_000,
    "road": 50_000,
    "runway": 20_000,
    "building": 360_000,
    "prop": 35_000,
}

MATERIALS = {
    "terrain": (0.78, 0.84, 0.81),
    "coast_beach": (0.84, 0.80, 0.68),
    "water": (0.52, 0.70, 0.78),
    "vegetation": (0.66, 0.76, 0.64),
    "road": (0.74, 0.76, 0.74),
    "runway": (0.68, 0.70, 0.70),
    "building": (0.92, 0.95, 0.94),
    "prop": (0.82, 0.82, 0.78),
}


def array_to_tris(flat: array) -> np.ndarray:
    if len(flat) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    tris = np.asarray(flat, dtype=np.float32).reshape((-1, 3, 3))
    return tris[np.isfinite(tris).all(axis=(1, 2))]


def sample_evenly(tris: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0:
        return tris[:0]
    if len(tris) <= max_count:
        return tris
    idx = np.linspace(0, len(tris) - 1, max_count, dtype=np.int64)
    return tris[idx]


def parse_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be min,max")
    lo, hi = (float(part.strip()) for part in parts)
    if hi <= lo:
        raise argparse.ArgumentTypeError("range upper bound must be greater than lower bound")
    return (lo, hi)


def crop_bounds(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    bounds = {}
    for axis in AXIS_INDEX:
        value = getattr(args, f"crop_bounds_{axis}")
        if value is not None:
            bounds[axis] = (value[0] - args.crop_margin, value[1] + args.crop_margin)
    return bounds


def path_ids_for(payload: dict, category: str, expected_len: int) -> np.ndarray | None:
    path_ids = payload.get("triangle_path_ids", {}).get(category)
    if path_ids is None or len(path_ids) == 0:
        return None
    ids = np.asarray(path_ids, dtype=np.int64)
    if len(ids) != expected_len:
        return None
    return ids


def grouped_indices(ids: np.ndarray):
    if len(ids) == 0:
        return
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    breaks = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(sorted_ids)]))
    for start, end in zip(starts, ends):
        yield int(sorted_ids[start]), order[start:end]


def apply_centroid_crop(
    tris: np.ndarray,
    ids: np.ndarray | None,
    bounds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray | None]:
    if not bounds or len(tris) == 0:
        return tris, ids
    centroids = tris.mean(axis=1)
    keep = np.ones(len(tris), dtype=bool)
    for axis, (lo, hi) in bounds.items():
        coord = centroids[:, AXIS_INDEX[axis]]
        keep &= (coord >= lo) & (coord <= hi)
    ids_out = ids[keep] if ids is not None else None
    return tris[keep], ids_out


def bbox_intersects_bounds(
    bounds_min: np.ndarray, bounds_max: np.ndarray, bounds: dict[str, tuple[float, float]]
) -> bool:
    for axis, (lo, hi) in bounds.items():
        idx = AXIS_INDEX[axis]
        if bounds_max[idx] < lo or bounds_min[idx] > hi:
            return False
    return True


def path_for_manifest(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def filter_by_path_names(
    args: argparse.Namespace,
    payload: dict,
    category: str,
    tris: np.ndarray,
    ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    stats = {"paths_before_name_filter": 0, "paths_after_name_filter": 0}
    source_paths = payload.get("source_paths", [])
    if ids is None or len(tris) == 0:
        return tris, ids, stats

    unique_ids = np.unique(ids)
    stats["paths_before_name_filter"] = int(len(unique_ids))
    exclude = [part.lower() for part in args.exclude_path_contains]
    include = [part.lower() for part in args.include_path_contains]
    max_id = int(max(int(ids.max(initial=0)), len(source_paths) - 1))
    path_keep = np.ones(max_id + 1, dtype=bool)
    for path_id in unique_ids:
        path_id_int = int(path_id)
        path = source_paths[path_id_int].lower() if path_id_int < len(source_paths) else ""
        if include and not any(token in path for token in include):
            path_keep[path_id_int] = False
            continue
        if exclude and any(token in path for token in exclude):
            path_keep[path_id_int] = False

    keep = path_keep[ids]
    tris = tris[keep]
    ids = ids[keep]
    stats["paths_after_name_filter"] = int(len(np.unique(ids))) if len(ids) else 0
    return tris, ids, stats


def cap_triangles_per_source_path(
    tris: np.ndarray,
    ids: np.ndarray | None,
    max_count: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if ids is None or max_count <= 0 or len(tris) == 0:
        return tris, ids, 0
    selected = []
    capped_paths = 0
    for _, idx in grouped_indices(ids):
        if len(idx) > max_count:
            idx = idx[np.linspace(0, len(idx) - 1, max_count, dtype=np.int64)]
            capped_paths += 1
        selected.append(idx)
    if not selected:
        return tris[:0], ids[:0], capped_paths
    selected_idx = np.concatenate(selected)
    return tris[selected_idx], ids[selected_idx], capped_paths


def select_whole_source_paths(
    args: argparse.Namespace,
    category: str,
    tris: np.ndarray,
    ids: np.ndarray | None,
    bounds: dict[str, tuple[float, float]],
    budget: int,
) -> tuple[np.ndarray, dict]:
    stats = {
        "preserve_source_paths": bool(args.preserve_source_paths and category != "terrain"),
        "candidate_source_paths": 0,
        "candidate_triangles": 0,
        "selected_source_paths": 0,
        "selected_triangles_before_write": 0,
    }
    if budget <= 0 or len(tris) == 0:
        return tris[:0], stats
    if ids is None:
        tris, _ = apply_centroid_crop(tris, None, bounds)
        return sample_evenly(tris, budget), stats

    if bounds:
        focus = np.array(
            [
                (bounds[axis][0] + bounds[axis][1]) * 0.5 if axis in bounds else 0.0
                for axis in ("x", "y", "z")
            ],
            dtype=np.float64,
        )
        distance_axes = [AXIS_INDEX[axis] for axis in bounds]
    else:
        flat = tris.reshape((-1, 3))
        focus = (flat.min(axis=0) + flat.max(axis=0)) * 0.5
        distance_axes = [0, 1]

    groups = []
    for path_id, idx in grouped_indices(ids):
        group_tris = tris[idx]
        flat = group_tris.reshape((-1, 3))
        bounds_min = flat.min(axis=0)
        bounds_max = flat.max(axis=0)
        if bounds and not bbox_intersects_bounds(bounds_min, bounds_max, bounds):
            continue
        center = (bounds_min + bounds_max) * 0.5
        distance = float(np.linalg.norm(center[distance_axes] - focus[distance_axes]))
        groups.append((distance, -len(idx), path_id, idx))

    stats["candidate_source_paths"] = int(len(groups))
    stats["candidate_triangles"] = int(sum(len(group[3]) for group in groups))
    groups.sort(key=lambda group: (group[0], group[1]))

    selected = []
    total = 0
    for _, _, _, idx in groups:
        if total + len(idx) <= budget or not selected:
            selected.append(idx)
            total += len(idx)
        if total >= budget:
            break
    if not selected:
        return tris[:0], stats
    selected_idx = np.concatenate(selected)
    stats["selected_source_paths"] = int(len(selected))
    stats["selected_triangles_before_write"] = int(len(selected_idx))
    return tris[selected_idx], stats


def write_mtl(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, rgb in MATERIALS.items():
            handle.write(f"newmtl {name}\n")
            handle.write(f"Kd {rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f}\n")
            handle.write("Ka 0.0800 0.0800 0.0800\n")
            handle.write("Ks 0.0800 0.0800 0.0800\n")
            handle.write("Ns 20.0000\n\n")


def write_obj(args: argparse.Namespace, payload: dict) -> dict:
    mtl_path = args.output.with_suffix(".mtl")
    write_mtl(mtl_path)
    counts = {}
    written = {}
    path_stats = {}
    bounds = crop_bounds(args)
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    vertex_index = 1

    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# Caldera local mesh exported from source USD cache\n")
        handle.write(f"mtllib {mtl_path.name}\n")
        for category in CATEGORIES:
            tris = array_to_tris(payload["triangles"].get(category, array("f")))
            counts[category] = int(len(tris))
            ids = path_ids_for(payload, category, len(tris))
            tris, ids, category_path_stats = filter_by_path_names(
                args, payload, category, tris, ids
            )
            budget = getattr(args, f"{category}_triangles")
            if args.preserve_source_paths and category != "terrain":
                tris, preserve_stats = select_whole_source_paths(
                    args, category, tris, ids, bounds, budget
                )
                category_path_stats.update(preserve_stats)
            else:
                tris, ids = apply_centroid_crop(tris, ids, bounds)
                capped_paths = 0
                if category != "terrain":
                    tris, ids, capped_paths = cap_triangles_per_source_path(
                        tris,
                        ids,
                        args.max_triangles_per_source_path,
                    )
                category_path_stats["capped_source_paths"] = int(capped_paths)
                tris = sample_evenly(tris, budget)
            path_stats[category] = category_path_stats
            written[category] = int(len(tris))
            if len(tris) == 0:
                continue
            flat = tris.reshape((-1, 3))
            bounds_min = np.minimum(bounds_min, flat.min(axis=0))
            bounds_max = np.maximum(bounds_max, flat.max(axis=0))
            handle.write(f"\no {category}\nusemtl {category}\n")
            for tri in tris:
                for point in tri:
                    handle.write(f"v {point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n")
                handle.write(f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}\n")
                vertex_index += 3

    return {
        "obj": path_for_manifest(args.output),
        "mtl": path_for_manifest(mtl_path),
        "source_cache": path_for_manifest(args.cache),
        "input_triangles": counts,
        "written_triangles": written,
        "path_selection": path_stats,
        "crop_bounds": bounds,
        "vertices_written": int(vertex_index - 1),
        "bounds": {
            "x": [float(bounds_min[0]), float(bounds_max[0])],
            "y": [float(bounds_min[1]), float(bounds_max[1])],
            "z": [float(bounds_min[2]), float(bounds_max[2])],
        },
    }


def add_category_args(parser: argparse.ArgumentParser) -> None:
    for category, default in DEFAULT_BUDGETS.items():
        parser.add_argument(f"--{category.replace('_', '-')}-triangles", type=int, default=default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-path-contains", action="append", default=[])
    parser.add_argument("--include-path-contains", action="append", default=[])
    parser.add_argument("--max-triangles-per-source-path", type=int, default=0)
    parser.add_argument(
        "--preserve-source-paths",
        action="store_true",
        help="Select whole source paths for non-terrain categories instead of sampling triangles directly.",
    )
    parser.add_argument("--crop-bounds-x", type=parse_range)
    parser.add_argument("--crop-bounds-y", type=parse_range)
    parser.add_argument("--crop-bounds-z", type=parse_range)
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.0,
        help="Expand crop bounds on every provided axis before selecting triangles/source paths.",
    )
    add_category_args(parser)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    manifest = write_obj(args, payload)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"obj_written={args.output}")
    print(f"manifest_written={manifest_path}")
    print(
        json.dumps(
            {k: manifest[k] for k in ("input_triangles", "written_triangles", "bounds")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
