"""Render a Caldera OBJ mesh in Blender as a clay/wireframe scene."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import json
import math
import sys
from pathlib import Path


def parse_vec(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("vector must be x,y,z")
    return tuple(float(part.strip()) for part in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--camera-location", type=parse_vec)
    parser.add_argument("--camera-target", type=parse_vec)
    parser.add_argument("--orthographic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ortho-scale", type=float, default=700.0)
    parser.add_argument("--lens", type=float, default=35.0)
    parser.add_argument("--resolution-x", type=int, default=1800)
    parser.add_argument("--resolution-y", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--wire-thickness", type=float, default=0.0018)
    parser.add_argument("--wire-categories", default="building,road,runway,prop")
    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_WORKBENCH"])
    parser.add_argument("--base-plane", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-json", type=Path)
    parser.add_argument("--trajectory-elevation-offset", type=float, default=1.2)
    parser.add_argument("--trajectory-radius", type=float, default=1.4)
    parser.add_argument("--trajectory-marker-radius", type=float, default=4.0)
    parser.add_argument("--trajectory-color", type=parse_vec, default=(0.02, 0.88, 1.0))
    parser.add_argument("--feature-marker-json", type=Path)
    parser.add_argument("--feature-marker-z-offset", type=float, default=5.0)
    parser.add_argument("--feature-marker-radius", type=float, default=5.5)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def look_at(obj, target) -> None:
    import mathutils

    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def bounds_from_objects(objects):
    import mathutils

    mins = mathutils.Vector((math.inf, math.inf, math.inf))
    maxs = mathutils.Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    return mins, maxs


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.68):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def load_trajectory_points(path: Path | None, z_offset: float) -> list[tuple[float, float, float]]:
    if path is None:
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    points = []
    for row in rows:
        points.append((float(row["x"]), float(row["y"]), float(row["z"]) + z_offset))
    return points


def add_trajectory(
    points: list[tuple[float, float, float]],
    radius: float,
    marker_radius: float,
    color: tuple[float, float, float],
) -> None:
    if len(points) < 2:
        return
    import bpy

    line_mat = make_mat("trajectory_line", (color[0], color[1], color[2], 1.0), 0.38)
    start_mat = make_mat("trajectory_start", (0.10, 0.95, 0.42, 1.0), 0.38)
    end_mat = make_mat("trajectory_end", (1.0, 0.42, 0.12, 1.0), 0.38)

    curve = bpy.data.curves.new("player_trajectory_curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    polyline = curve.splines.new("POLY")
    polyline.points.add(len(points) - 1)
    for idx, point in enumerate(points):
        polyline.points[idx].co = (point[0], point[1], point[2], 1.0)
    curve.materials.append(line_mat)
    obj = bpy.data.objects.new("player_trajectory", curve)
    obj.color = (color[0], color[1], color[2], 1.0)
    bpy.context.collection.objects.link(obj)

    for name, point, mat in (
        ("trajectory_start_marker", points[0], start_mat),
        ("trajectory_end_marker", points[-1], end_mat),
    ):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=24, ring_count=12, radius=marker_radius, location=point
        )
        marker = bpy.context.object
        marker.name = name
        marker.data.materials.append(mat)
        bpy.ops.object.shade_smooth()


def load_feature_markers(path: Path | None, z_offset: float) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers = payload.get("markers", payload if isinstance(payload, list) else [])
    out = []
    for marker in markers:
        color = marker.get("color", [1.0, 0.2, 0.05])
        out.append(
            {
                "label": str(marker.get("label", "marker")),
                "x": float(marker["x"]),
                "y": float(marker["y"]),
                "z": float(marker["z"]) + z_offset,
                "radius": float(marker.get("radius", 0.0)),
                "color": (float(color[0]), float(color[1]), float(color[2])),
            }
        )
    return out


def add_feature_markers(markers: list[dict], default_radius: float) -> None:
    if not markers:
        return
    import bpy

    for idx, marker in enumerate(markers, start=1):
        color = marker["color"]
        mat = make_mat(
            f"feature_marker_{idx}_{marker['label']}", (color[0], color[1], color[2], 1.0), 0.32
        )
        radius = marker["radius"] or default_radius
        location = (marker["x"], marker["y"], marker["z"])
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=radius, location=location
        )
        obj = bpy.context.object
        obj.name = f"feature_marker_{idx}_{marker['label']}"
        obj.data.materials.append(mat)
        bpy.ops.object.shade_smooth()


def main() -> None:
    import bpy
    import mathutils

    args = parse_args()
    args.obj = args.obj.resolve()
    args.output = args.output.resolve()
    if args.trajectory_json is not None:
        args.trajectory_json = args.trajectory_json.resolve()
    if args.feature_marker_json is not None:
        args.feature_marker_json = args.feature_marker_json.resolve()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(args.obj), forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.obj(filepath=str(args.obj), axis_forward="Y", axis_up="Z")

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise SystemExit(f"No mesh objects imported from {args.obj}")

    clay = make_mat("caldera_clay", (0.90, 0.94, 0.94, 1.0), 0.76)
    terrain_mat = make_mat("terrain_clay", (0.76, 0.83, 0.80, 1.0), 0.84)
    road_mat = make_mat("road_clay", (0.72, 0.75, 0.74, 1.0), 0.78)
    wire_mat = make_mat("wire_dark", (0.025, 0.035, 0.035, 1.0), 0.50)
    wire_categories = {
        part.strip().lower() for part in args.wire_categories.split(",") if part.strip()
    }

    for obj in mesh_objects:
        name = obj.name.lower()
        obj.data.materials.clear()
        if "terrain" in name:
            obj.data.materials.append(terrain_mat)
        elif "road" in name or "runway" in name:
            obj.data.materials.append(road_mat)
        else:
            obj.data.materials.append(clay)
        obj.select_set(True)
        mod = obj.modifiers.new("WeightedNormals", "WEIGHTED_NORMAL")
        mod.keep_sharp = True
        category_name = name.split(".")[0]
        if category_name in wire_categories:
            wire = obj.modifiers.new("WireOverlay", "WIREFRAME")
            wire.thickness = args.wire_thickness
            wire.use_replace = False
            obj.data.materials.append(wire_mat)
            wire.material_offset = len(obj.data.materials) - 1

    trajectory_points = load_trajectory_points(
        args.trajectory_json, args.trajectory_elevation_offset
    )
    add_trajectory(
        trajectory_points,
        args.trajectory_radius,
        args.trajectory_marker_radius,
        args.trajectory_color,
    )
    feature_markers = load_feature_markers(args.feature_marker_json, args.feature_marker_z_offset)
    add_feature_markers(feature_markers, args.feature_marker_radius)

    mins, maxs = bounds_from_objects(mesh_objects)
    center = (mins + maxs) * 0.5
    span = max(maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z)

    bpy.ops.object.light_add(type="SUN", location=(center.x - 200, center.y - 350, center.z + 500))
    sun = bpy.context.object
    sun.name = "Caldera Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(52), math.radians(0), math.radians(-35))

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    target = args.camera_target or (center.x, center.y, center.z + 12)
    if args.camera_location:
        camera.location = args.camera_location
    else:
        camera.location = (center.x + span * 0.74, center.y - span * 0.86, center.z + span * 0.48)
    look_at(camera, target)
    if args.orthographic:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = args.ortho_scale
    else:
        camera.data.type = "PERSP"
        camera.data.lens = args.lens
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = args.engine
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
    elif args.engine == "BLENDER_WORKBENCH":
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_valley_factor = 1.2
        scene.display.shading.cavity_ridge_factor = 0.6
    scene.world.color = (0.86, 0.91, 0.92)
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if args.base_plane:
        bpy.ops.mesh.primitive_plane_add(
            size=span * 1.6, location=(center.x, center.y, mins.z - 0.08)
        )
        plane = bpy.context.object
        plane.name = "matte_ground_reference"
        plane.data.materials.append(make_mat("matte_ground", (0.80, 0.87, 0.87, 1.0), 0.9))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)

    manifest = {
        "obj": str(args.obj),
        "output": str(args.output),
        "camera_location": list(camera.location),
        "camera_target": list(target),
        "orthographic": args.orthographic,
        "ortho_scale": camera.data.ortho_scale if args.orthographic else None,
        "bounds": {
            "min": list(mins),
            "max": list(maxs),
            "center": list(center),
            "span": float(span),
        },
        "mesh_objects": [obj.name for obj in mesh_objects],
        "trajectory_json": str(args.trajectory_json) if args.trajectory_json else None,
        "trajectory_points": len(trajectory_points),
        "feature_marker_json": str(args.feature_marker_json) if args.feature_marker_json else None,
        "feature_markers": len(feature_markers),
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
