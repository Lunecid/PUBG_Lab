"""Create an RQ3 feature-design board from Caldera geometry features.

The board explains which geometry-derived features Warzone / Caldera can
support for constructs that official PUBG telemetry cannot directly observe.
It uses existing rendered assets and the extracted terrain-experience feature
CSV; it does not train or evaluate a survival model.
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
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = REPO_ROOT
OUT_DIR = ROOT / "reports" / "caldera_3d" / "rq3_feature_design"
FEATURE_CSV = (
    ROOT
    / "reports"
    / "caldera_3d"
    / "single_agent"
    / "world_build"
    / "match_31_player_25_terrain_experience_features.csv"
)
BASE_WORLD_RENDER = (
    ROOT
    / "reports"
    / "caldera_3d"
    / "single_agent"
    / "world_build"
    / "match_31_player_25_local_world_object_core_zup_trajectory_grounded_workbench.png"
)
MARKER_WORLD_RENDER = OUT_DIR / "caldera_rq3_feature_marker_world_tile.png"
TOPDOWN_IMAGE = (
    ROOT / "reports" / "caldera_3d" / "single_agent" / "match_31_player_25_topdown_snapshot.png"
)
EXPERIENCE_IMAGE = (
    ROOT / "reports" / "caldera_3d" / "single_agent" / "match_31_player_25_experience_snapshot.png"
)

MARKERS_JSON = OUT_DIR / "caldera_rq3_feature_markers.json"
TABLE_MD = OUT_DIR / "caldera_rq3_feature_design_table.md"
MARKER_TOPDOWN_OUT = OUT_DIR / "caldera_rq3_feature_marker_topdown.png"
BOARD_OUT = OUT_DIR / "caldera_rq3_feature_design_board.png"
MANIFEST_OUT = OUT_DIR / "caldera_rq3_feature_design_manifest.json"

TOPDOWN_BOUNDS = {
    "x": (-1566.0, 1558.0),
    "y": (-1563.0, 1561.0),
}

BG = (15, 21, 22)
PANEL = (21, 29, 30)
PANEL_2 = (25, 35, 36)
EDGE = (74, 88, 88)
TEXT = (237, 245, 242)
MUTED = (172, 191, 187)
CYAN = (25, 196, 224)
ORANGE = (241, 149, 65)
YELLOW = (241, 204, 74)
PURPLE = (172, 105, 239)
BLUE = (57, 154, 232)
GREEN = (108, 190, 103)
RED = (232, 91, 76)


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


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def load_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(FEATURE_CSV.open(encoding="utf-8")))


def select_feature_markers(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [
        {
            "id": "rough_terrain",
            "label": "rough terrain",
            "short_label": "Rough terrain",
            "color": [ORANGE[0] / 255, ORANGE[1] / 255, ORANGE[2] / 255],
            "row": max(rows, key=lambda row: f(row, "terrain_slope_deg", -1.0)),
            "feature_family": "Terrain shape",
            "designed_features": [
                "terrain_slope_deg",
                "local_relief_r50_m",
                "terrain_ruggedness_r50_m",
            ],
            "pubg_gap": "Coordinates do not say whether the route crosses steep or rugged terrain.",
        },
        {
            "id": "relative_elevation",
            "label": "relative elevation",
            "short_label": "Relative elevation",
            "color": [YELLOW[0] / 255, YELLOW[1] / 255, YELLOW[2] / 255],
            "row": max(rows, key=lambda row: f(row, "relative_elevation_r50_m", -1.0)),
            "feature_family": "Local high-ground context",
            "designed_features": [
                "relative_elevation_r50_m",
                "high_ground_local_r50",
                "forward_grade",
            ],
            "pubg_gap": "Raw z does not encode high or low ground without terrain-neighborhood context.",
        },
        {
            "id": "building_affordance",
            "label": "building affordance",
            "short_label": "Building affordance",
            "color": [PURPLE[0] / 255, PURPLE[1] / 255, PURPLE[2] / 255],
            "row": min(rows, key=lambda row: f(row, "building_dist_m", 1e9)),
            "feature_family": "Compound / building geometry",
            "designed_features": [
                "building_dist_m",
                "building_tri_density_r25_ha",
                "cover_building_r25",
            ],
            "pubg_gap": "Official telemetry does not expose building mesh, wall mass, or compound geometry.",
        },
        {
            "id": "route_corridor",
            "label": "route corridor",
            "short_label": "Route corridor",
            "color": [BLUE[0] / 255, BLUE[1] / 255, BLUE[2] / 255],
            "row": min(rows, key=lambda row: f(row, "road_or_runway_dist_m", 1e9)),
            "feature_family": "Route topology",
            "designed_features": [
                "road_dist_m",
                "runway_dist_m",
                "road_or_runway_dist_m",
                "on_corridor_proxy",
            ],
            "pubg_gap": "Coordinates do not label roads, runways, bridge-like corridors, or open terrain.",
        },
        {
            "id": "terrain_grounding",
            "label": "terrain grounding",
            "short_label": "Terrain grounding",
            "color": [GREEN[0] / 255, GREEN[1] / 255, GREEN[2] / 255],
            "row": max(rows, key=lambda row: f(row, "height_above_terrain_m", -1.0)),
            "feature_family": "Surface grounding",
            "designed_features": [
                "terrain_z_m",
                "height_above_terrain_m",
                "terrain_lookup_xy_dist_m",
            ],
            "pubg_gap": "Actor z exists, but the terrain surface below the actor is absent.",
        },
    ]
    markers = []
    for item in selected:
        row = item["row"]
        markers.append(
            {
                "id": item["id"],
                "label": item["label"],
                "short_label": item["short_label"],
                "feature_family": item["feature_family"],
                "designed_features": item["designed_features"],
                "pubg_gap": item["pubg_gap"],
                "x": f(row, "x"),
                "y": f(row, "y"),
                "z": f(row, "z"),
                "t": f(row, "t"),
                "radius": 6.0,
                "color": item["color"],
                "values": {
                    "height_above_terrain_m": f(row, "height_above_terrain_m"),
                    "terrain_slope_deg": f(row, "terrain_slope_deg"),
                    "local_relief_r50_m": f(row, "local_relief_r50_m"),
                    "relative_elevation_r50_m": f(row, "relative_elevation_r50_m"),
                    "building_dist_m": f(row, "building_dist_m"),
                    "road_or_runway_dist_m": f(row, "road_or_runway_dist_m"),
                    "tactical_state": row.get("tactical_state", ""),
                },
            }
        )
    return markers


def write_marker_outputs(markers: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "Feature exemplar markers for the Caldera RQ3 geometry-feature render.",
        "source_feature_csv": str(FEATURE_CSV.relative_to(ROOT)),
        "markers": markers,
    }
    MARKERS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Caldera RQ3 Geometry Feature Design Table",
        "",
        "| Feature family | Designed features | Example | PUBG telemetry gap |",
        "| --- | --- | --- | --- |",
    ]
    for marker in markers:
        values = marker["values"]
        example = (
            f"t={marker['t']:.3f}s; "
            f"slope={values['terrain_slope_deg']:.2f}deg; "
            f"relief50={values['local_relief_r50_m']:.2f}m; "
            f"building_dist={values['building_dist_m']:.2f}m; "
            f"road/runway_dist={values['road_or_runway_dist_m']:.2f}m"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    marker["feature_family"],
                    ", ".join(f"`{name}`" for name in marker["designed_features"]),
                    example,
                    marker["pubg_gap"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary: these are descriptive geometry-derived construct features, not Caldera survival or rank labels.",
        ]
    )
    TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, PANEL)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def topdown_xy_to_px(x: float, y: float, size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    x0, x1 = TOPDOWN_BOUNDS["x"]
    y0, y1 = TOPDOWN_BOUNDS["y"]
    px = int(round((x - x0) / (x1 - x0) * width))
    py = int(round((y1 - y) / (y1 - y0) * height))
    return px, py


def make_marker_topdown(markers: list[dict[str, Any]]) -> None:
    image = Image.open(TOPDOWN_IMAGE).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = load_font(28, True)
    small = load_font(20)
    for idx, marker in enumerate(markers, start=1):
        color = tuple(int(c * 255) for c in marker["color"])
        x, y = topdown_xy_to_px(marker["x"], marker["y"], image.size)
        radius = 30
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, 86),
            outline=(*color, 255),
            width=6,
        )
        draw.text((x - 8, y - 17), str(idx), fill=(255, 255, 255, 255), font=label_font)
        label = marker["short_label"]
        box_x = min(max(x + 35, 20), image.width - 360)
        box_y = min(max(y - 28, 80), image.height - 76)
        draw.rounded_rectangle(
            (box_x, box_y, box_x + 330, box_y + 58),
            radius=10,
            fill=(9, 14, 15, 220),
            outline=(*color, 230),
            width=2,
        )
        draw.text((box_x + 14, box_y + 9), f"{idx}. {label}", fill=(245, 250, 248, 255), font=small)
    MARKER_TOPDOWN_OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(MARKER_TOPDOWN_OUT)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_gap: int = 6,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if text_size(draw, trial, font)[0] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    line_h = text_size(draw, "Ag", font)[1] + line_gap
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def card(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    body: str,
    color: tuple[int, int, int],
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(rect, radius=10, fill=PANEL_2, outline=EDGE, width=1)
    draw.rounded_rectangle(
        (rect[0] + 18, rect[1] + 20, rect[0] + 34, rect[1] + 92), radius=8, fill=color
    )
    draw.text((rect[0] + 50, rect[1] + 18), title, fill=TEXT, font=title_font)
    draw_wrapped(
        draw, (rect[0] + 50, rect[1] + 58), body, body_font, MUTED, rect[2] - rect[0] - 78, 5
    )


def make_board(markers: list[dict[str, Any]]) -> None:
    width, height = 3000, 1900
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title = load_font(58, True)
    h2 = load_font(32, True)
    h3 = load_font(25, True)
    body = load_font(22)
    small = load_font(18)
    tiny = load_font(16)

    draw.text((56, 38), "RQ3 Feature Design From Warzone / Caldera Geometry", fill=TEXT, font=title)
    draw.text(
        (58, 105),
        "A geometry-rich world model shows spatial constructs that official PUBG telemetry cannot directly observe.",
        fill=MUTED,
        font=body,
    )

    render_path = MARKER_WORLD_RENDER if MARKER_WORLD_RENDER.exists() else BASE_WORLD_RENDER
    world = fit_image(render_path, (1780, 1110))
    image.paste(world, (56, 165))
    draw.rounded_rectangle((56, 165, 1836, 1275), radius=12, outline=EDGE, width=2)
    draw.rectangle((57, 166, 1835, 260), fill=(13, 19, 20))
    draw.text(
        (82, 190), "A. Object-preserved 3D world tile + observed trajectory", fill=TEXT, font=h2
    )
    if MARKER_WORLD_RENDER.exists():
        subtitle = "Colored markers indicate feature exemplars selected from geometry-derived trajectory features."
    else:
        subtitle = (
            "3D spatial substrate; selected feature exemplars are marked in the map panel below."
        )
    draw.text((82, 232), subtitle, fill=MUTED, font=small)

    x0, y0 = 1880, 165
    draw.rounded_rectangle((x0, y0, 2944, 1275), radius=12, fill=PANEL, outline=EDGE, width=2)
    draw.text((x0 + 30, y0 + 24), "B. Feature families this enables", fill=TEXT, font=h2)
    draw.text(
        (x0 + 30, y0 + 66),
        "These are construct features, not supervised outcome labels.",
        fill=MUTED,
        font=small,
    )

    marker_colors = {
        marker["id"]: tuple(int(c * 255) for c in marker["color"]) for marker in markers
    }
    descriptions = [
        (
            "Surface grounding",
            "`terrain_z_m`, `height_above_terrain_m`, terrain lookup distance",
            marker_colors["terrain_grounding"],
        ),
        (
            "Terrain shape",
            "`terrain_slope_deg`, local relief, ruggedness across 25/50/100m neighborhoods",
            marker_colors["rough_terrain"],
        ),
        (
            "Relative elevation",
            "`relative_elevation_r50_m`, high-ground local flags, forward grade",
            marker_colors["relative_elevation"],
        ),
        (
            "Building affordance",
            "nearest building distance, building density, cover-building radius proxies",
            marker_colors["building_affordance"],
        ),
        (
            "Route topology",
            "road/runway distance and corridor-contact proxy",
            marker_colors["route_corridor"],
        ),
    ]
    cy = y0 + 118
    for idx, (name, desc, color) in enumerate(descriptions):
        card(draw, (x0 + 30, cy, x0 + 1034, cy + 146), name, desc, color, h3, body)
        cy += 166

    topdown_source = MARKER_TOPDOWN_OUT if MARKER_TOPDOWN_OUT.exists() else TOPDOWN_IMAGE
    topdown = fit_image(topdown_source, (860, 560))
    image.paste(topdown, (56, 1320))
    draw.rounded_rectangle((56, 1320, 916, 1880), radius=12, outline=EDGE, width=2)
    draw.rectangle((57, 1321, 915, 1395), fill=(13, 19, 20))
    draw.text((82, 1345), "C. Feature exemplar locations on semantic layer", fill=TEXT, font=h2)

    draw.rounded_rectangle((956, 1320, 2944, 1880), radius=12, fill=PANEL, outline=EDGE, width=2)
    draw.text((986, 1345), "D. Selected feature exemplars", fill=TEXT, font=h2)

    headers = ["Marker", "Example", "What it lets us design"]
    col_x = [986, 1280, 1885]
    for x, header in zip(col_x, headers):
        draw.text((x, 1394), header, fill=MUTED, font=h3)
    draw.line((986, 1430, 2910, 1430), fill=EDGE, width=1)

    row_y = 1450
    for marker in markers:
        color = tuple(int(c * 255) for c in marker["color"])
        values = marker["values"]
        draw.ellipse((990, row_y + 8, 1018, row_y + 36), fill=color)
        draw.text((1030, row_y + 4), marker["short_label"], fill=TEXT, font=body)
        example = (
            f"t={marker['t']:.1f}s | slope {values['terrain_slope_deg']:.1f}deg | "
            f"relief50 {values['local_relief_r50_m']:.1f}m | "
            f"bldg {values['building_dist_m']:.1f}m | route {values['road_or_runway_dist_m']:.1f}m"
        )
        draw_wrapped(draw, (1280, row_y + 4), example, small, MUTED, 560, 3)
        design = ", ".join(marker["designed_features"])
        draw_wrapped(draw, (1885, row_y + 4), design, small, TEXT, 1000, 3)
        row_y += 74

    note = (
        "Boundary: Warzone / Caldera supplies geometry-rich construct measurements. "
        "It does not provide the supervised PUBG phase-survival labels used in RQ4; LOS/occlusion is a future ray-cast extension."
    )
    draw_wrapped(draw, (986, 1818), note, tiny, MUTED, 1850, 4)

    ImageOps.expand(image, border=0).save(BOARD_OUT)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    markers = select_feature_markers(rows)
    write_marker_outputs(markers)
    make_marker_topdown(markers)
    make_board(markers)
    manifest = {
        "purpose": "RQ3 board explaining geometry-derived feature designs enabled by Warzone / Caldera.",
        "source_feature_csv": str(FEATURE_CSV.relative_to(ROOT)),
        "marker_json": str(MARKERS_JSON.relative_to(ROOT)),
        "marker_topdown": str(MARKER_TOPDOWN_OUT.relative_to(ROOT)),
        "board": str(BOARD_OUT.relative_to(ROOT)),
        "feature_table_md": str(TABLE_MD.relative_to(ROOT)),
        "world_render_used": str(
            (
                MARKER_WORLD_RENDER if MARKER_WORLD_RENDER.exists() else BASE_WORLD_RENDER
            ).relative_to(ROOT)
        ),
        "markers": [
            {k: marker[k] for k in ("id", "label", "t", "x", "y", "z", "feature_family")}
            for marker in markers
        ],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"markers={MARKERS_JSON}")
    print(f"marker_topdown={MARKER_TOPDOWN_OUT}")
    print(f"table={TABLE_MD}")
    print(f"board={BOARD_OUT}")
    print(f"manifest={MANIFEST_OUT}")


if __name__ == "__main__":
    main()
