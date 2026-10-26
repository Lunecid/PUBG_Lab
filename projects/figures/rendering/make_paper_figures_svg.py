"""Generate publication figures for the PUBG STDB/DynamicSTGNN paper.

This script draws the two core figures directly as SVG and renders them to
high-resolution PNG with headless Chrome. The layouts are built from the
paper's current method specification rather than from prior rendered figures:

* pipeline_overview: telemetry -> STDB -> tensors -> prediction/evaluation
* dynstgnn_architecture: tensor pack -> two ST blocks -> hazard head

The output filenames intentionally match the LaTeX references.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import html
import math
import subprocess
from pathlib import Path


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

INK = "#253746"
SUB = "#4B5C6C"
MUTE = "#8A98A7"
RULE = "#5D7184"
GRID = "#D3DBE3"
BG = "#FFFFFF"
SOFT = "#F6F8FB"
DB = "#EAF1F8"
DATA = "#E8F4EC"
GRAPH = "#F0EAF8"
MODEL = "#FFF0E4"
OUT = "#E6F0FA"
WARN = "#FFF7D8"

PROX = "#2867B2"
COMBAT = "#C0392B"
COPRESS = "#8E5FB8"


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        ]

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        *,
        stroke: str = RULE,
        sw: float = 1.5,
        rx: float = 9,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str,
        *,
        stroke: str = RULE,
        sw: float = 1.5,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def ellipse(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        fill: str,
        *,
        stroke: str = RULE,
        sw: float = 1.5,
    ) -> None:
        self.parts.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = RULE,
        sw: float = 2,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"{dash_attr}/>'
        )

    def path(self, d: str, *, fill: str = "none", stroke: str = RULE, sw: float = 1.5) -> None:
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = INK,
        sw: float = 2,
        dash: str | None = None,
        head: float = 10,
    ) -> None:
        self.line(x1, y1, x2, y2, color=color, sw=sw, dash=dash)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        left = (x2 - head * ux + 0.55 * head * px, y2 - head * uy + 0.55 * head * py)
        right = (x2 - head * ux - 0.55 * head * px, y2 - head * uy - 0.55 * head * py)
        self.path(
            f"M {left[0]:.1f} {left[1]:.1f} L {x2:.1f} {y2:.1f} L {right[0]:.1f} {right[1]:.1f}",
            stroke=color,
            sw=sw,
        )

    def text(
        self,
        text: str,
        x: float,
        y: float,
        size: float,
        *,
        weight: int = 400,
        color: str = SUB,
        anchor: str = "middle",
        italic: bool = False,
    ) -> None:
        style = " font-style='italic'" if italic else ""
        safe = html.escape(text)
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'dominant-baseline="central"{style}>{safe}</text>'
        )

    def multiline(
        self,
        lines: list[str],
        x: float,
        y: float,
        size: float,
        *,
        step: float = 20,
        color: str = SUB,
        weight: int = 400,
        anchor: str = "middle",
    ) -> None:
        for idx, line in enumerate(lines):
            self.text(line, x, y + idx * step, size, color=color, weight=weight, anchor=anchor)

    def tag(self, label: str, x: float, y: float, w: float, *, color: str = MUTE) -> None:
        self.rect(x, y, w, 24, "#FFFFFF", stroke="#B8C3CF", sw=1.0, rx=12)
        self.text(label, x + w / 2, y + 12, 10.5, weight=700, color=color)

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def small_graph(svg: Svg, cx: float, cy: float, scale: float = 1.0) -> None:
    pts = [
        (cx - 42 * scale, cy - 20 * scale),
        (cx + 8 * scale, cy - 34 * scale),
        (cx + 42 * scale, cy - 4 * scale),
        (cx + 16 * scale, cy + 32 * scale),
        (cx - 38 * scale, cy + 22 * scale),
    ]
    svg.line(*pts[0], *pts[1], color=PROX, sw=2.1 * scale)
    svg.line(*pts[3], *pts[4], color=PROX, sw=2.1 * scale)
    svg.arrow(*pts[1], *pts[2], color=COMBAT, sw=2.1 * scale, head=8 * scale)
    svg.line(*pts[0], *pts[3], color=COPRESS, sw=2.1 * scale, dash="5 4")
    for x, y in pts:
        svg.circle(x, y, 6.5 * scale, "#FFFFFF", stroke=INK, sw=1.6 * scale)


def tensor_table(svg: Svg, x: float, y: float) -> None:
    svg.rect(x, y, 185, 92, "#FFFFFF", stroke="#AAB7C4", sw=1.2, rx=6)
    for i in range(1, 5):
        svg.line(x, y + i * 18, x + 185, y + i * 18, color=GRID, sw=1)
    for i in range(1, 4):
        svg.line(x + i * 46, y, x + i * 46, y + 92, color=GRID, sw=1)
    svg.text("m", x + 23, y + 12, 9.5, weight=700, color=MUTE)
    svg.text("team", x + 69, y + 12, 9.5, weight=700, color=MUTE)
    svg.text("phase", x + 115, y + 12, 9.5, weight=700, color=MUTE)
    svg.text("66-d", x + 160, y + 12, 9.5, weight=700, color=MUTE)
    for r, values in enumerate(
        [("1", "T12", "3", "..."), ("1", "T19", "3", "..."), ("2", "T04", "5", "...")]
    ):
        for c, value in enumerate(values):
            svg.text(value, x + 23 + c * 46, y + 33 + r * 18, 9.5, color=SUB)


def tensor_stack(svg: Svg, x: float, y: float, w: float, h: float, color: str, label: str) -> None:
    for off in (16, 8, 0):
        svg.rect(
            x + off, y - off, w, h, color if off == 0 else "#FFFFFF", stroke="#9EADBA", sw=1.1, rx=5
        )
    svg.text(label, x + w / 2, y + h / 2, 13, weight=700, color=INK)


def fig_pipeline(path: Path) -> None:
    svg = Svg(1680, 620)
    svg.text(
        "Core pipeline: phase-aligned representation and next-phase hazard prediction",
        840,
        36,
        22,
        weight=700,
        color=INK,
    )
    svg.line(80, 68, 1600, 68, color=GRID, sw=1.2)

    # Phase rail.
    svg.text("phase clock", 130, 92, 12, weight=700, color=MUTE, anchor="start")
    for idx in range(8):
        x = 245 + idx * 115
        radius = 28 - idx * 2.3
        svg.circle(x, 92, radius, "#FFFFFF", stroke="#B7C3D0", sw=1.2)
        svg.text(f"k={idx + 1}", x, 92, 10.5, weight=600, color=SUB)
        if idx < 7:
            svg.arrow(x + radius + 5, 92, x + 105 - radius, 92, color="#B7C3D0", sw=1.2, head=6)
    svg.text("shrinking safe-zone phases", 1215, 92, 12, italic=True, color=MUTE)

    # Left lane labels.
    svg.rect(45, 140, 74, 180, "#FFFFFF", stroke=GRID, sw=1.0, rx=8)
    svg.text("RQ1", 82, 178, 15, weight=700, color=INK)
    svg.multiline(["build a", "phase-aligned", "STDB"], 82, 220, 10.5, step=15, color=MUTE)
    svg.rect(45, 345, 74, 170, "#FFFFFF", stroke=GRID, sw=1.0, rx=8)
    svg.text("RQ2", 82, 385, 15, weight=700, color=INK)
    svg.multiline(["predict", "next-phase", "hazard"], 82, 425, 10.5, step=15, color=MUTE)

    # Main transformation blocks.
    svg.rect(150, 150, 220, 155, SOFT, rx=8)
    svg.tag("source", 168, 166, 64)
    svg.text("Official telemetry", 260, 197, 17, weight=700, color=INK)
    svg.multiline(
        ["time-stamped events", "actors and teams", "locations and state"], 260, 235, 13, step=25
    )

    svg.arrow(385, 227, 445, 227, color=INK, sw=2.2)
    svg.text("align time + space", 415, 204, 11.5, italic=True, color=MUTE)

    svg.rect(460, 132, 285, 190, DB, rx=8)
    svg.tag("representation", 482, 150, 118)
    svg.text("STDB objects", 602, 187, 17, weight=700, color=INK)
    object_tiles = [
        ("moving teams", 490, 215),
        ("dynamic zones", 620, 215),
        ("spatial events", 490, 266),
        ("phase intervals", 620, 266),
    ]
    for label, x, y in object_tiles:
        svg.rect(x, y, 105, 34, "#FFFFFF", stroke="#B5C1CE", sw=1.0, rx=6)
        svg.text(label, x + 52.5, y + 17, 10.8, weight=600, color=SUB)

    svg.arrow(760, 205, 830, 205, color=INK, sw=2.2)
    svg.arrow(760, 275, 830, 375, color=INK, sw=2.2)

    svg.rect(850, 140, 285, 142, DATA, rx=8)
    svg.tag("tensor X", 872, 156, 78)
    svg.text("Team-phase features", 995, 187, 16.5, weight=700, color=INK)
    tensor_table(svg, 902, 212)
    svg.text("one row per (match, team, phase)", 995, 318, 11.5, color=MUTE)

    svg.rect(850, 340, 285, 135, GRAPH, rx=8)
    svg.tag("tensor A", 872, 356, 78)
    svg.text("Dynamic team relations", 995, 387, 16.5, weight=700, color=INK)
    small_graph(svg, 1060, 414, 0.62)
    svg.text("proximity", 890, 458, 10.5, weight=700, color=PROX, anchor="start")
    svg.text("combat", 960, 458, 10.5, weight=700, color=COMBAT, anchor="start")
    svg.text("co-pressure", 1025, 458, 10.5, weight=700, color=COPRESS, anchor="start")

    svg.arrow(1150, 205, 1220, 295, color=INK, sw=2.2)
    svg.arrow(1150, 405, 1220, 335, color=INK, sw=2.2)

    svg.rect(1225, 235, 235, 160, MODEL, rx=8)
    svg.tag("model", 1245, 252, 62)
    svg.text("DynamicSTGNN", 1342, 289, 18, weight=700, color=INK)
    svg.multiline(
        ["X and A sequences", "2 gated ST blocks", "causal phase state"], 1342, 325, 12.5, step=23
    )

    svg.arrow(1475, 315, 1530, 315, color=INK, sw=2.2)

    svg.rect(1535, 225, 120, 92, OUT, rx=8)
    svg.text("hazard", 1595, 253, 15, weight=700, color=INK)
    svg.text("p(elim", 1595, 282, 12.5, italic=True, color=SUB)
    svg.text("before k+1)", 1595, 303, 12.5, italic=True, color=SUB)
    svg.rect(1535, 340, 120, 58, "#FFFFFF", stroke="#AAB7C4", sw=1.1, dash="5 4", rx=8)
    svg.text("rank / top-k", 1595, 362, 11.5, color=MUTE)
    svg.text("auxiliary", 1595, 382, 10.5, color=MUTE)

    # Evaluation strip.
    svg.rect(150, 535, 1305, 52, "#FFFFFF", stroke=GRID, sw=1.2, rx=8)
    svg.text("evaluation conditions", 178, 561, 12, weight=700, color=MUTE, anchor="start")
    chips = [
        ("HGB feature-only", 360, 548, 136, "#FFFFFF"),
        ("no-edge neural", 520, 548, 126, "#FFFFFF"),
        ("proximity", 670, 548, 95, "#EEF5FF"),
        ("prox + combat", 790, 548, 126, "#FFF0F0"),
        ("full graph", 940, 548, 98, "#F4EEFC"),
        ("aligned test cells", 1085, 548, 140, "#FFFFFF"),
    ]
    for label, x, y, w, fill in chips:
        svg.rect(x, y, w, 26, fill, stroke="#B9C4CF", sw=1.0, rx=13)
        svg.text(label, x + w / 2, y + 13, 10.5, weight=600, color=SUB)

    svg.save(path)


def fig_architecture(path: Path) -> None:
    svg = Svg(1680, 760)
    svg.text(
        "DynamicSTGNN architecture for phase-level team elimination hazard",
        840,
        38,
        22,
        weight=700,
        color=INK,
    )
    svg.line(80, 68, 1600, 68, color=GRID, sw=1.2)

    # Input pack.
    svg.rect(55, 105, 360, 565, "#FFFFFF", stroke=GRID, sw=1.2, rx=10)
    svg.text("Input tensor pack", 235, 135, 18, weight=700, color=INK)
    tensor_stack(svg, 105, 175, 125, 72, DATA, "X")
    svg.text("T x N x F", 290, 197, 14, weight=700, color=INK)
    svg.text("8 x <=128 x 66", 290, 225, 12, color=MUTE)
    tensor_stack(svg, 105, 305, 125, 72, GRAPH, "A")
    svg.text("T x E x N x N", 295, 327, 14, weight=700, color=INK)
    svg.text("8 x 3 x <=128 x <=128", 295, 355, 12, color=MUTE)
    svg.text("relation channels", 235, 424, 13, weight=700, color=INK)
    legend = [(PROX, "proximity"), (COMBAT, "combat directed"), (COPRESS, "co-pressure")]
    for idx, (color, label) in enumerate(legend):
        y = 455 + idx * 33
        if idx == 1:
            svg.arrow(115, y, 160, y, color=color, sw=2.4, head=7)
        else:
            svg.line(115, y, 160, y, color=color, sw=2.4, dash="5 4" if idx == 2 else None)
        svg.text(label, 175, y, 12, color=SUB, anchor="start")
    svg.rect(105, 585, 260, 42, WARN, stroke="#D3BC5F", sw=1.1, rx=7)
    svg.text("target mask keeps only alive cut phases", 235, 606, 11.5, weight=600, color=SUB)

    svg.arrow(430, 390, 500, 390, color=INK, sw=2.3)

    # Core network.
    svg.rect(510, 105, 585, 565, "#FFFFFF", stroke=GRID, sw=1.2, rx=10)
    svg.text("Spatio-temporal core", 802, 135, 18, weight=700, color=INK)
    svg.rect(640, 165, 325, 56, SOFT, rx=8)
    svg.text("node embedding", 802, 187, 15, weight=700, color=INK)
    svg.text("Linear 66 -> 64", 802, 209, 11.5, color=MUTE)
    svg.arrow(802, 225, 802, 255, color=INK, sw=1.8)

    block_y = [265, 445]
    for idx, y in enumerate(block_y, start=1):
        svg.rect(555, y, 495, 135, DB, rx=9)
        svg.tag(f"ST block {idx}", 575, y + 14, 82)
        x0 = 595
        labels = [
            ("causal temporal", MODEL, 665),
            ("hetero graph mix", GRAPH, 805),
            ("causal temporal", MODEL, 945),
        ]
        for label, fill, cx in labels:
            svg.rect(cx - 58, y + 54, 116, 42, fill, stroke="#9EADBA", sw=1.1, rx=7)
            svg.text(label, cx, y + 75, 10.8, weight=700, color=SUB)
        svg.arrow(x0 + 130, y + 75, x0 + 152, y + 75, color=INK, sw=1.5, head=6)
        svg.arrow(x0 + 270, y + 75, x0 + 292, y + 75, color=INK, sw=1.5, head=6)
        svg.circle(1015, y + 75, 15, "#FFFFFF", stroke=RULE, sw=1.2)
        svg.text("+", 1015, y + 75, 13, weight=700, color=INK)
        svg.path(
            f"M 592 {y + 117} C 700 {y + 145}, 925 {y + 145}, 1015 {y + 92}", stroke=MUTE, sw=1.2
        )
        svg.text("residual + LayerNorm", 805, y + 122, 10.5, color=MUTE)
        if idx == 1:
            svg.arrow(802, y + 140, 802, 438, color=INK, sw=1.8)

    svg.rect(638, 610, 330, 38, "#F9FBFD", stroke="#B9C4CF", sw=1.1, rx=7)
    svg.text(
        "causal padding: state at phase k sees phases <= k", 803, 629, 11.5, weight=600, color=SUB
    )

    svg.arrow(1110, 390, 1180, 390, color=INK, sw=2.3)

    # Heads and outputs.
    svg.rect(1195, 105, 430, 565, "#FFFFFF", stroke=GRID, sw=1.2, rx=10)
    svg.text("Prediction heads", 1410, 135, 18, weight=700, color=INK)
    svg.rect(1245, 190, 330, 150, OUT, rx=9)
    svg.text("next-phase hazard head", 1410, 218, 16, weight=700, color=INK)
    svg.text("uses hidden states for cut phases 1-7", 1410, 246, 12, color=MUTE)
    for idx in range(7):
        x = 1280 + idx * 42
        svg.circle(x, 292, 13, "#FFFFFF", stroke=RULE, sw=1.2)
        svg.text(str(idx + 1), x, 292, 9.5, weight=700, color=SUB)
        if idx < 6:
            svg.line(x + 14, 292, x + 28, 292, color=GRID, sw=1.4)
    svg.text("p(eliminated before phase k+1)", 1410, 323, 11.5, italic=True, color=SUB)

    svg.rect(1245, 380, 150, 70, "#FFFFFF", stroke="#AAB7C4", sw=1.1, dash="5 4", rx=8)
    svg.text("final-rank", 1320, 405, 12, weight=700, color=MUTE)
    svg.text("regularizer", 1320, 430, 10.5, color=MUTE)
    svg.rect(1425, 380, 150, 70, "#FFFFFF", stroke="#AAB7C4", sw=1.1, dash="5 4", rx=8)
    svg.text("top-k", 1500, 405, 12, weight=700, color=MUTE)
    svg.text("regularizer", 1500, 430, 10.5, color=MUTE)

    svg.rect(1245, 510, 330, 78, "#F9FBFD", stroke="#B9C4CF", sw=1.1, rx=8)
    svg.text(
        "loss = hazard BCE + 0.3 rank L1 + 0.3 top-k BCE", 1410, 535, 11.5, weight=600, color=SUB
    )
    svg.text("evaluation reports AUC, AP, BCE, Brier, ECE", 1410, 563, 11.5, color=MUTE)

    svg.save(path)


def render(svg_path: Path, png_path: Path, width: int, height: int, scale: float = 2.5) -> None:
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            f"--screenshot={png_path}",
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={scale}",
            "--default-background-color=FFFFFFFF",
            f"file://{svg_path.resolve()}",
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    out = Path("paper/figures")
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)

    pipeline_svg = src / "pipeline_overview.svg"
    arch_svg = src / "dynstgnn_architecture.svg"
    fig_pipeline(pipeline_svg)
    fig_architecture(arch_svg)
    render(pipeline_svg, out / "pipeline_overview.png", 1680, 620)
    render(arch_svg, out / "dynstgnn_architecture.png", 1680, 760)
    print("written: pipeline_overview.png, dynstgnn_architecture.png (+ SVG sources)")


if __name__ == "__main__":
    main()
