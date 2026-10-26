"""Render the frozen training graph and held-out coverage, without refitting."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
import base64
import io
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pubg-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pubg-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba, Normalize
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np

from pubg_lab.spatial.erangel.erangel_100m_extract import OUT, ROOT, WIDTH, EXTENT

COLORS = {
    "repeated_observation": "#39d3cb",
    "flagged": "#ff9c42",
    "low_support": "#ffe575",
    "unobserved": "#ed638d",
}


def main():
    graph = json.loads((OUT / "graph.json").read_text())
    m = json.loads((OUT / "metrics.json").read_text())
    nodes = graph["nodes"]
    xy = np.array([[n["x_m"], n["y_m"]] for n in nodes])
    bg = (
        Image.open(ROOT / "data/spatial/erangel/source/official_notext.png")
        .convert("RGB")
        .resize((1600, 1600))
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#101e2c",
            "axes.facecolor": "#101e2c",
            "text.color": "#e9f1f7",
            "axes.labelcolor": "#c8d5df",
            "xtick.color": "#c8d5df",
            "ytick.color": "#c8d5df",
            "axes.edgecolor": "#51606d",
            "savefig.facecolor": "#101e2c",
        }
    )

    def base(ax, title, bounds=None):
        ax.imshow(bg, extent=(0, EXTENT, EXTENT, 0), alpha=0.55)
        ax.set(xlim=(0, EXTENT), ylim=(EXTENT, 0), title=title, xlabel="x (m)", ylabel="y (m)")
        if bounds:
            x0, y0, size = bounds
            ax.set(xlim=(x0, x0 + size), ylim=(y0 + size, y0))
        ax.set_aspect("equal")

    def cells(ax, kind, opacity=0.64):
        rgba = np.zeros((WIDTH, WIDTH, 4))
        for n in nodes:
            idx = n["node_id"]
            land = n["historical_land_fraction"] >= 0.75
            if kind == "support":
                if n["match_support"]:
                    rgba[idx // WIDTH, idx % WIDTH] = plt.cm.viridis(
                        min(1, np.log10(1 + n["match_support"]) / np.log10(2142))
                    )
                    rgba[idx // WIDTH, idx % WIDTH, 3] = 0.82
                elif land:
                    rgba[idx // WIDTH, idx % WIDTH] = to_rgba(COLORS["unobserved"], 0.8)
            elif land or n["match_support"]:
                rgba[idx // WIDTH, idx % WIDTH] = to_rgba(COLORS[n["category"]], opacity)
        # pcolormesh preserves the final 60m boundary cells.
        bounds = np.minimum(np.arange(WIDTH + 1) * 100, EXTENT)
        ax.pcolormesh(
            bounds, bounds, np.zeros((WIDTH, WIDTH)), color=rgba.reshape(-1, 4), shading="flat"
        )
        mesh = ax.collections[-1]
        mesh.set_array(None)
        mesh.set_facecolors(rgba.reshape(-1, 4))

    def links(ax, threshold=3):
        for key, color, style, lw in [
            ("observed_edges", "#44ece7", "solid", 0.65),
            ("inferred_edges", "#f181ff", "dashed", 1.15),
        ]:
            ee = [
                e for e in graph[key] if key != "observed_edges" or e["match_support"] >= threshold
            ]
            segments = np.array([[xy[e["source"]], xy[e["target"]]] for e in ee])
            if len(segments):
                ax.add_collection(
                    LineCollection(
                        segments, colors=color, linewidths=lw, linestyles=style, alpha=0.8
                    )
                )

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.7), layout="constrained")
    base(axes[0], "A  Training coverage")
    cells(axes[0], "support")
    sm = plt.cm.ScalarMappable(norm=Normalize(0, np.log10(2142)), cmap="viridis")
    cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.035, pad=0.04)
    cb.set_ticks(np.log10(np.array([0, 10, 100, 1000]) + 1), labels=["0", "10", "100", "1000"])
    cb.set_label("Distinct training matches per cell")
    base(axes[1], "B  Observed and inferred connections")
    links(axes[1], 3)
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#44ece7", label="Observed: at least 3 matches"),
            Line2D([0], [0], color="#f181ff", ls="--", label="Inferred: historical land adjacency"),
        ],
        loc="lower left",
        facecolor="#122334",
        labelcolor="white",
        fontsize=8,
    )
    base(axes[2], "C  Uncertainty of cell representation")
    cells(axes[2], "uncertainty")
    labels = {
        "repeated_observation": "10+ matches; no extra flag",
        "flagged": "10+ matches; extra warning",
        "low_support": "1–9 matches",
        "unobserved": "Unobserved historical land",
    }
    axes[2].legend(
        handles=[
            Line2D([0], [0], marker="s", ls="", color=v, label=labels[k]) for k, v in COLORS.items()
        ],
        loc="lower left",
        facecolor="#122334",
        labelcolor="white",
        fontsize=8,
    )
    fig.suptitle(
        f"Erangel 100 m spatial approximation | {m['n_train']:,} training / {m['n_validation']:,} held-out matches\n40.2 inferred from date; historical rasters are not exact patch assets",
        fontsize=17,
    )
    fig.supxlabel(
        "Training data only. Nonvehicle means no vehicle recorded. Lines connect cell centers, not verified physical routes.",
        fontsize=11,
    )
    fig.savefig(OUT / "overview.png", dpi=165)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2), layout="constrained")
    for ax, title, bounds in zip(
        axes,
        ["West bridge / coast", "Pochinki / fields", "Northeast / Stalber"],
        [(2800, 5100, 1600), (3100, 3500, 1600), (5700, 500, 1600)],
    ):
        base(ax, title, bounds)
        cells(ax, "uncertainty", 0.30)
        links(ax, 3)
    fig.suptitle(
        "100 m detail | cyan: observed (3+ matches), magenta dashed: inferred", fontsize=16
    )
    fig.supxlabel(
        "Orange: warning flag · yellow: low support · pink: unobserved historical land · teal: repeated observation\nBridge, terrain, building floors and collision geometry remain unresolved.",
        fontsize=11,
    )
    fig.savefig(OUT / "detail.png", dpi=165)
    plt.close(fig)
    groups = [("All", m["validation_counts"])] + [
        (k, v) for k, v in m["validation_by_match_type"].items()
    ]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]}
    )
    observed_values = [100 * v["transitions_observed"] / v["transitions"] for _, v in groups]
    ax.barh(range(len(groups)), observed_values, color="#44d8cf")
    for i, value in enumerate(observed_values):
        ax.text(50, i, f"{value:.3f}%", ha="center", va="center", color="#101e2c", fontsize=12)
    ax.set(
        yticks=range(len(groups)),
        yticklabels=[f"{k} (n={v['transitions']:,})" for k, v in groups],
        xlim=(0, 100),
        xlabel="Observed in training (%)",
        title="All held-out nonself transitions",
    )
    left = np.zeros(len(groups))
    for label, color in [("inferred", "#d27ce8"), ("uncovered", "#ed638d")]:
        values = np.array(
            [100 * v.get("transitions_" + label, 0) / v["transitions"] for _, v in groups]
        )
        ax2.barh(range(len(groups)), values, left=left, color=color, label=label)
        left += values
    for i, (_, v) in enumerate(groups):
        ax2.text(
            left[i] + 0.003,
            i,
            f"{v.get('transitions_inferred', 0)} / {v.get('transitions_uncovered', 0)}",
            va="center",
            fontsize=9,
        )
    ax2.set(
        yticks=[],
        xlim=(0, 0.2),
        xlabel="Share of ALL held-out transitions (%)",
        title="Not observed in training: magnified",
    )
    ax.invert_yaxis()
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", facecolor="#122334", labelcolor="white", fontsize=9)
    fig.suptitle("Validation coverage of the frozen graph — not prediction accuracy", fontsize=15)
    fig.supxlabel(
        "Right labels: inferred / uncovered transition counts. The two panels intentionally use different x-axis scales.",
        fontsize=10,
    )
    fig.savefig(OUT / "validation.png", dpi=160)
    plt.close(fig)
    stream = io.BytesIO()
    bg.save(stream, format="JPEG", quality=85)
    payload = dict(
        metrics=m,
        nodes=nodes,
        observed=[
            [
                e["source"],
                e["target"],
                e["match_support"],
                e["transition_count"],
                e["human_count"],
                e["bot_count"],
            ]
            for e in graph["observed_edges"]
        ],
        inferred=[[e["source"], e["target"]] for e in graph["inferred_edges"]],
        vehicle=[
            [e["source"], e["target"], e["match_support"]] for e in graph["vehicle_observed_edges"]
        ],
    )
    template = (ROOT / "tools/erangel_100m_viewer.html.template").read_text()
    (OUT / "viewer.html").write_text(
        template.replace("__DATA__", json.dumps(payload, separators=(",", ":"))).replace(
            "__MAP__", base64.b64encode(stream.getvalue()).decode()
        )
    )
    print("Created overview.png, detail.png, validation.png and self-contained viewer.html")


if __name__ == "__main__":
    main()
