#!/usr/bin/env python3
"""Concept figure of the SAC 2027 manuscript: what each evaluation unit compares and what each loss can learn.

The running example of Section 2: two matches at one phase, each with one team eliminated in the next phase and one
survivor (scores 0.60/0.40 in match A, 0.50/0.30 in match B).  Panel A marks the same-match pairs (within-match AUC) and
the different-match pairs (also in within-phase AUC); panel B adds 0.30 to every score of match A, which leaves the
same-match pairs in order and flips one different-match pair; panel C states what a risk-set loss and cross-entropy
see of such a shift.  No data are read.

Usage: python tools/sac2027_figure_concept.py <output-dir>
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))

from matplotlib.patches import FancyBboxPatch  # noqa: E402

import pubg_lab.projects.battle_royale.make_result_figures_v2 as F  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper/figures"
ELIM, SURV, GOOD, BAD, GREY, LIGHT = (
    "#D55E00",
    "#2F5597",
    "#333333",
    "#C0392B",
    "#666666",
    "#F2F2F2",
)
XA, XB = 0.28, 0.72


def scores_panel(ax, title, scores, flipped):
    """scores: {'A': (eliminated, survivor), 'B': (...)}; flipped: different-match pairs drawn as wrongly ordered."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1.02)
    ax.axis("off")
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left", pad=2)
    for x, name in ((XA, "Match A"), (XB, "Match B")):
        ax.text(x, 0.17, name, ha="center", va="bottom", fontsize=7, color=GREY)
    pos = {
        (m, r): ((XA if m == "A" else XB), v)
        for m, (e, s) in scores.items()
        for r, v in (("e", e), ("s", s))
    }
    # different-match pairs first (behind): eliminated of one match against the survivor of the other
    for em, sm in (("A", "B"), ("B", "A")):
        (x0, y0), (x1, y1) = pos[(em, "e")], pos[(sm, "s")]
        wrong = (em, sm) in flipped
        ax.plot(
            [x0, x1],
            [y0, y1],
            ls=(0, (3, 2)),
            lw=1.0 if wrong else 0.7,
            color=BAD if wrong else "#9A9A9A",
            zorder=1,
        )
        if wrong:
            ax.text(0.63, 0.69, "wrong order", ha="center", va="bottom", fontsize=7, color=BAD)
    for m in ("A", "B"):  # same-match pair: a short bracket beside the two points
        (x, ye), (_, ys) = pos[(m, "e")], pos[(m, "s")]
        side = -0.07 if m == "A" else 0.07
        ax.plot(
            [x + side * 0.6, x + side, x + side, x + side * 0.6],
            [ye, ye, ys, ys],
            color=GOOD,
            lw=0.9,
            zorder=2,
        )
    for (m, r), (x, y) in pos.items():
        ax.plot(
            [x],
            [y],
            marker="o",
            ms=6,
            mfc=ELIM if r == "e" else "white",
            mec=ELIM if r == "e" else SURV,
            mew=1.1,
            zorder=3,
        )
        ax.text(
            x + (-0.2 if m == "A" else 0.2),
            y,
            f"{y:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color=GOOD,
        )


def loss_box(fig, left, bottom, width, height, title, lines):
    ax = F.add_axes_cm(fig, left, bottom, width, height)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.96,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            fc=LIGHT,
            ec="#BBBBBB",
            lw=0.6,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.06, 0.9, title, ha="left", va="top", fontsize=7, fontweight="bold", transform=ax.transAxes
    )
    ax.text(
        0.06,
        0.7,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=7,
        color=GREY,
        transform=ax.transAxes,
        linespacing=1.2,
    )


def main():
    F.setup_style()
    W, H = 8.4, 6.6
    fig = F.new_figure(W, H)
    top_b, top_h = 2.95, 3.15
    a = F.add_axes_cm(fig, 0.05, top_b, 4.05, top_h)
    scores_panel(
        a, "A  Two matches at one phase", {"A": (0.60, 0.40), "B": (0.50, 0.30)}, flipped=set()
    )
    b = F.add_axes_cm(fig, 4.3, top_b, 4.05, top_h)
    scores_panel(
        b,
        "B  Add 0.30 to every score of A",
        {"A": (0.90, 0.70), "B": (0.50, 0.30)},
        flipped={("B", "A")},
    )
    # legend strip
    lg = F.add_axes_cm(fig, 0.05, 2.42, 8.3, 0.42)
    lg.axis("off")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    items = (
        (0.02, "o", ELIM, ELIM, "eliminated next phase"),
        (0.375, "o", "white", SURV, "survives"),
    )
    for x, mk, fc, ec, label in items:
        lg.plot([x], [0.5], marker=mk, ms=5, mfc=fc, mec=ec, mew=1.0)
        lg.text(x + 0.025, 0.5, label, va="center", fontsize=7, color=GOOD)
    lg.plot([0.56, 0.56, 0.585], [0.8, 0.2, 0.2], color=GOOD, lw=0.9)
    lg.text(0.6, 0.5, "same match", va="center", fontsize=7, color=GOOD)
    lg.plot([0.765, 0.81], [0.5, 0.5], ls=(0, (3, 2)), lw=0.8, color="#9A9A9A")
    lg.text(0.82, 0.5, "other match", va="center", fontsize=7, color=GOOD)
    # panel C: what each loss sees of the shift in B
    fig.text(
        0.05 / W,
        2.3 / H,
        "C  What each training loss sees of that shift",
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="top",
    )
    loss_box(
        fig,
        0.05,
        0.05,
        4.1,
        1.95,
        "Risk-set loss (Cox type)",
        (
            "compares eliminated teams",
            "with survivors of their match:",
            "unchanged, no signal for",
            "the match's risk level",
        ),
    )
    loss_box(
        fig,
        4.25,
        0.05,
        4.1,
        1.95,
        "Cross-entropy (BCE)",
        (
            "compares probabilities with",
            "labels: changed, so it learns",
            "the match's level from its",
            "number of eliminations",
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    notes = F.save_and_check(fig, OUT / "fig_concept", W, H, True)
    print("\n".join(notes) if notes else "QA ok")


if __name__ == "__main__":
    main()
