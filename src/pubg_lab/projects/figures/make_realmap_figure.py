"""Real-match graph on the actual Erangel (Baltic_Main) map.

Renders what the model ACTUALLY sees: real team positions from a real match,
our three relation channels (proximity / combat / co-pressure) as edges, and the
current safe zone — all projected onto the in-game map image.

Projection (verified against webapp/network_analysis):
    ARENA_SIZE_M = 8160 ; no y-flip ; no rotation for Baltic_Main
    px = pos_x / 8160 * IMG_W ;  py = pos_y / 8160 * IMG_H

Snapshot choice: per phase we plot the moment the teams are settled INSIDE the
zone (max teams inside_blue, with the winner inside) — so the winner is never
drawn outside the safe area, and nearby teams produce visible proximity edges.
Safe zone is the TRUE current circle, recovered by least squares from each
team's signed distance to the blue boundary:  |pos_i - C| = R - dist_blue_i.

Outputs (docs/presentation_figures/):
    fig_realmap_hero[_deck].png       — one representative phase, large
    fig_realmap_evolution[_deck].png  — 4 phases (zone + graph contracting)
    fig_realmap_heroP{1,3,4}.png      — candidate hero phases
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

from pubg_lab.data.stdb.conn import connect

ARENA = 8160.0
MID = "f6513861-33a3-4a86-a082-c2026e2a5a4a"  # Baltic_Main squad; winner stays in-zone all phases
HERO_PHASE = 4
EVO_PHASES = (1, 3, 5, 6)
MAP_PATH = "webapp/public/maps/baltic_main_2k.jpg"
OUT = Path("docs/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

# deck color tokens
PROX, COMBAT, COPRESS = "#1d4ed8", "#d62728", "#d97706"
NODE, NODE_EDGE, WINNER = "#1f2430", "#ffffff", "#d97706"
INK, MUTED = "#1f2430", "#6b7280"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def pick_snapshot(cur, p, win):
    """Snapshot in phase p where the winner is inside the zone and the most teams are settled inside."""
    cur.execute(
        """WITH wt AS (SELECT snapshot_t FROM stdb.team_snapshot_features
                     WHERE match_id=%s AND phase_idx=%s AND team_id=%s AND inside_blue)
                   SELECT s.snapshot_t FROM stdb.team_snapshot_features s JOIN wt ON s.snapshot_t=wt.snapshot_t
                   WHERE s.match_id=%s AND s.phase_idx=%s
                   GROUP BY 1 ORDER BY sum(CASE WHEN s.inside_blue THEN 1 ELSE 0 END) DESC, s.snapshot_t ASC
                   LIMIT 1""",
        (MID, p, win, MID, p),
    )
    r = cur.fetchone()
    if r:
        return r[0]
    cur.execute(
        """SELECT snapshot_t FROM stdb.team_snapshot_features WHERE match_id=%s AND phase_idx=%s
                   GROUP BY 1 ORDER BY sum(CASE WHEN inside_blue THEN 1 ELSE 0 END) DESC, snapshot_t ASC
                   LIMIT 1""",
        (MID, p),
    )
    return cur.fetchone()[0]


def fit_zone(P, d):
    """Current safe circle from |pos-C| = R - dist_blue (least squares); robust fallback."""
    P = np.asarray(P, float)
    d = np.asarray(d, float)
    if len(P) >= 4:
        A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], -2 * d, -np.ones(len(d))])
        rhs = P[:, 0] ** 2 + P[:, 1] ** 2 - d**2
        a, b, R, _ = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if 50 < R < 8200 and np.std(np.hypot(P[:, 0] - a, P[:, 1] - b) + d) < 0.15 * R:
            return float(a), float(b), float(R)
    a, b = float(np.median(P[:, 0])), float(np.median(P[:, 1]))  # fallback
    R = float(np.median(np.hypot(P[:, 0] - a, P[:, 1] - b) + d))
    return a, b, max(R, 50.0)


def fetch():
    with connect() as c:
        cur = c.cursor()
        cur.execute(
            "SELECT team_id FROM stdb.team_phase_labels WHERE match_id=%s AND final_winner LIMIT 1",
            (MID,),
        )
        win = cur.fetchone()[0]
        cur.execute(
            """SELECT DISTINCT phase_idx FROM stdb.team_snapshot_features
                       WHERE match_id=%s AND phase_idx IS NOT NULL ORDER BY 1""",
            (MID,),
        )
        phases = [r[0] for r in cur.fetchall()]
        pos, zonefit = {}, {}
        for p in phases:
            t = pick_snapshot(cur, p, win)
            cur.execute(
                """SELECT team_id, pos_x, pos_y, dist_blue_boundary_m, alive_n
                           FROM stdb.team_snapshot_features WHERE match_id=%s AND phase_idx=%s AND snapshot_t=%s""",
                (MID, p, t),
            )
            rows = cur.fetchall()
            pos[p] = {
                tid: dict(x=float(x), y=float(y), alive=int(a or 0)) for tid, x, y, _, a in rows
            }
            P = [(float(x), float(y)) for _, x, y, _, _ in rows]
            d = [float(db) for *_, db, _ in rows]
            zonefit[p] = fit_zone(P, d)
        cur.execute(
            """SELECT phase_idx, edge_type, src_team_id, dst_team_id, weight,
                              avg_distance_m, damage_sum, kill_count, groggy_count
                       FROM stdb.team_phase_edge_cache WHERE match_id=%s AND phase_idx IS NOT NULL""",
            (MID,),
        )
        edges = {}
        for ph, et, s, dd, w, dist, dmg, kil, grog in cur.fetchall():
            edges.setdefault(ph, {}).setdefault(et, []).append(
                dict(
                    s=s,
                    d=dd,
                    w=float(w or 0),
                    dist=float(dist or 0),
                    dmg=float(dmg or 0),
                    kil=int(kil or 0),
                    grog=int(grog or 0),
                )
            )
        cur.execute(
            "SELECT team_id, final_rank FROM stdb.team_phase_labels WHERE match_id=%s GROUP BY 1,2",
            (MID,),
        )
        rank = {tid: int(r) for tid, r in cur.fetchall() if r is not None}
        cur.close()
    return pos, edges, zonefit, rank


def m2px(v, img):
    return v / ARENA * img


def draw_panel(
    ax,
    img,
    pos,
    edges,
    zonefit,
    rank,
    phase,
    show_prox=True,
    hero=False,
    crop_to="zone",
    gscale=1.0,
):
    IMG = img.shape[0]
    ax.imshow(img, extent=[0, IMG, IMG, 0], zorder=0)  # origin top-left, no flip
    ax.add_patch(plt.Rectangle((0, 0), IMG, IMG, facecolor="white", alpha=0.24, zorder=1))

    pp = pos[phase]
    P = lambda tid: (m2px(pp[tid]["x"], IMG), m2px(pp[tid]["y"], IMG))

    # ---- safe zone: TRUE current circle (winner is inside it) ----
    cx, cy, r = zonefit[phase]
    ax.add_patch(
        Circle(
            (m2px(cx, IMG), m2px(cy, IMG)),
            m2px(r, IMG),
            fill=False,
            ec=PROX,
            lw=2.2,
            ls=(0, (6, 4)),
            alpha=0.95,
            zorder=2,
        )
    )

    # ---- edges: proximity (light, weight-scaled) < copressure < combat (on top) ----
    ep = edges.get(phase, {})
    if show_prox:
        pw = [e["w"] for e in ep.get("proximity", []) if e["s"] in pp and e["d"] in pp]
        wmax = max(pw + [1e-9])
        for e in ep.get("proximity", []):
            if e["s"] in pp and e["d"] in pp:
                wn = e["w"] / wmax
                if wn < 0.06:
                    continue
                (x0, y0), (x1, y1) = P(e["s"]), P(e["d"])
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=PROX,
                    lw=(0.7 + 1.9 * wn) * gscale,
                    alpha=0.18 + 0.44 * wn,
                    zorder=3,
                    solid_capstyle="round",
                )  # visible but weight-scaled
    for e in ep.get("copressure", []):
        if e["s"] in pp and e["d"] in pp:
            (x0, y0), (x1, y1) = P(e["s"]), P(e["d"])
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=COPRESS,
                lw=2.6 * gscale,
                alpha=0.92,
                zorder=5,
                solid_capstyle="round",
            )
    cmax = max([e["dmg"] for e in ep.get("combat", [])] + [1])
    for e in ep.get("combat", []):
        if e["s"] in pp and e["d"] in pp:
            (x0, y0), (x1, y1) = P(e["s"]), P(e["d"])
            lw = (1.6 + 3.0 * (e["dmg"] / cmax)) * gscale
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=(13 if hero else 8) * gscale,
                    color=COMBAT,
                    lw=lw,
                    alpha=0.95,
                    shrinkA=8 * gscale,
                    shrinkB=8 * gscale,
                    zorder=6,
                )
            )

    # ---- nodes: real team positions, size ~ players alive ----
    for tid, t in pp.items():
        x, y = P(tid)
        is_win = rank.get(tid) == 1
        ms = ((78 if hero else 34) + (26 if hero else 12) * (t["alive"] - 1)) * gscale**2
        ax.scatter(
            [x], [y], s=ms * 1.7, facecolor="white", edgecolor="none", alpha=0.85, zorder=7.5
        )
        ax.scatter(
            [x],
            [y],
            s=ms,
            facecolor=WINNER if is_win else NODE,
            edgecolor=NODE_EDGE,
            linewidth=(1.6 if hero else 0.9) * gscale,
            zorder=8,
        )
        if is_win:
            ax.scatter(
                [x],
                [y],
                s=ms * 3.4,
                facecolor="none",
                edgecolor=WINNER,
                linewidth=1.8 * gscale,
                zorder=8.5,
            )
            ax.annotate(
                "WINNER",
                (x, y),
                textcoords="offset points",
                xytext=(0, 16 if hero else 11),
                ha="center",
                fontsize=(11 if hero else 7.5),
                fontweight="bold",
                color=WINNER,
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
            )

    # ---- framing ----
    xs = [m2px(t["x"], IMG) for t in pp.values()]
    ys = [m2px(t["y"], IMG) for t in pp.values()]
    zc = (m2px(cx, IMG), m2px(cy, IMG))
    zr = m2px(r, IMG)
    if crop_to == "teams":
        pad = m2px(420, IMG)
        lo_x, hi_x = min(xs) - pad, max(xs) + pad
        lo_y, hi_y = min(ys) - pad, max(ys) + pad
    else:
        pad = m2px(700, IMG)
        lo_x, hi_x = min(min(xs), zc[0] - zr) - pad, max(max(xs), zc[0] + zr) + pad
        lo_y, hi_y = min(min(ys), zc[1] - zr) - pad, max(max(ys), zc[1] + zr) + pad
    side = min(max(hi_x - lo_x, hi_y - lo_y), IMG)
    mx, my = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
    mx = min(max(mx, side / 2), IMG - side / 2)
    my = min(max(my, side / 2), IMG - side / 2)
    ax.set_xlim(mx - side / 2, mx + side / 2)
    ax.set_ylim(my + side / 2, my - side / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#cbd2d9")
        s.set_linewidth(1.0)
    return len(pp), r, side


def scalebar(ax, img_px):
    km_px = m2px(1000.0, img_px)
    x0 = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.06
    y0 = ax.get_ylim()[0] - (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.07
    ax.plot([x0, x0 + km_px], [y0, y0], color=INK, lw=2.5, solid_capstyle="butt", zorder=10)
    ax.text(
        x0 + km_px / 2, y0, "1 km", ha="center", va="bottom", fontsize=7.5, color=INK, zorder=10
    )


def hero_figure(img, pos, edges, zonefit, rank, bare, phase=HERO_PHASE, name=None):
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    n, r, side = draw_panel(
        ax, img, pos, edges, zonefit, rank, phase, show_prox=True, hero=True, crop_to="zone"
    )  # frame the safe circle so teams sit inside it
    scalebar(ax, img.shape[0])
    if not bare:
        ax.set_title(
            f"PUBG match on Erangel — what the model sees at Phase {phase}",
            fontsize=13.5,
            color=INK,
            pad=10,
            loc="left",
            fontweight="bold",
        )
        ec = edges.get(phase, {})
        ax.text(
            0.0,
            1.005,
            f"{n} teams · {len(ec.get('proximity', []))} proximity · "
            f"{len(ec.get('combat', []))} combat · {len(ec.get('copressure', []))} co-pressure edges "
            f"· safe-zone radius {r:,.0f} m",
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
            va="bottom",
        )
        handles = [
            Line2D([0], [0], color=PROX, lw=2, ls=(0, (6, 4)), label="Safe zone (current)"),
            Line2D([0], [0], color=PROX, lw=1.2, alpha=0.5, label="Proximity  (who is near whom)"),
            Line2D(
                [0], [0], color=COPRESS, lw=2, label="Co-pressure  (squeezed together by the zone)"
            ),
            Line2D([0], [0], color=COMBAT, lw=2.4, label="Combat  (directed: who damages whom)"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=NODE,
                markeredgecolor="white",
                markersize=9,
                label="Team  (size = players alive)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=WINNER,
                markeredgecolor=WINNER,
                markersize=10,
                label="Match winner",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            fontsize=8.0,
            edgecolor="#cbd2d9",
            handlelength=2.0,
            borderpad=0.7,
        )
    fig.tight_layout()
    if name is None:
        name = "fig_realmap_hero_deck.png" if bare else "fig_realmap_hero.png"
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", pad_inches=0.04 if bare else 0.1)
    print("wrote", name)
    plt.close(fig)


def evolution_figure(img, pos, edges, zonefit, rank, bare):
    show = [p for p in EVO_PHASES if p in pos]
    fig, axes = plt.subplots(1, len(show), figsize=(4.4 * len(show), 4.9))
    for ax, ph in zip(np.atleast_1d(axes), show):
        n, r, side = draw_panel(
            ax, img, pos, edges, zonefit, rank, ph, show_prox=(ph <= 3), hero=False
        )
        ec = edges.get(ph, {})
        ax.set_title(f"Phase {ph}", fontsize=12.5, color=INK, loc="left", fontweight="bold", pad=16)
        ax.text(
            0.0,
            1.012,
            f"{n} teams · zone {r:,.0f} m",
            transform=ax.transAxes,
            fontsize=8.4,
            color=MUTED,
            va="bottom",
        )
        ax.text(
            0.0,
            -0.045,
            f"combat {len(ec.get('combat', []))} · "
            f"co-press {len(ec.get('copressure', []))} · prox {len(ec.get('proximity', []))}",
            transform=ax.transAxes,
            fontsize=8.0,
            color=MUTED,
            va="top",
        )
    if not bare:
        fig.suptitle(
            "The same relations, a contracting habitat: real match across phases on Erangel",
            fontsize=13.5,
            color=INK,
            x=0.012,
            ha="left",
            fontweight="bold",
            y=1.02,
        )
        handles = [
            Line2D([0], [0], color=PROX, lw=2, ls=(0, (6, 4)), label="Safe zone"),
            Line2D([0], [0], color=PROX, lw=1.2, alpha=0.5, label="Proximity"),
            Line2D([0], [0], color=COPRESS, lw=2, label="Co-pressure"),
            Line2D([0], [0], color=COMBAT, lw=2.4, label="Combat (directed)"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=WINNER,
                markeredgecolor=WINNER,
                markersize=9,
                label="Winner",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            frameon=False,
            fontsize=8.6,
            bbox_to_anchor=(0.5, -0.04),
            handlelength=2.0,
        )
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    name = "fig_realmap_evolution_deck.png" if bare else "fig_realmap_evolution.png"
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", pad_inches=0.04 if bare else 0.1)
    print("wrote", name)
    plt.close(fig)


def panel_image(img, pos, edges, zonefit, rank, phase):
    """One bare map panel (no baked text except the WINNER pin) for HTML-labelled grids."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    draw_panel(
        ax, img, pos, edges, zonefit, rank, phase, show_prox=(phase <= 3), hero=False, gscale=1.25
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / f"fig_realmap_p{phase}.png", dpi=175, bbox_inches="tight", pad_inches=0.015)
    print(f"wrote fig_realmap_p{phase}.png")
    plt.close(fig)


def report_evo_numbers(pos, edges, zonefit):
    """Print the numbers the deck's evolution-stats strip should quote (so HTML stays in sync)."""

    def ec(p, t):
        return len(edges.get(p, {}).get(t, []))

    phs = sorted(pos)
    combat = [ec(p, "combat") for p in phs]
    copress = [ec(p, "copressure") for p in phs]
    teams = [len(pos[p]) for p in phs]
    rad = [zonefit[p][2] for p in phs]
    print("=== EVO STATS (update HTML) ===")
    print(f"  phases: {phs}")
    print(f"  combat:  {combat[0]} -> {combat[-1]}  (per phase {combat})")
    print(
        f"  copress: {copress[0]} -> {max(copress)}  peak@P{phs[int(np.argmax(copress))]} (per phase {copress})"
    )
    print(f"  teams:   {teams[0]} -> {teams[-1]}")
    print(f"  radius:  {rad[0]:,.0f} -> {rad[-1]:,.0f} m  (per phase {[round(x) for x in rad]})")


def main():
    if not os.path.exists(MAP_PATH):
        raise SystemExit(f"map image not found: {MAP_PATH}")
    img = mpimg.imread(MAP_PATH)
    pos, edges, zonefit, rank = fetch()
    print("teams/phase:", {ph: len(pos[ph]) for ph in sorted(pos)})
    for ph in (1, 3, 4):
        if ph in pos:
            hero_figure(
                img, pos, edges, zonefit, rank, True, phase=ph, name=f"fig_realmap_heroP{ph}.png"
            )
    for bare in (False, True):
        hero_figure(img, pos, edges, zonefit, rank, bare)
        evolution_figure(img, pos, edges, zonefit, rank, bare)
    for ph in EVO_PHASES:  # bare single panels for the HTML-labelled grid
        if ph in pos:
            panel_image(img, pos, edges, zonefit, rank, ph)
    report_evo_numbers(pos, edges, zonefit)


if __name__ == "__main__":
    main()
