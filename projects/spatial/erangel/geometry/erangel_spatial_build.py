"""Version-aware, diagnostic Erangel registration and observed-transition graph.

No mesh walkability labels are produced. Height residuals use player altitude,
not surveyed ground truth. Only fit matches/blocks influence calibration.
"""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, binary_erosion
from scipy.optimize import least_squares

ROOT = REPO_ROOT
OUT = ROOT / "reports/erangel_alignment_2026-10-15"
ASSETS = ROOT / "data/spatial/erangel/source"
EXTENT = 8160.0
CELL = 50


def risk_regions():
    pois = json.loads(
        (
            ROOT / "reports/spatial_source_inventory_2026-10-15/pubggs_erangel_embedded_map.json"
        ).read_text()
    )["layers"]["pois"]
    refs = {p["name"]: p for p in pois}
    # Coarse analyst-selected buffers, NOT official change polygons or exhaustive masks.
    radii = {
        "pochinki": 650,
        "georgopol": 800,
        "hospital": 450,
        "ruins": 500,
        "boatyard": 450,
        "mylta": 700,
        "stalber": 700,
        "shelter": 550,
        "sosnovkamilitarybase": 850,
        "ferrypier": 450,
    }
    rows = [
        dict(name=k, x_m=refs[k]["x"] / 100, y_m=refs[k]["y"] / 100, radius_m=v)
        for k, v in radii.items()
    ]
    rows += [
        dict(name="west_bridge", x_m=3530, y_m=5690, radius_m=500),
        dict(name="east_bridge", x_m=5460, y_m=5480, radius_m=500),
        dict(name="southwest_plain", x_m=2500, y_m=4900, radius_m=650),
    ]
    return rows


def risk_at(x, y, regions):
    r = np.zeros(np.shape(x), dtype=bool)
    for q in regions:
        r |= (x - q["x_m"]) ** 2 + (y - q["y_m"]) ** 2 <= q["radius_m"] ** 2
    return r


def stats(a):
    return (
        dict(
            n=len(a),
            median_abs_m=float(np.median(np.abs(a))),
            p90_abs_m=float(np.percentile(np.abs(a), 90)),
            mean_abs_m=float(np.mean(np.abs(a))),
            median_signed_m=float(np.median(a)),
        )
        if len(a)
        else {"n": 0}
    )


def main():
    h = np.asarray(Image.open(ASSETS / "height_2019.png")).astype(np.float32)
    rgb = np.asarray(Image.open(ASSETS / "official_notext.png").resize((1024, 1024))).astype(float)
    data = np.load(OUT / "telemetry_probe.npz")
    p = data["points"]
    pairs = data["pairs"]
    x, y, z = p[:, 2], p[:, 3], p[:, 4]
    regions = risk_regions()
    risk = risk_at(x, y, regions)
    water = (rgb[:, :, 2] > rgb[:, :, 0] * 1.25) & (rgb[:, :, 1] > rgb[:, :, 0] * 1.3)
    px = np.minimum((x / EXTENT * 1024).astype(int), 1023)
    py = np.minimum((y / EXTENT * 1024).astype(int), 1023)
    on_water = water[py, px]
    # Region holdout in addition to match holdout; calibration never sees those blocks.
    block = ((x // 500).astype(int) * 31 + (y // 500).astype(int) * 17) % 5
    fit = (p[:, 0] < 32) & (block != 0) & ~risk & ~on_water
    idx = np.where(fit)[0]
    # One deterministic point per 25m spatial cell prevents stationary crowding.
    _, first = np.unique(np.floor(p[idx, 2:4] / 25).astype(int), axis=0, return_index=True)
    idx = idx[first]
    rng = np.random.default_rng(42)
    if len(idx) > 6000:
        idx = np.sort(rng.choice(idx, 6000, replace=False))

    def sample(xx, yy, k=(0.0078, 1, 0, 0, 1)):
        a, b, dx, dy, s = k
        xx, yy = np.atleast_1d(xx), np.atleast_1d(yy)
        u = (xx * s + dx) * h.shape[1] / EXTENT
        v = (yy * s + dy) * h.shape[0] / EXTENT
        return map_coordinates(h, [v, u], order=1, mode="nearest", output=np.float64)

    def predict(xx, yy, k):
        return k[0] * (sample(xx, yy, k) - 32768) + k[1]

    base_h = sample(x[idx], y[idx])
    base = least_squares(
        lambda k: k[0] * (base_h - 32768) + k[1] - z[idx],
        [0.0078, 1],
        bounds=([0.002, -20], [0.03, 20]),
        loss="soft_l1",
        f_scale=2,
    )
    k0 = [*base.x, 0, 0, 1]
    opt = least_squares(
        lambda k: predict(x[idx], y[idx], k) - z[idx],
        k0,
        bounds=([0.002, -20, -64, -64, 0.98], [0.03, 20, 64, 64, 1.02]),
        loss="soft_l1",
        f_scale=2,
        x_scale=[0.008, 2, 20, 20, 0.01],
        max_nfev=100,
    )
    k = opt.x
    pred = predict(x, y, k)
    residual = z - pred
    test = p[:, 0] >= 32
    # Gross orientation falsification, not independent landmark accuracy.
    orient = {}
    for name, xx, yy in [
        ("xy", x, y),
        ("flip_x", EXTENT - x, y),
        ("flip_y", x, EXTENT - y),
        ("flip_xy", EXTENT - x, EXTENT - y),
        ("swap", y, x),
        ("swap_flip_x", EXTENT - y, x),
        ("swap_flip_y", y, EXTENT - x),
        ("swap_flip_xy", EXTENT - y, EXTENT - x),
    ]:
        hh = sample(xx[idx], yy[idx])
        kk = least_squares(
            lambda q: q[0] * (hh - 32768) + q[1] - z[idx],
            base.x,
            bounds=([0.00001, -300], [0.04, 300]),
            loss="soft_l1",
            f_scale=2,
        )
        heldh = sample(xx[test], yy[test])
        orient[name] = stats(z[test] - (kk.x[0] * (heldh - 32768) + kk.x[1]))
    # Silhouette agreement of two weak proxies; no survey/coastline labels.
    hh = np.asarray(
        Image.open(ASSETS / "height_2019.png").resize((1024, 1024), Image.Resampling.NEAREST)
    )
    land_h = hh >= 32768
    land_rgb = ~water
    coast_iou = float(np.sum(land_h & land_rgb) / np.sum(land_h | land_rgb))
    orientation_iou = {
        name: float(np.sum(arr & land_rgb) / np.sum(arr | land_rgb))
        for name, arr in [
            ("xy", land_h),
            ("flip_x", land_h[:, ::-1]),
            ("flip_y", land_h[::-1]),
            ("swap", land_h.T),
        ]
    }
    # Observed graph, built only from the 32 fit matches; no synthetic straight-line routes.
    cells = np.floor(p[:, 2:4] / CELL).astype(int)
    cm = defaultdict(set)
    cp = defaultdict(list)
    em = defaultdict(list)
    ematches = defaultdict(set)
    for i, c in enumerate(cells):
        if p[i, 0] < 32:
            cm[tuple(c)].add(int(p[i, 0]))
            cp[tuple(c)].append(i)
    for row in pairs:
        i, j = int(row[0]), int(row[1])
        a, b = tuple(cells[i]), tuple(cells[j])
        if p[i, 0] < 32 and a != b:
            em[a, b].append(row[2:])
            ematches[a, b].add(int(p[i, 0]))
    order = sorted(cm)
    ids = {c: i for i, c in enumerate(order)}
    nodes = []
    for c in order:
        lo = np.array(c) * CELL
        center = (lo + np.minimum(lo + CELL, EXTENT)) / 2
        ii = np.asarray(cp[c])
        raw = sample(*center, k=k).item()
        nodes.append(
            dict(
                node_id=ids[c],
                cell_x=int(c[0]),
                cell_y=int(c[1]),
                x_m=center[0],
                y_m=center[1],
                height_raw=raw,
                player_z_proxy_m=k[0] * (raw - 32768) + k[1],
                match_support=len(cm[c]),
                point_count=len(ii),
                median_abs_player_z_residual_m=float(np.median(np.abs(residual[ii]))),
                known_change_buffer=bool(risk_at(*center, regions)),
                water_color_proxy=bool(
                    water[
                        min(int(center[1] / EXTENT * 1024), 1023),
                        min(int(center[0] / EXTENT * 1024), 1023),
                    ]
                ),
                exact_patch_match=False,
                traversability_verified=False,
            )
        )
    edges = []
    for (a, b), v in sorted(em.items()):
        v = np.asarray(v)
        n1, n2 = nodes[ids[a]], nodes[ids[b]]
        edges.append(
            dict(
                source=ids[a],
                target=ids[b],
                count=len(v),
                match_support=len(ematches[a, b]),
                median_dt_s=float(np.median(v[:, 0])),
                median_displacement_m=float(np.median(v[:, 1])),
                known_change_buffer=n1["known_change_buffer"] or n2["known_change_buffer"],
            )
        )
    held = []
    for row in pairs:
        i, j = int(row[0]), int(row[1])
        a, b = tuple(cells[i]), tuple(cells[j])
        if p[i, 0] >= 32 and a != b:
            held.append((a, b))
    metrics = dict(
        n_nodes=len(nodes),
        n_directed_nonself_edges=len(edges),
        edges_supported_by_at_least_3_matches=sum(e["match_support"] >= 3 for e in edges),
        heldout_nonself_pairs=len(held),
        heldout_both_cells_seen=float(np.mean([a in ids and b in ids for a, b in held])),
        heldout_directed_edge_seen=float(np.mean([(a, b) in em for a, b in held])),
        known_change_buffer_nodes=sum(n["known_change_buffer"] for n in nodes),
    )
    for name, rows in [("nodes", nodes), ("edges", edges)]:
        with (OUT / (name + ".csv")).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "graph.json").write_text(json.dumps(dict(cell_size_m=CELL, nodes=nodes, edges=edges)))
    # Separate terrain-only adjacency candidate. Height threshold is only a land
    # proxy; a heightfield lacks bridge decks, interiors and collision geometry.
    axis = np.arange(CELL / 2, EXTENT, CELL)
    gx, gy = np.meshgrid(axis, axis)
    gh = sample(gx.ravel(), gy.ravel(), k).reshape(gx.shape)
    land = gh >= 32768
    loc = np.argwhere(land)
    lookup = {tuple(c): i for i, c in enumerate(loc)}
    tn = []
    te = []
    for cy, cx in loc:
        nid = lookup[cy, cx]
        xx = float(gx[cy, cx])
        yy = float(gy[cy, cx])
        raw = float(gh[cy, cx])
        tn.append(
            dict(
                node_id=nid,
                x_m=xx,
                y_m=yy,
                height_raw=raw,
                height_scaled_proxy_m=float(k[0] * (raw - 32768)),
                known_change_buffer=bool(risk_at(xx, yy, regions)),
                exact_patch_match=False,
                traversability_verified=False,
            )
        )
        for dy, dx in [(0, 1), (1, 0)]:
            c2 = (cy + dy, cx + dx)
            if c2 in lookup:
                te.append(
                    dict(
                        source=nid,
                        target=lookup[c2],
                        horizontal_m=CELL,
                        rise_proxy_m=float(k[0] * (gh[c2] - raw)),
                        grade_proxy=float(abs(k[0] * (gh[c2] - raw)) / CELL),
                    )
                )
    for name, rows in [("terrain_nodes", tn), ("terrain_edges", te)]:
        with (OUT / (name + ".csv")).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "terrain_graph.json").write_text(json.dumps(dict(cell_size_m=CELL, nodes=tn, edges=te)))
    terrain_metrics = dict(
        nodes=len(tn),
        undirected_4neighbor_edges=len(te),
        median_grade_proxy=float(np.median([e["grade_proxy"] for e in te])),
        definition="height>=32768 land proxy; no slope rejection; not navmesh",
    )
    np.savez_compressed(
        OUT / "alignment_arrays.npz",
        predicted_player_z=pred,
        residual=residual,
        risk=risk,
        water=on_water,
        calibration_indices=idx,
        spatial_block=block,
    )
    result = dict(
        status="DIAGNOSTIC_ONLY_VERSION_MISMATCH",
        world_extent_m=EXTENT,
        calibration_point_count=len(idx),
        calibration_excludes_heldout_matches_and_spatial_blocks=True,
        vertical_target="player altitude proxy, not ground elevation",
        nominal_world_to_official_pixel="u=x_cm*8192/816000; v=y_cm*8192/816000; y increases down",
        height_sampling="u=(x_m*s+dx_m)*8192/8160; v=(y_m*s+dy_m)*8192/8160",
        height_to_player_z="a*(height_raw-32768)+b",
        fit_parameters=dict(
            zip(["a_m_per_raw", "b_player_z_offset_m", "dx_m", "dy_m", "scale"], map(float, k))
        ),
        optimizer_success=bool(opt.success),
        baseline_parameters=k0,
        baseline_heldout=stats(z[test] - predict(x[test], y[test], k0)),
        fitted_heldout_all=stats(residual[test]),
        fitted_heldout_outside_known_buffers=stats(residual[test & ~risk & ~on_water]),
        fitted_heldout_inside_known_buffers=stats(residual[test & risk]),
        fitted_heldout_unseen_spatial_blocks=stats(
            residual[test & (block == 0) & ~risk & ~on_water]
        ),
        orientation_heldout=orient,
        silhouette_proxy_iou=coast_iou,
        silhouette_orientation_iou=orientation_iou,
        graph=metrics,
        terrain_candidate_graph=terrain_metrics,
        known_change_buffers=regions,
        limitations=[
            "No exact game build in DB",
            "Known terrain changes between 2019 heightmap and 2026 telemetry",
            "Official map 2024 commit is not a 40.2 map",
            "Player z is not ground truth",
            "Change buffers are approximate, non-exhaustive; outside does not mean unchanged",
            "Observed endpoint transitions are not collision-free straight paths",
        ],
    )
    (OUT / "alignment_metrics.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                q: result[q]
                for q in [
                    "fit_parameters",
                    "fitted_heldout_all",
                    "fitted_heldout_unseen_spatial_blocks",
                    "silhouette_proxy_iou",
                    "graph",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
