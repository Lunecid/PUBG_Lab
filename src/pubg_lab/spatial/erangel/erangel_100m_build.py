"""Build 100m graph from TRAIN matches only; then evaluate held-out matches.

Confidence is descriptive support, not calibrated traversability probability.
Old raster and current observations remain separate features.
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from pubg_lab.spatial.erangel.erangel_spatial_build import risk_regions, risk_at
from pubg_lab.spatial.erangel.erangel_100m_extract import OUT, ROOT, EXTENT, CELL, WIDTH

N = WIDTH * WIDTH


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prior_grid():
    h = np.asarray(Image.open(ROOT / "data/spatial/erangel/source/height_2019.png")).astype(
        np.float32
    )
    rgb = np.asarray(
        Image.open(ROOT / "data/spatial/erangel/source/official_notext.png").resize((1024, 1024))
    ).astype(float)
    ids = np.arange(N)
    gx = ids % WIDTH
    gy = ids // WIDTH
    x = (gx * CELL + np.minimum((gx + 1) * CELL, EXTENT)) / 2
    y = (gy * CELL + np.minimum((gy + 1) * CELL, EXTENT)) / 2
    raw = map_coordinates(h, [y * 8192 / EXTENT, x * 8192 / EXTENT], order=1, mode="nearest")
    # 16 interior samples per cell retain coast ambiguity instead of a center-only mask.
    land = np.zeros(N)
    water = np.zeros(N)
    for a in [0.125, 0.375, 0.625, 0.875]:
        for b in [0.125, 0.375, 0.625, 0.875]:
            xx = gx * CELL + np.minimum(CELL, EXTENT - gx * CELL) * a
            yy = gy * CELL + np.minimum(CELL, EXTENT - gy * CELL) * b
            hh = map_coordinates(
                h, [yy * 8192 / EXTENT, xx * 8192 / EXTENT], order=1, mode="nearest"
            )
            rr = rgb[
                np.minimum((yy * 1024 / EXTENT).astype(int), 1023),
                np.minimum((xx * 1024 / EXTENT).astype(int), 1023),
            ]
            land += (hh >= 32768) / 16
            water += ((rr[:, 2] > rr[:, 0] * 1.25) & (rr[:, 1] > rr[:, 0] * 1.3)) / 16
    regions = risk_regions()
    return x, y, raw, land, water, risk_at(x, y, regions), regions


def aggregate_training(rows):
    points = np.zeros((N, 3, 2), dtype=np.int64)
    support = np.zeros(N, dtype=int)
    kind_support = np.zeros((N, 3), dtype=int)
    type_support = {k: np.zeros(N, dtype=int) for k in ["official", "competitive", "airoyale"]}
    med = defaultdict(list)
    edges = {0: defaultdict(lambda: np.zeros(10)), 1: defaultdict(lambda: np.zeros(10))}
    used = []
    for r in rows:
        if r["split"] != "train":
            continue
        used.append(r["match_id"])
        d = np.load(OUT / "match_cache" / f"{int(r['match_index']):04d}.npz")
        nn = d["nodes"]
        ee = d["edges"]
        local = defaultdict(list)
        for n in nn:
            cell, kind, motion, count = map(int, n[:4])
            points[cell, kind, motion] += count
            if motion == 0:
                local[cell].append(n)
        for cell, group in local.items():
            support[cell] += 1
            type_support[r["match_type"]][cell] += 1
            kinds = {int(n[1]) for n in group}
            for kind in kinds:
                kind_support[cell, kind] += 1
            # One vote per match-cell. Each vote summarizes the available player-kind groups.
            g = np.asarray(group)
            med[cell].append(
                [np.median(g[:, 4]), np.median(g[:, 7]), np.max(g[:, 6]) - np.min(g[:, 5])]
            )
        for motion in [0, 1]:
            local_e = defaultdict(list)
            for e in ee:
                if int(e[3]) == motion:
                    local_e[int(e[0]), int(e[1])].append(e)
            for pair, group in local_e.items():
                arr = edges[motion][pair]
                g = np.asarray(group)
                arr[0] += g[:, 4].sum()
                arr[1] += 1
                arr[2] += g[:, 5].sum()
                arr[3] += g[:, 6].sum()
                for kind in [0, 1, 2]:
                    arr[4 + kind] += g[g[:, 2] == kind, 4].sum()
                arr[7 + ["official", "competitive", "airoyale"].index(r["match_type"])] += 1
    return points, support, kind_support, type_support, med, edges, used


def main():
    extraction = json.loads((OUT / "extraction.json").read_text())
    assert extraction["status"] == "complete"
    rows = list(csv.DictReader((OUT / "match_split.csv").open()))
    train = {r["match_id"] for r in rows if r["split"] == "train"}
    val = {r["match_id"] for r in rows if r["split"] == "validation"}
    assert not train & val and len(train | val) == len(rows)
    x, y, raw, land, water, risk, regions = prior_grid()
    points, support, ks, ts, med, edges, used = aggregate_training(rows)
    z = np.full(N, np.nan)
    delta = z.copy()
    spread = z.copy()
    for cell, v in med.items():
        values = np.asarray(v)
        z[cell], delta[cell], spread[cell] = np.median(values, axis=0)
    calibration = (
        (support >= 10)
        & (land >= 0.75)
        & (water < 0.25)
        & ~risk
        & (spread <= 3)
        & np.isfinite(delta)
    )
    assert calibration.sum() > 0
    offset = float(np.median(delta[calibration]))
    res = delta - offset
    nodes = []
    for i in range(N):
        flags = []
        if support[i] == 0:
            flags.append("unobserved")
        elif support[i] < 10:
            flags.append("low_match_support")
        if np.isfinite(res[i]) and abs(res[i]) > 5:
            flags.append("height_disagreement")
        if np.isfinite(spread[i]) and spread[i] > 6:
            flags.append("vertical_variability")
        if risk[i]:
            flags.append("known_change_buffer")
        if (land[i] >= 0.75 and water[i] >= 0.75) or (land[i] <= 0.25 and water[i] <= 0.25):
            flags.append("land_water_disagreement")
        if 0.25 < land[i] < 0.75:
            flags.append("coastal_mixed_cell")
        if support[i] > 0 and land[i] < 0.25:
            flags.append("observed_over_water_prior")
        category = (
            "unobserved"
            if support[i] == 0
            else "low_support"
            if support[i] < 10
            else "flagged"
            if flags
            else "repeated_observation"
        )
        nodes.append(
            dict(
                node_id=i,
                cell_x=i % WIDTH,
                cell_y=i // WIDTH,
                x_m=float(x[i]),
                y_m=float(y[i]),
                width_m=float(min(CELL, EXTENT - (i % WIDTH) * CELL)),
                height_m=float(min(CELL, EXTENT - (i // WIDTH) * CELL)),
                historical_height_proxy_m=float((raw[i] - 32768) / 128),
                historical_land_fraction=float(land[i]),
                rgb_water_fraction=float(water[i]),
                observed_player_z_summary_m=float(z[i]) if np.isfinite(z[i]) else None,
                height_residual_summary_m=float(res[i]) if np.isfinite(res[i]) else None,
                vertical_spread_summary_m=float(spread[i]) if np.isfinite(spread[i]) else None,
                match_support=int(support[i]),
                nonvehicle_points=int(points[i, :, 0].sum()),
                vehicle_points=int(points[i, :, 1].sum()),
                human_points=int(points[i, 1, 0]),
                bot_points=int(points[i, 2, 0]),
                unknown_kind_points=int(points[i, 0, 0]),
                human_match_support=int(ks[i, 1]),
                bot_match_support=int(ks[i, 2]),
                official_match_support=int(ts["official"][i]),
                competitive_match_support=int(ts["competitive"][i]),
                airoyale_match_support=int(ts["airoyale"][i]),
                category=category,
                uncertainty_reasons="|".join(flags),
                known_change_buffer=bool(risk[i]),
                exact_patch_match=False,
                traversability_verified=False,
            )
        )

    def observed_rows(motion):
        result = []
        for (a, b), v in sorted(edges[motion].items()):
            result.append(
                dict(
                    source=a,
                    target=b,
                    edge_type="observed",
                    motion="vehicle" if motion else "nonvehicle",
                    transition_count=int(v[0]),
                    match_support=int(v[1]),
                    mean_dt_s=float(v[2] / v[0]),
                    mean_displacement_m=float(v[3] / v[0]),
                    unknown_count=int(v[4]),
                    human_count=int(v[5]),
                    bot_count=int(v[6]),
                    official_match_support=int(v[7]),
                    competitive_match_support=int(v[8]),
                    airoyale_match_support=int(v[9]),
                    uncertain=bool(
                        v[1] < 3
                        or nodes[a]["category"] != "repeated_observation"
                        or nodes[b]["category"] != "repeated_observation"
                    ),
                    straight_segment_verified=False,
                )
            )
        return result

    observed = observed_rows(0)
    vehicle = observed_rows(1)
    inferred = []
    for i in range(N):
        if land[i] < 0.75:
            continue
        for dx, dy in [(1, 0), (0, 1)]:
            xx = i % WIDTH + dx
            yy = i // WIDTH + dy
            if xx >= WIDTH or yy >= WIDTH:
                continue
            j = yy * WIDTH + xx
            if land[j] < 0.75:
                continue
            # Historical 4-neighbor adjacency is a candidate, not certified walkability.
            length = float(np.hypot(x[j] - x[i], y[j] - y[i]))
            grade = abs(float(raw[j] - raw[i]) / 128) / length
            for a, b in [(i, j), (j, i)]:
                if (a, b) not in edges[0]:
                    inferred.append(
                        dict(
                            source=a,
                            target=b,
                            edge_type="inferred",
                            motion="nonvehicle_candidate",
                            historical_grade_proxy=grade,
                            uncertain=True,
                            reason="historical_land_4neighbor; no observed directed transition",
                            traversability_verified=False,
                        )
                    )
    for name, data in [
        ("nodes", nodes),
        ("observed_edges", observed),
        ("inferred_edges", inferred),
        ("vehicle_observed_edges", vehicle),
    ]:
        write_csv(OUT / (name + ".csv"), data)
    graph = dict(
        extent_m=EXTENT,
        cell_m=CELL,
        grid_width=WIDTH,
        nodes=nodes,
        observed_edges=observed,
        inferred_edges=inferred,
        vehicle_observed_edges=vehicle,
    )
    (OUT / "graph.json").write_text(json.dumps(graph, separators=(",", ":")))
    # Freeze graph before reading any validation aggregates.
    graph_hash = hashlib.sha256((OUT / "graph.json").read_bytes()).hexdigest()
    inferred_keys = {(e["source"], e["target"]) for e in inferred}
    validation = []
    for r in rows:
        if r["split"] != "validation":
            continue
        d = np.load(OUT / "match_cache" / f"{int(r['match_index']):04d}.npz")
        nn = d["nodes"]
        ee = d["edges"]
        counter = Counter()
        for n in nn:
            if n[2] != 0:
                continue
            cell, kind, count = int(n[0]), int(n[1]), int(n[3])
            counter["positions"] += count
            counter["positions_in_observed_cells"] += count * int(support[cell] > 0)
            counter["positions_in_repeated_cells"] += count * int(support[cell] >= 10)
        for e in ee:
            if e[3] != 0:
                continue
            a, b, kind, count = int(e[0]), int(e[1]), int(e[2]), int(e[4])
            counter["transitions"] += count
            label = (
                "observed"
                if (a, b) in edges[0]
                else "inferred"
                if (a, b) in inferred_keys
                else "uncovered"
            )
            counter["transitions_" + label] += count
            counter[f"kind_{kind}_transitions"] += count
            counter[f"kind_{kind}_transitions_{label}"] += count
            counter["transitions_both_cells_seen"] += count * int(support[a] > 0 and support[b] > 0)
        validation.append(
            dict(match_id=r["match_id"], match_type=r["match_type"], mode=r["mode"], **counter)
        )
    total = sum(
        (Counter({k: v for k, v in r.items() if isinstance(v, int)}) for r in validation), Counter()
    )
    macro = [
        r.get("transitions_observed", 0) / r["transitions"]
        for r in validation
        if r.get("transitions", 0) > 0
    ]
    by_type = {}
    for typ in ["official", "competitive", "airoyale"]:
        sub = sum(
            (
                Counter({k: v for k, v in r.items() if isinstance(v, int)})
                for r in validation
                if r["match_type"] == typ
            ),
            Counter(),
        )
        by_type[typ] = dict(sub)
    prior_land = land >= 0.75
    summary = dict(
        status="complete",
        patch="40.2 inferred, not exact build",
        n_matches=len(rows),
        n_train=len(train),
        n_validation=len(val),
        population_by_type=dict(Counter(r["match_type"] for r in rows)),
        grid_nodes=N,
        historical_land_candidate_nodes=int(prior_land.sum()),
        observed_nonvehicle_nodes=int((support > 0).sum()),
        observed_historical_land_candidate_nodes=int(((support > 0) & prior_land).sum()),
        land_candidate_observation_coverage=float(
            ((support > 0) & prior_land).sum() / prior_land.sum()
        ),
        repeated_land_candidate_coverage=float(
            ((support >= 10) & prior_land).sum() / prior_land.sum()
        ),
        node_categories=dict(Counter(n["category"] for n in nodes)),
        uncertainty_reason_counts=dict(
            Counter(f for n in nodes for f in n["uncertainty_reasons"].split("|") if f)
        ),
        observed_directed_edges=len(observed),
        inferred_directed_edges=len(inferred),
        vehicle_directed_edges=len(vehicle),
        observed_edges_at_least_3_matches=sum(e["match_support"] >= 3 for e in observed),
        height_offset_m=offset,
        height_offset_calibration_cells=int(calibration.sum()),
        validation_counts=dict(total),
        validation_observed_edge_micro=float(total["transitions_observed"] / total["transitions"]),
        validation_observed_edge_macro=float(np.mean(macro)),
        validation_by_match_type=by_type,
        extraction_totals=extraction["totals"],
        graph_sha256=graph_hash,
        training_matches_sha256=hashlib.sha256("\n".join(sorted(used)).encode()).hexdigest(),
        known_change_buffers=regions,
        interpretation="Coverage and support, not route prediction or collision accuracy. Full grid includes unknown water cells. All node confidence flags are descriptive heuristics.",
    )
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "validation_matches.json").write_text(json.dumps(validation, indent=2))
    fields = sorted(set().union(*(r.keys() for r in validation)))
    with (OUT / "validation_matches.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(validation)
    assert graph_hash == hashlib.sha256((OUT / "graph.json").read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "n_matches",
                    "n_train",
                    "n_validation",
                    "historical_land_candidate_nodes",
                    "observed_nonvehicle_nodes",
                    "land_candidate_observation_coverage",
                    "node_categories",
                    "observed_directed_edges",
                    "inferred_directed_edges",
                    "validation_observed_edge_micro",
                    "validation_observed_edge_macro",
                    "height_offset_m",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
