"""Independent structural/accounting checks on the completed run."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
import csv
import hashlib
import json
from collections import Counter

import numpy as np
from projects.spatial.erangel.ingestion.erangel_100m_extract import OUT


def main():
    m = json.loads((OUT / "metrics.json").read_text())
    g = json.loads((OUT / "graph.json").read_text())
    rows = list(csv.DictReader((OUT / "match_split.csv").open()))
    n = g["nodes"]
    checks = []
    train = {r["match_id"] for r in rows if r["split"] == "train"}
    val = {r["match_id"] for r in rows if r["split"] == "validation"}
    assert len(train) == 2141 and len(val) == 534 and not train & val
    checks.append("2675 unique matches; 2141 training and 534 validation; no match overlap")
    counts = Counter()
    observed = Counter()
    support = Counter()
    pos = np.zeros((6724, 2), dtype=np.int64)
    validation_pairs = set()
    for r in rows:
        with np.load(OUT / "match_cache" / f"{int(r['match_index']):04d}.npz") as d:
            assert str(d["match_id"]) == r["match_id"]
            c = json.loads(str(d["counts"]))
            counts.update(c)
            assert c.get("raw_positions", 0) == c.get("retained_positions", 0) + sum(
                v for k, v in c.items() if k.startswith("excluded_")
            )
            nn, ee = d["nodes"], d["edges"]
            assert int(nn[:, 3].sum()) == c.get("retained_positions", 0)
            assert int(ee[:, 4].sum()) == c.get("nonself_pairs", 0)
            if r["split"] == "train":
                for a in nn:
                    pos[int(a[0]), int(a[2])] += int(a[3])
                support.update(set(int(a[0]) for a in nn if a[2] == 0))
                for e in ee:
                    if e[3] == 0:
                        observed[int(e[0]), int(e[1])] += int(e[4])
            else:
                validation_pairs.update((int(e[0]), int(e[1])) for e in ee if e[3] == 0)
    assert dict(counts) == m["extraction_totals"]
    checks.append("Every match: raw = retained + exclusions; node/edge aggregates conserve counts")
    assert np.array_equal(pos[:, 0], [a["nonvehicle_points"] for a in n])
    assert np.array_equal(pos[:, 1], [a["vehicle_points"] for a in n])
    assert all(support[a["node_id"]] == a["match_support"] for a in n)
    assert observed == Counter(
        {(e["source"], e["target"]): e["transition_count"] for e in g["observed_edges"]}
    )
    checks.append(
        "Independent training-only recount equals all node support and observed edge counts"
    )
    infer = {(e["source"], e["target"]) for e in g["inferred_edges"]}
    assert len(infer) == len(g["inferred_edges"]) and not infer & set(observed)
    for a, b in infer:
        assert abs(n[a]["cell_x"] - n[b]["cell_x"]) + abs(n[a]["cell_y"] - n[b]["cell_y"]) == 1
        assert n[a]["historical_land_fraction"] >= 0.75 and n[b]["historical_land_fraction"] >= 0.75
    assert all(a != b for a, b in observed)
    checks.append(
        "Inferred edges are unique directed 4-neighbor historical-land candidates, disjoint from observed edges"
    )
    validation = json.loads((OUT / "validation_matches.json").read_text())
    assert {v["match_id"] for v in validation} == val
    for v in validation:
        assert v.get("transitions", 0) == sum(
            v.get("transitions_" + label, 0) for label in ["observed", "inferred", "uncovered"]
        )
    vc = m["validation_counts"]
    assert vc["transitions"] == sum(v["transitions"] for v in validation)
    assert m["graph_sha256"] == hashlib.sha256((OUT / "graph.json").read_bytes()).hexdigest()
    assert (
        m["training_matches_sha256"]
        == hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest()
    )
    checks.append(
        "Held-out totals partition exactly; graph and training-set hashes match the frozen artifacts"
    )
    result = {
        "status": "passed",
        "checks": checks,
        "training_nonvehicle_positions": int(pos[:, 0].sum()),
        "training_vehicle_positions": int(pos[:, 1].sum()),
        "validation_nonvehicle_positions": vc["positions"],
        "land_candidate_categories": dict(
            Counter(a["category"] for a in n if a["historical_land_fraction"] >= 0.75)
        ),
        "validation_unique_directed_edges": len(validation_pairs),
        "validation_unique_edges_observed": len(validation_pairs & set(observed)),
        "validation_unique_edges_inferred": len(validation_pairs & infer),
        "validation_unique_edges_uncovered": len(validation_pairs - set(observed) - infer),
        "validation_unique_edge_observation_coverage": len(validation_pairs & set(observed))
        / len(validation_pairs),
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
