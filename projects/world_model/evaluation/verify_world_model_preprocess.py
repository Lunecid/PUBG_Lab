"""Offline audit of the completed preprocessing; never opens protected matches."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.world_model.data.preprocess import build_bank, view_at, targets_at
from pubg_lab.world_model.data.dataset import WorldDataset


def verify(root, output):
    rows = list(csv.DictReader((root / "matches.csv").open()))
    allowed = {r["match_id"]: r for r in rows if r["eligible"] == "1"}
    denied = {r["match_id"] for r in rows if r["role"] not in ("train", "validation")}
    actual = {p.stem for p in (root / "banks").glob("*.npz")}
    assert not actual & denied
    assert actual == set(allowed)
    assert {r["patch"] for r in allowed.values() if r["role"] == "train"} == {"40.2"}
    assert {r["patch"] for r in allowed.values() if r["role"] == "validation"} == {"41.1"}
    coverage = defaultdict(Counter)
    post_end = Counter()
    for mid, r in allowed.items():
        with np.load(root / "banks" / f"{mid}.npz", allow_pickle=False) as bank:
            assert np.isfinite(bank["x"]).all()
            if r["end_time"] and r["start_time"]:
                end_s = (
                    datetime.fromisoformat(r["end_time"]) - datetime.fromisoformat(r["start_time"])
                ).total_seconds()
                assert not len(bank["times"]) or float(bank["times"][-1]) <= end_s
                assert (bank["target_source_time"][bank["target_valid"]] <= end_s).all()
            anchors = bank["anchors"]
            role = r["role"]
            coverage[role]["matches"] += 1
            coverage[role]["anchors"] += len(anchors)
            coverage[role]["matches_without_anchors"] += int(not len(anchors))
            if len(anchors):
                k, p = anchors.T
                q = k[:, None] + np.arange(1, 7)
                in_bounds = q < len(bank["times"])
                qc = np.minimum(q, len(bank["times"]) - 1)
                good = (
                    in_bounds
                    & bank["target_valid"][qc, p[:, None]]
                    & (bank["target_episode"][qc, p[:, None]] == bank["episode"][k, p][:, None])
                    & (bank["target_segment"][qc, p[:, None]] == bank["segment"][k, p][:, None])
                )
                coverage[role]["ego_any_future"] += int(good.any(1).sum())
                coverage[role]["ego_complete_60s"] += int(good.all(1).sum())
                for kind, name in [(1, "human"), (2, "bot"), (0, "other_roster")]:
                    coverage[role][name + "_anchors"] += int(
                        (bank["kind_eval_only"][p] == kind).sum()
                    )
        meta = json.loads((root / "meta" / f"{mid}.json").read_text())
        post_end.update(meta["post_end_rows_excluded"])
    # A deterministic sample includes every observed map/perspective/size/role group.
    groups = {}
    for mid, r in sorted(allowed.items()):
        groups.setdefault((r["map"], r["perspective"], r["nominal_team_size"], r["role"]), mid)
    checks = []
    deterministic = []
    for mid in groups.values():
        with gzip.open(root / "raw" / f"{mid}.json.gz", "rt") as f:
            raw = json.load(f)
        with np.load(root / "banks" / f"{mid}.npz", allow_pickle=False) as f:
            bank = dict(f)
        if len(deterministic) < 3:
            replay, _ = build_bank(raw)
            payload = io.BytesIO()
            np.savez_compressed(payload, **replay)
            rebuilt_sha = hashlib.sha256(payload.getvalue()).hexdigest()
            expected = json.loads((root / "meta" / f"{mid}.json").read_text())["bank_sha256"]
            assert rebuilt_sha == expected
            deterministic.append({"match_id": mid, "bank_sha256": expected, "byte_identical": True})
        if len(bank["times"]) < 13:
            checks.append({"match_id": mid, "short_match": True})
            continue
        k = min(60, len(bank["times"]) - 1)
        cut = float(bank["times"][k])
        prefix = {
            **raw,
            "tables": {key: [r for r in rs if r["t"] <= cut] for key, rs in raw["tables"].items()},
        }
        rebuilt, _ = build_bank(prefix)
        for key in [
            "x",
            "mask",
            "valid",
            "source_time",
            "known",
            "episode",
            "segment",
            "mobility",
            "game",
            "game_mask",
            "game_time",
        ]:
            np.testing.assert_array_equal(
                bank[key][: len(rebuilt["times"])], rebuilt[key], err_msg=mid + ":" + key
            )
        # Stored confidence has an initial-life placeholder for actors with later
        # positions. The loader exposes it only under the as-of known mask.
        n = len(rebuilt["times"])
        np.testing.assert_array_equal(
            np.where(bank["known"][:n], bank["confidence"][:n], 0),
            np.where(rebuilt["known"], rebuilt["confidence"], 0),
            err_msg=mid + ":masked_confidence",
        )
        prefix_anchors = rebuilt["anchors"]
        if len(prefix_anchors):
            pk, pp = map(int, prefix_anchors[-1])
            for visibility in ("observer", "server"):
                full_view = view_at(bank, pk, pp, visibility)
                prefix_view = view_at(rebuilt, pk, pp, visibility)
                for key in full_view:
                    np.testing.assert_array_equal(
                        full_view[key], prefix_view[key], err_msg=mid + ":" + visibility + ":" + key
                    )
        anchors = bank["anchors"]
        if not len(anchors):
            checks.append({"match_id": mid, "prefix_equal": True, "no_anchors": True})
            continue
        k, p = map(int, anchors[len(anchors) // 2])
        original = view_at(bank, k, p, "observer")
        mutated = {key: value.copy() for key, value in bank.items()}
        own = bank["team"] == bank["team"][p]
        mutated["x"][:, ~own] = 77777
        for other in np.flatnonzero(own):
            if other != p:
                mutated["x"][:, other, 8:] = 88888
        mutated["target"][:] = 99999
        mutated["future_damage"][:] = 99999
        modified = view_at(mutated, k, p, "observer")
        for key in original:
            np.testing.assert_array_equal(original[key], modified[key], err_msg=mid + ":" + key)
        y = targets_at(bank, k, p)
        checks.append(
            {
                "match_id": mid,
                "prefix_equal": True,
                "masked_confidence_equal": True,
                "prefix_views_equal": bool(len(prefix_anchors)),
                "observer_private_invariance": True,
                "future_label_invariance": True,
                "players": len(bank["team"]),
                "observer_shape": list(original["x"].shape),
                "target_shape": list(y["xy"].shape),
            }
        )
    # Domain holdouts must have no hidden pathway through priors or vocab.
    plans = []
    for f in sorted((root / "ablations").glob("*.json")):
        a = json.loads(f.read_text())
        train = set(a["train_match_ids"])
        dev = set(a["validation_match_ids"])
        assert train <= allowed.keys() and dev <= allowed.keys()
        assert not train & dev and not (train | dev) & denied
        assert all(allowed[m]["patch"] == "40.2" for m in train)
        assert all(allowed[m]["patch"] == "41.1" for m in dev)
        assert a["fit_priors_and_vocab_must_use_exact_train_ids"]
        name = a["name"]
        for field in ("map", "perspective", "nominal_team_size"):
            prefix = "holdout_" + field + "_"
            if name.startswith(prefix):
                value = name[len(prefix) :]
                assert all(allowed[m][field] != value for m in train)
                assert all(allowed[m][field] == value for m in dev)
        plans.append({"name": name, "train": len(train), "validation": len(dev), "passed": True})
    loader = WorldDataset(root, role="validation", view="observer")
    runtime = []
    for mid in loader.match_ids:
        b = loader.bank(mid)
        if not len(b["anchors"]):
            continue
        graph, target = loader.sample(mid, 0)
        # The next patch has no observed support in a strictly earlier-patch prior.
        assert not len(graph["observed_connections"])
        runtime.append(
            {
                "match_id": mid,
                "nodes": {
                    "players": len(graph["player_ids"]),
                    "teams": len(graph["team_ids"]),
                    "cells": len(graph["cell_coordinates"]),
                },
                "observed_edges": len(graph["observed_connections"]),
                "inferred_edges": len(graph["inferred_adjacency"]),
            }
        )
        if len(runtime) == 5:
            break
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed",
        "training_executed": False,
        "allowed_matches": len(actual),
        "coverage_by_role": {k: dict(v) for k, v in coverage.items()},
        "post_end_rows_excluded": dict(post_end),
        "full_bank_rebuild_checks": deterministic,
        "denied_detail_files": len(actual & denied),
        "domain_prefix_checks": checks,
        "ablation_checks": plans,
        "runtime_graph_examples": runtime,
        "limitations": [
            "Patch inferred from official schedule, not executable build.",
            "Observer is a telemetry/HUD proxy; visibility and zone announcement latency unverified.",
            "Rank targets are cross-field-consistent candidates, not independently adjudicated outcomes.",
            "Final elimination times remain masked.",
            "No pristine final test; later patch remains reserved.",
            "Ablation manifests validated; no ablation model was trained.",
        ],
    }
    (output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "prefix_groups": len(checks),
                "ablations": len(plans),
                "runtime_examples": len(runtime),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.root, a.output)
