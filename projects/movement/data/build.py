"""Materialize match banks, train-only vocab/spatial prior, and paired sample index."""

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
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.bank import read_raw, build_bank, FEATURES, MAX_ITEMS

OUT = ROOT / "data/processed/heterograph_diffusion_v1"
PRIOR = ROOT / "reports/erangel_100m_40_2_2026-10-15"


def one(row, vocab):
    raw = read_raw(OUT / "raw" / f"{row['match_id']}.json.gz")
    b = build_bank(raw, vocab)
    np.savez_compressed(OUT / "banks" / f"{row['match_id']}.npz", **b)
    ix = []
    stats = Counter()
    T, P = b["valid"].shape
    for k in range(3, T, 3):
        for p in np.flatnonzero(b["valid"][k]):
            members = np.flatnonzero(b["team"] == b["team"][p])
            if len(members) > 4:
                stats["anchors_excluded_roster_gt4"] += 1
                continue
            current = members[b["valid"][k, members]]
            future = [j for j in range(k + 1, min(k + 7, T))]
            has = bool(future and b["target_valid"][future, p].any())
            complete = bool(k + 6 < T and b["target_valid"][k + 1 : k + 7, current].all())
            ix.append(
                [
                    int(row["match_index"]),
                    k,
                    int(p),
                    int(b["kind"][p]),
                    int(has),
                    int(complete),
                    len(current),
                ]
            )
            stats["anchors"] += 1
            stats["ego_has_future"] += has
            stats["all_targets_complete_60s"] += complete
            stats["human_anchors"] += b["kind"][p] == 1
            stats["bot_anchors"] += b["kind"][p] == 2
            stats["current_vehicle_anchors"] += b["x"][k, p, 14] > 0
    stats.update(
        {
            "frames": T,
            "actors": P,
            "equipment_tokens_truncated": int(b["quality"][0]),
            "ambiguous_actor_frames": int(b["quality"][1]),
            "positive_post_first_death_rows_excluded": int(b["quality"][2]),
        }
    )
    return row["match_id"], row["split"], ix, {k: int(v) for k, v in stats.items()}


def static(rows):
    total = defaultdict(lambda: [0, 0])
    own = OUT / "spatial_match"
    own.mkdir(exist_ok=True)
    for r in rows:
        if r["split"] != "train":
            continue
        d = np.load(PRIOR / "match_cache" / f"{int(r['match_index']):04d}.npz")
        acc = Counter()
        for a, b, kind, motion, count, dt, dist in d["edges"]:
            acc[int(a), int(b), int(motion)] += int(count)
        local = np.array(
            [[*key, value] for key, value in sorted(acc.items())], dtype=np.int64
        ).reshape(-1, 4)
        np.savez_compressed(own / f"{r['match_id']}.npz", edges=local)
        for key, value in acc.items():
            total[key][0] += value
            total[key][1] += 1
    edges = np.array([[*key, *val] for key, val in sorted(total.items())], dtype=np.int64)
    nr = list(csv.DictReader((PRIOR / "nodes.csv").open()))
    terrain = np.array(
        [
            [
                float(r[k])
                for k in ("x_m", "y_m", "historical_height_proxy_m", "historical_land_fraction")
            ]
            + [float(r["known_change_buffer"] == "True")]
            for r in nr
        ],
        np.float32,
    )
    np.savez_compressed(OUT / "spatial_prior.npz", edges=edges, terrain=terrain)
    return {
        "train_matches": sum(r["split"] == "train" for r in rows),
        "directed_typed_edges": len(edges),
        "training_policy": "subtract own match counts and match support at load time; validation uses training prior only",
        "inferred_policy": "historical-land four-neighbor candidate iff no remaining observed nonvehicle edge; never certified traversable",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = list(csv.DictReader((OUT / "matches.csv").open()))
    assert len(rows) == 939
    vocab = {"items": {}, "vehicles": {}}
    isets = set()
    vsets = set()
    for r in rows:
        if r["split"] != "train":
            continue
        raw = read_raw(OUT / "raw" / f"{r['match_id']}.json.gz")
        isets.update(x["item"] for x in raw["tables"]["item_equip"] if x["item"])
        vsets.update(x["vehicle_class"] for x in raw["tables"]["positions"] if x["vehicle_class"])
    vocab["items"] = {s: i + 2 for i, s in enumerate(sorted(isets))}
    vocab["vehicles"] = {s: i + 2 for i, s in enumerate(sorted(vsets))}
    vocab["reserved"] = {
        "0": "padding/no token; not proof of unarmed",
        "1": "unseen training vocabulary",
    }
    (OUT / "vocabulary.json").write_text(json.dumps(vocab, indent=2))
    prior = static(rows)
    allix = {"train": [], "validation": []}
    report = {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = [pool.submit(one, r, vocab) for r in rows]
        for i, f in enumerate(as_completed(fs), 1):
            mid, split, ix, stats = f.result()
            allix[split].extend(ix)
            report[mid] = {"split": split, **stats}
            if i % 50 == 0:
                print(
                    json.dumps(
                        {"built": i, "total": len(rows), "seconds": round(time.time() - start)}
                    ),
                    flush=True,
                )
    for split, ix in allix.items():
        np.save(OUT / f"{split}_index.npy", np.array(sorted(ix), np.int32))
    contract = {
        "version": 1,
        "cohort": "939 Steam official TPP squad Erangel matches, patch 40.2 inferred from time window",
        "split_matches": dict(Counter(r["split"] for r in rows)),
        "history_offsets_s": [-30, -20, -10, 0],
        "future_offsets_s": [10, 20, 30, 40, 50, 60],
        "anchor_stride_s": 30,
        "input_position_age_max_s": 15,
        "target_nearest_tolerance_s": 5,
        "target_time_error": "signed sample source time minus desired future grid time is retained",
        "current_target_members": "ego first; currently valid landed non-first-dead roster members, padded to four; fixed at anchor",
        "future_team_target": "equal-weight center and pairwise distances of fixed current target members; valid only when all required future positions valid",
        "future_mask": "supervised loss/evaluation and past-only completion of missing training labels; never a condition feature or decoder attention mask",
        "views": ["observer", "server"],
        "observer": "own private evidence + own-team HUD proxies; no exact enemy positions, unshared teammate loadout, outgoing damage, alive-team count",
        "server": "all participants and teams recorded as of each historical frame; private gear, vehicles, damage and recent attacker coordinates; no future events or final rank; not full engine state",
        "zone": "same as-of raw safe_zone/poison_zone in both views; UI announcement timing unverified",
        "feature_names": FEATURES,
        "equipment_max_tokens": MAX_ITEMS,
        "position_scale_m": 1000,
        "spatial_prior": prior,
        "index_columns": [
            "original_match_index",
            "anchor_grid_index",
            "ego_actor_index",
            "analysis_actor_kind",
            "has_any_ego_future_label",
            "complete_joint_60s",
            "current_target_count",
        ],
        "actor_kind": "1 human prefix, 2 bot prefix; analysis metadata only, no model feature",
        "initial_death_policy": "censor after first recorded death including later recalls; explicitly counted",
        "raw_queries": "source_queries.json",
        "source_split_sha256": hashlib.sha256((PRIOR / "match_split.csv").read_bytes()).hexdigest(),
        "bank_code_sha256": hashlib.sha256(
            (ROOT / "src/pubg_lab/movement/data/bank.py").read_bytes()
        ).hexdigest(),
    }
    (OUT / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    (OUT / "per_match_quality.json").write_text(json.dumps(report, indent=2))
    summary = {
        "matches": len(rows),
        "splits": {
            s: {
                "samples": len(ix),
                "matches": sum(r["split"] == s for r in rows),
                "counts": dict(
                    sum(
                        (
                            Counter({k: v for k, v in stats.items() if k != "split"})
                            for stats in report.values()
                            if stats["split"] == s
                        ),
                        Counter(),
                    )
                ),
            }
            for s, ix in allix.items()
        },
        "vocabulary_sizes": {k: len(vocab[k]) + 2 for k in ("items", "vehicles")},
        "spatial_prior": prior,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
