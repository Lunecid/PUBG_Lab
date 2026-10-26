"""Rebuild fit-only priors and multi-life banks without opening locked-test raw."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import csv
import json
import sys
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.bank import read_raw
from pubg_lab.movement.bank_v3 import build_bank, MODE_NAMES

OLD = ROOT / "data/processed/heterograph_diffusion_v1"
OUT = ROOT / "data/processed/heterograph_diffusion_v3"


def one(row, vocab):
    assert row["split"] != "test"
    mid = row["match_id"]
    raw = read_raw(OLD / "raw" / f"{mid}.json.gz")
    b, stats, positions = build_bank(raw, vocab)
    np.savez_compressed(OUT / "banks" / f"{mid}.npz", **b)
    ix = []
    T, P = b["valid"].shape
    firstland = {}
    for r in raw["tables"]["parachute_landing"]:
        firstland.setdefault(r["actor"], r["t"])
    for k in range(3, T, 3):
        for p in np.flatnonzero(b["valid"][k]):
            if b["episode"][k, p] == 0 and b["times"][k] < firstland.get(int(p), 1e9):
                continue
            members = np.flatnonzero((b["team"] == b["team"][p]) & b["valid"][k])
            roster = np.sum(b["team"] == b["team"][p])
            if roster > 4:
                continue
            future = (
                b["target_valid"][k + 1 : min(k + 7, T), members]
                if False
                else b["target_valid"][k + 1 : min(k + 7, T)][:, members]
            )
            future = future & (
                b["target_episode"][k + 1 : min(k + 7, T)][:, members]
                == b["episode"][k, members][None]
            )
            egoj = list(members).index(p)
            has = bool(future[:, egoj].any())
            complete = len(future) == 6 and future.all()
            ix.append(
                [
                    int(row["match_index"]),
                    k,
                    int(p),
                    int(b["kind"][p]),
                    int(has),
                    int(complete),
                    len(members),
                ]
            )
    acc = Counter()
    if row["split"] == "train":
        for rs in positions:
            for a, d in zip(rs[:-1], rs[1:]):
                dt = d["t"] - a["t"]
                m = a["mode"]
                if not (
                    a["valid"]
                    and d["valid"]
                    and a["episode"] == d["episode"]
                    and m == d["mode"]
                    and m in (1, 2, 3)
                    and 5 <= dt <= 15
                ):
                    continue
                distance = np.linalg.norm(d["xyz"][:2] - a["xyz"][:2])
                if distance / dt > (12 if m == 1 else 90):
                    continue
                cell = lambda r: int(r["y"] // 100) * 82 + int(r["x"] // 100)
                aa, dd = cell(a), cell(d)
                if aa != dd:
                    acc[aa, dd, int(m != 1)] += 1
        np.savez_compressed(
            OUT / "spatial_match" / f"{mid}.npz",
            edges=np.array([[*k, v] for k, v in sorted(acc.items())], np.int64).reshape(-1, 4),
        )
    stats.update(
        anchors=len(ix),
        post_reentry_anchors=sum(b["episode"][r[1], r[2]] > 0 for r in ix),
        interpolated_targets=int(b["target_interpolated"].sum()),
    )
    return mid, row["split"], ix, stats, acc


def main():
    rows = list(csv.DictReader((OUT / "matches.csv").open()))
    fit = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "validation"]
    sets = {"items": set(), "vehicles": set()}
    for r in fit:
        raw = read_raw(OLD / "raw" / f"{r['match_id']}.json.gz")
        sets["items"].update(x["item"] for x in raw["tables"]["item_equip"] if x["item"])
        sets["vehicles"].update(
            x["vehicle_class"] for x in raw["tables"]["positions"] if x["vehicle_class"]
        )
    vocab = {k: {v: i + 2 for i, v in enumerate(sorted(s))} for k, s in sets.items()}
    (OUT / "vocabulary.json").write_text(json.dumps(vocab, indent=2))
    total = defaultdict(lambda: [0, 0])
    indices = {"train": [], "validation": []}
    quality = {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs = [pool.submit(one, r, vocab) for r in fit + dev]
        for i, f in enumerate(as_completed(fs), 1):
            mid, split, ix, stats, acc = f.result()
            indices[split] += ix
            quality[mid] = {"split": split, **{k: int(v) for k, v in stats.items()}}
            for key, value in acc.items():
                total[key][0] += value
                total[key][1] += 1
            if i % 50 == 0:
                print("BUILT", i, len(fit + dev), round(time.time() - start), flush=True)
    for split, ix in indices.items():
        np.save(OUT / f"{split}_index.npy", np.array(sorted(ix), np.int32))
    terrain = np.load(OLD / "spatial_prior.npz")["terrain"]
    edges = np.array([[*k, *v] for k, v in sorted(total.items())], np.int64).reshape(-1, 5)
    np.savez_compressed(OUT / "spatial_prior.npz", terrain=terrain, edges=edges)
    report = {
        "matches_built": len(quality),
        "split_counts": {"train": len(fit), "validation": len(dev), "test_locked_not_built": 94},
        "indices": {k: len(v) for k, v in indices.items()},
        "quality_totals": dict(
            sum(
                (Counter({k: v for k, v in r.items() if k != "split"}) for r in quality.values()),
                Counter(),
            )
        ),
        "spatial_edges": len(edges),
        "modes": MODE_NAMES,
        "test_status": "retrospective previously exposed, locked for v3; no bank/index/evaluation",
        "code_sha256": hashlib.sha256(
            (ROOT / "src/pubg_lab/movement/bank_v3.py").read_bytes()
        ).hexdigest(),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    (OUT / "per_match_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
