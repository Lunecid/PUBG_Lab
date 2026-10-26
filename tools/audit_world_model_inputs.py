"""Full allowed-bank audit; never opens protected match banks."""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, json, argparse, hashlib, sqlite3, time
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from world_model.indexed_dataset import IndexedWorldDataset
from world_model.inputs import ConditionVocabulary, load_sample
from tools.verify_world_model_encoder import ResourceSampler, sha


def initialize(root):
    global datasets, vocabulary, worker_memory
    torch.set_num_threads(1)
    worker_memory = ResourceSampler()
    worker_memory.__enter__()
    datasets = {r: IndexedWorldDataset(root, role=r) for r in ("train", "validation")}
    vocabulary = ConditionVocabulary.from_dataset(datasets["train"])


def audit(task):
    role, mid, support_hash = task
    ds = datasets[role]
    result = {"match_id": mid, "role": role, "errors": []}
    def check(value, name):
        if not bool(value):
            result["errors"].append(name)
    try:
        b = ds.bank(mid)
        t, p, f = b["x"].shape
        check(f == 17, "feature_width")
        check(np.isfinite(b["x"][b["mask"]]).all(), "nonfinite_observed_x")
        check(np.isfinite(b["game"][b["game_mask"]]).all(), "nonfinite_game")
        check(np.isfinite(b["target"][b["target_valid"]]).all(), "nonfinite_target")
        check(np.all(np.diff(b["times"]) > 0), "nonmonotone_time")
        check(np.all(b["source_time"][b["valid"]] <=
                     np.broadcast_to(b["times"][:, None], (t, p))[b["valid"]]+1e-6), "future_position")
        check(np.all(b["game_time"] <= b["times"]+1e-6), "future_game")
        check(np.all((b["mobility"] >= 0) & (b["mobility"] < 10)), "mode_range")
        a = b["anchors"]
        check(a.ndim == 2 and a.shape[1] == 2, "anchor_shape")
        check(np.all((a[:, 0] >= 12) & (a[:, 0] < t) & (a[:, 1] >= 0) & (a[:, 1] < p)),
              "anchor_bounds")
        check(np.all(b["valid"][a[:, 0], a[:, 1]] & b["known"][a[:, 0], a[:, 1]]), "anchor_invalid")
        meta = json.loads((ds.root/"meta"/f"{mid}.json").read_text())
        items = b["items_sparse"]
        if len(items):
            check(np.all((items >= 0) & (items < np.array([t, p, len(meta["item_vocabulary_local"])]))),
                  "item_bounds")
        if role == "train":
            edges = np.asarray(sorted(map(tuple, b["spatial_edges"])), dtype=np.int64).reshape(-1, 6)
            check(hashlib.sha256(edges.tobytes()).hexdigest() == support_hash, "catalog_bank_spatial_mismatch")
        ks = np.unique(a[:, 0])
        q = ks[:, None]+np.arange(1, 7)[None, :]  # 10s bank, 10s horizons
        qc = np.minimum(q, t-1)
        # K,H,P -> K,P,H
        future = (b["target_valid"][qc] & (q < t)[:, :, None]
                  & (b["target_episode"][qc] == b["episode"][ks, None, :])
                  & (b["target_segment"][qc] == b["segment"][ks, None, :])).transpose(0, 2, 1)
        current = b["valid"][ks] & b["known"][ks]
        future &= current[:, :, None]
        full = future.all(-1)
        lookup = {int(k): i for i, k in enumerate(ks)}
        ai = np.array([lookup[int(k)] for k in a[:, 0]], dtype=int)
        result.update(anchors=len(a), ego_full60=int(full[ai, a[:, 1]].sum()),
                      ego_any=int(future.any(-1)[ai, a[:, 1]].sum()), team_times=0,
                      joint_full_team_times=0, joint_full_anchors=0, pair_labels=0)
        for team in np.unique(b["team"]):
            ids = b["team"] == team
            n = current[:, ids].sum(-1)
            joint = ((~current[:, ids]) | full[:, ids]).all(-1) & (n > 0)
            result["team_times"] += int((n > 0).sum())
            result["joint_full_team_times"] += int(joint.sum())
            result["joint_full_anchors"] += int(n[joint].sum())
            nf = future[:, ids].sum(1)
            result["pair_labels"] += int((nf*(nf-1)//2).sum())
        if len(a):
            targets = None
            for view in ("observer", "server"):
                ds.view = view
                sample = load_sample(ds, mid, 0, vocabulary)
                g = sample.inputs
                for field in fields(g):
                    val = getattr(g, field.name)
                    if isinstance(val, torch.Tensor) and val.is_floating_point():
                        check(torch.isfinite(val).all(), "adapter_nonfinite_"+field.name)
                if view == "observer":
                    others = torch.arange(g.x.shape[1]) != g.ego_local_index
                    check(not g.mask[:, others, 8:].any(), "observer_private")
                    check(not g.game_mask[:, 2].any(), "observer_alive_teams")
                current_target = np.asarray(sample.targets["xy"])
                if targets is not None:
                    check(np.array_equal(targets, current_target, equal_nan=True), "view_target_mismatch")
                targets = current_target
            result["adapter_views"] = 2
        result["bank_sha256"] = sha(ds.root/"banks"/f"{mid}.npz")
        result["quality_issues"] = meta.get("issues", [])
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    worker_memory.measure()
    result["worker_pid"] = os.getpid()
    result["worker_peak_uss"] = worker_memory.max_uss
    if worker_memory.errors or worker_memory.over_limit:
        result["errors"].append("worker_resource_measurement")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    ds = IndexedWorldDataset(args.root)
    plan = ds.plan
    train, val = set(plan["train_match_ids"]), set(plan["validation_match_ids"])
    assert not train & val
    assert all(ds.metadata[m]["role"] == "train" for m in train)
    assert all(ds.metadata[m]["role"] == "validation" for m in val)
    # Only the authorized fit rows; no protected-match details fetched.
    support = defaultdict(list)
    db = sqlite3.connect(f"file:{ds.root/'catalog.sqlite'}?mode=ro", uri=True)
    for row in db.execute("SELECT match_id,sx,sy,dx,dy,mode,n FROM spatial_support WHERE match_id IN ("+
                          ",".join("?" for _ in train)+")", sorted(train)):
        support[row[0]].append(row[1:])
    db.close()
    hashes = {mid: hashlib.sha256(np.asarray(sorted(support[mid]), dtype=np.int64).reshape(-1, 6).tobytes()).hexdigest()
              for mid in train}
    del support
    tasks = [(r, m, hashes.get(m)) for r, ids in (("train", train), ("validation", val))
             for m in sorted(ids, key=lambda m: (ds.metadata[m]["geometry_key"], m))]
    started = time.monotonic()
    totals = {r: Counter() for r in ("train", "validation")}
    failures = []
    worker_peaks = {}
    with ResourceSampler() as memory, ProcessPoolExecutor(max_workers=4, initializer=initialize,
                                                          initargs=(args.root,)) as pool:
        with (out/"match_audit.jsonl").open("w") as handle:
            for i, row in enumerate(pool.map(audit, tasks, chunksize=8)):
                handle.write(json.dumps(row)+"\n")
                totals[row["role"]]["matches"] += 1
                for k in ("anchors", "ego_full60", "ego_any", "team_times", "joint_full_team_times",
                          "joint_full_anchors", "pair_labels", "adapter_views"):
                    totals[row["role"]][k] += row.get(k, 0)
                if row["errors"]:
                    failures.append({"match_id": row["match_id"], "errors": row["errors"]})
                # Include all audit worker private memory in hard ceiling.
                worker_peaks[row["worker_pid"]] = row["worker_peak_uss"]
                uss = memory.proc.memory_full_info().uss + sum(worker_peaks.values())
                memory.max_uss = max(memory.max_uss, uss)
                if uss >= 48*1024**3:
                    raise MemoryError("Audit aggregate USS ceiling")
                if (i+1) % 250 == 0:
                    print(json.dumps({"checked": i+1, "errors": len(failures),
                                      "elapsed_s": round(time.monotonic()-started)}), flush=True)
    summary = {"status": "PASS" if not failures else "FAIL", "totals": totals,
               "failures": failures, "elapsed_s": time.monotonic()-started,
               "max_aggregate_uss_bytes": memory.max_uss, "protected_details_read": False,
               "scope": "All allowed bank arrays/anchors; first anchor both adapters per nonempty bank",
               "source_sha256": sha(__file__), "manifest_sha256": sha(ds.root/"ablations/full.json")}
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
