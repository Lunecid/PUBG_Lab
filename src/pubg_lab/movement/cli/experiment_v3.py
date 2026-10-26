"""Fixed-budget V3 paired experiment. Test access is explicitly forbidden."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.dataset_v3 import TrajectoryDataset, collate, DEFAULT
from pubg_lab.movement.dataset import NODE_TYPES
from pubg_lab.movement.model_v3 import JointMotion, constant_velocity
from pubg_lab.movement.evaluation import metrics, match_macro
from pubg_lab.movement.cli.experiment_v2 import select_plan, evaluation_plan, worker_init

OUT = ROOT / "reports/heterograph_diffusion_experiment_v3"
SEEDS = [41, 42, 43]
STEPS = 2000
BATCH = 8
SOURCES = [
    "src/pubg_lab/movement/bank_v3.py",
    "src/pubg_lab/movement/dataset_v3.py",
    "src/pubg_lab/movement/model_v3.py",
    "src/pubg_lab/movement/dataset.py",
    "src/pubg_lab/movement/model.py",
    "src/pubg_lab/movement/evaluation.py",
    "src/pubg_lab/movement/cli/experiment_v3.py",
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False))


def prepare():
    ds = TrajectoryDataset()
    val = TrajectoryDataset(split="validation")
    ids = evaluation_plan(val, seed=20261017)
    np.save(OUT / "evaluation_indices.npy", ids)
    plans = {}
    for seed in SEEDS:
        plan = select_plan(ds, seed, STEPS, BATCH)
        np.save(OUT / f"training_indices_seed{seed}.npy", plan)
        plans[str(seed)] = {
            "sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
            "draws": plan.size,
            "unique_windows": len(np.unique(plan)),
            "matches": len(np.unique(ds.index[plan.flatten(), 0])),
        }
    spec = {
        "steps": STEPS,
        "batch_size": BATCH,
        "seeds": SEEDS,
        "views": ["observer", "server"],
        "models": ["regression", "diffusion"],
        "plans": plans,
        "evaluation_count": len(ids),
        "evaluation_indices_sha256": sha(OUT / "evaluation_indices.npy"),
        "split_lock_sha256": sha(DEFAULT / "split_lock.json"),
        "source_sha256": {s: sha(ROOT / s) for s in SOURCES},
        "checkpoint": "Final EMA only; no dev selection",
        "test": "Locked retrospective holdout, no test loading; historical v2 exposure disclosed",
        "hypotheses": "Compare same-input independently trained joint regression vs smooth residual diffusion; observer vs server. Direct current-circle geometry must have measured sensitivity, not assumed benefit.",
        "primary": ["ego_meanpathADE_m", "team_meanpathADE_m", "joint_energy_score_m"],
        "secondary": ["ego_expectedADE_m", "pair_distance_MAE_m"],
        "zone_ablation": "Observer regression and diffusion seed41 retrained without circle numerical features or radial loss, same plan and budget. Single-seed exploratory ablation.",
        "posthoc_policy": "No tuning after comparison. Condition masking sensitivity is not a retrained ablation.",
    }
    p = OUT / "experiment_spec.json"
    if p.exists():
        assert json.loads(p.read_text()) == spec
    else:
        write(p, spec)
    print("PREPARED", len(ids), flush=True)


def hide_zone(c):
    c = dict(c)
    c["zone_geometry"] = torch.zeros_like(c["zone_geometry"])
    c["x"] = c["x"].clone()
    c["feature_mask"] = c["feature_mask"].clone()
    z = c["node_type"] == NODE_TYPES.index("zone")
    c["x"][z] = 0
    c["feature_mask"][z] = False
    return c


def train(view, kind, seed, workers=2, nozone=False):
    spec = json.loads((OUT / "experiment_spec.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in spec["source_sha256"].items())
    plan = np.load(OUT / f"training_indices_seed{seed}.npy")
    name = f"{view}_{kind}_{seed}" + ("_nozone" if nozone else "")
    path = OUT / f"{name}.pt"
    if path.exists():
        ck = torch.load(path, weights_only=False)
        assert ck["steps"] == STEPS and ck["source_sha256"] == spec["source_sha256"]
        print("COMPLETE_EXISTS", name, flush=True)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TrajectoryDataset(view=view)
    m = JointMotion(len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4, kind)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(
        ds,
        batch_sampler=plan.tolist(),
        collate_fn=collate,
        num_workers=workers,
        worker_init_fn=worker_init,
        prefetch_factor=2 if workers else None,
    )
    start = time.monotonic()
    logs = []
    recent = []
    with (OUT / f"{name}.jsonl").open("w") as f:
        for step, b in enumerate(loader, 1):
            lr = min(1, step / 100) * (
                5e-5
                + 4.5e-4 * 0.5 * (1 + math.cos(math.pi * max(0, step - 100) / max(1, STEPS - 100)))
            )
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss, parts = m.loss(
                hide_zone(b["conditions"]) if nozone else b["conditions"], b["targets"]
            )
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0, error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for a, z in zip(ema.parameters(), m.parameters()):
                    a.lerp_(z, 0.01)
            recent.append(float(loss.detach()))
            if step % 100 == 0 or step == 1 or step == STEPS:
                row = {
                    "step": step,
                    "loss_mean": float(np.mean(recent)),
                    "lr": lr,
                    "grad_norm": float(norm),
                    "elapsed_s": time.monotonic() - start,
                    **parts,
                }
                recent = []
                logs.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(name, row, flush=True)
            if step == STEPS:
                break
    torch.save(
        {
            "model": ema.state_dict(),
            "raw_model": m.state_dict(),
            "optimizer": opt.state_dict(),
            "kind": kind,
            "view": view,
            "seed": seed,
            "nozone": nozone,
            "steps": STEPS,
            "parameters": sum(p.numel() for p in m.parameters()),
            "plan_sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
            "source_sha256": spec["source_sha256"],
            "log": logs,
            "elapsed_s": time.monotonic() - start,
        },
        path,
    )
    print("TRAIN_COMPLETE", name, flush=True)


def evaluate():
    ids = np.load(OUT / "evaluation_indices.npy")
    predictions = {}
    meta = []
    targets = []
    presence = []
    initial = []
    base = []
    for view in ["observer", "server"]:
        ds = TrajectoryDataset(split="validation", view=view)
        models = {}
        records = {}
        for kind in ["regression", "diffusion"]:
            for seed in SEEDS:
                name = f"{view}_{kind}_{seed}"
                ck = torch.load(OUT / f"{name}.pt", weights_only=False)
                m = JointMotion(
                    len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4, kind
                )
                m.load_state_dict(ck["model"])
                m.eval()
                models[name] = (seed, m)
                records[name] = []
                predictions[name] = []
        for number, index in enumerate(ids):
            s = ds[int(index)]
            b = collate([s])
            c = b["conditions"]
            t = s["targets"]
            p = s["conditions"]["current_presence"]
            md = {k: v for k, v in s["meta"].items() if k != "origin_m"}
            md["index"] = int(index)
            if view == "observer":
                meta.append({**md, "origin_m": s["meta"]["origin_m"].tolist()})
                targets.append(t)
                presence.append(p)
                initial.append(s["conditions"]["initial_xy"])
                cv = constant_velocity(c)[0].numpy()[None]
                stationary = np.repeat(s["conditions"]["initial_xy"][:, None], 6, axis=1)[None]
                base.append(
                    {
                        **md,
                        "metrics": metrics(cv, t, p),
                        "stationary_metrics": metrics(stationary, t, p),
                    }
                )
                predictions.setdefault("constant_velocity", []).append(cv)
                predictions.setdefault("stationary", []).append(stationary)
            else:
                assert md["sample_id"] == meta[number]["sample_id"]
            for name, (seed, m) in models.items():
                torch.manual_seed(seed * 100000 + number)
                pred = m.sample(c, 20)[0].numpy()
                assert np.isfinite(pred).all()
                predictions[name].append(pred)
                records[name].append({**md, "metrics": metrics(pred, t, p)})
            if (number + 1) % 100 == 0:
                print("EVAL", view, number + 1, len(ids), flush=True)
        for name, r in records.items():
            write(OUT / f"evaluation_{name}.json", {"records": r, "match_macro": match_macro(r)})
    write(OUT / "evaluation_baselines.json", {"records": base})
    write(OUT / "evaluation_samples.json", meta)
    np.savez_compressed(OUT / "predictions.npz", **{k: np.stack(v) for k, v in predictions.items()})
    np.savez_compressed(
        OUT / "evaluation_targets.npz",
        **{k: np.stack([t[k] for t in targets]) for k in targets[0]},
        presence=np.stack(presence),
        initial_xy=np.stack(initial),
    )
    print("EVALUATION_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train-seed", "train-ablation", "evaluate"])
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.stage == "prepare":
        prepare()
    elif a.stage == "evaluate":
        evaluate()
    elif a.stage == "train-ablation":
        for kind in ["regression", "diffusion"]:
            train("observer", kind, 41, a.workers, nozone=True)
    else:
        for view in ["observer", "server"]:
            for kind in ["regression", "diffusion"]:
                train(view, kind, a.seed, a.workers)


if __name__ == "__main__":
    main()
