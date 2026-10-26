"""Fixed-budget paired experiment. Validation is evaluated only after all training.
Run: python tools/movement_diffusion/experiment_v2.py --output reports/heterograph_diffusion_experiment_v2
"""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.dataset import TrajectoryDataset, collate
from pubg_lab.movement.models.model import JointGraphDiffusion, constant_velocity
from pubg_lab.movement.evaluation.evaluation import metrics, match_macro


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def worker_init(_):
    torch.set_num_threads(1)


def select_plan(ds, seed, steps, batch):
    eligible = np.flatnonzero(ds.index[:, 4] > 0)
    groups = {
        int(m): eligible[ds.index[eligible, 0] == m] for m in np.unique(ds.index[eligible, 0])
    }
    rng = np.random.default_rng(seed)
    mids = sorted(groups)
    plan = []
    # Two independently drawn matches per batch; four windows per match. Uniform match weight.
    for _ in range(steps):
        ids = []
        for m in rng.choice(mids, 2):
            ids.extend(rng.choice(groups[int(m)], batch // 2).tolist())
        plan.append(ids)
    return np.asarray(plan, np.int64)


def evaluation_plan(ds, seed=20261015, per_match=8):
    rng = np.random.default_rng(seed)
    ids = []
    for m in np.unique(ds.index[:, 0]):
        eligible = np.flatnonzero((ds.index[:, 0] == m) & (ds.index[:, 4] > 0))
        ids.extend(
            sorted(rng.choice(eligible, min(per_match, len(eligible)), replace=False).tolist())
        )
    return np.asarray(ids, np.int64)


def train(args, view, seed, plan):
    path = args.output / f"{view}_seed{seed}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert (
            saved["step"] == args.steps
            and saved["plan_sha256"] == hashlib.sha256(plan.tobytes()).hexdigest()
        )
        print("reuse completed", path, flush=True)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = TrajectoryDataset(view=view)
    loader = DataLoader(
        ds,
        batch_sampler=plan.tolist(),
        collate_fn=collate,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        prefetch_factor=2 if args.workers else None,
    )
    model = JointGraphDiffusion(
        len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4
    )
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    start = time.monotonic()
    running = []
    logs = []
    modelsum = digest(ROOT / "src/pubg_lab/movement/models/model.py")
    logfile = args.output / f"{view}_seed{seed}.jsonl"
    with logfile.open("w") as f:
        for step, batch in enumerate(loader, 1):
            warm = min(1.0, step / 100)
            fraction = max(0, (step - 100) / max(1, args.steps - 100))
            lr = warm * (3e-5 + (3e-4 - 3e-5) * 0.5 * (1 + math.cos(math.pi * fraction)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(batch["conditions"], batch["targets"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at {view} {seed} {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters()):
                    a.lerp_(b, 0.005)
            running.append(float(loss.detach()))
            if step % 50 == 0 or step == 1 or step == args.steps:
                row = {
                    "step": step,
                    "loss_mean_last50": float(np.mean(running)),
                    "loss": float(loss.detach()),
                    "lr": lr,
                    "grad_norm": float(norm),
                    "elapsed_s": time.monotonic() - start,
                    **parts,
                }
                running = []
                logs.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(view, seed, row, flush=True)
            if step == args.steps:
                break
    payload = {
        "model": ema.state_dict(),
        "raw_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.steps,
        "seed": seed,
        "view": view,
        "architecture": "joint_member_equivariant_residual_v2",
        "hidden": 64,
        "diffusion_steps": 32,
        "residual_scale_km": 0.2,
        "parameters": sum(p.numel() for p in model.parameters()),
        "plan_sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
        "model_code_sha256": modelsum,
        "training_log": logs,
        "elapsed_s": time.monotonic() - start,
        "vocabulary": ds.builder.vocab,
    }
    torch.save(payload, path)
    print("TRAIN_COMPLETE", str(path), payload["elapsed_s"], flush=True)


def evaluate(args, ids):
    base = []
    predictions = {}
    eval_meta = []
    targs = []
    pres = []
    initials = []
    for view in ["observer", "server"]:
        ds = TrajectoryDataset(split="validation", view=view)
        models = []
        for seed in args.seeds:
            ck = torch.load(
                args.output / f"{view}_seed{seed}.pt", map_location="cpu", weights_only=False
            )
            m = JointGraphDiffusion(
                len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4
            )
            m.load_state_dict(ck["model"])
            m.eval()
            models.append(m)
            predictions[f"{view}_{seed}"] = []
        records = {seed: [] for seed in args.seeds}
        for number, index in enumerate(ids):
            sample = ds[int(index)]
            batch = collate([sample])
            c = batch["conditions"]
            target = sample["targets"]
            p = sample["conditions"]["current_presence"]
            meta = {k: v for k, v in sample["meta"].items() if k != "origin_m"}
            # Slice metadata as-of only. Subgroups are descriptive, never training controls.
            mi, k, ego, *_ = ds.index[index]
            bank = ds.bank(ds.rows[int(mi)]["match_id"])
            meta.update(
                current_members=int(p.sum()),
                vehicle=bool(bank["x"][k, ego, 14]),
                combat=bool(bank["x"][k, ego, 10] > 0),
                phase=int(bank["game"][k, 0]),
                index=int(index),
            )
            if view == "observer":
                cv = constant_velocity(c)[0].numpy()
                stationary = np.repeat(sample["conditions"]["initial_xy"][:, None], 6, axis=1)
                base.append(
                    {
                        **meta,
                        "metrics": metrics(cv[None], target, p),
                        "stationary_metrics": metrics(stationary[None], target, p),
                    }
                )
                eval_meta.append({**meta, "origin_m": sample["meta"]["origin_m"].tolist()})
                targs.append(target)
                pres.append(p)
                initials.append(sample["conditions"]["initial_xy"])
                predictions.setdefault("constant_velocity", []).append(cv[None])
                predictions.setdefault("stationary", []).append(stationary[None])
            else:
                assert meta["sample_id"] == eval_meta[number]["sample_id"]
            for seed, m in zip(args.seeds, models):
                torch.manual_seed(seed * 100000 + number)
                pred = m.sample(c, args.scenarios)[0].numpy()
                assert np.isfinite(pred).all()
                predictions[f"{view}_{seed}"].append(pred)
                records[seed].append({**meta, "metrics": metrics(pred, target, p)})
            if (number + 1) % 100 == 0:
                print("EVAL", view, number + 1, "/", len(ids), flush=True)
        for seed in args.seeds:
            write(
                args.output / f"evaluation_{view}_{seed}.json",
                {"records": records[seed], "match_macro": match_macro(records[seed])},
            )
    write(
        args.output / "evaluation_baselines.json",
        {
            "records": base,
            "cv_match_macro": match_macro(base),
            "stationary_match_macro": match_macro(
                [{**r, "metrics": r["stationary_metrics"]} for r in base]
            ),
        },
    )
    write(args.output / "evaluation_samples.json", eval_meta)
    np.savez_compressed(
        args.output / "predictions.npz", **{k: np.stack(v) for k, v in predictions.items()}
    )
    np.savez_compressed(
        args.output / "evaluation_targets.npz",
        **{k: np.stack([v[k] for v in targs]) for k in targs[0]},
        presence=np.stack(pres),
        initial_xy=np.stack(initials),
    )
    print("EVALUATION_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    assert args.batch_size % 2 == 0
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    trainset = TrajectoryDataset()
    val = TrajectoryDataset(split="validation")
    ids = evaluation_plan(val)
    np.save(args.output / "evaluation_indices.npy", ids)
    plans = {seed: select_plan(trainset, seed, args.steps, args.batch_size) for seed in args.seeds}
    for seed, plan in plans.items():
        np.save(args.output / f"training_indices_seed{seed}.npy", plan)
    config = {
        "status": "Fixed before validation metric inspection; local exploratory experiment, not preregistration",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "learning_rate": "100 step warmup to 3e-4, cosine decay to 3e-5",
        "ema_decay": 0.995,
        "checkpoint_selection": "Final update only, no validation tuning",
        "evaluation": "8 uniformly sampled eligible windows per held-out match, seed20261015; 188 matches",
        "primary_metrics": ["ego_meanpathADE_m", "team_meanpathADE_m", "joint_energy_score_m"],
        "secondary": "expected ADE, best-of-20 ADE/FDE, pair distance MAE; best-of-K is oracle coverage, not comparable alone to deterministic CV",
        "training_population": "all 751 training matches eligible; not all 1.45m windows visited",
        "plans": {
            str(seed): {
                "sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
                "draws": int(plan.size),
                "unique_windows": int(len(np.unique(plan))),
                "matches": int(len(np.unique(trainset.index[plan.flatten(), 0]))),
            }
            for seed, plan in plans.items()
        },
        "evaluation_count": len(ids),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "src/pubg_lab/movement/models/model.py",
                ROOT / "src/pubg_lab/movement/data/dataset.py",
                ROOT / "src/pubg_lab/movement/evaluation/evaluation.py",
                Path(__file__),
            ]
        },
    }
    spec = args.output / "experiment_spec.json"
    if spec.exists():
        assert json.loads(spec.read_text()) == config, (
            "Experiment specification changed; use a new output directory"
        )
    else:
        write(spec, config)
    if not args.eval_only:
        for seed in args.seeds:
            for view in ["observer", "server"]:
                train(args, view, seed, plans[seed])
    evaluate(args, ids)


if __name__ == "__main__":
    main()
