"""Preregistered bounded CPU pretraining and frozen representation probe."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import argparse
import time
import copy
import platform
import csv
import traceback
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import torch
import psutil
from pubg_lab.world_model.indexed_dataset import IndexedWorldDataset
from pubg_lab.world_model.inputs import ConditionVocabulary, load_sample
from pubg_lab.world_model.encoder import WorldEncoder
from pubg_lab.world_model.pretraining import (
    PretrainHeads,
    PathProbe,
    pretrain_losses,
    supervised_targets,
    masked_mse,
    pair_targets,
)
from pubg_lab.world_model.cli.verify_world_model_encoder import ResourceSampler, tensor_hash, sha

SEED = 20261020
REPO = REPO_ROOT


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def prepare(root, audit_path, out):
    audit = json.loads(Path(audit_path).read_text())
    assert audit["status"] == "PASS", "Full input audit must pass before training"
    out.mkdir(parents=True, exist_ok=False)
    data = {"samples": {}, "selection": {}, "metadata": {}}
    for role, limit, per_match in (("train", 32, 8), ("validation", 16, 4)):
        ds = IndexedWorldDataset(root, role=role)
        vocab = ConditionVocabulary.from_dataset(ds)
        data.update(
            item_count=len(ds.item_vocabulary) + 2,
            condition_sizes=vocab.sizes,
            vocabulary=vocab.tokens,
            item_vocabulary=ds.item_vocabulary,
        )
        selected, samples = [], {"observer": [], "server": []}
        for mid in sorted(ds.match_ids, key=digest):
            bank = ds.bank(mid)
            if not len(bank["anchors"]):
                continue
            candidates = defaultdict(list)
            seen = set()
            for index, (k, ego) in enumerate(bank["anchors"]):
                team = int(bank["team"][ego])
                key = (team, int(k))
                if key not in seen:
                    seen.add(key)
                    candidates[team].append((index, int(k), int(ego)))
            teams = sorted(candidates, key=lambda team: digest((mid, team)))
            for team in teams:
                candidates[team].sort(key=lambda row: digest((mid, team, row[1])))
            chosen = []
            depth = 0
            while len(chosen) < per_match:
                more = [
                    (team, candidates[team][depth])
                    for team in teams
                    if len(candidates[team]) > depth
                ]
                if not more:
                    break
                chosen.extend(more[: per_match - len(chosen)])
                depth += 1
            for team, (index, k, ego) in chosen:
                for view in samples:
                    ds.view = view
                    s = load_sample(ds, mid, index, vocab)
                    s.provenance.update(
                        team_id=team,
                        role=role,
                        map=ds.metadata[mid]["map"],
                        nominal_team_size=ds.metadata[mid]["nominal_team_size"],
                        perspective=ds.metadata[mid]["perspective"],
                        kind_eval_only=bank["kind_eval_only"][
                            s.inputs.actor_ids[s.inputs.target_player_indices]
                        ].tolist(),
                    )
                    samples[view].append(s)
                selected.append(
                    {
                        "match_id": mid,
                        "anchor_index": index,
                        "frame": k,
                        "ego": ego,
                        "team_id": team,
                    }
                )
            data["metadata"][mid] = ds.metadata[mid]
            if len({x["match_id"] for x in selected}) >= limit:
                break
        data["samples"][role], data["selection"][role] = samples, selected
        print(f"prepared {role}: {len(selected)} anchors", flush=True)
    torch.save(data, out / "samples.pt")
    manifest = {k: v for k, v in data.items() if k != "samples"}
    manifest.update(
        audit_sha256=sha(audit_path),
        source_sha256=sha(__file__),
        samples_sha256=sha(out / "samples.pt"),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def order(samples, steps):
    groups = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(samples):
        groups[s.provenance["match_id"]][s.provenance["team_id"]].append(i)
    rng = np.random.default_rng(SEED)
    mids = sorted(groups)
    rows = []
    for _ in range(steps):
        batch = []
        for _ in range(2):
            mid = mids[int(rng.integers(len(mids)))]
            teams = sorted(groups[mid])
            team = teams[int(rng.integers(len(teams)))]
            ids = groups[mid][team]
            batch.append(ids[int(rng.integers(len(ids)))])
        rows.append(batch)
    return rows


def adam(params):
    return torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4, foreach=False)


def loss_summary(encoder, heads, samples):
    values = defaultdict(list)
    with torch.no_grad():
        for i, s in enumerate(samples):
            _, losses, _ = pretrain_losses(encoder, heads, s, SEED + i)
            for name, value in losses.items():
                if value is not None:
                    values[name].append(float(value))
    return {k: float(np.mean(v)) for k, v in values.items()}


def evaluate_prediction(sample, delta):
    g = sample.inputs
    y = supervised_targets(sample)
    pred = (delta + g.x[-1, g.target_player_indices, None, :2]) * 100
    truth = y["xy"] * 100
    mask = y["mask"]
    distance = (pred - truth).norm(dim=-1)
    pd, pm, _ = pair_targets(pred, mask)
    td, _, _ = pair_targets(truth, mask)

    def avg(value, valid):
        return float(value[valid].mean()) if valid.any() else None

    return (
        {
            "ADE": avg(distance, mask),
            "FDE": avg(distance[:, -1], mask[:, -1]),
            "pair_MAE": avg((pd - td).abs(), pm),
            "valid_points": int(mask.sum()),
            "valid_pairs": int(pm.sum()),
            "current_members": len(mask),
            "complete60": bool(mask.all()),
        },
        pred,
        truth,
        mask,
    )


def aggregate(rows):
    result = {}
    for arm in ("pretrained", "random", "cv"):
        selected = [r for r in rows if r["arm"] == arm]
        result[arm] = {}
        for metric in ("ADE", "FDE", "pair_MAE"):
            by_match_team = defaultdict(lambda: defaultdict(list))
            for r in selected:
                if r[metric] is not None:
                    by_match_team[r["match_id"]][r["team_id"]].append(r[metric])
            by_match = {
                m: float(np.mean([np.mean(v) for v in teams.values()]))
                for m, teams in by_match_team.items()
            }
            result[arm][metric] = {
                "mean": float(np.mean(list(by_match.values()))) if by_match else None,
                "per_match": by_match,
                "n_matches": len(by_match),
            }
    return result


def run_view(data, view, out, memory):
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > 600:
            raise TimeoutError("Preregistered 600 s/view ceiling")
        if memory.over_limit or memory.errors:
            raise MemoryError("Private memory measurement/limit")

    train, dev = data["samples"]["train"][view], data["samples"]["validation"][view]
    torch.manual_seed(SEED)
    encoder = WorldEncoder(data["item_count"], data["condition_sizes"])
    random_encoder = copy.deepcopy(encoder)
    heads = PretrainHeads()
    parameters = list(encoder.parameters()) + list(heads.parameters())
    optimizer = adam(parameters)
    before_hash = tensor_hash(encoder.state_dict().items())
    before_loss = loss_summary(encoder, heads, train[:8])
    task_gradient_norms = {}
    for i, sample in enumerate(train[:8]):
        _, parts, _ = pretrain_losses(encoder, heads, sample, SEED + i)
        for name, value in parts.items():
            if name not in task_gradient_norms and value is not None:
                grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                task_gradient_norms[name] = float(
                    torch.sqrt(sum(g.square().sum() for g in grads if g is not None))
                )
    curve, gradient_groups, support = [], defaultdict(list), Counter()
    for step, batch in enumerate(order(train, 256)):
        guard()
        optimizer.zero_grad(set_to_none=True)
        values, tasks = [], defaultdict(list)
        for n, i in enumerate(batch):
            loss, parts, counts = pretrain_losses(encoder, heads, train[i], SEED + 2 * step + n)
            support.update(counts)
            if loss is not None:
                values.append(loss)
                for k, v in parts.items():
                    if v is not None:
                        tasks[k].append(float(v.detach()))
        if not values:
            continue
        total = torch.stack(values).mean()
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite pretraining loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                gradient_groups[name.split(".")[0]].append(float(param.grad.norm()))
        optimizer.step()
        curve.append(
            {
                "step": step + 1,
                "total": float(total.detach()),
                "grad_norm_before_clip": float(norm),
                **{k: float(np.mean(v)) for k, v in tasks.items()},
            }
        )
        if (step + 1) % 32 == 0:
            print(json.dumps({"view": view, "stage": "pretrain", **curve[-1]}), flush=True)
    after_hash = tensor_hash(encoder.state_dict().items())
    assert before_hash != after_hash, "Encoder parameters did not update"
    after_loss = loss_summary(encoder, heads, train[:8])
    dev_loss = loss_summary(encoder, heads, dev)
    rows, prediction_tensors, state = (
        [],
        [],
        {"encoder": encoder.state_dict(), "heads": heads.state_dict()},
    )
    probe_curves = {}
    for arm, enc in (("pretrained", encoder), ("random", random_encoder)):
        enc.eval()
        enc.requires_grad_(False)
        with torch.no_grad():
            cached = [enc(s.inputs) for s in train]
            cached_dev = [enc(s.inputs) for s in dev]
        torch.manual_seed(SEED + 1)
        probe = PathProbe()
        opt = adam(probe.parameters())
        probe_curve = []
        for step, batch in enumerate(order(train, 128)):
            guard()
            opt.zero_grad(set_to_none=True)
            values = []
            for i in batch:
                y = supervised_targets(train[i])
                loss = masked_mse(
                    probe(cached[i], train[i].inputs), y["delta"], y["mask"][..., None]
                )
                if loss is not None:
                    values.append(loss)
            if not values:
                continue
            loss = torch.stack(values).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            probe_curve.append({"step": step + 1, "loss": float(loss.detach())})
        probe_curves[arm] = probe_curve
        state[arm + "_probe"] = probe.state_dict()
        with torch.no_grad():
            for i, (s, latent) in enumerate(zip(dev, cached_dev)):
                delta = probe(latent, s.inputs)
                metrics, pred, truth, mask = evaluate_prediction(s, delta)
                rows.append({**s.provenance, "arm": arm, **metrics})
                prediction_tensors.append((f"{arm}_{i}", pred))
                if arm == "pretrained":
                    g = s.inputs
                    cv = (
                        g.x[-1, g.target_player_indices, None, 3:5]
                        * torch.arange(1, 7)[None, :, None]
                    )
                    cm, cp, _, _ = evaluate_prediction(s, cv)
                    fallback = ~g.mask[-1, g.target_player_indices, 3:5].all(-1)
                    rows.append(
                        {
                            **s.provenance,
                            "arm": "cv",
                            **cm,
                            "cv_fallback_players": int(fallback.sum()),
                        }
                    )
                    prediction_tensors += [
                        (f"cv_{i}", cp),
                        (f"truth_{i}", truth),
                        (f"mask_{i}", mask),
                    ]
    torch.save(state, out / f"{view}_checkpoint.pt")
    torch.save(dict(prediction_tensors), out / f"{view}_predictions.pt")
    (out / f"{view}_rows.json").write_text(json.dumps(rows, indent=2))
    state_hash = tensor_hash(
        (f"{group}.{k}", v) for group, d in state.items() for k, v in d.items()
    )
    summary = {
        "view": view,
        "elapsed_s": time.monotonic() - started,
        "optimizer_steps": {
            "pretrain": len(curve),
            "probes": {k: len(v) for k, v in probe_curves.items()},
        },
        "encoder_before_hash": before_hash,
        "encoder_after_hash": after_hash,
        "state_hash": state_hash,
        "prediction_hash": tensor_hash(prediction_tensors),
        "fixed_train_loss_before": before_loss,
        "fixed_train_loss_after": after_loss,
        "development_pretrain_losses": dev_loss,
        "task_training_support": dict(support),
        "initial_unweighted_task_gradient_norms": task_gradient_norms,
        "gradient_group_mean": {k: float(np.mean(v)) for k, v in gradient_groups.items()},
        "curve": curve,
        "probe_curves": probe_curves,
        "metrics": aggregate(rows),
    }
    (out / f"{view}_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "view": view,
                "finished": True,
                "elapsed_s": summary["elapsed_s"],
                "metrics": {
                    arm: {k: v["mean"] for k, v in m.items()}
                    for arm, m in summary["metrics"].items()
                },
            }
        ),
        flush=True,
    )
    return {k: v for k, v in summary.items() if k not in ("curve", "probe_curves")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--audit")
    ap.add_argument("--prepared")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    competing = [
        (p.pid, p.info["name"])
        for p in psutil.process_iter(["name"])
        if any(s in (p.info["name"] or "").lower() for s in ("strata", "llama"))
    ]
    if competing:
        raise RuntimeError(f"Competing process review: {competing}")
    out = Path(args.output)
    if args.prepare:
        with ResourceSampler():
            prepare(args.root, args.audit, out)
        return
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    sources = list((REPO / "src/pubg_lab/world_model").glob("*.py")) + [
        Path(__file__),
        REPO / "src/pubg_lab/common/paths.py",
        REPO / "src/pubg_lab/common/__init__.py",
        REPO / "src/pubg_lab/__init__.py",
        REPO / "pyproject.toml",
        REPO / "docs/experiment_design/WORLD_MODEL_G2_CPU_PROTOCOL_2026-10-20.md",
    ]
    hashes = {}
    for path in sources:
        relative = path.relative_to(REPO)
        dst = source / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
        hashes[str(relative)] = sha(path)
    receipt = {
        "device": "cpu",
        "seed": SEED,
        "threads": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sources": hashes,
        "prepared_sha256": sha(Path(args.prepared) / "samples.pt"),
        "protected_details_read": False,
        "views": {},
    }
    try:
        with ResourceSampler() as memory:
            data = torch.load(
                Path(args.prepared) / "samples.pt", weights_only=False, map_location="cpu"
            )
            for view in ("observer", "server"):
                receipt["views"][view] = run_view(data, view, out, memory)
            receipt["max_uss_bytes"] = memory.max_uss
            receipt["status"] = "PASS"
    except Exception:
        receipt["status"] = "FAIL"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
