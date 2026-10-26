"""G4 preregistered CPU pilot; fixed fit universe, sampling and prior ablations."""

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
import hashlib
import sqlite3
import time
import copy
import platform
import traceback
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import torch
import psutil
from pubg_lab.world_model.cohorts import CohortDataset
from pubg_lab.world_model.temporal_prior import TemporalPriorDataset, PastWorldEncoder
from pubg_lab.world_model.inputs import ConditionVocabulary
from pubg_lab.world_model.pretraining import (
    PretrainHeads,
    PathProbe,
    pretrain_losses,
    supervised_targets,
    masked_mse,
    pair_targets,
)
from pubg_lab.world_model.cli.verify_world_model_encoder import ResourceSampler, tensor_hash, sha
from pubg_lab.world_model.cli.train_world_model_g2 import evaluate_prediction, adam

REPO = REPO_ROOT
STRATA = ("emergency30", "reentry30", "emergency_air", "reentry_air")


def digest(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


def choose(root, g3, out):
    plan = json.loads((g3 / "index_02/primary_br_manifest.json").read_text())
    selected = {}
    for role, limit in (("train", 32), ("validation", 16)):
        ds = CohortDataset(root, g3 / "index_02/primary_br_manifest.json", role)
        records = []
        for mid in sorted(ds.match_ids, key=digest):
            b = ds.bank(mid)
            by_team = defaultdict(list)
            seen = set()
            for k, a in b["anchors"]:
                if k < 12 or not b["valid"][k, a] or not b["known"][k, a]:
                    continue
                key = int(b["team"][a]), int(k)
                if key in seen:
                    continue
                seen.add(key)
                by_team[key[0]].append((int(k), int(a)))
            choices = []
            teams = sorted(by_team, key=lambda t: digest((mid, t)))
            for t in teams:
                by_team[t].sort(key=lambda r: digest((mid, t, r)))
            depth = 0
            while len(choices) < 4:
                more = [(t, by_team[t][depth]) for t in teams if len(by_team[t]) > depth]
                if not more:
                    break
                choices += more[: 4 - len(choices)]
                depth += 1
            records += [
                dict(match_id=mid, frame=k, actor=a, team_id=t, stratum="natural", role=role)
                for t, (k, a) in choices
            ]
            if len({r["match_id"] for r in records}) >= limit:
                break
        selected[role] = records
    db = sqlite3.connect(f"file:{g3 / 'index_02/cohort_events.sqlite'}?mode=ro", uri=True)
    pool = defaultdict(list)
    for mid, k, a, role, kind, team in db.execute("""
        SELECT s.match_id,s.frame,s.actor,s.role,e.kind,e.team
        FROM selected s JOIN selected_events se USING(match_id,frame,actor)
        JOIN events e ON e.event_id=se.event_id
        WHERE s.cohort IN ('normal_br','ranked_br')
          AND e.kind IN ('emergency_pickup','reentry_direct')"""):
        st = "emergency30" if kind == "emergency_pickup" else "reentry30"
        pool[role, st].append(
            dict(match_id=mid, frame=k, actor=a, team_id=team, stratum=st, role=role)
        )
    for line in (g3 / "short_transport_01/samples.jsonl").open():
        r = json.loads(line)
        st = "emergency_air" if r["kind"] == "emergency_pickup" else "reentry_air"
        team = db.execute("SELECT team FROM events WHERE event_id=?", (r["event_id"],)).fetchone()[
            0
        ]
        pool[r["role"], st].append(
            dict(
                match_id=r["match_id"],
                frame=r["frame"],
                actor=r["actor"],
                team_id=team,
                stratum=st,
                role=r["role"],
            )
        )
    db.close()
    for role, n in (("train", 16), ("validation", 8)):
        seen = {(r["match_id"], r["frame"], r["actor"]) for r in selected[role]}
        for st in STRATA:
            mids = set()
            chosen = []
            for r in sorted(
                pool[role, st], key=lambda r: digest((r["match_id"], r["frame"], r["actor"]))
            ):
                key = r["match_id"], r["frame"], r["actor"]
                if key in seen or r["match_id"] in mids:
                    continue
                assert r["match_id"] in plan[role + "_match_ids"]
                seen.add(key)
                mids.add(r["match_id"])
                chosen.append(r)
                if len(chosen) == n:
                    break
            assert len(chosen) == n, (role, st)
            selected[role] += chosen
    actual = dict(
        cohort="g4_pilot",
        train_match_ids=sorted({r["match_id"] for r in selected["train"]}),
        validation_match_ids=sorted({r["match_id"] for r in selected["validation"]}),
    )
    (out / "fit_manifest.json").write_text(json.dumps(actual, indent=2))
    (out / "selection.json").write_text(json.dumps(selected, indent=2))
    return selected


def prepare(root, g3, out, memory):
    selected = choose(root, g3, out)
    data = dict(samples={}, selection=selected)
    checks = []
    for role in ("train", "validation"):
        ds = TemporalPriorDataset(
            root, out / "fit_manifest.json", g3 / "prior_01/historical_prior.sqlite", role
        )
        v = ConditionVocabulary.from_dataset(ds)
        data.update(
            item_count=len(ds.item_vocabulary) + 2,
            condition_sizes=v.sizes,
            vocabulary=v.tokens,
            item_vocabulary=ds.item_vocabulary,
        )
        data["samples"][role] = {}
        for view in ("observer", "server"):
            ds.view = view
            policies = {}
            for policy in ("H0", "H1"):
                ds.policy = policy
                ss = []
                # Preserve the locked order; cohort/selection tags never enter inputs.
                for r in selected[role]:
                    s = ds.load(r["match_id"], r["frame"], r["actor"], v)
                    s.provenance.update(role=role, stratum=r["stratum"])
                    ss.append(s)
                policies[policy] = ss
                checks.append(
                    dict(
                        role=role,
                        view=view,
                        policy=policy,
                        samples=len(ss),
                        edges=sum(len(s.inputs.past_edges) for s in ss),
                        matches=len({s.provenance["match_id"] for s in ss}),
                    )
                )
                print(json.dumps(checks[-1]), flush=True)
            if role == "train":
                for a, b in zip(policies["H0"], policies["H1"]):
                    assert input_hash(a) == input_hash(b), "Train H0/H1 differ"
            data["samples"][role][view] = policies
        ds.prior.close()
    # Actual-data gradient gate before ANY optimizer step.
    sample = next(s for s in data["samples"]["train"]["observer"]["H1"] if len(s.inputs.past_edges))
    torch.manual_seed(20261020)
    enc = PastWorldEncoder(data["item_count"], data["condition_sizes"])
    heads = PretrainHeads(d=64)
    loss, parts, _ = pretrain_losses(enc, heads, sample, 20261020)
    loss.backward()
    grads = {
        n: float(p.grad.norm())
        for n, p in enc.named_parameters()
        if n.startswith("spatial.") and p.grad is not None
    }
    for suffix in (
        "message.weight",
        "gate.weight",
        "support.weight",
        "mode.weight",
        "patch.weight",
    ):
        assert all(
            np.isfinite(grads[f"spatial.{i}.{suffix}"]) and grads[f"spatial.{i}.{suffix}"] > 0
            for i in range(2)
        )
    assert not memory.errors and not memory.over_limit
    torch.save(data, out / "samples.pt")
    summary = dict(
        status="PASS",
        checks=checks,
        gradient_norms=grads,
        samples_sha256=sha(out / "samples.pt"),
        fit_manifest_sha256=sha(out / "fit_manifest.json"),
        protocol_sha256=sha(
            REPO / "docs/experiment_design/WORLD_MODEL_G4_CPU_PROTOCOL_2026-10-20.md"
        ),
        actual_fit_matches=len(
            json.loads((out / "fit_manifest.json").read_text())["train_match_ids"]
        ),
        protected_details_read=False,
        max_uss_bytes=memory.max_uss,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def input_hash(sample):
    from dataclasses import fields

    return tensor_hash(
        (f.name, getattr(sample.inputs, f.name))
        for f in fields(sample.inputs)
        if isinstance(getattr(sample.inputs, f.name), torch.Tensor)
    )


def schedule(samples, steps, seed, mixed):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, s in enumerate(samples):
        p = s.provenance
        groups[p["stratum"]][p["match_id"]][p["team_id"]].append(i)
    rng = np.random.default_rng(seed)

    def draw(st):
        mids = sorted(groups[st])
        m = mids[int(rng.integers(len(mids)))]
        teams = sorted(groups[st][m])
        t = teams[int(rng.integers(len(teams)))]
        ids = groups[st][m][t]
        return ids[int(rng.integers(len(ids)))]

    return [
        [draw("natural") for _ in range(3 if mixed else 4)]
        + ([draw(STRATA[step % 4])] if mixed else [])
        for step in range(steps)
    ]


def metrics30(sample, delta):
    result, pred, truth, mask = evaluate_prediction(sample, delta)
    m = mask[:, :3]
    dist = (pred[:, :3] - truth[:, :3]).norm(dim=-1)
    pd, pm, _ = pair_targets(pred[:, :3], m)
    td, _, _ = pair_targets(truth[:, :3], m)
    result["ADE30"] = float(dist[m].mean()) if m.any() else None
    result["pair30"] = float((pd - td).abs()[pm].mean()) if pm.any() else None
    return result, pred, truth, mask


def train_view(data, view, seed, out, memory):
    start = time.monotonic()

    def guard():
        if time.monotonic() - start > 1200:
            raise TimeoutError("1200 s seed/view ceiling")
        if memory.over_limit or memory.errors:
            raise MemoryError("USS ceiling/measurement")

    train = data["samples"]["train"][view]["H1"]
    dev = data["samples"]["validation"][view]
    torch.manual_seed(seed)
    initial = PastWorldEncoder(data["item_count"], data["condition_sizes"])
    state = {}
    curves = {}
    rows = []
    predictions = {}
    support = {}
    gradients = {}
    for arm in ("natural", "mixed", "random"):
        guard()
        enc = copy.deepcopy(initial)
        torch.manual_seed(seed + 1)
        heads = PretrainHeads(d=64)
        if arm != "random":
            params = list(enc.parameters()) + list(heads.parameters())
            opt = adam(params)
            curve = []
            counts = Counter()
            exposure = Counter()
            for step, batch in enumerate(schedule(train, 192, seed, arm == "mixed")):
                guard()
                opt.zero_grad(set_to_none=True)
                losses = []
                for j, i in enumerate(batch):
                    loss, parts, c = pretrain_losses(enc, heads, train[i], seed + 4 * step + j)
                    counts.update(c)
                    exposure[train[i].provenance["stratum"]] += 1
                    if loss is not None:
                        losses.append(loss)
                if not losses:
                    raise RuntimeError("No active task")
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Pretrain nonfinite")
                loss.backward()
                if step == 0:
                    gradients[arm] = {
                        n: float(p.grad.norm())
                        for n, p in enc.named_parameters()
                        if n.startswith("spatial.") and p.grad is not None
                    }
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                curve.append(dict(step=step + 1, loss=float(loss.detach()), gradient=float(norm)))
                if (step + 1) % 48 == 0:
                    print(
                        json.dumps(
                            dict(seed=seed, view=view, arm=arm, stage="pretrain", **curve[-1])
                        ),
                        flush=True,
                    )
            curves[arm] = curve
            support[arm] = dict(tasks=dict(counts), exposure=dict(exposure))
        enc.eval()
        enc.requires_grad_(False)
        with torch.no_grad():
            cached = [enc(s.inputs) for s in train]
            cd = {p: [enc(s.inputs) for s in samples] for p, samples in dev.items()}
        torch.manual_seed(seed + 2)
        probe = PathProbe(d=64)
        opt = adam(probe.parameters())
        pc = []
        for step, batch in enumerate(schedule(train, 96, seed + 3, True)):
            guard()
            opt.zero_grad(set_to_none=True)
            losses = []
            for i in batch:
                y = supervised_targets(train[i])
                loss = masked_mse(
                    probe(cached[i], train[i].inputs), y["delta"], y["mask"][..., None]
                )
                if loss is not None:
                    losses.append(loss)
            if not losses:
                raise RuntimeError("No probe target")
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            pc.append(float(loss.detach()))
        curves[arm + "_probe"] = pc
        state[arm + "_encoder"] = enc.state_dict()
        state[arm + "_heads"] = heads.state_dict()
        state[arm + "_probe"] = probe.state_dict()
        with torch.no_grad():
            for policy, ss in dev.items():
                for i, (s, z) in enumerate(zip(ss, cd[policy])):
                    delta = probe(z, s.inputs)
                    metrics, pred, truth, mask = metrics30(s, delta)
                    y = supervised_targets(s)
                    o = heads(z, s.inputs)
                    modepred = o["mode"].argmax(-1)
                    for mode in (4, 5):
                        m = y["mode_mask"] & (y["mode"] == mode)
                        metrics[f"mode{mode}_correct"] = int((modepred[m] == mode).sum())
                        metrics[f"mode{mode}_n"] = int(m.sum())
                    rows.append(dict(**s.provenance, arm=arm, seed=seed, **metrics))
                    predictions[f"{arm}_{policy}_{i}"] = pred
                    if arm == "natural":
                        g = s.inputs
                        cv = (
                            g.x[-1, g.target_player_indices, None, 3:5]
                            * torch.arange(1, 7)[None, :, None]
                        )
                        cm, cp, _, _ = metrics30(s, cv)
                        rows.append(dict(**s.provenance, arm="cv", seed=seed, **cm))
                        predictions[f"cv_{policy}_{i}"] = cp
                        predictions[f"truth_{policy}_{i}"] = truth
                        predictions[f"mask_{policy}_{i}"] = mask
        print(
            json.dumps(
                dict(
                    seed=seed,
                    view=view,
                    arm=arm,
                    stage="finished",
                    elapsed_s=round(time.monotonic() - start, 1),
                )
            ),
            flush=True,
        )
    torch.save(state, out / f"{view}_checkpoint.pt")
    torch.save(predictions, out / f"{view}_predictions.pt")
    (out / f"{view}_rows.json").write_text(json.dumps(rows, indent=2))
    summary = dict(
        view=view,
        seed=seed,
        elapsed_s=time.monotonic() - start,
        state_hash=tensor_hash((f"{a}.{k}", v) for a, d in state.items() for k, v in d.items()),
        prediction_hash=tensor_hash(predictions.items()),
        curves=curves,
        support=support,
        gradient_norms=gradients,
    )
    (out / f"{view}_summary.json").write_text(json.dumps(summary, indent=2))
    return {k: v for k, v in summary.items() if k not in ("curves", "gradient_norms")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--root")
    p.add_argument("--g3")
    p.add_argument("--prepared")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261020)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    competing = [
        p.info["name"]
        for p in psutil.process_iter(["name"])
        if any(n in (p.info["name"] or "").lower() for n in ("strata", "llama"))
    ]
    if competing:
        raise RuntimeError(f"Competing processes {competing}")
    sources = [
        "world_model/" + n
        for n in (
            "temporal_prior.py",
            "encoder.py",
            "historical_prior.py",
            "inputs.py",
            "pretraining.py",
            "cohorts.py",
        )
    ]
    sources += [
        "src/pubg_lab/world_model/cli/train_world_model_g4.py",
        "docs/experiment_design/WORLD_MODEL_G4_CPU_PROTOCOL_2026-10-20.md",
    ]
    sources += ["docs/experiment_design/WORLD_MODEL_G4_CPU_PROTOCOL_V2_2026-10-20.md"]
    for rel in sources:
        dst = out / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / rel).read_bytes())
    receipt = dict(
        device="cpu",
        threads=1,
        seed=a.seed,
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        torch=torch.__version__,
        gpu_used=False,
        competing_processes=competing,
        protected_details_read=False,
        sources={r: sha(REPO / r) for r in sources},
    )
    try:
        with ResourceSampler() as memory:
            if a.prepare:
                prepare(Path(a.root), Path(a.g3), out, memory)
            else:
                gate = json.loads((Path(a.prepared) / "summary.json").read_text())
                assert gate["status"] == "PASS"
                assert gate["samples_sha256"] == sha(Path(a.prepared) / "samples.pt")
                receipt["prepared_sha256"] = gate["samples_sha256"]
                data = torch.load(
                    Path(a.prepared) / "samples.pt", weights_only=False, map_location="cpu"
                )
                receipt["views"] = {
                    v: train_view(data, v, a.seed, out, memory) for v in ("observer", "server")
                }
            receipt.update(status="PASS", max_uss_bytes=memory.max_uss)
    except Exception:
        receipt.update(status="FAIL", error=traceback.format_exc())
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
