"""Pre-prediction, mode-stratified secondary evaluation; no tuning or test access."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.dataset_v3 import TrajectoryDataset, collate, DEFAULT
from pubg_lab.movement.models.model_v3 import JointMotion, constant_velocity
from pubg_lab.movement.evaluation.evaluation import metrics, match_macro

OUT = ROOT / "reports/heterograph_diffusion_experiment_v3"
GROUPS = {4: "emergency_pickup", 5: "redeploy_aircraft"}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def prepare():
    assert not (OUT / "predictions.npz").exists(), "Plan must precede validation predictions"
    index = np.load(DEFAULT / "validation_index.npy")
    rows = {int(r["match_index"]): r for r in csv.DictReader((DEFAULT / "matches.csv").open())}
    rng = np.random.default_rng(20261019)
    plan = []
    for mi in np.unique(index[:, 0]):
        r = rows[int(mi)]
        assert r["split"] == "validation"
        eligible = np.flatnonzero((index[:, 0] == mi) & (index[:, 4] > 0))
        with np.load(DEFAULT / "banks" / (r["match_id"] + ".npz")) as b:
            modes = b["mobility"][index[eligible, 1], index[eligible, 2]]
        for mode, group in GROUPS.items():
            ids = eligible[modes == mode]
            if len(ids):
                for i in sorted(rng.choice(ids, min(2, len(ids)), replace=False).tolist()):
                    plan.append(
                        {"index": i, "match_id": r["match_id"], "mode": mode, "group": group}
                    )
    doc = {
        "before_validation_predictions": True,
        "reason": "Primary random development windows contain only 8 pickup and 1 redeploy-aircraft anchors. Add a separate bounded diagnostic, not a change to primary comparison.",
        "sampling": "Uniform random up to2 windows per development match per directly recorded mode, seed20261019; eligibility requires at least one future ego label. Match-macro within mode; no prevalence-weighted pooled claim.",
        "status": "Exploratory secondary audit; frozen weights, no tuning; confidence intervals not confirmatory after selecting rare strata.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan": plan,
        "counts": {
            g: {
                "windows": sum(x["group"] == g for x in plan),
                "matches": len({x["match_id"] for x in plan if x["group"] == g}),
            }
            for g in GROUPS.values()
        },
    }
    write("transport_audit_plan.json", doc)
    print(json.dumps(doc["counts"], indent=2))


def evaluate():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((OUT / "transport_audit_plan.json").read_text())
    assert protocol["source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = json.loads((OUT / "experiment_spec.json").read_text())
    records = {"constant_velocity": [], "stationary": []}
    for view in ["observer", "server"]:
        ds = TrajectoryDataset(split="validation", view=view)
        models = {}
        for kind in ["regression", "diffusion"]:
            for seed in [41, 42, 43]:
                key = f"{view}_{kind}_{seed}"
                ck = torch.load(OUT / (key + ".pt"), weights_only=False, map_location="cpu")
                assert ck["steps"] == 2000 and ck["source_sha256"] == spec["source_sha256"]
                model = JointMotion(
                    len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4, kind
                )
                model.load_state_dict(ck["model"])
                model.eval()
                models[key] = (seed, model)
                records[key] = []
        for number, item in enumerate(protocol["plan"]):
            s = ds[item["index"]]
            assert (
                s["meta"]["match_id"] == item["match_id"] and s["meta"]["mobility"] == item["mode"]
            )
            c = collate([s])["conditions"]
            p = s["conditions"]["current_presence"]
            base = {**item, "sample_id": s["meta"]["sample_id"]}
            if view == "observer":
                cv = constant_velocity(c)[0].numpy()[None]
                stationary = np.repeat(s["conditions"]["initial_xy"][:, None], 6, axis=1)[None]
                for key, pred in [("constant_velocity", cv), ("stationary", stationary)]:
                    records[key].append({**base, "metrics": metrics(pred, s["targets"], p)})
            for key, (seed, model) in models.items():
                torch.manual_seed(seed * 100000 + 50000 + number)
                pred = model.sample(c, 20)[0].numpy()
                assert np.isfinite(pred).all()
                records[key].append({**base, "metrics": metrics(pred, s["targets"], p)})
            if (number + 1) % 50 == 0:
                print("TRANSPORT", view, number + 1, len(protocol["plan"]), flush=True)
    summary = {}
    for group in GROUPS.values():
        summary[group] = {"counts": protocol["counts"][group], "models": {}}
        for model in [
            "constant_velocity",
            "stationary",
            "observer_regression",
            "observer_diffusion",
            "server_regression",
            "server_diffusion",
        ]:
            keys = (
                [model]
                if model in ["constant_velocity", "stationary"]
                else [f"{model}_{seed}" for seed in [41, 42, 43]]
            )
            values = [match_macro([r for r in records[k] if r["group"] == group]) for k in keys]
            summary[group]["models"][model] = {
                metric: {
                    "mean": float(np.mean([v[metric]["mean_over_matches"] for v in values])),
                    "seed_means": [v[metric]["mean_over_matches"] for v in values],
                    "matches": values[0][metric]["matches"],
                }
                for metric in values[0]
            }
    write(
        "transport_audit_results.json",
        {
            "protocol": {k: v for k, v in protocol.items() if k != "plan"},
            "summary": summary,
            "records": records,
        },
    )
    print("TRANSPORT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "evaluate"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else evaluate()
