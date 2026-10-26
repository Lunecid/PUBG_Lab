"""Diagnose frozen V3 predictions, without test access or fitting."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.dataset_v3 import TrajectoryDataset, collate
from pubg_lab.movement.model_v3 import JointMotion
from pubg_lab.movement.evaluation import metrics, match_macro

OLD = ROOT / "reports/heterograph_diffusion_experiment_v3"
OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    meta = json.loads((OLD / "evaluation_samples.json").read_text())
    t = dict(np.load(OLD / "evaluation_targets.npz"))
    pred = np.load(OLD / "predictions.npz")
    out = {}
    means = {}
    for view in ["observer", "server"]:
        ds = TrajectoryDataset(split="validation", view=view)
        models = {}
        for seed in [41, 42, 43]:
            name = f"{view}_diffusion_{seed}"
            m = JointMotion(
                len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4, "diffusion"
            )
            m.load_state_dict(torch.load(OLD / f"{name}.pt", weights_only=False)["model"])
            m.eval()
            models[name] = m
            means[name] = []
        with torch.no_grad():
            for i, md in enumerate(meta):
                c = collate([ds[md["index"]]])["conditions"]
                for name, m in models.items():
                    means[name].append(m.mean(m.encode(c), c)[0].numpy())
                if (i + 1) % 500 == 0:
                    print("MEAN_HEAD", view, i + 1, flush=True)
    for name, values in means.items():
        mu = np.stack(values)
        x = pred[name]
        y = t["trajectory"]
        v = t["trajectory_valid"] & t["presence"][:, :, None]
        ensemble = x.mean(1)
        d = x - ensemble[:, None]
        w = t["presence"]
        center = (d * w[:, None, :, None, None]).sum(2) / w.sum(1)[:, None, None, None]
        rel = d - center[:, :, None]
        records = []
        head = []
        avg = []
        for i, md in enumerate(meta):
            a = (
                (np.square(x[i] - y[i]) * v[i][None, ..., None]).sum((1, 2, 3)).mean()
                / v[i].sum()
                * 1e6
            )
            bias = (np.square(ensemble[i] - y[i]) * v[i][..., None]).sum() / v[i].sum() * 1e6
            variance = (
                (np.square(d[i]) * v[i][None, ..., None]).sum((1, 2, 3)).mean() / v[i].sum() * 1e6
            )
            assert abs(a - bias - variance) < max(1.0, a * 2e-5)
            common = t["team_valid"][i]
            vals = {
                "expected_squared_error_m2": float(a),
                "ensemble_squared_error_m2": float(bias),
                "sample_variance_m2": float(variance),
                "mean_head_to_ensemble_m": float(
                    np.linalg.norm(mu[i] - ensemble[i], axis=-1)[v[i]].mean() * 1000
                ),
            }
            if common.any():
                cc = np.square(center[i][:, common]).sum(-1).mean()
                rr = np.square(rel[i][:, w[i]][:, :, common]).sum(-1).mean()
                vals.update(
                    center_variance_m2=float(cc * 1e6),
                    relative_variance_m2=float(rr * 1e6),
                    relative_variance_share=float(rr / (cc + rr)),
                )
            records.append({"match_id": md["match_id"], "metrics": vals})
            target = {k: z[i] for k, z in t.items() if k not in ["presence", "initial_xy"]}
            head.append({"match_id": md["match_id"], "metrics": metrics(mu[i][None], target, w[i])})
            avg.append(
                {"match_id": md["match_id"], "metrics": metrics(ensemble[i][None], target, w[i])}
            )
        out[name] = {
            "decomposition": match_macro(records),
            "mean_head_metrics": match_macro(head),
            "ensemble_as_single_path_metrics": match_macro(avg),
        }
    np.savez_compressed(OUT / "v3_mean_heads.npz", **{k: np.stack(v) for k, v in means.items()})
    (OUT / "v3_error_diagnosis.json").write_text(
        json.dumps(
            {
                "models": out,
                "interpretation": "Descriptive algebraic decomposition; MSE identity is not ADE identity. Shared vs relative variance uses only all-current-member-valid horizons. Causes from architecture and exposure remain hypotheses without controlled retraining.",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                k: {
                    "head_ADE": r["mean_head_metrics"]["ego_meanpathADE_m"]["mean_over_matches"],
                    "ensemble_pair": r["ensemble_as_single_path_metrics"]["pair_distance_MAE_m"][
                        "mean_over_matches"
                    ],
                    **{a: z["mean_over_matches"] for a, z in r["decomposition"].items()},
                }
                for k, r in out.items()
            },
            indent=2,
        ),
        flush=True,
    )


def monte_carlo_diagnosis():
    p = np.load(OLD / "predictions.npz")
    h = np.load(OUT / "v3_mean_heads.npz")
    t = np.load(OLD / "evaluation_targets.npz")
    valid = t["trajectory_valid"]
    result = {}
    for key in h.files:
        x = p[key]
        mu = x.mean(1)
        variance = np.square(x - mu[:, None]).sum(-1).mean(1)
        drift = np.square(mu - h[key]).sum(-1)
        result[key] = {
            "head_to_ensemble_RMS_m": float(np.sqrt(drift[valid].mean()) * 1000),
            "estimated_MC20_RMS_m": float(np.sqrt((variance / 19)[valid].mean()) * 1000),
            "estimated_squared_systematic_shift_m2": float(
                (drift - variance / 19)[valid].mean() * 1e6
            ),
        }
    (OUT / "v3_monte_carlo_diagnosis.json").write_text(
        json.dumps(
            {
                "method": "Valid-point-weighted squared drift minus unbiased within-ensemble variance/K. Descriptive unbiased noisy estimate under iid scenarios, not proof of systematic bias. Negative estimates possible. Distinct from match-macro ADE.",
                "models": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
    monte_carlo_diagnosis()
