"""Frozen V3 cases plus error-independent direct-transport cases; no case cherry picking."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import json
import sys
import shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.dataset_v3 import TrajectoryDataset, collate
from pubg_lab.movement.models.model_v3 import JointMotion
from pubg_lab.movement.evaluation.evaluation import metrics

OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"
OLD = ROOT / "reports/heterograph_diffusion_experiment_v3"
COLORS = ["#0780a8", "#cf6a2a", "#884dac", "#5f912b"]


def read(p):
    return json.loads(p.read_text())


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    old = np.load(OLD / "predictions.npz")
    new = np.load(OUT / "predictions.npz")
    nt = np.load(OUT / "transport_predictions.npz")
    t = dict(np.load(OUT / "evaluation_targets.npz"))
    meta = read(OUT / "evaluation_samples.json")
    cases = []
    for case in read(OUT / "visualization_cases.json"):
        i = case["evaluation_position"]
        base = read(OLD / f"viewer_samples/{i:04d}.json")
        cases.append(
            {
                **base,
                "label": case["label"],
                "selection": "Unchanged V3 illustration; selected previously by median V3 error, not representative evidence.",
                "predictions": {
                    **{
                        "v3_" + k: old[k][i].tolist()
                        for k in old.files
                        if k.startswith(("observer", "server"))
                    },
                    **{"v4_" + k: new[k][i].tolist() for k in new.files},
                },
                "source": "primary",
                "source_position": i,
            }
        )
    ds = TrajectoryDataset(split="validation")
    oldmodels = {}
    for v in ["observer", "server"]:
        for kind in ["regression", "diffusion"]:
            for seed in [41, 42, 43]:
                name = f"{v}_{kind}_{seed}"
                m = JointMotion(
                    len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4, kind
                )
                m.load_state_dict(torch.load(OLD / f"{name}.pt", weights_only=False)["model"])
                m.eval()
                oldmodels[name] = m
    for case in read(OUT / "visual_transport_plan.json"):
        i = case["transport_position"]
        s = ds[case["index"]]
        target = s["targets"]
        base = {
            "meta": {**s["meta"], "origin_m": s["meta"]["origin_m"].tolist()},
            "label": case["group"],
            "selection": case["selection"],
            "source": "transport",
            "source_position": i,
            "presence": s["conditions"]["current_presence"].tolist(),
            "initial_xy": s["conditions"]["initial_xy"].tolist(),
            "truth": target["trajectory"].tolist(),
            "truth_valid": target["trajectory_valid"].tolist(),
            "team_valid": target["team_valid"].tolist(),
            "zones_asof": [],
            "predictions": {"v4_" + k: nt[k][i].tolist() for k in nt.files},
        }
        geo = s["conditions"]["zone_geometry"][0]
        for z in range(2):
            if geo[z * 7 + 6]:
                base["zones_asof"].append(
                    {
                        "kind": ["safe", "poison"][z],
                        "xy": (
                            s["meta"]["origin_m"]
                            + (s["conditions"]["initial_xy"][0] + geo[z * 7 : z * 7 + 2]) * 1000
                        ).tolist(),
                        "radius": float(geo[z * 7 + 2] * 1000),
                    }
                )
        for view in ["observer", "server"]:
            sample = TrajectoryDataset(split="validation", view=view)[case["index"]]
            c = collate([sample])["conditions"]
            for kind in ["regression", "diffusion"]:
                for seed in [41, 42, 43]:
                    key = f"{view}_{kind}_{seed}"
                    torch.manual_seed(seed * 100000 + 50000 + i)
                    base["predictions"]["v3_" + key] = (
                        oldmodels[key].sample(c, 20)[0].numpy().tolist()
                    )
        cases.append(base)
    image = np.asarray(Image.open(ROOT / "reports/heterograph_case_40_2_2026-10-15/map.jpg"))
    for n, case in enumerate(cases, 1):
        origin = np.array(case["meta"]["origin_m"])
        p = np.array(case["presence"])
        initial = np.array(case["initial_xy"]) * 1000 + origin
        y = np.array(case["truth"]) * 1000 + origin
        valid = np.array(case["truth_valid"])
        keys = [
            "v3_observer_regression_42",
            "v3_observer_diffusion_42",
            "v4_observer_regression_42",
            "v4_observer_diffusion_42",
            "v4_observer_independent_42",
        ]
        pts = np.concatenate(
            [initial[p], y[valid]]
            + [
                (np.array(case["predictions"][k]).mean(0) * 1000 + origin)[p].reshape(-1, 2)
                for k in keys
            ]
        )
        lo, hi = pts.min(0), pts.max(0)
        span = max((hi - lo).max() * 1.15, 500)
        center = (hi + lo) / 2
        case["bounds"] = [(center - span / 2).tolist(), (center + span / 2).tolist()]
        fig, axes = plt.subplots(
            2, 5, figsize=(20, 8), layout="constrained", gridspec_kw={"height_ratios": [2, 1]}
        )
        case["model_metrics"] = {}
        target = {
            "trajectory": np.array(case["truth"]),
            "trajectory_valid": valid,
            "team_center": np.array(case["truth"])[p].mean(0),
            "team_valid": np.array(case["team_valid"]),
        }
        for key, pr in case["predictions"].items():
            case["model_metrics"][key] = metrics(np.array(pr), target, p)
        for col, key in enumerate(keys):
            ax = axes[0, col]
            samples = np.array(case["predictions"][key]) * 1000 + origin
            mean = samples.mean(0)
            ax.imshow(image, extent=(0, 8160, 8160, 0), alpha=0.35)
            for member in np.flatnonzero(p):
                truth = np.vstack(
                    [initial[member], np.where(valid[member, :, None], y[member], np.nan)]
                )
                forecast = np.vstack([initial[member], mean[member]])
                ax.plot(*truth.T, color=COLORS[member], lw=2.4)
                ax.plot(*forecast.T, color=COLORS[member], ls="--", lw=1.8)
            if len(samples) > 1:
                for line in samples[:, 0]:
                    ax.plot(*np.vstack([initial[0], line]).T, color=COLORS[0], alpha=0.15, lw=0.7)
            ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
            ax.set_ylim(center[1] + span / 2, center[1] - span / 2)
            ax.set_aspect("equal")
            ax.set_title(key.replace("_42", "").replace("_", " "), fontsize=10)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax = axes[1, col]
            dist = []
            true = []
            for a in np.flatnonzero(p):
                for b in np.flatnonzero(p):
                    if a >= b:
                        continue
                    ok = valid[a] & valid[b]
                    dist.append(
                        np.where(
                            ok[None], np.linalg.norm(samples[:, a] - samples[:, b], axis=-1), np.nan
                        )
                    )
                    true.append(np.where(ok, np.linalg.norm(y[a] - y[b], axis=-1), np.nan))
            if dist:
                d = np.stack(dist)
                truth = np.stack(true)
                h = np.arange(10, 61, 10)
                for j in range(len(dist)):
                    ax.plot(h, truth[j], color="#222", alpha=0.6, lw=1.1)
                    ax.plot(h, d[j].mean(0), ls="--", color="#087fa9", alpha=0.65, lw=1.1)
                if len(samples) > 1:
                    band = np.nanmean(d, axis=0)
                    ax.fill_between(
                        h, *np.nanquantile(band, [0.1, 0.9], axis=0), alpha=0.15, color="#087fa9"
                    )
            ax.set_xlabel("Future time (s)")
            ax.set_ylabel("Pair distance (m)")
            ax.grid(alpha=0.2)
        upper = max(a.get_ylim()[1] for a in axes[1])
        [a.set_ylim(0, upper) for a in axes[1]]
        fig.suptitle(
            f"{case['label']} | {case['meta']['match_id'][:8]} | t={case['meta']['anchor_s']}s | fixed development illustration",
            fontsize=14,
        )
        fig.supxlabel(
            "Solid: telemetry labels; dashed: predicted mean. Faint: 20 ego scenarios. Bottom dashed: mean of scenario pair distances, not distances of the mean path. Seed42.\nBand: scenario 10-90% of mean pair distance, not confidence interval. Shared crop may omit scenarios; historical basemap is not collision geometry.",
            fontsize=9,
        )
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"validation_case_{n}.{ext}", dpi=150)
        plt.close(fig)
        case["figure"] = f"validation_case_{n}.png"
    (OUT / "viewer_cases.json").write_text(
        json.dumps(cases, separators=(",", ":"), allow_nan=False)
    )
    shutil.copyfile(ROOT / "reports/heterograph_case_40_2_2026-10-15/map.jpg", OUT / "map.jpg")
    shutil.copyfile(ROOT / "projects/movement/viewers/viewer_v4.html", OUT / "viewer.html")
    print("V4_VISUALS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
