"""All planned metrics, same-match contrasts and illustrative validation figures."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
import csv
import sys
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.dataset_v3 import TrajectoryDataset
from pubg_lab.movement.cli.analyze_experiment_v2 import match_values

OUT = ROOT / "reports/heterograph_diffusion_experiment_v3"
SEEDS = [41, 42, 43]
MODELS = [
    "constant_velocity",
    "stationary",
    "observer_regression",
    "observer_diffusion",
    "server_regression",
    "server_diffusion",
]
NAMES = [
    "Constant velocity",
    "Last position",
    "Observer regression",
    "Observer diffusion",
    "Server regression",
    "Server diffusion",
]
COLORS = ["#5c6778", "#9da6b1", "#2b9357", "#167dad", "#b89a2c", "#c2682b"]
KEYS = [
    "ego_meanpathADE_m",
    "team_meanpathADE_m",
    "joint_energy_score_m",
    "ego_expectedADE_m",
    "pair_distance_MAE_m",
]
PAIRS = [
    ("observer_diffusion", "observer_regression"),
    ("server_diffusion", "server_regression"),
    ("server_regression", "observer_regression"),
    ("server_diffusion", "observer_diffusion"),
]


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, d):
    (OUT / name).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False))


def summarize(records, mask=None):
    result = {}
    vectors = {}
    for name, replicas in records.items():
        result[name] = {}
        vectors[name] = {}
        for key in KEYS:
            values = [match_values(rs, key, mask) for rs in replicas]
            mids = sorted(set.intersection(*[set(v) for v in values]))
            if not mids:
                continue
            a = np.array([[v[mid] for mid in mids] for v in values])
            m = a.mean(0)
            rng = np.random.default_rng(231)
            boot = m[rng.integers(len(m), size=(3000, len(m)))].mean(1)
            result[name][key] = {
                "mean": float(m.mean()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "seed_means": a.mean(1).tolist(),
                "matches": len(m),
            }
            vectors[name][key] = dict(zip(mids, m))
    contrasts = {}
    for a, b in PAIRS:
        contrasts[a + "_minus_" + b] = {}
        for key in KEYS:
            if key not in vectors[a] or key not in vectors[b]:
                continue
            mids = sorted(vectors[a][key].keys() & vectors[b][key].keys())
            d = np.array([vectors[a][key][m] - vectors[b][key][m] for m in mids])
            rng = np.random.default_rng(232)
            boot = d[rng.integers(len(d), size=(5000, len(d)))].mean(1)
            r = {
                "difference_m": float(d.mean()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "matches": len(mids),
            }
            if key in KEYS[:3]:
                r["familywise95_12comparisons"] = np.quantile(
                    boot, [0.05 / 24, 1 - 0.05 / 24]
                ).tolist()
            contrasts[a + "_minus_" + b][key] = r
    return {"models": result, "contrasts": contrasts}


def plots(summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    for ax, key, title in zip(
        axes,
        KEYS[:3],
        ["Individual mean-path ADE", "Team-centroid mean-path ADE", "Joint energy score"],
    ):
        for i, model in enumerate(MODELS):
            r = summary["models"][model][key]
            lo, hi = r["ci95"]
            ax.errorbar(
                r["mean"],
                i,
                xerr=[[r["mean"] - lo], [hi - r["mean"]]],
                fmt="o",
                color=COLORS[i],
                capsize=3,
            )
            ax.text(hi + 2, i, f"{r['mean']:.1f}", va="center", fontsize=9)
        ax.set_yticks(range(len(MODELS)), NAMES)
        ax.invert_yaxis()
        ax.set_xlim(0, max(summary["models"][m][key]["ci95"][1] for m in MODELS) * 1.2 + 5)
        ax.set_title(title)
        ax.set_xlabel("meters, lower is better")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "V3 | Development matches only | 188 matches, 1,504 windows, 3 training seeds", fontsize=15
    )
    fig.supxlabel(
        "Match-level averages and bootstrap 95% intervals. Locked test not evaluated. Historical test exposure disclosed.",
        fontsize=10,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"metrics_comparison.{ext}", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, model in zip(axes.ravel(), MODELS[2:]):
        for seed in SEEDS:
            rows = [json.loads(x) for x in (OUT / f"{model}_{seed}.jsonl").read_text().splitlines()]
            ax.plot([r["step"] for r in rows], [r["loss_mean"] for r in rows], label=str(seed))
        ax.set_title(model)
        ax.set_xlabel("updates")
        ax.set_ylabel("training objective")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Training curves | objective differs between regression and diffusion")
    fig.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(fig)


def cases(meta, t, pred, records):
    # Rules locked before evaluation. Select by data condition then median diffusion
    # error within each subgroup; illustrative only, never representative evidence.
    err = np.array(
        [
            np.mean(
                [
                    r[i]["metrics"].get("joint_meanpathADE_m", 0)
                    for r in records["observer_diffusion"]
                ]
            )
            for i in range(len(meta))
        ]
    )
    common = t["team_valid"].all(1) & (t["presence"].sum(1) >= 2)
    groups = [
        ("Reentry episode", np.array([m["episode"] > 0 for m in meta])),
        (
            "Air transport / unresolved exit",
            np.array([m["mobility"] in [4, 5, 7, 8] for m in meta]),
        ),
        ("Ground vehicle", np.array([m["mobility"] == 2 for m in meta])),
        (
            "Human without vehicle evidence",
            np.array([m["actor_kind"] == 1 and m["mobility"] == 1 for m in meta]),
        ),
    ]
    selected = []
    for label, mask in groups:
        ids = np.flatnonzero(common & mask)
        if not len(ids):
            ids = np.flatnonzero(t["team_valid"].any(1) & mask)
        if len(ids):
            selected.append((label, int(ids[np.argsort(err[ids])[len(ids) // 2]])))
    image = np.asarray(Image.open(ROOT / "reports/heterograph_case_40_2_2026-10-15/map.jpg"))
    pc = ["#007ca8", "#bd5e24", "#754baa", "#63852b"]
    manifest = []
    cols = [
        "constant_velocity",
        "observer_regression_42",
        "observer_diffusion_42",
        "server_regression_42",
        "server_diffusion_42",
    ]
    for n, (label, i) in enumerate(selected, 1):
        origin = np.array(meta[i]["origin_m"])
        presence = t["presence"][i]
        initial = t["initial_xy"][i] * 1000 + origin
        y = t["trajectory"][i] * 1000 + origin
        valid = t["trajectory_valid"][i]
        pts = np.concatenate(
            [initial[presence], y[valid]]
            + [(pred[c][i].mean(0) * 1000 + origin)[presence].reshape(-1, 2) for c in cols]
        )
        lo = pts.min(0)
        hi = pts.max(0)
        span = max((hi - lo).max() * 1.15, 500)
        center = (hi + lo) / 2
        fig, axes = plt.subplots(
            2, 5, figsize=(20, 8), gridspec_kw={"height_ratios": [2, 1]}, layout="constrained"
        )
        for col, key in enumerate(cols):
            ax = axes[0, col]
            ax.imshow(image, extent=(0, 8160, 8160, 0), alpha=0.38)
            samples = pred[key][i] * 1000 + origin
            mean = samples.mean(0)
            for member in np.flatnonzero(presence):
                truth = np.vstack(
                    [initial[member], np.where(valid[member, :, None], y[member], np.nan)]
                )
                forecast = np.vstack([initial[member], mean[member]])
                ax.plot(*truth.T, color=pc[member], lw=2.4)
                ax.plot(*forecast.T, color=pc[member], ls="--", lw=1.6)
            for line in samples[:, 0]:
                if len(samples) > 1:
                    ax.plot(*np.vstack([initial[0], line]).T, color=pc[0], alpha=0.12, lw=0.6)
            ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
            ax.set_ylim(center[1] + span / 2, center[1] - span / 2)
            ax.set_title(key.replace("_42", "").replace("_", "\n"), fontsize=11)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal")
            ax = axes[1, col]
            base = initial[presence].mean(0)
            true = np.linalg.norm(t["team_center"][i] * 1000 + origin - base, axis=-1)
            d = np.linalg.norm(samples[:, presence].mean(1) - base, axis=-1)
            h = np.arange(10, 61, 10)
            ax.plot(
                h,
                np.where(t["team_valid"][i], true, np.nan),
                "-o",
                color="#222",
                label="True center",
            )
            ax.plot(h, d.mean(0), "--o", color="#147eaa", label="Mean scenario distance")
            ax.fill_between(h, *np.quantile(d, [0.1, 0.9], axis=0), color="#147eaa", alpha=0.15)
            ax.set_xlabel("Horizon (s)")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.legend(fontsize=8)
        upper = max(ax.get_ylim()[1] for ax in axes[1])
        [ax.set_ylim(0, upper) for ax in axes[1]]
        fig.suptitle(
            f"{label} | {meta[i]['match_id'][:8]} | t={meta[i]['anchor_s']}s | episode={meta[i]['episode']} | current mode={meta[i]['mobility']}"
        )
        fig.supxlabel(
            "Solid: true telemetry / label-only interpolation. Dashed: mean prediction. Faint: 20 ego scenarios. Seed42. Historical basemap is not collision geometry.\nShared map scales; crop may omit scenarios. Band: scenario 10–90 percentiles, not confidence interval. Mode7 is unresolved after aircraft exit. Median-error selection.",
            fontsize=10,
        )
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"validation_case_{n}.{ext}", dpi=140)
        plt.close(fig)
        manifest.append(
            {
                "label": label,
                "evaluation_position": i,
                "sample_id": meta[i]["sample_id"],
                "figure": f"validation_case_{n}.png",
                "selection": "median observer diffusion joint meanpath ADE in data-defined subgroup; prefer complete multiactor windows",
            }
        )
    write("visualization_cases.json", manifest)
    return manifest


def main():
    baseline = read("evaluation_baselines.json")["records"]
    records = {
        "constant_velocity": [baseline],
        "stationary": [[{**r, "metrics": r["stationary_metrics"]} for r in baseline]],
    }
    for model in MODELS[2:]:
        records[model] = [read(f"evaluation_{model}_{seed}.json")["records"] for seed in SEEDS]
    meta = read("evaluation_samples.json")
    ids = [r["sample_id"] for r in baseline]
    assert len(ids) == 1504 and all(
        [r["sample_id"] for r in replica] == ids for rs in records.values() for replica in rs
    )
    overall = summarize(records)
    groups = {}
    for name, mask in {
        "human": np.array([m["actor_kind"] == 1 for m in meta]),
        "bot": np.array([m["actor_kind"] == 2 for m in meta]),
        "reentry": np.array([m["episode"] > 0 for m in meta]),
        "first_life": np.array([m["episode"] == 0 for m in meta]),
        "aerial": np.array([m["mobility"] in [4, 5, 7, 8] for m in meta]),
        "land_vehicle": np.array([m["mobility"] == 2 for m in meta]),
        "nonvehicle_candidate": np.array([m["mobility"] == 1 for m in meta]),
    }.items():
        groups[name] = {"windows": int(mask.sum()), **summarize(records, mask)}
    write("comparison_summary.json", {"overall": overall, "subgroups": groups})
    plots(overall)
    t = dict(np.load(OUT / "evaluation_targets.npz"))
    pred = dict(np.load(OUT / "predictions.npz"))
    manifest = cases(meta, t, pred, records)
    folder = OUT / "viewer_samples"
    folder.mkdir(exist_ok=True)
    ds = TrajectoryDataset(split="validation")
    mode_names = json.loads((ds.root / "summary.json").read_text())["modes"]
    for i, m in enumerate(meta):
        sample = ds[m["index"]]
        mi, k, ego, *_ = ds.index[m["index"]]
        b = ds.bank(m["match_id"])
        zones = []
        for z, start in enumerate([3, 6]):
            if b["game_mask"][k, start : start + 3].all() and b["game"][k, start + 2] > 0:
                zones.append(
                    {
                        "kind": ["safe", "poison"][z],
                        "xy": b["game"][k, start : start + 2].tolist(),
                        "radius": float(b["game"][k, start + 2]),
                    }
                )
        obj = {
            "meta": m,
            "mode_name": mode_names[m["mobility"]],
            "zones_asof": zones,
            "presence": t["presence"][i].tolist(),
            "initial_xy": t["initial_xy"][i].tolist(),
            "truth": t["trajectory"][i].tolist(),
            "truth_valid": t["trajectory_valid"][i].tolist(),
            "team_valid": t["team_valid"][i].tolist(),
            "time_error_s": t["time_error_s"][i].tolist(),
            "predictions": {key: np.round(value[i], 6).tolist() for key, value in pred.items()},
            "metrics": {
                model: {str(seed): records[model][j][i]["metrics"] for j, seed in enumerate(SEEDS)}
                for model in MODELS[2:]
            },
            "baseline_metrics": {key: records[key][0][i]["metrics"] for key in MODELS[:2]},
        }
        (folder / f"{i:04d}.json").write_text(json.dumps(obj, separators=(",", ":")))
    write("viewer_index.json", {"samples": meta, "cases": manifest})
    shutil.copyfile(ROOT / "reports/heterograph_case_40_2_2026-10-15/map.jpg", OUT / "map.jpg")
    shutil.copyfile(ROOT / "tools/movement_diffusion/viewer_v3.html", OUT / "viewer.html")
    with (OUT / "metrics_table.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *KEYS])
        writer.writerows([[m, *[overall["models"][m][k]["mean"] for k in KEYS]] for m in MODELS])
    print(
        json.dumps(
            {m: {k: overall["models"][m][k]["mean"] for k in KEYS} for m in MODELS}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
