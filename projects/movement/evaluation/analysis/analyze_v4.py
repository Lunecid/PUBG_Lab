"""All fixed V4 comparisons; descriptive match bootstrap, frozen development cases."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.evaluation.evaluation import match_macro

OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"
OLD = ROOT / "reports/heterograph_diffusion_experiment_v3"
KEYS = [
    "ego_meanpathADE_m",
    "team_meanpathADE_m",
    "pair_distance_MAE_m",
    "joint_energy_score_m",
    "ego_expectedADE_m",
    "ego_minADE_m",
]


def read(p):
    return json.loads(p.read_text())


def write(n, d):
    (OUT / n).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False))


def vectors(replicas, key, mask=None):
    result = []
    for rs in replicas:
        groups = defaultdict(list)
        for i, r in enumerate(rs):
            if (mask is None or mask[i]) and key in r["metrics"]:
                groups[r["match_id"]].append(r["metrics"][key])
        result.append({m: float(np.mean(v)) for m, v in groups.items()})
    mids = sorted(set.intersection(*[set(r) for r in result]))
    arr = np.array([[r[m] for m in mids] for r in result])
    return mids, arr


def summarize(records, mask=None):
    result = {}
    for name, replicas in records.items():
        result[name] = {}
        for key in KEYS:
            mids, a = vectors(replicas, key, mask)
            if not mids:
                continue
            mean = a.mean(0)
            rng = np.random.default_rng(20261021)
            boot = mean[rng.integers(len(mean), size=(3000, len(mean)))].mean(1)
            result[name][key] = {
                "mean": float(mean.mean()),
                "matches": len(mids),
                "seed_means": a.mean(1).tolist(),
                "descriptive_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
    return result


def contrast(a, b):
    result = {}
    for key in KEYS:
        ma, va = vectors(a, key)
        mb, vb = vectors(b, key)
        aa = dict(zip(ma, va.mean(0)))
        bb = dict(zip(mb, vb.mean(0)))
        mids = sorted(aa.keys() & bb.keys())
        d = np.array([aa[m] - bb[m] for m in mids])
        rng = np.random.default_rng(20261022)
        boot = d[rng.integers(len(d), size=(5000, len(d)))].mean(1)
        result[key] = {
            "difference_m": float(d.mean()),
            "descriptive_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "matches": len(d),
        }
    return result


def load_records():
    result = {}
    for version, root in [("v3", OLD), ("v4", OUT)]:
        for view in ["observer", "server"]:
            for kind in ["regression", "diffusion"] + (
                ["independent"] if version == "v4" and view == "observer" else []
            ):
                result[f"{version}_{view}_{kind}"] = [
                    read(root / f"evaluation_{view}_{kind}_{seed}.json")["records"]
                    for seed in [41, 42, 43]
                ]
    base = read(OLD / "evaluation_baselines.json")["records"]
    result["constant_velocity"] = [base]
    result["stationary"] = [[{**r, "metrics": r["stationary_metrics"]} for r in base]]
    return result


def dispersion(pred, t, meta):
    out = {}
    for name in pred.files:
        if "diffusion" not in name and "independent" not in name:
            continue
        x = pred[name]
        p = t["presence"]
        valid = t["trajectory_valid"]
        center = (x * p[:, None, :, None, None]).sum(2) / p.sum(1)[:, None, None, None]
        rel = x - center[:, :, None]
        records = []
        for i, md in enumerate(meta):
            vals = {}
            ok = t["team_valid"][i]
            if ok.any():
                vals["center_variance_m2"] = float(center[i][:, ok].var(0).sum(-1).mean() * 1e6)
                vals["relative_variance_m2"] = float(
                    rel[i][:, p[i]][:, :, ok].var(0).sum(-1).mean() * 1e6
                )
            lo, hi = np.quantile(x[i], [0.1, 0.9], axis=0)
            covered = (t["trajectory"][i] >= lo) & (t["trajectory"][i] <= hi)
            vals["coordinate_10_90_coverage"] = float(covered[valid[i]].mean())
            vals["coordinate_10_90_width_m"] = float((hi - lo)[valid[i]].mean() * 1000)
            records.append({"match_id": md["match_id"], "metrics": vals})
        out[name] = match_macro(records)
    return out


def plots(summary):
    names = [
        "v3_observer_regression",
        "v3_observer_diffusion",
        "v4_observer_regression",
        "v4_observer_diffusion",
        "v4_observer_independent",
        "v4_server_regression",
        "v4_server_diffusion",
    ]
    labels = [
        "V3 observer regression",
        "V3 observer diffusion",
        "V4 observer regression",
        "V4 observer diffusion",
        "V4 independent residual",
        "V4 server regression",
        "V4 server diffusion",
    ]
    colors = ["#81958a", "#7f9ba7", "#26965c", "#087fa9", "#9e829e", "#b29b32", "#cf703b"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6), layout="constrained")
    for ax, key, title in zip(
        axes,
        KEYS[:4],
        [
            "Individual mean-path ADE",
            "Team-center mean-path ADE",
            "Pair-distance MAE (scenarios)",
            "Joint energy score",
        ],
    ):
        for i, (name, color) in enumerate(zip(names, colors)):
            d = summary[name][key]
            lo, hi = d["descriptive_ci95"]
            ax.errorbar(
                d["mean"],
                i,
                xerr=[[d["mean"] - lo], [hi - d["mean"]]],
                fmt="o",
                color=color,
                capsize=3,
            )
            ax.annotate(
                f"{d['mean']:.1f}",
                (d["mean"], i),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_yticks(range(len(names)), labels if ax == axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("meters; lower is better")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("V4 | Identical 188 development matches / 1,504 windows / 3 seeds", fontsize=15)
    fig.supxlabel(
        "Descriptive match-bootstrap 95% intervals; repeated development set. Diffusion uses K=20 independent joint scenarios. Test remains locked.",
        fontsize=9,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"metrics_comparison.{ext}", dpi=160)
    plt.close(fig)


def main():
    records = load_records()
    meta = read(OUT / "evaluation_samples.json")
    ids = [r["sample_id"] for r in meta]
    for rs in records.values():
        for replica in rs:
            assert [r["sample_id"] for r in replica] == ids
    overall = summarize(records)
    groups = {}
    for group, mask in {
        "human": np.array([m["actor_kind"] == 1 for m in meta]),
        "bot": np.array([m["actor_kind"] == 2 for m in meta]),
        "post_reentry": np.array([m["episode"] > 0 for m in meta]),
        "nonvehicle_candidate": np.array([m["mobility"] == 1 for m in meta]),
        "land_vehicle": np.array([m["mobility"] == 2 for m in meta]),
    }.items():
        groups[group] = {"windows": int(mask.sum()), "models": summarize(records, mask)}
    contrasts = {}
    for view in ["observer", "server"]:
        for a, b in [
            (f"v4_{view}_diffusion", f"v3_{view}_diffusion"),
            (f"v4_{view}_regression", f"v3_{view}_regression"),
            (f"v4_{view}_diffusion", f"v4_{view}_regression"),
        ]:
            contrasts[a + "_minus_" + b] = contrast(records[a], records[b])
    contrasts["structured_minus_independent"] = contrast(
        records["v4_observer_diffusion"], records["v4_observer_independent"]
    )
    uniform = [read(OUT / "evaluation_observer_regression_41_uniform.json")["records"]]
    balanced = [records["v4_observer_regression"][0]]
    sampling = {
        "uniform": summarize({"uniform": uniform})["uniform"],
        "balanced": summarize({"balanced": balanced})["balanced"],
        "balanced_minus_uniform": contrast(balanced, uniform),
        "status": "One-seed exploratory control; same V4 regression architecture/loss/updates, sampling differs.",
    }
    transport = {}
    oldtr = read(OLD / "transport_audit_results.json")
    newtr = read(OUT / "transport_audit_results.json")
    for group in ["emergency_pickup", "redeploy_aircraft"]:
        rs = {}
        for version, doc in [("v3", oldtr), ("v4", newtr)]:
            for view in ["observer", "server"]:
                for kind in ["regression", "diffusion"] + (
                    ["independent"] if version == "v4" and view == "observer" else []
                ):
                    rs[f"{version}_{view}_{kind}"] = [
                        [r for r in doc["records"][f"{view}_{kind}_{seed}"] if r["group"] == group]
                        for seed in [41, 42, 43]
                    ]
        transport[group] = {"models": summarize(rs), "counts": newtr["counts"][group]}
        ua = [
            [r for r in newtr["records"]["observer_regression_41_uniform"] if r["group"] == group]
        ]
        ba = [rs["v4_observer_regression"][0]]
        transport[group]["sampling_control"] = {
            "uniform": summarize({"u": ua})["u"],
            "balanced": summarize({"b": ba})["b"],
            "difference": contrast(ba, ua),
        }
        transport[group]["v4_minus_v3_observer_diffusion"] = contrast(
            rs["v4_observer_diffusion"], rs["v3_observer_diffusion"]
        )
    doc = {
        "overall": overall,
        "contrasts": contrasts,
        "subgroups": groups,
        "sampling_control": sampling,
        "transport": transport,
        "inference": "Exploratory repeated-development comparisons, fixed3seeds. Match-bootstrap intervals exclude seed/participant uncertainty and are not multiplicity-adjusted confirmatory tests.",
    }
    write("comparison_summary.json", doc)
    plots(overall)
    with (OUT / "metrics_table.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["model", *KEYS])
        w.writerows([[m, *[overall[m][k]["mean"] for k in KEYS]] for m in overall])
    t = dict(np.load(OUT / "evaluation_targets.npz"))
    write(
        "generation_diagnostics.json",
        {
            "v3": dispersion(np.load(OLD / "predictions.npz"), t, meta),
            "v4": dispersion(np.load(OUT / "predictions.npz"), t, meta),
            "note": "Coordinatewise 10-90% empirical bands are not joint trajectory coverage, confidence intervals, or guaranteed calibrated uncertainty. Lower dispersion may indicate collapse; assess energy and oracle coverage together.",
        },
    )
    print(
        json.dumps({m: {k: overall[m][k]["mean"] for k in KEYS[:4]} for m in overall}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
