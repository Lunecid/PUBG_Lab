"""Saved-run diagnostics only; never fit or select checkpoints from development."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def match_means(rows, metric):
    group = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get(metric) is not None:
            group[r["match_id"]][r["team_id"]].append(r[metric])
    return {m: float(np.mean([np.mean(v) for v in teams.values()])) for m, teams in group.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, out = Path(a.run), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [20261020, 20261021, 20261022]
    rows = []
    summaries = {}
    receipts = []
    for seed in seeds:
        folder = root / f"seed_{seed}_02"
        receipt = json.loads((folder / "receipt.json").read_text())
        assert receipt["status"] == "PASS"
        receipts.append(receipt)
        for view in ("observer", "server"):
            rows += json.loads((folder / f"{view}_rows.json").read_text())
            summaries[seed, view] = json.loads((folder / f"{view}_summary.json").read_text())

    def select(view, policy, arm, stratum, seed):
        return [
            r
            for r in rows
            if r["view"] == view
            and r["spatial_prior_policy"] == policy
            and r["arm"] == arm
            and r["seed"] == seed
            and (r["stratum"] == stratum if stratum != "rare" else r["stratum"] != "natural")
        ]

    aggregate = []
    for view in ("observer", "server"):
        for policy in ("H0", "H1"):
            for arm in ("natural", "mixed", "random", "cv"):
                for st in (
                    "natural",
                    "rare",
                    "emergency30",
                    "reentry30",
                    "emergency_air",
                    "reentry_air",
                ):
                    for metric in ("ADE30", "pair30", "ADE", "FDE", "pair_MAE"):
                        per_seed = []
                        for seed in seeds:
                            mm = match_means(select(view, policy, arm, st, seed), metric)
                            per_seed.append(
                                dict(
                                    seed=seed,
                                    mean=float(np.mean(list(mm.values()))) if mm else None,
                                    n_matches=len(mm),
                                )
                            )
                        valid = [r["mean"] for r in per_seed if r["mean"] is not None]
                        aggregate.append(
                            dict(
                                view=view,
                                policy=policy,
                                arm=arm,
                                stratum=st,
                                metric=metric,
                                mean=float(np.mean(valid)) if valid else None,
                                per_seed=per_seed,
                            )
                        )

    def paired(view, policy, st, metric, arm0="natural", arm1="mixed", policy0=None):
        differences = []
        mids = None
        for seed in seeds:
            a = match_means(select(view, policy0 or policy, arm0, st, seed), metric)
            b = match_means(select(view, policy, arm1, st, seed), metric)
            common = set(a) & set(b)
            mids = common if mids is None else mids & common
            differences.append({m: b[m] - a[m] for m in common})
        mids = sorted(mids)
        if not mids:
            return None
        matrix = np.array([[d[m] for m in mids] for d in differences])
        rng = np.random.default_rng(20261020)
        indices = rng.integers(0, len(mids), size=(2000, len(mids)))
        boot = matrix[:, indices].mean((0, 2))
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        return dict(
            mean=float(matrix.mean()),
            ci95=ci,
            seed_differences=matrix.mean(1).tolist(),
            n_matches=len(mids),
            all_seeds_negative=bool((matrix.mean(1) < 0).all()),
            improvement=bool(ci[1] < 0 and (matrix.mean(1) < 0).all()),
        )

    primary = paired("observer", "H1", "natural", "ADE30")
    diagnostics = dict(
        primary=primary,
        aggregates=aggregate,
        paired_auxiliary={},
        mode_support={},
        exposure={},
        reproduction={},
        max_process_uss_bytes=max(r["max_uss_bytes"] for r in receipts),
        protected_details_read=False,
    )
    for view in ("observer", "server"):
        for st in ("natural", "rare"):
            for metric in ("ADE30", "pair30"):
                diagnostics["paired_auxiliary"][f"{view}_{st}_{metric}_mixed_minus_natural"] = (
                    paired(view, "H1", st, metric)
                )
                diagnostics["paired_auxiliary"][f"{view}_{st}_{metric}_H1_minus_H0"] = paired(
                    view, "H1", st, metric, "mixed", "mixed", "H0"
                )
        for seed in seeds:
            diagnostics["exposure"][f"{view}_{seed}"] = summaries[seed, view]["support"]
        for arm in ("natural", "mixed"):
            ss = [
                r
                for r in rows
                if r["view"] == view and r["spatial_prior_policy"] == "H1" and r["arm"] == arm
            ]
            diagnostics["mode_support"][f"{view}_{arm}"] = {
                str(mode): dict(
                    correct=sum(r[f"mode{mode}_correct"] for r in ss),
                    count=sum(r[f"mode{mode}_n"] for r in ss),
                )
                for mode in (4, 5)
            }
        rep = root / "seed_20261020_repeat_02"
        if rep.exists() and (rep / "receipt.json").exists():
            rs = json.loads((rep / f"{view}_summary.json").read_text())
            diagnostics["reproduction"][view] = {
                k: rs[k] == summaries[20261020, view][k] for k in ("state_hash", "prediction_hash")
            }
            assert all(diagnostics["reproduction"][view].values())
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180}
    )
    colors = {"natural": "#2475a5", "mixed": "#ce7326", "random": "#848c93", "cv": "#41915c"}

    def lookup(view, policy, arm, st, metric):
        return next(
            r
            for r in aggregate
            if (r["view"], r["policy"], r["arm"], r["stratum"], r["metric"])
            == (view, policy, arm, st, metric)
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for i, st in enumerate(("natural", "rare")):
        for j, metric in enumerate(("ADE30", "pair30")):
            ax = axes[i, j]
            for k, arm in enumerate(("natural", "mixed", "random", "cv")):
                vals = [lookup(v, "H1", arm, st, metric)["mean"] for v in ("observer", "server")]
                bars = ax.bar(
                    np.arange(2) + (k - 1.5) * 0.18, vals, 0.18, color=colors[arm], label=arm
                )
                ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
            ax.set(
                xticks=[0, 1],
                xticklabels=["Observer", "Server"],
                ylabel="Metres",
                title=f"{st.capitalize()} development / {metric}",
            )
            ax.margins(y=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Three-seed averages; same frozen-probe training mixture (75:25)", fontsize=13)
    fig.savefig(out / "comparison.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, view in zip(axes, ("observer", "server")):
        for k, arm in enumerate(("natural", "mixed")):
            vals = [lookup(view, p, arm, "natural", "ADE30")["mean"] for p in ("H0", "H1")]
            ax.plot([0, 1], vals, "o-", label=arm, color=colors[arm])
            ax.set(
                xticks=[0, 1],
                xticklabels=["H0 exact patch", "H1 past patches"],
                title=view,
                ylabel="Natural development ADE30 (m)",
            )
        ax.legend(frameon=False)
        ax.set_ylim(60, 68)
    fig.suptitle("Same checkpoints: availability of past-patch evidence")
    fig.savefig(out / "prior_effect.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, view in zip(axes, ("observer", "server")):
        for arm in ("natural", "mixed"):
            for seed in seeds:
                curve = summaries[seed, view]["curves"][arm]
                yy = np.array([r["loss"] for r in curve])
                ax.plot(
                    np.arange(16, len(yy) + 1),
                    np.convolve(yy, np.ones(16) / 16, "valid"),
                    color=colors[arm],
                    alpha=0.55,
                    label=arm if seed == seeds[0] else None,
                )
        ax.set(title=view, xlabel="Optimizer steps", ylabel="16-step mean weighted pretrain loss")
        ax.legend(frameon=False)
    fig.suptitle("Different sampling distributions: curves are optimization diagnostics")
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)
    data = torch.load(root / "prepared_01/samples.pt", weights_only=False, map_location="cpu")
    ss = data["samples"]["validation"]["observer"]["H1"]
    pred = torch.load(
        root / "seed_20261020_02/observer_predictions.pt", weights_only=False, map_location="cpu"
    )
    indices = [
        next(i for i, s in enumerate(ss) if s.provenance["stratum"] == st)
        for st in ("natural", "emergency_air", "reentry_air")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, i in zip(axes, indices):
        s = ss[i]
        truth = pred[f"truth_H1_{i}"].numpy()
        mask = pred[f"mask_H1_{i}"].numpy()
        start = s.inputs.x[-1, s.inputs.target_player_indices, :2].numpy() * 100
        for arm in ("natural", "mixed", "cv"):
            pp = pred[f"{arm}_H1_{i}"].numpy()
            for j in range(len(pp)):
                visible = np.where(mask[j, :, None], pp[j], np.nan)
                ax.plot(
                    np.r_[start[j, 0], visible[:, 0]],
                    np.r_[start[j, 1], visible[:, 1]],
                    color=colors[arm],
                    ls="--",
                    alpha=0.7,
                    label=arm if j == 0 else None,
                )
        for j in range(len(truth)):
            visible = np.where(mask[j, :, None], truth[j], np.nan)
            ax.plot(
                np.r_[start[j, 0], visible[:, 0]],
                np.r_[start[j, 1], visible[:, 1]],
                "k.-",
                lw=1.7,
                label="truth" if j == 0 else None,
            )
        ax.scatter(start[:, 0], start[:, 1], marker="x", color="black")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(title=s.provenance["stratum"], xlabel="Relative x (m)", ylabel="Relative y (m)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed first examples / observer / seed 20261020 / only valid horizons")
    fig.savefig(out / "trajectories.png")
    plt.close(fig)
    (out / "figure_selection.json").write_text(
        json.dumps([dict(index=i, **ss[i].provenance) for i in indices], indent=2)
    )
    print(
        json.dumps(
            {
                "primary": primary,
                "reproduction": diagnostics["reproduction"],
                "metrics": [
                    r
                    for r in aggregate
                    if r["view"] == "observer"
                    and r["policy"] == "H1"
                    and r["metric"] in ("ADE30", "pair30")
                    and r["stratum"] in ("natural", "rare")
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
