"""Read-only result diagnostics and standalone scientific figures."""

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
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pubg_lab.world_model.encoder import WorldEncoder
from pubg_lab.world_model.pretraining import PretrainHeads, pretrain_losses, supervised_targets
from pubg_lab.world_model.cli.verify_world_model_encoder import tensor_hash, ResourceSampler, sha


def kinds(sample):
    # NumPy indexing with a length-one torch tensor serialized a scalar in
    # prepared v1 metadata. Evaluation-only metadata; normalize without rewriting.
    return np.atleast_1d(sample.provenance["kind_eval_only"]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    data = torch.load(root / "prepared/samples.pt", weights_only=False)
    summaries = {
        v: json.loads((root / f"cpu_run_01/{v}_summary.json").read_text())
        for v in ("observer", "server")
    }
    repeated = {
        v: json.loads((root / f"cpu_run_02/{v}_summary.json").read_text()) for v in summaries
    }
    reproduction = {
        v: {k: summaries[v][k] == repeated[v][k] for k in ("state_hash", "prediction_hash")}
        for v in summaries
    }
    assert all(all(row.values()) for row in reproduction.values())
    diagnostics = {
        "reproduction": reproduction,
        "cohort": {},
        "task_diagnostics": {},
        "paired": {},
        "subgroups": {},
    }
    for role in ("train", "validation"):
        ss = data["samples"][role]["server"]
        mids = {s.provenance["match_id"] for s in ss}
        diagnostics["cohort"][role] = {
            "matches": len(mids),
            "anchors": len(ss),
            "map_matches": dict(Counter(data["metadata"][m]["map"] for m in mids)),
            "perspective_matches": dict(Counter(data["metadata"][m]["perspective"] for m in mids)),
            "team_size_matches": dict(
                Counter(data["metadata"][m]["nominal_team_size"] for m in mids)
            ),
            "kind_player_anchors": dict(Counter(k for s in ss for k in kinds(s))),
            "current_mode_player_anchors": dict(
                Counter(
                    int(k)
                    for s in ss
                    for k in s.inputs.mobility[-1, s.inputs.target_player_indices]
                )
            ),
            "target_mode_points": dict(
                Counter(
                    int(k)
                    for s in ss
                    for k in supervised_targets(s)["mode"][supervised_targets(s)["mode_mask"]]
                )
            ),
            "observed_edges_total": sum(len(s.inputs.observed) for s in ss),
            "inferred_edges_total": sum(len(s.inputs.inferred) for s in ss),
            "complete60_anchors": sum(s.targets["joint_complete_60s"] for s in ss),
        }
    for view in summaries:
        # Additional post-run TRAIN-only diagnostic: first up to 8 supported
        # examples per task, selected by label availability, never by loss value.
        torch.manual_seed(20261020)
        initial = WorldEncoder(data["item_count"], data["condition_sizes"])
        head_initial = PretrainHeads()
        assert tensor_hash(initial.state_dict().items()) == summaries[view]["encoder_before_hash"]
        final = WorldEncoder(data["item_count"], data["condition_sizes"])
        head_final = PretrainHeads()
        checkpoint = torch.load(root / f"cpu_run_01/{view}_checkpoint.pt", weights_only=True)
        final.load_state_dict(checkpoint["encoder"])
        head_final.load_state_dict(checkpoint["heads"])
        collected = defaultdict(list)
        grad = {}
        selected = defaultdict(list)
        for i, sample in enumerate(data["samples"]["train"][view]):
            _, before, counts = pretrain_losses(initial, head_initial, sample, 20261020 + i)
            with torch.no_grad():
                _, after, _ = pretrain_losses(final, head_final, sample, 20261020 + i)
            for task, loss in before.items():
                if loss is not None and len(collected[task]) < 8:
                    collected[task].append((float(loss.detach()), float(after[task])))
                    selected[task].append(i)
                    if task not in grad:
                        gradients = torch.autograd.grad(
                            loss,
                            list(initial.parameters()) + list(head_initial.parameters()),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        grad[task] = float(
                            torch.sqrt(sum(g.square().sum() for g in gradients if g is not None))
                        )
            if len(collected) == 4 and all(len(v) == 8 for v in collected.values()):
                break
        diagnostics["task_diagnostics"][view] = {
            "posthoc_train_only": True,
            "selection_rule": "first 8 label-supported examples per task",
            "sample_indices": dict(selected),
            "initial_unweighted_gradient_norms": grad,
            "mean_before_after": {k: np.mean(v, axis=0).tolist() for k, v in collected.items()},
        }
        diagnostics["paired"][view] = {}
        for metric in ("ADE", "FDE", "pair_MAE"):
            a = summaries[view]["metrics"]["pretrained"][metric]["per_match"]
            b = summaries[view]["metrics"]["random"][metric]["per_match"]
            differences = {m: a[m] - b[m] for m in sorted(a.keys() & b.keys())}
            diagnostics["paired"][view][metric] = {
                "per_match": differences,
                "mean": float(np.mean(list(differences.values()))),
                "improved": sum(v < 0 for v in differences.values()),
                "n": len(differences),
            }
        predictions = torch.load(root / f"cpu_run_01/{view}_predictions.pt", weights_only=True)
        subgroup = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, s in enumerate(data["samples"]["validation"][view]):
            for arm in ("pretrained", "random", "cv"):
                error = (predictions[f"{arm}_{i}"] - predictions[f"truth_{i}"]).norm(dim=-1)
                mask = predictions[f"mask_{i}"]
                for actor, kind in enumerate(kinds(s)):
                    if mask[actor].any():
                        subgroup[arm][str(kind)][s.provenance["match_id"]].append(
                            float(error[actor][mask[actor]].mean())
                        )
        diagnostics["subgroups"][view] = {
            arm: {
                kind: {
                    "match_macro_actor_ADE": float(np.mean([np.mean(x) for x in groups.values()])),
                    "n_matches": len(groups),
                    "actor_anchors": sum(map(len, groups.values())),
                }
                for kind, groups in kinds.items()
            }
            for arm, kinds in subgroup.items()
        }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180}
    )
    colors = {"pretrained": "#236B8E", "random": "#A9B2BB", "cv": "#D08A35"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7))
    for row, view in enumerate(summaries):
        for col, metric in enumerate(("ADE", "FDE", "pair_MAE")):
            ax = axes[row, col]
            vals = [summaries[view]["metrics"][arm][metric]["mean"] for arm in colors]
            ax.bar(range(3), vals, color=list(colors.values()))
            ax.set_xticks(range(3), ["Pretrained\nfrozen", "Random\nfrozen", "Constant\nvelocity"])
            ax.set_ylim(0, max(vals) * 1.25)
            for i, val in enumerate(vals):
                ax.text(i, val + max(vals) * 0.03, f"{val:.2f}", ha="center")
            n = summaries[view]["metrics"]["pretrained"][metric]["n_matches"]
            ax.set_title(f"{view.title()} · {metric} · {n} matches")
            ax.set_ylabel("Error (m), lower is better")
    fig.suptitle(
        "Patch 40.2 → 41.1 development | fixed 256 + 128 updates, one seed\nMatch macro averages; this is a frozen representation probe, not Diffusion",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "development_metrics.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric in zip(axes, ("ADE", "FDE", "pair_MAE")):
        for j, view in enumerate(summaries):
            values = list(diagnostics["paired"][view][metric]["per_match"].values())
            ax.scatter(
                np.linspace(j - 0.15, j + 0.15, len(values)),
                values,
                color=("#236B8E" if j == 0 else "#8D5E9A"),
                alpha=0.8,
            )
            ax.scatter([j], [np.mean(values)], marker="_", s=500, color="black")
        ax.axhline(0, color="#777777", ls="--", lw=1)
        ax.set_xticks([0, 1], ["Observer", "Server"])
        ax.set_title(metric)
        ax.set_ylabel("Pretrained − random error (m)")
    fig.suptitle("Each dot is one development match; negative favors pretraining")
    fig.tight_layout()
    fig.savefig(out / "paired_match_differences.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for row, view in enumerate(summaries):
        curve = summaries[view]["curve"]
        for ax, task in zip(axes[row], ("health", "motion", "mode", "pair")):
            points = [(r["step"], r[task]) for r in curve if task in r]
            x, y = np.array(points).T
            ax.plot(x, y, alpha=0.25, lw=0.7, color="#236B8E")
            smooth = [np.median(y[max(0, i - 15) : i + 1]) for i in range(len(y))]
            ax.plot(x, smooth, lw=1.6, color="#236B8E")
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title(f"{view.title()} · {task}")
            ax.set_xlabel("Optimizer update")
            ax.set_ylabel("Unweighted task loss (symlog)")
    fig.suptitle(
        "Training losses | thin: sampled batches; thick: trailing 16 observed-task median\nSupports differ by task. Gaussian density NLL may be negative."
    )
    fig.tight_layout()
    fig.savefig(out / "training_losses.png")
    plt.close(fig)

    # Illustration selected only by CURRENT membership, not prediction error.
    ss = data["samples"]["validation"]["observer"]
    chosen = next((i for i, s in enumerate(ss) if len(s.inputs.target_player_indices) >= 2), 0)
    s = ss[chosen]
    g = s.inputs
    predictions = torch.load(root / "cpu_run_01/observer_predictions.pt", weights_only=True)
    truth, mask = predictions[f"truth_{chosen}"], predictions[f"mask_{chosen}"]
    current = g.x[-1, g.target_player_indices, :2] * 100
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, arm in zip(axes, colors):
        pred = predictions[f"{arm}_{chosen}"]
        for p in range(len(current)):
            color = plt.get_cmap("tab10")(p % 10)
            gt = torch.cat([current[p : p + 1], truth[p]]).numpy()
            valid = np.r_[True, mask[p].numpy()]
            gt[~valid] = np.nan
            pp = torch.cat([current[p : p + 1], pred[p]]).numpy()
            ax.plot(gt[:, 0], gt[:, 1], "-o", color=color, ms=3, lw=2, label=f"Player {p + 1}")
            ax.plot(pp[:, 0], pp[:, 1], "--", color=color, lw=1.4)
            ax.scatter(*current[p], marker="s", color=color, s=25)
        ax.set_title(arm.title())
        ax.set_xlabel("Relative map X (m)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Relative map Y (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Observer paths: first multi-member development anchor | t={s.provenance['t0']:.0f}s\nSolid: observed future; dashed: prediction; square: current position; gaps: missing labels"
    )
    fig.tight_layout()
    fig.savefig(out / "example_team_paths.png")
    plt.close(fig)
    (out / "example_provenance.json").write_text(
        json.dumps({"sample_index": chosen, **s.provenance}, indent=2)
    )
    print(
        json.dumps(
            {
                "reproduction": reproduction,
                "cohort": diagnostics["cohort"],
                "task_diagnostics": diagnostics["task_diagnostics"],
                "example_index": chosen,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    with ResourceSampler():
        main()
