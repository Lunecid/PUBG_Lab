"""Matched horizon diagnostics, retaining actual per-horizon availability masks."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"
OLD = ROOT / "reports/heterograph_diffusion_experiment_v3"


def main():
    t = np.load(OUT / "evaluation_targets.npz")
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    y = t["trajectory"] * 1000
    valid = t["trajectory_valid"]
    p = t["presence"]
    result = {}
    for version, root in [("v3", OLD), ("v4", OUT)]:
        pred = np.load(root / "predictions.npz")
        for kind in ["regression", "diffusion"]:
            allseed = []
            for seed in [41, 42, 43]:
                x = pred[f"observer_{kind}_{seed}"] * 1000
                mean = x.mean(1)
                series = []
                for f in range(6):
                    groups = defaultdict(lambda: defaultdict(list))
                    for i, md in enumerate(meta):
                        vals = {}
                        if valid[i, 0, f]:
                            vals["ego_ADE"] = float(np.linalg.norm(mean[i, 0, f] - y[i, 0, f]))
                        if t["team_valid"][i, f]:
                            vals["team_ADE"] = float(
                                np.linalg.norm(mean[i, p[i], f].mean(0) - y[i, p[i], f].mean(0))
                            )
                        pair = []
                        for a in range(4):
                            for b in range(a + 1, 4):
                                if p[i, a] and p[i, b] and valid[i, a, f] and valid[i, b, f]:
                                    pair.extend(
                                        abs(
                                            np.linalg.norm(x[i, :, a, f] - x[i, :, b, f], axis=-1)
                                            - np.linalg.norm(y[i, a, f] - y[i, b, f])
                                        ).tolist()
                                    )
                        if pair:
                            vals["pair_MAE"] = float(np.mean(pair))
                        for k, v in vals.items():
                            groups[k][md["match_id"]].append(v)
                    series.append(
                        {
                            k: float(np.mean([np.mean(v) for v in rows.values()]))
                            for k, rows in groups.items()
                        }
                    )
                allseed.append(series)
            result[f"{version}_{kind}"] = {
                k: [float(np.mean([s[f][k] for s in allseed])) for f in range(6)]
                for k in ["ego_ADE", "team_ADE", "pair_MAE"]
            }
    (OUT / "horizon_diagnostics.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, k, title in zip(
        axes,
        ["ego_ADE", "team_ADE", "pair_MAE"],
        ["Individual mean path", "Team center mean path", "Scenario pair distance"],
    ):
        for name, values in result.items():
            ax.plot(range(10, 61, 10), values[k], "-o", label=name)
        ax.set_title(title)
        ax.set_xlabel("Future horizon (s)")
        ax.set_ylabel("Error (m)")
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Observer model | Same development windows | Match means, 3 seeds")
    fig.supxlabel(
        "Availability masks retained; later horizons can have fewer observed targets.", fontsize=9
    )
    fig.savefig(OUT / "horizon_diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
