"""Post-evaluation residual mean drift; no refitting or changed primary estimates."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"


def main():
    x = np.load(OUT / "predictions.npz")
    t = np.load(OUT / "evaluation_targets.npz")
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    groups = {
        "all": np.ones(len(meta), bool),
        "nonvehicle": np.array([m["mobility"] == 1 for m in meta]),
        "land_vehicle": np.array([m["mobility"] == 2 for m in meta]),
        "post_reentry": np.array([m["episode"] > 0 for m in meta]),
    }
    result = {}
    for view in ["observer", "server"]:
        for seed in [41, 42, 43]:
            key = f"{view}_diffusion_{seed}"
            pr = x[key]
            head = x[f"{view}_regression_{seed}"][:, 0]
            mean = pr.mean(1)
            variance = np.square(pr - mean[:, None]).sum(-1).mean(1)
            drift = np.square(mean - head).sum(-1)
            valid = t["trajectory_valid"]
            result[key] = {}
            for group, ids in groups.items():
                mask = valid & ids[:, None, None]
                result[key][group] = {
                    "head_to_ensemble_RMS_m": float(np.sqrt(drift[mask].mean()) * 1000),
                    "estimated_MC20_RMS_m": float(np.sqrt((variance / 19)[mask].mean()) * 1000),
                    "estimated_squared_systematic_shift_m2": float(
                        (drift - variance / 19)[mask].mean() * 1e6
                    ),
                }
    (OUT / "v4_residual_drift_diagnosis.json").write_text(
        json.dumps(
            {
                "method": "Post-evaluation diagnosis, no retraining. Valid-point-weighted squared mean drift minus unbiased sample variance/K; noisy descriptive estimate, not causal proof.",
                "models": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
