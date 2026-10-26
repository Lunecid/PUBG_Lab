"""Verify saved checkpoints, paired sample coverage, and exact inference replay."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.dataset import TrajectoryDataset, collate
from pubg_lab.movement.model import JointGraphDiffusion, constant_velocity

OUT = ROOT / "reports/heterograph_diffusion_experiment_v2"


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    spec = json.loads((OUT / "experiment_spec.json").read_text())
    checkpoint_checks = []
    current_hash = hashlib.sha256(
        (ROOT / "src/pubg_lab/movement/model.py").read_bytes()
    ).hexdigest()
    pred = dict(np.load(OUT / "predictions.npz"))
    targets = dict(np.load(OUT / "evaluation_targets.npz"))
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    indices = np.load(OUT / "evaluation_indices.npy")
    assert len(meta) == 1504 and len(set(x["sample_id"] for x in meta)) == 1504
    for view in ["observer", "server"]:
        d = TrajectoryDataset(split="validation", view=view)
        sample = d[int(indices[0])]
        batch = collate([sample])
        assert sample["meta"]["sample_id"] == meta[0]["sample_id"]
        for key, value in sample["targets"].items():
            np.testing.assert_array_equal(value, targets[key][0])
        for seed in [41, 42, 43]:
            path = OUT / f"{view}_seed{seed}.pt"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            plan = np.load(OUT / f"training_indices_seed{seed}.npy")
            assert (
                ck["step"] == 2500
                and ck["parameters"] == 204044
                and ck["seed"] == seed
                and ck["view"] == view
            )
            assert (
                ck["plan_sha256"]
                == hashlib.sha256(plan.tobytes()).hexdigest()
                == spec["plans"][str(seed)]["sha256"]
            )
            assert ck["model_code_sha256"] == current_hash
            log = [
                json.loads(x) for x in (OUT / f"{view}_seed{seed}.jsonl").read_text().splitlines()
            ]
            assert log == ck["training_log"] and log[-1]["step"] == 2500
            model = JointGraphDiffusion(
                len(d.builder.vocab["items"]) + len(d.builder.vocab["vehicles"]) + 4
            )
            model.load_state_dict(ck["model"])
            model.eval()
            torch.manual_seed(seed * 100000)
            a = model.sample(batch["conditions"], 20)[0].numpy()
            np.testing.assert_array_equal(a, pred[f"{view}_{seed}"][0])
            assert (
                pred[f"{view}_{seed}"].shape == (1504, 20, 4, 6, 2)
                and np.isfinite(pred[f"{view}_{seed}"]).all()
            )
            checkpoint_checks.append(
                {
                    "view": view,
                    "seed": seed,
                    "steps": ck["step"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "training_plan_matches": True,
                    "training_log_matches_checkpoint": True,
                    "first_validation_sample_bitwise_reproduced": True,
                    "elapsed_training_s": ck["elapsed_s"],
                }
            )
        np.testing.assert_array_equal(
            constant_velocity(batch["conditions"])[0].numpy(), pred["constant_velocity"][0, 0]
        )
    stats = {}
    for seed in [41, 42, 43]:
        a = pred[f"observer_{seed}"]
        b = pred[f"server_{seed}"]
        stats[str(seed)] = {
            "mean_absolute_coordinate_difference_m": float(np.abs(a - b).mean() * 1000),
            "max_absolute_coordinate_difference_m": float(np.abs(a - b).max() * 1000),
        }
        assert not np.array_equal(a, b)
    all_records = []
    for view in ["observer", "server"]:
        for seed in [41, 42, 43]:
            r = json.loads((OUT / f"evaluation_{view}_{seed}.json").read_text())["records"]
            assert [x["sample_id"] for x in r] == [x["sample_id"] for x in meta]
            all_records.extend(r)
    checks = {
        "completed_models": len(checkpoint_checks),
        "steps_per_model": 2500,
        "total_optimizer_updates": 15000,
        "total_training_window_presentations": 120000,
        "evaluation_matches": len(set(x["match_id"] for x in meta)),
        "evaluation_windows": len(meta),
        "generated_joint_scenarios": 1504 * 20 * 6,
        "checkpoint_checks": checkpoint_checks,
        "paired_prediction_differences": stats,
        "all_6x1504_record_ID_sequences_equal": True,
        "full_training_rerun_performed": False,
        "inference_replay": "First validation sample, all six checkpoints, exact float32 match",
    }
    (OUT / "final_verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    manifest = {}
    for path in sorted(OUT.rglob("*")):
        if (
            path.is_file()
            and path.name not in ["artifact_manifest.json"]
            and path.suffix not in [".log"]
        ):
            manifest[str(path.relative_to(OUT))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "files": manifest,
                "note": "Run again after any artifact edits to refresh checksums. Training code was independently frozen before validation in experiment_spec.json.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
