"""Verify model lineage, frozen plans, exact inference replay, and locked tests."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import csv
import hashlib
import json
import platform
import sys
import shutil
from pathlib import Path
import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.cli.experiment_v4 import OUT, OLD, verify, load_model
from pubg_lab.movement.dataset_v3 import TrajectoryDataset, collate, DEFAULT


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    spec = verify()
    pred = np.load(OUT / "predictions.npz")
    checks = []
    expected = (
        [
            f"{v}_{k}_{s}"
            for v in ["observer", "server"]
            for k in ["regression", "diffusion"]
            for s in [41, 42, 43]
        ]
        + [f"observer_independent_{s}" for s in [41, 42, 43]]
        + ["observer_regression_41_uniform"]
    )
    assert set(pred.files) == set(expected)
    assert {p.stem for p in OUT.glob("*.pt") if not p.name.startswith("train_features_")} == set(
        expected
    )
    index = int(np.load(OUT / "evaluation_indices.npy")[0])
    shapes = {}
    for seed, digest in spec["plans"].items():
        assert sha(OUT / f"training_indices_seed{seed}.npy") == digest
    assert (
        sha(OUT / "evaluation_indices.npy")
        == spec["evaluation_sha256"]
        == sha(OLD / "evaluation_indices.npy")
    )
    assert (
        sha(OUT / "transport_audit_plan.json")
        == spec["transport_plan_sha256"]
        == sha(OLD / "transport_audit_plan.json")
    )
    for name in expected:
        ck = torch.load(OUT / (name + ".pt"), weights_only=False)
        assert ck["source_sha256"] == spec["sources"]
        assert ck["steps"] == (2000 if ck["kind"] == "regression" else 4000)
        assert all(torch.isfinite(x).all() for x in ck["model"].values())
        view, kind, seed = name.split("_")[:3]
        seed = int(seed)
        uniform = name.endswith("_uniform")
        ds = TrajectoryDataset(split="validation", view=view)
        model = load_model(ds, view, kind, seed, uniform)
        batch = collate([ds[index]])
        c = batch["conditions"]
        torch.manual_seed(seed * 100000)
        replay = model.sample(c, 20)[0].numpy()
        np.testing.assert_array_equal(replay, pred[name][0])
        assert pred[name].shape == (1504, 1 if kind == "regression" else 20, 4, 6, 2)
        assert np.isfinite(pred[name]).all()
        plan_path = (OLD if uniform else OUT) / f"training_indices_seed{seed}.npy"
        expected_plan = (
            hashlib.sha256(np.load(plan_path).tobytes()).hexdigest()
            if kind == "regression"
            else sha(plan_path)
        )
        assert ck["plan_sha256"] == expected_plan
        if kind != "regression":
            original = torch.load(OUT / f"{view}_regression_{seed}.pt", weights_only=False)["model"]
            assert ck["regressor_sha256"] == sha(OUT / f"{view}_regression_{seed}.pt")
            for k, v in original.items():
                torch.testing.assert_close(ck["model"]["regressor." + k], v, rtol=0, atol=0)
            cache = torch.load(OUT / f"train_features_{view}_{seed}.pt", weights_only=False)
            assert cache["plan_sha256"] == sha(OUT / f"training_indices_seed{seed}.npy")
            assert len(cache["context"]["h"]) == 16000
        checks.append(
            {
                "name": name,
                "steps": ck["steps"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "exact_first_validation_prediction_replay": True,
                "checkpoint_sha256": sha(OUT / (name + ".pt")),
            }
        )
        if not shapes:
            shapes = {
                "conditions": {
                    k: list(v.shape) for k, v in c.items() if isinstance(v, torch.Tensor)
                },
                "targets": {k: list(v.shape) for k, v in batch["targets"].items()},
                "output": list(model.sample(c, 20).shape),
            }
    trans = json.loads((OUT / "transport_audit_results.json").read_text())
    plan = json.loads((OUT / "transport_audit_plan.json").read_text())["plan"]
    for rs in trans["records"].values():
        assert len(rs) == 371 and [(r["index"], r["match_id"]) for r in rs] == [
            (r["index"], r["match_id"]) for r in plan
        ]
    rows = list(csv.DictReader((DEFAULT / "matches.csv").open()))
    test = {r["match_id"] for r in rows if r["split"] == "test"}
    assert len(test) == 94
    assert not any((DEFAULT / "banks" / (m + ".npz")).exists() for m in test)
    assert not (DEFAULT / "test_index.npy").exists()
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    assert len({r["match_id"] for r in meta}) == 188 and not test & {r["match_id"] for r in meta}
    doc = {
        "checkpoints": checks,
        "source_and_plan_hashes_verified": True,
        "regression_backbones_identical_in_diffusion": True,
        "cache_features_fit_only_and_linked_to_regressor": True,
        "test_predictions": 0,
        "test_matches_locked": 94,
        "historically_pristine_test": False,
        "primary_windows": 1504,
        "validation_matches": 188,
        "secondary_transport_windows": 371,
        "full_training_rerun": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
        },
        "runtime_shapes_B1": shapes,
    }
    (OUT / "delivery_verification.json").write_text(json.dumps(doc, indent=2))
    print("V4_VERIFIED", len(checks), flush=True)
    for name in [
        "analyze_v4.py",
        "visualize_v4.py",
        "viewer_v4.html",
        "finalize_v4.py",
        "audit_v4.py",
        "diagnose_v4.py",
        "report_v4.py",
        "horizons_v4.py",
        "finish_v4.py",
        "residual_drift_v4.py",
    ]:
        p = ROOT / "tools/movement_diffusion" / name
        if p.exists():
            dest = OUT / "presentation_snapshot" / name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, dest)
    tests = OUT / "presentation_snapshot/tests"
    tests.mkdir(exist_ok=True)
    for p in (ROOT / "tests/movement_diffusion").glob("*.py"):
        shutil.copyfile(p, tests / p.name)
    assert "32 passed" in (OUT / "tests_final.log").read_text()
    manifest = {
        str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(OUT.rglob("*"))
        if p.is_file()
        and p.name != "artifact_manifest.json"
        and p.suffix != ".log"
        and "__pycache__" not in p.parts
    }
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
