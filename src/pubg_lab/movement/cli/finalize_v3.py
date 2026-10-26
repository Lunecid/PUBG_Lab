"""Verify delivered development artifacts without opening locked test data."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_diffusion_experiment_v3"
DATA = ROOT / "data/processed/heterograph_diffusion_v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads((OUT / "experiment_spec.json").read_text())
    rows = list(csv.DictReader((DATA / "matches.csv").open()))
    split = {
        key: {r["match_id"] for r in rows if r["split"] == key}
        for key in ["train", "validation", "test"]
    }
    assert [len(split[k]) for k in split] == [657, 188, 94]
    assert not any(
        split[a] & split[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    )
    assert not (DATA / "test_index.npy").exists()
    assert not any((DATA / "banks" / (mid + ".npz")).exists() for mid in split["test"])
    assert not any((DATA / "spatial_match" / (mid + ".npz")).exists() for mid in split["test"])
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    assert len(meta) == 1504 and {m["match_id"] for m in meta} == split["validation"]
    assert len({m["sample_id"] for m in meta}) == 1504
    for source, digest in spec["source_sha256"].items():
        assert sha(ROOT / source) == digest, source
    expected = [
        f"{v}_{k}_{seed}"
        for v in ["observer", "server"]
        for k in ["regression", "diffusion"]
        for seed in [41, 42, 43]
    ]
    expected += [f"observer_{k}_41_nozone" for k in ["regression", "diffusion"]]
    assert {p.stem for p in OUT.glob("*.pt")} == set(expected)
    for name in expected:
        ck = torch.load(OUT / (name + ".pt"), map_location="cpu", weights_only=False)
        assert ck["steps"] == 2000 and ck["source_sha256"] == spec["source_sha256"]
        assert ck["plan_sha256"] == spec["plans"][str(ck["seed"])]["sha256"]
        assert all(torch.isfinite(x).all() for x in ck["model"].values())
    with np.load(OUT / "predictions.npz") as predictions:
        assert len(predictions.files) == 14
        assert all(
            len(predictions[k]) == 1504 and np.isfinite(predictions[k]).all()
            for k in predictions.files
        )
    assert len(list((OUT / "viewer_samples").glob("*.json"))) == 1504
    transport = json.loads((OUT / "transport_audit_results.json").read_text())
    assert len(transport["records"]) == 14
    for records in transport["records"].values():
        assert len(records) == 371
        assert all(r["match_id"] in split["validation"] for r in records)
        assert len({r["sample_id"] for r in records}) == 371
    report = OUT / "REPORT_KO.md"
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
        if not target.startswith(("http:", "https:", "#")):
            assert (report.parent / target).resolve().exists(), target
    # Keep the pre-training snapshot intact. Store later presentation/QA sources separately.
    for name in [
        "report_v3.py",
        "finalize_v3.py",
        "viewer_v3.html",
        "state_figures_v3.py",
        "transport_audit_v3.py",
        "coverage_v3.py",
        "analyze_v3.py",
    ]:
        dest = OUT / "presentation_snapshot" / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "tools/movement_diffusion" / name, dest)
    checks = {
        "frozen_training_sources_match": True,
        "checkpoints_verified": 14,
        "primary_validation_windows": 1504,
        "validation_matches": 188,
        "secondary_transport_windows": 371,
        "test_banks": 0,
        "test_spatial_files": 0,
        "test_predictions": 0,
        "report_local_links_valid": True,
        "historically_pristine_test_available": False,
    }
    (OUT / "delivery_verification.json").write_text(json.dumps(checks, indent=2))
    excluded = {
        "aborted_initial_mobility_classification",
        "pre_joint_residual_fix",
        "pre_ground_transport_reset",
        "__pycache__",
    }
    manifest = {}
    for p in sorted(OUT.rglob("*")):
        rel = p.relative_to(OUT)
        if (
            p.is_file()
            and not (set(rel.parts) & excluded)
            and p.name != "artifact_manifest.json"
            and p.suffix != ".log"
        ):
            manifest[str(rel)] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**checks, "hashed_artifacts": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
