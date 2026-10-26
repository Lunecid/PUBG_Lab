#!/usr/bin/env python3
"""Reproducible source and numerical fidelity checks against the pinned archive."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO / "third_party/krafton_pubg_winprob"))
import torch
from src.models.backbone import TransformerBackbone


def main():
    torch.set_num_threads(2)
    root = REPO / "third_party/krafton_pubg_winprob"
    archive = REPO / "reports/krafton_pubg_winprob_audit_2026-10-02/upstream_99ed9d1.zip"
    out = REPO / "reports/sac2027_patch_v14/baseline_audit"
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert sha == "7cdfc8fade1e38538f264ab0f096de65a76420dfeab43ecb2798916522df7b30"
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0]
        rows = []
        for name in z.namelist():
            rel = name[len(prefix) :]
            if not rel or name.endswith("/"):
                continue
            before = z.read(name)
            after = (root / rel).read_bytes()
            rows.append(
                dict(
                    path=rel,
                    unchanged=before == after,
                    upstream_sha256=hashlib.sha256(before).hexdigest(),
                    adapted_sha256=hashlib.sha256(after).hexdigest(),
                )
            )
        # Load the archived class; its imported modules are independently
        # confirmed byte-identical above. Restore upstream's four-map constant.
        namespace = {"__name__": "audit_upstream_backbone"}
        exec(
            compile(z.read(prefix + "src/models/backbone.py"), "pinned_upstream_backbone", "exec"),
            namespace,
        )
        namespace["NUM_MAPS"] = 4
    expected = {
        "src/data/dataset.py",
        "src/data/dataset_v2.py",
        "src/models/backbone.py",
        "src/training/calibration.py",
        "src/training/evaluation.py",
        "src/training/inference.py",
        "src/training/inference_with_json.py",
        "src/training/metrics.py",
        "src/training/trainer.py",
    }
    assert {r["path"] for r in rows if not r["unchanged"]} == expected
    torch.manual_seed(42)
    kw = dict(input_dim=57, embed_dim=16, num_heads=4, num_layers=2, dropout=0.0)
    original = namespace["TransformerBackbone"](**kw).eval()
    adapted = TransformerBackbone(**kw, num_maps=4).eval()
    adapted.load_state_dict(original.state_dict())
    x = torch.randn(2, 19, 57)
    pos = torch.rand(2, 19, 4, 3)
    pos[:, :, -1] = float("nan")
    blue = torch.tensor([[0.5, 0.5, 0.4], [0.4, 0.3, 0.2]])
    white = blue / 2
    maps = torch.tensor([0, 3])
    valid = torch.ones(2, 19, dtype=torch.bool)
    a = original(x, pos, blue, white, maps)
    b = adapted(x, pos, blue, white, maps, valid_mask=valid)
    assert torch.allclose(a, b, atol=1e-6)
    weights = torch.randn_like(a)
    (a * weights).sum().backward()
    (b * weights).sum().backward()
    gradient_delta = max(
        float((dict(original.named_parameters())[n].grad - p.grad).abs().max())
        for n, p in adapted.named_parameters()
        if p.grad is not None
    )
    assert gradient_delta < 1e-5
    with torch.no_grad():
        small = adapted(x[:, :3], pos[:, :3], blue, white, maps, valid_mask=valid[:, :3])
        keys = valid.clone()
        keys[:, 3:] = False
        padded = adapted(x, pos, blue, white, maps, valid_mask=keys)[:, :3]
        mutant = x.clone()
        mutant[:, 3:] = 1e4
        changed = adapted(mutant, pos, blue, white, maps, valid_mask=keys)[:, :3]
    assert torch.allclose(small, padded, atol=1e-6)
    assert torch.allclose(padded, changed, atol=1e-6)
    result = dict(
        upstream_commit="99ed9d1fa5f1df0cb242bb42ed8a511e6c972896",
        archive_sha256=sha,
        files=rows,
        unchanged_files=sum(r["unchanged"] for r in rows),
        changed_files=len(expected),
        forward_max_abs_delta=float((a - b).detach().abs().max()),
        gradient_max_abs_delta=gradient_delta,
        padding_max_abs_delta=float((small - padded).abs().max()),
        padding_mutation_max_abs_delta=float((padded - changed).abs().max()),
        scope="Source and synthetic functional fidelity, not reproduction of original empirical performance",
    )
    (out / "krafton_source_and_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
