"""Real paired batches: optimizer steps, generation, masks, and runtime tensor shapes."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.dataset import TrajectoryDataset, collate
from pubg_lab.movement.model import JointGraphDiffusion


def main():
    torch.set_num_threads(2)
    out = {}
    target_hash = []
    for view in ("observer", "server"):
        ds = TrajectoryDataset(view=view)
        indices = [len(ds) // 3, 2 * len(ds) // 3]
        samples = [ds[i] for i in indices]
        batch = collate(samples)
        torch.manual_seed(20261015)
        categories = len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4
        model = JointGraphDiffusion(categories)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logs = []
        started = time.time()
        for step in range(3):
            opt.zero_grad()
            loss, parts = model.loss(batch["conditions"], batch["targets"])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            opt.step()
            logs.append({"step": step, "loss": float(loss.detach()), **parts})
        model.eval()
        torch.manual_seed(123)
        generated = model.sample(batch["conditions"], scenarios=2)
        assert generated.shape == (2, 2, 4, 6, 2) and torch.isfinite(generated).all()
        out[view] = {
            "sample_ids": [s["meta"]["sample_id"] for s in samples],
            "condition_shapes": {
                k: list(v.shape) for k, v in batch["conditions"].items() if torch.is_tensor(v)
            },
            "target_shapes": {k: list(v.shape) for k, v in batch["targets"].items()},
            "optimizer_steps": logs,
            "generated_shape": list(generated.shape),
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": time.time() - started,
        }
        torch.save(model.state_dict(), ROOT / "reports/heterograph_dataset_v1" / f"smoke_{view}.pt")
        target_hash.append(batch["targets"]["trajectory"])
    assert torch.equal(*target_hash)
    out["interpretation"] = (
        "Three optimizer steps on two paired real samples: integration smoke test only, no performance comparison."
    )
    out["paired_targets_identical"] = True
    (ROOT / "reports/heterograph_dataset_v1/smoke.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
