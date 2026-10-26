"""Runtime shape / gradient / seed reproducibility probe on training samples only."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.dataset import TrajectoryDataset, collate, NODE_TYPES, RELATIONS
from pubg_lab.movement.models.model import JointGraphDiffusion

OUT = ROOT / "reports/heterograph_diffusion_experiment_v2"


def main():
    torch.set_num_threads(1)
    ds = TrajectoryDataset()
    ck = torch.load(OUT / "observer_seed41.pt", map_location="cpu", weights_only=False)
    m = JointGraphDiffusion(len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4)
    m.load_state_dict(ck["model"])
    b = collate([ds[len(ds) // 3], ds[2 * len(ds) // 3]])
    c = b["conditions"]
    loss, _ = m.loss(c, b["targets"])
    loss.backward()
    grads = {
        typ: float(layer.weight.grad.norm()) if layer.weight.grad is not None else None
        for typ, layer in zip(NODE_TYPES, m.projections)
    }
    relgrad = {
        typ: float(m.relation.weight.grad[i].norm())
        for i, typ in enumerate(RELATIONS)
        if (c["edge_type"] == i).any()
    }
    m.eval()
    torch.manual_seed(123)
    a = m.sample(c, 20)
    torch.manual_seed(123)
    repeat = m.sample(c, 20)
    assert torch.equal(a, repeat)
    assert torch.isfinite(a).all()
    result = {
        "training_samples_only": [s["sample_id"] for s in b["meta"]],
        "model_parameters": sum(p.numel() for p in m.parameters()),
        "conditions": {k: list(v.shape) for k, v in c.items() if hasattr(v, "shape")},
        "encoded_members": list(m.encode(c).shape),
        "targets": {k: list(v.shape) for k, v in b["targets"].items()},
        "joint_samples": list(a.shape),
        "terminal_alpha_bar": float(m.alpha_bar[-1]),
        "type_projection_gradient_norms": grads,
        "present_relation_embedding_gradient_norms": relgrad,
        "fixed_seed_bitwise_repeatable": True,
        "finite_samples": True,
        "meaning": "Gradient availability is wiring evidence, not evidence of predictive value for that feature type.",
    }
    (OUT / "runtime_model_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
