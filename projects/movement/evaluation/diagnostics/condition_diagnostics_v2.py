"""Post-evaluation sensitivity diagnostic. NOT a retrained feature ablation.
Same random noise and CV reference; change the learned condition or zone values.
"""

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
from pubg_lab.movement.data.dataset import TrajectoryDataset, collate, NODE_TYPES
from pubg_lab.movement.models.model import JointGraphDiffusion

OUT = ROOT / "reports/heterograph_diffusion_experiment_v2"


def main():
    torch.set_num_threads(2)
    ids = np.load(OUT / "evaluation_indices.npy")
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    groups = {}
    for i, row in enumerate(meta):
        groups.setdefault(row["match_id"], []).append(i)
    rng = np.random.default_rng(20261016)
    selected = np.array([rng.choice(v) for v in groups.values()])
    result = {}
    for view in ["observer", "server"]:
        ds = TrajectoryDataset(split="validation", view=view)
        models = []
        for seed in [41, 42, 43]:
            ck = torch.load(OUT / f"{view}_seed{seed}.pt", map_location="cpu", weights_only=False)
            m = JointGraphDiffusion(
                len(ds.builder.vocab["items"]) + len(ds.builder.vocab["vehicles"]) + 4
            )
            m.load_state_dict(ck["model"])
            m.eval()
            models.append((seed, m))
        records = []
        for number in selected:
            sample = ds[int(ids[number])]
            c = collate([sample])["conditions"]
            zone = c["node_type"] == NODE_TYPES.index("zone")
            cz = dict(c)
            cz["x"] = c["x"].clone()
            cz["feature_mask"] = c["feature_mask"].clone()
            cz["x"][zone] = 0
            cz["feature_mask"][zone] = False
            for seed, m in models:
                with torch.no_grad():
                    h = m.encode(c)
                original = m.encode

                def get(condition):
                    torch.manual_seed(seed * 100000 + int(number))
                    return m.sample(condition, 20)[0].numpy()

                base = get(c)
                m.encode = lambda _: torch.zeros_like(h)
                try:
                    no_condition = get(c)
                finally:
                    m.encode = original
                no_zone = get(cz)
                p = sample["conditions"]["current_presence"]
                records.append(
                    {
                        "sample_id": sample["meta"]["sample_id"],
                        "seed": seed,
                        "zero_learned_condition_meanpath_shift_m": float(
                            np.linalg.norm(
                                base.mean(0)[p] - no_condition.mean(0)[p], axis=-1
                            ).mean()
                            * 1000
                        ),
                        "masked_zone_features_meanpath_shift_m": float(
                            np.linalg.norm(base.mean(0)[p] - no_zone.mean(0)[p], axis=-1).mean()
                            * 1000
                        ),
                        "generated_spread_mean_m": float(
                            np.linalg.norm(base[:, p] - base.mean(0)[None, p], axis=-1).mean()
                            * 1000
                        ),
                    }
                )
        keys = [
            "zero_learned_condition_meanpath_shift_m",
            "masked_zone_features_meanpath_shift_m",
            "generated_spread_mean_m",
        ]
        result[view] = {
            "windows": len(selected),
            "seeds": 3,
            "summary": {
                k: {
                    "mean": float(np.mean([r[k] for r in records])),
                    "max": float(max(r[k] for r in records)),
                }
                for k in keys
            },
            "records": records,
        }
    report = {
        "status": "POST_HOC_DIAGNOSTIC_NOT_CONFIRMATORY_ABLATION",
        "method": "One randomly selected evaluation window per each of 188 matches (selection seed 20261016, independent of errors), all three training seeds, K20. An earlier every-16th-window probe was retained separately because the sorted evaluation order overrepresented early windows. CV reference and noise held fixed. Zero the entire learned condition (including encoded past and current state), or mask only zone node numerical features. Topology and CV motion remain. Neither intervention represents a retrained baseline; effects are output sensitivity, not causal feature importance. No model selection or retraining follows this diagnostic.",
        "results": result,
    }
    (OUT / "condition_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
