"""Post-run support and parameter audit; no fitting and no protected data loader."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import fields

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import torch
from pubg_lab.world_model.training.pretraining import supervised_targets
from projects.world_model.training.train_world_model_g4 import schedule, STRATA
from projects.world_model.common.resources import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, out = Path(a.run), Path(a.output)
    gate = json.loads((root / "prepared_01/summary.json").read_text())
    assert sha(root / "prepared_01/samples.pt") == gate["samples_sha256"]
    data = torch.load(root / "prepared_01/samples.pt", weights_only=False, map_location="cpu")
    strata = {}
    for role in ("train", "validation"):
        groups = defaultdict(list)
        for s in data["samples"][role]["observer"]["H1"]:
            g = s.inputs
            y = supervised_targets(s)
            i = int(torch.where(g.target_player_indices == g.ego_local_index)[0][0])
            m = y["mask"][i, :3]
            for f in fields(g):
                x = getattr(g, f.name)
                if isinstance(x, torch.Tensor):
                    assert torch.isfinite(x).all(), f.name
            assert not len(g.observed) and not g.cells[:, 2:6].any()
            assert not (
                {tuple(r[:2]) for r in g.past_edges.tolist()}
                & {tuple(r) for r in g.inferred.tolist()}
            )
            groups[s.provenance["stratum"]].append(
                dict(
                    ego_current_mode=int(g.mobility[-1, g.ego_local_index]),
                    velocity_known=bool(g.mask[-1, g.ego_local_index, 3:5].all()),
                    speed=float(g.x[-1, g.ego_local_index, 3:5].norm() * 10),
                    valid_history=int(g.history_valid[:, g.ego_local_index].sum()),
                    position_age_s=float(g.x[-1, g.ego_local_index, 6] * 120),
                    displacement=float((y["delta"][i, :3][m] * 100).norm(dim=-1).mean())
                    if m.any()
                    else None,
                    mode4_points=int(((y["mode"] == 4) & y["mask"]).sum()),
                    mode5_points=int(((y["mode"] == 5) & y["mask"]).sum()),
                    label_points=int(y["mask"].sum()),
                )
            )
        strata[role] = {
            st: dict(
                samples=len(rr),
                ego_current_modes=dict(Counter(r["ego_current_mode"] for r in rr)),
                velocity_known=sum(r["velocity_known"] for r in rr),
                mean_ego_speed_mps=float(np.mean([r["speed"] for r in rr])),
                mean_history_points=float(np.mean([r["valid_history"] for r in rr])),
                mean_position_age_s=float(np.mean([r["position_age_s"] for r in rr])),
                mean_ego_displacement30_m=float(
                    np.mean([r["displacement"] for r in rr if r["displacement"] is not None])
                ),
                mode4_points=sum(r["mode4_points"] for r in rr),
                mode5_points=sum(r["mode5_points"] for r in rr),
                label_points=sum(r["label_points"] for r in rr),
            )
            for st, rr in groups.items()
        }
    checks = {}
    exposure = {}
    for seed in (20261020, 20261021, 20261022):
        train = data["samples"]["train"]["observer"]["H1"]
        counts = [
            Counter(supervised_targets(s)["mode"][supervised_targets(s)["mode_mask"]].tolist())
            for s in train
        ]
        exposure[str(seed)] = {}
        for arm in ("natural", "mixed"):
            c = Counter()
            for batch in schedule(train, 192, seed, arm == "mixed"):
                for i in batch:
                    c.update(counts[i])
            exposure[str(seed)][arm] = dict(c)
        for view in ("observer", "server"):
            state = torch.load(
                root / f"seed_{seed}_02/{view}_checkpoint.pt",
                weights_only=False,
                map_location="cpu",
            )
            assert all(torch.isfinite(t).all() for group in state.values() for t in group.values())
            checks[f"{seed}_{view}"] = {
                arm: dict(
                    encoder_changed=any(
                        not torch.equal(t, state["random_encoder"][k])
                        for k, t in state[arm + "_encoder"].items()
                    ),
                    flag_weights_zero=all(
                        not state[arm + "_encoder"][f"spatial.{i}.{part}.weight"][:, 3:]
                        .any()
                        .item()
                        for i in (0, 1)
                        for part in ("support", "gate")
                    ),
                )
                for arm in ("natural", "mixed")
            }
    assert all(
        v["encoder_changed"] and v["flag_weights_zero"] for r in checks.values() for v in r.values()
    )
    result = dict(
        strata=strata,
        checkpoint_checks=checks,
        mode_exposures=exposure,
        status="PASS",
        protected_details_read=False,
        prepared_sha256=gate["samples_sha256"],
    )
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({"status": "PASS", "mode_exposures": exposure, "checkpoint_groups": len(checks)})
    )


if __name__ == "__main__":
    main()
