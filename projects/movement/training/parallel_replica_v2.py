"""Run one independent preplanned seed; no changes to the frozen experiment."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((REPO_ROOT / "tools/movement_diffusion")))
from projects.movement.training.experiment_v2 import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    args.output = ROOT / "reports/heterograph_diffusion_experiment_v2"
    args.steps = 2500
    args.batch_size = 8
    args.workers = 2
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan = np.load(args.output / f"training_indices_seed{args.seed}.npy")
    for view in ["observer", "server"]:
        train(args, view, args.seed, plan)
    print("REPLICA_COMPLETE", args.seed, flush=True)


if __name__ == "__main__":
    main()
