"""Fit-only sampling metadata; evaluation plans are byte-identical V3 copies."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import csv
import json
import hashlib
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))
from pubg_lab.movement.data.dataset_v3 import DEFAULT

OLD = ROOT / "reports/heterograph_diffusion_experiment_v3"
OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"


def main():
    assert not list(OUT.glob("*.pt"))
    idx = np.load(DEFAULT / "train_index.npy")
    rows = {int(r["match_index"]): r for r in csv.DictReader((DEFAULT / "matches.csv").open())}
    mode = np.zeros(len(idx), np.int8)
    episode = np.zeros(len(idx), np.int16)
    for n, mi in enumerate(np.unique(idx[:, 0])):
        row = rows[int(mi)]
        assert row["split"] == "train"
        ids = np.flatnonzero(idx[:, 0] == mi)
        with np.load(DEFAULT / "banks" / f"{row['match_id']}.npz") as b:
            mode[ids] = b["mobility"][idx[ids, 1], idx[ids, 2]]
            episode[ids] = b["episode"][idx[ids, 1], idx[ids, 2]]
        if n % 100 == 0:
            print("TRAIN_METADATA", n, flush=True)
    eligible = idx[:, 4] > 0
    pop = {
        "pickup": eligible & (mode == 4),
        "redeploy": eligible & (mode == 5),
        "post_reentry": eligible & (episode > 0) & ~np.isin(mode, [4, 5]),
    }
    pools = {
        g: {int(m): np.flatnonzero(mask & (idx[:, 0] == m)) for m in np.unique(idx[mask, 0])}
        for g, mask in pop.items()
    }
    coverage = {
        "pool": {
            g: {"windows": int(mask.sum()), "matches": len(pools[g])} for g, mask in pop.items()
        },
        "plans": {},
    }
    for seed in [41, 42, 43]:
        rng = np.random.default_rng(seed + 20261020)
        old = np.load(OLD / f"training_indices_seed{seed}.npy")
        plan = old.copy()
        for step in range(len(plan)):
            for slot, g in [(3, "pickup" if step % 2 == 0 else "redeploy"), (7, "post_reentry")]:
                mid = int(rng.choice(sorted(pools[g])))
                plan[step, slot] = rng.choice(pools[g][mid])
        np.save(OUT / f"training_indices_seed{seed}.npy", plan)
        chosen = plan.ravel()
        coverage["plans"][str(seed)] = {
            "draws": len(chosen),
            "unique_windows": len(np.unique(chosen)),
            "matches": len(np.unique(idx[chosen, 0])),
            "mode_counts": {
                str(m): int((mode[chosen] == m).sum()) for m in np.unique(mode[chosen])
            },
            "reentry_draws": int((episode[chosen] > 0).sum()),
            "sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
        }
    np.savez_compressed(OUT / "train_sampling_metadata.npz", mode=mode, episode=episode)
    for name in [
        "evaluation_indices.npy",
        "transport_audit_plan.json",
        "evaluation_samples.json",
        "evaluation_targets.npz",
        "visualization_cases.json",
        "evaluation_baselines.json",
    ]:
        shutil.copyfile(OLD / name, OUT / name)
    (OUT / "sampling_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
