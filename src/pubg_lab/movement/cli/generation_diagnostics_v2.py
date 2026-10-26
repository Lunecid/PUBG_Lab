"""Distribution and simple numerical feasibility diagnostics; no collision claims."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = (REPO_ROOT) / "reports/heterograph_diffusion_experiment_v2"


def main():
    meta = json.loads((OUT / "evaluation_samples.json").read_text())
    p = dict(np.load(OUT / "predictions.npz"))
    t = dict(np.load(OUT / "evaluation_targets.npz"))
    orig = np.array([r["origin_m"] for r in meta])
    result = {}
    for model, a in p.items():
        # [window,K,member,future,xy], all numerical diagnostics are per window first.
        ab = a * 1000 + orig[:, None, None, None]
        active = np.broadcast_to(t["presence"][:, None, :, None], a.shape[:-1])
        outside = ((ab < 0) | (ab >= 8160)).any(-1)
        counts = active.sum((1, 2, 3))
        outside_rate = (outside & active).sum((1, 2, 3)) / counts
        start = np.broadcast_to(t["initial_xy"][:, None, :, None], (len(a), a.shape[1], 4, 1, 2))
        path = np.concatenate([start, a], axis=3) * 1000
        length = np.linalg.norm(np.diff(path, axis=3), axis=-1).sum(3)
        truepath = np.concatenate([t["initial_xy"][:, :, None], t["trajectory"]], axis=2) * 1000
        true_len = np.linalg.norm(np.diff(truepath, axis=2), axis=-1).sum(2)
        full = t["trajectory_valid"].all(2) & t["presence"]
        error = np.abs(length - true_len[:, None])
        fulln = full.sum(1)
        length_error = (error * full[:, None]).sum((1, 2)) / np.maximum(fulln * a.shape[1], 1)
        metrics = {
            "outside_map_coordinate_fraction": outside_rate,
            "mean_path_length_error_complete_member_m": np.where(fulln > 0, length_error, np.nan),
        }
        end = a[:, :, 0, -1] * 1000
        true = t["trajectory"][:, 0, -1] * 1000
        center = end.mean(1)
        r = np.linalg.norm(end - center[:, None], axis=-1)
        endvalid = t["trajectory_valid"][:, 0, -1]
        if a.shape[1] > 1:
            for q in [0.5, 0.8, 0.9]:
                radius = np.quantile(r, q, axis=1)
                metrics[f"ego_endpoint_radial_{int(q * 100)}pct_coverage"] = np.where(
                    endvalid, np.linalg.norm(true - center, axis=-1) <= radius, np.nan
                )
                metrics[f"ego_endpoint_radial_{int(q * 100)}pct_radius_m"] = np.where(
                    endvalid, radius, np.nan
                )
        grouped = {}
        for key, values in metrics.items():
            g = defaultdict(list)
            for m, v in zip(meta, values):
                if np.isfinite(v):
                    g[m["match_id"]].append(float(v))
            grouped[key] = {
                "match_macro": float(np.mean([np.mean(v) for v in g.values()])),
                "matches": len(g),
                "windows": int(np.isfinite(values).sum()),
            }
        result[model] = grouped
    report = {
        "definition": "Radial coverage uses the empirical quantile of 20 generated endpoint distances from their sample mean; it is a descriptive calibration check, not a confidence interval or joint trajectory coverage. Outside-map rate tests only coordinate box [0,8160), not terrain or collisions. Path length is evaluated only for members with all six targets. All aggregate diagnostics are match-macro.",
        "models": result,
    }
    (OUT / "generation_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
