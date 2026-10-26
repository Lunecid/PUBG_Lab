#!/usr/bin/env python3
"""Explicit 2x2 objective x validation probability-mapping analysis, no rounded pooled arithmetic."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from projects.sac2027.training.sac2027_fair_tune import BASE, M, save


def design(frame, phased):
    z = -frame.rank_score.to_numpy()
    return (
        np.column_stack([z, np.eye(7)[frame.phase.to_numpy(dtype=int) - 1]])
        if phased
        else z[:, None]
    )


def main():
    out = BASE / "mapping_factorial"
    out.mkdir(parents=True, exist_ok=True)
    pooled = {}
    summaries = []
    for cohort in ("erangel_tpp_squad", "taego_tpp_squad"):
        frames = {}
        calibrations = []
        for objective, obit in (("weighted_full", 0), ("bce", 1)):
            for seed in (42, 13, 2026):
                path = BASE / "objective_controls" / cohort / objective / f"seed{seed}"
                va = M.read_predictions(path / "pred_val_raw.csv")
                te = M.read_predictions(path / "pred_test_raw.csv")
                for bbit in (0, 1):
                    cal = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs").fit(
                        design(va, bbit), va.y
                    )
                    pred = te.copy()
                    p = cal.predict_proba(design(te, bbit))[:, 1]
                    pred["p_hazard"] = p
                    pred["rank_score"] = -np.log(
                        np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1)
                    )
                    key = f"B{bbit}O{obit}_s{seed}"
                    frames[key] = pred
                    calibrations.append(
                        dict(
                            cell=key,
                            coef=cal.coef_,
                            intercept=cal.intercept_,
                            fit_split="41.1 validation",
                        )
                    )
        contrasts = {"mapping_shapley": {}, "objective_shapley": {}, "joint_gain": {}}
        for seed in (42, 13, 2026):
            for cell, coef in {"B1O0": 0.5, "B0O0": -0.5, "B1O1": 0.5, "B0O1": -0.5}.items():
                contrasts["mapping_shapley"][f"{cell}_s{seed}"] = coef / 3
            for cell, coef in {"B0O1": 0.5, "B0O0": -0.5, "B1O1": 0.5, "B1O0": -0.5}.items():
                contrasts["objective_shapley"][f"{cell}_s{seed}"] = coef / 3
            contrasts["joint_gain"][f"B1O1_s{seed}"] = 1 / 3
            contrasts["joint_gain"][f"B0O0_s{seed}"] = -1 / 3
        result = M.paired_contrasts(
            frames,
            contrasts,
            n_boot=2000,
            seed=20261008 + (cohort.startswith("taego")),
            rank_col="rank_score",
            metrics=("auc", "auc_within_phase", "nll", "brier"),
            return_samples=True,
        )
        samples = result.pop("samples")
        for metric in ("auc", "auc_within_phase", "nll", "brier"):
            assert np.allclose(
                samples["mapping_shapley"][metric] + samples["objective_shapley"][metric],
                samples["joint_gain"][metric],
                atol=1e-12,
            )
            for contrast in contrasts:
                pooled.setdefault((contrast, metric), []).append(
                    (result["contrasts"][contrast][metric]["estimate"], samples[contrast][metric])
                )
        save(out / f"{cohort}.json", dict(results=result, calibrations=calibrations))
    for (contrast, metric), items in pooled.items():
        estimate = float(np.mean([v[0] for v in items]))
        draws = np.mean([v[1] for v in items], axis=0)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        summaries.append(
            dict(
                contrast=contrast,
                metric=metric,
                estimate=estimate,
                ci_low=float(lo),
                ci_high=float(hi),
                weighting="equal cohort and equal training seed",
            )
        )
    pd.DataFrame(summaries).to_csv(out / "pooled.csv", index=False)
    save(
        out / "provenance.json",
        dict(
            design="explicit 2x2 objective x global/phase-aware logistic validation mapping",
            note="NLL/Brier differences are losses: negative is improvement. This is not a factorial decomposition of all input and architecture changes.",
        ),
    )


if __name__ == "__main__":
    main()
