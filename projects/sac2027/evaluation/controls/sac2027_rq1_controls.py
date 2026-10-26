#!/usr/bin/env python3
"""AI-adjusted RQ1 and movement-component diagnostics, estimated only on patch 40.2."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import importlib.util
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.models.hgb as H
import pubg_lab.survival.ptr.evaluation.ecomodel as E
from projects.sac2027.training.sac2027_fair_tune import BASE, DUMP, save

path = REPO / "reports/bigcomp2027_results_v2/ecological_model/robustness_v12/robustness_v12.py"
spec = importlib.util.spec_from_file_location("rq1_helpers_v12", path)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
OUT = BASE / "rq1_controls"
BASE_TERMS = tuple(E.RQ1_TERMS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(BASE / "cohort_counts.csv")
    cohorts = counts.loc[counts.analysis_role == "main", "cohort"].tolist()
    train = {
        c: H.build_rows(D.load_dump(DUMP / c / "train.pt", labels="v2"), "HGB-cut", "v2")
        for c in cohorts
    }
    variants = {
        "unadjusted": dict(terms=BASE_TERMS),
        "ai_adjusted": dict(
            terms=BASE_TERMS + ("ai_share",),
            sources={"ai_share": "team_ai_frac"},
            construct={"ai_share": "control"},
        ),
        "human_only": dict(terms=BASE_TERMS, human=True),
        "radius100_ai": dict(
            terms=BASE_TERMS + ("ai_share",),
            sources={"ai_share": "team_ai_frac", "local_crowding": "hab_local_enemy_count_100m"},
            construct={"ai_share": "control"},
        ),
        "radius500_ai": dict(
            terms=BASE_TERMS + ("ai_share",),
            sources={"ai_share": "team_ai_frac", "local_crowding": "hab_local_enemy_count_500m"},
            construct={"ai_share": "control"},
        ),
        "movement_components_ai": dict(
            terms=tuple(t for t in BASE_TERMS if t != "migration_deficit")
            + ("required_speed", "observed_speed", "ai_share"),
            sources={
                "required_speed": "hab_required_speed_to_safe_mps",
                "observed_speed": "hab_observed_speed_mps",
                "ai_share": "team_ai_frac",
            },
            construct={
                "required_speed": "movement_component",
                "observed_speed": "movement_component",
                "ai_share": "control",
            },
        ),
        "movement_components_human": dict(
            terms=tuple(t for t in BASE_TERMS if t != "migration_deficit")
            + ("required_speed", "observed_speed"),
            human=True,
            sources={
                "required_speed": "hab_required_speed_to_safe_mps",
                "observed_speed": "hab_observed_speed_mps",
            },
            construct={
                "required_speed": "movement_component",
                "observed_speed": "movement_component",
            },
        ),
    }
    diagnostics = []
    for c, rows in train.items():
        names = list(rows.feature_names)
        ai = rows.X[:, names.index("team_ai_frac")]
        for pop, mask in (("human", ai == 0), ("AI", ai == 1)):
            x = rows.X[mask]
            speed = x[:, names.index("hab_observed_speed_mps")] * 40
            distance = x[:, names.index("hab_distance_to_target_safe_m")] * 8160
            required = x[:, names.index("hab_required_speed_to_safe_mps")] * 40
            deficit = x[:, names.index("hab_migration_deficit_mps")] * 40
            if not len(x):
                continue
            diagnostics.append(
                dict(
                    cohort=c,
                    population=pop,
                    n_rows=len(x),
                    fraction_d_positive=float(np.mean(distance > 0)),
                    speed_median=float(np.median(speed)),
                    speed_q25=float(np.quantile(speed, 0.25)),
                    speed_q75=float(np.quantile(speed, 0.75)),
                    deficit_negative_fraction=float(np.mean(deficit < 0)),
                    formula_max_error=float(np.max(np.abs(deficit - (required - speed)))),
                )
            )
    pd.DataFrame(diagnostics).to_csv(OUT / "movement_diagnostics_train.csv", index=False)
    summary = {}
    for name, variant in variants.items():
        estimates = []
        info = []
        ames = []
        with R.ecomodel_patched(
            terms=variant["terms"],
            sources=variant.get("sources"),
            construct=variant.get("construct"),
        ):
            for c, original in train.items():
                rows = original
                if variant.get("human"):
                    rows = R.subset_rows(
                        rows, rows.X[:, list(rows.feature_names).index("team_ai_frac")] == 0
                    )
                if len(rows.keys.match_id.unique()) < 50:
                    info.append(
                        dict(
                            cohort=c, skipped="fewer than 50 training match clusters in this subset"
                        )
                    )
                    continue
                est, fitinfo = R.fit_cohort(c, rows)
                estimates.append(est)
                info.append(fitinfo)
                if name in ("ai_adjusted", "human_only"):
                    E.RQ1_UNITS["ai_share"] = (1.0, "AI team indicator")
                    inputs = E.make_rq1_inputs(rows)
                    model = E.fit_rq1("phase", inputs, rows.y, units=E.rq1_units())
                    ame = E.ame_table(model, inputs, c, "M-phase")
                    # A +1 change is inappropriate for AI teams already at 1.
                    ames.append(ame[ame.covariate != "ai_share"])
                print(name, c, fitinfo["n_train_rows"], flush=True)
            all_est = pd.concat(estimates, ignore_index=True)
            main, groups = R.pool(all_est)
            R.tidy(main, groups, name).to_csv(OUT / f"tidy_{name}.csv", index=False)
            all_est.to_csv(OUT / f"cohort_estimates_{name}.csv", index=False)
            if ames:
                all_ame = pd.concat(ames, ignore_index=True)
                all_ame.to_csv(OUT / f"ame_{name}_per_cohort.csv", index=False)
                E.ame_pooled(all_ame).to_csv(OUT / f"ame_{name}_pooled.csv", index=False)
        summary[name] = info
        save(OUT / "fit_diagnostics.json", summary)
    save(
        OUT / "complete.json",
        dict(
            cohorts=cohorts,
            estimation_patch="40.2",
            n_variants=len(variants),
            interpretation="observational conditional associations; deficit is a retrospective speed residual, not directional progress",
        ),
    )


if __name__ == "__main__":
    main()
