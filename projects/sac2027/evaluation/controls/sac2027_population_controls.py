#!/usr/bin/env python3
"""Train-only population/AI-input controls with frozen full-population settings."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import copy
import json
import sys
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
from projects.sac2027.training.sac2027_fair_tune import (
    BASE,
    DUMP,
    OUT,
    T,
    M,
    save,
    fit,
    arrays,
    torch,
    np,
)


class ControlledDataset:
    def __init__(self, base, *, remove_ai=False, human_loss=False):
        self.base = base
        self.remove_ai = remove_ai
        self.human_loss = human_loss

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_batch(self, indices):
        batch = self.base.get_batch(indices)
        if self.remove_ai:
            batch["x"] = batch["x"].clone()
            # PTR appends X + 2-D position + AI share; only this last token
            # input is ablated, without changing masks or team context.
            assert batch["x"].shape[-1] == 153
            batch["x"][..., -1] = 0
        if self.human_loss:
            mask = batch["target_mask"].clone()
            for bi, index in enumerate(indices):
                ai = torch.as_tensor(self.base.matches[index]["team_ai_frac"])
                mask[bi, :, : len(ai)] &= (ai == 0)[None, :]
            batch["target_mask"] = mask
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--architecture", choices=["ptr", "krafton", "gbdt"], required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--controls", default="no_ai_input,human_loss")
    args = ap.parse_args()
    torch.set_num_threads(2)
    setting = json.loads((OUT / args.cohort / args.architecture / "selection.json").read_text())[
        "setting"
    ]
    base = T.prepare(
        T.parse_args(
            [
                "--dump-dir",
                str(DUMP),
                "--cohort",
                args.cohort,
                "--arm",
                "M11R-norank",
                "--out-dir",
                str(BASE / "population_controls"),
            ]
        )
    )
    controls = args.controls.split(",")
    assert set(controls) <= {"no_ai_input", "human_loss", "mixed_cpu"}
    for control in controls:
        prep = copy.copy(base)
        prep.datasets = {
            s: ControlledDataset(
                ds,
                remove_ai=control == "no_ai_input",
                human_loss=control == "human_loss" and s in ("train", "val"),
            )
            for s, ds in base.datasets.items()
        }
        for seed in (42, 13, 2026):
            dest = (
                BASE
                / "population_controls"
                / args.cohort
                / args.architecture
                / control
                / f"seed{seed}"
            )
            if (dest / "metrics.json").exists():
                continue
            print(args.cohort, args.architecture, control, seed, flush=True)
            if args.architecture == "gbdt":
                from sklearn.ensemble import HistGradientBoostingClassifier
                from threadpoolctl import threadpool_limits
                import joblib

                x, r = arrays(prep.datasets["train"])
                model = HistGradientBoostingClassifier(
                    **setting, early_stopping=False, random_state=seed
                )
                with threadpool_limits(limits=2):
                    model.fit(x, r.y)
                dest.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, dest / "best.joblib")
            else:
                model, _ = fit(prep, args.architecture, setting, seed, T._device(args.device), dest)
            result = {}
            for split in ("val", "test"):
                # Predict all held-out teams, then report human-only metrics too.
                ds = ControlledDataset(base.datasets[split], remove_ai=control == "no_ai_input")
                if args.architecture == "gbdt":
                    x, frame = arrays(ds)
                    with threadpool_limits(limits=2):
                        p = model.predict_proba(x)[:, 1]
                    frame["p_hazard"] = p
                    frame["rank_score"] = -np.log(
                        np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1)
                    )
                else:
                    frame = T.predict(model, ds, T._device(args.device), 8, rank_from_hazard=True)
                M.write_predictions(frame, dest / f"pred_{split}.csv")
                result[split] = M.hazard_metrics(frame.y, frame.p_hazard)
            save(
                dest / "metrics.json",
                dict(
                    control=control,
                    seed=seed,
                    setting=setting,
                    metrics=result,
                    standardizer="same train tokens as mixed training; opponents retained as context",
                    validation_selection="human target rows only"
                    if control == "human_loss" and args.architecture != "gbdt"
                    else "original frozen setting; no test selection",
                ),
            )
            del model
            if args.device == "mps" and args.architecture != "gbdt":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
