#!/usr/bin/env python3
"""Frozen equal-trial-budget search. No test predictions until all trials finish."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import fcntl
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = REPO_ROOT
sys.path[:0] = [str(REPO), str(REPO / "third_party/krafton_pubg_winprob")]
import numpy as np
import pandas as pd
import torch
from torch import nn
import pubg_lab.survival.ptr.training.train as T
import pubg_lab.survival.ptr.evaluation.metrics as M
from pubg_lab.survival.ptr.training.loss import hazard_bce
from pubg_lab.survival.ptr.models.model import PTRFormer
from pubg_lab.survival.ptr.common.util import set_seed, to_jsonable
from projects.krafton_baseline.models.ourfeat import TokenOnlyBackbone
from src.models.heads import HazardHead

BASE = REPO / "reports/sac2027_patch_v14"
DUMP = BASE / "dumps/phase_v2"
OUT = BASE / "tuning"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
    temp.replace(path)


def search_spec():
    rng = np.random.default_rng(20261008)
    nn_configs = [
        dict(d_model=128, n_heads=4, n_layers=2, dropout=0.1, lr=0.001, weight_decay=0.0001),
        dict(d_model=64, n_heads=8, n_layers=8, dropout=0.1, lr=0.0001, weight_decay=0.0001),
    ]
    for _ in range(14):
        nn_configs.append(
            dict(
                d_model=int(rng.choice([64, 128])),
                n_heads=int(rng.choice([4, 8])),
                n_layers=int(rng.choice([1, 2, 4, 8])),
                dropout=float(rng.choice([0, 0.1, 0.2, 0.3])),
                lr=float(10 ** rng.uniform(-4, np.log10(0.003))),
                weight_decay=float(10 ** rng.uniform(-6, -2)),
            )
        )
    gb_configs = [
        dict(
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=0.01,
            max_leaf_nodes=31,
            min_samples_leaf=20,
        )
    ]
    for _ in range(15):
        gb_configs.append(
            dict(
                max_iter=int(rng.choice([100, 200, 400])),
                learning_rate=float(10 ** rng.uniform(-2, -0.7)),
                l2_regularization=float(10 ** rng.uniform(-4, 1)),
                max_leaf_nodes=int(rng.choice([15, 31, 63])),
                min_samples_leaf=int(rng.choice([10, 20, 40, 80])),
            )
        )
    return dict(
        random_seed=20261008,
        selection="minimum validation NLL; trial index breaks ties",
        neural_trials=nn_configs,
        gbdt_trials=gb_configs,
        epochs=12,
        patience=3,
        batch_matches=8,
        seeds=[42, 13, 2026],
        standardization="train only; identical 153 token columns for all models",
        note="14 random + 2 reference neural settings; 15 random + 1 reference GBDT setting; equal trials, not equal FLOPs",
    )


def freeze():
    spec = search_spec()
    path = OUT / "search_space.json"
    if path.exists():
        assert json.loads(path.read_text()) == spec
    else:
        save(path, spec)
    return spec


class KRAFTONMatched(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = TokenOnlyBackbone(
            cfg.in_features, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout
        )
        self.head = HazardHead(cfg.d_model, dropout=cfg.dropout)

    def forward(self, batch):
        x = batch["x"]
        b, p, n, f = x.shape
        keys = batch["node_mask"].reshape(b * p, n).clone()
        # Empty phases carry no targets. A dummy key avoids all-masked attention NaNs.
        empty = ~keys.any(dim=1)
        keys[empty, 0] = True
        flat = x.reshape(b * p, n, f)
        h = self.backbone(flat, valid_mask=keys)
        z = self.head(h).reshape(b, p, n)
        z = torch.where(batch["query_mask"], z, torch.zeros_like(z))
        return {"hazard_logit": z, "rank_score": -z}


def build_model(cfg, architecture, seed):
    set_seed(seed)
    return PTRFormer(cfg, init_seed=seed) if architecture == "ptr" else KRAFTONMatched(cfg)


def fit(prep, architecture, setting, seed, device, dest):
    cfg = replace(
        prep.config, **{k: setting[k] for k in ("d_model", "n_heads", "n_layers", "dropout")}
    )
    model = build_model(cfg, architecture, seed).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=setting["lr"], weight_decay=setting["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    stop = T.EarlyStopper(3)
    rng = np.random.default_rng(seed)
    history = []
    best = None
    t0 = time.perf_counter()
    for epoch in range(1, 13):
        model.train()
        total = steps = 0
        for ids in T._chunks(rng.permutation(len(prep.datasets["train"])), 8):
            batch = T._to_device(prep.datasets["train"].get_batch(ids), device)
            loss = hazard_bce(model(batch)["hazard_logit"], batch["y_hazard"], batch["target_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        scheduler.step()
        score = T.evaluate_nll(model, prep.datasets["val"], device, 8)
        if not np.isfinite(score):
            raise RuntimeError("nonfinite validation loss")
        improved = stop.update(score)
        history.append(
            dict(
                epoch=epoch,
                train_loss=total / max(1, steps),
                val_nll=score,
                elapsed_sec=time.perf_counter() - t0,
            )
        )
        print(architecture, seed, dest.name, "epoch", epoch, "val_nll", round(score, 6), flush=True)
        if improved:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stop.should_stop:
            break
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=best,
            config=asdict(cfg),
            standardizer=prep.standardizer.state_dict(),
            architecture=architecture,
            seed=seed,
            setting=setting,
        ),
        dest / "best.pt",
    )
    result = dict(
        architecture=architecture,
        seed=seed,
        setting=setting,
        history=history,
        best_epoch=stop.best_epoch,
        val_nll=stop.best,
        train_sec=time.perf_counter() - t0,
        n_parameters=sum(p.numel() for p in model.parameters()),
    )
    save(dest / "trial.json", result)
    model.load_state_dict(best)
    return model, result


def arrays(ds):
    rows = []
    xs = []
    for i in range(len(ds)):
        batch = ds.get_batch([i])
        p, n = batch["target_mask"][0].nonzero(as_tuple=True)
        xs.append(batch["x"][0, p, n].numpy())
        rows.append(
            pd.DataFrame(
                dict(
                    match_id=batch["match_ids"][0],
                    team_id=batch["team_ids"][0, n].numpy(),
                    phase=p.numpy() + 1,
                    y=batch["y_hazard"][0, p, n].numpy(),
                    final_rank_norm=batch["final_rank_norm"][0, n].numpy(),
                    top5=batch["top5"][0, n].numpy(),
                    match_type=batch["match_types"][0],
                )
            )
        )
    return np.concatenate(xs), pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--cohort", choices=["erangel_tpp_squad", "taego_tpp_squad"])
    ap.add_argument("--architecture", choices=["ptr", "krafton", "gbdt"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    spec = freeze()
    if args.freeze_only:
        return
    assert args.cohort and args.architecture
    dest = OUT / args.cohort / args.architecture
    dest.mkdir(parents=True, exist_ok=True)
    run_lock = (dest / ".run.lock").open("a")
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX)
    torch.set_num_threads(2)
    prep = T.prepare(
        T.parse_args(
            [
                "--dump-dir",
                str(DUMP),
                "--cohort",
                args.cohort,
                "--arm",
                "M11R-norank",
                "--out-dir",
                str(OUT),
            ]
        )
    )
    device = T._device(args.device) if args.architecture != "gbdt" else torch.device("cpu")
    trials = spec["gbdt_trials" if args.architecture == "gbdt" else "neural_trials"]
    if args.architecture == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
        import joblib

        train_x, train_rows = arrays(prep.datasets["train"])
        val_x, val_rows = arrays(prep.datasets["val"])
    records = []
    for i, setting in enumerate(trials):
        folder = dest / f"trial{i:02d}"
        if (folder / "trial.json").exists():
            record = json.loads((folder / "trial.json").read_text())
        elif args.architecture == "gbdt":
            t0 = time.perf_counter()
            model = HistGradientBoostingClassifier(**setting, early_stopping=False, random_state=42)
            with threadpool_limits(limits=2):
                model.fit(train_x, train_rows.y)
                p = model.predict_proba(val_x)[:, 1]
            record = dict(
                setting=setting,
                val_nll=M.hazard_metrics(val_rows.y, p)["nll"],
                train_sec=time.perf_counter() - t0,
            )
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / "best.joblib")
            save(folder / "trial.json", record)
            print("gbdt", i, record["val_nll"], flush=True)
        else:
            model, record = fit(prep, args.architecture, setting, 42, device, folder)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        assert record["setting"] == setting
        records.append(dict(trial=i, **record))
    chosen = min(records, key=lambda r: (r["val_nll"], r["trial"]))
    save(
        dest / "selection.json",
        dict(
            chosen_trial=chosen["trial"],
            setting=chosen["setting"],
            val_nll=chosen["val_nll"],
            n_trials=len(records),
        ),
    )
    # Selection is frozen on disk before any test prediction.
    for seed in spec["seeds"]:
        target = dest / f"selected_seed{seed}"
        if (target / "metrics.json").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        folder = dest / f"trial{chosen['trial']:02d}"
        if args.architecture == "gbdt":
            model = (
                joblib.load(folder / "best.joblib")
                if seed == 42
                else HistGradientBoostingClassifier(
                    **chosen["setting"], early_stopping=False, random_state=seed
                )
            )
            with threadpool_limits(limits=2):
                if seed != 42:
                    model.fit(train_x, train_rows.y)
            joblib.dump(model, target / "best.joblib")
        else:
            if seed == 42:
                ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
                model = build_model(T.PTRConfig(**ckpt["config"]), args.architecture, seed).to(
                    device
                )
                model.load_state_dict(ckpt["state_dict"])
                torch.save(ckpt, target / "best.pt")
            else:
                model, _ = fit(prep, args.architecture, chosen["setting"], seed, device, target)
        results = {}
        for split in ("val", "test"):
            if args.architecture == "gbdt":
                x, frame = arrays(prep.datasets[split])
                with threadpool_limits(limits=2):
                    p = model.predict_proba(x)[:, 1]
                frame["p_hazard"] = p
                frame["rank_score"] = -np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
            else:
                frame = T.predict(model, prep.datasets[split], device, 8, rank_from_hazard=True)
            M.write_predictions(frame, target / f"pred_{split}.csv")
            results[split] = M.hazard_metrics(frame.y, frame.p_hazard)
        save(target / "metrics.json", dict(seed=seed, selection=chosen, metrics=results))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
