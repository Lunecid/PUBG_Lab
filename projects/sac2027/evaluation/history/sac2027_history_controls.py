#!/usr/bin/env python3
"""Causal prefix-order and summary controls. Never permute across a forecast cutoff."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import hashlib
import json
import sys
from dataclasses import replace, asdict
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
    torch,
    np,
    build_model,
)
from pubg_lab.survival.ptr.training.loss import hazard_bce
from pubg_lab.survival.ptr.common.util import set_seed

PHASE_KEYS = (
    "x",
    "node_mask",
    "query_mask",
    "target_mask",
    "y_hazard",
    "pos",
    "pos_valid",
    "zone",
    "zone_std",
    "zone_valid",
    "rel",
)
KINDS = (
    "krafton_current_padded",
    "krafton_history_summary",
    "ptr_ordered_prefix",
    "ptr_shuffled_prefix",
)


class PrefixOrder(torch.nn.Module):
    def __init__(self, base, shuffle=False):
        super().__init__()
        self.base = base
        self.shuffle = shuffle

    def forward(self, batch):
        b, p, n, _ = batch["x"].shape
        outputs = []
        for cut in range(p):
            # A separate prefix for each forecast keeps every source phase <= cut.
            orders = []
            for mid in batch["match_ids"]:
                past = np.arange(cut)
                if self.shuffle:
                    key = int(hashlib.sha256(f"20261008|{mid}|{cut}".encode()).hexdigest()[:8], 16)
                    past = np.random.default_rng(key).permutation(past)
                orders.append(np.r_[past, cut])
            order = torch.as_tensor(np.array(orders), device=batch["x"].device, dtype=torch.long)
            prefix = {**batch}
            for name in PHASE_KEYS:
                if name in batch:
                    tensor = batch[name]
                    indices = order.reshape(b, cut + 1, *([1] * (tensor.ndim - 2))).expand(
                        b, cut + 1, *tensor.shape[2:]
                    )
                    prefix[name] = torch.gather(tensor, 1, indices)
            outputs.append(self.base(prefix)["hazard_logit"][:, -1])
        z = torch.stack(outputs, dim=1)
        return dict(hazard_logit=z, rank_score=-z)


class PastSummary(torch.nn.Module):
    def __init__(self, base, include_history=True):
        super().__init__()
        self.base = base
        self.include_history = include_history

    def forward(self, batch):
        # A query can be added by retrospective target eligibility (comeback).
        # Only causally observed keys may contribute to another time's history.
        x = batch["x"]
        mask = batch["node_mask"].to(x.dtype)[..., None]
        total = torch.cumsum(x * mask, dim=1) - x * mask
        count = torch.cumsum(mask, dim=1) - mask
        mean = total / count.clamp(min=1)
        if not self.include_history:
            mean = torch.zeros_like(mean)
            count = torch.zeros_like(count)
        augmented = torch.cat([x, mean, count / 7.0], dim=-1)
        return self.base({**batch, "x": augmented})


def model_for(cfg, kind, seed):
    if kind.startswith("ptr"):
        return PrefixOrder(build_model(cfg, "ptr", seed), shuffle=kind == "ptr_shuffled_prefix")
    return PastSummary(
        build_model(replace(cfg, in_features=2 * cfg.in_features + 1), "krafton", seed),
        include_history=kind == "krafton_history_summary",
    )


def run(prep, cohort, kind, setting, seed, device):
    dest = BASE / "history_controls" / cohort / kind / f"seed{seed}"
    if (dest / "metrics.json").exists():
        return
    set_seed(seed)
    cfg = replace(
        prep.config, **{k: setting[k] for k in ("d_model", "n_heads", "n_layers", "dropout")}
    )
    model = model_for(cfg, kind, seed).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=setting["lr"], weight_decay=setting["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    stop = T.EarlyStopper(3)
    rng = np.random.default_rng(seed)
    history = []
    best = None
    for epoch in range(1, 13):
        model.train()
        for ids in T._chunks(rng.permutation(len(prep.datasets["train"])), 8):
            batch = T._to_device(prep.datasets["train"].get_batch(ids), device)
            loss = hazard_bce(model(batch)["hazard_logit"], batch["y_hazard"], batch["target_mask"])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()
        nll = T.evaluate_nll(model, prep.datasets["val"], device, 8)
        assert np.isfinite(nll)
        improved = stop.update(nll)
        history.append(dict(epoch=epoch, val_nll=nll))
        print(cohort, kind, seed, epoch, round(nll, 6), flush=True)
        if improved:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stop.should_stop:
            break
    model.load_state_dict(best)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=best,
            kind=kind,
            config=asdict(cfg),
            standardizer=prep.standardizer.state_dict(),
            seed=seed,
        ),
        dest / "best.pt",
    )
    result = {}
    for split in ("val", "test"):
        frame = T.predict(model, prep.datasets[split], device, 8, rank_from_hazard=True)
        M.write_predictions(frame, dest / f"pred_{split}.csv")
        result[split] = M.hazard_metrics(frame.y, frame.p_hazard)
    save(
        dest / "metrics.json",
        dict(
            kind=kind,
            setting=setting,
            seed=seed,
            best_epoch=stop.best_epoch,
            history=history,
            metrics=result,
            note="Current token retained. Prefix shuffle retains original phase-related feature values, so it tests sensitivity to sequence presentation, not absence of all temporal information.",
        ),
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--kind", choices=KINDS)
    args = ap.parse_args()
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
                str(BASE / "history_controls"),
            ]
        )
    )
    for kind in [args.kind] if args.kind else KINDS:
        architecture = "ptr" if kind.startswith("ptr") else "krafton"
        setting = json.loads((OUT / args.cohort / architecture / "selection.json").read_text())[
            "setting"
        ]
        for seed in (42, 13, 2026):
            run(prep, args.cohort, kind, setting, seed, T._device(args.device))


if __name__ == "__main__":
    main()
