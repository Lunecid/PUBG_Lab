#!/usr/bin/env python3
"""Matched objective/horizon controls; validation-only stopping and calibration."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
from projects.sac2027.training.sac2027_fair_tune import (
    BASE,
    DUMP,
    OUT,
    T,
    M,
    build_model,
    save,
    torch,
    np,
    pd,
)
from pubg_lab.survival.ptr.training.loss import hazard_bce
from sklearn.linear_model import LogisticRegression

OBJECTIVES = (
    "weighted_full",
    "weighted_truncated",
    "cox_full",
    "cox_truncated",
    "bce",
    "cox_truncated_bce",
    "cond_logit",
    "cox_truncated_bce_w025",
    "cox_truncated_bce_w05",
    "cox_truncated_bce_w2",
    "cox_truncated_bce_w4",
)
# cox_truncated_bce adds a pointwise BCE term (weight fixed before training) to next-phase Cox; the BCE term alone
# carries a gradient for a match-phase offset (reports/sac2027_patch_v14/COXBCE_PLAN_KO_2026-10-09.md).
# The *_w* variants sweep that weight and cond_logit conditions on the per-match-phase event count
# (reports/sac2027_patch_v14/CONDLOGIT_PLAN_KO_2026-10-09.md, written before these runs).
COMBINED_BCE_WEIGHT = 1.0
COMBINED_WEIGHTS = {
    "cox_truncated_bce": 1.0,
    "cox_truncated_bce_w025": 0.25,
    "cox_truncated_bce_w05": 0.5,
    "cox_truncated_bce_w2": 2.0,
    "cox_truncated_bce_w4": 4.0,
}


def cond_logit_loss(z, y, valid):
    """Conditional logistic likelihood given the number of next-phase eliminations in each match-phase.

    For a risk set R with event set E (|E| = e), P(E | e) = exp(sum_E s) / e_e(exp s_j : j in R), where e_e is the
    elementary symmetric polynomial of degree e, computed in log space by the standard one-item-at-a-time recursion.
    Adding a constant to every score of a match-phase cancels (d loss / d c = 0), and only the binary next-phase labels
    enter, so the loss has neither the match-common signal of BCE nor the within-phase ordering of Cox.
    Normalised per event over informative sets (0 < e < |R|), matching the per-event scale of survival_loss.
    """
    z = z.flatten(0, 1)
    valid = valid.flatten(0, 1)
    y = torch.nan_to_num(y.flatten(0, 1)).bool() & valid
    g, n = z.shape
    s = torch.where(valid, z, torch.zeros_like(z))
    e = y.sum(-1)
    size = valid.sum(-1)
    kmax = int(e.max().item()) if g else 0
    # Empty states hold a finite floor instead of -inf: logaddexp(-inf, -inf) has a NaN gradient, and the floor is far
    # below any real log-sum of scores, so exp(floor - real) underflows to exactly 0.
    dp = torch.full((g, kmax + 1), -1e6, device=z.device, dtype=z.dtype)
    dp[:, 0] = 0.0

    def lae(a, b):
        m = torch.maximum(a, b)
        return m + torch.log1p(torch.exp(-(a - b).abs()))

    for j in range(n):
        vj = valid[:, j]
        if kmax == 0 or not bool(vj.any()):
            continue
        upd = lae(dp[:, 1:], dp[:, :-1] + s[:, j : j + 1])
        dp = torch.cat([dp[:, :1], torch.where(vj[:, None], upd, dp[:, 1:])], dim=1)
    logden = dp.gather(1, e.clamp(max=kmax)[:, None]).squeeze(1)
    terms = logden - (s * y).sum(-1)
    active = (e > 0) & (e < size)
    return torch.where(active, terms, torch.zeros_like(terms)).sum() / e[active].sum().clamp(min=1)


def survival_loss(score, observed_time, event, valid, *, weighted=False, end=None):
    """Within-match/cut risk sets. Standard Cox uses Breslow ties.

    The weighted variant retains upstream survival-complement terms, time-gap
    weights and per-group sum reduction. Truncation changes event eligibility
    only; full follow-up weights are retained to isolate the horizon effect.
    """
    g, n = score.shape
    ti = observed_time[:, :, None]
    tj = observed_time[:, None, :]
    risk = (tj >= ti) & valid[:, None, :]
    # A finite dummy denominator for inactive rows; they contribute zero.
    eye = torch.eye(n, device=score.device, dtype=torch.bool)[None]
    risk = risk | ((~risk.any(-1))[:, :, None] & eye)
    logits = score[:, None, :].expand(g, n, n).masked_fill(~risk, float("-inf"))
    logden = torch.logsumexp(logits, dim=-1)
    event = event & valid
    terms = logden - score
    if weighted:
        logp = score[:, None, :] - logden[:, :, None]
        probs = torch.exp(logp.masked_fill(~risk, float("-inf"))).clamp(max=1 - 1e-7)
        survivors = risk & ~eye
        complement = torch.where(
            survivors, -torch.log(1 - probs + 1e-7), torch.zeros_like(probs)
        ).sum(-1)
        weight = ((end[:, None] - observed_time) / end[:, None].clamp(min=1.0)).clamp(0, 1)
        terms = (terms + complement) * weight
        active = valid.sum(-1) >= 2
        return (
            torch.where(event, terms, torch.zeros_like(terms)).sum(-1) * active
        ).sum() / active.sum().clamp(min=1)
    return torch.where(event, terms, torch.zeros_like(terms)).sum() / event.sum().clamp(min=1)


def labels(ds, indices, batch, device, *, upstream_weighted=False):
    b, p, n = batch["target_mask"].shape
    times = torch.zeros(b, p, n)
    events = torch.zeros(b, p, n, dtype=torch.bool)
    ends = torch.zeros(b, p)
    for bi, index in enumerate(indices):
        m = ds.matches[index]
        N = len(m["team_ids"])
        death = m["team_death_s"].float()
        end = float(m["match_end_s"])
        if upstream_weighted:
            # Exactly the censoring convention of adaptation.ourfeat.t_win_of.
            # Using match_end alone also changes all time-gap weights.
            finite = death[torch.isfinite(death)]
            end = max(2000.0, end, float(finite.max()) + 1.0 if len(finite) else 0.0)
        times[bi, :, :N] = torch.nan_to_num(death, nan=end, posinf=end)[None, :]
        events[bi, :, :N] = (torch.isfinite(death) & (death <= end))[None, :]
        ends[bi] = end
    if upstream_weighted:
        times = times / 2000.0
        ends = ends / 2000.0
    return (
        times.to(device).flatten(0, 1),
        events.to(device).flatten(0, 1),
        ends.to(device).flatten(),
    )


def loss_for(objective, model, batch, ds, indices, device):
    z = model(batch)["hazard_logit"]
    if objective == "bce":
        return hazard_bce(z, batch["y_hazard"], batch["target_mask"])
    if objective == "cond_logit":
        return cond_logit_loss(z, batch["y_hazard"], batch["target_mask"])
    if objective in COMBINED_WEIGHTS:
        return loss_for_scores("cox_truncated", z, batch, ds, indices, device) + COMBINED_WEIGHTS[
            objective
        ] * hazard_bce(z, batch["y_hazard"], batch["target_mask"])
    return loss_for_scores(objective, z, batch, ds, indices, device)


def loss_for_scores(objective, z, batch, ds, indices, device):
    t, event, end = labels(
        ds, indices, batch, device, upstream_weighted=objective.startswith("weighted")
    )
    if objective.endswith("truncated"):
        event = event & batch["y_hazard"].bool().flatten(0, 1)
    if objective.startswith("weighted"):
        # Upstream winner/max-time exclusion. A horizon changes event inclusion,
        # not time weights or the risk sets of the retained earlier events.
        event = event & (t < end[:, None])
    return survival_loss(
        z.flatten(0, 1),
        t,
        event,
        batch["target_mask"].flatten(0, 1),
        weighted=objective.startswith("weighted"),
        end=end,
    )


def phase_auc(frame):
    suite = M.MetricSuite(frame)
    return suite.evaluate(np.ones(suite.n_matches), ("auc_within_phase",))["auc_within_phase"]


def calibrator_inputs(frame):
    score = -frame.rank_score.to_numpy()
    return np.column_stack([score, np.eye(7)[frame.phase.to_numpy(dtype=int) - 1]])


# Checkpoint selection: (a) validation within-phase AUC, the original rule; (b) the trained objective's own validation
# loss (OBJECTIVE_REPLICATION_PLAN_KO_2026-10-09.md).  One run can track both: each criterion keeps its own patience and
# picks its best epoch only among the epochs before its own stop, so (a) matches a run that stops on (a) alone.
SELECTIONS = {
    "within_phase_auc": ("objective_controls", "validation within-phase AUC"),
    "own_loss": ("objective_controls_ownloss", "validation loss of the trained objective"),
}


def val_loss(objective, model, ds, device):
    model.eval()
    total = steps = 0
    with torch.no_grad():
        for indices in T._chunks(np.arange(len(ds)), 8):
            batch = T._to_device(ds.get_batch(indices), device)
            total += float(loss_for(objective, model, batch, ds, indices, device))
            steps += 1
    return total / max(steps, 1)


def finish(
    model,
    state,
    prep,
    cfg,
    setting,
    objective,
    seed,
    dest,
    criterion,
    best_epoch,
    history,
    start,
    device,
):
    model.load_state_dict(state)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=state,
            config=asdict(cfg),
            standardizer=prep.standardizer.state_dict(),
            objective=objective,
            seed=seed,
        ),
        dest / "best.pt",
    )
    val = T.predict(model, prep.datasets["val"], device, 8, rank_from_hazard=True)
    cal = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs").fit(
        calibrator_inputs(val), val.y
    )
    save(
        dest / "selection.json",
        dict(
            best_epoch=best_epoch,
            criterion=criterion,
            calibration="validation logistic slope + phase intercepts; same val reused for early stopping",
            coef=cal.coef_,
            intercept=cal.intercept_,
        ),
    )
    result = {}
    for split in ("val", "test"):
        frame = (
            val.copy()
            if split == "val"
            else T.predict(model, prep.datasets[split], device, 8, rank_from_hazard=True)
        )
        result[split + "_raw"] = M.hazard_metrics(frame.y, frame.p_hazard)
        raw_phase_auc = phase_auc(frame)
        M.write_predictions(frame, dest / f"pred_{split}_raw.csv")
        frame["p_hazard"] = cal.predict_proba(calibrator_inputs(frame))[:, 1]
        # Keep raw ranking for within-phase discrimination; probability metrics
        # use the common phase-aware validation mapping.
        M.write_predictions(frame, dest / f"pred_{split}.csv")
        result[split] = M.hazard_metrics(frame.y, frame.p_hazard)
        result[split]["within_phase_auc_raw"] = raw_phase_auc
    save(
        dest / "metrics.json",
        dict(
            objective=objective,
            setting=setting,
            seed=seed,
            selection=criterion,
            history=history,
            metrics=result,
            elapsed_sec=time.perf_counter() - start,
        ),
    )


def run(prep, setting, cohort, objective, seed, device, selections=("within_phase_auc",)):
    todo = [
        s
        for s in selections
        if not (
            BASE / SELECTIONS[s][0] / cohort / objective / f"seed{seed}" / "metrics.json"
        ).exists()
    ]
    if not todo:
        return
    cfg = replace(
        prep.config, **{k: setting[k] for k in ("d_model", "n_heads", "n_layers", "dropout")}
    )
    model = build_model(cfg, "krafton", seed).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=setting["lr"], weight_decay=setting["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    stops = {s: T.EarlyStopper(3) for s in todo}
    best = {s: None for s in todo}
    done = {s: False for s in todo}
    rng = np.random.default_rng(seed)
    history = []
    start = time.perf_counter()
    for epoch in range(1, 13):
        model.train()
        total = steps = 0
        for indices in T._chunks(rng.permutation(len(prep.datasets["train"])), 8):
            batch = T._to_device(prep.datasets["train"].get_batch(indices), device)
            loss = loss_for(objective, model, batch, prep.datasets["train"], indices, device)
            assert torch.isfinite(loss), (objective, epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        scheduler.step()
        val = T.predict(model, prep.datasets["val"], device, 8, rank_from_hazard=True)
        auc = phase_auc(val)
        record = dict(epoch=epoch, train_loss=total / max(steps, 1), val_within_phase_auc=auc)
        criteria = {"within_phase_auc": -auc}
        if "own_loss" in todo:
            criteria["own_loss"] = record["val_own_loss"] = val_loss(
                objective, model, prep.datasets["val"], device
            )
        history.append(record)
        print(
            cohort,
            objective,
            seed,
            epoch,
            round(auc, 6),
            round(record.get("val_own_loss", float("nan")), 6),
            flush=True,
        )
        for s in todo:
            if done[s]:
                continue
            if stops[s].update(criteria[s]):
                best[s] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            done[s] = stops[s].should_stop
        if all(done.values()):
            break
    for s in todo:
        finish(
            model,
            best[s],
            prep,
            cfg,
            setting,
            objective,
            seed,
            BASE / SELECTIONS[s][0] / cohort / objective / f"seed{seed}",
            SELECTIONS[s][1],
            stops[s].best_epoch,
            history,
            start,
            device,
        )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=OBJECTIVES)
    ap.add_argument(
        "--select", choices=("within_phase_auc", "own_loss", "both"), default="within_phase_auc"
    )
    args = ap.parse_args()
    torch.set_num_threads(2)
    # A setting frozen before training avoids choosing the objective comparison's
    # recipe by optimizing BCE and thereby favoring BCE over Cox.
    setting = json.loads((OUT / "search_space.json").read_text())["neural_trials"][0]
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
                str(BASE / "objective_controls"),
            ]
        )
    )
    selections = ("within_phase_auc", "own_loss") if args.select == "both" else (args.select,)
    for seed in (42, 13, 2026):
        for objective in args.objectives:
            run(prep, setting, args.cohort, objective, seed, T._device(args.device), selections)


if __name__ == "__main__":
    main()
