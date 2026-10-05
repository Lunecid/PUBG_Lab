"""PTR loss: hazard BCE over target_mask + 0.1 * pairwise logistic rank loss per (match, phase).

Rank direction (design doc v2.2 section 4.1; documented in ``model_stgcn/ptr/loss.py``):
``final_rank_norm = final_rank / n_teams`` (stdb/rq1_feature_etl.py:698; stdb/queries.py:658-667 reads it as
``y_rank`` with the fallback ``rank / N`` and ``top5 = rank <= 5``), so a SMALLER value is a better placement.
``rank_score`` s is survival-like (higher = better survival); every pair (i better than j) contributes
``log(1 + exp(-(s_i - s_j)))``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from model_stgcn.ptr.loss import hazard_bce, pairwise_rank_loss, ptr_loss


def softplus(x):
    return math.log1p(math.exp(x))


def test_rank_loss_matches_hand_computation_for_both_reductions():
    # one match, two phases; fr = final_rank_norm (small = good placement); team 0 finishes best
    fr = torch.tensor([[0.2, 0.5, 0.9]])
    score = torch.tensor([[[3.0, 1.0, 0.0], [0.5, 9.0, -1.0]]])  # [B=1, P=2, N=3], higher = survives longer
    tm = torch.tensor([[[True, True, True], [True, False, True]]])
    # phase 0: pairs (better, worse) = (0,1) (0,2) (1,2) with s_i - s_j = 2, 3, 1
    phase0 = [softplus(-2.0), softplus(-3.0), softplus(-1.0)]
    # phase 1: only teams 0 and 2 are targets; team 0 is better: 0.5 - (-1.0) = 1.5
    phase1 = [softplus(-1.5)]
    pair_mean = (sum(phase0) + sum(phase1)) / 4  # literal "average over pairs"
    group_mean = (sum(phase0) / 3 + sum(phase1) / 1) / 2  # every (match, phase) weighs the same
    assert float(pairwise_rank_loss(score, fr, tm)) == pytest.approx(pair_mean, rel=1e-5)
    assert float(pairwise_rank_loss(score, fr, tm, "pair")) == pytest.approx(pair_mean, rel=1e-5)
    assert float(pairwise_rank_loss(score, fr, tm, "group")) == pytest.approx(group_mean, rel=1e-5)
    assert pair_mean != pytest.approx(group_mean, rel=1e-3)
    with pytest.raises(ValueError):
        pairwise_rank_loss(score, fr, tm, "mean")


def test_rank_loss_direction_better_placement_should_score_higher():
    fr = torch.tensor([[0.1, 0.4, 0.7, 1.0]])  # team 0 won, team 3 went out first
    tm = torch.ones(1, 1, 4, dtype=torch.bool)
    aligned = torch.tensor([[[2.0, 1.0, -1.0, -2.0]]])  # decreasing with final_rank_norm
    reversed_ = -aligned
    for reduction in ("pair", "group"):
        assert pairwise_rank_loss(aligned, fr, tm, reduction) < pairwise_rank_loss(reversed_, fr, tm, reduction)
    # descent raises the winner's score and lowers the first-eliminated team's score
    s = aligned.clone().requires_grad_(True)
    pairwise_rank_loss(s, fr, tm).backward()
    assert s.grad[0, 0, 0] < 0 and s.grad[0, 0, 3] > 0


def test_rank_loss_ignores_ties_and_non_targets_and_has_no_cross_group_pairs():
    fr = torch.tensor([[0.5, 0.5, 0.9]])  # teams 0 and 1 tie, team 2 finished worst
    tm = torch.tensor([[[True, True, True]]])
    score = torch.tensor([[[1.0, 0.0, 5.0]]])
    # valid pairs (better, worse): (0,2) -> s0 - s2 = -4 and (1,2) -> s1 - s2 = -5; the tied pair is skipped
    expect = (softplus(4.0) + softplus(5.0)) / 2
    assert float(pairwise_rank_loss(score, fr, tm)) == pytest.approx(expect, rel=1e-5)
    # a non-target team never contributes, even with an extreme score
    tm2 = torch.tensor([[[True, False, True]]])
    score2 = torch.tensor([[[1.0, 1e6, 5.0]]])
    assert float(pairwise_rank_loss(score2, fr, tm2)) == pytest.approx(softplus(4.0), rel=1e-5)
    # pairs never cross phases: a single target per phase gives no pair at all
    tm3 = torch.tensor([[[True, False, False], [False, False, True]]])
    score3 = torch.zeros(1, 2, 3)
    for reduction in ("pair", "group"):
        assert float(pairwise_rank_loss(score3, fr, tm3, reduction)) == 0.0
    # pairs never cross matches either: two matches, one target team each -> no pair
    fr2 = torch.tensor([[0.2, 0.9], [0.9, 0.2]])
    tm4 = torch.tensor([[[True, False]], [[False, True]]])
    assert float(pairwise_rank_loss(torch.randn(2, 1, 2), fr2, tm4)) == 0.0


def test_empty_risk_set_gives_finite_zero_loss_and_finite_gradients():
    B, P, N = 2, 3, 5
    logit = torch.randn(B, P, N, requires_grad=True)
    score = torch.randn(B, P, N, requires_grad=True)
    batch = {
        "target_mask": torch.zeros(B, P, N, dtype=torch.bool),
        "y_hazard": torch.ones(B, P, N),
        "final_rank_norm": torch.rand(B, N),
    }
    for reduction in ("pair", "group"):
        loss, parts = ptr_loss({"hazard_logit": logit, "rank_score": score}, batch, rank_reduction=reduction)
        assert torch.isfinite(loss) and float(loss.detach()) == 0.0
        assert float(parts["n_target"]) == 0
        loss.backward()
        assert torch.isfinite(logit.grad).all() and torch.isfinite(score.grad).all()
    assert float(hazard_bce(logit.detach(), batch["y_hazard"], batch["target_mask"])) == 0.0
    # targets exist but no pair does (one target per group): rank part is exactly zero, BCE is not
    batch["target_mask"][:, :, 0] = True
    loss, parts = ptr_loss({"hazard_logit": logit.detach(), "rank_score": score.detach()}, batch)
    assert float(parts["rank"]) == 0.0 and float(parts["bce"]) > 0


def test_nan_labels_and_nan_outputs_at_masked_positions_do_not_poison_loss_or_grads():
    B, P, N = 2, 3, 6
    g = torch.Generator().manual_seed(0)
    tm = torch.rand(B, P, N, generator=g) < 0.5
    y = (torch.rand(B, P, N, generator=g) < 0.3).float()
    y[~tm] = float("nan")
    fr = torch.rand(B, N, generator=g)
    fr[0, 5] = float("nan")
    tm[0, :, 5] = False  # NaN rank only where the team is not a target
    logit = torch.randn(B, P, N, generator=g)
    score = torch.randn(B, P, N, generator=g)
    logit[~tm] = float("nan")
    score[~tm] = float("nan")
    logit.requires_grad_(True)
    score.requires_grad_(True)
    loss, _ = ptr_loss(
        {"hazard_logit": logit, "rank_score": score},
        {"target_mask": tm, "y_hazard": y, "final_rank_norm": fr},
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logit.grad).all() and torch.isfinite(score.grad).all()
    assert (logit.grad[~tm] == 0).all() and (score.grad[~tm] == 0).all()
    # a target team whose final rank is unknown drops out of the rank pairs instead of ranking as "best"
    tm = torch.ones(1, 1, 3, dtype=torch.bool)
    fr = torch.tensor([[0.3, float("nan"), 0.9]])
    score = torch.tensor([[[1.0, 100.0, 0.0]]])
    assert float(pairwise_rank_loss(score, fr, tm)) == pytest.approx(softplus(-1.0), rel=1e-5)


def test_total_loss_is_bce_plus_point_one_rank():
    B, P, N = 2, 3, 5
    g = torch.Generator().manual_seed(1)
    tm = torch.rand(B, P, N, generator=g) < 0.7
    batch = {
        "target_mask": tm,
        "y_hazard": (torch.rand(B, P, N, generator=g) < 0.4).float(),
        "final_rank_norm": torch.rand(B, N, generator=g),
    }
    out = {"hazard_logit": torch.randn(B, P, N, generator=g), "rank_score": torch.randn(B, P, N, generator=g)}
    loss, parts = ptr_loss(out, batch)
    manual_bce = F.binary_cross_entropy_with_logits(out["hazard_logit"][tm], batch["y_hazard"][tm])
    assert float(parts["bce"]) == pytest.approx(float(manual_bce), rel=1e-5)
    rank = pairwise_rank_loss(out["rank_score"], batch["final_rank_norm"], tm)
    assert float(parts["rank"]) == pytest.approx(float(rank), rel=1e-5)
    assert float(loss) == pytest.approx(float(manual_bce) + 0.1 * float(rank), rel=1e-5)
    loss0, _ = ptr_loss(out, batch, rank_weight=0.0)
    assert float(loss0) == pytest.approx(float(manual_bce), rel=1e-5)
    loss_g, parts_g = ptr_loss(out, batch, rank_reduction="group")
    rank_g = pairwise_rank_loss(out["rank_score"], batch["final_rank_norm"], tm, "group")
    assert float(parts_g["rank"]) == pytest.approx(float(rank_g), rel=1e-5)
    assert float(loss_g) == pytest.approx(float(manual_bce) + 0.1 * float(rank_g), rel=1e-5)
