"""PTR loss: hazard BCE over ``target_mask`` + ``rank_weight`` * pairwise logistic rank loss.

Rank direction (design doc v2.2, section 4.1)
---------------------------------------------
``final_rank_norm = final_rank / n_teams`` (``stdb/rq1_feature_etl.py:698``).  ``stdb/queries.py:658-667``
reads that column as the rank target ``y_rank`` (fallback ``float(rank) / N``) and derives
``top5 = rank <= 5`` from the same rank, so rank 1 = winner and **a smaller ``final_rank_norm`` is a better
placement**.

``rank_score`` ``s`` is a **survival-like** score: a higher ``s`` means better survival.  For every pair of
target teams of the same (match, phase) in which ``i`` finished better than ``j``
(``final_rank_norm_i < final_rank_norm_j``) the loss is ``log(1 + exp(-(s_i - s_j)))`` = ``softplus(-(s_i - s_j))``,
which pushes ``s_i > s_j``.  The ranking metrics (:mod:`model_stgcn.ptr.metrics`) use ``risk = -s``.

Reduction (``rank_reduction``): ``"pair"`` (default) is the plain mean over all valid pairs of the batch,
the literal "average over pairs" of the design doc; ``"group"`` first averages the pairs of every (match,
phase) group and then the groups (each group weighs the same, like the group-averaged metrics).  Pairs never
cross (match, phase) groups, involve only target teams, and skip ties / non-finite ranks.  Empty risk sets give
exactly 0 (finite loss and gradients); NaN labels / outputs at masked positions cannot reach the loss or its
gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

RANK_REDUCTIONS = ("pair", "group")


def hazard_bce(logit: torch.Tensor, y: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean BCE-with-logits over target rows; 0 when there are none."""
    zero = torch.zeros((), dtype=logit.dtype, device=logit.device)
    logit = torch.where(target_mask, logit, zero)
    y = torch.where(target_mask, torch.nan_to_num(y), zero)
    per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    n = target_mask.sum().clamp(min=1).to(per.dtype)
    return torch.where(target_mask, per, zero).sum() / n


def pairwise_rank_loss(
    score: torch.Tensor,
    final_rank_norm: torch.Tensor,
    target_mask: torch.Tensor,
    reduction: str = "pair",
) -> torch.Tensor:
    """``score [B,P,N]`` (higher = better survival), ``final_rank_norm [B,N]`` (smaller = better placement)."""
    if reduction not in RANK_REDUCTIONS:
        raise ValueError(f"rank reduction must be one of {RANK_REDUCTIONS}, got '{reduction}'")
    zero = torch.zeros((), dtype=score.dtype, device=score.device)
    score = torch.where(target_mask, score, zero)
    rank_ok = torch.isfinite(final_rank_norm)
    fr = torch.where(rank_ok, final_rank_norm, zero)
    diff = score.unsqueeze(-1) - score.unsqueeze(-2)  # s_i - s_j  [B,P,N,N]
    better = (
        fr[:, None, :, None] < fr[:, None, None, :]
    )  # i finished better than j (strict: no ties)
    usable = target_mask & rank_ok[:, None, :]
    valid = usable.unsqueeze(-1) & usable.unsqueeze(-2) & better
    loss = torch.where(valid, F.softplus(-diff), zero)  # log(1 + exp(-(s_i - s_j)))
    pairs = valid.sum(dim=(-1, -2))  # [B,P]
    if reduction == "pair":
        return loss.sum() / pairs.sum().clamp(min=1).to(loss.dtype)
    per_group = loss.sum(dim=(-1, -2)) / pairs.clamp(min=1).to(loss.dtype)
    has_pairs = pairs > 0
    n_groups = has_pairs.sum().clamp(min=1).to(loss.dtype)
    return torch.where(has_pairs, per_group, zero).sum() / n_groups


def ptr_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    rank_weight: float = 0.1,
    rank_reduction: str = "pair",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tm = batch["target_mask"].bool()
    bce = hazard_bce(out["hazard_logit"], batch["y_hazard"], tm)
    rank = pairwise_rank_loss(out["rank_score"], batch["final_rank_norm"], tm, rank_reduction)
    total = bce + rank_weight * rank
    return total, {"bce": bce.detach(), "rank": rank.detach(), "n_target": tm.sum().detach()}
