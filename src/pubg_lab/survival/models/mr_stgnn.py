"""
Multi-Resolution hierarchical ST-GNN (MR-STGNN).

  Tier-S (snapshot 10s)  ── SnapshotEncoder ──► snapshot embeddings (B,T_s,N,Cs)
        │ pool_snapshots_to_phase (phase-local masked mean)
        ▼
  phase node repr (B,T_p,N,Cs)  ─concat─  phase aggregate features X_p (F_p)
        │ DynamicSTGNN (phase tier, causal over phases) — REUSED
        ▼
  hazard_phase (B,T_p-1,N)        ← main label (k→k+1)
  survival_snapshot (B,T_s,N)     ← deep-supervision aux label (per 10s)

Causality / no leak:
  * snapshot encoder is left-pad causal (embedding at t sees <= t).
  * phase pooling is phase-local; cross-phase cumulative dependence is the
    phase tier's causal temporal conv (hidden at phase k sees <= k).
  * adjacencies are row-stochastic (validated by the reused DynamicSTGNN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pubg_lab.survival.models.dyn_stgnn import DynamicSTGNN, dynamic_stgnn_loss
from pubg_lab.survival.models.snapshot_encoder import SnapshotEncoder, pool_snapshots_to_phase


class MRSTGNN(nn.Module):
    def __init__(
        self,
        snap_in_features: int,
        phase_in_features: int,
        *,
        snap_hidden: int = 48,
        snap_mid: int = 96,
        snap_blocks: int = 2,
        snap_edge_types: int = 2,
        phase_hidden: int = 64,
        phase_mid: int = 128,
        phase_blocks: int = 2,
        phase_edge_types: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.snap_encoder = SnapshotEncoder(
            in_features=snap_in_features,
            hidden=snap_hidden,
            mid=snap_mid,
            num_blocks=snap_blocks,
            num_edge_types=snap_edge_types,
            dropout=dropout,
        )
        # phase tier consumes [phase features ; pooled snapshot embedding]
        self.phase_model = DynamicSTGNN(
            in_features=phase_in_features + self.snap_encoder.out_dim,
            hidden=phase_hidden,
            mid=phase_mid,
            num_blocks=phase_blocks,
            num_edge_types=phase_edge_types,
            dropout=dropout,
        )
        # snapshot-level survival hazard head (deep supervision)
        self.survival_head = nn.Sequential(
            nn.Linear(self.snap_encoder.out_dim, self.snap_encoder.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.snap_encoder.out_dim // 2, 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,  # (B, T_s, N, F_s)
        A_s: torch.Tensor,  # (B, T_s, E_s, N, N)
        phase_of_snap: torch.Tensor,  # (B, T_s)
        node_mask_s: torch.Tensor,  # (B, T_s, N)
        X_p: torch.Tensor,  # (B, T_p, N, F_p)
        A_p: torch.Tensor,  # (B, T_p, E_p, N, N)
        *,
        snap_edge_mask: torch.Tensor | None = None,
        phase_edge_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_phase = X_p.shape[1]
        snap_emb = self.snap_encoder(X_s, A_s, edge_active_mask=snap_edge_mask)  # (B,T_s,N,Cs)
        pooled = pool_snapshots_to_phase(
            snap_emb, phase_of_snap, node_mask_s, t_phase=t_phase
        )  # (B,T_p,N,Cs)
        X_p_aug = torch.cat([X_p, pooled], dim=-1)  # (B,T_p,N,F_p+Cs)
        hazard, rank, topk = self.phase_model(
            X_p_aug, A_p, edge_active_mask=phase_edge_mask, phase_mask=phase_mask
        )
        survival = self.survival_head(snap_emb).squeeze(-1)  # (B,T_s,N)
        return hazard, survival, rank, topk


def mr_survival_targets(
    death_snap_idx: torch.Tensor,  # (B, N) int, -1 = censored/absent
    node_mask_s: torch.Tensor,  # (B, T_s, N) bool
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build snapshot-level discrete-time hazard targets + at-risk mask.

    For each alive (b, t, team): label = 1 if the team's death snapshot index
    falls in the PAST-causal-safe future window (t, t+horizon], else 0; at risk
    iff the team is alive at t (node_mask_s) and not already dead. The input
    features for snapshot t are <= t (encoder causal), the LABEL looks ahead —
    standard discrete-time survival, no input leak.
    """
    B, T_s, N = node_mask_s.shape
    device = node_mask_s.device
    t_idx = torch.arange(T_s, device=device).view(1, T_s, 1)  # (1,T_s,1)
    death = death_snap_idx.view(B, 1, N)  # (B,1,N)
    has_death = death >= 0
    # at risk: alive at t (node present) — node_mask already means alive
    at_risk = node_mask_s
    # event within (t, t+horizon]
    within = has_death & (death > t_idx) & (death <= t_idx + horizon)
    target = (within & at_risk).to(torch.float32)
    return target, at_risk.to(torch.float32)


def mr_loss(
    hazard: torch.Tensor,
    survival: torch.Tensor,
    rank: torch.Tensor,
    topk: torch.Tensor,
    batch: dict,
    *,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.3,
    delta: float = 0.3,
    horizon: int = 3,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MR-STGNN loss = phase loss (hazard+rank+topk) + delta * snapshot survival.

    Phase part reuses the tuned phase-tier loss verbatim, so the phase-only
    baseline (delta=0) is numerically identical to the current pipeline.
    """
    phase_total, parts = dynamic_stgnn_loss(
        hazard,
        rank,
        topk,
        batch["y_hazard"],
        batch["target_mask"],
        batch["y_rank"],
        batch["y_topk"],
        batch["team_mask"],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        hazard_pos_weight=hazard_pos_weight,
    )

    tgt, at_risk = mr_survival_targets(
        batch["death_snap_idx"], batch["node_mask_s"], horizon=horizon
    )
    surv_bce = F.binary_cross_entropy_with_logits(survival, tgt, reduction="none")
    surv_denom = at_risk.sum().clamp(min=1.0)
    l_survival = (surv_bce * at_risk).sum() / surv_denom

    total = phase_total + delta * l_survival
    parts = dict(parts)
    parts["loss"] = float(total.detach().item())
    parts["l_survival"] = float(l_survival.detach().item())
    parts["survival_pos"] = float(tgt.sum().detach().item())
    return total, parts
