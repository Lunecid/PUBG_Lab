"""
Snapshot (10s) encoder + phase pooling for the MR-STGNN Tier-S.

Reuses the phase tier's DynamicSTBlock (causal temporal-gated conv + dense
heterogeneous spatial GCN) over the full snapshot sequence, then pools each
snapshot embedding into its phase to hand a per-phase vector to the phase
tier. Causality: DynamicSTBlock's temporal conv is left-pad causal, so the
embedding at snapshot t observes only snapshots <= t. Phase pooling is
phase-local (snapshots whose phase_of_snap == k), and cumulative dependence
across phases is then handled by the phase tier's own causal convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pubg_lab.survival.dyn_stgnn import DynamicSTBlock


class SnapshotEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 48,
        mid: int = 96,
        num_blocks: int = 2,
        num_edge_types: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            DynamicSTBlock(hidden, mid, hidden, num_edge_types, dropout=dropout)
            for _ in range(num_blocks)
        )
        self.out_dim = hidden

    def forward(
        self,
        X_s: torch.Tensor,  # (B, T_s, N, F)
        A_s: torch.Tensor,  # (B, T_s, E, N, N)
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (B, T_s, N, C)
        h = self.input(X_s).permute(0, 3, 1, 2)  # (B, C, T_s, N)
        for block in self.blocks:
            h = block(h, A_s, edge_active_mask=edge_active_mask)
        return h.permute(0, 2, 3, 1)  # (B, T_s, N, C)


def pool_snapshots_to_phase(
    snap_emb: torch.Tensor,  # (B, T_s, N, C)
    phase_of_snap: torch.Tensor,  # (B, T_s) int, 0 = landing/pre-phase
    node_mask_s: torch.Tensor,  # (B, T_s, N) bool
    *,
    t_phase: int = 8,
) -> torch.Tensor:  # (B, t_phase, N, C)
    """Masked mean of snapshot embeddings within each phase.

    phase_of_snap==0 (landing, before phase 1) folds into phase 1 — it is past
    context for the first cut. Output index p corresponds to phase_idx p+1
    (1..t_phase), matching the phase tier's tensor layout.
    """
    B, T_s, N, C = snap_emb.shape
    device = snap_emb.device
    # phase id per snapshot, clipped into [1, t_phase], then to 0-based slot
    pid = phase_of_snap.clamp(min=1, max=t_phase).long() - 1  # (B, T_s) in [0, t_phase-1]
    m = node_mask_s.to(snap_emb.dtype).unsqueeze(-1)  # (B, T_s, N, 1)
    masked = snap_emb * m

    out = snap_emb.new_zeros((B, t_phase, N, C))
    denom = snap_emb.new_zeros((B, t_phase, N, 1))
    idx = pid.view(B, T_s, 1, 1).expand(B, T_s, N, C)
    out.scatter_add_(1, idx, masked)
    denom.scatter_add_(1, pid.view(B, T_s, 1, 1).expand(B, T_s, N, 1), m)
    return out / denom.clamp(min=1.0)
