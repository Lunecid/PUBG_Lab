"""Snapshot-resolution ST survival model with phase-cut outputs.

This is the model contract we actually want for a minimum-resolution ST setup:

    X_s/A_s at every 10s snapshot -> causal snapshot encoder -> phase-cut heads

The phase index is used only to choose output cut points.  It is not used to
average raw features into a phase-level input tensor before the ST encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_stgcn.snapshot_encoder import SnapshotEncoder


def phase_cut_indices(
    phase_of_snap: torch.Tensor,
    node_mask_s: torch.Tensor,
    *,
    t_phase: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return last observed snapshot index for each phase cut.

    For cut phase k, the selected snapshot is the last real snapshot whose
    clipped phase id is <= k.  phase_of_snap==0 means pre-phase/landing and is
    treated as phase-1 context.  Padded snapshots are excluded through
    node_mask_s.any(-1).
    """
    if phase_of_snap.dim() != 2:
        raise ValueError("phase_of_snap must have shape (B, T_s)")
    if node_mask_s.dim() != 3:
        raise ValueError("node_mask_s must have shape (B, T_s, N)")
    if phase_of_snap.shape[:2] != node_mask_s.shape[:2]:
        raise ValueError("phase_of_snap and node_mask_s disagree on B/T_s")

    B, T_s = phase_of_snap.shape
    device = phase_of_snap.device
    valid_snap = node_mask_s.any(dim=-1)
    pid = phase_of_snap.clamp(min=1, max=t_phase).long()
    time_idx = torch.arange(T_s, device=device).view(1, T_s).expand(B, T_s)

    idxs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    neg_one = time_idx.new_full((B, T_s), -1)
    for phase in range(1, t_phase + 1):
        eligible = valid_snap & (pid <= phase)
        idx = torch.where(eligible, time_idx, neg_one).amax(dim=1)
        masks.append(idx >= 0)
        idxs.append(idx.clamp(min=0))

    return torch.stack(idxs, dim=1), torch.stack(masks, dim=1)


def gather_phase_cut_embeddings(
    snap_emb: torch.Tensor,
    phase_of_snap: torch.Tensor,
    node_mask_s: torch.Tensor,
    *,
    t_phase: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather snapshot embeddings at each causal phase cut."""
    if snap_emb.dim() != 4:
        raise ValueError("snap_emb must have shape (B, T_s, N, C)")
    B, _T_s, N, C = snap_emb.shape
    cut_idx, cut_mask = phase_cut_indices(
        phase_of_snap,
        node_mask_s,
        t_phase=t_phase,
    )
    gather_idx = cut_idx.view(B, t_phase, 1, 1).expand(B, t_phase, N, C)
    cuts = snap_emb.gather(dim=1, index=gather_idx)
    cuts = cuts * cut_mask.to(dtype=snap_emb.dtype).view(B, t_phase, 1, 1)
    return cuts, cut_mask


class SnapshotPhaseSurvivalGNN(nn.Module):
    """Causal 10s-snapshot STGNN that predicts next-phase elimination hazard."""

    def __init__(
        self,
        snap_in_features: int,
        *,
        snap_hidden: int = 48,
        snap_mid: int = 96,
        snap_blocks: int = 2,
        snap_edge_types: int = 2,
        t_phase: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t_phase = t_phase
        self.snap_encoder = SnapshotEncoder(
            in_features=snap_in_features,
            hidden=snap_hidden,
            mid=snap_mid,
            num_blocks=snap_blocks,
            num_edge_types=snap_edge_types,
            dropout=dropout,
        )
        head_hidden = max(snap_hidden // 2, 1)
        self.hazard_head = nn.Sequential(
            nn.Linear(self.snap_encoder.out_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
        *,
        edge_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hazard logits and cut validity.

        hazard_logits[:, k] predicts elimination during phase k+2 from all
        10s snapshots observed through phase k+1.
        """
        snap_emb = self.snap_encoder(X_s, A_s, edge_active_mask=edge_active_mask)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            snap_emb,
            phase_of_snap,
            node_mask_s,
            t_phase=self.t_phase,
        )
        hazard_logits = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard_logits, cut_mask[:, :-1]

    def survival_probability(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
        *,
        edge_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hazard_logits, cut_mask = self(
            X_s,
            A_s,
            phase_of_snap,
            node_mask_s,
            edge_active_mask=edge_active_mask,
        )
        return 1.0 - torch.sigmoid(hazard_logits), cut_mask


def snapshot_phase_hazard_loss(
    hazard_logits: torch.Tensor,
    y_hazard: torch.Tensor,
    target_mask: torch.Tensor,
    cut_mask: torch.Tensor | None = None,
    *,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked BCE over phase-cut next-phase hazard labels."""
    mask = target_mask.bool()
    if cut_mask is not None:
        mask = mask & cut_mask.unsqueeze(-1).bool()
    mask_f = mask.to(dtype=hazard_logits.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    pos_weight = torch.tensor(
        hazard_pos_weight,
        dtype=hazard_logits.dtype,
        device=hazard_logits.device,
    )
    bce = F.binary_cross_entropy_with_logits(
        hazard_logits,
        y_hazard.to(dtype=hazard_logits.dtype),
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = (bce * mask_f).sum() / denom
    with torch.no_grad():
        probs = torch.sigmoid(hazard_logits)
        pred = probs >= 0.5
        truth = y_hazard >= 0.5
        correct = ((pred == truth) & mask).sum()
        acc = correct.to(dtype=hazard_logits.dtype) / denom
        positives = (y_hazard.to(dtype=hazard_logits.dtype) * mask_f).sum()
    return loss, {
        "loss": float(loss.detach().item()),
        "hazard_acc": float(acc.detach().item()),
        "hazard_pos": float(positives.detach().item()),
        "hazard_n": float(denom.detach().item()),
    }
