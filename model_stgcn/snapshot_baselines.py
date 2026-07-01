"""Minimal no-graph snapshot baselines for pre-experiment validation.

These modules are intentionally small. They establish temporal-only baselines
before any GNN contribution is claimed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_stgcn.snapshot_phase_survival import gather_phase_cut_embeddings


class SnapshotLastOnlyHazard(nn.Module):
    """Use only the last observed snapshot at each phase cut."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden: int = 48,
        t_phase: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t_phase = t_phase
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 1), 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(X_s)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            h,
            phase_of_snap,
            node_mask_s,
            t_phase=self.t_phase,
        )
        hazard = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard, cut_mask[:, :-1]


class CausalTemporalConv(nn.Module):
    """Node-wise causal temporal convolution over 10s snapshots."""

    def __init__(
        self,
        hidden: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(hidden, hidden, kernel_size)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, C) -> apply the same causal conv to each node sequence.
        B, T, N, C = h.shape
        z = h.permute(0, 2, 3, 1).reshape(B * N, C, T)
        z = torch.nn.functional.pad(z, (self.pad, 0))
        z = self.conv(z).transpose(1, 2).reshape(B, N, T, C).permute(0, 2, 1, 3)
        return self.norm(h + self.dropout(torch.relu(z)))


class SnapshotTCNNoGraphHazard(nn.Module):
    """Temporal-only snapshot baseline with no team-to-team message passing."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden: int = 48,
        num_layers: int = 2,
        kernel_size: int = 3,
        t_phase: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t_phase = t_phase
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            CausalTemporalConv(hidden, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers)
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 1), 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input(X_s)
        for layer in self.layers:
            h = layer(h)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            h,
            phase_of_snap,
            node_mask_s,
            t_phase=self.t_phase,
        )
        hazard = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard, cut_mask[:, :-1]
