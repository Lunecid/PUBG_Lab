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


class SnapshotGRUNoGraphHazard(nn.Module):
    """Causal snapshot-level GRU baseline with no graph channels.

    Each team node consumes its 10s feature sequence with a unidirectional GRU.
    The hidden state gathered at each phase cut is therefore a cumulative
    summary of all observed within-phase and previous-phase behavior.
    """

    def __init__(
        self,
        in_features: int,
        *,
        hidden: int = 48,
        num_layers: int = 1,
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
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=max(int(num_layers), 1),
            batch_first=True,
            dropout=dropout if int(num_layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
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
        B, T, N, _F = X_s.shape
        mask = node_mask_s.bool()
        x = self.input(X_s.float()) * mask.unsqueeze(-1).to(X_s.dtype)
        seq = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h, _ = self.gru(seq)
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        h = self.dropout(self.norm(h)) * mask.unsqueeze(-1).to(h.dtype)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            h,
            phase_of_snap,
            mask,
            t_phase=self.t_phase,
        )
        hazard = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard, cut_mask[:, :-1]


class SnapshotLSTMNoGraphHazard(nn.Module):
    """Causal snapshot-level LSTM baseline with no graph channels."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden: int = 48,
        num_layers: int = 1,
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
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=max(int(num_layers), 1),
            batch_first=True,
            dropout=dropout if int(num_layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
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
        B, T, N, _F = X_s.shape
        mask = node_mask_s.bool()
        x = self.input(X_s.float()) * mask.unsqueeze(-1).to(X_s.dtype)
        seq = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h, _ = self.lstm(seq)
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        h = self.dropout(self.norm(h)) * mask.unsqueeze(-1).to(h.dtype)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            h,
            phase_of_snap,
            mask,
            t_phase=self.t_phase,
        )
        hazard = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard, cut_mask[:, :-1]


class SnapshotTransformerNoGraphHazard(nn.Module):
    """Causal Transformer encoder over each team's snapshot sequence."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden: int = 48,
        num_layers: int = 2,
        num_heads: int = 4,
        t_phase: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t_phase = t_phase
        heads = max(1, min(int(num_heads), hidden))
        while hidden % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=max(hidden * 4, 1),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 1), 1),
        )

    @staticmethod
    def _positional_encoding(
        length: int,
        hidden: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(hidden, 1))
        )
        pe = torch.zeros((length, hidden), device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        if hidden > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(
        self,
        X_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _F = X_s.shape
        mask = node_mask_s.bool()
        x = self.input(X_s.float())
        x = x + self._positional_encoding(
            T,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        ).view(1, T, 1, -1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        seq = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        seq_mask = mask.permute(0, 2, 1).reshape(B * N, T)
        padding_mask = ~seq_mask
        all_padded = ~seq_mask.any(dim=1)
        if bool(all_padded.any()):
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False
        causal_mask = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=X_s.device),
            diagonal=1,
        )
        h = self.encoder(
            seq,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        h = self.dropout(self.norm(h)) * mask.unsqueeze(-1).to(h.dtype)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            h,
            phase_of_snap,
            mask,
            t_phase=self.t_phase,
        )
        hazard = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard, cut_mask[:, :-1]


class SnapshotHierGRUNoGraphHazard(nn.Module):
    """Hierarchical recurrent survival baseline with no graph channels.

    The model mirrors the Battle Royale survival story:

    1. a within-phase GRUCell accumulates 10s team snapshots into one
       phase-state vector z_i(k);
    2. a phase-level GRU accumulates z_i(1..k);
    3. a hazard head predicts next-phase elimination risk at each phase cut.

    It intentionally consumes only node/snapshot features.  Relation or combat
    edge channels are outside this baseline.
    """

    def __init__(
        self,
        in_features: int,
        *,
        hidden: int = 48,
        num_layers: int = 1,
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
        self.intra_phase_cell = nn.GRUCell(hidden, hidden)
        self.phase_gru = nn.GRU(
            hidden,
            hidden,
            num_layers=max(int(num_layers), 1),
            batch_first=True,
            dropout=dropout if int(num_layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 1), 1),
        )

    def _encode_phase(
        self,
        x: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
        phase_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N, H = x.shape
        h = x.new_zeros((B, N, H))
        present = node_mask_s.new_zeros((B, N), dtype=torch.bool)
        pid = phase_of_snap.clamp(min=1, max=self.t_phase).long()
        for t in range(T):
            step_mask = node_mask_s[:, t].bool() & (pid[:, t].view(B, 1) == phase_idx)
            if not bool(step_mask.any()):
                continue
            h_next = self.intra_phase_cell(
                x[:, t].reshape(B * N, H),
                h.reshape(B * N, H),
            ).reshape(B, N, H)
            h = torch.where(step_mask.unsqueeze(-1), h_next, h)
            present = present | step_mask
        return h, present

    def forward(
        self,
        X_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _T, N, _F = X_s.shape
        x = self.input(X_s.float())

        phase_states: list[torch.Tensor] = []
        phase_node_masks: list[torch.Tensor] = []
        for phase_idx in range(1, self.t_phase + 1):
            z, present = self._encode_phase(x, phase_of_snap, node_mask_s, phase_idx)
            phase_states.append(z)
            phase_node_masks.append(present)

        z_phase = torch.stack(phase_states, dim=1)
        phase_node_mask = torch.stack(phase_node_masks, dim=1)
        z_phase = self.norm(z_phase) * phase_node_mask.unsqueeze(-1).to(z_phase.dtype)

        seq = z_phase.permute(0, 2, 1, 3).reshape(B * N, self.t_phase, -1)
        phase_h, _ = self.phase_gru(seq)
        phase_h = phase_h.reshape(B, N, self.t_phase, -1).permute(0, 2, 1, 3)
        phase_h = self.dropout(phase_h)
        phase_h = phase_h * phase_node_mask.unsqueeze(-1).to(phase_h.dtype)

        hazard = self.hazard_head(phase_h[:, :-1]).squeeze(-1)
        cut_mask = phase_node_mask.any(dim=-1)[:, :-1]
        return hazard, cut_mask
