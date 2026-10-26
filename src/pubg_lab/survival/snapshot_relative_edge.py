"""Snapshot relative-edge hazard model.

This pilot tests whether contact-gated pairwise relative state carries signal
beyond interaction topology.  It derives edge features from existing snapshot
node state and snapshot relation gates, then uses edge-conditioned message
passing before the phase-cut hazard head.

The default ``observed`` gate preserves the original pilot behavior.  The
``knn_combat`` gate is the sparse snapshot-first follow-up: each alive source
team keeps its k nearest alive teams by current position, with combat edges
always retained.  The ``feature_roll`` null preserves that sparse mask but
deterministically rolls edge features across edges within each snapshot.  The
``topology_roll`` null preserves each source row's degree but rolls destinations
within the alive candidate set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pubg_lab.survival.snapshot_baselines import CausalTemporalConv
from pubg_lab.survival.snapshot_phase_survival import gather_phase_cut_embeddings


class EdgeConditionedSnapshotLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        *,
        edge_hidden: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden, hidden)
        self.msg_proj = nn.Linear(hidden, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1),
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        # h: (B, T, N, C), edge_feat: (B, T, N, N, F)
        logits = self.edge_mlp(edge_feat).squeeze(-1)
        logits = logits.masked_fill(~edge_mask.bool(), -30.0)
        attn = torch.softmax(logits, dim=-1)
        attn = attn * edge_mask.any(dim=-1, keepdim=True).to(dtype=attn.dtype)
        msg = torch.matmul(attn, self.msg_proj(h))
        z = torch.relu(self.self_proj(h) + msg)
        return self.norm(h + self.dropout(z))


class SnapshotRelativeEdgeHazard(nn.Module):
    """Contact-gated relative-advantage snapshot pilot."""

    edge_feature_names = [
        "prox_weight",
        "combat_weight",
        "contact_gate",
        "delta_z",
        "delta_alive",
        "delta_health",
        "delta_blue_boundary",
        "delta_target_center",
        "delta_rel_zone_x",
        "delta_rel_zone_y",
    ]

    def __init__(
        self,
        in_features: int,
        *,
        feature_names: list[str],
        hidden: int = 48,
        num_layers: int = 1,
        edge_hidden: int = 16,
        kernel_size: int = 3,
        t_phase: int = 8,
        dropout: float = 0.1,
        relative_edge_gate: str = "observed",
        relative_edge_k: int = 5,
        relative_edge_null: str = "none",
    ) -> None:
        super().__init__()
        if relative_edge_gate not in {"observed", "knn_combat"}:
            raise ValueError(f"unknown relative_edge_gate={relative_edge_gate}")
        if relative_edge_null not in {"none", "feature_roll", "topology_roll"}:
            raise ValueError(f"unknown relative_edge_null={relative_edge_null}")
        self.t_phase = t_phase
        self.relative_edge_gate = relative_edge_gate
        self.relative_edge_k = int(relative_edge_k)
        self.relative_edge_null = relative_edge_null
        self.feature_names = list(feature_names)
        self.feature_index = {name: i for i, name in enumerate(feature_names)}
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        edge_dim = len(self.edge_feature_names)
        self.edge_layers = nn.ModuleList(
            EdgeConditionedSnapshotLayer(
                hidden,
                edge_dim,
                edge_hidden=edge_hidden,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.temporal_layers = nn.ModuleList(
            CausalTemporalConv(hidden, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers)
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 1), 1),
        )

    def _feature(self, X_s: torch.Tensor, name: str) -> torch.Tensor:
        idx = self.feature_index.get(name)
        if idx is None:
            return X_s.new_zeros(X_s.shape[:-1])
        return X_s[..., idx]

    def _knn_combat_mask(
        self,
        X_s: torch.Tensor,
        combat: torch.Tensor,
        alive_block: torch.Tensor,
        eye: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse directed gate: per-source kNN in xy plus all combat edges."""
        candidate = alive_block & ~eye
        combat_gate = (combat > 0) & candidate
        k = max(0, min(self.relative_edge_k, max(int(X_s.shape[-2]) - 1, 0)))
        if k == 0:
            return combat_gate

        x = self._feature(X_s, "pos_x")
        y = self._feature(X_s, "pos_y")
        dx = x[:, :, :, None] - x[:, :, None, :]
        dy = y[:, :, :, None] - y[:, :, None, :]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-12)
        inf = torch.full_like(dist, float("inf"))
        masked_dist = torch.where(candidate, dist, inf)
        vals, idx = torch.topk(masked_dist, k, dim=-1, largest=False)
        finite = torch.isfinite(vals)
        knn = torch.zeros_like(candidate)
        knn.scatter_(-1, idx, finite)
        return (knn | combat_gate) & candidate

    def _apply_edge_topology_null(
        self,
        edge_mask: torch.Tensor,
        alive_block: torch.Tensor,
        eye: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve per-row degree while breaking destination semantics."""
        if self.relative_edge_null != "topology_roll":
            return edge_mask
        out = torch.zeros_like(edge_mask)
        candidate = alive_block & ~eye
        bsz, t_max, n_max = edge_mask.shape[:3]
        for b in range(bsz):
            for t in range(t_max):
                for src in range(n_max):
                    dst = torch.nonzero(edge_mask[b, t, src], as_tuple=False).flatten()
                    if dst.numel() == 0:
                        continue
                    cand = torch.nonzero(candidate[b, t, src], as_tuple=False).flatten()
                    if cand.numel() <= 1:
                        out[b, t, src, dst] = True
                        continue
                    shift = max(int(cand.numel()) // 2, 1)
                    match = cand[:, None] == dst[None, :]
                    pos = torch.argmax(match.to(torch.int64), dim=0)
                    rolled = cand[(pos + shift) % cand.numel()]
                    out[b, t, src, rolled] = True
        return out

    def _apply_edge_feature_null(
        self,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve sparse topology but break feature-to-edge semantics."""
        if self.relative_edge_null == "none":
            return edge_feat
        out = edge_feat.clone()
        feature_cols = torch.tensor(
            [0, 1, 3, 4, 5, 6, 7, 8, 9],
            dtype=torch.long,
            device=edge_feat.device,
        )
        bsz, t_max = edge_mask.shape[:2]
        for b in range(bsz):
            for t in range(t_max):
                idx = torch.nonzero(edge_mask[b, t], as_tuple=False)
                if idx.shape[0] <= 1:
                    continue
                vals = edge_feat[b, t, idx[:, 0], idx[:, 1]][:, feature_cols]
                replacement = out[b, t, idx[:, 0], idx[:, 1]].clone()
                replacement[:, feature_cols] = vals.roll(1, dims=0)
                out[b, t, idx[:, 0], idx[:, 1]] = replacement
        return out

    def build_edge_features(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        node_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if A_s.dim() != 5:
            raise ValueError("A_s must have shape (B, T_s, E, N, N)")
        prox = A_s[:, :, 0]
        combat = A_s[:, :, 1] if A_s.shape[2] > 1 else torch.zeros_like(prox)
        alive = node_mask_s.bool()
        alive_block = alive[:, :, :, None] & alive[:, :, None, :]
        N = A_s.shape[-1]
        eye = torch.eye(N, dtype=torch.bool, device=A_s.device).view(1, 1, N, N)
        observed_mask = ((prox > 0) | (combat > 0)) & alive_block & ~eye
        if self.relative_edge_gate == "observed":
            edge_mask = observed_mask
        else:
            edge_mask = self._knn_combat_mask(X_s, combat, alive_block, eye)
        edge_mask = self._apply_edge_topology_null(edge_mask, alive_block, eye)

        def delta(name: str) -> torch.Tensor:
            v = self._feature(X_s, name)
            return v[:, :, :, None] - v[:, :, None, :]

        contact = edge_mask.to(dtype=X_s.dtype)
        edge_feat = torch.stack(
            [
                prox,
                combat,
                contact,
                delta("pos_z"),
                delta("alive_n"),
                delta("mean_health"),
                delta("dist_blue_boundary_m"),
                delta("dist_target_center_m"),
                delta("rel_zone_x"),
                delta("rel_zone_y"),
            ],
            dim=-1,
        )
        edge_feat = edge_feat * contact.unsqueeze(-1)
        edge_feat = self._apply_edge_feature_null(edge_feat, edge_mask)
        return edge_feat, edge_mask

    def forward(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input(X_s)
        edge_feat, edge_mask = self.build_edge_features(X_s, A_s, node_mask_s)
        for edge_layer, temporal_layer in zip(self.edge_layers, self.temporal_layers):
            h = edge_layer(h, edge_feat, edge_mask)
            h = temporal_layer(h)
            h = h * node_mask_s.to(dtype=h.dtype).unsqueeze(-1)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            h,
            phase_of_snap,
            node_mask_s,
            t_phase=self.t_phase,
        )
        hazard = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard, cut_mask[:, :-1]
