"""Factorized V3 GNN models.

The main contract is explicit:

    X_node: team/node attributes
    G:      global phase context
    A:      dynamic team graph

The graph encoder is pluggable so the final experiment can compare no-graph,
GCN, GraphSAGE, GAT, relational GCN, and the current STGNN-style block under
the same input contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pubg_lab.survival.models.dyn_stgnn import CausalTemporalGatedConv, dynamic_stgnn_loss
from pubg_lab.survival.models.mr_stgnn import mr_survival_targets
from pubg_lab.survival.models.snapshot_encoder import SnapshotEncoder, pool_snapshots_to_phase

GRAPH_ENCODERS = ("none", "gcn", "sage", "gat", "rgcn", "mamba")
GLOBAL_FUSIONS = ("none", "concat", "add", "film")
TEAM_DECODERS = ("mlp", "attention")


def _edge_mask(
    edge_active_mask: torch.Tensor | None,
    num_edge_types: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if edge_active_mask is None:
        return torch.ones(num_edge_types, dtype=torch.bool, device=device)
    return edge_active_mask.to(device=device, dtype=torch.bool)


def _combine_adjacency(A: torch.Tensor, edge_active_mask: torch.Tensor | None) -> torch.Tensor:
    mask = _edge_mask(edge_active_mask, A.shape[2], device=A.device)
    if not bool(mask.any()):
        return A.new_zeros((A.shape[0], A.shape[1], A.shape[3], A.shape[4]))
    out = A[:, :, mask].sum(dim=2)
    rs = out.sum(dim=-1, keepdim=True)
    return torch.where(rs > 0, out / rs.clamp(min=1e-9), out)


class NodeGlobalEncoder(nn.Module):
    def __init__(
        self,
        node_features: int,
        global_features: int,
        hidden: int,
        *,
        fusion: str = "concat",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion not in GLOBAL_FUSIONS:
            raise ValueError(f"unknown global fusion: {fusion}")
        self.fusion = fusion
        self.node = nn.Sequential(nn.Linear(node_features, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.global_features = global_features
        self.global_mlp = (
            nn.Sequential(nn.Linear(global_features, hidden), nn.ReLU(), nn.Dropout(dropout))
            if global_features > 0
            else None
        )
        self.concat = (
            nn.Linear(2 * hidden, hidden) if fusion == "concat" and global_features > 0 else None
        )
        self.film = (
            nn.Linear(hidden, 2 * hidden) if fusion == "film" and global_features > 0 else None
        )

    def forward(self, X_node: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        h = self.node(X_node)
        if self.fusion == "none" or self.global_mlp is None or G.shape[-1] == 0:
            return h

        g = self.global_mlp(G).unsqueeze(2).expand(-1, -1, X_node.shape[2], -1)
        if self.fusion == "concat":
            assert self.concat is not None
            return F.relu(self.concat(torch.cat([h, g], dim=-1)))
        if self.fusion == "add":
            return h + g
        if self.fusion == "film":
            assert self.film is not None
            gamma, beta = self.film(g).chunk(2, dim=-1)
            return h * (1.0 + torch.tanh(gamma)) + beta
        raise ValueError(f"unknown global fusion: {self.fusion}")


class DenseGraphLayer(nn.Module):
    def __init__(self, hidden: int, num_edge_types: int, encoder: str) -> None:
        super().__init__()
        if encoder not in GRAPH_ENCODERS:
            raise ValueError(f"unknown graph encoder: {encoder}")
        self.encoder = encoder
        self.num_edge_types = num_edge_types
        self.skip = nn.Linear(hidden, hidden)

        if encoder == "gcn":
            self.lin = nn.Linear(hidden, hidden, bias=False)
        elif encoder == "sage":
            self.lin = nn.Linear(2 * hidden, hidden)
        elif encoder == "gat":
            self.lin = nn.Linear(hidden, hidden, bias=False)
            self.attn_src = nn.Parameter(torch.empty(hidden))
            self.attn_dst = nn.Parameter(torch.empty(hidden))
            nn.init.xavier_uniform_(self.attn_src.view(1, -1))
            nn.init.xavier_uniform_(self.attn_dst.view(1, -1))
            self.leaky_relu = nn.LeakyReLU(0.2)
        elif encoder in {"rgcn", "mamba"}:
            self.edge_linears = nn.ModuleList(
                nn.Linear(hidden, hidden, bias=False) for _ in range(num_edge_types)
            )
            self.edge_logits = nn.Parameter(torch.zeros(num_edge_types))

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B,T,N,H), A: (B,T,E,N,N)
        if self.encoder == "none":
            return self.skip(x)

        if self.encoder in {"rgcn", "mamba"}:
            return self._rgcn(x, A, edge_active_mask)

        A2 = _combine_adjacency(A, edge_active_mask)
        neigh = torch.matmul(A2, x)

        if self.encoder == "gcn":
            return self.lin(neigh) + self.skip(x)
        if self.encoder == "sage":
            return self.lin(torch.cat([x, neigh], dim=-1))
        if self.encoder == "gat":
            return self._gat(x, A2)
        raise ValueError(f"unknown graph encoder: {self.encoder}")

    def _rgcn(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mask = _edge_mask(edge_active_mask, self.num_edge_types, device=A.device)
        out = x.new_zeros(x.shape)
        if bool(mask.any()):
            logits = self.edge_logits.to(A.device).masked_fill(~mask, -1e9)
            edge_w = torch.softmax(logits, dim=0)
            for e, lin in enumerate(self.edge_linears):
                if not bool(mask[e]):
                    continue
                out = out + edge_w[e] * lin(torch.matmul(A[:, :, e], x))
        return out + self.skip(x)

    def _gat(self, x: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
        Wh = self.lin(x)
        src = (Wh * self.attn_src.to(Wh.device)).sum(dim=-1).unsqueeze(-1)
        dst = (Wh * self.attn_dst.to(Wh.device)).sum(dim=-1).unsqueeze(-2)
        scores = self.leaky_relu(src + dst)
        mask = A2 > 0
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(mask, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=-1, keepdim=True)
        attn = torch.where(denom > 0, attn / denom.clamp(min=1e-9), attn)
        return torch.matmul(attn, Wh) + self.skip(x)


class CausalMambaTemporalMixer(nn.Module):
    """Small causal selective state-space mixer over match phases.

    This keeps the V3 model self-contained while giving the Mamba comparison a
    genuinely recurrent, input-gated temporal state instead of another TCN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        state_dim: int = 16,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        inner = max(out_features * expansion, in_features)
        self.inner = inner
        self.state_dim = state_dim
        self.in_proj = nn.Linear(in_features, 2 * inner)
        self.dt_proj = nn.Linear(inner, inner)
        self.b_proj = nn.Linear(inner, state_dim)
        self.c_proj = nn.Linear(inner, state_dim)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).repeat(inner, 1)
        )
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,C).  Each team sequence is scanned causally over T.
        b, t, n, _ = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        u, gate = self.in_proj(flat).chunk(2, dim=-1)
        u = F.silu(u)
        gate = F.silu(gate)

        a = -torch.exp(self.a_log).to(device=x.device, dtype=x.dtype)
        state = flat.new_zeros((b * n, self.inner, self.state_dim))
        outs: list[torch.Tensor] = []
        for step in range(t):
            u_t = u[:, step]
            dt = F.softplus(self.dt_proj(u_t)).clamp(max=20.0)
            b_t = self.b_proj(u_t)
            c_t = self.c_proj(u_t)
            decay = torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0))
            update = dt.unsqueeze(-1) * u_t.unsqueeze(-1) * b_t.unsqueeze(1)
            state = decay * state + update
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + self.d.to(x.device, x.dtype) * u_t
            outs.append(y_t * gate[:, step])

        y = torch.stack(outs, dim=1)
        y = self.dropout(self.out_proj(y))
        return y.reshape(b, n, t, -1).permute(0, 2, 1, 3)


class GraphTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        mid: int,
        num_edge_types: int,
        *,
        graph_encoder: str,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_mamba = graph_encoder == "mamba"
        if self.use_mamba:
            self.input_proj = nn.Linear(hidden, mid)
            self.mamba = CausalMambaTemporalMixer(mid, hidden, dropout=dropout)
        else:
            self.t1 = CausalTemporalGatedConv(hidden, mid, kernel_size)
            self.t2 = CausalTemporalGatedConv(mid, hidden, kernel_size)
        self.graph = DenseGraphLayer(mid, num_edge_types, graph_encoder)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B,T,N,H)
        if self.use_mamba:
            h = F.silu(self.input_proj(x))
            h = F.relu(self.graph(h, A, edge_active_mask=edge_active_mask))
            h = self.mamba(h)
            h = self.dropout(self.norm(h))
            return F.relu(h + x)

        h = self.t1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        h = F.relu(self.graph(h, A, edge_active_mask=edge_active_mask))
        h = self.t2(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        h = self.dropout(self.norm(h))
        return F.relu(h + x)


class TeamSetDecoder(nn.Module):
    """Decode team outcomes from an encoded team set.

    The `attention` variant compares teams within the same cut.  It accepts
    either (B,N,H) tensors or cut-phase tensors (B,T,N,H), flattening the
    leading batch/time axes internally.
    """

    def __init__(
        self,
        hidden: int,
        *,
        decoder: str = "mlp",
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if decoder not in TEAM_DECODERS:
            raise ValueError(f"unknown team decoder: {decoder}")
        self.decoder = decoder
        if decoder == "attention":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            self.encoder = None

    def forward(
        self,
        pooled: torch.Tensor,
        *,
        team_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is None:
            return pooled
        original_shape = pooled.shape
        if pooled.dim() == 4:
            b, t, n, h = pooled.shape
            pooled = pooled.reshape(b * t, n, h)
            if team_mask is not None:
                team_mask = team_mask[:, None, :].expand(b, t, n).reshape(b * t, n)
        key_padding_mask = None
        if team_mask is not None:
            key_padding_mask = ~team_mask.to(dtype=torch.bool, device=pooled.device)
        out = self.encoder(pooled, src_key_padding_mask=key_padding_mask)
        if len(original_shape) == 4:
            out = out.reshape(*original_shape)
        return out


class FactorizedPhaseGNN(nn.Module):
    def __init__(
        self,
        node_features: int,
        global_features: int,
        *,
        hidden: int = 64,
        mid: int = 128,
        num_blocks: int = 2,
        num_edge_types: int = 3,
        graph_encoder: str = "rgcn",
        global_fusion: str = "concat",
        team_decoder: str = "mlp",
        decoder_heads: int = 4,
        decoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_encoder = graph_encoder
        self.global_fusion = global_fusion
        self.team_decoder_name = team_decoder
        self.input = NodeGlobalEncoder(
            node_features,
            global_features,
            hidden,
            fusion=global_fusion,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            GraphTemporalBlock(
                hidden,
                mid,
                num_edge_types,
                graph_encoder=graph_encoder,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )
        self.team_decoder = TeamSetDecoder(
            hidden,
            decoder=team_decoder,
            heads=decoder_heads,
            layers=decoder_layers,
            dropout=dropout,
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.rank_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.topk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(
        self,
        X_node: torch.Tensor,
        G: torch.Tensor,
        A: torch.Tensor,
        *,
        edge_active_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
        team_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input(X_node, G)
        for block in self.blocks:
            h = block(h, A, edge_active_mask=edge_active_mask)

        # Every supervised forecast must be causal at the cut phase.  Hazard,
        # final-rank, and top-k heads therefore read h[:, p] only when
        # predicting outcomes after cut phase p; no full-sequence pooling.
        cut_h = h[:, :-1]
        hazard_logits = self.hazard_head(cut_h).squeeze(-1)
        decoded = self.team_decoder(cut_h, team_mask=team_mask)
        rank_pred = self.rank_head(decoded).squeeze(-1)
        topk_logits = self.topk_head(decoded)
        return hazard_logits, rank_pred, topk_logits


class FactorizedMRGNN(nn.Module):
    def __init__(
        self,
        snap_features: int,
        node_features: int,
        global_features: int,
        *,
        snap_hidden: int = 48,
        snap_mid: int = 96,
        snap_blocks: int = 2,
        snap_edge_types: int = 2,
        phase_hidden: int = 64,
        phase_mid: int = 128,
        phase_blocks: int = 2,
        phase_edge_types: int = 3,
        graph_encoder: str = "rgcn",
        global_fusion: str = "concat",
        team_decoder: str = "mlp",
        decoder_heads: int = 4,
        decoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.snap_encoder = SnapshotEncoder(
            in_features=snap_features,
            hidden=snap_hidden,
            mid=snap_mid,
            num_blocks=snap_blocks,
            num_edge_types=snap_edge_types,
            dropout=dropout,
        )
        self.phase_model = FactorizedPhaseGNN(
            node_features + self.snap_encoder.out_dim,
            global_features,
            hidden=phase_hidden,
            mid=phase_mid,
            num_blocks=phase_blocks,
            num_edge_types=phase_edge_types,
            graph_encoder=graph_encoder,
            global_fusion=global_fusion,
            team_decoder=team_decoder,
            decoder_heads=decoder_heads,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )
        self.survival_head = nn.Sequential(
            nn.Linear(self.snap_encoder.out_dim, self.snap_encoder.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.snap_encoder.out_dim // 2, 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
        X_node: torch.Tensor,
        G: torch.Tensor,
        A: torch.Tensor,
        *,
        snap_edge_mask: torch.Tensor | None = None,
        phase_edge_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
        team_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        snap_emb = self.snap_encoder(X_s, A_s, edge_active_mask=snap_edge_mask)
        pooled = pool_snapshots_to_phase(
            snap_emb,
            phase_of_snap,
            node_mask_s,
            t_phase=X_node.shape[1],
        )
        X_aug = torch.cat([X_node, pooled], dim=-1)
        hazard, rank, topk = self.phase_model(
            X_aug,
            G,
            A,
            edge_active_mask=phase_edge_mask,
            phase_mask=phase_mask,
            team_mask=team_mask,
        )
        survival = self.survival_head(snap_emb).squeeze(-1)
        return hazard, survival, rank, topk


def factorized_phase_loss(
    hazard: torch.Tensor,
    rank: torch.Tensor,
    topk: torch.Tensor,
    batch: dict,
    *,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.3,
    pairwise_rank_weight: float = 0.0,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_f = batch["target_mask"].float()
    hazard_denom = target_f.sum().clamp(min=1.0)

    pos_weight = torch.tensor(
        hazard_pos_weight,
        dtype=hazard.dtype,
        device=hazard.device,
    )
    hazard_bce = F.binary_cross_entropy_with_logits(
        hazard,
        batch["y_hazard"],
        reduction="none",
        pos_weight=pos_weight,
    )
    l_hazard = (hazard_bce * target_f).sum() / hazard_denom

    y_rank = batch["y_rank"].unsqueeze(1).expand_as(rank)
    l_rank = ((rank - y_rank).abs() * target_f).sum() / hazard_denom

    y_topk = batch["y_topk"].unsqueeze(1).expand_as(topk)
    topk_bce = F.binary_cross_entropy_with_logits(topk, y_topk, reduction="none")
    l_topk = (topk_bce * target_f.unsqueeze(-1)).sum() / (hazard_denom * y_topk.size(-1))

    total = alpha * l_hazard + beta * l_rank + gamma * l_topk
    parts = {
        "loss": float(total.detach().item()),
        "l_hazard": float(l_hazard.detach().item()),
        "l_rank": float(l_rank.detach().item()),
        "l_topk": float(l_topk.detach().item()),
        "hazard_pos": float((batch["y_hazard"] * target_f).sum().detach().item()),
        "hazard_n": float(target_f.sum().detach().item()),
    }
    if pairwise_rank_weight > 0:
        pair = pairwise_forecast_rank_loss(rank, batch["y_rank"], batch["target_mask"])
        total = total + pairwise_rank_weight * pair
        parts = dict(parts)
        parts["loss"] = float(total.detach().item())
        parts["l_rank_pair"] = float(pair.detach().item())
    return total, parts


def pairwise_forecast_rank_loss(
    rank_pred: torch.Tensor,
    y_rank: torch.Tensor,
    rank_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Pairwise final-rank ordering loss at every causal cut phase."""
    pred_diff = rank_pred.unsqueeze(3) - rank_pred.unsqueeze(2)
    true_diff = y_rank[:, None, :, None] - y_rank[:, None, None, :]
    valid = rank_mask.unsqueeze(3) & rank_mask.unsqueeze(2) & (true_diff.abs() > 1e-6)
    if not bool(valid.any()):
        return rank_pred.new_tensor(0.0)
    sign = true_diff.sign()
    loss = F.softplus(-(sign * pred_diff) / temperature)
    return loss[valid].mean()


def pairwise_rank_loss(
    rank_pred: torch.Tensor,
    y_rank: torch.Tensor,
    team_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Pairwise ordering loss for final rank.

    Lower rank is better. For every valid pair, the predicted rank difference
    should have the same sign as the true rank difference.
    """
    pred_diff = rank_pred.unsqueeze(2) - rank_pred.unsqueeze(1)
    true_diff = y_rank.unsqueeze(2) - y_rank.unsqueeze(1)
    valid = team_mask.unsqueeze(2) & team_mask.unsqueeze(1) & (true_diff.abs() > 1e-6)
    if not bool(valid.any()):
        return rank_pred.new_tensor(0.0)
    sign = true_diff.sign()
    loss = F.softplus(-(sign * pred_diff) / temperature)
    return loss[valid].mean()


def factorized_mr_loss(
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
    pairwise_rank_weight: float = 0.0,
    horizon: int = 3,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    phase_total, parts = factorized_phase_loss(
        hazard,
        rank,
        topk,
        batch,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pairwise_rank_weight=pairwise_rank_weight,
        hazard_pos_weight=hazard_pos_weight,
    )
    tgt, at_risk = mr_survival_targets(
        batch["death_snap_idx"],
        batch["node_mask_s"],
        horizon=horizon,
    )
    surv_bce = F.binary_cross_entropy_with_logits(survival, tgt, reduction="none")
    l_survival = (surv_bce * at_risk).sum() / at_risk.sum().clamp(min=1.0)
    total = phase_total + delta * l_survival
    out = dict(parts)
    out["loss"] = float(total.detach().item())
    out["l_survival"] = float(l_survival.detach().item())
    out["survival_pos"] = float(tgt.sum().detach().item())
    return total, out
