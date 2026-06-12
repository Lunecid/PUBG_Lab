"""
Dynamic Spatio-Temporal GNN for phase-level PUBG team survival modeling.

Input:
    X : (B, T=8, N<=128 padded, F)
    A : (B, T=8, E=3, N<=128 padded, N<=128 padded)

Outputs:
    hazard_logits : (B, T-1=7, N)    next-phase elimination hazard
    rank_pred     : (B, N)           normalized final-rank auxiliary
    topk_logits   : (B, N, 2)        [top5, top10] auxiliary logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalGatedConv(nn.Module):
    """Causal 1D temporal convolution with GLU activation."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv2d(
            c_in,
            2 * c_out,
            kernel_size=(kernel_size, 1),
            padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        x = F.pad(x, (0, 0, self.left_pad, 0))
        h = self.conv(x)
        a, b = h.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class HeterogeneousSpatialGCN(nn.Module):
    """Dense per-phase graph conv over multiple edge types."""

    def __init__(self, c_in: int, c_out: int, num_edge_types: int) -> None:
        super().__init__()
        self.edge_linears = nn.ModuleList(
            nn.Linear(c_in, c_out, bias=False) for _ in range(num_edge_types)
        )
        self.edge_logits = nn.Parameter(torch.zeros(num_edge_types))
        self.skip = nn.Linear(c_in, c_out)

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, C, T, N), A: (B, T, E, N, N)
        xp = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        if edge_active_mask is None:
            edge_active_mask = torch.ones_like(self.edge_logits, dtype=torch.bool)
        edge_active_mask = edge_active_mask.to(device=self.edge_logits.device, dtype=torch.bool)
        out = torch.zeros(
            (*xp.shape[:-1], self.skip.out_features),
            dtype=xp.dtype,
            device=xp.device,
        )
        if edge_active_mask.any():
            masked_logits = self.edge_logits.masked_fill(~edge_active_mask, -1e9)
            edge_w = torch.softmax(masked_logits, dim=0)
        else:
            edge_w = torch.zeros_like(self.edge_logits)
        for e, lin in enumerate(self.edge_linears):
            if not bool(edge_active_mask[e]):
                continue
            ae = A[:, :, e]         # (B, T, N, N)
            agg = torch.matmul(ae, xp)
            out = out + edge_w[e] * lin(agg)
        out = out + self.skip(xp)
        return out.permute(0, 3, 1, 2)

    def edge_weights(self, edge_active_mask: torch.Tensor | None = None) -> torch.Tensor:
        if edge_active_mask is None:
            return torch.softmax(self.edge_logits.detach(), dim=0)
        mask = edge_active_mask.to(device=self.edge_logits.device, dtype=torch.bool)
        if not mask.any():
            return torch.zeros_like(self.edge_logits.detach())
        masked_logits = self.edge_logits.detach().masked_fill(~mask, -1e9)
        return torch.softmax(masked_logits, dim=0)


class DynamicSTBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hid: int,
        c_out: int,
        num_edge_types: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t1 = CausalTemporalGatedConv(c_in, c_hid, kernel_size)
        self.s = HeterogeneousSpatialGCN(c_hid, c_hid, num_edge_types)
        self.t2 = CausalTemporalGatedConv(c_hid, c_out, kernel_size)
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.t1(x)
        h = F.relu(self.s(h, A, edge_active_mask=edge_active_mask))
        h = self.t2(h)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)
        h = self.dropout(h)
        return F.relu(h + self.proj(x))


class DynamicSTGNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        mid: int = 128,
        num_blocks: int = 2,
        num_edge_types: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            blocks.append(
                DynamicSTBlock(
                    hidden,
                    mid,
                    hidden,
                    num_edge_types,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

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
        X: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # X: (B,T,N,F) -> embed -> (B,C,T,N)
        h = self.input(X)
        h = h.permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h, A, edge_active_mask=edge_active_mask)
        htnc = h.permute(0, 2, 3, 1)  # (B,T,N,C)

        # hidden[t] only observes X<=t because temporal convs are causal.
        hazard_logits = self.hazard_head(htnc[:, :-1]).squeeze(-1)

        # Auxiliary heads use observed sequence summary. Masking is handled in loss.
        pooled = htnc.mean(dim=1)
        rank_pred = self.rank_head(pooled).squeeze(-1)
        topk_logits = self.topk_head(pooled)
        return hazard_logits, rank_pred, topk_logits

    def edge_weights(self, edge_active_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        return [
            block.s.edge_weights(edge_active_mask=edge_active_mask)
            for block in self.blocks
            if isinstance(block, DynamicSTBlock)
        ]


def dynamic_stgnn_loss(
    hazard_logits: torch.Tensor,
    rank_pred: torch.Tensor,
    topk_logits: torch.Tensor,
    y_hazard: torch.Tensor,
    target_mask: torch.Tensor,
    y_rank: torch.Tensor,
    y_topk: torch.Tensor,
    team_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.3,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_f = target_mask.float()
    team_f = team_mask.float()
    hazard_denom = target_f.sum().clamp(min=1.0)
    team_denom = team_f.sum().clamp(min=1.0)

    pos_weight = torch.tensor(
        hazard_pos_weight,
        dtype=hazard_logits.dtype,
        device=hazard_logits.device,
    )
    hazard_bce = F.binary_cross_entropy_with_logits(
        hazard_logits,
        y_hazard,
        reduction="none",
        pos_weight=pos_weight,
    )
    l_hazard = (hazard_bce * target_f).sum() / hazard_denom

    l_rank = ((rank_pred - y_rank).abs() * team_f).sum() / team_denom

    topk_bce = F.binary_cross_entropy_with_logits(topk_logits, y_topk, reduction="none")
    l_topk = (topk_bce * team_f.unsqueeze(-1)).sum() / (team_denom * y_topk.size(-1))

    total = alpha * l_hazard + beta * l_rank + gamma * l_topk
    parts = {
        "loss": float(total.detach().item()),
        "l_hazard": float(l_hazard.detach().item()),
        "l_rank": float(l_rank.detach().item()),
        "l_topk": float(l_topk.detach().item()),
    }
    return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E, N, F_ = 2, 8, 3, 128, 66
    X = torch.randn(B, T, N, F_)
    A = torch.rand(B, T, E, N, N)
    A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    model = DynamicSTGNN(in_features=F_)
    yh, yr, yk = model(X, A)
    print(tuple(yh.shape), tuple(yr.shape), tuple(yk.shape))
