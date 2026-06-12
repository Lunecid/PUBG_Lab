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
        self._adjacency_checked = False

    def _validate_adjacency(self, A: torch.Tensor) -> None:
        """첫 forward에서 1회만: A가 row-stochastic(또는 zero-row)인지 검사.

        stdb.queries._fetch_phase_edges가 보장하는 불변식이 ETL/캐시 변경으로
        조용히 깨지는 것을 학습 시작 시점에 잡는다 (CODE_REVIEW C6).
        """
        if A.dim() != 5:
            raise ValueError(f"A must be (B,T,E,N,N), got {tuple(A.shape)}")
        rs = A.sum(dim=-1)
        ok = torch.isclose(rs, torch.ones_like(rs), atol=1e-3) | torch.isclose(
            rs, torch.zeros_like(rs), atol=1e-6
        )
        if not bool(ok.all()):
            bad = (~ok).nonzero()[0].tolist()
            raise ValueError(
                f"adjacency rows must sum to 1 (or 0 for edge-less rows); "
                f"first violation at (b,t,e,n)={bad}, row_sum={rs[tuple(bad)]:.4f}"
            )

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        edge_active_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._adjacency_checked:
            self._validate_adjacency(A)
            self._adjacency_checked = True

        # X: (B,T,N,F) -> embed -> (B,C,T,N)
        h = self.input(X)
        h = h.permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h, A, edge_active_mask=edge_active_mask)
        htnc = h.permute(0, 2, 3, 1)  # (B,T,N,C)

        # hidden[t] only observes X<=t because temporal convs are causal.
        hazard_logits = self.hazard_head(htnc[:, :-1]).squeeze(-1)

        # Auxiliary heads use observed sequence summary.  Padded phases carry
        # bias-driven garbage (input(0) != 0), so average only real phases when
        # phase_mask (B,T) is given; per-node target masking stays in the loss.
        if phase_mask is not None:
            m = phase_mask.to(dtype=htnc.dtype, device=htnc.device)
            m = m.view(m.shape[0], m.shape[1], 1, 1)  # (B,T,1,1)
            pooled = (htnc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
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
        # 타겟이 전무한 배치(denom clamp로 loss=0)를 학습 곡선에서 식별하기 위함
        "hazard_denom": float(target_f.sum().detach().item()),
    }
    return total, parts


if __name__ == "__main__":
    # 실제 파이프라인 분포를 흉내낸 스모크: 희소 A, zero-row 허용, alive에만
    # self-loop, 패딩 노드/phase는 전부 0 (stdb.queries._fetch_phase_edges 참조)
    torch.manual_seed(0)
    B, T, E, N, F_ = 2, 8, 3, 128, 66
    n_alive, n_phase = 24, 6  # 실 노드 24팀, 실 phase 6개

    X = torch.zeros(B, T, N, F_)
    X[:, :n_phase, :n_alive] = torch.randn(B, n_phase, n_alive, F_)

    A = torch.zeros(B, T, E, N, N)
    dense = torch.rand(B, n_phase, E, n_alive, n_alive)
    dense = dense * (torch.rand_like(dense) < 0.2)          # ~80% 희소
    eye = torch.eye(n_alive)
    dense[:, :, 0] = dense[:, :, 0] + eye                    # proximity self-loop
    rs = dense.sum(dim=-1, keepdim=True)
    dense = torch.where(rs > 0, dense / rs.clamp(min=1e-9), dense)  # zero-row 유지
    A[:, :n_phase, :, :n_alive, :n_alive] = dense

    phase_mask = torch.zeros(B, T, dtype=torch.bool)
    phase_mask[:, :n_phase] = True

    model = DynamicSTGNN(in_features=F_).eval()
    with torch.no_grad():
        yh, yr, yk = model(X, A, phase_mask=phase_mask)
    print(tuple(yh.shape), tuple(yr.shape), tuple(yk.shape))

    # 정규화 위반 시 명확히 실패하는지 확인
    try:
        bad = A.clone()
        bad[:, 0, 0, 0, :n_alive] = 0.5
        model._adjacency_checked = False
        model(X, bad)
        raise SystemExit("validation should have failed")
    except ValueError as e:
        print("adjacency validation OK:", str(e)[:60])
