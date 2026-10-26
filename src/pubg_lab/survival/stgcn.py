"""
model_stgcn/stgcn.py
====================
Spatio-Temporal Graph Convolutional Network — Yu et al. (2018) skeleton with a
*dynamic* per-timestep adjacency matrix `A_t`.

Differences from the original STGCN:
  • A_t is given per timestep (B, T, N, N), not shared across time.
  • Spatial conv is implemented as a dense per-timestep batched op
    `H' = A_t @ H @ W`, which is fine because N is small (≤ 25 for PUBG).
  • Two output heads: a regression head (final-rank prediction) and a binary
    classification head ([P(top5), P(top10)]).

Input/output tensors:
  X     : (B, T, N, F_in)
  A_t   : (B, T, N, N)       row-normalised, includes self-loops
  rank  : (B, N)              continuous rank prediction (lower = better)
  topk  : (B, N, 2)           logits for [top5, top10]

The architecture follows two ST-Conv blocks with the structure
  [TemporalGated → SpatialGCN → TemporalGated] from Yu et al.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGatedConv(nn.Module):
    """1D convolution along the time axis with a Gated Linear Unit (GLU)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        # output 2*c_out, split into (a, b); GLU = a * sigmoid(b)
        self.conv = nn.Conv2d(
            c_in,
            2 * c_out,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, N)
        h = self.conv(x)
        a, b = h.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DynamicSpatialGCN(nn.Module):
    """Per-timestep dense graph conv with a time-varying adjacency."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, x: torch.Tensor, A_t: torch.Tensor) -> torch.Tensor:
        # x:   (B, C_in, T, N)
        # A_t: (B, T, N, N)
        b, c, t, n = x.shape
        xp = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        xa = torch.matmul(A_t, xp)  # (B, T, N, C)
        xo = self.lin(xa)  # (B, T, N, C_out)
        return xo.permute(0, 3, 1, 2)  # (B, C_out, T, N)


class STConvBlock(nn.Module):
    """[Temporal-Gated → Spatial GCN → Temporal-Gated] with norm + residual."""

    def __init__(
        self,
        c_in: int,
        c_hid: int,
        c_out: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t1 = TemporalGatedConv(c_in, c_hid, kernel_size)
        self.s = DynamicSpatialGCN(c_hid, c_hid)
        self.t2 = TemporalGatedConv(c_hid, c_out, kernel_size)
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        # 1×1 projection for the residual (handles c_in != c_out)
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, A_t: torch.Tensor) -> torch.Tensor:
        h = self.t1(x)
        h = F.relu(self.s(h, A_t))
        h = self.t2(h)
        # LayerNorm acts on the channel dimension: (B,C,T,N) → (B,T,N,C) → back
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)
        h = self.dropout(h)
        return F.relu(h + self.proj(x))


class STGCN(nn.Module):
    """The full network — stacks two ST-Conv blocks, pools over time, two heads."""

    def __init__(
        self,
        in_features: int = 8,
        hidden: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            c_in = in_features if i == 0 else hidden
            blocks.append(STConvBlock(c_in, hidden, hidden, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.head_cls = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, X: torch.Tensor, A_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # X: (B, T, N, F)  → (B, F, T, N)
        h = X.permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h, A_t)
        # Temporal pool: average over T (more stable than last-step for variable phases)
        h = h.mean(dim=2)  # (B, C, N)
        h = h.permute(0, 2, 1)  # (B, N, C)
        rank = self.head_reg(h).squeeze(-1)  # (B, N)
        topk = self.head_cls(h)  # (B, N, 2)
        return rank, topk


# ── Losses ───────────────────────────────────────────────────────────────────


def stgcn_loss(
    rank_pred: torch.Tensor,
    topk_logits: torch.Tensor,
    rank_true: torch.Tensor,
    topk_true: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Combined multi-task loss.

    Regression: per-node L1 between predicted and true rank
                (rank is divided by 25 to keep gradient scale comparable to BCE).
    Classification: BCE-with-logits on top5 / top10 columns.

    All terms are masked by `mask` (B, N) so padded slots contribute 0.
    """
    mf = mask.float()
    denom = mf.sum().clamp(min=1.0)

    # MAE on normalised rank
    rank_norm_t = rank_true / 25.0
    l_reg = (mf * (rank_pred - rank_norm_t).abs()).sum() / denom

    # BCE on each top-k head
    bce = F.binary_cross_entropy_with_logits(topk_logits, topk_true, reduction="none")
    l_cls_top5 = (mf * bce[..., 0]).sum() / denom
    l_cls_top10 = (mf * bce[..., 1]).sum() / denom

    total = alpha * l_reg + beta * l_cls_top5 + gamma * l_cls_top10
    parts = {
        "loss": total.detach().item(),
        "l_reg": l_reg.detach().item(),
        "l_top5": l_cls_top5.detach().item(),
        "l_top10": l_cls_top10.detach().item(),
    }
    return total, parts


if __name__ == "__main__":
    # tiny smoke test
    torch.manual_seed(0)
    B, T, N, F_ = 2, 20, 25, 8
    X = torch.randn(B, T, N, F_)
    A = torch.softmax(torch.randn(B, T, N, N), dim=-1)
    model = STGCN(in_features=F_)
    rank, topk = model(X, A)
    y_reg = torch.randint(1, 26, (B, N)).float()
    y_cls = torch.bernoulli(torch.full((B, N, 2), 0.3))
    mask = torch.ones(B, N, dtype=torch.bool)
    loss, parts = stgcn_loss(rank, topk, y_reg, y_cls, mask)
    print("rank pred:", tuple(rank.shape), "  topk logits:", tuple(topk.shape))
    print("loss:", parts)
