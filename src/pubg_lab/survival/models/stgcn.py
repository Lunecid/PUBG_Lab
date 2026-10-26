"""
[한] 시공간 그래프 합성곱 신경망 (Spatio-Temporal Graph Convolutional Network: STGCN).
    - 참조: Yu et al. (2018) 골격을 기반으로 PUBG 게임의 타임스텝별 동적 인접 행렬(A_t)을 지원하도록 확장.
    - 원본 STGCN과의 차이점:
        • 고정 인접 행렬 대신 시점별 동적 인접 행렬 A_t (B, T, N, N) 사용.
        • 소규모 노드 수(PUBG 최대 25개 팀)에 적합한 조밀 배치 행렬 곱 H' = A_t @ H @ W 사용.
        • 2개의 출력 헤드: 연속 최종 순위 회귀 헤드 및 [P(top5), P(top10)] 이진 분류 헤드.
    - 입력/출력 텐서:
        X     : (B, T, N, F_in)   - 노드 특성
        A_t   : (B, T, N, N)      - 행 정규화 및 self-loop가 포함된 시점별 인접 행렬
        rank  : (B, N)            - 정규화된 최종 순위 예측 (낮을수록 우수)
        topk  : (B, N, 2)         - [Top-5, Top-10] 잔존 여부 로짓

[EN] Spatio-Temporal Graph Convolutional Network (STGCN).
    - Reference: Extends Yu et al. (2018) skeleton with dynamic per-timestep adjacency matrix `A_t`.
    - Key differences from original STGCN:
        • Adjacency A_t varies per timestep (B, T, N, N), rather than being static across time.
        • Dense batched spatial convolution H' = A_t @ H @ W optimized for small N (<= 25 teams in PUBG).
        • Dual prediction heads: rank regression head and binary classification head ([P(top5), P(top10)]).
    - Input/Output tensors:
        X     : (B, T, N, F_in)   - Node feature sequence
        A_t   : (B, T, N, N)      - Row-normalized dynamic adjacency with self-loops
        rank  : (B, N)            - Continuous normalized rank prediction (lower is better)
        topk  : (B, N, 2)         - Survival classification logits for [top5, top10]
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
# [한] 직접 실행과 패키지 임포트 모두를 지원하기 위한 경로 설정 블록
# [EN] Path setup block supporting both direct script execution and package imports
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGatedConv(nn.Module):
    """
    [한] 시간 축 1D 합성곱 및 GLU 활성화 (Temporal Gated Convolution with GLU).
         - 역할: 시간 축을 따라 1D 합성곱을 수행하고 GLU(Gated Linear Unit) 게이팅을 통해 비선형 시간 특징을 추출합니다.
         - 구현: 출력 채널을 2*c_out으로 확장한 후 채널을 반으로 나누어 (a, b)를 얻고 a * sigmoid(b)를 계산합니다.

    [EN] 1D Convolution along the Time Axis with a Gated Linear Unit (GLU).
         - Role: Extracts nonlinear temporal sequence patterns via 1D convolution combined with GLU gating.
         - Implementation: Expands channels to 2*c_out, splits into (a, b), and applies a * sigmoid(b).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        # [한] GLU 게이팅을 위해 출력 채널을 2*c_out으로 설정하는 2D Conv
        # [EN] Conv2d with kernel (kernel_size, 1) producing 2*c_out channels for GLU activation
        self.conv = nn.Conv2d(
            c_in,
            2 * c_out,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [한] x 형상: (B, C_in, T, N)
        # [EN] x shape: (B, C_in, T, N)
        h = self.conv(x)
        a, b = h.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class DynamicSpatialGCN(nn.Module):
    """
    [한] 동적 인접 행렬을 사용하는 시점별 조밀 공간 그래프 합성곱 (Dynamic Spatial GCN).
         - 역할: 시간에 따라 변화하는 팀 간 그래프 A_t로부터 각 시점마다 이웃 팀들의 상태를 집약합니다.
         - 수식: H' = A_t @ H @ W
         - 구현: (B, T, N, C) 형태로 전치 후 행렬 곱을 수행하고 선형 변환(Linear)을 적용합니다.

    [EN] Dense Spatial Graph Convolution with Time-Varying Adjacency Matrix.
         - Role: Aggregates neighbor representations at each timestep using dynamic adjacency A_t.
         - Formulation: H' = A_t @ H @ W
         - Implementation: Permutes to (B, T, N, C), multiplies with A_t, and projects via Linear layer.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, x: torch.Tensor, A_t: torch.Tensor) -> torch.Tensor:
        # [한] x: (B, C_in, T, N), A_t: (B, T, N, N)
        # [EN] x: (B, C_in, T, N), A_t: (B, T, N, N)
        b, c, t, n = x.shape
        xp = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        xa = torch.matmul(A_t, xp)  # (B, T, N, C) - 이웃 노드 특성 집약 / neighbor aggregation
        xo = self.lin(xa)           # (B, T, N, C_out) - 선형 특징 사영 / linear feature projection
        return xo.permute(0, 3, 1, 2)  # (B, C_out, T, N)


class STConvBlock(nn.Module):
    """
    [한] 시공간 합성곱 블록 (Spatio-Temporal Convolutional Block).
         - 역할: [Temporal-Gated -> Spatial GCN -> Temporal-Gated] 샌드위치 구조로
           시간적 전이와 공간적 팀 상호작용을 통합하고 잔차 연결 및 LayerNorm으로 안정화합니다.

    [EN] Spatio-Temporal Convolutional Block.
         - Role: Integrates spatial and temporal interactions via [Temporal-Gated -> Spatial GCN -> Temporal-Gated]
           pipeline with residual connection and LayerNorm.
    """

    def __init__(
        self,
        c_in: int,
        c_hid: int,
        c_out: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # [한] 1단계 시간 게이트 합성곱
        # [EN] First temporal gated convolution stage
        self.t1 = TemporalGatedConv(c_in, c_hid, kernel_size)
        # [한] 2단계 동적 공간 그래프 합성곱
        # [EN] Second dynamic spatial GCN stage
        self.s = DynamicSpatialGCN(c_hid, c_hid)
        # [한] 3단계 시간 게이트 합성곱
        # [EN] Third temporal gated convolution stage
        self.t2 = TemporalGatedConv(c_hid, c_out, kernel_size)
        # [한] 채널 차원 정규화를 위한 LayerNorm 및 드롭아웃
        # [EN] LayerNorm along channel dimension and dropout for regularization
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        # [한] 입력/출력 채널 불일치를 맞추기 위한 1x1 투영 잔차 연결
        # [EN] 1x1 projection for residual connection (handles c_in != c_out)
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, A_t: torch.Tensor) -> torch.Tensor:
        h = self.t1(x)
        h = F.relu(self.s(h, A_t))
        h = self.t2(h)
        # [한] 채널 축 LayerNorm 적용을 위한 차원 전치: (B,C,T,N) -> (B,T,N,C) -> (B,C,T,N)
        # [EN] Permute for channel-wise LayerNorm: (B,C,T,N) -> (B,T,N,C) -> (B,C,T,N)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)
        h = self.dropout(h)
        # [한] 잔차 연결 및 ReLU 활성화
        # [EN] Residual shortcut addition and ReLU
        return F.relu(h + self.proj(x))


class STGCN(nn.Module):
    """
    [한] PUBG 전체 STGCN 네트워크 (Full STGCN Network).
         - 역할: 2개의 ST-Conv 블록을 적층하고, 시간 축 평균 풀링을 거쳐
           최종 순위 예측(회귀)과 Top-5/Top-10 잔존 예측(분류)을 동시에 수행합니다.

    [EN] Full STGCN Architecture.
         - Role: Stacks two ST-Conv blocks, performs temporal mean pooling, and feeds into
           two output heads for continuous rank regression and top-k survival classification.
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # [한] 지정된 개수의 STConvBlock 적층
        # [EN] Stack of num_blocks STConvBlocks
        blocks = []
        for i in range(num_blocks):
            c_in = in_features if i == 0 else hidden
            blocks.append(STConvBlock(c_in, hidden, hidden, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        # [한] 순위 회귀 예측 헤드 (출력: 1개 스칼라 순위 점수)
        # [EN] Rank regression head (outputs 1 continuous rank score per node)
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # [한] Top-K 잔존 분류 예측 헤드 (출력: [Top-5, Top-10] 2개 로짓)
        # [EN] Top-K survival classification head (outputs 2 logits for [top5, top10])
        self.head_cls = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, X: torch.Tensor, A_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [한] X: (B, T, N, F) -> (B, F, T, N) 전치 후 블록 통과
        # [EN] Permute X from (B, T, N, F) to (B, F, T, N) and pass through ST-Conv blocks
        h = X.permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h, A_t)

        # [한] 시간 축 평균 풀링: 가변 페이즈 길이에 대해 마지막 시점보다 평균 풀링이 안정적임
        # [EN] Temporal mean pooling: averaging over T is more robust than last-step for variable phase lengths
        h = h.mean(dim=2)       # (B, C, N)
        h = h.permute(0, 2, 1)  # (B, N, C)

        # [한] 회귀 및 분류 헤드 예측 산출
        # [EN] Output predictions from regression and classification heads
        rank = self.head_reg(h).squeeze(-1)  # (B, N)
        topk = self.head_cls(h)              # (B, N, 2)
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
    """
    [한] STGCN 결합 다중 작업 손실 함수 (Combined Multi-task Loss).
         - 회귀 손실: 예측 순위와 실제 정규화 순위(rank / 25.0) 간의 노드별 L1 오차
         - 분류 손실: Top-5 및 Top-10 컬럼에 대한 BCEWithLogitsLoss
         - 마스킹: mask (B, N)를 적용하여 패딩된 슬롯의 기여도를 0으로 처리합니다.

    [EN] Combined Multi-task Loss for STGCN.
         - Regression loss: Per-node L1 between predicted and true normalized rank (divided by 25.0)
         - Classification loss: BCE-with-logits on top5 and top10 columns
         - Masking: Masked by `mask` (B, N) so padded team slots contribute 0
    """
    mf = mask.float()
    denom = mf.sum().clamp(min=1.0)

    # [한] 1. 정규화 순위 MAE 손실 (순위를 25로 나누어 BCE와 그래디언트 스케일 균형 유지)
    # [EN] 1. MAE on normalized rank (divided by 25 to balance gradient scales with BCE)
    rank_norm_t = rank_true / 25.0
    l_reg = (mf * (rank_pred - rank_norm_t).abs()).sum() / denom

    # [한] 2. 각 Top-K 헤드에 대한 BCE 손실
    # [EN] 2. BCE loss on each top-k head
    bce = F.binary_cross_entropy_with_logits(topk_logits, topk_true, reduction="none")
    l_cls_top5 = (mf * bce[..., 0]).sum() / denom
    l_cls_top10 = (mf * bce[..., 1]).sum() / denom

    # [한] 총 손실 가중합 계산
    # [EN] Compute total weighted loss
    total = alpha * l_reg + beta * l_cls_top5 + gamma * l_cls_top10
    parts = {
        "loss": total.detach().item(),
        "l_reg": l_reg.detach().item(),
        "l_top5": l_cls_top5.detach().item(),
        "l_top10": l_cls_top10.detach().item(),
    }
    return total, parts


if __name__ == "__main__":
    # [한] 소규모 스모크 테스트 (순전파 형상 및 손실 계산 검증)
    # [EN] Tiny smoke test validating forward shapes and loss computation
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
