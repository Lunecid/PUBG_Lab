"""
[한] MR-STGNN Tier-S를 위한 스냅샷(10초 주기) 인코더 및 페이즈 풀링 모듈.
    - 역할: 전체 10초 스냅샷 시퀀스에 걸쳐 페이즈 티어의 DynamicSTBlock(인과적 시간 게이트 합성곱 +
           조밀 이종 공간 GCN)을 재사용하여 미시적 표현을 학습한 후, 각 스냅샷 임베딩을 소속 페이즈로
           풀링하여 페이즈 티어에 전달합니다.
    - 인과성 보장 (A3 causality):
        DynamicSTBlock의 시간 합성곱은 좌측 패딩 인과적 Conv이므로 시점 t의 임베딩은 오직 <= t 시점만 관측합니다.
        페이즈 풀링은 페이즈 국소적(phase-local: phase_of_snap == k)으로 수행되며, 페이즈 간 누적 의존성은
        상위 페이즈 티어의 인과적 합성곱이 처리합니다.

[EN] Snapshot (10s) encoder + phase pooling for MR-STGNN Tier-S.
    - Role: Reuses the phase tier's DynamicSTBlock (causal temporal-gated conv + dense
           heterogeneous spatial GCN) over fine-grained snapshot sequences, pooling snapshot embeddings
           into their corresponding game phases.
    - Causality guarantees (A3 causality):
        DynamicSTBlock's temporal conv uses left-padding so snapshot t observes only <= t.
        Phase pooling is strictly phase-local, and cross-phase temporal dynamics are handled by the
        phase tier's own causal convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pubg_lab.survival.models.dyn_stgnn import DynamicSTBlock


class SnapshotEncoder(nn.Module):
    """
    [한] 10초 스냅샷 그래프 인코더 (Snapshot Spatio-Temporal Graph Encoder).
         - 역할: 10초 주기로 샘플링된 미시적 플레이어/팀 이동 및 이종 상호작용 그래프를 입력받아
           인과적 시공간 블록을 통해 풍부한 미시 표현 벡터를 추출합니다.
         - 구조: 입력 선형 투영 레이어 + num_blocks개의 DynamicSTBlock 스택.

    [EN] 10-Second Snapshot Spatio-Temporal Graph Encoder.
         - Role: Processes micro-level player/team movements and interaction graphs sampled at 10s intervals
           to generate fine-grained spatio-temporal representations.
         - Structure: Input projection layer + stack of num_blocks DynamicSTBlocks.
    """

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
        # [한] 입력 스냅샷 피처를 은닉 차원으로 매핑하는 선형 투영 레이어
        # [EN] Linear projection layer mapping input snapshot features to hidden dimension
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # [한] 미시 시공간 상호작용을 포착하는 DynamicSTBlock 적층 모듈 리스트
        # [EN] ModuleList of stacked DynamicSTBlocks capturing micro spatio-temporal dynamics
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
        # [한] 입력 피처 투영 및 합성곱 차원 (B, C, T_s, N)으로 전치
        # [EN] Project input features and permute to (B, C, T_s, N) for convolution operations
        h = self.input(X_s).permute(0, 3, 1, 2)

        # [한] 적층된 시공간 블록을 순차 통과하여 인과적 미시 상호작용 표현 갱신
        # [EN] Pass through stacked spatio-temporal blocks sequentially
        for block in self.blocks:
            h = block(h, A_s, edge_active_mask=edge_active_mask)

        # [한] (B, T_s, N, C) 형상으로 복원하여 반환
        # [EN] Permute back and return tensor of shape (B, T_s, N, C)
        return h.permute(0, 2, 3, 1)


def pool_snapshots_to_phase(
    snap_emb: torch.Tensor,  # (B, T_s, N, C)
    phase_of_snap: torch.Tensor,  # (B, T_s) int, 0 = landing/pre-phase
    node_mask_s: torch.Tensor,  # (B, T_s, N) bool
    *,
    t_phase: int = 8,
) -> torch.Tensor:  # (B, t_phase, N, C)
    """
    [한] 각 게임 페이즈 내부의 스냅샷 임베딩에 대한 마스크 가중 평균 풀링 (Phase-local Masked Pooling).
         - 역할: 10초 스냅샷 단위의 미시 표현들을 소속 페이즈(1..t_phase)별로 집약하여,
           상위 페이즈 티어 모델이 사용할 수 있는 거시 노드 표현 (B, t_phase, N, C)을 생성합니다.
         - 특성: phase_of_snap == 0(낙하/페이즈 1 이전 초기 구간)은 페이즈 1의 과거 맥락으로 편입(fold)됩니다.
           출력 인덱스 p는 페이즈 (p+1)에 해당하여 페이즈 티어 텐서 레이아웃과 일치합니다.

    [EN] Masked Mean Pooling of Snapshot Embeddings within Each Phase.
         - Role: Aggregates 10-second snapshot representations within their corresponding game phases
           to produce coarse-level phase node representations (B, t_phase, N, C).
         - Properties: phase_of_snap == 0 (landing/pre-phase 1) is folded into phase 1 as historical context.
           Output index p corresponds to phase (p+1), perfectly aligning with phase-tier tensor formats.
    """
    B, T_s, N, C = snap_emb.shape
    device = snap_emb.device

    # [한] 스냅샷별 페이즈 ID를 [1, t_phase]로 클리핑한 후 0 기반 인덱스로 변환: (B, T_s) in [0, t_phase-1]
    # [EN] Clip phase IDs into [1, t_phase] and convert to 0-based indices: (B, T_s) in [0, t_phase-1]
    pid = phase_of_snap.clamp(min=1, max=t_phase).long() - 1

    # [한] 유효 생존 노드 마스크 적용: (B, T_s, N, 1)
    # [EN] Apply node alive mask to ignore dead/padded teams: (B, T_s, N, 1)
    m = node_mask_s.to(snap_emb.dtype).unsqueeze(-1)
    masked = snap_emb * m

    # [한] 페이즈별 집약을 위한 빈 텐서 준비
    # [EN] Allocate accumulator and denominator tensors for scatter pooling
    out = snap_emb.new_zeros((B, t_phase, N, C))
    denom = snap_emb.new_zeros((B, t_phase, N, 1))

    # [한] scatter_add_ 연산을 이용해 소속 페이즈 인덱스 위치에 마스킹된 임베딩과 카운트를 누적
    # [EN] Accumulate masked embeddings and counts into phase bins using scatter_add_
    idx = pid.view(B, T_s, 1, 1).expand(B, T_s, N, C)
    out.scatter_add_(1, idx, masked)
    denom.scatter_add_(1, pid.view(B, T_s, 1, 1).expand(B, T_s, N, 1), m)

    # [한] 0으로 나누기 방지(clamp min=1.0)를 적용한 평균 계산 후 반환
    # [EN] Divide accumulated sum by count with zero-division guard (clamp min=1.0)
    return out / denom.clamp(min=1.0)
