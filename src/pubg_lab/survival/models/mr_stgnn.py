"""
[한] 다해상도 계층형 시공간 그래프 신경망 (Multi-Resolution Hierarchical ST-GNN: MR-STGNN).
    - 구조:
        1) Tier-S (스냅샷 10초 주기): SnapshotEncoder -> 스냅샷 임베딩 (B, T_s, N, Cs)
        2) pool_snapshots_to_phase: 페이즈 내부의 스냅샷 임베딩들을 마스크 가중 평균 풀링 -> (B, T_p, N, Cs)
        3) Tier-P (페이즈 단위): 페이즈 특성 X_p와 풀링된 임베딩 결합 -> DynamicSTGNN 재사용
    - 출력:
        hazard_phase (B, T_p-1, N)    : 페이즈 전이(k -> k+1) 탈락 위험도 (주 라벨)
        survival_snapshot (B, T_s, N) : 10초 스냅샷 단위 생존 위험도 (심층 감독 보조 라벨)
        rank, topk                    : 최종 순위 및 Top-K 보조 예측
    - 인과성 및 정보 누설 방지 (A3 법칙):
        * 스냅샷 인코더는 좌측 패딩 인과적 Conv 사용 (시점 t의 임베딩은 <= t 정보만 참조).
        * 페이즈 풀링은 페이즈 내부에 한정되며, 페이즈 간 누적 의존성은 DynamicSTGNN의 인과적 Conv가 처리.
        * 인접 행렬은 DynamicSTGNN 불변식 검증을 통해 row-stochastic 보장.

[EN] Multi-Resolution Hierarchical Spatio-Temporal GNN (MR-STGNN).
    - Architecture:
        1) Tier-S (snapshot 10s): SnapshotEncoder -> snapshot embeddings (B, T_s, N, Cs)
        2) pool_snapshots_to_phase: Phase-local masked mean pooling -> phase node repr (B, T_p, N, Cs)
        3) Tier-P (phase-level): Concatenates phase features X_p with pooled embeddings -> DynamicSTGNN
    - Outputs:
        hazard_phase (B, T_p-1, N)    : Next-phase elimination hazard (Primary label)
        survival_snapshot (B, T_s, N) : Per-10s snapshot survival hazard (Deep supervision auxiliary label)
        rank, topk                    : Normalized rank and top-k survival logits
    - Causality and Leak-Free Guarantees (A3 Causality):
        * Snapshot encoder is strictly causal with left-padding (t sees only <= t).
        * Phase pooling is phase-local; cross-phase dynamics handled causally by DynamicSTGNN.
        * Adjacencies verified row-stochastic by reused DynamicSTGNN invariants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pubg_lab.survival.models.dyn_stgnn import DynamicSTGNN, dynamic_stgnn_loss
from pubg_lab.survival.models.snapshot_encoder import SnapshotEncoder, pool_snapshots_to_phase


class MRSTGNN(nn.Module):
    """
    [한] MR-STGNN 복합 신경망 모듈 (Multi-Resolution Spatio-Temporal GNN Module).
         - 역할: 고해상도(10초 단위) 스냅샷 상호작용과 저해상도(수 분 단위) 페이즈 구조를
           계층적으로 결합하여 단기 전술적 교전과 장기 전략적 생존을 동시에 포착합니다.
         - 구성:
           1) snap_encoder: 10초 단위 플레이어/팀 이동 및 교전 그래프 인코더
           2) phase_model: 페이즈 단위 거시적 동적 시공간 GNN (DynamicSTGNN)
           3) survival_head: 스냅샷 단위 심층 감독(Deep Supervision)을 위한 보조 위험도 헤드

    [EN] Multi-Resolution Spatio-Temporal GNN Module.
         - Role: Hierarchically bridges fine-grained (10s snapshots) tactical engagements and coarse-grained
           (multi-minute game phases) strategic macro-movements for joint survival estimation.
         - Components:
           1) snap_encoder: 10s snapshot spatio-temporal graph encoder
           2) phase_model: Coarse phase-level DynamicSTGNN
           3) survival_head: Auxiliary hazard readout head for snapshot-level deep supervision
    """

    def __init__(
        self,
        snap_in_features: int,
        phase_in_features: int,
        *,
        snap_hidden: int = 48,
        snap_mid: int = 96,
        snap_blocks: int = 2,
        snap_edge_types: int = 2,
        phase_hidden: int = 64,
        phase_mid: int = 128,
        phase_blocks: int = 2,
        phase_edge_types: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # [한] 1. 스냅샷 티어 인코더 (10초 단위 스냅샷 그래프 특성 및 엣지 임베딩)
        # [EN] 1. Snapshot tier encoder (processes 10s interval snapshot features and multi-relational graphs)
        self.snap_encoder = SnapshotEncoder(
            in_features=snap_in_features,
            hidden=snap_hidden,
            mid=snap_mid,
            num_blocks=snap_blocks,
            num_edge_types=snap_edge_types,
            dropout=dropout,
        )

        # [한] 2. 페이즈 티어 모델: 기존 페이즈 특성에 풀링된 스냅샷 임베딩을 결합(concatenation)하여 입력받음
        # [EN] 2. Phase tier model: Consumes [phase features ; pooled snapshot embeddings]
        self.phase_model = DynamicSTGNN(
            in_features=phase_in_features + self.snap_encoder.out_dim,
            hidden=phase_hidden,
            mid=phase_mid,
            num_blocks=phase_blocks,
            num_edge_types=phase_edge_types,
            dropout=dropout,
        )

        # [한] 3. 스냅샷 레벨 생존 위험도 보조 헤드 (중간 계층 표현력 강화를 위한 심층 감독)
        # [EN] 3. Snapshot-level survival hazard head (deep supervision for intermediate representations)
        self.survival_head = nn.Sequential(
            nn.Linear(self.snap_encoder.out_dim, self.snap_encoder.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.snap_encoder.out_dim // 2, 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,  # (B, T_s, N, F_s)
        A_s: torch.Tensor,  # (B, T_s, E_s, N, N)
        phase_of_snap: torch.Tensor,  # (B, T_s)
        node_mask_s: torch.Tensor,  # (B, T_s, N)
        X_p: torch.Tensor,  # (B, T_p, N, F_p)
        A_p: torch.Tensor,  # (B, T_p, E_p, N, N)
        *,
        snap_edge_mask: torch.Tensor | None = None,
        phase_edge_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # [한] 페이즈 수 확인 및 10초 스냅샷 인코딩 수행 -> (B, T_s, N, Cs)
        # [EN] Check number of phases and compute snapshot embeddings -> (B, T_s, N, Cs)
        t_phase = X_p.shape[1]
        snap_emb = self.snap_encoder(X_s, A_s, edge_active_mask=snap_edge_mask)

        # [한] 스냅샷 임베딩을 각 페이즈 단위로 마스크 가중 평균 풀링 -> (B, T_p, N, Cs)
        # [EN] Pool snapshot embeddings into their respective phases via masked mean -> (B, T_p, N, Cs)
        pooled = pool_snapshots_to_phase(
            snap_emb, phase_of_snap, node_mask_s, t_phase=t_phase
        )

        # [한] 원래 페이즈 특성 X_p와 풀링된 미시적 임베딩을 채널 축으로 결합
        # [EN] Concatenate macro phase features with pooled micro representations along feature axis
        X_p_aug = torch.cat([X_p, pooled], dim=-1)

        # [한] 페이즈 티어 DynamicSTGNN 순전파 실행 (위험도, 순위, Top-K 예측)
        # [EN] Forward through phase-tier DynamicSTGNN (hazard, rank, topk)
        hazard, rank, topk = self.phase_model(
            X_p_aug, A_p, edge_active_mask=phase_edge_mask, phase_mask=phase_mask
        )

        # [한] 스냅샷 단위 생존 위험도 로짓 산출 (B, T_s, N)
        # [EN] Compute snapshot-level survival hazard logits (B, T_s, N)
        survival = self.survival_head(snap_emb).squeeze(-1)
        return hazard, survival, rank, topk


def mr_survival_targets(
    death_snap_idx: torch.Tensor,  # (B, N) int, -1 = censored/absent
    node_mask_s: torch.Tensor,  # (B, T_s, N) bool
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [한] 스냅샷 레벨 이산시간 위험도 타겟 및 위험 노출(at-risk) 마스크 생성 함수.
         - 역할: 각 생존 노드(b, t, team)에 대해, 팀의 사망 스냅샷 인덱스가 미래 윈도우 (t, t+horizon] 구간에
           속하는지 확인하여 라벨(1 또는 0)을 부여하고, 현재 시점 t에 생존해 있는 노드만을 at-risk 마스크로 지정합니다.
         - 인과성: t 시점의 입력 피처는 오직 <= t 시점만 참조(인코더 인과성)하며, 타겟 라벨은 미래 사건 발생 여부를
           조사하는 표준 이산시간 생존분석(Discrete-time survival analysis) 규약에 부합하여 정보 누설이 없습니다.

    [EN] Build snapshot-level discrete-time hazard targets + at-risk mask.
         - Role: For each alive team node at snapshot t, sets target = 1 if death occurs within the causal
           forward evaluation window (t, t + horizon], and 0 otherwise. Marks team as at-risk if alive at t.
         - Causality: Input features for snapshot t observe only <= t (strictly causal encoder), while labels
           properly observe future events according to standard discrete-time survival analysis principles.
    """
    B, T_s, N = node_mask_s.shape
    device = node_mask_s.device
    t_idx = torch.arange(T_s, device=device).view(1, T_s, 1)  # (1, T_s, 1)
    death = death_snap_idx.view(B, 1, N)  # (B, 1, N)
    has_death = death >= 0

    # [한] at-risk 정의: 시점 t에 노드가 생존해 있는 상태 (node_mask_s)
    # [EN] at-risk condition: team node is alive and present at snapshot t
    at_risk = node_mask_s

    # [한] 사망 이벤트 발생 윈도우: t < death <= t + horizon
    # [EN] Event occurrence window: within (t, t + horizon]
    within = has_death & (death > t_idx) & (death <= t_idx + horizon)
    target = (within & at_risk).to(torch.float32)
    return target, at_risk.to(torch.float32)


def mr_loss(
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
    horizon: int = 3,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    [한] MR-STGNN 복합 계층 손실 함수 (Multi-Resolution Loss Function).
         - 총 손실 = L_phase (hazard + rank + topk) + delta * L_survival (스냅샷 심층 감독)
         - 페이즈 손실부는 DynamicSTGNN의 검증된 손실 함수를 그대로 재사용하여,
           delta=0 설정 시 기존 페이즈 모델 기준선과 수치적으로 완전히 동일한 결과를 보장합니다.

    [EN] Multi-Resolution Loss for MR-STGNN.
         - Total Loss = L_phase (hazard + rank + topk) + delta * L_survival (snapshot deep supervision).
         - Reuses the validated dynamic_stgnn_loss verbatim, guaranteeing that delta=0 numerically
           matches the single-resolution phase baseline.
    """
    # [한] 1. 페이즈 티어 다중 작업 손실 계산 (위험도 + 순위 + Top-K)
    # [EN] 1. Compute phase-tier multi-task loss (hazard + rank + topk)
    phase_total, parts = dynamic_stgnn_loss(
        hazard,
        rank,
        topk,
        batch["y_hazard"],
        batch["target_mask"],
        batch["y_rank"],
        batch["y_topk"],
        batch["team_mask"],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        hazard_pos_weight=hazard_pos_weight,
    )

    # [한] 2. 스냅샷 티어 이산시간 생존 타겟 생성 및 BCE 손실 계산
    # [EN] 2. Generate snapshot-tier discrete-time targets and compute BCE loss
    tgt, at_risk = mr_survival_targets(
        batch["death_snap_idx"], batch["node_mask_s"], horizon=horizon
    )
    surv_bce = F.binary_cross_entropy_with_logits(survival, tgt, reduction="none")
    surv_denom = at_risk.sum().clamp(min=1.0)
    l_survival = (surv_bce * at_risk).sum() / surv_denom

    # [한] 3. 페이즈 손실과 스냅샷 보조 손실 결합
    # [EN] 3. Combine phase loss with weighted snapshot auxiliary loss
    total = phase_total + delta * l_survival
    parts = dict(parts)
    parts["loss"] = float(total.detach().item())
    parts["l_survival"] = float(l_survival.detach().item())
    parts["survival_pos"] = float(tgt.sum().detach().item())
    return total, parts
