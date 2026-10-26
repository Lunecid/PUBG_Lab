"""
[한] 페이즈 컷 출력을 산출하는 스냅샷 해상도 시공간 생존 모델 (Snapshot-Resolution ST Survival Model with Phase-Cut Outputs).
    - 구조:
        10초 단위 스냅샷 시퀀스(X_s, A_s) -> 인과적 스냅샷 인코더(SnapshotEncoder) -> 페이즈 컷 시점 헤드(phase-cut heads)
    - 특징:
        페이즈 인덱스는 오직 출력 평가 시점(cut points)을 결정하는 데에만 사용되며,
        ST 인코더 통과 전에 원시 피처를 페이즈 단위로 사전 평균화(aggregation)하지 않고 미시 시계열을 온전히 보존합니다.

[EN] Snapshot-Resolution Spatio-Temporal Survival Model with Phase-Cut Outputs.
    - Architecture:
        Every 10s snapshot (X_s, A_s) -> Causal snapshot encoder -> Phase-cut heads
    - Properties:
        The phase index is used solely to select causal output cut points. Raw features are not
        coarsely averaged into phase-level tensors prior to the ST encoder, preserving temporal fidelity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pubg_lab.survival.models.snapshot_encoder import SnapshotEncoder


def phase_cut_indices(
    phase_of_snap: torch.Tensor,
    node_mask_s: torch.Tensor,
    *,
    t_phase: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [한] 각 페이즈 컷에 해당하는 마지막 관측 스냅샷의 시간 인덱스 반환 함수.
         - 역할: 컷 페이즈 k에 대해 클리핑된 페이즈 ID가 <= k인 실제 스냅샷 중 가장 마지막(최신) 스냅샷 인덱스를 찾습니다.
         - 처리: phase_of_snap == 0(낙하/페이즈 1 이전)은 페이즈 1의 과거 맥락으로 간주하며, 패딩된 스냅샷은 제외합니다.

    [EN] Return Last Observed Snapshot Index for Each Phase Cut.
         - Role: For phase cut k, selects the last observed snapshot index whose clipped phase ID is <= k.
         - Handling: Treats phase_of_snap == 0 as phase-1 landing context; excludes padded snapshots.
    """
    if phase_of_snap.dim() != 2:
        raise ValueError("phase_of_snap must have shape (B, T_s)")
    if node_mask_s.dim() != 3:
        raise ValueError("node_mask_s must have shape (B, T_s, N)")
    if phase_of_snap.shape[:2] != node_mask_s.shape[:2]:
        raise ValueError("phase_of_snap and node_mask_s disagree on B/T_s")

    B, T_s = phase_of_snap.shape
    device = phase_of_snap.device
    # [한] 하나 이상의 팀 노드가 살아있는 유효 스냅샷 마스크
    # [EN] Valid snapshot mask where at least one team is alive
    valid_snap = node_mask_s.any(dim=-1)
    pid = phase_of_snap.clamp(min=1, max=t_phase).long()
    time_idx = torch.arange(T_s, device=device).view(1, T_s).expand(B, T_s)

    idxs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    neg_one = time_idx.new_full((B, T_s), -1)

    # [한] 각 페이즈(1..t_phase)별로 해당 페이즈 이하에서 가장 늦은 스냅샷 인덱스를 최대값으로 추출
    # [EN] For each phase cut (1..t_phase), extract the maximum snapshot index satisfying pid <= phase
    for phase in range(1, t_phase + 1):
        eligible = valid_snap & (pid <= phase)
        idx = torch.where(eligible, time_idx, neg_one).amax(dim=1)
        masks.append(idx >= 0)
        idxs.append(idx.clamp(min=0))

    return torch.stack(idxs, dim=1), torch.stack(masks, dim=1)


def gather_phase_cut_embeddings(
    snap_emb: torch.Tensor,
    phase_of_snap: torch.Tensor,
    node_mask_s: torch.Tensor,
    *,
    t_phase: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [한] 인과적 페이즈 컷 시점의 스냅샷 임베딩 수집 함수 (Gather Phase-cut Embeddings).
         - 역할: 전체 스냅샷 임베딩 snap_emb (B, T_s, N, C)로부터 각 페이즈 종료 컷 시점에 해당하는
           스냅샷 벡터들을 선별(gather)하여 (B, t_phase, N, C) 형태로 반환합니다.

    [EN] Gather Snapshot Embeddings at Each Causal Phase Cut.
         - Role: Gathers embedding slices at the specific snapshot indices corresponding to each phase
           cut boundary, returning tensor of shape (B, t_phase, N, C).
    """
    if snap_emb.dim() != 4:
        raise ValueError("snap_emb must have shape (B, T_s, N, C)")
    B, _T_s, N, C = snap_emb.shape
    cut_idx, cut_mask = phase_cut_indices(
        phase_of_snap,
        node_mask_s,
        t_phase=t_phase,
    )
    # [한] 컷 인덱스 확장 및 gather 연산을 통한 임베딩 추출
    # [EN] Expand gather index and gather cut embeddings
    gather_idx = cut_idx.view(B, t_phase, 1, 1).expand(B, t_phase, N, C)
    cuts = snap_emb.gather(dim=1, index=gather_idx)
    cuts = cuts * cut_mask.to(dtype=snap_emb.dtype).view(B, t_phase, 1, 1)
    return cuts, cut_mask


class SnapshotPhaseSurvivalGNN(nn.Module):
    """
    [한] 스냅샷 인코더 기반 페이즈 탈락 위험도 예측 GNN (Snapshot Phase Survival GNN).
         - 역할: 10초 주기의 미시적 스냅샷 시퀀스를 인과적으로 인코딩한 후,
           각 페이즈 컷 시점의 상태로부터 다음 페이즈 탈락 위험도(hazard logits)를 직접 예측합니다.
         - 인과성: 시점 k의 컷 상태는 phase <= k 시점까지 관측된 10초 스냅샷들만 참조하므로
           미래 누설 없이 (k+1)번째 페이즈 탈락 위험도를 타당하게 예측합니다.

    [EN] Snapshot-Resolution GNN for Next-Phase Elimination Hazard Prediction.
         - Role: Causally encodes 10s micro-snapshot sequences and directly predicts elimination
           hazard for the upcoming phase from representations gathered at each phase cut.
         - Causality: State at cut k observes snapshots observed through phase k, causally predicting
           elimination during phase k+1 without future leakage.
    """

    def __init__(
        self,
        snap_in_features: int,
        *,
        snap_hidden: int = 48,
        snap_mid: int = 96,
        snap_blocks: int = 2,
        snap_edge_types: int = 2,
        t_phase: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t_phase = t_phase
        # [한] 10초 단위 미시 스냅샷 시공간 인코더
        # [EN] 10-second micro-snapshot spatio-temporal encoder
        self.snap_encoder = SnapshotEncoder(
            in_features=snap_in_features,
            hidden=snap_hidden,
            mid=snap_mid,
            num_blocks=snap_blocks,
            num_edge_types=snap_edge_types,
            dropout=dropout,
        )
        head_hidden = max(snap_hidden // 2, 1)
        # [한] 다음 페이즈 탈락 위험도 예측 헤드
        # [EN] Next-phase elimination hazard prediction head
        self.hazard_head = nn.Sequential(
            nn.Linear(self.snap_encoder.out_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
        *,
        edge_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [한] 위험도 로짓 및 컷 유효성 마스크 반환.
             hazard_logits[:, k]는 페이즈 (k+1)까지 관측된 모든 10초 스냅샷으로부터
             페이즈 (k+2) 동안의 탈락 위험도를 예측합니다.

        [EN] Returns hazard logits and cut validity mask.
             hazard_logits[:, k] predicts elimination during phase k+2 from all snapshots observed through phase k+1.
        """
        snap_emb = self.snap_encoder(X_s, A_s, edge_active_mask=edge_active_mask)
        cut_h, cut_mask = gather_phase_cut_embeddings(
            snap_emb,
            phase_of_snap,
            node_mask_s,
            t_phase=self.t_phase,
        )
        # [한] 마지막 페이즈를 제외한 (T-1)개 전이에 대해 탈락 위험도 산출: (B, t_phase-1, N)
        # [EN] Compute hazard logits for (t_phase - 1) transitions: (B, t_phase-1, N)
        hazard_logits = self.hazard_head(cut_h[:, :-1]).squeeze(-1)
        return hazard_logits, cut_mask[:, :-1]

    def survival_probability(
        self,
        X_s: torch.Tensor,
        A_s: torch.Tensor,
        phase_of_snap: torch.Tensor,
        node_mask_s: torch.Tensor,
        *,
        edge_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [한] 단일 페이즈 생존 확률 반환: P(생존) = 1.0 - sigmoid(hazard).
        [EN] Returns single-phase survival probability: P(survival) = 1.0 - sigmoid(hazard).
        """
        hazard_logits, cut_mask = self(
            X_s,
            A_s,
            phase_of_snap,
            node_mask_s,
            edge_active_mask=edge_active_mask,
        )
        return 1.0 - torch.sigmoid(hazard_logits), cut_mask


def snapshot_phase_hazard_loss(
    hazard_logits: torch.Tensor,
    y_hazard: torch.Tensor,
    target_mask: torch.Tensor,
    cut_mask: torch.Tensor | None = None,
    *,
    hazard_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    [한] 페이즈 컷 위험도 라벨에 대한 마스크 BCE 손실 함수 (Masked BCE Loss).
         - 역할: 유효 타겟 및 유효 페이즈 컷에 대해서만 탈락 위험도 BCE를 계산하고 정확도/양성비율을 집계합니다.

    [EN] Masked BCE over Phase-Cut Next-Phase Hazard Labels.
         - Role: Computes BCE loss exclusively over valid target nodes and valid cut positions,
           tracking batch accuracy and positive sample statistics.
    """
    mask = target_mask.bool()
    if cut_mask is not None:
        mask = mask & cut_mask.unsqueeze(-1).bool()
    mask_f = mask.to(dtype=hazard_logits.dtype)
    denom = mask_f.sum().clamp(min=1.0)

    # [한] 탈락 희귀도 조절용 pos_weight 적용 BCE
    # [EN] BCE with pos_weight for class imbalance
    pos_weight = torch.tensor(
        hazard_pos_weight,
        dtype=hazard_logits.dtype,
        device=hazard_logits.device,
    )
    bce = F.binary_cross_entropy_with_logits(
        hazard_logits,
        y_hazard.to(dtype=hazard_logits.dtype),
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = (bce * mask_f).sum() / denom

    # [한] 모니터링을 위한 정확도 및 양성 샘플 수 산출 (그래디언트 제외)
    # [EN] Compute diagnostic metrics (accuracy and count of positive events) without gradients
    with torch.no_grad():
        probs = torch.sigmoid(hazard_logits)
        pred = probs >= 0.5
        truth = y_hazard >= 0.5
        correct = ((pred == truth) & mask).sum()
        acc = correct.to(dtype=hazard_logits.dtype) / denom
        positives = (y_hazard.to(dtype=hazard_logits.dtype) * mask_f).sum()

    return loss, {
        "loss": float(loss.detach().item()),
        "hazard_acc": float(acc.detach().item()),
        "hazard_pos": float(positives.detach().item()),
        "hazard_n": float(denom.detach().item()),
    }
