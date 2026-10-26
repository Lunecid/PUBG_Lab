"""
[한] PUBG 월드 모델 G2 사전학습 손실 및 헤드 모듈 (World Model G2 Pretraining Heads and Losses).
    - 참조 문헌:
        • 타입화 표현: Hu et al. 2020, HGT, arXiv:2003.01332
        • 마스킹 재구성(Masked Reconstruction): Hou et al. 2022, GraphMAE, arXiv:2205.10803 (Sec 3.2)
        • 맥락 사전학습(Context Pretraining): Forecast-MAE, arXiv:2308.09882 (Sec 3)
    - 역할:
        CPU 환경에서 계산 가능한 범위 내에서 다음 4가지 보조 사전학습 목표를 최적화:
        1) P1: 체력 마스킹 재구성 손실 (Health Masked MSE)
        2) P2: 결합 GMM 이동 궤적 NLL 손실 (Joint GMM Motion NLL)
        3) P3: 이동 모드 분류 손실 (Mobility Mode Cross Entropy)
        4) Pair: 팀원/상대 개체 간 상대 거리 보존 손실 (Pairwise Distance MSE)

[EN] Bounded CPU Pretraining Heads and Losses for G2 Feasibility Verification.
    - References:
        • Typed representations: Hu et al. 2020, HGT, arXiv:2003.01332
        • Masked reconstruction: Hou et al. 2022, GraphMAE, arXiv:2205.10803 (Sec 3.2)
        • Context pretraining: Forecast-MAE, arXiv:2308.09882 (Sec 3)
    - Role:
        Optimizes four pretraining proxy objectives within CPU memory and time budgets:
        1) P1: Masked health reconstruction (Health MSE)
        2) P2: Joint GMM trajectory distribution (Motion NLL)
        3) P3: Mobility transit mode classification (Cross Entropy)
        4) Pair: Inter-agent pairwise distance preservation (Pairwise MSE)
"""

from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """
    [한] 마스크된 유효 영역에 대해서만 계산하는 평균 제곱 오차 (Masked Mean Squared Error).
         - 유효한 마스크가 하나도 없는 경우 None을 반환하여 손실 계산에서 안전하게 배제.

    [EN] Masked Mean Squared Error.
         - Returns None if no valid masked elements exist, safely excluding term from loss.
    """
    mask = torch.broadcast_to(mask, prediction.shape)
    if not mask.any():
        return None
    return (prediction[mask] - target[mask]).square().mean()


def joint_gmm_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """
    [한] 가우시안 혼합 모델(GMM) 기반의 결합 이동 궤적 음의 로그 우도 (Joint Marginal GMM NLL).
         - 텐서 형상: mean (M, K, H, 2), log_std (M, K, H, 2), logits (K,), target (M, H, 2), mask (M, H)
           (M: 대상 개체 수, K: 가우시안 성분 수, H: 미래 예측 지평선, 2: 2D 델타 좌표)
         - 수치 안정성: 결측 라벨은 산술 연산 전에 0으로 대체(NaN*0 누수 방지).
           각 혼합 성분 내부에서 관측 차원들을 합산한 후 성분 간 log-sum-exp를 적용하여 최종 NLL 계산.

    [EN] Joint Marginal Gaussian Mixture Model (GMM) Negative Log-Likelihood.
         - Shapes: mean (M, K, H, 2), log_std (M, K, H, 2), logits (K,), target (M, H, 2), mask (M, H)
         - Numerical Stability: Replaces missing labels before arithmetic to avoid NaN*0 propagation;
           sums observed dimensions within each component, then applies log-sum-exp over mixture components.
    """
    if not mask.any():
        return None
    observed = mask[:, None, :, None].expand_as(mean)
    y = torch.where(mask[:, :, None], target, torch.zeros_like(target))
    terms = (
        0.5 * ((y[:, None] - mean) * torch.exp(-log_std)).square()
        + log_std
        + 0.5 * math.log(2 * math.pi)
    )
    nll_component = torch.where(observed, terms, torch.zeros_like(terms)).sum((0, 2, 3))
    return -torch.logsumexp(logits.log_softmax(0) - nll_component, 0) / (mask.sum() * 2)


def pair_targets(xy: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [한] 모든 개체 쌍(pair) 간의 유클리드 거리 및 동시 유효 마스크 산출 함수.
         - triu_indices를 이용해 중복 없는 상삼각 쌍 인덱스를 추출하고 상대 거리를 계산합니다.

    [EN] Computes Pairwise Euclidean Distances and Joint Validity Mask between Entity Pairs.
         - Extracts unique upper-triangular pair combinations and calculates inter-entity distances.
    """
    pairs = torch.triu_indices(len(xy), len(xy), 1, device=xy.device)
    clean = torch.where(mask[:, :, None], xy, torch.zeros_like(xy))
    distance = (clean[pairs[0]] - clean[pairs[1]]).norm(dim=-1)
    return distance, mask[pairs[0]] & mask[pairs[1]], pairs


def corrupt_health(g, seed: int) -> tuple:
    """
    [한] P1 사전학습용 체력 마스킹 오염 함수 (Health History Masking for GraphMAE Pretraining).
         - 역할: 선정된 초점 플레이어들의 전체 체력(HP) 이력을 0으로 마스킹하여,
           인코더가 주변 맥락과 이전 교전 상태로부터 체력을 복원하도록 학습시킵니다.
         - 반환: 오염된 입력 g_corrupted, 원래 체력 타겟 target, 마스킹된 플레이어 불리언 마스크 selected.

    [EN] Health History Masking for P1 GraphMAE Reconstruction Pretraining.
         - Role: Masks the entire health trajectory of selected focal actors to zero, training
           the encoder to reconstruct actor state from spatial context and interaction history.
         - Returns: Corrupted inputs, true target health values, and selected actor boolean mask.
    """
    generator = torch.Generator().manual_seed(seed)
    candidates = g.target_player_indices
    candidates = candidates[g.mask[-1, candidates, 5]]
    selected = torch.zeros(len(g.actor_ids), dtype=torch.bool)
    if len(candidates):
        choose = torch.rand(len(candidates), generator=generator) < 0.3
        if not choose.any():
            choose[torch.randint(len(candidates), (1,), generator=generator)] = True
        selected[candidates[choose]] = True
    x, mask = g.x.clone(), g.mask.clone()
    target = x[-1, :, 5].clone()
    x[:, selected, 5] = 0
    mask[:, selected, 5] = False
    return replace(g, x=x, mask=mask), target, selected


def supervised_targets(sample) -> dict[str, torch.Tensor]:
    """
    [한] 지도학습 궤적 및 이동 모드 정답 텐서 구성 함수.
         - 원점 기준 100m 정규화 델타 좌표, 유효 마스크, 이동 모드 라벨 추출.

    [EN] Prepares Supervised Future Trajectory and Transit Mode Ground Truth Tensors.
         - Extracts 100m-normalized delta coordinates, validity masks, and discrete mobility modes.
    """
    g = sample.inputs
    rows = sample.targets["selected_rows"]
    ids = g.target_player_indices
    raw = torch.as_tensor(sample.targets["xy"][rows], dtype=torch.float32)
    mask = torch.as_tensor(sample.targets["mask"][rows], dtype=torch.bool)
    xy = (raw - g.origin_m) / 100
    delta = xy - g.x[-1, ids, None, :2]
    xy = torch.where(mask[:, :, None], xy, torch.zeros_like(xy))
    delta = torch.where(mask[:, :, None], delta, torch.zeros_like(delta))
    mode = torch.as_tensor(sample.targets["mode"][rows], dtype=torch.long)
    return {"xy": xy, "delta": delta, "mask": mask, "mode": mode, "mode_mask": mask & (mode > 0)}


class PretrainHeads(nn.Module):
    """
    [한] 월드 모델 다목적 사전학습 헤드 (Multi-task Pretraining Heads).
         - health: 마스킹된 체력 복원 선형 헤드 (P1)
         - motion: GMM 기반 미래 이동 궤적 예측 헤드 (K개 성분, H개 스텝, 평균 2D + 로그 표준편차 2D) (P2)
         - mixture: 팀 및 게임 컨텍스트를 입력받는 GMM 성분 가중치 로짓 헤드
         - mode: 미래 이동 수단(도보, 차량 등 10개 클래스) 분류 헤드 (P3)
         - pair: 개체 간 상대 거리 예측 헤드

    [EN] Multi-task Pretraining Prediction Heads.
         - health: Linear head for masked health reconstruction (P1)
         - motion: Trajectory GMM predicting 2D mean and log-std across K components and H steps (P2)
         - mixture: Mixture component logits conditioned on team and game context
         - mode: Future mobility mode classification logits (P3)
         - pair: Pairwise distance prediction head preserving relative entity geometry
    """

    def __init__(self, d: int = 128, components: int = 4, horizon: int = 6) -> None:
        super().__init__()
        self.components, self.horizon = components, horizon
        self.health = nn.Linear(d, 1)
        self.motion = nn.Linear(d, components * horizon * 4)
        self.mixture = nn.Linear(d * 2, components)
        self.mode = nn.Linear(d, horizon * 10)
        self.pair = nn.Sequential(nn.Linear(d * 3 + 1, d), nn.GELU(), nn.Linear(d, horizon))

    def forward(self, latent: dict[str, torch.Tensor], g) -> dict[str, torch.Tensor]:
        ids = g.target_player_indices
        player = latent["player"][ids]
        own = g.members[g.members[:, 0] == g.ego_local_index, 1]
        team = latent["team"][own[0]] if len(own) else player.mean(0)

        # [한] GMM 이동 궤적 평균 및 클리핑된 로그 표준편차 분리
        # [EN] Unpack GMM trajectory means and clamped log-std
        motion = self.motion(player).reshape(len(ids), self.components, self.horizon, 4)
        mean = motion[..., :2]

        # [한] 개체 쌍 간의 상대 거리 예측을 위한 입력 벡터 구성 (대칭성 반영)
        # [EN] Construct pairwise input vectors reflecting symmetric relations
        pairs = torch.triu_indices(len(ids), len(ids), 1, device=player.device)
        i, j = pairs
        distance = (g.x[-1, ids[i], :2] - g.x[-1, ids[j], :2]).norm(dim=-1, keepdim=True)
        pair_input = torch.cat(
            [
                player[i] + player[j],
                (player[i] - player[j]).abs(),
                team.expand(len(i), -1),
                distance,
            ],
            -1,
        )

        return {
            "mean": mean,
            "log_std": motion[..., 2:].clamp(-4, 4),
            "logits": self.mixture(torch.cat([team, latent["game"]])),
            "mode": self.mode(player).reshape(len(ids), self.horizon, 10),
            "pair": F.softplus(self.pair(pair_input)),
        }


def pretrain_losses(encoder: nn.Module, heads: PretrainHeads, sample, seed: int) -> tuple:
    """
    [한] 복합 사전학습 손실 계산 함수 (Combined Multi-task Pretraining Loss).
         - 총 손실 = 1.0 * L_health + 1.0 * L_motion + 0.1 * L_mode + 0.1 * L_pair
         - 유효한 손실 항목들만 동적으로 가중합을 산출하여 안정적 최적화 보장.

    [EN] Computes Combined Multi-Task Pretraining Losses.
         - Total = 1.0 * L_health + 1.0 * L_motion + 0.1 * L_mode + 0.1 * L_pair
         - Dynamically aggregates active loss terms with sample support tracking.
    """
    g = sample.inputs
    y = supervised_targets(sample)
    out = heads(encoder(g), g)

    # [한] P1 체력 마스킹 오염 및 복원 손실
    # [EN] P1 Masked health corruption and reconstruction loss
    masked, hp, selected = corrupt_health(g, seed)
    hp_pred = heads.health(encoder(masked)["player"]).squeeze(-1)

    # [한] 쌍 거리 타겟 및 개별 손실 항목 산출
    # [EN] Pairwise distance targets and individual loss term evaluation
    pair_y, pair_mask, _ = pair_targets(y["xy"], y["mask"])
    losses = {
        "health": masked_mse(hp_pred, hp, selected),
        "motion": joint_gmm_nll(out["mean"], out["log_std"], out["logits"], y["delta"], y["mask"]),
        "mode": F.cross_entropy(out["mode"][y["mode_mask"]], y["mode"][y["mode_mask"]])
        if y["mode_mask"].any()
        else None,
        "pair": masked_mse(out["pair"], pair_y, pair_mask),
    }
    weights = {"health": 1.0, "motion": 1.0, "mode": 0.1, "pair": 0.1}
    active = [weights[k] * v for k, v in losses.items() if v is not None]
    total = sum(active) if active else None
    support = {
        "health": int(selected.sum()),
        "motion": int(y["mask"].sum()),
        "mode": int(y["mode_mask"].sum()),
        "pair": int(pair_mask.sum()),
    }
    return total, losses, support


class PathProbe(nn.Module):
    """
    [한] 표현력 검증을 위한 동결 인코더 경로 선형 탐침 (Linear Path Probe for Frozen Latent Representations).
         - 역할: 동결된 월드 모델 인코더의 플레이어 잠재 표현에 단순 선형 계층만을 연결하여
           표현 자체에 내재된 미래 궤적 정보의 풍부도를 검증(Probe)합니다.

    [EN] Linear Path Probe for Evaluating Frozen Latent Representations.
         - Role: Evaluates trajectory information quality in frozen representations via a single linear projection.
    """

    def __init__(self, d: int = 128, horizon: int = 6) -> None:
        super().__init__()
        self.horizon = horizon
        self.linear = nn.Linear(d, horizon * 2)

    def forward(self, latent: dict[str, torch.Tensor], g) -> torch.Tensor:
        ids = g.target_player_indices
        return self.linear(latent["player"][ids]).reshape(len(ids), self.horizon, 2)
