"""
[한] PUBG 팀 중심 및 상대 대형 예측 / 동결 평균 잔차 디퓨전 모델 (Model v4).
    - 핵심 원리:
        • 팀 이동 경로를 '팀 중심(Center) 궤적'과 '팀원 간 상대 대형(Relative Formation)'으로 명시적 분해 (A1 대칭/보존 법칙).
        • 동결된 회귀 모델(FormationRegressor)이 결정론적 평균 경로를 예측하고,
          디퓨전 모델(FormationDiffusion)이 잔차(Residual) 불확실성을 생성적(Generative)으로 샘플링.
    - 인과성 및 정보 누설 방지 (A3 법칙):
        • 미래 시점의 스쿼드 멤버십, 미래 이동 모드, 미래 안전구역(자기장) 정보를 조건 변수로 일절 사용하지 않음.

[EN] Team-Center / Relative-Formation Prediction and Frozen-Mean Residual Diffusion (Model v4).
    - Core Principles:
        • Explicitly decomposes squad trajectories into team macro-center and micro-relative formations (A1 symmetry).
        • Frozen regressor predicts deterministic mean paths, while a structured diffusion model generates residual paths.
    - Strict Causality (A3 Causality):
        • No future team membership, future mobility modes, or future circle parameters are used as conditioning variables.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from pubg_lab.movement.models.model_v3 import JointMotion as Previous


def center_relative(x: torch.Tensor, presence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [한] 팀원 생존 여부(presence)를 반영한 팀 무게중심 및 상대 변위 분해 함수.
         - center: 생존 팀원들의 위치 가중 평균 (팀 전체 거시 중심 좌표)
         - relative: 각 팀원의 팀 중심 대비 상대 변위 벡터 (대형 형태)

    [EN] Decomposes Player Trajectories into Team Center and Relative Formations.
         - center: Presence-weighted average position of active squad members
         - relative: Relative displacement vector of each member from the team center
    """
    w = presence.to(x.dtype)[:, :, None, None]
    center = (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)
    return center, (x - center) * w


def pair_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    scale: float = 0.2,
) -> torch.Tensor:
    """
    [한] 실제 미래 팀원 간 상대 거리 보존 손실 (Pairwise Distance Formation Loss).
         - 역할: 팀원들 사이의 실제 유클리드 거리 차이를 Smooth L1으로 평가하여 스쿼드 대형 형태를 보존합니다.
         - 인덱싱: 4인 스쿼드의 중복 없는 상삼각 쌍 인덱스를 추출하여 상호 유효한 팀원에 대해서만 계산.

    [EN] Actual Future Pairwise Distance Loss for Preserving Squad Formation Shape.
         - Role: Penalizes deviations in pairwise Euclidean distances between squad members via Smooth L1.
         - Indexing: Evaluates upper-triangular pairs across mutually valid squad members.
    """
    pairs = torch.triu(torch.ones(4, 4, device=pred.device, dtype=torch.bool), 1)
    mask = valid[:, :, None] & valid[:, None, :] & pairs[None, :, :, None]
    pd = torch.linalg.vector_norm(pred[:, :, None] - pred[:, None, :], dim=-1)
    td = torch.linalg.vector_norm(truth[:, :, None] - truth[:, None, :], dim=-1)
    error = F.smooth_l1_loss(pd / scale, td / scale, reduction="none")
    return (error * mask).sum() / mask.sum().clamp_min(1)


class FormationRegressor(Previous):
    """
    [한] 팀 중심 및 대형 결정론적 회귀 모델 (Formation Deterministic Regressor).
         - 역할: 맥락 피처(c)로부터 팀 중심 궤적과 팀원별 상대 변위를 예측하여 기준 평균 경로(mean)를 제공합니다.
         - 손실 구성:
           1) point: 개별 플레이어 궤적 Smooth L1 오차
           2) team: 팀 중심 궤적 Smooth L1 오차
           3) mode: 이동 수단(보행/차량) 교차 엔트로피 손실
           4) radial: 안전구역(Bluezone/Whitezone) 경계와의 반경 거리 오차
           5) formation: 팀원 간 상대 거리(pair_loss) 대형 보존 손실

    [EN] Deterministic Squad Formation Regressor.
         - Role: Predicts deterministic team center trajectories and member relative offsets, supplying the baseline mean.
         - Multi-task Loss:
           1) point: Per-player trajectory Smooth L1
           2) team: Team macro-center Smooth L1
           3) mode: Mobility mode cross-entropy
           4) radial: Distance to safe zone circular boundaries
           5) formation: Pairwise inter-member formation distance loss
    """

    def __init__(self, categories) -> None:
        super().__init__(categories, "regression")
        # [한] 풀링된 팀 표현으로부터 팀 중심 이동 경로를 예측하는 헤드 (초기 가중치 0 초기화)
        # [EN] Projection head predicting team macro-center movement (zero-initialized)
        self.center_head = nn.Sequential(nn.Linear(110, 48), nn.SiLU(), nn.Linear(48, 6))
        nn.init.zeros_(self.center_head[-1].weight)
        nn.init.zeros_(self.center_head[-1].bias)

    def mean(self, h: torch.Tensor, c: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        [한] 맥락 임베딩 h로부터 팀 중심과 상대 대형을 결합한 평균 궤적 산출.
        [EN] Computes combined mean trajectory from team center and relative offsets.
        """
        _, relative = center_relative(self.mean_head(h).reshape(-1, 4, 3, 2), c["current_presence"])
        pooled = (h * c["current_presence"][:, :, None]).sum(1) / c["current_presence"].sum(1)[
            :, None
        ].clamp_min(1)
        center = self.center_head(pooled).reshape(-1, 1, 3, 2)
        return self.curve((center + relative) * self.scale, c)

    def loss(self, c: dict[str, torch.Tensor], t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.encode(c)
        mean = self.mean(h, c)
        valid = t["trajectory_valid"] & c["current_presence"][:, :, None]

        # [한] 1. 개별 포인트 Smooth L1 오차
        # [EN] 1. Individual trajectory Smooth L1 error
        point = (
            F.smooth_l1_loss(mean / self.scale, t["trajectory"] / self.scale, reduction="none")
            * valid[..., None]
        ).sum() / (valid.sum() * 2).clamp_min(1)

        # [한] 2. 팀 중심 궤적 오차
        # [EN] 2. Team macro-center error
        center = center_relative(mean, c["current_presence"])[0][:, 0]
        tm = t["team_valid"]
        team = (
            F.smooth_l1_loss(center / self.scale, t["team_center"] / self.scale, reduction="none")
            * tm[..., None]
        ).sum() / (tm.sum() * 2).clamp_min(1)

        # [한] 3. 이동 모드 분류 손실
        # [EN] 3. Mobility mode classification loss
        logits = self.mode_head(h).reshape(-1, 4, 6, 10)
        mm = t["mode_valid"] & valid
        ce = F.cross_entropy(
            logits.flatten(0, 2), t["mode"].flatten(), reduction="none"
        ).reshape_as(valid)
        mode = (ce * mm).sum() / mm.sum().clamp_min(1)

        # [한] 4. 안전구역 원형 경계 반경 거리 오차
        # [EN] 4. Safe zone boundary radial distance error
        radial = point * 0
        for z in range(2):
            geo = c["zone_geometry"][:, :, z * 7 : (z + 1) * 7]
            zc = c["initial_xy"] + geo[:, :, :2]
            pr = torch.linalg.vector_norm(mean - zc[:, :, None], dim=-1) - geo[:, :, 2, None]
            tr = (
                torch.linalg.vector_norm(t["trajectory"] - zc[:, :, None], dim=-1)
                - geo[:, :, 2, None]
            )
            zm = valid & geo[:, :, 6, None].bool()
            radial += (
                F.smooth_l1_loss(pr / self.scale, tr / self.scale, reduction="none") * zm
            ).sum() / zm.sum().clamp_min(1)

        # [한] 5. 팀원 간 상대 거리 대형 보존 손실
        # [EN] 5. Inter-member formation distance preservation loss
        formation = pair_loss(mean, t["trajectory"], valid)

        loss = point + 0.1 * team + 0.05 * mode + 0.05 * radial + 0.2 * formation
        return loss, {
            k: float(v.detach())
            for k, v in dict(
                point=point, team=team, mode=mode, radial=radial, formation=formation
            ).items()
        }


class FormationDiffusion(nn.Module):
    """
    [한] 팀 대형 구조화 잔차 디퓨전 생성 모델 (Structured Formation Residual Diffusion Model).
         - 역할: 동결된 FormationRegressor가 제공하는 결정론적 평균 경로를 기준으로,
           미래 궤적의 잔차(Residual) 분포를 노이즈 제거 과정(Reverse Diffusion)을 통해 조건부 샘플링합니다.
         - 구조화(structured=True) 방식:
           팀 중심 잔차(center_denoiser)와 대형 분산/상대 잔차(relative_denoiser)를 분리하여 디노이징.

    [EN] Structured Formation Residual Diffusion Model.
         - Role: Generates future stochastic trajectory residuals conditioned on frozen regressor baseline means.
         - Structured Mode (structured=True):
           Decouples denoising into team center macro-noise and member-relative formation noise.
    """

    def __init__(self, regressor: FormationRegressor, structured: bool = True) -> None:
        super().__init__()
        self.regressor = regressor
        # [한] 사전학습된 회귀기 가중치 고정 (동결 참조선 원칙 준수)
        # [EN] Freeze regressor weights (strict frozen baseline protocol)
        self.regressor.requires_grad_(False)
        self.structured = structured
        self.steps = 32
        self.register_buffer("center_scale", torch.tensor(0.2))
        self.register_buffer("relative_scale", torch.tensor(0.2))
        self.register_buffer("independent_scale", torch.tensor(0.2))

        def mlp(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 6)
            )

        if structured:
            self.center_denoiser = mlp(116 + 6 + 6 + 16)
            self.relative_denoiser = mlp(116 * 2 + 6 + 6 + 16)
        else:
            self.independent_denoiser = mlp(116 + 6 + 6 + 16)

    @torch.no_grad()
    def context(self, c: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        [한] 동결 회귀기로부터 맥락 표현, 기준 평균 경로 및 팀원 존재 여부 캐시 추출.
        [EN] Extracts context representations, baseline mean paths, and presence flags from frozen regressor.
        """
        h = self.regressor.encode(c)
        mean = self.regressor.mean(h, c)
        displacement = mean[:, :, [1, 3, 5]] - c["initial_xy"][:, :, None]
        return dict(
            h=torch.cat([h, displacement.flatten(2)], -1), mean=mean, presence=c["current_presence"]
        )

    def denoise(self, y: torch.Tensor, step: torch.Tensor, ctx: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        [한] 확산 스텝(step) 및 맥락(ctx) 조건부 노이즈 제거(denoising) 순전파.
        [EN] Conditional denoising forward pass conditioned on diffusion timestep and context.
        """
        h, p = ctx["h"], ctx["presence"]
        # [한] 정현파 시간 스텝 임베딩 (Sinusoidal timestep embedding)
        # [EN] Sinusoidal timestep embedding
        f = torch.exp(torch.arange(8, device=y.device) * (-math.log(10000) / 7))
        e = step[:, None] * f[None]
        e = torch.cat([e.sin(), e.cos()], -1)
        pooled_h = (h * p[:, :, None]).sum(1) / p.sum(1)[:, None].clamp_min(1)

        if self.structured:
            center, relative = y[:, :1], center_relative(y[:, 1:], p)[1]
            spread = (relative.square() * p[:, :, None, None]).sum(1) / p.sum(1)[
                :, None, None
            ].clamp_min(1)
            pc = self.center_denoiser(
                torch.cat([pooled_h, center.flatten(1), spread.flatten(1), e], -1)
            ).reshape(-1, 1, 3, 2)
            pr = self.relative_denoiser(
                torch.cat(
                    [
                        h,
                        pooled_h[:, None].expand(-1, 4, -1),
                        center.flatten(2).expand(-1, 4, -1),
                        relative.flatten(2),
                        e[:, None].expand(-1, 4, -1),
                    ],
                    -1,
                )
            ).reshape(-1, 4, 3, 2)
            return torch.cat([pc, center_relative(pr, p)[1]], 1)

        pooled = center_relative(y, p)[0].flatten(2).expand(-1, 4, -1)
        return (
            self.independent_denoiser(
                torch.cat([h, y.flatten(2), pooled, e[:, None].expand(-1, 4, -1)], -1)
            ).reshape(-1, 4, 3, 2)
            * p[:, :, None, None]
        )

    def encode_residual(self, residual: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        if not self.structured:
            return residual / self.independent_scale
        center, relative = center_relative(residual, presence)
        return torch.cat([center / self.center_scale, relative / self.relative_scale], 1)

    def decode_residual(self, latent: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        if not self.structured:
            return latent * self.independent_scale
        return (
            latent[:, :1] * self.center_scale
            + center_relative(latent[:, 1:], presence)[1] * self.relative_scale
        )

    def noise(self, ctx: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        [한] 팀 중심 및 상대 대형의 사전 가우시안 노이즈 텐서 생성.
        [EN] Generates prior Gaussian noise tensor for team center and member relatives.
        """
        p = ctx["presence"]
        if self.structured:
            center = torch.randn(len(p), 1, 3, 2, device=p.device)
            relative = center_relative(torch.randn(len(p), 4, 3, 2, device=p.device), p)[1]
            return torch.cat([center, relative], 1)
        return torch.randn(len(p), 4, 3, 2, device=p.device) * p[:, :, None, None]

    def loss_cached(self, ctx: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        [한] x0-예측 기반 디퓨전 재구성 손실 및 대형 보존 손실 계산.
        [EN] Evaluates x0-prediction diffusion reconstruction loss and pair formation loss.
        """
        p = ctx["presence"]
        km = target["trajectory_valid"][:, :, [1, 3, 5]] & p[:, :, None]
        complete = (km | ~p[:, :, None]).all(1)
        valid = complete[:, None] & p[:, :, None]
        residual = torch.where(
            valid[..., None],
            target["trajectory"][:, :, [1, 3, 5]] - ctx["mean"][:, :, [1, 3, 5]],
            0.0,
        )
        clean = self.encode_residual(residual, p)
        step = torch.randint(self.steps, (len(p),), device=p.device)
        ab = self.regressor.encoder.alpha_bar[step][:, None, None, None]
        noise = self.noise(ctx)
        noisy = ab.sqrt() * clean + (1 - ab).sqrt() * noise
        output = self.denoise(noisy, step, ctx)
        latent_mask = torch.cat([complete[:, None], valid], 1) if self.structured else valid
        reconstruction = ((output - clean).square() * latent_mask[..., None]).sum() / (
            2 * latent_mask.sum()
        ).clamp_min(1)
        formation = reconstruction * 0
        if self.structured:
            path = ctx["mean"][:, :, [1, 3, 5]] + self.decode_residual(output, p)
            formation = pair_loss(path, target["trajectory"][:, :, [1, 3, 5]], valid)
        loss = reconstruction + 0.2 * formation
        return loss, dict(
            reconstruction=float(reconstruction.detach()), formation=float(formation.detach())
        )

    @torch.no_grad()
    def sample(self, c: dict[str, torch.Tensor], scenarios: int = 20) -> torch.Tensor:
        """
        [한] DDIM 역방향 디퓨전 샘플링을 통해 다중 미래 시나리오 경로 생성.
             - scenarios: 생성할 확률적 미래 경로 시나리오 개수 (기본 20개).

        [EN] Generates Stochastic Multi-Scenario Trajectories via Reverse Diffusion Sampling.
             - scenarios: Number of stochastic future trajectory trajectories to sample (default 20).
        """
        ctx0 = self.context(c)
        ctx = {k: v.repeat_interleave(scenarios, 0) for k, v in ctx0.items()}
        y = self.noise(ctx)
        for step in reversed(range(self.steps)):
            a = self.regressor.encoder.alpha_bar[step]
            prev = (
                self.regressor.encoder.alpha_bar[step - 1]
                if step
                else torch.tensor(1.0, device=y.device)
            )
            clean = self.denoise(y, torch.full((len(y),), step, device=y.device), ctx)
            eps = (y - a.sqrt() * clean) / (1 - a).sqrt().clamp_min(1e-6)
            y = prev.sqrt() * clean + (1 - prev).sqrt() * eps
        residual = self.decode_residual(y, ctx["presence"])
        residual = torch.einsum("tk,bpkd->bptd", self.regressor.basis, residual).reshape(
            len(ctx0["h"]), scenarios, 4, 6, 2
        )
        return ctx0["mean"][:, None] + residual
