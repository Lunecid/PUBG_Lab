"""
Hazard Head
============
Phase-conditioned discrete-time hazard 예측.

h_i(k) = P(T_i = k | T_i >= k, x_i(k))

같은 팀 상태라도 1페이즈의 200m 이동 vs 최종 페이즈의 200m 이동은 위험이 다름.
zone phase가 baseline hazard 변화를 명시적으로 조건화.

입력: temporal 출력 [hidden_dim] + phase context
출력: hazard logit (scalar), risk score (scalar)
"""

import torch
import torch.nn as nn


class HazardHead(nn.Module):
    """
    Phase-conditioned hazard prediction head.

    조건화 정보:
      - zone_phase: 현재 자기장 페이즈 (임베딩)
      - safe_area_ratio: 안전구역 비율
      - alive_ratio: 생존 비율
      - game_progress: 경기 진행도

    출력:
      - hazard_logit: sigmoid 전 hazard 확률 (BCE loss용)
      - risk_score: ranking loss용 연속값
    """

    def __init__(
        self,
        hidden_dim=128,
        max_phases=9,
        phase_embed_dim=16,
        n_context_features=4,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # phase 임베딩
        self.phase_embed = nn.Embedding(max_phases + 1, phase_embed_dim)

        # context 프로젝션 (safe_area_ratio, alive_ratio, game_progress, density)
        context_dim = phase_embed_dim + n_context_features
        self.context_proj = nn.Linear(context_dim, hidden_dim // 4)

        # hazard 예측 MLP
        head_input_dim = hidden_dim + hidden_dim // 4
        self.hazard_mlp = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # risk score (ranking 보조 loss용, hazard와 별도 head)
        self.risk_mlp = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_temporal, zone_phase, context_features):
        """
        Parameters:
            h_temporal: Tensor [hidden_dim] — temporal encoder 출력
            zone_phase: int or Tensor — 현재 zone phase (0~max_phases)
            context_features: Tensor [4] — [safe_area_ratio, alive_ratio,
                                             game_progress, density]

        Returns:
            hazard_logit: Tensor [1] — sigmoid 전 hazard
            risk_score: Tensor [1] — ranking용 risk
        """
        # phase 임베딩
        if isinstance(zone_phase, int):
            zone_phase = torch.tensor([zone_phase], device=h_temporal.device)
        phase_emb = self.phase_embed(zone_phase.clamp(max=self.phase_embed.num_embeddings - 1))
        if phase_emb.dim() > 1:
            phase_emb = phase_emb.squeeze(0)

        # context 조립
        ctx = torch.cat([phase_emb, context_features], dim=-1)
        ctx_proj = self.context_proj(ctx)

        # 결합
        combined = torch.cat([h_temporal, ctx_proj], dim=-1)

        hazard_logit = self.hazard_mlp(combined).squeeze(-1)
        risk_score = self.risk_mlp(combined).squeeze(-1)

        return hazard_logit, risk_score

    def forward_batch(self, h_temporal, zone_phases, context_features):
        """
        배치 처리.

        Parameters:
            h_temporal: Tensor [B, hidden_dim]
            zone_phases: Tensor [B] (int)
            context_features: Tensor [B, 4]

        Returns:
            hazard_logits: Tensor [B]
            risk_scores: Tensor [B]
        """
        phase_emb = self.phase_embed(
            zone_phases.clamp(max=self.phase_embed.num_embeddings - 1)
        )  # [B, phase_embed_dim]

        ctx = torch.cat([phase_emb, context_features], dim=-1)  # [B, context_dim]
        ctx_proj = self.context_proj(ctx)  # [B, hidden_dim//4]

        combined = torch.cat([h_temporal, ctx_proj], dim=-1)  # [B, head_input_dim]

        hazard_logits = self.hazard_mlp(combined).squeeze(-1)  # [B]
        risk_scores = self.risk_mlp(combined).squeeze(-1)  # [B]

        return hazard_logits, risk_scores
