"""
Group Pooling
==============
Attention 기반으로 agent 임베딩 → team 임베딩으로 풀링.

생태학 유비: 개체(individual) → 종/개체군(population) 수준 집계.
팀 내 기여도가 다른 플레이어를 attention으로 가중 평균.

입력: agent 임베딩 [n_players, hidden_dim] + team 마스크
출력: team 임베딩 [n_teams, hidden_dim]
"""

import torch
import torch.nn as nn


class GroupPooling(nn.Module):
    """
    Attention-weighted agent → group 풀링.

    각 팀에 대해:
      1. 팀 소속 agent 임베딩 추출
      2. attention score 계산 (MLP → scalar)
      3. softmax → 가중 평균
      4. 결과 = 팀 표현 벡터
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # attention score 계산
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 풀링 후 변환
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, agent_h, team_idx, n_teams):
        """
        Parameters:
            agent_h: Tensor [n_agents, hidden_dim] — agent encoder 출력
            team_idx: Tensor [n_agents] — 각 agent의 팀 인덱스 (글로벌)
            n_teams: int — 총 팀 수

        Returns:
            team_h: Tensor [n_teams, hidden_dim] — 팀 임베딩
            attn_weights: Tensor [n_agents] — attention 가중치 (해석용)
        """
        device = agent_h.device
        n_agents = agent_h.shape[0]

        # attention 점수 계산
        attn_scores = self.attn(agent_h).squeeze(-1)  # [n_agents]

        # 팀별 softmax
        attn_weights = torch.zeros(n_agents, device=device)
        team_h = torch.zeros(n_teams, self.hidden_dim, device=device)

        for t in range(n_teams):
            mask = (team_idx == t)
            if mask.sum() == 0:
                continue

            scores = attn_scores[mask]
            weights = torch.softmax(scores, dim=0)
            attn_weights[mask] = weights

            members = agent_h[mask]  # [k, hidden_dim]
            team_h[t] = (weights.unsqueeze(-1) * members).sum(dim=0)

        team_h = self.output_proj(team_h)
        return team_h, attn_weights


class GroupPoolingScatter(nn.Module):
    """
    scatter 기반 효율적 구현 (배치 처리 최적화).
    torch_scatter가 있으면 사용, 없으면 loop fallback.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, agent_h, team_idx, n_teams):
        """scatter 기반 풀링. API는 GroupPooling과 동일."""
        device = agent_h.device

        attn_logits = self.attn(agent_h).squeeze(-1)  # [n_agents]

        # 팀별 softmax: max-subtract for numerical stability
        max_per_team = torch.full((n_teams,), float('-inf'), device=device)
        max_per_team.scatter_reduce_(0, team_idx, attn_logits, reduce='amax', include_self=False)
        attn_stable = attn_logits - max_per_team[team_idx]
        attn_exp = attn_stable.exp()

        # 팀별 합
        sum_exp = torch.zeros(n_teams, device=device)
        sum_exp.scatter_add_(0, team_idx, attn_exp)
        attn_weights = attn_exp / (sum_exp[team_idx] + 1e-8)

        # 가중 합
        weighted = agent_h * attn_weights.unsqueeze(-1)  # [n_agents, hidden_dim]
        team_h = torch.zeros(n_teams, self.hidden_dim, device=device)
        team_h.scatter_add_(0, team_idx.unsqueeze(-1).expand_as(weighted), weighted)

        team_h = self.output_proj(team_h)
        return team_h, attn_weights
