"""
Temporal Module
================
GRU로 L 스텝의 팀 임베딩 시퀀스를 처리.

시간 축을 따라 hazard trajectory를 모델링:
  h_team(t-L+1), ..., h_team(t) → GRU → h_temporal(t)

zone context 시퀀스도 함께 입력하여 phase 변화를 반영.
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    GRU 기반 시간 인코더.

    입력: 팀 임베딩 시퀀스 [L, hidden_dim] + zone 시퀀스 [L, zone_dim]
    출력: 시간 집약 표현 [hidden_dim]
    """

    def __init__(
        self,
        hidden_dim=128,
        zone_dim=10,
        n_gru_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # team embedding 정규화 (zone_proj와 scale 맞춤)
        self.team_norm = nn.LayerNorm(hidden_dim)

        # zone context 프로젝션
        self.zone_proj = nn.Linear(zone_dim, hidden_dim // 4)

        # GRU 입력: team_h + zone_proj
        gru_input_dim = hidden_dim + hidden_dim // 4
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, team_h_seq, zone_seq):
        """
        Parameters:
            team_h_seq: Tensor [L, hidden_dim] — L 스텝의 팀 임베딩
            zone_seq: Tensor [L, zone_dim] — L 스텝의 zone context

        Returns:
            h_out: Tensor [hidden_dim] — 시간 집약 표현
        """
        L = team_h_seq.shape[0]

        # team embedding 정규화 + zone 프로젝션
        team_h_normed = self.team_norm(team_h_seq)
        z = self.zone_proj(zone_seq)  # [L, hidden_dim//4]

        # GRU 입력 조립
        gru_input = torch.cat([team_h_normed, z], dim=-1)  # [L, gru_input_dim]
        gru_input = gru_input.unsqueeze(0)  # [1, L, gru_input_dim]

        output, h_n = self.gru(gru_input)  # output: [1, L, hidden_dim]
        h_out = h_n[-1, 0]  # 마지막 GRU layer의 마지막 hidden state

        return self.output_norm(h_out)

    def forward_batch(self, team_h_seqs, zone_seqs):
        """
        배치 처리.

        Parameters:
            team_h_seqs: Tensor [B, L, hidden_dim]
            zone_seqs: Tensor [B, L, zone_dim]

        Returns:
            h_out: Tensor [B, hidden_dim]
        """
        team_h_normed = self.team_norm(team_h_seqs)
        z = self.zone_proj(zone_seqs)  # [B, L, hidden_dim//4]
        gru_input = torch.cat([team_h_normed, z], dim=-1)  # [B, L, gru_input_dim]

        output, h_n = self.gru(gru_input)  # h_n: [n_layers, B, hidden_dim]
        h_out = h_n[-1]  # [B, hidden_dim]

        return self.output_norm(h_out)
