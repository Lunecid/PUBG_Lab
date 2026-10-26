"""
Agent Encoder
==============
HeteroGNN으로 플레이어 그래프를 인코딩.

입력: HeteroData (player 노드 39d, ally/encounter 엣지)
출력: 플레이어별 임베딩 [n_players, hidden_dim]

ally 엣지와 encounter 엣지를 별도 message passing으로 처리한 뒤 합산.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeConv(MessagePassing):
    """엣지 피처를 포함하는 message passing 레이어."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target, x_j: source, edge_attr: 엣지 피처
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(inp)


class AgentEncoder(nn.Module):
    """
    플레이어 그래프 인코더.

    처리 흐름:
      1. 노드 피처 → 초기 임베딩 (MLP)
      2. ally MP + encounter MP (각각 별도 가중치)
      3. 2~3층 반복, skip connection
      4. 출력: [n_players, hidden_dim]
    """

    def __init__(
        self,
        agent_feat_dim=39,
        ally_edge_dim=2,
        encounter_edge_dim=2,
        hidden_dim=128,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 초기 임베딩
        self.input_proj = nn.Sequential(
            nn.Linear(agent_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 각 레이어: ally conv + encounter conv + 결합
        self.ally_convs = nn.ModuleList()
        self.enc_convs = nn.ModuleList()
        self.combine_norms = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()

        for _ in range(n_layers):
            self.ally_convs.append(
                EdgeConv(hidden_dim, ally_edge_dim, hidden_dim)
            )
            self.enc_convs.append(
                EdgeConv(hidden_dim, encounter_edge_dim, hidden_dim)
            )
            self.combine_norms.append(nn.LayerNorm(hidden_dim))
            self.combine_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

    def forward(self, player_graph):
        """
        Parameters:
            player_graph: HeteroData with:
                - player.x: [n, 39]
                - (player, ally, player).edge_index: [2, e_ally]
                - (player, ally, player).edge_attr: [e_ally, 2]
                - (player, encounter, player).edge_index: [2, e_enc]
                - (player, encounter, player).edge_attr: [e_enc, 2]

        Returns:
            h: Tensor [n, hidden_dim]
        """
        x = player_graph["player"].x
        h = self.input_proj(x)

        ally_ei = player_graph["player", "ally", "player"].edge_index
        ally_ea = player_graph["player", "ally", "player"].edge_attr
        enc_ei = player_graph["player", "encounter", "player"].edge_index
        enc_ea = player_graph["player", "encounter", "player"].edge_attr

        for i in range(self.n_layers):
            # ally message
            h_ally = self.ally_convs[i](h, ally_ei, ally_ea) if ally_ei.shape[1] > 0 else torch.zeros_like(h)
            # encounter message
            h_enc = self.enc_convs[i](h, enc_ei, enc_ea) if enc_ei.shape[1] > 0 else torch.zeros_like(h)

            # 합산 + skip connection
            h = h + h_ally + h_enc
            h = self.combine_norms[i](h)
            h = h + self.combine_mlps[i](h)

        return h
