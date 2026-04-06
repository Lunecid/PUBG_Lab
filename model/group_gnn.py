"""
Group GNN
==========
팀-팀 상호작용 그래프에서 경쟁 압력을 전파하는 GNN.

생태학 유비: Lotka-Volterra 경쟁 계수 α_ij를 attention으로 학습.
team_graph의 엣지 피처 (최근접 거리, 중심 거리, encounter 수, 데미지 교환)를
사용해 팀 간 경쟁 강도를 attention weight로 표현.

입력: team 임베딩 [n_teams, hidden_dim] + team_graph (Data)
출력: 팀 임베딩 [n_teams, hidden_dim] + α_ij attention weights
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class CompetitionConv(MessagePassing):
    """
    팀-팀 경쟁 메시지 전파.

    attention weight = α_ij (경쟁 계수 해석).
    엣지 피처: [min_dist, center_dist, edge_count, dmg_exchange]
    """

    def __init__(self, hidden_dim, edge_dim=4, n_heads=4):
        super().__init__(aggr="add", flow="source_to_target")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # attention 계산: query/key from node, bias from edge
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, n_heads)

        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Returns:
            out: [n_teams, hidden_dim]
            alpha: [n_edges, n_heads] — attention weights (α_ij 해석용)
        """
        self._alpha = None
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        alpha = self._alpha
        self._alpha = None
        return out, alpha

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # multi-head attention
        q = self.W_q(x_i).view(-1, self.n_heads, self.head_dim)
        k = self.W_k(x_j).view(-1, self.n_heads, self.head_dim)
        v = self.W_v(x_j).view(-1, self.n_heads, self.head_dim)

        # attention score: dot-product with learned temperature
        base_scale = self.head_dim ** 0.5
        effective_scale = base_scale * self.attn_scale.abs().clamp(min=0.01)
        attn = (q * k).sum(dim=-1) / effective_scale  # [E, n_heads]
        edge_bias = self.edge_proj(edge_attr)  # [E, n_heads]
        attn = attn + edge_bias

        # softmax per target node
        from torch_geometric.utils import softmax
        alpha = softmax(attn, index, num_nodes=size_i)  # [E, n_heads]
        self._alpha = alpha.detach()

        # 가중 value
        out = alpha.unsqueeze(-1) * v  # [E, n_heads, head_dim]
        return out.view(-1, self.n_heads * self.head_dim)

    def update(self, aggr_out):
        return self.out_proj(aggr_out)


class GroupGNN(nn.Module):
    """
    팀-팀 상호작용 GNN (2~3층).

    각 레이어에서:
      1. CompetitionConv (multi-head attention + edge features)
      2. LayerNorm + skip connection
      3. FFN

    α_ij = attention weight → 경쟁 계수 해석 (시각화 §7-(3))
    """

    def __init__(
        self,
        hidden_dim=128,
        team_feat_dim=8,
        edge_dim=4,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        # 팀 그래프 노드 피처 → hidden_dim 프로젝션
        # (group_pooling 출력 + team_graph.x 결합)
        self.input_proj = nn.Linear(hidden_dim + team_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(CompetitionConv(hidden_dim, edge_dim, n_heads))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            ))

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, team_h_pooled, team_graph):
        """
        Parameters:
            team_h_pooled: Tensor [n_teams_alive, hidden_dim] — group pooling 출력
            team_graph: Data with:
                - x: [n_teams_alive, team_feat_dim]
                - edge_index: [2, n_edges]
                - edge_attr: [n_edges, edge_dim]

        Returns:
            team_h: Tensor [n_teams_alive, hidden_dim]
            alphas: list of Tensor [n_edges, n_heads] per layer
        """
        n_alive = team_graph.num_nodes
        if n_alive == 0:
            return team_h_pooled, []

        # pooled 임베딩 + team_graph 노드 피처 결합
        # team_h_pooled가 전체 팀(alive+dead) 포함할 수 있으므로,
        # team_graph.alive_teams로 alive 팀만 추출
        alive_teams = team_graph.alive_teams
        if len(alive_teams) > 0 and team_h_pooled.shape[0] > n_alive:
            pooled_alive = team_h_pooled[alive_teams]
        else:
            pooled_alive = team_h_pooled[:n_alive]

        h = self.input_proj(torch.cat([pooled_alive, team_graph.x], dim=-1))

        edge_index = team_graph.edge_index
        edge_attr = team_graph.edge_attr

        alphas = []
        for i in range(self.n_layers):
            if edge_index.shape[1] > 0:
                h_msg, alpha = self.convs[i](h, edge_index, edge_attr)
                alphas.append(alpha)
                h = self.norms[i](h + h_msg)
            else:
                alphas.append(None)
                h = self.norms[i](h)
            h = h + self.ffns[i](h)

        h = self.final_norm(h)
        return h, alphas
