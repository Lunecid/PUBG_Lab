"""Untrained G1 typed temporal/set encoder; no decoder or optimizer.

Typed messages: inspired by HGT (Hu et al., arXiv:2003.01332), not its reproduction.
Joint entity-set interface: MotionDiffuser (Jiang et al., arXiv:2306.03083).
Mathematical contract and limitations: WORLD_MODEL_G1_IMPLEMENTATION_2026-10-20.md.
"""

import torch
from torch import nn

from .inputs import EncoderInput


def mean_to(messages, destinations, size):
    out = messages.new_zeros((size, messages.shape[-1]))
    counts = messages.new_zeros((size, 1))
    if len(messages):
        out.index_add_(0, destinations, messages)
        counts.index_add_(0, destinations, messages.new_ones((len(messages), 1)))
    return out / counts.clamp_min(1)


class TemporalBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.heads = heads

    def forward(self, x, valid):
        # x=P,L,d. Invalid queries attend to themselves to avoid all-masked NaNs;
        # they are then zeroed, and never serve as keys for valid queries.
        P, L, _ = x.shape
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None] & valid[:, None, :]
        allowed = allowed | (
            torch.eye(L, dtype=torch.bool, device=x.device)[None] & ~valid[:, :, None]
        )
        blocked = (~allowed).repeat_interleave(self.heads, dim=0)
        h, _ = self.attention(x, x, x, attn_mask=blocked, need_weights=False)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x * valid[:, :, None]


class SpatialBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.observed = nn.Linear(d, d, bias=False)
        self.hypothetical = nn.Linear(d, d, bias=False)
        self.mode = nn.Embedding(10, d)
        self.support = nn.Linear(2, d, bias=False)
        self.hypothesis_gate = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(d)

    def forward(self, cells, observed, inferred):
        if not len(cells):
            return cells
        update = torch.zeros_like(cells)
        if len(observed):
            src, dst, mode, n, matches = observed.unbind(1)
            support = torch.stack([n, matches], -1).float()
            # Evidence-strength scaling, not a calibrated passability probability.
            weight = (matches.float() / (matches + 1)).unsqueeze(-1)
            msg = (
                self.observed(cells[src]) + self.mode(mode) + self.support(torch.log1p(support))
            ) * weight
            update = update + mean_to(msg, dst, len(cells))
        if len(inferred):
            src, dst = inferred.unbind(1)
            msg = self.hypothetical(cells[src]) * self.hypothesis_gate.sigmoid()
            update = update + mean_to(msg, dst, len(cells))
        return self.norm(cells + update)


class WorldEncoder(nn.Module):
    def __init__(self, item_count, condition_sizes, d=128, heads=4):
        super().__init__()
        self.d = d
        self.player_input = nn.Linear(36, d)
        self.equipment = nn.Embedding(item_count, d, padding_idx=0)
        self.mobility = nn.Embedding(10, d, padding_idx=0)
        self.confidence = nn.Linear(2, d, bias=False)
        self.contact = nn.Linear(6, d, bias=False)
        self.time = nn.Linear(1, d, bias=False)
        self.temporal = nn.ModuleList([TemporalBlock(d, heads) for _ in range(2)])
        self.game_input = nn.Linear(18, d)
        self.game_temporal = TemporalBlock(d, heads)
        self.conditions = nn.ModuleList(
            [nn.Embedding(n, d, padding_idx=0) for n in condition_sizes]
        )
        self.zone_input = nn.Linear(6, d)
        self.zone_relative = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.cell_input = nn.Linear(8, d)
        self.spatial = nn.ModuleList([SpatialBlock(d) for _ in range(2)])
        self.cell_to_player = nn.Linear(d, d, bias=False)
        self.context_player = nn.Linear(d, d, bias=False)
        self.context_geometry = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.context_score = nn.Sequential(nn.Tanh(), nn.Linear(d, 1))
        self.team_member = nn.Linear(d, d)
        # Ordered relative vectors, squared terms and norm retain formation shape.
        self.team_pair = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.team_count = nn.Linear(3, d)
        self.team_norm = nn.LayerNorm(d)
        self.team_to_player = nn.Linear(d, d, bias=False)
        self.player_norm = nn.LayerNorm(d)

    def player_context(self, player, positions, valid):
        """All currently permitted entities, O(P²d); no server-only lookup here.

        Observer has own-team HUD proxies only. Server additionally has opponents.
        Shared pair operators preserve player permutation equivariance.
        """
        indices = torch.nonzero(valid, as_tuple=True)[0]
        if len(indices) < 2:
            return torch.zeros_like(player)
        points = positions[indices]
        delta = points[None] - points[:, None]  # destination query -> source entity
        relation = torch.cat([delta, delta.norm(dim=-1, keepdim=True)], -1)
        messages = self.context_player(player[indices])[None] + self.context_geometry(relation)
        scores = self.context_score(messages).squeeze(-1)
        scores = scores.masked_fill(
            torch.eye(len(indices), dtype=torch.bool, device=player.device), -torch.inf
        )
        context = (scores.softmax(-1)[:, :, None] * messages).sum(1)
        return torch.zeros_like(player).index_copy(0, indices, context)

    def encode_spatial(self, g):
        cells = self.cell_input(g.cells)
        for block in self.spatial:
            cells = block(cells, g.observed, g.inferred)
        return cells

    def forward(self, inputs):
        if not isinstance(inputs, EncoderInput):
            raise TypeError("WorldEncoder accepts EncoderInput only, never a target dictionary")
        g = inputs
        L, P, _ = g.x.shape
        valid = g.history_valid
        x = self.player_input(torch.cat([g.x, g.mask.float(), g.absolute_xy], -1))
        item_context = x.new_zeros((L * P, self.d))
        if len(g.items):
            frame, player, token = g.items.unbind(1)
            item_context = mean_to(self.equipment(token), frame * P + player, L * P)
        x = (
            x
            + item_context.reshape(L, P, self.d)
            + self.mobility(g.mobility) * g.mobility_mask[:, :, None]
            + self.confidence(g.confidence)
            + self.contact(torch.cat([g.x[:, :, 9:12], g.mask[:, :, 9:12].float()], -1))
        )
        time = torch.arange(1 - L, 1, dtype=x.dtype, device=x.device).reshape(L, 1) / 12
        x = x + self.time(time)[:, None, :]
        x = x.transpose(0, 1)
        for block in self.temporal:
            x = block(x, valid.T)
        player = x[:, -1]
        cells = self.encode_spatial(g)
        if len(g.located):
            p, c = g.located.unbind(1)
            player = player + mean_to(self.cell_to_player(cells[c]), p, P)

        game_history = self.game_input(torch.cat([g.game, g.game_mask.float()], -1))
        game_history = game_history + self.time(time)
        game = self.game_temporal(game_history[None], g.game_mask.any(-1)[None])[0, -1]
        for j, embed in enumerate(self.conditions):
            game = game + embed(g.conditions[j])
        zone = self.zone_input(torch.cat([g.zone, g.zone_mask.float()], -1))
        zone_valid = g.zone_mask.all(-1)
        zone = zone * zone_valid[:, None]
        relative = g.zone[None, :, :2] - g.x[-1, :, None, :2]
        boundary = relative.norm(dim=-1) - g.zone[None, :, 2]
        spatial_valid = valid[-1, :, None] & zone_valid[None, :]
        geometry = torch.cat([relative, boundary[:, :, None]], -1)
        geometry = geometry * spatial_valid[:, :, None]
        relative_context = self.zone_relative(
            torch.cat([geometry.reshape(P, 6), spatial_valid.float()], -1)
        )
        player = (player + relative_context + game + zone.sum(0)) * valid[-1, :, None]
        player = player + self.player_context(player, g.x[-1, :, :2], valid[-1])

        team_rows = []
        for q in range(g.team_count):
            indices = g.members[g.members[:, 1] == q, 0]
            selected = indices[valid[-1, indices]]
            if len(selected):
                pooled = self.team_member(player[selected]).mean(0)
            else:
                pooled = player.new_zeros(self.d)
            pair = player.new_zeros(self.d)
            n = len(selected)
            if n > 1:
                points = g.x[-1, selected, :2]
                delta = (points[:, None] - points[None])[
                    ~torch.eye(n, dtype=torch.bool, device=player.device)
                ]
                pair = self.team_pair(
                    torch.cat([delta, delta.square(), delta.norm(dim=-1, keepdim=True)], -1)
                ).mean(0)
            count = player.new_tensor([len(indices) / 10, n / 10, float(n > 1)])
            team_rows.append(self.team_norm(pooled + pair + self.team_count(count)))
        team = torch.stack(team_rows) if team_rows else player.new_zeros((0, self.d))
        if len(g.members):
            p, q = g.members.unbind(1)
            player = player + mean_to(self.team_to_player(team[q]), p, P)
        player = self.player_norm(player) * valid[-1, :, None]
        return {
            "player": player,
            "team": team,
            "cell": cells,
            "zone": zone,
            "game": game,
            "player_valid": valid[-1],
        }

    def forward_batch(self, inputs):
        return [self(g) for g in inputs]
