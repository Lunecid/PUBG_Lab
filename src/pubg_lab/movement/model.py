"""Joint, member-equivariant graph-conditioned residual diffusion (v prediction).

Coordinates are ego-relative km. Residuals around a causal constant-velocity
forecast are divided by 0.2 km. No future labels enter encode() or sample().
"""

import math
import torch
from torch import nn
from .dataset import NODE_TYPES, RELATIONS, WIDTH, EDGE_WIDTH


def constant_velocity(c):
    offsets = torch.arange(10, 61, 10, device=c["initial_xy"].device).float()
    age = c["position_age_s"]
    return c["initial_xy"][:, :, None] + c["velocity"][:, :, None] * (
        offsets[None, None, :, None] + age[:, :, None, None]
    )


class JointGraphDiffusion(nn.Module):
    def __init__(self, categories, hidden=64, diffusion_steps=32):
        super().__init__()
        self.steps, self.hidden, self.scale = diffusion_steps, hidden, 0.2
        self.projections = nn.ModuleList([nn.Linear(WIDTH * 2, hidden) for _ in NODE_TYPES])
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden)
        self.category = nn.Embedding(categories, hidden, padding_idx=0)
        self.relation = nn.Embedding(len(RELATIONS), hidden)
        self.edge = nn.Linear(EDGE_WIDTH * 2, hidden)
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.update = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.LayerNorm(hidden))
                for _ in range(2)
            ]
        )
        self.pool_types = [
            NODE_TYPES.index(s) for s in ("player", "zone", "game", "cell", "region")
        ]
        self.pool_project = nn.Linear(hidden * len(self.pool_types), hidden)
        self.temporal = nn.GRU(hidden * 3, hidden, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.SiLU())
        self.noisy_project = nn.Sequential(
            nn.Linear(12 + hidden + 16, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)
        )
        self.joint_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, 12)
        )
        grid = torch.linspace(0, 1, diffusion_steps + 1, dtype=torch.float64)
        ab = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - ab[1:] / ab[:-1]).clamp(max=0.999).float()
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))
        self.register_buffer("betas", beta)

    def encode(self, c):
        features = torch.cat([c["x"] * c["feature_mask"], c["feature_mask"].float()], -1)
        h = torch.zeros(len(features), self.hidden, device=features.device)
        for typ, layer in enumerate(self.projections):
            take = c["node_type"] == typ
            if take.any():
                h[take] = layer(features[take])
        h = h + self.type_embedding(c["node_type"]) + self.category(c["category"])
        src, dst = c["edge_index"]
        edge = self.edge(
            torch.cat([c["edge_attr"] * c["edge_mask"], c["edge_mask"].float()], -1)
        ) + self.relation(c["edge_type"])
        degree = torch.bincount(dst, minlength=len(h)).clamp_min(1).float()[:, None]
        for mlp, update in zip(self.msg, self.update):
            message = torch.nn.functional.silu(mlp(h[src]) + edge)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, message)
            h = h + update(torch.cat([h, agg / degree], -1))
        B = c["batch_size"]
        pools = []
        # Pool each type separately: 441 map regions cannot dilute players/zones.
        for typ in self.pool_types:
            take = c["node_type"] == typ
            pool = torch.zeros(B * 4, self.hidden, device=h.device)
            pool.index_add_(0, c["node_graph"][take], h[take])
            count = torch.bincount(c["node_graph"][take], minlength=B * 4).clamp_min(1)
            pools.append(pool / count[:, None])
        global_context = self.pool_project(torch.cat(pools, -1)).reshape(B, 4, self.hidden)
        ids = c["target_nodes"]
        members = h[ids.clamp_min(0)] * (ids >= 0)[..., None]
        team = h[c["team_nodes"]][:, :, None].expand(-1, -1, 4, -1)
        glob = global_context[:, :, None].expand(-1, -1, 4, -1)
        sequence = torch.cat([members, team, glob], -1).permute(0, 2, 1, 3).reshape(B * 4, 4, -1)
        history = self.temporal(sequence)[1][-1].reshape(B, 4, self.hidden)
        state = torch.cat(
            [c["initial_xy"], c["velocity"] * 30, c["position_age_s"][:, :, None] / 15], -1
        )
        return self.condition(torch.cat([history, state], -1)) * c["current_presence"][:, :, None]

    def predict_v(self, noisy, step, condition, presence):
        freq = torch.exp(torch.arange(8, device=noisy.device) * (-math.log(10000) / 7))
        phase = step.float()[:, None] * freq[None]
        emb = torch.cat([phase.sin(), phase.cos()], -1)[:, None].expand(-1, 4, -1)
        h = self.noisy_project(torch.cat([noisy.flatten(2), condition, emb], -1))
        joint = self.joint_attention(h, h, h, key_padding_mask=~presence, need_weights=False)[0]
        return (
            self.denoiser(torch.cat([h, joint], -1)).reshape(-1, 4, 6, 2)
            * presence[:, :, None, None]
        )

    def loss(self, conditions, targets):
        c = self.encode(conditions)
        t = torch.randint(self.steps, (len(c),), device=c.device)
        a = self.alpha_bar[t][:, None, None, None]
        baseline = constant_velocity(conditions)
        mask = targets["trajectory_valid"] & conditions["current_presence"][:, :, None]
        clean = torch.where(mask[..., None], (targets["trajectory"] - baseline) / self.scale, 0.0)
        noise = torch.randn_like(clean) * conditions["current_presence"][:, :, None, None]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        v = a.sqrt() * noise - (1 - a).sqrt() * clean
        pred = self.predict_v(noisy, t, c, conditions["current_presence"])
        denoise = ((pred - v).square() * mask[..., None]).sum() / (mask.sum() * 2).clamp_min(1)
        x0 = baseline + self.scale * (a.sqrt() * noisy - (1 - a).sqrt() * pred)
        weights = conditions["current_presence"].float()
        center = (x0 * weights[:, :, None, None]).sum(1) / weights.sum(1)[:, None, None].clamp_min(
            1
        )
        tm = targets["team_valid"]
        team = (((center - targets["team_center"]) / self.scale).square() * tm[..., None]).sum() / (
            tm.sum() * 2
        ).clamp_min(1)
        return denoise + 0.1 * team, {
            "denoising": float(denoise.detach()),
            "team_center": float(team.detach()),
            "valid_future_points": int(mask.sum()),
        }

    @torch.no_grad()
    def sample(self, conditions, scenarios=20):
        c = self.encode(conditions).repeat_interleave(scenarios, 0)
        presence = conditions["current_presence"].repeat_interleave(scenarios, 0)
        active = presence[:, :, None, None]
        y = torch.randn(len(c), 4, 6, 2, device=c.device) * active
        for t in reversed(range(self.steps)):
            a = self.alpha_bar[t]
            prev = self.alpha_bar[t - 1] if t else torch.tensor(1.0, device=c.device)
            v = self.predict_v(y, torch.full((len(c),), t, device=c.device), c, presence)
            clean = a.sqrt() * y - (1 - a).sqrt() * v
            eps = (1 - a).sqrt() * y + a.sqrt() * v
            y = (prev.sqrt() * clean + (1 - prev).sqrt() * eps) * active
        y = y.reshape(conditions["batch_size"], scenarios, 4, 6, 2)
        return constant_velocity(conditions)[:, None] + self.scale * y
