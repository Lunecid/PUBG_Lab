"""Team-center/relative-formation prediction and frozen-mean residual diffusion.
No future membership, future modes or future circles are conditioning variables.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model_v3 import JointMotion as Previous


def center_relative(x, presence):
    w = presence.to(x.dtype)[:, :, None, None]
    center = (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)
    return center, (x - center) * w


def pair_loss(pred, truth, valid, scale=.2):
    """Actual future pair distances, not a constant-tight-formation constraint."""
    pairs = torch.triu(torch.ones(4, 4, device=pred.device, dtype=torch.bool), 1)
    mask = valid[:, :, None] & valid[:, None, :] & pairs[None, :, :, None]
    pd = torch.linalg.vector_norm(pred[:, :, None] - pred[:, None, :], dim=-1)
    td = torch.linalg.vector_norm(truth[:, :, None] - truth[:, None, :], dim=-1)
    error = F.smooth_l1_loss(pd / scale, td / scale, reduction='none')
    return (error * mask).sum() / mask.sum().clamp_min(1)


class FormationRegressor(Previous):
    def __init__(self, categories):
        super().__init__(categories, 'regression')
        self.center_head = nn.Sequential(nn.Linear(110, 48), nn.SiLU(), nn.Linear(48, 6))
        nn.init.zeros_(self.center_head[-1].weight)
        nn.init.zeros_(self.center_head[-1].bias)

    def mean(self, h, c):
        _, relative = center_relative(self.mean_head(h).reshape(-1, 4, 3, 2), c['current_presence'])
        pooled = (h * c['current_presence'][:, :, None]).sum(1) / c['current_presence'].sum(1)[:, None].clamp_min(1)
        center = self.center_head(pooled).reshape(-1, 1, 3, 2)
        return self.curve((center + relative) * self.scale, c)

    def loss(self, c, t):
        h = self.encode(c)
        mean = self.mean(h, c)
        valid = t['trajectory_valid'] & c['current_presence'][:, :, None]
        point = (F.smooth_l1_loss(mean / self.scale, t['trajectory'] / self.scale, reduction='none') * valid[..., None]).sum() / (valid.sum() * 2).clamp_min(1)
        center = center_relative(mean, c['current_presence'])[0][:, 0]
        tm = t['team_valid']
        team = (F.smooth_l1_loss(center / self.scale, t['team_center'] / self.scale, reduction='none') * tm[..., None]).sum() / (tm.sum() * 2).clamp_min(1)
        logits = self.mode_head(h).reshape(-1, 4, 6, 10)
        mm = t['mode_valid'] & valid
        ce = F.cross_entropy(logits.flatten(0, 2), t['mode'].flatten(), reduction='none').reshape_as(valid)
        mode = (ce * mm).sum() / mm.sum().clamp_min(1)
        radial = point * 0
        for z in range(2):
            geo = c['zone_geometry'][:, :, z * 7:(z + 1) * 7]
            zc = c['initial_xy'] + geo[:, :, :2]
            pr = torch.linalg.vector_norm(mean - zc[:, :, None], dim=-1) - geo[:, :, 2, None]
            tr = torch.linalg.vector_norm(t['trajectory'] - zc[:, :, None], dim=-1) - geo[:, :, 2, None]
            zm = valid & geo[:, :, 6, None].bool()
            radial += (F.smooth_l1_loss(pr / self.scale, tr / self.scale, reduction='none') * zm).sum() / zm.sum().clamp_min(1)
        formation = pair_loss(mean, t['trajectory'], valid)
        loss = point + .1 * team + .05 * mode + .05 * radial + .2 * formation
        return loss, {k: float(v.detach()) for k, v in dict(point=point, team=team, mode=mode, radial=radial, formation=formation).items()}


class FormationDiffusion(nn.Module):
    def __init__(self, regressor, structured=True):
        super().__init__()
        self.regressor = regressor
        self.regressor.requires_grad_(False)
        self.structured = structured
        self.steps = 32
        self.register_buffer('center_scale', torch.tensor(.2))
        self.register_buffer('relative_scale', torch.tensor(.2))
        self.register_buffer('independent_scale', torch.tensor(.2))
        def mlp(width):
            return nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 6))
        if structured:
            self.center_denoiser = mlp(116 + 6 + 6 + 16)
            self.relative_denoiser = mlp(116 * 2 + 6 + 6 + 16)
        else:
            self.independent_denoiser = mlp(116 + 6 + 6 + 16)

    @torch.no_grad()
    def context(self, c):
        h = self.regressor.encode(c)
        mean = self.regressor.mean(h, c)
        displacement = mean[:, :, [1, 3, 5]] - c['initial_xy'][:, :, None]
        return dict(h=torch.cat([h, displacement.flatten(2)], -1), mean=mean, presence=c['current_presence'])

    def denoise(self, y, step, ctx):
        h, p = ctx['h'], ctx['presence']
        f = torch.exp(torch.arange(8, device=y.device) * (-math.log(10000) / 7))
        e = step[:, None] * f[None]
        e = torch.cat([e.sin(), e.cos()], -1)
        pooled_h = (h * p[:, :, None]).sum(1) / p.sum(1)[:, None].clamp_min(1)
        if self.structured:
            center, relative = y[:, :1], center_relative(y[:, 1:], p)[1]
            spread = (relative.square() * p[:, :, None, None]).sum(1) / p.sum(1)[:, None, None].clamp_min(1)
            pc = self.center_denoiser(torch.cat([pooled_h, center.flatten(1), spread.flatten(1), e], -1)).reshape(-1, 1, 3, 2)
            pr = self.relative_denoiser(torch.cat([h, pooled_h[:, None].expand(-1, 4, -1), center.flatten(2).expand(-1, 4, -1), relative.flatten(2), e[:, None].expand(-1, 4, -1)], -1)).reshape(-1, 4, 3, 2)
            return torch.cat([pc, center_relative(pr, p)[1]], 1)
        pooled = center_relative(y, p)[0].flatten(2).expand(-1, 4, -1)
        return self.independent_denoiser(torch.cat([h, y.flatten(2), pooled, e[:, None].expand(-1, 4, -1)], -1)).reshape(-1, 4, 3, 2) * p[:, :, None, None]

    def encode_residual(self, residual, presence):
        if not self.structured:
            return residual / self.independent_scale
        center, relative = center_relative(residual, presence)
        return torch.cat([center / self.center_scale, relative / self.relative_scale], 1)

    def decode_residual(self, latent, presence):
        if not self.structured:
            return latent * self.independent_scale
        return latent[:, :1] * self.center_scale + center_relative(latent[:, 1:], presence)[1] * self.relative_scale

    def noise(self, ctx):
        p = ctx['presence']
        if self.structured:
            center = torch.randn(len(p), 1, 3, 2, device=p.device)
            relative = center_relative(torch.randn(len(p), 4, 3, 2, device=p.device), p)[1]
            return torch.cat([center, relative], 1)
        return torch.randn(len(p), 4, 3, 2, device=p.device) * p[:, :, None, None]

    def loss_cached(self, ctx, target):
        p = ctx['presence']
        km = target['trajectory_valid'][:, :, [1, 3, 5]] & p[:, :, None]
        # Incomplete membership labels never redefine the team center.
        complete = (km | ~p[:, :, None]).all(1)
        valid = complete[:, None] & p[:, :, None]
        residual = torch.where(valid[..., None], target['trajectory'][:, :, [1, 3, 5]] - ctx['mean'][:, :, [1, 3, 5]], 0.)
        clean = self.encode_residual(residual, p)
        step = torch.randint(self.steps, (len(p),), device=p.device)
        ab = self.regressor.encoder.alpha_bar[step][:, None, None, None]
        noise = self.noise(ctx)
        noisy = ab.sqrt() * clean + (1 - ab).sqrt() * noise
        output = self.denoise(noisy, step, ctx)
        latent_mask = torch.cat([complete[:, None], valid], 1) if self.structured else valid
        reconstruction = ((output - clean).square() * latent_mask[..., None]).sum() / (2 * latent_mask.sum()).clamp_min(1)
        formation = reconstruction * 0
        if self.structured:
            path = ctx['mean'][:, :, [1, 3, 5]] + self.decode_residual(output, p)
            formation = pair_loss(path, target['trajectory'][:, :, [1, 3, 5]], valid)
        # x0 prediction eliminates untrained v-prediction's residual-noise floor.
        loss = reconstruction + .2 * formation
        return loss, dict(reconstruction=float(reconstruction.detach()), formation=float(formation.detach()))

    @torch.no_grad()
    def sample(self, c, scenarios=20):
        ctx0 = self.context(c)
        ctx = {k: v.repeat_interleave(scenarios, 0) for k, v in ctx0.items()}
        y = self.noise(ctx)
        for step in reversed(range(self.steps)):
            a = self.regressor.encoder.alpha_bar[step]
            prev = self.regressor.encoder.alpha_bar[step - 1] if step else torch.tensor(1., device=y.device)
            clean = self.denoise(y, torch.full((len(y),), step, device=y.device), ctx)
            eps = (y - a.sqrt() * clean) / (1 - a).sqrt().clamp_min(1e-6)
            y = prev.sqrt() * clean + (1 - prev).sqrt() * eps
        residual = self.decode_residual(y, ctx['presence'])
        residual = torch.einsum('tk,bpkd->bptd', self.regressor.basis, residual).reshape(len(ctx0['h']), scenarios, 4, 6, 2)
        return ctx0['mean'][:, None] + residual
