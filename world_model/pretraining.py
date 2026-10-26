"""G2 P1/P2/P3: bounded CPU feasibility heads, not a full game simulator.

Ref: Hu et al. 2020, HGT, arXiv:2003.01332 (typed representation);
Hou et al. 2022, GraphMAE, arXiv:2205.10803, sec 3.2 (masked reconstruction);
Forecast-MAE, arXiv:2308.09882, sec 3 (context pretraining, no future input here).
Joint component semantics follow the V3 project contract, not independent actors.
"""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mse(prediction, target, mask):
    mask = torch.broadcast_to(mask, prediction.shape)
    if not mask.any():
        return None
    return (prediction[mask]-target[mask]).square().mean()


def joint_gmm_nll(mean, log_std, logits, target, mask):
    """M,K,H,2; M,H,2; M,H -> joint marginal NLL per observed coordinate.

    Missing labels are replaced BEFORE arithmetic; NaN*0 is not a mask.
    Sum observed dimensions inside each component, then log-sum-exp components.
    """
    if not mask.any():
        return None
    observed = mask[:, None, :, None].expand_as(mean)
    y = torch.where(mask[:, :, None], target, torch.zeros_like(target))
    terms = .5*((y[:, None]-mean)*torch.exp(-log_std)).square()+log_std+.5*math.log(2*math.pi)
    nll_component = torch.where(observed, terms, torch.zeros_like(terms)).sum((0, 2, 3))
    return -torch.logsumexp(logits.log_softmax(0)-nll_component, 0)/(mask.sum()*2)


def pair_targets(xy, mask):
    pairs = torch.triu_indices(len(xy), len(xy), 1, device=xy.device)
    clean = torch.where(mask[:, :, None], xy, torch.zeros_like(xy))
    distance = (clean[pairs[0]]-clean[pairs[1]]).norm(dim=-1)
    return distance, mask[pairs[0]] & mask[pairs[1]], pairs


def corrupt_health(g, seed):
    """P1 masks the entire health history of selected focal actors only."""
    generator = torch.Generator().manual_seed(seed)
    candidates = g.target_player_indices
    candidates = candidates[g.mask[-1, candidates, 5]]
    selected = torch.zeros(len(g.actor_ids), dtype=torch.bool)
    if len(candidates):
        choose = torch.rand(len(candidates), generator=generator) < .3
        if not choose.any():
            choose[torch.randint(len(candidates), (1,), generator=generator)] = True
        selected[candidates[choose]] = True
    x, mask = g.x.clone(), g.mask.clone()
    target = x[-1, :, 5].clone()
    x[:, selected, 5] = 0
    mask[:, selected, 5] = False
    return replace(g, x=x, mask=mask), target, selected


def supervised_targets(sample):
    g = sample.inputs
    rows = sample.targets["selected_rows"]
    ids = g.target_player_indices
    raw = torch.as_tensor(sample.targets["xy"][rows], dtype=torch.float32)
    mask = torch.as_tensor(sample.targets["mask"][rows], dtype=torch.bool)
    xy = (raw-g.origin_m)/100
    delta = xy-g.x[-1, ids, None, :2]
    xy = torch.where(mask[:, :, None], xy, torch.zeros_like(xy))
    delta = torch.where(mask[:, :, None], delta, torch.zeros_like(delta))
    mode = torch.as_tensor(sample.targets["mode"][rows], dtype=torch.long)
    return {"xy": xy, "delta": delta, "mask": mask, "mode": mode,
            "mode_mask": mask & (mode > 0)}


class PretrainHeads(nn.Module):
    def __init__(self, d=128, components=4, horizon=6):
        super().__init__()
        self.components, self.horizon = components, horizon
        self.health = nn.Linear(d, 1)
        self.motion = nn.Linear(d, components*horizon*4)
        self.mixture = nn.Linear(d*2, components)
        self.mode = nn.Linear(d, horizon*10)
        self.pair = nn.Sequential(nn.Linear(d*3+1, d), nn.GELU(), nn.Linear(d, horizon))

    def forward(self, latent, g):
        ids = g.target_player_indices
        player = latent["player"][ids]
        own = g.members[g.members[:, 0] == g.ego_local_index, 1]
        team = latent["team"][own[0]] if len(own) else player.mean(0)
        motion = self.motion(player).reshape(len(ids), self.components, self.horizon, 4)
        mean = motion[..., :2]
        pairs = torch.triu_indices(len(ids), len(ids), 1, device=player.device)
        i, j = pairs
        distance = (g.x[-1, ids[i], :2]-g.x[-1, ids[j], :2]).norm(dim=-1, keepdim=True)
        pair_input = torch.cat([player[i]+player[j], (player[i]-player[j]).abs(),
                                team.expand(len(i), -1), distance], -1)
        return {"mean": mean, "log_std": motion[..., 2:].clamp(-4, 4),
                "logits": self.mixture(torch.cat([team, latent["game"]])),
                "mode": self.mode(player).reshape(len(ids), self.horizon, 10),
                "pair": F.softplus(self.pair(pair_input))}


def pretrain_losses(encoder, heads, sample, seed):
    g = sample.inputs
    y = supervised_targets(sample)
    out = heads(encoder(g), g)
    masked, hp, selected = corrupt_health(g, seed)
    hp_pred = heads.health(encoder(masked)["player"]).squeeze(-1)
    pair_y, pair_mask, _ = pair_targets(y["xy"], y["mask"])
    losses = {
        "health": masked_mse(hp_pred, hp, selected),
        "motion": joint_gmm_nll(out["mean"], out["log_std"], out["logits"], y["delta"], y["mask"]),
        "mode": F.cross_entropy(out["mode"][y["mode_mask"]], y["mode"][y["mode_mask"]])
                if y["mode_mask"].any() else None,
        "pair": masked_mse(out["pair"], pair_y, pair_mask),
    }
    weights = {"health": 1., "motion": 1., "mode": .1, "pair": .1}
    active = [weights[k]*v for k, v in losses.items() if v is not None]
    total = sum(active) if active else None
    support = {"health": int(selected.sum()), "motion": int(y["mask"].sum()),
               "mode": int(y["mode_mask"].sum()), "pair": int(pair_mask.sum())}
    return total, losses, support


class PathProbe(nn.Module):
    def __init__(self, d=128, horizon=6):
        super().__init__()
        self.horizon = horizon
        self.linear = nn.Linear(d, horizon*2)

    def forward(self, latent, g):
        ids = g.target_player_indices
        return self.linear(latent["player"][ids]).reshape(len(ids), self.horizon, 2)
