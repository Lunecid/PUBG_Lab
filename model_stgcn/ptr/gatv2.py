"""Snapshot GATv2 baseline using the pinned PyG 2.7.0 operator directly.

Relation layout is [query/receiver i, key/sender j]; PyG edge_index is
[sender j, receiver i]. Four edge features preserve proximity, outgoing
combat, incoming combat and co-pressure in the receiver's frame. No distance
threshold or graph sparsification is introduced. Each match-phase is a
separate graph. This is a node-prediction adaptation, not the paper's dataset
or full training recipe.
"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def graph_edges(batch):
    key, query = batch['node_mask'], batch['query_mask']
    b, p, n = key.shape
    raw = batch['rel'].permute(0, 1, 3, 4, 2)
    valid = query[..., :, None] & key[..., None, :]
    rel = torch.where(valid[..., None], raw, torch.zeros_like(raw))
    adjacency = (rel > 0).any(-1) & valid
    adjacency = adjacency & ~torch.eye(n, dtype=torch.bool, device=key.device)
    bi, pi, receiver, sender = adjacency.nonzero(as_tuple=True)
    offset = (bi * p + pi) * n
    index = torch.stack((offset + sender, offset + receiver))
    attributes = rel[bi, pi, receiver, sender]
    # Explicit zero-feature self loops avoid PyG's default mean-edge fill.
    own = query.reshape(-1).nonzero(as_tuple=True)[0]
    index = torch.cat((index, torch.stack((own, own))), dim=1)
    attributes = torch.cat((attributes, attributes.new_zeros(len(own), rel.shape[-1])))
    return index, attributes


class GraphBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv = GATv2Conv(d, d // cfg.n_heads, heads=cfg.n_heads,
            concat=True, edge_dim=4, dropout=cfg.dropout, add_self_loops=False,
            share_weights=False, residual=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(4*d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, index, attributes, mask):
        x = x + self.drop(self.conv(self.ln1(x), index, attributes))
        x = torch.where(mask, x, torch.zeros_like(x))
        return torch.where(mask, x + self.drop(self.ffn(self.ln2(x))), torch.zeros_like(x))


class SnapshotGATv2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.token = nn.Sequential(nn.Linear(cfg.in_features, d), nn.GELU(), nn.Linear(d, d))
        self.phase = nn.Embedding(cfg.n_phases, d)
        nn.init.normal_(self.phase.weight, std=.02)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(GraphBlock(cfg) for _ in range(cfg.n_layers))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, batch):
        q = batch['query_mask']; b, p, n = q.shape
        x = torch.where(q[..., None], batch['x'], torch.zeros_like(batch['x']))
        x = self.token(x) + self.phase(torch.arange(p, device=x.device))[None, :, None]
        x = torch.where(q[..., None], self.drop(x), torch.zeros_like(x)).reshape(b*p*n, -1)
        index, attributes = graph_edges(batch)
        mask = q.reshape(-1, 1)
        for block in self.blocks:
            x = block(x, index, attributes, mask)
        z = self.head(x).reshape(b, p, n)
        z = torch.where(q, z, torch.zeros_like(z))
        return dict(hazard_logit=z, rank_score=-z)
