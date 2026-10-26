"""G4 single-owner past-only spatial evidence.

Ref: G3 prior interface / G4 CPU protocol. Typed messages extend the local
HGT-inspired encoder (Hu et al., arXiv:2003.01332), not HGT reproduction.
"""

from dataclasses import dataclass, fields
import numpy as np
import torch
from torch import nn
from .cohorts import CohortDataset
from .dataset import WorldDataset
from .historical_prior import HistoricalPrior, timestamp
from .inputs import EncoderInput, ModelSample, adapt_graph
from .encoder import WorldEncoder, mean_to


@dataclass(frozen=True)
class PastInput(EncoderInput):
    past_edges: torch.Tensor  # E,5 src,dst,mode,count,donor matches
    past_numeric: torch.Tensor  # E,3 age_days/30, known_change, geometry_verified
    past_patch: torch.Tensor  # E; source patch token from actual-fit vocabulary


class TemporalPriorDataset(CohortDataset):
    def __init__(self, root, manifest, prior_path, role="train", view="observer", policy="H1"):
        super().__init__(root, manifest, role, view)
        if policy not in ("H0", "H1"):
            raise ValueError("Unknown past evidence policy")
        self.policy = policy
        self.prior = HistoricalPrior(prior_path, self.fit_ids)

    def support(self, geometry):
        # No legacy aggregate read, including its cell counts and future donors.
        return {}, {}

    def graph_at(self, mid, frame, actor):
        bank = self.bank(mid)
        if frame < 12 or not bank["valid"][frame, actor] or not bank["known"][frame, actor]:
            raise ValueError("Current known/valid ego and 120s history required")
        overlay = dict(bank)
        overlay["anchors"] = np.array([[frame, actor]], dtype=np.int64)
        self.bank_cache[mid] = overlay
        try:
            # Intentionally bypass IndexedWorldDataset and G1 subtraction.
            graph, targets = WorldDataset.sample(self, mid, 0)
        finally:
            self.bank_cache[mid] = bank
        meta = self.metadata[mid]
        rows = self.prior.query(
            map_name=meta["map"],
            target_mid=mid,
            target_start=timestamp(meta["start_time"]),
            target_patch=meta["patch"],
            cells=graph["cell_coordinates"],
            source_patch=meta["patch"] if self.policy == "H0" else None,
        )
        lookup = {tuple(map(int, c)): i for i, c in enumerate(graph["cell_coordinates"])}
        edges, numeric, patches = [], [], []
        for r in rows:
            edges.append(
                [
                    lookup[r["sx"], r["sy"]],
                    lookup[r["dx"], r["dy"]],
                    r["mode"],
                    r["count"],
                    r["donor_matches"],
                ]
            )
            numeric.append([r["age_days"] / 30, r["known_change"], int(r["geometry_verified"])])
            # Current corpus has one source patch. Refuse ambiguous future
            # multi-patch aggregation rather than silently choosing one token.
            if len(r["source_patches"]) != 1:
                raise ValueError("Multi-patch edges require patch-specific aggregation")
            patches.append(r["source_patches"][0])
        graph["past_edges"] = np.array(edges, np.int64).reshape(-1, 5)
        graph["past_numeric"] = np.array(numeric, np.float32).reshape(-1, 3)
        graph["past_patches"] = patches
        used = {(r[0], r[1]) for r in edges}
        graph["inferred_adjacency"] = np.array(
            [r for r in graph["inferred_adjacency"] if tuple(r) not in used], np.int64
        ).reshape(-1, 2)
        assert len(graph["observed_connections"]) == 0
        assert not graph["cell_features"][:, 2:].any()
        assert len({tuple(r[:3]) for r in edges}) == len(edges)
        return graph, targets

    def load(self, mid, frame, actor, vocabulary):
        graph, targets = self.graph_at(mid, frame, actor)
        bank = self.bank(mid)
        focal = np.flatnonzero((bank["team"] == bank["team"][actor]) & bank["known"][frame])
        base = adapt_graph(
            graph,
            ego_id=actor,
            frame_end=frame,
            t0=float(bank["times"][frame]),
            view=self.view,
            vocabulary=vocabulary,
            focal_ids=focal,
        )
        g = PastInput(
            **{f.name: getattr(base, f.name) for f in fields(base)},
            past_edges=torch.as_tensor(graph["past_edges"]),
            past_numeric=torch.as_tensor(graph["past_numeric"]),
            past_patch=torch.tensor(
                [vocabulary.tokens["patch"].get(p, 1) for p in graph["past_patches"]],
                dtype=torch.long,
            ),
        )
        lookup = {int(p): i for i, p in enumerate(targets["members"])}
        targets = dict(targets)
        targets["selected_rows"] = np.array(
            [lookup[int(g.actor_ids[p])] for p in g.target_player_indices], np.int64
        )
        targets["joint_complete_60s"] = bool(
            len(targets["selected_rows"]) and targets["mask"][targets["selected_rows"]].all()
        )
        return ModelSample(
            g,
            targets,
            dict(
                match_id=mid,
                frame=int(frame),
                ego=int(actor),
                team_id=int(bank["team"][actor]),
                view=self.view,
                patch=self.metadata[mid]["patch"],
                spatial_prior_policy=self.policy,
                spatial_donor_policy="actual_fit_end_before_target_start",
                map=self.metadata[mid]["map"],
                past_edges=len(g.past_edges),
                kind_eval_only=bank["kind_eval_only"][
                    g.actor_ids[g.target_player_indices]
                ].tolist(),
            ),
        )


class PastSpatialBlock(nn.Module):
    def __init__(self, d, patch_count):
        super().__init__()
        self.message = nn.Linear(d, d, bias=False)
        self.mode = nn.Embedding(10, d)
        self.patch = nn.Embedding(patch_count, d, padding_idx=0)
        self.support = nn.Linear(5, d, bias=False)
        self.gate = nn.Linear(5, 1)
        # G4 fit corpus has constant-zero change/verified flags. Do not activate
        # random, untrained coefficients when the next patch changes the flag.
        # They remain zero with AdamW while training inputs for them are zero.
        with torch.no_grad():
            self.support.weight[:, 3:].zero_()
            self.gate.weight[:, 3:].zero_()
        self.hypothetical = nn.Linear(d, d, bias=False)
        self.hypothesis_gate = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(d)

    def forward(self, cells, g):
        update = torch.zeros_like(cells)
        if len(g.past_edges):
            src, dst, mode, n, nm = g.past_edges.unbind(1)
            features = torch.cat(
                [torch.log1p(torch.stack([n, nm], -1).float()), g.past_numeric], -1
            )
            message = (
                self.message(cells[src])
                + self.mode(mode)
                + self.patch(g.past_patch)
                + self.support(features)
            )
            update = mean_to(message * self.gate(features).sigmoid(), dst, len(cells))
        if len(g.inferred):
            src, dst = g.inferred.unbind(1)
            update = update + mean_to(
                self.hypothetical(cells[src]) * self.hypothesis_gate.sigmoid(), dst, len(cells)
            )
        return self.norm(cells + update)


class PastWorldEncoder(WorldEncoder):
    def __init__(self, item_count, condition_sizes, d=64, heads=4):
        super().__init__(item_count, condition_sizes, d, heads)
        self.spatial = nn.ModuleList([PastSpatialBlock(d, condition_sizes[-1]) for _ in range(2)])

    def encode_spatial(self, g):
        if not isinstance(g, PastInput):
            raise TypeError("PastInput required")
        if len(g.observed) or torch.any(g.cells[:, 2:6] != 0):
            raise ValueError("Legacy evidence must not duplicate historical channel")
        cells = self.cell_input(g.cells)
        for block in self.spatial:
            cells = block(cells, g)
        return cells
