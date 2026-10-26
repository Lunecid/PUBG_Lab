"""
Dataset + collate for the MR-STGNN: each item carries BOTH the phase tier
tensors (150-d V2 node features + phase graph) and the snapshot tier tensors
(team_snapshot_features + snapshot graph + death index), fetched for the same
match and indexed on the SAME team order (_phase_team_order) so a team keeps
one node index across resolutions.

Lazy fetch is ~0.3s/match (snapshot graph dominates); use CachedMRDataset for
small-scale runs and validation. A disk/.pt cache for full-DB training is a
later optimization (see plan).
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch.utils.data import Dataset

from pubg_lab.survival.data.dataset import _phase_to_torch
from pubg_lab.data.stdb.queries.queries import fetch_phase_match_tensors
from pubg_lab.data.stdb.queries.snapshot_queries import fetch_snapshot_match_tensors

logger = logging.getLogger("pubg_lab.survival.data.mr_dataset")

_SNAP_TENSOR_KEYS = ("X_s", "A_s", "phase_of_snap", "node_mask_s", "death_snap_idx")


def fetch_mr(match_id: str, *, include_v2: bool = True) -> dict | None:
    pt = fetch_phase_match_tensors(match_id, include_v2_features=include_v2)
    if pt is None:
        return None
    st = fetch_snapshot_match_tensors(match_id)
    if st is None:
        return None
    d = _phase_to_torch(pt)
    d["X_s"] = torch.from_numpy(st.X_s.copy())
    # A_s is ~99% zero (sparse graph). Store as COO so a cached match costs
    # ~1-2MB instead of 26MB dense; mr_collate re-densifies per batch only.
    # Phase A ("A") stays dense — it is small (8x3x128x128).
    d["A_s"] = torch.from_numpy(st.A_s.copy()).to_sparse()
    d["phase_of_snap"] = torch.from_numpy(st.phase_of_snap.copy())
    d["node_mask_s"] = torch.from_numpy(st.node_mask.copy())
    d["death_snap_idx"] = torch.from_numpy(st.death_snap_idx.copy())
    return d


class LazyMRDataset(Dataset):
    def __init__(self, match_ids: Iterable[str], *, include_v2: bool = True) -> None:
        self.match_ids = list(match_ids)
        self.include_v2 = include_v2

    def __len__(self) -> int:
        return len(self.match_ids)

    def __getitem__(self, idx: int) -> dict:
        mid = self.match_ids[idx]
        d = fetch_mr(mid, include_v2=self.include_v2)
        if d is None:
            raise RuntimeError(f"match {mid} not MR-ready")
        return d


class CachedMRDataset(Dataset):
    def __init__(self, match_ids: Iterable[str], *, include_v2: bool = True) -> None:
        self.items: list[dict] = []
        for mid in match_ids:
            d = fetch_mr(mid, include_v2=include_v2)
            if d is None:
                continue
            self.items.append(d)
        logger.info("CachedMRDataset: %d matches loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def mr_collate(batch: list[dict]) -> dict:
    out: dict = {}
    sample = batch[0]
    for k, v in sample.items():
        if k == "A_s":
            # re-densify the sparse snapshot adjacency for this batch only
            out[k] = torch.stack([b[k].to_dense() for b in batch])
        elif torch.is_tensor(v):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
