"""Dataset for snapshot-resolution phase survival training.

Each sample keeps the 10s snapshot tensors as the model input and uses the
existing phase labels only as supervision at phase cut points.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch.utils.data import Dataset

from stdb.queries import fetch_phase_match_tensors
from stdb.snapshot_queries import fetch_snapshot_match_tensors

logger = logging.getLogger("model_stgcn.snapshot_phase_dataset")


def fetch_snapshot_phase(match_id: str, *, include_edges: bool = True) -> dict | None:
    phase = fetch_phase_match_tensors(match_id)
    if phase is None:
        return None
    snap = fetch_snapshot_match_tensors(match_id, include_edges=include_edges)
    if snap is None:
        return None
    item = {
        "X_s": torch.from_numpy(snap.X_s.copy()),
        "phase_of_snap": torch.from_numpy(snap.phase_of_snap.copy()),
        "node_mask_s": torch.from_numpy(snap.node_mask.copy()),
        "snapshot_ts": torch.from_numpy(snap.snapshot_ts.copy()),
        "y_hazard": torch.from_numpy(phase.y_hazard.copy()),
        "target_mask": torch.from_numpy(phase.target_mask.copy()),
        "phase_mask": torch.from_numpy(phase.phase_mask.copy()),
        "team_mask": torch.from_numpy(phase.team_mask.copy()),
        "match_id": match_id,
        "team_ids": phase.team_ids,
        "phase_indices": phase.phase_indices,
    }
    if snap.A_s is not None:
        item["A_s"] = torch.from_numpy(snap.A_s.copy()).to_sparse()
    return item


class LazySnapshotPhaseDataset(Dataset):
    def __init__(self, match_ids: Iterable[str], *, include_edges: bool = True) -> None:
        self.match_ids = list(match_ids)
        self.include_edges = include_edges

    def __len__(self) -> int:
        return len(self.match_ids)

    def __getitem__(self, idx: int) -> dict:
        mid = self.match_ids[idx]
        item = fetch_snapshot_phase(mid, include_edges=self.include_edges)
        if item is None:
            raise RuntimeError(f"match {mid} is not snapshot-phase-ready")
        return item


class CachedSnapshotPhaseDataset(Dataset):
    def __init__(self, match_ids: Iterable[str], *, include_edges: bool = True) -> None:
        self.items: list[dict] = []
        for mid in match_ids:
            item = fetch_snapshot_phase(mid, include_edges=include_edges)
            if item is None:
                continue
            self.items.append(item)
        logger.info("CachedSnapshotPhaseDataset: %d matches loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def snapshot_phase_collate(batch: list[dict]) -> dict:
    out: dict = {}
    sample = batch[0]
    for key, value in sample.items():
        if key == "A_s":
            out[key] = torch.stack([b[key].to_dense() for b in batch])
        elif torch.is_tensor(value):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out
