"""
model_stgcn/dataset.py
======================
PyTorch Dataset wrapping `stdb.queries.fetch_match_tensors`.

Two flavours:
  • LazyMatchDataset    – fetches from PG on every __getitem__
  • CachedMatchDataset  – fetches once into memory.  Use only for small
                          training sets (a few thousand matches OK).
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch.utils.data import Dataset

from model_stgcn.splits import split_ids
from stdb.queries import (
    MatchTensors,
    PhaseMatchTensors,
    fetch_match_tensors,
    fetch_phase_match_tensors,
)

logger = logging.getLogger("model_stgcn.dataset")


def _to_torch(m: MatchTensors) -> dict[str, torch.Tensor]:
    return {
        "X":     torch.from_numpy(m.X.copy()),
        "A_t":   torch.from_numpy(m.A_t.copy()),
        "y_reg": torch.from_numpy(m.y_reg.copy()),
        "y_cls": torch.from_numpy(m.y_cls.copy()),
        "mask":  torch.from_numpy(m.mask.copy()),
        "match_id": m.match_id,
    }


def _phase_to_torch(m: PhaseMatchTensors) -> dict[str, torch.Tensor | str | list]:
    return {
        "X": torch.from_numpy(m.X.copy()),
        "A": torch.from_numpy(m.A.copy()),
        "y_hazard": torch.from_numpy(m.y_hazard.copy()),
        "target_mask": torch.from_numpy(m.target_mask.copy()),
        "y_rank": torch.from_numpy(m.y_rank.copy()),
        "y_topk": torch.from_numpy(m.y_topk.copy()),
        "team_mask": torch.from_numpy(m.team_mask.copy()),
        "node_mask": torch.from_numpy(m.node_mask.copy()),
        "phase_mask": torch.from_numpy(m.phase_mask.copy()),
        "match_id": m.match_id,
        "team_ids": m.team_ids,
        "phase_indices": m.phase_indices,
        "feature_names": m.feature_names,
        "edge_types": m.edge_types,
    }


class LazyMatchDataset(Dataset):
    def __init__(self, match_ids: Iterable[str]) -> None:
        self.match_ids = list(match_ids)

    def __len__(self) -> int:
        return len(self.match_ids)

    def __getitem__(self, idx: int) -> dict:
        mid = self.match_ids[idx]
        mt = fetch_match_tensors(mid)
        if mt is None:
            raise RuntimeError(f"match {mid} not in stdb")
        return _to_torch(mt)


class CachedMatchDataset(Dataset):
    def __init__(self, match_ids: Iterable[str]) -> None:
        self.items: list[dict] = []
        for mid in match_ids:
            mt = fetch_match_tensors(mid)
            if mt is None:
                logger.warning("skip %s — not in stdb", mid)
                continue
            self.items.append(_to_torch(mt))
        logger.info("CachedMatchDataset: %d matches loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class LazyPhaseMatchDataset(Dataset):
    """Lazy dataset for phase-level DynamicSTGNN tensors."""

    def __init__(self, match_ids: Iterable[str]) -> None:
        self.match_ids = list(match_ids)

    def __len__(self) -> int:
        return len(self.match_ids)

    def __getitem__(self, idx: int) -> dict:
        mid = self.match_ids[idx]
        mt = fetch_phase_match_tensors(mid)
        if mt is None:
            raise RuntimeError(f"match {mid} not phase-ready")
        return _phase_to_torch(mt)


class CachedPhaseMatchDataset(Dataset):
    """In-memory dataset for phase-level tensors. Use for smoke tests/small runs."""

    def __init__(self, match_ids: Iterable[str]) -> None:
        self.items: list[dict] = []
        for mid in match_ids:
            mt = fetch_phase_match_tensors(mid)
            if mt is None:
                logger.warning("skip %s — not phase-ready", mid)
                continue
            self.items.append(_phase_to_torch(mt))
        logger.info("CachedPhaseMatchDataset: %d matches loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    out = {
        "X":     torch.stack([b["X"]     for b in batch]),
        "A_t":   torch.stack([b["A_t"]   for b in batch]),
        "y_reg": torch.stack([b["y_reg"] for b in batch]),
        "y_cls": torch.stack([b["y_cls"] for b in batch]),
        "mask":  torch.stack([b["mask"]  for b in batch]),
        "match_id": [b["match_id"] for b in batch],
    }
    return out


def phase_collate(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "A": torch.stack([b["A"] for b in batch]),
        "y_hazard": torch.stack([b["y_hazard"] for b in batch]),
        "target_mask": torch.stack([b["target_mask"] for b in batch]),
        "y_rank": torch.stack([b["y_rank"] for b in batch]),
        "y_topk": torch.stack([b["y_topk"] for b in batch]),
        "team_mask": torch.stack([b["team_mask"] for b in batch]),
        "node_mask": torch.stack([b["node_mask"] for b in batch]),
        "phase_mask": torch.stack([b["phase_mask"] for b in batch]),
        "match_id": [b["match_id"] for b in batch],
        "team_ids": [b["team_ids"] for b in batch],
        "phase_indices": [b["phase_indices"] for b in batch],
        "feature_names": batch[0]["feature_names"],
        "edge_types": batch[0]["edge_types"],
    }


def train_val_test_split(
    ids: list[str],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    return split_ids(ids, val_frac=val_frac, test_frac=test_frac, seed=seed)
