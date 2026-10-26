"""Factorized V3 tensors.

This module deliberately reuses the existing STDB tensor builders, then splits
the legacy phase node vector into:

    X_node: team-specific attributes
    G:      phase/global context
    A:      dynamic team graph

That makes the final experiment input contract explicit without invalidating
the already reproduced legacy results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pubg_lab.survival.v3.data.spec import (
    FeatureIndexSplit,
    assert_no_forbidden_input_features,
    feature_names_for_preset,
    split_feature_names,
)
from pubg_lab.data.stdb.queries.queries import PhaseMatchTensors, fetch_phase_match_tensors
from pubg_lab.data.stdb.queries.snapshot_queries import (
    SnapshotTensors,
    fetch_snapshot_match_tensors,
)

logger = logging.getLogger("pubg_lab.survival.v3.data.factorized_data")


@dataclass
class FactorizedPhaseTensors:
    match_id: str
    X_node: np.ndarray
    G: np.ndarray
    A: np.ndarray
    y_hazard: np.ndarray
    target_mask: np.ndarray
    y_rank: np.ndarray
    y_topk: np.ndarray
    team_mask: np.ndarray
    node_mask: np.ndarray
    phase_mask: np.ndarray
    team_ids: list[int]
    phase_indices: list[int]
    node_feature_names: list[str]
    global_feature_names: list[str]
    edge_types: list[str]


@dataclass
class FactorizedMRTensors:
    phase: FactorizedPhaseTensors
    X_s: np.ndarray
    A_s: np.ndarray
    phase_of_snap: np.ndarray
    node_mask_s: np.ndarray
    death_snap_idx: np.ndarray
    snapshot_ts: np.ndarray


def _global_from_broadcast(
    X: np.ndarray,
    global_indices: tuple[int, ...],
    team_mask: np.ndarray,
    phase_mask: np.ndarray,
) -> np.ndarray:
    """Extract one global vector per phase from the broadcast legacy tensor."""
    T = X.shape[0]
    G = np.zeros((T, len(global_indices)), dtype=np.float32)
    if not global_indices:
        return G
    real_team_idx = np.where(team_mask)[0]
    if real_team_idx.size == 0:
        return G
    source_i = int(real_team_idx[0])
    G[:, :] = X[:, source_i, list(global_indices)]
    G[~phase_mask.astype(bool)] = 0.0
    return G


def factorize_phase_tensors(mt: PhaseMatchTensors) -> FactorizedPhaseTensors:
    assert_no_forbidden_input_features(mt.feature_names)
    split = split_feature_names(mt.feature_names)
    X_node = mt.X[:, :, list(split.node_indices)].astype(np.float32, copy=True)
    G = _global_from_broadcast(
        mt.X,
        split.global_indices,
        mt.team_mask,
        mt.phase_mask,
    )
    return FactorizedPhaseTensors(
        match_id=mt.match_id,
        X_node=X_node,
        G=G,
        A=mt.A.astype(np.float32, copy=True),
        y_hazard=mt.y_hazard.astype(np.float32, copy=True),
        target_mask=mt.target_mask.astype(bool, copy=True),
        y_rank=mt.y_rank.astype(np.float32, copy=True),
        y_topk=mt.y_topk.astype(np.float32, copy=True),
        team_mask=mt.team_mask.astype(bool, copy=True),
        node_mask=mt.node_mask.astype(bool, copy=True),
        phase_mask=mt.phase_mask.astype(bool, copy=True),
        team_ids=list(mt.team_ids),
        phase_indices=list(mt.phase_indices),
        node_feature_names=list(split.node_feature_names),
        global_feature_names=list(split.global_feature_names),
        edge_types=list(mt.edge_types),
    )


def apply_feature_preset(
    m: FactorizedPhaseTensors, feature_preset: str = "full"
) -> FactorizedPhaseTensors:
    if feature_preset == "full":
        return m
    available_names = list(m.global_feature_names) + list(m.node_feature_names)
    keep_names = set(feature_names_for_preset(feature_preset, available_names))
    global_indices = [i for i, name in enumerate(m.global_feature_names) if name in keep_names]
    node_indices = [i for i, name in enumerate(m.node_feature_names) if name in keep_names]
    if not global_indices:
        raise ValueError(f"feature_preset={feature_preset!r} selected no global features")
    if not node_indices:
        raise ValueError(f"feature_preset={feature_preset!r} selected no node features")
    return FactorizedPhaseTensors(
        match_id=m.match_id,
        X_node=m.X_node[:, :, node_indices].astype(np.float32, copy=True),
        G=m.G[:, global_indices].astype(np.float32, copy=True),
        A=m.A,
        y_hazard=m.y_hazard,
        target_mask=m.target_mask,
        y_rank=m.y_rank,
        y_topk=m.y_topk,
        team_mask=m.team_mask,
        node_mask=m.node_mask,
        phase_mask=m.phase_mask,
        team_ids=m.team_ids,
        phase_indices=m.phase_indices,
        node_feature_names=[m.node_feature_names[i] for i in node_indices],
        global_feature_names=[m.global_feature_names[i] for i in global_indices],
        edge_types=m.edge_types,
    )


def fetch_factorized_phase_match(
    match_id: str,
    *,
    include_v2_features: bool = True,
    include_combat_in: bool = False,
    feature_preset: str = "full",
) -> FactorizedPhaseTensors | None:
    mt = fetch_phase_match_tensors(
        match_id,
        include_v2_features=include_v2_features,
        include_combat_in=include_combat_in,
    )
    if mt is None:
        return None
    return apply_feature_preset(factorize_phase_tensors(mt), feature_preset)


def fetch_factorized_mr_match(
    match_id: str,
    *,
    include_v2_features: bool = True,
    include_combat_in: bool = False,
    feature_preset: str = "full",
) -> FactorizedMRTensors | None:
    phase = fetch_factorized_phase_match(
        match_id,
        include_v2_features=include_v2_features,
        include_combat_in=include_combat_in,
        feature_preset=feature_preset,
    )
    if phase is None:
        return None
    snap = fetch_snapshot_match_tensors(match_id)
    if snap is None:
        return None
    return FactorizedMRTensors(
        phase=phase,
        X_s=snap.X_s.astype(np.float32, copy=True),
        A_s=snap.A_s.astype(np.float32, copy=True),
        phase_of_snap=snap.phase_of_snap.astype(np.int64, copy=True),
        node_mask_s=snap.node_mask.astype(bool, copy=True),
        death_snap_idx=snap.death_snap_idx.astype(np.int64, copy=True),
        snapshot_ts=snap.snapshot_ts.astype(np.int64, copy=True),
    )


def _phase_to_torch(m: FactorizedPhaseTensors) -> dict:
    return {
        "X_node": torch.from_numpy(m.X_node.copy()),
        "G": torch.from_numpy(m.G.copy()),
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
        "node_feature_names": m.node_feature_names,
        "global_feature_names": m.global_feature_names,
        "edge_types": m.edge_types,
    }


def _mr_to_torch(m: FactorizedMRTensors) -> dict:
    out = _phase_to_torch(m.phase)
    out["X_s"] = torch.from_numpy(m.X_s.copy())
    out["A_s"] = torch.from_numpy(m.A_s.copy()).to_sparse()
    out["phase_of_snap"] = torch.from_numpy(m.phase_of_snap.copy())
    out["node_mask_s"] = torch.from_numpy(m.node_mask_s.copy())
    out["death_snap_idx"] = torch.from_numpy(m.death_snap_idx.copy())
    out["snapshot_ts"] = torch.from_numpy(m.snapshot_ts.copy())
    return out


class LazyFactorizedPhaseDataset(Dataset):
    def __init__(
        self,
        match_ids: Iterable[str],
        *,
        include_v2_features: bool = True,
        include_combat_in: bool = False,
        feature_preset: str = "full",
    ) -> None:
        self.match_ids = list(match_ids)
        self.include_v2_features = include_v2_features
        self.include_combat_in = include_combat_in
        self.feature_preset = feature_preset

    def __len__(self) -> int:
        return len(self.match_ids)

    def __getitem__(self, idx: int) -> dict:
        mid = self.match_ids[idx]
        item = fetch_factorized_phase_match(
            mid,
            include_v2_features=self.include_v2_features,
            include_combat_in=self.include_combat_in,
            feature_preset=self.feature_preset,
        )
        if item is None:
            raise RuntimeError(f"match {mid} is not factorized-phase-ready")
        return _phase_to_torch(item)


class CachedFactorizedPhaseDataset(Dataset):
    def __init__(
        self,
        match_ids: Iterable[str],
        *,
        include_v2_features: bool = True,
        include_combat_in: bool = False,
        feature_preset: str = "full",
    ) -> None:
        self.items: list[dict] = []
        for mid in match_ids:
            item = fetch_factorized_phase_match(
                mid,
                include_v2_features=include_v2_features,
                include_combat_in=include_combat_in,
                feature_preset=feature_preset,
            )
            if item is not None:
                self.items.append(_phase_to_torch(item))
        logger.info("CachedFactorizedPhaseDataset: %d matches loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class LazyFactorizedMRDataset(Dataset):
    def __init__(
        self,
        match_ids: Iterable[str],
        *,
        include_v2_features: bool = True,
        include_combat_in: bool = False,
        feature_preset: str = "full",
    ) -> None:
        self.match_ids = list(match_ids)
        self.include_v2_features = include_v2_features
        self.include_combat_in = include_combat_in
        self.feature_preset = feature_preset

    def __len__(self) -> int:
        return len(self.match_ids)

    def __getitem__(self, idx: int) -> dict:
        mid = self.match_ids[idx]
        item = fetch_factorized_mr_match(
            mid,
            include_v2_features=self.include_v2_features,
            include_combat_in=self.include_combat_in,
            feature_preset=self.feature_preset,
        )
        if item is None:
            raise RuntimeError(f"match {mid} is not factorized-MR-ready")
        return _mr_to_torch(item)


class CachedFactorizedMRDataset(Dataset):
    def __init__(
        self,
        match_ids: Iterable[str],
        *,
        include_v2_features: bool = True,
        include_combat_in: bool = False,
        feature_preset: str = "full",
    ) -> None:
        self.items: list[dict] = []
        for mid in match_ids:
            item = fetch_factorized_mr_match(
                mid,
                include_v2_features=include_v2_features,
                include_combat_in=include_combat_in,
                feature_preset=feature_preset,
            )
            if item is not None:
                self.items.append(_mr_to_torch(item))
        logger.info("CachedFactorizedMRDataset: %d matches loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def factorized_collate(batch: list[dict]) -> dict:
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


def feature_split_from_item(item: dict) -> FeatureIndexSplit:
    return split_feature_names(
        list(item["global_feature_names"]) + list(item["node_feature_names"])
    )
