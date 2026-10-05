"""Dataset for snapshot-resolution phase survival training.

Each sample keeps the 10s snapshot tensors as the model input and uses the
existing phase labels only as supervision at phase cut points.
"""

from __future__ import annotations

import logging
import hashlib
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from stdb.queries import fetch_phase_match_tensors
from stdb.snapshot_queries import fetch_snapshot_match_tensors

logger = logging.getLogger("model_stgcn.snapshot_phase_dataset")

SNAPSHOT_PHASE_CACHE_VERSION = "snapshot_phase_dataset_v1"


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
        "y_rank": torch.from_numpy(phase.y_rank.copy()),
        "y_topk": torch.from_numpy(phase.y_topk.copy()),
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
    def __init__(
        self,
        match_ids: Iterable[str],
        *,
        include_edges: bool = True,
        cache_dir: str | Path | None = None,
        cache_label: str | None = None,
    ) -> None:
        ids = list(match_ids)
        cache_path = _cache_path(
            ids,
            include_edges=include_edges,
            cache_dir=cache_dir,
            cache_label=cache_label,
        )
        if cache_path is not None and cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                if _cache_payload_matches(payload, ids, include_edges=include_edges):
                    self.items = list(payload["items"])
                    logger.info(
                        "CachedSnapshotPhaseDataset: disk cache hit path=%s loaded=%d",
                        cache_path,
                        len(self.items),
                    )
                    return
                logger.warning("CachedSnapshotPhaseDataset: cache metadata mismatch path=%s", cache_path)
            except Exception as exc:  # pragma: no cover - defensive cache recovery
                logger.warning(
                    "CachedSnapshotPhaseDataset: failed to read cache path=%s error=%s",
                    cache_path,
                    exc,
                )

        self.items: list[dict] = []
        started = time.time()
        total = len(ids)
        for idx, mid in enumerate(ids, start=1):
            item = fetch_snapshot_phase(mid, include_edges=include_edges)
            if item is None:
                continue
            self.items.append(item)
            if idx == 1 or idx % 250 == 0 or idx == total:
                logger.info(
                    "CachedSnapshotPhaseDataset: loaded=%d/%d kept=%d elapsed=%.1fs",
                    idx,
                    total,
                    len(self.items),
                    time.time() - started,
                )
        logger.info(
            "CachedSnapshotPhaseDataset: %d/%d matches loaded in %.1fs",
            len(self.items),
            total,
            time.time() - started,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": SNAPSHOT_PHASE_CACHE_VERSION,
                "include_edges": bool(include_edges),
                "match_ids": ids,
                "items": self.items,
            }
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(cache_path)
            logger.info(
                "CachedSnapshotPhaseDataset: wrote disk cache path=%s items=%d",
                cache_path,
                len(self.items),
            )

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


def _cache_path(
    ids: list[str],
    *,
    include_edges: bool,
    cache_dir: str | Path | None,
    cache_label: str | None,
) -> Path | None:
    if cache_dir is None:
        return None
    digest_src = "\n".join(ids).encode("utf-8")
    digest = hashlib.sha1(digest_src).hexdigest()[:16]
    label = _safe_cache_label(cache_label or "split")
    edge_tag = "edges" if include_edges else "no_edges"
    return Path(cache_dir) / f"{label}_{edge_tag}_{digest}.pt"


def _safe_cache_label(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return out.strip("_") or "split"


def _cache_payload_matches(payload: object, ids: list[str], *, include_edges: bool) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("version") == SNAPSHOT_PHASE_CACHE_VERSION
        and bool(payload.get("include_edges")) == bool(include_edges)
        and list(payload.get("match_ids") or []) == ids
        and isinstance(payload.get("items"), list)
    )
