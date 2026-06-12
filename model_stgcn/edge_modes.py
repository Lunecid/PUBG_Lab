"""Edge-channel ablation helpers for phase-level DynamicSTGNN experiments."""

from __future__ import annotations

import torch

from stdb.queries import PHASE_EDGE_TYPES


EDGE_MODE_CHOICES = (
    "full",
    "none",
    "proximity",
    "proximity_combat",
    "proximity_copressure",
    "combat_only",
    "copressure_only",
)


def active_edge_indices(edge_mode: str, edge_types: list[str] | tuple[str, ...] = PHASE_EDGE_TYPES) -> list[int]:
    """Return channel indices kept for an ablation mode."""
    edge_index = {name: i for i, name in enumerate(edge_types)}
    if edge_mode == "full":
        return list(range(len(edge_types)))
    if edge_mode == "none":
        return []
    if edge_mode == "proximity":
        return [edge_index["proximity"]]
    if edge_mode == "proximity_combat":
        return [edge_index["proximity"], edge_index["combat"]]
    if edge_mode == "proximity_copressure":
        return [edge_index["proximity"], edge_index["copressure"]]
    if edge_mode == "combat_only":
        return [edge_index["combat"]]
    if edge_mode == "copressure_only":
        return [edge_index["copressure"]]
    raise ValueError(f"unknown edge_mode: {edge_mode}")


def apply_edge_mode(
    A: torch.Tensor,
    edge_mode: str,
    edge_types: list[str] | tuple[str, ...] = PHASE_EDGE_TYPES,
) -> torch.Tensor:
    """Zero inactive edge channels while preserving tensor shape.

    The DynamicSTGNN residual path remains active. Therefore `none` is a
    sequence-only neural baseline with no neighbor aggregation.
    """
    keep = active_edge_indices(edge_mode, edge_types)
    if len(keep) == len(edge_types):
        return A
    out = torch.zeros_like(A)
    if keep:
        out[:, :, keep] = A[:, :, keep]
    return out


def edge_active_mask(
    edge_mode: str,
    edge_types: list[str] | tuple[str, ...] = PHASE_EDGE_TYPES,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask for edge channels active in a model run."""
    mask = torch.zeros(len(edge_types), dtype=torch.bool, device=device)
    keep = active_edge_indices(edge_mode, edge_types)
    if keep:
        mask[keep] = True
    return mask
