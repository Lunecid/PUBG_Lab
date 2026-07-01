"""Snapshot graph null controls.

These controls operate on snapshot-resolution adjacency tensors with shape
``(B, T_s, E, N, N)``.  They are intentionally local to alive teams at each
snapshot so padded/dead nodes never enter the randomized graph.
"""

from __future__ import annotations

import torch


SNAPSHOT_EDGE_CONTROL_CHOICES = ("observed", "edge_shuffle", "gate_only")


def _active_indices(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask.bool(), as_tuple=False).flatten()


def _edge_shuffle_edges(
    A: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle alive-block adjacency entries per snapshot and relation channel."""
    out = torch.zeros_like(A)
    bsz, t_max, e_max, _n, _n2 = A.shape
    for b in range(bsz):
        for t in range(t_max):
            active_nodes = _active_indices(node_mask[b, t])
            if active_nodes.numel() <= 1:
                continue
            for e in range(e_max):
                sub = A[b, t, e][active_nodes][:, active_nodes]
                flat = sub.flatten()
                perm = torch.randperm(flat.numel(), generator=generator).to(A.device)
                shuffled = flat[perm].reshape_as(sub)
                out[b, t, e][active_nodes[:, None], active_nodes[None, :]] = shuffled
    return out


def _row_normalize_edges(A: torch.Tensor) -> torch.Tensor:
    row_sum = A.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > 0, A / row_sum.clamp_min(1e-12), A)


def _gate_only_edges(A: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Keep only binary contact gates inside alive blocks."""
    gate = (A > 0).to(dtype=A.dtype)
    alive = node_mask.bool()
    alive_block = alive[:, :, None, :, None] & alive[:, :, None, None, :]
    return torch.where(alive_block, gate, torch.zeros_like(gate))


def apply_snapshot_edge_control(
    A: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    control: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if control == "observed":
        return A
    if generator is None:
        generator = torch.Generator()
    if control == "edge_shuffle":
        return _row_normalize_edges(_edge_shuffle_edges(A, node_mask, generator=generator))
    if control == "gate_only":
        return _row_normalize_edges(_gate_only_edges(A, node_mask))
    raise ValueError(f"unknown snapshot edge control: {control}")
