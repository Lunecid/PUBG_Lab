"""Phase-graph randomization controls for V3 factorized experiments."""

from __future__ import annotations

import torch


EDGE_CONTROL_CHOICES = ("observed", "node_shuffle", "edge_shuffle")


def _active_indices(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask.bool(), as_tuple=False).flatten()


def _node_shuffle_edges(A: torch.Tensor, node_mask: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Relabel alive nodes per match-phase, preserving graph structure."""
    out = torch.zeros_like(A)
    bsz, t_max, e_max, _n, _n2 = A.shape
    for b in range(bsz):
        for t in range(t_max):
            active_nodes = _active_indices(node_mask[b, t])
            if active_nodes.numel() <= 1:
                continue
            perm = torch.randperm(active_nodes.numel(), generator=generator).to(A.device)
            for e in range(e_max):
                sub = A[b, t, e][active_nodes][:, active_nodes]
                shuffled = sub[perm][:, perm]
                out[b, t, e][active_nodes[:, None], active_nodes[None, :]] = shuffled
    return out


def _edge_shuffle_edges(A: torch.Tensor, node_mask: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Shuffle alive-block edge entries per match-phase-channel."""
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


def apply_phase_edge_control(
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
    if control == "node_shuffle":
        return _row_normalize_edges(_node_shuffle_edges(A, node_mask, generator=generator))
    if control == "edge_shuffle":
        return _row_normalize_edges(_edge_shuffle_edges(A, node_mask, generator=generator))
    raise ValueError(f"unknown edge_control: {control}")
