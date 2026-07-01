from __future__ import annotations

import torch

from model_stgcn.snapshot_relative_edge import SnapshotRelativeEdgeHazard


FEATURE_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rel_zone_x",
    "rel_zone_y",
    "speed_mps",
    "dist_blue_boundary_m",
    "inside_blue",
    "dist_target_center_m",
    "mean_health",
    "min_health",
    "alive_n",
    "vehicle_ratio",
]


def test_snapshot_relative_edge_forward_shape():
    B, Ts, N, F = 2, 18, 6, len(FEATURE_NAMES)
    torch.manual_seed(3)
    X = torch.randn(B, Ts, N, F)
    A = torch.zeros(B, Ts, 2, N, N)
    A[:, :, 0, :, :] = 1.0 / N
    phase_of_snap = (torch.arange(Ts).view(1, Ts).repeat(B, 1) // 2).clamp(max=8)
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    model = SnapshotRelativeEdgeHazard(
        F,
        feature_names=FEATURE_NAMES,
        hidden=8,
        num_layers=1,
        edge_hidden=4,
        dropout=0.0,
    )

    hazard, cut_mask = model(X, A, phase_of_snap, node_mask)

    assert hazard.shape == (B, 7, N)
    assert cut_mask.shape == (B, 7)


def test_snapshot_relative_edge_features_are_contact_gated_and_directed():
    B, Ts, N, F = 1, 1, 3, len(FEATURE_NAMES)
    X = torch.zeros(B, Ts, N, F)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X[0, 0, 0, idx["pos_z"]] = 0.25
    X[0, 0, 1, idx["pos_z"]] = 0.10
    X[0, 0, 0, idx["alive_n"]] = 1.0
    X[0, 0, 1, idx["alive_n"]] = 0.5
    X[0, 0, 0, idx["mean_health"]] = 0.8
    X[0, 0, 1, idx["mean_health"]] = 0.2
    A = torch.zeros(B, Ts, 2, N, N)
    A[0, 0, 0, 0, 0] = 1.0
    A[0, 0, 0, 0, 1] = 1.0
    A[0, 0, 1, 1, 0] = 1.0
    node_mask = torch.tensor([[[True, True, False]]])
    model = SnapshotRelativeEdgeHazard(F, feature_names=FEATURE_NAMES, hidden=8)

    edge_feat, edge_mask = model.build_edge_features(X, A, node_mask)
    name_idx = {name: i for i, name in enumerate(model.edge_feature_names)}

    assert not bool(edge_mask[0, 0, 0, 0])
    assert bool(edge_mask[0, 0, 0, 1])
    assert bool(edge_mask[0, 0, 1, 0])
    assert not bool(edge_mask[0, 0, 0, 2])
    assert torch.isclose(edge_feat[0, 0, 0, 1, name_idx["delta_z"]], torch.tensor(0.15))
    assert torch.isclose(edge_feat[0, 0, 1, 0, name_idx["delta_z"]], torch.tensor(-0.15))
    assert torch.isclose(edge_feat[0, 0, 0, 1, name_idx["delta_alive"]], torch.tensor(0.5))
    assert torch.isclose(edge_feat[0, 0, 0, 1, name_idx["delta_health"]], torch.tensor(0.6))


def test_sparse_relative_edge_keeps_knn_and_combat_not_complete_graph():
    B, Ts, N, F = 1, 1, 4, len(FEATURE_NAMES)
    X = torch.zeros(B, Ts, N, F)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X[0, 0, :, idx["pos_x"]] = torch.tensor([0.0, 0.1, 0.5, 0.9])
    A = torch.zeros(B, Ts, 2, N, N)
    A[0, 0, 0] = 1.0
    A[0, 0, 1, 0, 3] = 1.0
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    model = SnapshotRelativeEdgeHazard(
        F,
        feature_names=FEATURE_NAMES,
        hidden=8,
        relative_edge_gate="knn_combat",
        relative_edge_k=1,
    )

    edge_feat, edge_mask = model.build_edge_features(X, A, node_mask)
    name_idx = {name: i for i, name in enumerate(model.edge_feature_names)}

    assert int(edge_mask[0, 0].sum()) == 5
    assert bool(edge_mask[0, 0, 0, 1])
    assert bool(edge_mask[0, 0, 0, 3])
    assert not bool(edge_mask[0, 0, 0, 2])
    assert torch.isclose(edge_feat[0, 0, 0, 3, name_idx["contact_gate"]], torch.tensor(1.0))


def test_sparse_relative_edge_feature_roll_null_preserves_mask_and_breaks_alignment():
    B, Ts, N, F = 1, 1, 4, len(FEATURE_NAMES)
    X = torch.zeros(B, Ts, N, F)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X[0, 0, :, idx["pos_x"]] = torch.tensor([0.0, 0.1, 0.5, 0.9])
    X[0, 0, :, idx["pos_z"]] = torch.tensor([0.0, 1.0, 3.0, 6.0])
    A = torch.zeros(B, Ts, 2, N, N)
    A[0, 0, 0] = 1.0
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    base = SnapshotRelativeEdgeHazard(
        F,
        feature_names=FEATURE_NAMES,
        hidden=8,
        relative_edge_gate="knn_combat",
        relative_edge_k=1,
        relative_edge_null="none",
    )
    null = SnapshotRelativeEdgeHazard(
        F,
        feature_names=FEATURE_NAMES,
        hidden=8,
        relative_edge_gate="knn_combat",
        relative_edge_k=1,
        relative_edge_null="feature_roll",
    )

    base_feat, base_mask = base.build_edge_features(X, A, node_mask)
    null_feat, null_mask = null.build_edge_features(X, A, node_mask)
    name_idx = {name: i for i, name in enumerate(base.edge_feature_names)}
    delta_idx = name_idx["delta_z"]
    contact_idx = name_idx["contact_gate"]

    torch.testing.assert_close(base_mask, null_mask)
    torch.testing.assert_close(base_feat[base_mask][..., contact_idx], null_feat[null_mask][..., contact_idx])
    assert not torch.allclose(base_feat[base_mask][..., delta_idx], null_feat[null_mask][..., delta_idx])
    torch.testing.assert_close(
        torch.sort(base_feat[base_mask][..., delta_idx]).values,
        torch.sort(null_feat[null_mask][..., delta_idx]).values,
    )


def test_sparse_relative_edge_topology_roll_null_preserves_row_degree():
    B, Ts, N, F = 1, 1, 5, len(FEATURE_NAMES)
    X = torch.zeros(B, Ts, N, F)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X[0, 0, :, idx["pos_x"]] = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0])
    A = torch.zeros(B, Ts, 2, N, N)
    A[0, 0, 0] = 1.0
    node_mask = torch.ones(B, Ts, N, dtype=torch.bool)
    base = SnapshotRelativeEdgeHazard(
        F,
        feature_names=FEATURE_NAMES,
        hidden=8,
        relative_edge_gate="knn_combat",
        relative_edge_k=1,
    )
    null = SnapshotRelativeEdgeHazard(
        F,
        feature_names=FEATURE_NAMES,
        hidden=8,
        relative_edge_gate="knn_combat",
        relative_edge_k=1,
        relative_edge_null="topology_roll",
    )

    _base_feat, base_mask = base.build_edge_features(X, A, node_mask)
    null_feat, null_mask = null.build_edge_features(X, A, node_mask)
    name_idx = {name: i for i, name in enumerate(null.edge_feature_names)}

    assert int(base_mask.sum()) == int(null_mask.sum())
    torch.testing.assert_close(base_mask.sum(dim=-1), null_mask.sum(dim=-1))
    assert not torch.equal(base_mask, null_mask)
    assert not bool(torch.diagonal(null_mask[0, 0]).any())
    torch.testing.assert_close(
        null_feat[null_mask][..., name_idx["contact_gate"]],
        torch.ones(int(null_mask.sum())),
    )
