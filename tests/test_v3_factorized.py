from __future__ import annotations

import numpy as np
import torch

from pubg_lab.survival.v3.data.factorized_data import _global_from_broadcast
from pubg_lab.survival.v3.experiments.experiment_matrix import EXPERIMENT_GROUPS
from pubg_lab.survival.v3.models.models import FactorizedMRGNN, FactorizedPhaseGNN, GRAPH_ENCODERS
from pubg_lab.survival.v3.experiments.run_matrix import _is_train_complete
from pubg_lab.survival.v3.data.spec import FEATURE_PRESETS, GLOBAL_FEATURES, OM_PRESET_FEATURES, feature_names_for_preset, expected_phase_feature_names, split_feature_names


def test_v3_feature_split_has_node_and_global_roles():
    split = split_feature_names(expected_phase_feature_names(include_v2=True))
    assert split.node_dim > 0
    assert split.global_dim == len([f for f in expected_phase_feature_names(include_v2=True) if f in set(GLOBAL_FEATURES)])
    assert "phase_idx_norm" in split.global_feature_names
    assert "alive_players_start_norm" in split.node_feature_names
    assert "hab_crowding_pressure" in split.global_feature_names
    assert "hab_local_enemy_count_200m" in split.node_feature_names


def test_global_extraction_uses_broadcast_source_and_masks_padded_phases():
    X = np.zeros((3, 4, 5), dtype=np.float32)
    X[:, 0, [1, 3]] = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    team_mask = np.array([True, True, False, False])
    phase_mask = np.array([True, True, False])
    G = _global_from_broadcast(X, (1, 3), team_mask, phase_mask)
    np.testing.assert_allclose(G[0], [1.0, 10.0])
    np.testing.assert_allclose(G[1], [2.0, 20.0])
    np.testing.assert_allclose(G[2], [0.0, 0.0])


def test_feature_presets_select_nonempty_node_and_global_features():
    names = expected_phase_feature_names(include_v2=True)
    for preset in FEATURE_PRESETS:
        selected = feature_names_for_preset(preset, names)
        split = split_feature_names(selected)
        assert split.node_dim > 0, preset
        assert split.global_dim > 0, preset
    assert "net_combat_out_degree" in feature_names_for_preset("network", names)
    assert "hab_crowding_pressure" in feature_names_for_preset("habitat", names)
    assert "path_length_norm" in feature_names_for_preset('stdb', names)
    assert "net_combat_out_degree" not in feature_names_for_preset("base66", names)


def test_om_feature_presets_are_clean_cumulative_ladder():
    names = expected_phase_feature_names(include_v2=True)
    order = [
        "om0_context",
        "om1_survival_capital",
        "om2_movement",
        "om3_pressure_habitat",
        "om4_threat_exposure",
        "om5_relation_structure",
        "om6_spatial_proxy",
    ]
    selected = {preset: set(feature_names_for_preset(preset, names)) for preset in order}

    for prev, nxt in zip(order, order[1:]):
        assert selected[prev] < selected[nxt]

    assert "alive_players_start_norm" not in selected["om0_context"]
    assert "alive_players_start_norm" in selected["om1_survival_capital"]
    assert "path_length_norm" in selected["om2_movement"]
    assert "hab_crowding_pressure" in selected["om3_pressure_habitat"]
    assert "damage_taken_norm" in selected["om4_threat_exposure"]
    assert "net_prox_degree_200m" in selected["om5_relation_structure"]
    assert "avg_enemy_density_100m_norm" in selected["om6_spatial_proxy"]
    assert "avg_enemy_density_100m_norm" not in selected["om5_relation_structure"]

    om6z = set(feature_names_for_preset("om6z_spatial_affordance", names))
    assert selected["om5_relation_structure"] < om6z
    assert "team_z_mean_norm" in om6z
    assert "contact_delta_z_mean_norm" in om6z
    assert "high_ground_x_boundary_pressure" in om6z
    assert "avg_enemy_density_100m_norm" not in om6z


def test_om_ladder_excludes_resource_sensitivity_features():
    resource_features = {
        "pickup_count_norm",
        "equip_count_norm",
        "use_count_norm",
        "unique_item_count_norm",
        "weapon_pickup_count_norm",
        "weapon_equip_count_norm",
        "ammo_pickup_count_norm",
        "armor_pickup_count_norm",
        "throwable_pickup_count_norm",
        "heal_use_count_norm",
        "boost_use_count_norm",
        "carepackage_pickup_count_norm",
        "lootbox_pickup_count_norm",
        "vehicle_ratio",
        "vehicle_speed_mean_norm",
        "vehicle_speed_max_norm",
    }
    for preset, features in OM_PRESET_FEATURES.items():
        assert not (set(features) & resource_features), preset


def test_om_ladder_experiment_group_uses_clean_presets_in_order():
    rows = EXPERIMENT_GROUPS["om_ladder"]
    assert [row["feature_preset"] for row in rows] == [
        "om0_context",
        "om1_survival_capital",
        "om2_movement",
        "om3_pressure_habitat",
        "om4_threat_exposure",
        "om5_relation_structure",
        "om6_spatial_proxy",
    ]
    assert all(row["graph_encoder"] == "none" for row in rows)
    assert all(row["edge_mode"] == "none" for row in rows)

    z_rows = EXPERIMENT_GROUPS["om6z_sensitivity"]
    assert [row["feature_preset"] for row in z_rows] == ["om6z_spatial_affordance"]
    assert all(row["graph_encoder"] == "none" for row in z_rows)
    assert all(row["edge_mode"] == "none" for row in z_rows)


def test_run_matrix_does_not_treat_partial_checkpoint_as_complete(tmp_path):
    checkpoint = tmp_path / "model.pt"
    eval_json = tmp_path / "eval_test.json"
    train_done = tmp_path / "train_complete.json"

    assert not _is_train_complete(checkpoint, eval_json, train_done)
    checkpoint.write_text("partial", encoding="utf-8")
    assert not _is_train_complete(checkpoint, eval_json, train_done)
    train_done.write_text("{}", encoding="utf-8")
    assert _is_train_complete(checkpoint, eval_json, train_done)
    train_done.unlink()
    eval_json.write_text("{}", encoding="utf-8")
    assert _is_train_complete(checkpoint, eval_json, train_done)


def test_factorized_phase_model_forward_all_graph_encoders():
    B, T, N, Fn, Fg, E = 2, 8, 6, 12, 4, 3
    Xn = torch.randn(B, T, N, Fn)
    G = torch.randn(B, T, Fg)
    A = torch.zeros(B, T, E, N, N)
    eye = torch.eye(N).view(1, 1, 1, N, N)
    A[:, :, 0:1] = eye
    phase_mask = torch.ones(B, T, dtype=torch.bool)
    for enc in GRAPH_ENCODERS:
        model = FactorizedPhaseGNN(
            Fn,
            Fg,
            hidden=16,
            mid=24,
            num_blocks=1,
            num_edge_types=E,
            graph_encoder=enc,
            global_fusion="concat",
        )
        hazard, rank, topk = model(Xn, G, A, phase_mask=phase_mask)
        assert hazard.shape == (B, T - 1, N)
        assert rank.shape == (B, T - 1, N)
        assert topk.shape == (B, T - 1, N, 2)


def test_factorized_mr_model_forward_smoke():
    B, Tp, Ts, N, Fn, Fg, Fs, Ep, Es = 2, 8, 20, 6, 10, 4, 5, 3, 2
    Xs = torch.randn(B, Ts, N, Fs)
    As = torch.zeros(B, Ts, Es, N, N)
    Ap = torch.zeros(B, Tp, Ep, N, N)
    eye = torch.eye(N)
    As[:, :, 0] = eye
    Ap[:, :, 0] = eye
    phase_of_snap = torch.arange(Ts).view(1, Ts).repeat(B, 1) % Tp + 1
    node_mask_s = torch.ones(B, Ts, N, dtype=torch.bool)
    Xn = torch.randn(B, Tp, N, Fn)
    G = torch.randn(B, Tp, Fg)
    phase_mask = torch.ones(B, Tp, dtype=torch.bool)
    model = FactorizedMRGNN(
        Fs,
        Fn,
        Fg,
        snap_hidden=8,
        snap_mid=12,
        phase_hidden=16,
        phase_mid=24,
        phase_blocks=1,
        graph_encoder="rgcn",
    )
    hazard, survival, rank, topk = model(
        Xs,
        As,
        phase_of_snap,
        node_mask_s,
        Xn,
        G,
        Ap,
        phase_mask=phase_mask,
    )
    assert hazard.shape == (B, Tp - 1, N)
    assert survival.shape == (B, Ts, N)
    assert rank.shape == (B, Tp - 1, N)
    assert topk.shape == (B, Tp - 1, N, 2)
