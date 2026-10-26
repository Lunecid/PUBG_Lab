"""Tests for model_stgcn/dump_phase_tier.py (design v2.3: one clock + cap, labels v2, event-based alive
counts, population rules).

DB-free part: a synthetic ``RawMatch`` exercises every convention documented in the module.
DB-gated part (``requires_db``): the population rules on the real split manifests and the independent
re-derivation of one real match (tools/verify_dump_match.py).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

import pubg_lab.survival.data.dump_phase_tier as D
import pubg_lab.data.stdb.temporal.causal_cut as causal_cut
from tests.conftest import requires_db

F = D.N_FEATURES
MAP = 8160.0
INF = float("inf")
DELTA = 8.0  # game clock lags the event clock by 8 s in the synthetic match


def _members():
    """Team 11 dies in phase 3 (last kill event 340 s, a bot without kill event at ts 349.3 + 0.7),
    team 12 dies in phase 2 (kills at 240 / 250 s), team 15 wins."""
    return [
        (11, "account.a", "byplayer", 339.3, 340.0),
        (11, "ai.1", "byplayer", 349.3, None),
        (12, "account.b", "byplayer", 239.3, 240.0),
        (12, "account.c", "byplayer", 249.4, 250.0),
        (15, "account.d", "alive", 1640.0, None),
        (15, "ai.2", "alive", None, None),
    ]


def _raw(**over) -> D.RawMatch:
    """3 teams (ids 11, 12, 15), phases 1..3 present (phase 3 is the last one of the match).

    Phase bounds on the event clock W: p1 (90, 190), p2 (200, 300), p3 (310, 400).  With delta = 8 the
    windows on the game clock G are [82, 172], [192, 292], [302, 392]: cut grid points 170, 280, 380.
    Team 12 dies in phase 2.  Team 15 is flagged not alive at the start of phase 3 although its
    phase-3 pipeline target is defined (the node_mask raise).
    """
    N, P_loader = 3, D.N_PHASES
    X = np.zeros((P_loader, 128, F), dtype=np.float32)
    for pos in range(3):
        X[pos, :N, 0] = pos + 1  # phase tag
        X[pos, :N, 1] = np.arange(N)  # team tag
    node = np.zeros((P_loader, 128), dtype=bool)
    node[0, :3] = True
    node[1, :3] = True
    node[2, [0, 2]] = True  # team 12 dead in phase 3, team 15 forgotten by the label table
    node[2, 2] = False
    y = np.zeros((D.N_PHASES - 1, 128), dtype=np.float32)
    tgt = np.zeros((D.N_PHASES - 1, 128), dtype=bool)
    tgt[0, :3] = True  # cut 1: everyone alive at the end
    tgt[1, [0, 2]] = True  # cut 2: team 12 died during phase 2 -> not alive at end
    y[0, 1] = 1.0  # team 12 eliminated in phase 2 (hazard at cut 1)
    tgt[2, 2] = True  # cut 3: team 15 target defined although node_mask says dead at start
    rank = np.zeros(128, dtype=np.float32)
    rank[:3] = [1 / 3, 3 / 3, 2 / 3]
    topk = np.zeros((128, 2), dtype=np.float32)
    topk[:3] = [[1, 1], [0, 1], [1, 1]]
    edges = [
        (1, "proximity", 11, 12, 0.9), (1, "proximity", 12, 11, 0.9),
        (1, "combat", 12, 11, 4.2),            # attacker 12 -> victim 11
        (2, "copressure", 11, 15, 0.3), (2, "copressure", 15, 11, 0.3),
        (2, "proximity", 11, 11, 1.0),         # self loop: dropped
        (2, "proximity", 11, 99, 0.2),         # unknown team: dropped
        (9, "proximity", 11, 12, 0.5),         # phase outside 1..8 present set: dropped
    ]  # fmt: skip
    bounds = {1: (90.0, 190.0), 2: (200.0, 300.0), 3: (310.0, 400.0)}
    zone_rows = {  # keyed by the G grid time
        ("safe", 170): (4000.0, 4100.0, 2000.0), ("poison", 170): (3900.0, 4000.0, 1000.0),
        ("safe", 280): (3900.0, 4000.0, 1000.0),  # poison missing at 280 -> zone_valid False
        ("safe", 380): (3800.0, 3900.0, 500.0), ("poison", 380): (3700.0, 3800.0, 250.0),
        ("safe", 180): (9999.0, 9999.0, 9999.0),  # after the phase-1 cut: must never be read
        ("poison", 180): (9999.0, 9999.0, 9999.0),
    }  # fmt: skip
    pos_rows = {  # (G time, x_m, y_m)
        11: [(81.0, 1111.0, 1111.0), (82.0, 1000.0, 1000.0), (170.0, 2000.0, 3000.0), (180.0, 7777.0, 7777.0),
             (192.0, 1.0, 1.0), (280.0, 3000.0, 3100.0), (290.0, 8888.0, 8888.0)],
        12: [(82.0, 500.0, 600.0), (160.0, 800.0, 900.0)],   # no row at 170 -> falls back to 160
        15: [(81.0, 55.0, 66.0), (200.0, 10.0, 20.0)],       # t = 81 is before the window start (82)
    }  # fmt: skip
    base = dict(
        match_id="m-test",
        map_name="Baltic_Main",
        team_ids=[11, 12, 15],
        phase_indices=[1, 2, 3],
        X_loader=X,
        node_mask_loader=node,
        y_hazard_loader=y,
        target_mask_loader=tgt,
        y_rank=rank,
        y_topk=topk,
        edge_rows=edges,
        bounds=bounds,
        zone_rows=zone_rows,
        pos_rows=pos_rows,
        clock_delta_s=DELTA,
        clock_delta_median_s=DELTA,
        last_phase_idx=3,
        match_type="official",
        team_won={11: False, 12: False, 15: True},
        members=_members(),
        monster_damage={"account.a": 30.0, "account.c": 5.0, "account.nobody": 99.0},
    )
    base.update(over)
    return D.RawMatch(**base)


def _col(name: str) -> int:
    return D.FEATURE_NAMES.index(name)


# ------------------------------------------------------------------------------ feature groups
def test_feature_layout_is_150_and_unique():
    assert D.N_FEATURES == 150 == len(D.FEATURE_NAMES) == len(set(D.FEATURE_NAMES))
    assert D.FEATURE_NAMES[0] == "phase_idx_norm"


def test_feature_groups_have_zone_and_habitat_families_without_alive_counts():
    groups = D.build_feature_groups()
    assert {"zone", "habitat", "alive_counts", "replaced"} <= set(groups)
    names = D.FEATURE_NAMES
    assert len(groups["habitat"]) == 31
    assert all(names[c].startswith("hab_") for c in groups["habitat"])
    assert len(groups["zone"]) == 13
    for name, cols in groups.items():
        assert cols == sorted(set(cols)), name
        assert all(0 <= c < D.N_FEATURES for c in cols), name
    assert not set(groups["zone"]) & set(groups["habitat"])
    # design v2.2: the headcount variables are not spatial, the noHab arms keep them
    alive = {names[c] for c in groups["alive_counts"]}
    assert alive == {
        "avg_alive_teams_norm", "avg_alive_players_norm", "hab_alive_teams",
        "alive_players_start_norm", "alive_players_end_norm", "alive_players_min_norm", "alive_players_avg_norm",
    }  # fmt: skip
    assert not (set(groups["zone"]) | set(groups["habitat"])) & set(groups["alive_counts"])
    assert "hab_safe_area_per_alive_team" in {names[c] for c in groups["habitat"]}  # spatial: removed with noHab
    # strict sub-cuts are contained in the headline family
    assert set(groups["zone_geometry"]) <= set(groups["zone"])
    assert set(groups["zone_local"]) | set(groups["zone_global"]) == set(groups["zone"])
    assert len(groups["network"]) == 43 and len(groups["z_affordance"]) == 9
    # the *-noHab arms drop zone + habitat: 44 columns, 106 remain
    dropped = set(groups["zone"]) | set(groups["habitat"])
    assert len(dropped) == 44 and D.N_FEATURES - len(dropped) == 106


def test_replaced_features_are_the_four_alive_player_columns():
    assert D.REPLACED_FEATURES == (
        "alive_players_start_norm", "alive_players_end_norm", "alive_players_min_norm", "alive_players_avg_norm",
    )  # fmt: skip
    assert all(n in D.FEATURE_NAMES for n in D.REPLACED_FEATURES)
    assert [D.build_feature_groups()["replaced"]] == [sorted(_col(n) for n in D.REPLACED_FEATURES)]


def test_zone_columns_match_the_ptr_data_layer_when_present():
    ptr_data = pytest.importorskip('pubg_lab.survival.ptr.data.data')
    assert tuple(D.ZONE_COLUMNS) == tuple(ptr_data.ZONE_COLUMNS)


def test_map_size_table_covers_every_required_cohort():
    for spec in D.REQUIRED_COHORTS:
        assert spec.map_name in D.MAP_SIZE_M, spec.map_name


# ------------------------------------------------------------------------------ death model
def test_member_death_times_alive_kill_event_and_time_survived_fallback():
    members = [
        (1, "account.a", "alive", 1700.0, 1650.0),        # revived: alive wins; its -50 s pair stays in the offset sample
        (1, "account.b", "byplayer", 500.0, 500.9),       # kill event: 0.9 s after time_survived
        (2, "account.c", "byplayer", 600.0, 600.5),       # offset sample 0.5 s
        (2, "account.d", "suicide", 700.0, None),         # no kill event -> ts + median offset
        (2, "account.f", "byplayer", 800.0, 800.9),
        (3, "ai.1", "byplayer", 100.0, 400.0),            # bots never define the offset
        (3, "account.e", "byplayer", None, None),         # nothing known -> inf
    ]
    d, off = D.member_death_times(members)
    # offset sample = every human member with a kill event and a time_survived (the audit's rule, revived
    # players included): median(-50, 0.5, 0.9, 0.9) = 0.7
    assert off == pytest.approx(0.7)
    assert d[0] == INF
    assert d[1] == pytest.approx(500.9) and d[2] == pytest.approx(600.5)
    assert d[3] == pytest.approx(700.0 + 0.7)
    assert d[4] == pytest.approx(800.9)
    assert d[5] == pytest.approx(400.0) and d[6] == INF


def test_member_death_times_use_the_documented_fallback_offset_without_human_pairs():
    d, off = D.member_death_times([(1, "ai.1", "byplayer", 100.0, None), (1, "ai.2", "byplayer", 50.0, 300.0)])
    assert off == D.FALLBACK_OFFSET_S == 0.70
    assert d == [pytest.approx(100.7), pytest.approx(300.0)]


def test_team_death_times_take_the_max_and_stay_infinite_while_a_member_lives():
    t, per_team, off = D.team_death_times(_members(), [11, 12, 15, 99])
    assert t[0] == pytest.approx(350.0) and t[1] == pytest.approx(250.0)
    assert t[2] == INF, "a team with a living member (the winner) is censored"
    assert math.isnan(t[3]), "no roster -> unknown, never a death time"
    assert sorted(per_team) == [11, 12, 15] and off == pytest.approx(0.7)


def test_alive_counts_step_function_and_time_average():
    deaths = [240.0, 250.0]
    assert D.alive_counts(deaths, 200.0, 300.0) == (2.0, 0.0, 0.0, pytest.approx(0.4 + 0.5))
    assert D.alive_counts(deaths, 90.0, 190.0) == (2.0, 2.0, 2.0, pytest.approx(2.0))
    assert D.alive_counts(deaths, 310.0, 400.0) == (0.0, 0.0, 0.0, 0.0)
    assert D.alive_counts([INF, INF], 200.0, 300.0) == (2.0, 2.0, 2.0, 2.0)
    n_s, n_e, n_min, n_avg = D.alive_counts([200.0, 301.0], 200.0, 300.0)  # death exactly at the start: dead
    assert (n_s, n_e, n_min) == (1.0, 1.0, 1.0) and n_avg == pytest.approx(0.0 + 1.0)
    assert D.alive_counts([5.0], 100.0, 100.0)[3] == 0.0  # zero-length phase: avg falls back to the start count


# ------------------------------------------------------------------------------ labels v2
def test_labels_v2_semantics_terminal_phase_and_censoring():
    bounds = {1: (90.0, 190.0), 2: (200.0, 300.0), 3: (310.0, 400.0)}
    T = np.array([350.0, 250.0, INF, np.nan])
    tgt, y, node = D.labels_v2(T, bounds, last_phase_idx=3)
    assert tgt.shape == y.shape == node.shape == (D.N_PHASES, 4)
    assert tgt[0].tolist() == [True, True, True, False]  # T > phase_end_s[1]; unknown team never a target
    assert tgt[1].tolist() == [True, False, True, False]  # team 12 died before the end of phase 2
    assert not tgt[2:].any(), "phase k+1 must exist: cut 3 has no phase 4"
    assert y[0].tolist() == [0.0, 1.0, 0.0, 0.0]  # team 12 dies inside phase 2 (<= phase_end_s[2])
    assert y[1].tolist() == [1.0, 0.0, 0.0, 0.0], "k+1 = 3 is the last phase: match end instead of its bound"
    assert (y[~tgt] == 0).all()
    assert node[0].tolist() == [True, True, True, False]
    assert node[2].tolist() == [True, False, True, False]  # T > phase_start_s[3] = 310
    assert not node[3:].any()


def test_labels_v2_use_the_next_phase_bound_unless_it_is_the_last_phase():
    bounds = {1: (90.0, 190.0), 2: (200.0, 300.0), 3: (310.0, 400.0)}
    T = np.array([350.0, 450.0, INF])
    _, y, _ = D.labels_v2(T, bounds, last_phase_idx=9)  # the match goes on after phase 3
    assert y[1].tolist() == [1.0, 0.0, 0.0], "350 <= phase_end_s[3] = 400 -> positive; 450 is later; winner censored"
    _, y_last, _ = D.labels_v2(T, bounds, last_phase_idx=3)
    assert y_last[1].tolist() == [1.0, 1.0, 0.0], "at the last phase every finite death counts"


def test_labels_v2_need_both_phases_to_exist():
    bounds = {1: (90.0, 190.0), 3: (310.0, 400.0)}  # phase 2 missing
    tgt, y, _ = D.labels_v2(np.array([350.0]), bounds, last_phase_idx=3)
    assert not tgt.any() and not y.any()


def test_label_agreement_confusion_matrix_and_hazards():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    agree = D.label_agreement([m])
    # cut 1: identical (2 negatives, 1 positive); cut 2: team 11 is a v2 positive that the pipeline calls
    # negative (the pipeline's alive-at-end snapshot is stale); team 12 outside both risk sets; team 15 negative
    assert agree["confusion_rows_pipeline_cols_v2"] == [[1, 0, 0], [0, 3, 1], [0, 0, 1]]
    assert agree["rows"] == 6 and agree["agree"] == pytest.approx(5 / 6)
    assert agree["positives_pipeline"] == 1 and agree["positives_v2"] == 2
    assert agree["positive_precision_of_pipeline"] == 1.0 and agree["positive_recall_of_pipeline"] == 0.5
    assert agree["hazard_by_cut_pipeline"][0] == pytest.approx(1 / 3)
    assert agree["hazard_by_cut_v2"][1] == pytest.approx(1 / 2)
    assert agree["hazard_by_cut_v2"][2] is None  # no valid cut 3 (phase 4 missing)
    assert agree["winner_teams"] == 1 and agree["winner_teams_with_finite_death_s"] == 0
    # a winner whose members all died keeps a finite death time: counted and reported, never overridden
    raw = _raw(members=[(11, "account.a", "byplayer", 339.3, 340.0), (12, "account.b", "byplayer", 239.3, 240.0),
                        (15, "account.d", "byplayer", 399.0, 399.9)])
    odd = D.label_agreement([D.assemble_match(raw, map_size_m=MAP)])
    assert odd["winner_teams_with_finite_death_s"] == 1 and odd["v2_positives_on_those_winners"] == 1


# ------------------------------------------------------------------------------ assembly
def test_assembly_shapes_dtypes_and_phase_axis():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    P, N = D.N_PHASES, 3
    assert m["X"].shape == (P, N, F) and m["X"].dtype == torch.float32
    assert m["team_ids"].dtype == torch.int64 and m["team_ids"].tolist() == [11, 12, 15]
    for key in ("node_mask", "target_mask", "node_mask_v2", "target_mask_v2", "node_mask_label"):
        assert m[key].shape == (P, N) and m[key].dtype == torch.bool, key
    for key in ("y_hazard", "y_hazard_v2"):
        assert m[key].shape == (P, N) and m[key].dtype == torch.float32, key
    for key in ("final_rank_norm", "top5", "top10", "team_death_s", "team_ai_frac", "team_has_monster_dmg"):
        assert m[key].shape == (N,) and m[key].dtype == torch.float32, key
    assert m["team_won"].shape == (N,) and m["team_won"].dtype == torch.bool
    assert m["team_pos"].shape == (P, N, 2) and m["zone"].shape == (P, 8)
    assert m["zone_valid"].shape == (P,) and m["zone_valid"].dtype == torch.bool
    for key in ("phase_start_s", "phase_end_s", "cut_t", "cut_wall_s"):
        assert m[key].shape == (P,) and m[key].dtype == torch.float32, key
    # phase p lives at index p - 1; absent phases are zero / False
    assert m["X"][0, 0, 0] == 1 and m["X"][2, 0, 0] == 3
    assert not m["X"][3:].any() and not m["node_mask"][3:].any()
    assert m["phase_mask"].tolist() == [True, True, True] + [False] * 5
    assert torch.isfinite(m["X"]).all()
    assert m["match_type"] == "official" and m["clock_delta_s"] == DELTA and m["last_phase_idx"] == 3
    assert m["clock_delta_median_s"] == DELTA


def test_rank_direction_and_topk():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    assert m["final_rank_norm"].tolist() == pytest.approx([1 / 3, 1.0, 2 / 3])  # rank 1 -> smallest
    assert m["top5"].tolist() == [1.0, 0.0, 1.0] and m["top10"].tolist() == [1.0, 1.0, 1.0]


def test_pipeline_labels_are_untouched_and_node_mask_is_raised_over_targets():
    raw = _raw()
    m = D.assemble_match(raw, map_size_m=MAP)
    assert not (m["target_mask"] & ~m["node_mask"]).any()
    assert bool(m["node_mask"][2, 2]), "alive-at-end team 15 must be raised to alive-at-start"
    assert not bool(m["node_mask_label"][2, 2]) and (m["node_mask"] & ~m["node_mask_label"]).sum().item() == 1
    assert m["y_hazard"][0].tolist() == [0.0, 1.0, 0.0]
    assert (m["y_hazard"][~m["target_mask"]] == 0).all()
    assert not m["target_mask"][7].any(), "phase 8 is never a hazard target"
    assert m["target_mask"][0].tolist() == [True, True, True]
    assert torch.equal(m["y_hazard"][:7], torch.from_numpy(raw.y_hazard_loader[:, :3]) * m["target_mask"][:7])
    assert m["_stats"] == {"node_mask_raised": 1, "n_targets": 6}


def test_labels_v2_and_death_time_in_the_dump():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    assert m["team_death_s"][0] == pytest.approx(350.0) and m["team_death_s"][1] == pytest.approx(250.0)
    assert m["team_death_s"][2].item() == INF
    assert m["target_mask_v2"][0].tolist() == [True, True, True]
    assert m["target_mask_v2"][1].tolist() == [True, False, True]
    assert m["y_hazard_v2"][0].tolist() == [0.0, 1.0, 0.0] and m["y_hazard_v2"][1].tolist() == [1.0, 0.0, 0.0]
    assert m["node_mask_v2"][2].tolist() == [True, False, True]
    assert (m["y_hazard_v2"][~m["target_mask_v2"]] == 0).all()
    assert not m["target_mask_v2"][2:].any()


def test_labels_v2_follow_the_event_clock_phase_ends_not_the_game_clock():
    """A death between phase_end_s - delta and phase_end_s is still inside the phase on the event clock."""
    raw = _raw(
        members=[(11, "account.a", "byplayer", 192.0, 193.0), (12, "account.b", "alive", None, None),
                 (15, "account.c", "alive", None, None)],
        clock_delta_s=15.0,
    )
    m = D.assemble_match(raw, map_size_m=MAP)
    # team 11 dies at W = 193 > phase_end_s[1] = 190: it is in the risk set of cut 1 and dies in phase 2
    assert bool(m["target_mask_v2"][0, 0]) and m["y_hazard_v2"][0, 0] == 1.0
    assert m["phase_end_s"][0] == 190.0, "phase_end_s stays the stored event-clock number"


def test_event_based_alive_counts_replace_exactly_the_four_columns():
    raw = _raw()
    m = D.assemble_match(raw, map_size_m=MAP)
    X = m["X"].numpy()
    cs, ce, cm, ca = (_col(n) for n in D.REPLACED_FEATURES)
    # phase 1: everybody has both members alive on the whole phase
    assert X[0, :, [cs, ce, cm, ca]].T.tolist() == [[0.5] * 4] * 3
    # phase 2: team 12 loses both members inside the phase (kills at 240 and 250)
    assert X[1, 1, cs] == pytest.approx(0.5) and X[1, 1, ce] == 0.0 and X[1, 1, cm] == 0.0
    assert X[1, 1, ca] == pytest.approx((0.4 + 0.5) / 4)
    assert X[1, 0, [cs, ce, cm, ca]].tolist() == [0.5] * 4 and X[1, 2, [cs, ce, cm, ca]].tolist() == [0.5] * 4
    # phase 3: team 11 dies at 340 / 350 (bot via time_survived + offset), team 12 was dead before it started
    assert X[2, 0, cs] == pytest.approx(0.5) and X[2, 0, ce] == 0.0
    assert X[2, 0, ca] == pytest.approx(((340.0 - 310.0) / 90.0 + (350.0 - 310.0) / 90.0) / 4, abs=1e-4)
    assert X[2, 1, [cs, ce, cm, ca]].tolist() == [0.0] * 4
    assert X[2, 2, [cs, ce, cm, ca]].tolist() == [0.5] * 4
    # every other column is the loader's value
    others = [c for c in range(F) if c not in (cs, ce, cm, ca)]
    assert np.array_equal(X[:3][:, :, others], raw.X_loader[:3, :3][:, :, others])
    # no roster information: the loader values stay (no silent fabrication)
    bare = D.assemble_match(_raw(members=[]), map_size_m=MAP)
    assert torch.equal(bare["X"][:3, :, cs], torch.from_numpy(raw.X_loader[:3, :3, cs]))
    assert torch.isnan(bare["team_death_s"]).all() and not bare["target_mask_v2"].any()


def test_edges_are_raw_sorted_and_filtered():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    assert set(m["edges"]) == {"prox", "combat", "copress"}
    idx, w = m["edges"]["prox"]
    assert idx.dtype == torch.int64 and w.dtype == torch.float32
    assert idx.tolist() == [[0, 0, 1], [0, 1, 0]] and w.tolist() == pytest.approx([0.9, 0.9])
    cidx, cw = m["edges"]["combat"]
    assert cidx.tolist() == [[0, 1, 0]], "combat is attacker(12 -> index 1) -> victim(11 -> index 0)"
    assert cw.item() == pytest.approx(4.2)  # raw weight, no row normalisation
    pidx, _ = m["edges"]["copress"]
    assert pidx.tolist() == [[1, 0, 2], [1, 2, 0]]
    # the self loop, the unknown team and the phase-9 edge were dropped
    assert idx.shape[0] == 2


def test_empty_edge_types_are_well_formed():
    m = D.assemble_match(_raw(edge_rows=[]), map_size_m=MAP)
    for idx, w in m["edges"].values():
        assert idx.shape == (0, 3) and idx.dtype == torch.int64 and w.shape == (0,)


# ------------------------------------------------------------------------------ one clock + cap
def test_cut_time_follows_the_one_clock_window():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    ends, cuts, wall = m["phase_end_s"], m["cut_t"], m["cut_wall_s"]
    assert cuts[:3].tolist() == [170.0, 280.0, 380.0]  # floor((end - 8 - 10) / 10) * 10
    assert wall[:3].tolist() == [178.0, 288.0, 388.0]  # the same instants on the event clock
    assert (wall[:3] <= ends[:3] - causal_cut.GUARD_S).all()
    assert torch.isnan(cuts[3:]).all() and torch.isnan(wall[3:]).all()


def test_raised_clock_delta_moves_the_cut_and_is_reported_next_to_the_median():
    m = D.assemble_match(_raw(clock_delta_s=14.0, clock_delta_median_s=8.0), map_size_m=MAP)
    assert m["clock_delta_s"] == 14.0 and m["clock_delta_median_s"] == 8.0
    assert m["cut_t"][:3].tolist() == [160.0, 270.0, 370.0]  # floor((end - 14 - 10) / 10) * 10
    assert m["cut_wall_s"][:3].tolist() == [174.0, 284.0, 384.0]
    assert (m["cut_wall_s"][:3] <= m["phase_end_s"][:3] - causal_cut.GUARD_S).all()


def test_zero_clock_delta_reproduces_the_plain_cap():
    m = D.assemble_match(_raw(clock_delta_s=0.0), map_size_m=MAP)
    assert m["cut_t"][:3].tolist() == [180.0, 290.0, 390.0]
    assert m["cut_wall_s"][:3].tolist() == [180.0, 290.0, 390.0]


@pytest.mark.parametrize("delta", [0.0, 3.2, 8.0, 17.9, 41.9])
def test_cut_is_causal_on_the_event_clock_for_any_delta(delta):
    raw = _raw(clock_delta_s=delta)
    m = D.assemble_match(raw, map_size_m=MAP)
    for p in range(3):
        cut, wall = float(m["cut_t"][p]), float(m["cut_wall_s"][p])
        assert wall == pytest.approx(cut + delta, abs=1e-3)
        assert wall <= float(m["phase_end_s"][p]) - causal_cut.GUARD_S + 1e-3, "event-clock time of the cut"
        assert cut >= float(m["phase_start_s"][p]) - delta - 1e-6, "inside the shifted window"
        # a resampled grid point reads raw data up to 5 s later: still before the phase end on the event clock
        assert wall + 5.0 < float(m["phase_end_s"][p]) + 1e-3 or wall + 5.0 <= float(m["phase_end_s"][p])


def test_team_pos_uses_the_shifted_window_and_the_cut_snapshot():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    pos = m["team_pos"]
    assert pos[0, 0].tolist() == pytest.approx([2000.0 / MAP, 3000.0 / MAP])  # row at 170, not the one at 180
    assert pos[0, 1].tolist() == pytest.approx([800.0 / MAP, 900.0 / MAP])  # no row at 170 -> falls back to 160
    assert torch.isnan(pos[0, 2]).all(), "team 15's only phase-1 candidate (t = 81) lies before the shifted window start"
    assert pos[1, 0].tolist() == pytest.approx([3000.0 / MAP, 3100.0 / MAP])  # row at 280, not 290
    assert pos[1, 2].tolist() == pytest.approx([10.0 / MAP, 20.0 / MAP])
    assert torch.isnan(pos[1, 1]).all() and torch.isnan(pos[2, 0]).all()
    assert not torch.isclose(pos[0, 0, 0], torch.tensor(7777.0 / MAP))


def test_zone_row_layout_normalisation_and_validity():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    z = m["zone"]
    cols = {name: i for i, name in enumerate(D.ZONE_COLUMNS)}
    row = z[0]
    assert row[cols["cur_cx"]] == pytest.approx(4000.0 / MAP)
    assert row[cols["cur_cy"]] == pytest.approx(4100.0 / MAP)
    assert row[cols["cur_r"]] == pytest.approx(2000.0 / MAP)
    assert row[cols["next_r"]] == pytest.approx(1000.0 / MAP)
    assert row[cols["phase_frac"]] == pytest.approx((178.0 - 90.0) / 100.0)  # event-clock time of the cut
    assert row[cols["phase_no"]] == 1.0
    assert m["zone_valid"].tolist()[:3] == [True, False, True]  # poison missing at 280
    assert torch.isnan(z[1, cols["next_r"]]) and z[1, cols["cur_r"]] == pytest.approx(1000.0 / MAP)
    assert z[2, cols["phase_no"]] == 3.0 and z[2, cols["phase_frac"]] == pytest.approx((388.0 - 310.0) / 90.0)
    assert torch.isnan(z[5]).all()  # absent phase
    # nothing from the forbidden grid point (180 / 9999 m) reached the zone tensor
    geometry = z[:, :6]
    assert (geometry[~torch.isnan(geometry)].abs() < 1.0).all()  # 9999 m / 8160 m would be 1.22


def test_dump_is_blind_to_grid_rows_after_the_cut():
    """Rewriting every grid row after the one-clock cut changes nothing in the dump."""
    base = D.assemble_match(_raw(), map_size_m=MAP)
    raw = _raw()
    for (kind, t), _val in list(raw.zone_rows.items()):
        if t == 180:  # phase-1 grid point after the cut 170
            raw.zone_rows[(kind, t)] = (123.0, 456.0, 789.0)
    for tid, rows in raw.pos_rows.items():
        raw.pos_rows[tid] = [(t, 55.0, 66.0) if t in (180.0, 290.0) else (t, x, y) for t, x, y in rows]
    alt = D.assemble_match(raw, map_size_m=MAP)
    for key in ("team_pos", "zone", "zone_valid", "cut_t", "cut_wall_s", "X", "node_mask", "target_mask"):
        assert torch.equal(torch.nan_to_num(base[key].float(), nan=-1.0),
                           torch.nan_to_num(alt[key].float(), nan=-1.0)), key


def test_ignoring_the_clock_shift_would_read_the_forbidden_row():
    """Negative control: with delta = 0 the window ends at 180 and the rewritten row IS visible."""
    base = D.assemble_match(_raw(clock_delta_s=0.0), map_size_m=MAP)
    raw = _raw(clock_delta_s=0.0)
    raw.zone_rows[("safe", 180)] = (123.0, 456.0, 789.0)
    alt = D.assemble_match(raw, map_size_m=MAP)
    assert not torch.equal(torch.nan_to_num(base["zone"], nan=-1.0), torch.nan_to_num(alt["zone"], nan=-1.0))


def test_too_short_phase_has_no_cut():
    raw = _raw(bounds={1: (90.0, 90.0), 2: (200.0, 300.0), 3: (310.0, 400.0)})
    m = D.assemble_match(raw, map_size_m=MAP)
    assert torch.isnan(m["cut_t"][0]) and not m["zone_valid"][0]
    assert torch.isnan(m["team_pos"][0]).all()


def test_non_finite_features_are_rejected():
    raw = _raw()
    raw.X_loader[0, 0, 5] = float("nan")
    with pytest.raises(ValueError):
        D.assemble_match(raw, map_size_m=MAP)


def test_non_contiguous_phases_follow_the_phase_number_not_the_position():
    raw = _raw(phase_indices=[1, 3, 4], bounds={1: (90.0, 190.0), 3: (310.0, 400.0), 4: (410.0, 500.0)}, last_phase_idx=4)
    m = D.assemble_match(raw, map_size_m=MAP)
    assert m["phase_mask"].tolist() == [True, False, True, True] + [False] * 4
    assert m["X"][2, 0, 0] == 2 and m["X"][3, 0, 0] == 3  # loader positions 1, 2 -> phases 3, 4
    assert not m["X"][1, :, 0].any()
    assert not m["target_mask_v2"][0].any(), "phase 2 is missing: cut 1 has no next phase"
    # cut 3 -> phase 4 exists (the last one); teams 11 / 12 died before the end of phase 3, the winner is censored
    assert m["target_mask_v2"][2].tolist() == [False, False, True] and m["y_hazard_v2"][2].tolist() == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------------------ audit fields
def test_audit_fields_values_and_dtypes():
    m = D.assemble_match(_raw(), map_size_m=MAP)
    assert m["match_type"] == "official"
    assert m["team_won"].tolist() == [False, False, True]
    assert m["team_ai_frac"].tolist() == pytest.approx([0.5, 0.0, 0.5])
    assert m["team_has_monster_dmg"].tolist() == pytest.approx([30.0, 5.0, 0.0])  # damage amount, 0 if none
    no_roster = D.assemble_match(_raw(members=[]), map_size_m=MAP)
    assert torch.isnan(no_roster["team_ai_frac"]).all() and (no_roster["team_has_monster_dmg"] == 0).all()


# ------------------------------------------------------------------------------ population rules
class _TypeCursor:
    """Fake cursor answering the one query compute_population issues."""

    def __init__(self, types):
        self.types, self._rows = types, []

    def execute(self, sql, params=None):
        assert "pubg.matches" in sql and "match_type" in sql
        self._rows = [(m, self.types[m]) for m in params[0] if m in self.types]

    def fetchall(self):
        return self._rows


def _write_split(root: Path, spec, n_train, n_val, n_test, prefix):
    ids = {
        "train": [f"{prefix}-tr-{i:04d}" for i in range(n_train)],
        "val": [f"{prefix}-va-{i:04d}" for i in range(n_val)],
        "test": [f"{prefix}-te-{i:04d}" for i in range(n_test)],
    }
    path = root / spec.slug / "split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"splits": ids}), encoding="utf-8")
    return ids


def test_population_rules_exclude_airoyale_vikendi_and_small_cohorts(tmp_path):
    by_slug = {s.slug: s for s in D.REQUIRED_COHORTS}
    erangel, vik, kara = by_slug["erangel_tpp_squad"], by_slug["vikendi_tpp_squad"], by_slug["karakin_tpp_squad"]
    types = {}
    ids_e = _write_split(tmp_path, erangel, 168, 36, 36, "e")
    ids_v = _write_split(tmp_path, vik, 700, 150, 150, "v")
    ids_k = _write_split(tmp_path, kara, 140, 40, 40, "k")  # 220 matches
    for i, mid in enumerate(m for s in ids_e.values() for m in s):
        types[mid] = "airoyale" if i % 8 == 0 else ("official" if i % 2 else "competitive")  # 30 airoyale
    for mid in (m for s in ids_v.values() for m in s):
        types[mid] = "official"
    for i, mid in enumerate(m for s in ids_k.values() for m in s):
        types[mid] = "airoyale" if i % 7 == 0 else "official"  # 32 airoyale -> 188 left (< 200)
    population, kept = D.compute_population(_TypeCursor(types), [erangel, vik, kara], tmp_path)

    e = population["cohorts"]["erangel_tpp_squad"]
    assert e["before"] == {"train": 168, "val": 36, "test": 36, "total": 240}
    assert e["after"]["total"] == 210 and e["status"] == "kept" and e["excluded_by_match_type"] == {"airoyale": 30}
    assert sum(e["after"][s] for s in D.SPLITS) == 210
    assert "airoyale" not in e["match_type_after"] and e["match_type_before"]["airoyale"] == 30
    all_kept = [m for s in D.SPLITS for m in kept["erangel_tpp_squad"][s]]
    assert len(all_kept) == 210 and not any(types[m] == "airoyale" for m in all_kept)
    # the stored assignment is kept: train ids stay train, order preserved
    assert kept["erangel_tpp_squad"]["train"] == [m for m in ids_e["train"] if types[m] != "airoyale"]
    assert set(kept["erangel_tpp_squad"]["val"]) <= set(ids_e["val"])

    v = population["cohorts"]["vikendi_tpp_squad"]
    assert v["status"] == "excluded" and v["before"]["total"] == 1000 and v["after"]["total"] == 0
    assert all(not kept["vikendi_tpp_squad"][s] for s in D.SPLITS)

    k = population["cohorts"]["karakin_tpp_squad"]
    assert k["status"] == "dropped" and k["after"]["total"] == 188 < D.MIN_COHORT_MATCHES
    assert population["rules"]["exclude_match_types"] == ["airoyale"]
    assert population["rules"]["exclude_cohorts"] == ["vikendi_tpp_squad"]
    assert population["rules"]["min_matches_per_cohort"] == 200


# ------------------------------------------------------------------------------ meta + schema contract
def test_build_meta_documents_one_clock_replaced_features_and_population():
    spec = {s.slug: s for s in D.REQUIRED_COHORTS}["erangel_tpp_squad"]
    meta = D.build_meta(spec, "train", split_path=Path("x/split.json"), split_ids=["a", "b"], extra={"population": {"k": 1}})
    assert meta["fixb"] is True and meta["single_clock"] is True and meta["fixb_cut_guard_s"] == 10
    assert meta["replaced_features"] == list(D.REPLACED_FEATURES)
    assert meta["feature_names"] == D.FEATURE_NAMES and len(meta["feature_names"]) == 150
    assert {"zone", "habitat"} <= set(meta["feature_groups"])
    assert set(meta["clock"]) >= {"W", "G", "why", "feature_windows", "cut_t", "team_death_s"}
    assert "phase_end_s" in meta["clock"]["W"] and "delta" in meta["clock"]["G"]
    assert "clock_delta_median_s" in meta["clock"]["G"] and "raised" in meta["clock"]["G"]
    assert meta["population"] == {"k": 1} and meta["zone_columns"] == list(D.ZONE_COLUMNS)
    assert {"labels_v2", "alive_counts", "target_mask / y_hazard"} <= set(meta["conventions"])


@pytest.mark.parametrize("labels", ["v2", "pipeline"])
def test_dump_passes_the_ptr_validator_and_roundtrips_with_safe_load(tmp_path, labels):
    ptr_data = pytest.importorskip('pubg_lab.survival.ptr.data.data')
    matches = []
    for k in range(3):
        m = D.assemble_match(_raw(match_id=f"m-{k}"), map_size_m=MAP)
        m.pop("_stats")
        matches.append(m)
    spec = {s.slug: s for s in D.REQUIRED_COHORTS}["erangel_tpp_squad"]
    meta = D.build_meta(spec, "train", split_path=Path("x/split.json"), split_ids=["m-0", "m-1", "m-2"], extra={})
    dump = {"meta": meta, "matches": matches}
    ptr_data.validate_dump(dump, labels)
    path = tmp_path / "train.pt"
    torch.save(dump, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)  # no pickle fallback needed
    ptr_data.validate_dump(loaded, labels)
    keep = ptr_data.keep_columns(loaded["meta"], ("zone", "habitat"), keep_names=ptr_data.ALIVE_COUNT_FEATURES)
    assert len(keep) == D.N_FEATURES - 44
    assert loaded["meta"]["replaced_features"] == list(D.REPLACED_FEATURES)
    assert loaded["matches"][0]["team_death_s"][2].item() == INF


def test_required_since_demands_a_rebuild_after_the_drift_rule_for_raised_matches():
    since, bump = "2026-10-05T10:19:57+00:00", "2026-10-05T12:30:00+00:00"
    assert D.required_since(None, bump, True) is None
    assert D.required_since(since, bump, False) == since, "an unraised match only needs the one-clock rebuild"
    assert D.required_since(since, bump, True) == datetime.fromisoformat(bump).isoformat()
    assert D.required_since(bump, since, True) == datetime.fromisoformat(bump).isoformat(), "the later marker wins"
    assert D.required_since(since, None, True).startswith("9999-12-31"), "no marker: a raised match is always stale"
    assert D.required_since(since, "", False) == since


def test_stale_check_flags_missing_and_old_rows():
    class Cur:
        def __init__(self, answers):
            self.answers, self.table = answers, None

        def execute(self, sql, params=None):
            self.table = sql.split("stdb.")[1].split()[0]

        def fetchone(self):
            return (self.answers[self.table],)

    fresh = {t: False for t in D.FIXB_CREATED_AT_TABLES}
    assert D.stale_fixb_tables(Cur(fresh), "m", "2026-10-05T00:00:00+00:00") == []
    old = dict(fresh, team_phase_state=True, team_phase_habitat_features=None)
    assert D.stale_fixb_tables(Cur(old), "m", "2026-10-05T00:00:00+00:00") == ["team_phase_state", "team_phase_habitat_features"]
    assert D.stale_fixb_tables(Cur(dict(fresh, team_phase_z_affordance_features=None)), "m", "x") == [], "z rows may be absent"


# ------------------------------------------------------------------------------ DB-gated
@requires_db
def test_population_on_the_real_split_manifests():
    from pubg_lab.data.stdb.connection.conn import connect

    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            population, kept = D.compute_population(cur, D.REQUIRED_COHORTS)
            all_kept = [m for slug in kept for s in D.SPLITS for m in kept[slug][s]]
            cur.execute("SELECT count(*) FROM pubg.matches WHERE match_id = ANY(%s) AND match_type = 'airoyale'", (all_kept,))
            assert cur.fetchone()[0] == 0, "an airoyale match survived the exclusion"
    finally:
        conn.close()
    cohorts = population["cohorts"]
    assert cohorts["vikendi_tpp_squad"]["status"] == "excluded"
    for slug, rec in cohorts.items():
        assert rec["after"]["total"] <= rec["before"]["total"]
        assert (rec["status"] == "kept") == (rec["after"]["total"] >= D.MIN_COHORT_MATCHES and slug not in D.EXCLUDED_COHORTS)
        if rec["status"] == "kept":
            assert all(rec["after"][s] > 0 for s in D.SPLITS), slug


@requires_db
def test_one_real_match_passes_the_independent_verifier():
    """Build the dump entry of one Erangel match and re-derive every number with independent SQL."""
    from pubg_lab.data.stdb.connection.conn import connect
    from projects.battle_royale.evaluation.contracts.verify_dump_match import verify

    marker = Path("model_stgcn/etl_runs/fixb_2026-10-05/fixb_since.txt")
    since = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    bump_marker = Path("model_stgcn/etl_runs/fixb_2026-10-05/fixb_bump_since.txt")
    bump = bump_marker.read_text(encoding="utf-8").strip() if bump_marker.exists() else None
    split = json.loads(Path("model_stgcn/checkpoints/br_om_ladder_required_v1/erangel_tpp_squad/split.json").read_text())
    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            for mid in split["splits"]["test"][:40]:
                cur.execute("SELECT match_type FROM pubg.matches WHERE match_id = %s", (mid,))
                row = cur.fetchone()
                if row is None or row[0] in D.EXCLUDED_MATCH_TYPES:
                    continue
                if since:
                    raised = D.causal_cut.fetch_clock_delta_detail(cur, mid).raised
                    if D.stale_fixb_tables(cur, mid, D.required_since(since, bump, raised)):
                        pytest.skip("the one-clock ETL has not rebuilt this match yet (run tools/run_fixb_phase_etl.py)")
                raw = D.fetch_raw_match(cur, mid, "Baltic_Main")
                if raw is None:
                    continue
                m = D.assemble_match(raw, map_size_m=D.MAP_SIZE_M["Baltic_Main"])
                m.pop("_stats")
                results = verify(cur, m, "Baltic_Main", since)
                failed = [(n, d) for n, ok, d in results if not ok]
                assert not failed, f"{mid}: {failed}"
                return
    finally:
        conn.close()
    pytest.skip("no phase-ready Erangel TPP squad test match found")
