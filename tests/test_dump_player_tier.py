"""Tests for model_stgcn/dump_player_tier.py (stage 2: player-level dump).

DB-free part: a hand-built match (6 players, 3 teams, 2 phases) exercises every convention of the module with
numbers that can be checked by hand.  DB-gated part (``requires_db``): one real Erangel match goes through the
raw fetch, the assembly and the independent verifier of tools/verify_player_dump_match.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from model_stgcn import dump_phase_tier as D
from model_stgcn import dump_player_tier as PT
from tests.conftest import requires_db

SIZE = 8000.0
DELTA = 8.0
NAN = float("nan")
FEATURE = {name: i for i, name in enumerate(PT.PLAYER_FEATURE_NAMES)}


def _info() -> PT.PhaseInfo:
    """Teams 11 / 12 / 15; phase 1: [100, 200] cut 170 (W 178), phase 2: [210, 310] cut 270 (W 278); no other phase."""
    phase_mask = np.array([True, True] + [False] * 6)
    start = np.array([100.0, 210.0] + [NAN] * 6)
    end = np.array([200.0, 310.0] + [NAN] * 6)
    cut = np.array([170.0, 270.0] + [NAN] * 6)
    zone = np.full((8, 8), NAN)
    zone[0] = [0.5, 0.5, 0.25, 0.5, 0.5, 0.125, 0.9, 1.0]  # centre (4000, 4000) m, r = 2000 m; next r = 1000 m
    zone[1] = [0.5, 0.5, 0.125, NAN, NAN, NAN, 0.9, 2.0]  # no announced next circle
    return PT.PhaseInfo(
        match_id="p-test", team_ids=[11, 12, 15], phase_mask=phase_mask, phase_start_s=start, phase_end_s=end,
        cut_t=cut, delta_s=DELTA, zone=zone, map_size=SIZE,
    )  # fmt: skip


MEMBERS = [
    (11, "a1", "alive", 1500.0, None),          # never dies
    (11, "a2", "byplayer", 184.3, 185.0),       # killed at 185 (after the phase-1 cut at W = 178)
    (11, "a3", "byplayer", 194.3, 195.0),       # zone death at 105, revived, killed again at 195
    (12, "b1", "byplayer", 119.3, 120.0),       # killed at 120
    (15, "c1", "alive", 1500.0, None),
    (15, "ai.c2", "byplayer", 250.0, None),     # bot without a kill event: time_survived + offset
]  # fmt: skip
ORDER = ["a1", "a2", "a3", "b1", "ai.c2", "c1"]  # sorted by (team index, id)


def _pos(rows):
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]  # G, health, x, y, W


def _raw(**over) -> PT.RawPlayerMatch:
    positions = {
        "a1": _pos([(160, 100, 4000, 4000, 168), (170, 90, 4000, 4000, 178), (172, 80, 5000, 5000, 180),
                    (260, 70, 4100, 4000, 268), (270, 60, 4000, 4000, 278)]),
        "a2": _pos([(150, 100, 4300, 4000, 158)]),                           # 20 s stale at the cut -> invalid
        "a3": _pos([(160, 80, 4100, 4000, 168), (170, 75, 4100, 4000, 178)]),  # revived: fresh, health > 0
        "b1": _pos([(160, 0, 4200, 4100, 168), (165, 0, 4200, 4100, 173)]),    # corpse: fresh but health 0
        "ai.c2": _pos([(168, 100, 4250, 4000, 176)]),
        "c1": _pos([(170, 100, 4000, 4400, 178), (265, 100, 4000, 4200, 273)]),
    }  # fmt: skip
    kills = [  # (killer, victim, W)
        (None, "a3", 105.0), ("c1", "b1", 120.0), ("c1", "a2", 185.0), ("c1", "a3", 195.0),
    ]  # fmt: skip
    knocks = [("c1", "b1", 118.0), ("a1", "c1", 140.0), ("a1", "a2", 150.0), ("a1", "c1", 185.0), ("c1", "a1", 230.0)]
    damage = [  # (attacker, victim, damage, W)
        ("c1", "b1", 30.0, 119.0), ("c1", "b1", 20.0, 119.5), ("b1", "c1", 10.0, 110.0),
        ("a1", "a2", 15.0, 130.0),       # friendly fire: not dealt, but taken by a2
        (None, "a1", 5.0, 140.0),        # blue zone: taken
        ("a1", "a1", 7.0, 141.0),        # self-inflicted: ignored
        ("Monster.1", "a1", 12.0, 150.0),  # non-participant attacker: taken, never dealt
        ("a1", "c1", 50.0, 180.0),       # after the cut (178): ignored
        ("a1", "c1", 0.0, 150.0),        # zero damage: ignored
        ("a1", "ai.c2", 25.0, 100.0),    # exactly at the window start: counts
        ("c1", "a1", 40.0, 220.0),       # phase 2
    ]  # fmt: skip
    items = [("a1", "Heal", 110.0), ("a1", "Boost", 120.0), ("a1", "Heal", 179.0), ("c1", "Heal", 90.0)]
    base = dict(info=_info(), members=list(MEMBERS), positions=positions, kills=kills, knocks=knocks, damage=damage, items=items)
    base.update(over)
    return PT.RawPlayerMatch(**base)


def _row(pm, phase, name):
    return pm["Xp"][phase, ORDER.index(name)]


# ------------------------------------------------------------------------------ layout
def test_feature_layout_and_groups():
    assert PT.N_PLAYER_FEATURES == 13 == len(PT.PLAYER_FEATURE_NAMES) == len(set(PT.PLAYER_FEATURE_NAMES))
    cols = sorted(c for g in PT.PLAYER_FEATURE_GROUPS.values() for c in g)
    assert cols == list(range(13))
    assert set(PT.PLAYER_FEATURE_SCALES) == set(PT.PLAYER_FEATURE_NAMES)
    assert PT.EDGE_TYPES == ("teammate", "enemy_prox", "combat")
    assert PT.MAX_WORKERS == 2


def test_phase_info_from_a_phase_record_roundtrip():
    rec = {
        "match_id": "m", "team_ids": torch.tensor([3, 7]), "phase_mask": torch.tensor([True] * 3 + [False] * 5),
        "phase_start_s": torch.tensor([90.0] * 8), "phase_end_s": torch.tensor([190.0] * 8),
        "cut_t": torch.tensor([170.0, NAN] + [NAN] * 6), "clock_delta_s": 12.5, "zone": torch.zeros(8, 8), "map_size": 8160.0,
    }  # fmt: skip
    info = PT.phase_info_from_record(rec)
    assert info.team_ids == [3, 7] and info.delta_s == 12.5 and info.map_size == 8160.0
    assert info.phase_mask.dtype == bool and info.cut_t.dtype == np.float64 and math.isnan(info.cut_t[1])


# ------------------------------------------------------------------------------ players, masks, death times
def test_players_order_teams_ai_and_exclusions():
    pm = PT.assemble_player_match(_raw())
    assert pm["player_ids"] == ORDER
    assert pm["player_team_idx"].tolist() == [0, 0, 0, 1, 2, 2] and pm["player_team_idx"].dtype == torch.int64
    assert pm["player_is_ai"].tolist() == [False, False, False, False, True, False]
    assert "Monster.1" not in pm["player_ids"]
    assert pm["team_ids"].tolist() == [11, 12, 15]
    # a participant whose roster team is not in the phase dump is dropped and counted
    odd = PT.assemble_player_match(_raw(members=MEMBERS + [(99, "zz", "alive", 1.0, None)]))
    assert "zz" not in odd["player_ids"] and odd["_stats"]["n_players_dropped"] == 1


def test_death_times_node_mask_and_first_death():
    pm = PT.assemble_player_match(_raw())
    death = pm["player_death_s"].tolist()
    assert death[0] == math.inf and death[5] == math.inf
    assert death[1] == 185.0 and death[2] == 195.0 and death[3] == 120.0
    assert death[4] == pytest.approx(250.7)  # bot: time_survived 250 + the match offset 0.7 s (median over the humans)
    first = pm["player_first_death_s"].tolist()
    assert first[2] == 105.0 and first[1] == 185.0 and first[4] == pytest.approx(250.7) and first[0] == math.inf
    node = pm["player_node_mask"]
    assert node.shape == (8, 6) and node.dtype == torch.bool
    assert node[0].tolist() == [True, True, True, True, True, True]  # death > phase_start_s = 100
    assert node[1].tolist() == [True, False, False, False, True, True]  # death > 210
    assert not node[2:].any(), "absent phases carry no tokens"


# ------------------------------------------------------------------------------ features
def test_phase_one_features_by_hand():
    pm = PT.assemble_player_match(_raw())
    f = FEATURE
    a1 = _row(pm, 0, "a1")
    assert a1[f["rel_x_hm"]] == pytest.approx(-0.5) and a1[f["rel_y_hm"]] == pytest.approx(0.0)  # centroid of a1, a3
    assert a1[f["pos_x_norm"]] == pytest.approx(0.5) and a1[f["pos_y_norm"]] == pytest.approx(0.5)
    assert a1[f["health_norm"]] == pytest.approx(0.9), "the row at G = 172 lies after the cut and must not be read"
    assert a1[f["dmg_dealt_hp100"]] == pytest.approx(0.25), "25 HP to the bot at the window start; friendly / self / post-cut excluded"
    assert a1[f["dmg_taken_hp100"]] == pytest.approx(0.17), "5 zone + 12 from a non-participant; the self hit is excluded"
    assert a1[f["knocks_dealt"]] == 1 and a1[f["kills_phase"]] == 0 and a1[f["kills_cum"]] == 0
    assert a1[f["heals_cum"]] == 1, "Heal at 110 only: the Boost at 120 is in cum_use_boost, the Heal at 179 is after the cut"
    assert pm["cum_use_boost"][0, ORDER.index("a1")] == 1 and pm["cum_use_boost"][1, ORDER.index("a1")] == 1
    assert a1[f["zone_cur_signed_dist_r"]] == pytest.approx(1.0) and a1[f["zone_next_signed_dist_r"]] == pytest.approx(1.0)

    a3 = _row(pm, 0, "a3")
    assert a3[f["rel_x_hm"]] == pytest.approx(0.5) and a3[f["health_norm"]] == pytest.approx(0.75)
    assert a3[f["zone_cur_signed_dist_r"]] == pytest.approx(0.95) and a3[f["zone_next_signed_dist_r"]] == pytest.approx(0.9)

    a2 = _row(pm, 0, "a2")  # stale sample: positional columns are 0, event columns still count
    assert [float(a2[f[n]]) for n in ("rel_x_hm", "pos_x_norm", "health_norm", "zone_cur_signed_dist_r")] == [0.0] * 4
    assert a2[f["dmg_taken_hp100"]] == pytest.approx(0.15) and a2[f["dmg_dealt_hp100"]] == 0.0

    b1 = _row(pm, 0, "b1")  # corpse: fresh sample with health 0; alone in its team -> centroid = himself
    assert b1[f["health_norm"]] == 0.0 and b1[f["rel_x_hm"]] == 0.0 and b1[f["rel_y_hm"]] == 0.0
    assert b1[f["dmg_dealt_hp100"]] == pytest.approx(0.1) and b1[f["dmg_taken_hp100"]] == pytest.approx(0.5)
    d_b1 = math.hypot(200.0, 100.0)
    assert b1[f["zone_cur_signed_dist_r"]] == pytest.approx((2000.0 - d_b1) / 2000.0, abs=1e-6)
    assert b1[f["zone_next_signed_dist_r"]] == pytest.approx((1000.0 - d_b1) / 1000.0, abs=1e-6)

    c1 = _row(pm, 0, "c1")
    assert c1[f["rel_x_hm"]] == pytest.approx(-1.25) and c1[f["rel_y_hm"]] == pytest.approx(2.0)  # centroid (4125, 4200)
    assert c1[f["dmg_dealt_hp100"]] == pytest.approx(0.5) and c1[f["dmg_taken_hp100"]] == pytest.approx(0.1)
    assert c1[f["knocks_dealt"]] == 1 and c1[f["kills_phase"]] == 1 and c1[f["kills_cum"]] == 1
    assert c1[f["heals_cum"]] == 1, "cumulative counts include events before the phase start (Heal at 90)"
    assert c1[f["zone_cur_signed_dist_r"]] == pytest.approx(0.8) and c1[f["zone_next_signed_dist_r"]] == pytest.approx(0.6)

    c2 = _row(pm, 0, "ai.c2")
    assert c2[f["rel_x_hm"]] == pytest.approx(1.25) and c2[f["rel_y_hm"]] == pytest.approx(-2.0)
    assert c2[f["dmg_taken_hp100"]] == pytest.approx(0.25)


def test_phase_two_window_cumulative_counts_and_missing_zone_circle():
    pm = PT.assemble_player_match(_raw())
    f = FEATURE
    a1, c1 = _row(pm, 1, "a1"), _row(pm, 1, "c1")
    assert a1[f["dmg_taken_hp100"]] == pytest.approx(0.4) and a1[f["dmg_dealt_hp100"]] == 0.0, "only events in [210, 278]"
    assert a1[f["heals_cum"]] == 2, "cumulative: the Heal at 179 now counts (110 and 179)"
    assert c1[f["dmg_dealt_hp100"]] == pytest.approx(0.4) and c1[f["knocks_dealt"]] == 1
    assert c1[f["kills_phase"]] == 0 and c1[f["kills_cum"]] == 3, "b1 at 120, a3 at 195, a2 at 185 (none inside [210, 278])"
    # phase 2 has no announced next circle: its feature is 0 and the flag says so; the current circle exists
    assert bool(pm["zone_cur_valid"][1]) and not bool(pm["zone_next_valid"][1]) and bool(pm["zone_next_valid"][0])
    assert c1[f["zone_next_signed_dist_r"]] == 0.0
    d = math.hypot(0.0, 200.0)
    assert c1[f["zone_cur_signed_dist_r"]] == pytest.approx((1000.0 - d) / 1000.0, abs=1e-6)  # r = 0.125 * 8000
    assert _row(pm, 1, "ai.c2").abs().sum() == pytest.approx(float(_row(pm, 1, "ai.c2")[f["dmg_taken_hp100"]]))  # no sample


def test_zone_signed_distance_is_clipped():
    info = _info()
    info.zone[0, 5] = 0.0005  # a 4 m next circle: a player 5.6 km away would be at -1400 radii
    raw = _raw(info=info)
    raw.positions["a1"] = _pos([(170, 90, 8000, 8000, 178)])
    pm = PT.assemble_player_match(raw)
    row = _row(pm, 0, "a1")
    assert row[FEATURE["zone_next_signed_dist_r"]] == PT.ZONE_CLIP[0]
    assert row[FEATURE["zone_cur_signed_dist_r"]] == pytest.approx((2000.0 - math.hypot(4000.0, 4000.0)) / 2000.0, abs=1e-6)


# ------------------------------------------------------------------------------ alive at the cut, validity, coverage
def test_alive_at_the_cut_is_causal_and_handles_revived_players():
    pm = PT.assemble_player_match(_raw())
    alive, strict, valid = pm["player_alive_cut"][0], pm["player_alive_cut_strict"][0], pm["player_pos_valid"][0]
    assert strict.tolist() == [True, True, False, False, True, True], "no kill event up to W = 178 (a3 died at 105, b1 at 120)"
    assert alive.tolist() == [True, True, True, False, True, True], "a3 is revived (fresh health > 0 after the death); b1 is a corpse"
    assert valid.tolist() == [True, False, True, True, True, True]
    # a death AFTER the cut (a2 at 185) never makes a2 dead at the cut
    assert bool(alive[1]) and bool(strict[1])
    # phase 2 (W = 278): a2 / a3 / b1 are dead, the bot's fallback death (250.7) has passed, only a1 and c1 live
    assert pm["player_alive_cut"][1].tolist() == [True, False, False, False, False, True]
    assert pm["player_alive_cut"].dtype == torch.bool and not pm["player_alive_cut"][2:].any()


def test_coverage_counts_strict_and_last_death():
    pm = PT.assemble_player_match(_raw())
    st = pm["_stats"]
    assert st["coverage_alive"][:2] == [4, 2] and st["coverage_valid"][:2] == [3, 2]  # phase 1: a2 has no fresh sample
    assert st["coverage_alive_last"][0] == 5 and st["coverage_valid_last"][0] == 4  # a3 (last death 195) counts, a2 missing
    summary = PT.coverage_summary([st, st])
    assert summary["strict"]["alive_at_cut"][0] == 8 and summary["strict"]["with_sample"][0] == 6
    assert summary["strict"]["share_by_phase"][0] == pytest.approx(0.75)
    assert summary["last_death"]["share_by_phase"][0] == pytest.approx(0.8)
    assert summary["share_phases_1_7"] == summary["strict"]["share_phases_1_7"]
    assert summary["strict"]["share_by_phase"][5] is None


# ------------------------------------------------------------------------------ edges
def test_edges_by_hand():
    pm = PT.assemble_player_match(_raw())
    E = pm["edges_player"]
    ix = {n: i for i, n in enumerate(ORDER)}

    def edge_set(kind, phase):
        idx, w = E[kind]
        return {(int(i), int(j)): float(x) for (p, i, j), x in zip(idx.tolist(), w.tolist()) if p == phase}

    tm = edge_set("teammate", 0)
    assert set(tm) == {(ix["a1"], ix["a3"]), (ix["a3"], ix["a1"]), (ix["ai.c2"], ix["c1"]), (ix["c1"], ix["ai.c2"])}
    assert tm[(ix["a1"], ix["a3"])] == pytest.approx(math.exp(-100.0 / 200.0))
    assert tm[(ix["c1"], ix["ai.c2"])] == pytest.approx(math.exp(-math.hypot(250.0, 400.0) / 200.0))  # no distance cap

    prox = edge_set("enemy_prox", 0)
    assert set(prox) == {(ix["a1"], ix["ai.c2"]), (ix["ai.c2"], ix["a1"]), (ix["a3"], ix["ai.c2"]), (ix["ai.c2"], ix["a3"])}
    assert prox[(ix["a1"], ix["ai.c2"])] == pytest.approx(math.exp(-250.0 / 200.0))
    # c1 is 400 m from a1, the corpse b1 and the stale a2 take part in nothing
    assert all(ix["b1"] not in pair and ix["a2"] not in pair for pair in set(tm) | set(prox))
    prox2 = edge_set("enemy_prox", 1)
    assert set(prox2) == {(ix["a1"], ix["c1"]), (ix["c1"], ix["a1"])}
    assert prox2[(ix["a1"], ix["c1"])] == pytest.approx(math.exp(-200.0 / 200.0)), "a1 (4000, 4000) vs c1 (4000, 4200) at the phase-2 cut"

    combat = edge_set("combat", 0)
    assert set(combat) == {(ix["c1"], ix["b1"]), (ix["b1"], ix["c1"]), (ix["a1"], ix["c1"]), (ix["a1"], ix["ai.c2"])}
    assert combat[(ix["c1"], ix["b1"])] == pytest.approx(math.log1p(50.0 + 100.0 + 50.0)), "damage + kill + knock"
    assert combat[(ix["b1"], ix["c1"])] == pytest.approx(math.log1p(10.0))
    assert combat[(ix["a1"], ix["c1"])] == pytest.approx(math.log1p(50.0)), "the knock at 140; the hit at 180 is after the cut"
    assert combat[(ix["a1"], ix["ai.c2"])] == pytest.approx(math.log1p(25.0))
    combat2 = edge_set("combat", 1)
    assert set(combat2) == {(ix["c1"], ix["a1"])} and combat2[(ix["c1"], ix["a1"])] == pytest.approx(math.log1p(40.0 + 50.0))
    # teammates never fight: the friendly hit and the friendly knock create no combat edge
    assert (ix["a1"], ix["a2"]) not in combat


def test_edge_tensors_are_sorted_int64_float32_and_symmetric():
    pm = PT.assemble_player_match(_raw())
    for kind, (idx, w) in pm["edges_player"].items():
        assert idx.dtype == torch.int64 and w.dtype == torch.float32 and idx.shape[1] == 3 and w.shape == (idx.shape[0],)
        rows = idx.tolist()
        assert rows == sorted(rows), kind
    PT.validate_player_match({k: v for k, v in pm.items() if k != "_stats"})  # symmetric, in range, finite


# ------------------------------------------------------------------------------ causality
def test_nothing_after_the_cut_changes_the_phase_one_output():
    base = PT.assemble_player_match(_raw())
    raw = _raw()
    # rewrite every event after cut_wall_s = 178 and every position row after cut_t = 170
    raw.kills = [(a, v, w) if w <= 178 else (a, v, 9999.0) for a, v, w in raw.kills]
    raw.knocks = [(a, v, w) if w <= 178 else ("a1", "c1", w + 1.0) for a, v, w in raw.knocks]
    raw.damage = [(a, v, d, w) if w <= 178 else (a, v, d * 7.0, w + 1.0) for a, v, d, w in raw.damage]
    raw.items = [(a, s, w) if w <= 178 else (a, "Heal", w + 1.0) for a, s, w in raw.items]
    for pid, (t, h, x, y, w) in list(raw.positions.items()):
        keep = t <= 170
        raw.positions[pid] = (t[keep], h[keep], x[keep], y[keep], w[keep])
    alt = PT.assemble_player_match(raw)
    # the kills at 185 / 195 move to 9999: the death times change, but phase 1 only reads events up to 178
    for key in ("Xp",):
        assert torch.equal(base[key][0], alt[key][0]), key
    for kind in PT.EDGE_TYPES:
        for src in (base, alt):
            assert src["edges_player"][kind][0][:, 0].numel() >= 0
        b_idx, b_w = base["edges_player"][kind]
        a_idx, a_w = alt["edges_player"][kind]
        sel_b, sel_a = b_idx[:, 0] == 0, a_idx[:, 0] == 0
        assert torch.equal(b_idx[sel_b], a_idx[sel_a]) and torch.equal(b_w[sel_b], a_w[sel_a]), kind
    assert torch.equal(base["player_alive_cut"][0], alt["player_alive_cut"][0])


def test_phase_without_a_cut_is_empty_but_keeps_its_node_mask():
    info = _info()
    info.cut_t[1] = NAN
    pm = PT.assemble_player_match(_raw(info=info))
    assert not pm["Xp"][1].any() and not pm["player_pos_valid"][1].any() and not pm["player_alive_cut"][1].any()
    assert pm["player_node_mask"][1].tolist() == [True, False, False, False, True, True]
    assert (pm["edges_player"]["combat"][0][:, 0] == 0).all()
    assert torch.isnan(pm["cut_wall_s"][1])


# ------------------------------------------------------------------------------ schema / validation against a phase record
def _phase_record(pm_raw: PT.RawPlayerMatch) -> dict:
    info = pm_raw.info
    X = torch.zeros(8, 3, D.N_FEATURES)
    col = D.FEATURE_NAMES.index("alive_players_start_norm")
    X[0, :, col] = torch.tensor([3, 1, 2]) / 4.0  # phase 1 start (W = 100): a1 a2 a3 / b1 / c1 c2
    X[1, :, col] = torch.tensor([1, 0, 2]) / 4.0  # phase 2 start (W = 210): a1 / - / c1 c2
    return {
        "match_id": info.match_id, "team_ids": torch.tensor(info.team_ids), "phase_mask": torch.from_numpy(info.phase_mask),
        "cut_t": torch.tensor(info.cut_t, dtype=torch.float32), "phase_start_s": torch.tensor(info.phase_start_s, dtype=torch.float32),
        "phase_end_s": torch.tensor(info.phase_end_s, dtype=torch.float32), "clock_delta_s": info.delta_s, "map_size": info.map_size,
        "team_death_s": torch.tensor([math.inf, 120.0, math.inf]), "X": X,
    }  # fmt: skip


def test_validation_against_the_phase_record_and_roundtrip(tmp_path):
    raw = _raw()
    pm = PT.assemble_player_match(raw)
    pm.pop("_stats")
    PT.validate_player_match(pm, _phase_record(raw))
    path = tmp_path / "split.pt"
    torch.save({"meta": {"cohort": "unit"}, "matches": [pm]}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)  # no pickle fallback needed
    PT.validate_player_match(loaded["matches"][0], _phase_record(raw))
    assert loaded["matches"][0]["player_ids"] == ORDER and loaded["matches"][0]["player_death_s"][0].item() == math.inf

    bad = dict(_phase_record(raw))
    bad["team_death_s"] = torch.tensor([math.inf, 99.0, math.inf])
    with pytest.raises(ValueError):
        PT.validate_player_match(pm, bad)
    wrong_counts = _phase_record(raw)
    wrong_counts["X"] = wrong_counts["X"].clone()
    wrong_counts["X"][0, 0, D.FEATURE_NAMES.index("alive_players_start_norm")] = 0.25
    with pytest.raises(ValueError):
        PT.validate_player_match(pm, wrong_counts)
    broken = dict(pm)
    broken["Xp"] = pm["Xp"].clone()
    broken["Xp"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        PT.validate_player_match(broken)


def test_dump_is_deterministic_and_has_no_hidden_state():
    a, b = PT.assemble_player_match(_raw()), PT.assemble_player_match(_raw())
    for key in ("Xp", "player_node_mask", "player_alive_cut", "player_pos_valid", "player_death_s"):
        assert torch.equal(a[key], b[key]), key
    for kind in PT.EDGE_TYPES:
        assert torch.equal(a["edges_player"][kind][0], b["edges_player"][kind][0])


# ------------------------------------------------------------------------------ DB-gated
@requires_db
def test_one_real_erangel_match_passes_the_independent_verifier():
    """Raw fetch -> assembly -> tools/verify_player_dump_match.verify (plain SQL + python loops)."""
    from stdb.conn import connect
    from tools.verify_player_dump_match import verify

    phase_path = Path("model_stgcn/dumps/bigcomp2027_phase_v1/erangel_tpp_squad/test.pt")
    if not phase_path.exists():
        pytest.skip("the phase dump of erangel_tpp_squad is not available")
    phase = torch.load(phase_path, map_location="cpu", weights_only=True)
    rec = phase["matches"][0]
    info = PT.phase_info_from_record(rec)
    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            raw = PT.fetch_raw_player_match(cur, info)
            pm = PT.assemble_player_match(raw)
            pm.pop("_stats")
            PT.validate_player_match(pm, rec)
            results = verify(cur, pm, rec)
    finally:
        conn.close()
    failed = [(n, d) for n, ok, d in results if not ok]
    assert not failed, f"{rec['match_id']}: {failed}"
