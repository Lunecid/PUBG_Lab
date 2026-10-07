"""Unit tests of the leak-free phase-tier dump v2 (model_stgcn/phase_v2/*): DB-free, synthetic match.

Timeline (tests/phase_v2_synth.py): b_1 = 600, b_2 = 810, b_3 = 990 (phase 4 is final), a_k = b_k - 10, C_m = 1100.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from model_stgcn import dump_phase_tier as V1
from model_stgcn import dump_phase_tier_v2 as D2
from model_stgcn.phase_v2 import alive as AL
from model_stgcn.phase_v2 import assemble as AS
from model_stgcn.phase_v2 import features as FT
from model_stgcn.phase_v2 import grid as GR
from model_stgcn.phase_v2 import labels as LB
from model_stgcn.phase_v2 import phases as PH
from model_stgcn.phase_v2 import source as S
from tests.phase_v2_synth import END_W, KILLS, PHASE_EVENTS, SCHEDULE, TEAMS, make_source

MAP = 8160.0
TEAM_IDS = sorted(TEAMS)
TI = {t: i for i, t in enumerate(TEAM_IDS)}
COL = {n: i for i, n in enumerate(FT.FEATURE_NAMES)}


def drop_after(src: S.MatchSource, cut: float) -> S.MatchSource:
    """Independent re-implementation of the truncation (plain boolean masks per table, no use of S.truncate)."""

    def cutt(t):
        m = t["w"] < cut
        return {k: v[m] for k, v in t.items()}

    return replace(
        src, end_w=None, phase_events=cutt(src.phase_events), gs=cutt(src.gs), pos=cutt(src.pos),
        kills=cutt(src.kills), damage=cutt(src.damage), groggy=cutt(src.groggy),
        items={k: cutt(v) for k, v in src.items.items()}, cut_w=cut,
    )  # fmt: skip


def cut_features(src, k):
    ph = PH.phases_of(src)
    return FT.compute_cut_features(src, k, ph.cut(k), map_size=MAP)


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# --------------------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------------------
def test_phase_boundaries_come_from_events_and_the_final_phase_has_no_cut():
    ph = PH.phases_of(make_source())
    assert ph.indices == [1, 2, 3, 4]
    assert [ph.start(k) for k in (1, 2, 3, 4)] == [90.0, 600.0, 810.0, 990.0]
    assert [ph.boundary(k) for k in (1, 2, 3)] == [600.0, 810.0, 990.0]  # first event with ceil(isGame) = k + 1
    assert [ph.cut(k) for k in (1, 2, 3)] == [590.0, 800.0, 980.0]  # a_k = b_k - 10
    assert ph.boundary(4) is None and ph.cut(4) is None and not ph.has_cut(4)
    assert ph.last_phase == 4 and PH.monotone(ph)


def test_isgame_ceil_mapping_x0_and_x5():
    # x.5 (the x-th zone shrinks) opens the NEXT analysis phase; x.0 (zone x appears) stays inside phase x
    ph = PH.phases_from_events(np.array([10.0, 20.0, 30.0, 40.0]), np.array([0.1, 1.0, 1.5, 2.0]))
    assert ph.first_w == {1: 10.0, 2: 30.0}
    ph = PH.phases_from_events(np.array([5.0, 6.0]), np.array([0.0, np.nan]))  # isGame 0 / missing -> no phase
    assert ph.first_w == {}
    ph = PH.phases_from_events(np.array([30.0, 20.0]), np.array([2.5, 2.5]))  # earliest event of a phase wins
    assert ph.first_w == {3: 20.0}


# --------------------------------------------------------------------------------------
# causal alive status
# --------------------------------------------------------------------------------------
def test_kill_then_revive_needs_a_visible_healthy_sample():
    src = make_source()
    tl = AL.timelines_from_source(src)
    a1 = src.account_index()["a1"]
    # a1: killed 400, first healthy sample afterwards at 510.3, killed again 905
    assert AL.player_alive_at(tl, a1, 399.9)
    assert not AL.player_alive_at(tl, a1, 400.0)  # a kill AT t counts (<= t)
    assert not AL.player_alive_at(tl, a1, 510.2)  # revive evidence not observable yet
    assert AL.player_alive_at(tl, a1, 510.3)
    assert not AL.player_alive_at(tl, a1, 905.0)
    # accounts without a kill event are always alive
    assert AL.player_alive_at(tl, src.account_index()["c1"], 1e9)


def test_stale_health_sample_right_after_a_kill_is_not_revive_evidence():
    """Real case (solo match 8df7e412): a sample logged 1.6 s after the kill still carries health 10.  Revive evidence
    must be >= MIN_REVIVE_GAP_S after the kill (real redeploys appear >= 34 s later), else the team is censored."""
    src = make_source()
    ai = src.account_index()
    one = {"health": 10.0, "x": 1200.0, "y": 1300.0, "z": 20.0, "veh": False, "vspeed": np.nan, "team": 12, "acct": ai["b2"]}
    stale = make_source(extra_pos=[{**one, "w": 701.6, "g": 701}])  # b2 is killed at 700.0
    tl = AL.timelines_from_source(stale)
    assert not AL.player_alive_at(tl, ai["b2"], 702.0) and not AL.player_alive_at(tl, ai["b2"], 1e9)
    T = AL.team_elimination_times(stale.team_ids, stale.roster, stale.account_index(), tl)
    assert T[TI[12]] == 700.0  # not censored
    ev = AL.stale_health_samples(stale.kills["w"], stale.kills["victim"], stale.pos["w"], stale.pos["acct"], stale.pos["health"])
    assert [(e["acct"], e["kill_w"]) for e in ev] == [(ai["b2"], 700.0)]
    # the literal rule ("any later healthy sample") would have revived it
    lit = AL.build_timelines(stale.kills["w"], stale.kills["victim"], stale.pos["w"], stale.pos["acct"], stale.pos["health"], 0.0)
    assert lit[ai["b2"]].alive_at(1e9)
    # a healthy sample 25 s after the kill is revive evidence
    ok = make_source(extra_pos=[{**one, "w": 725.0, "g": 725, "health": 100.0}])
    tl = AL.timelines_from_source(ok)
    assert not AL.player_alive_at(tl, ai["b2"], 724.9) and AL.player_alive_at(tl, ai["b2"], 725.0)


def test_knocked_is_not_eliminated():
    src = make_source()
    # team 14: e1 is down (health 0, no kill) from 570 until killed at 1085, e2 alive until 1080 -> alive at 1000
    assert AL.team_alive_at(src, 1000.0)[TI[14]]
    # team 12 at 645: b1 is knocked (health 0), b2 alive
    assert AL.team_alive_at(src, 645.0)[TI[12]]
    # all members of team 15 knocked (health 0, no kill) for a while: still alive
    pos = {k: v.copy() for k, v in src.pos.items()}
    m = np.isin(pos["acct"], [src.account_index()["c1"], src.account_index()["c2"]]) & (pos["w"] > 700) & (pos["w"] < 760)
    pos["health"][m] = 0.0
    src2 = replace(src, pos=pos)
    assert AL.team_alive_at(src2, 750.0)[TI[15]]
    # the team is dead only after BOTH kill events
    assert AL.team_alive_at(src, 699.0)[TI[12]] and not AL.team_alive_at(src, 700.0)[TI[12]]


def test_alive_count_step_function_min_and_time_average():
    src = make_source()
    tl = AL.timelines_from_source(src)
    mem = AL.team_members(src.roster, src.account_index())
    # team 11, window [90, 590): 2 alive, a1 dies at 400 (-> 1), redeployed at 510.3 (-> 2)
    n = AL.alive_count_features(mem[11], tl, 90.0, 590.0)
    assert n[:3] == (2.0, 2.0, 1.0)
    assert n[3] == pytest.approx((2 * 310 + 1 * 110.3 + 2 * 79.7) / 500.0)
    # team 13: d1 dies 560, d2 dies 570 -> dead at the cut (no revive evidence before it)
    n = AL.alive_count_features(mem[13], tl, 90.0, 590.0)
    assert n[:3] == (2.0, 0.0, 0.0)
    assert n[3] == pytest.approx((2 * 470 + 1 * 10) / 500.0)
    # empty window: the count at the end, four times
    assert AL.alive_count_features(mem[11], tl, 600.0, 590.0) == (2.0, 2.0, 2.0, 2.0)


def test_alive_counts_enter_x_as_the_four_replaced_columns_over_4():
    cf = cut_features(make_source(), 1)
    row = {n: cf.X[TI[11], COL[n]] for n in FT.REPLACED_FEATURES}
    assert row["alive_players_start_norm"] == pytest.approx(2 / 4)
    assert row["alive_players_end_norm"] == pytest.approx(2 / 4)
    assert row["alive_players_min_norm"] == pytest.approx(1 / 4)
    assert row["alive_players_avg_norm"] == pytest.approx(((2 * 310 + 110.3 + 2 * 79.7) / 500.0) / 4, rel=1e-5)
    assert cf.X[TI[13], COL["alive_players_end_norm"]] == 0.0 and not cf.node_alive[TI[13]]
    assert np.array_equal(cf.alive_counts[TI[13]][:3], [2.0, 0.0, 0.0])


def test_kill_exactly_at_the_cut_is_not_visible():
    # d1 is killed at 800.0 = a_2: invisible at the cut (strict W < a_k) -> team 13 still alive at a_2
    cf = cut_features(make_source(), 2)
    assert cf.node_alive[TI[13]]
    assert cf.alive_counts[TI[13]][1] == 1.0


# --------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------
def build(src=None, **kw):
    return AS.build_match_v2(src or make_source(), map_size=MAP, **kw)


def test_elimination_time_and_risk_set_alive_at_the_boundary():
    rec = build()
    T = rec["team_death_s"].numpy()
    assert dict(zip(TEAM_IDS, T.tolist())) == {11: 905.0, 12: 700.0, 13: 800.0, 14: 1085.0, 15: float("inf")}
    tgt = rec["target_mask_v2"].numpy()
    ph = PH.phases_of(make_source())
    for k in (1, 2, 3):
        b = ph.boundary(k)
        assert np.array_equal(tgt[k - 1], T > b), f"cut {k}: risk set must be exactly {{T > b_k}}"
    # team 13 died at 800 = inside (a_2, b_2]: causally alive at the cut, not at risk (conditional selection)
    assert rec["node_mask"][1, TI[13]] and not tgt[1, TI[13]]
    assert not tgt[3].any() and not rec["cut_valid"][3] and not rec["cut_mask"][3]  # final phase: no cut


def test_hazard_label_window_and_terminal_interval():
    rec = build()
    y = rec["y_hazard_v2"].numpy()
    assert dict(zip(TEAM_IDS, y[0])) == {11: 0.0, 12: 1.0, 13: 1.0, 14: 0.0, 15: 0.0}  # (600, 810]
    assert dict(zip(TEAM_IDS, y[1])) == {11: 1.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0}  # (810, 990]: T = 905
    # cut 3: phase 4 is the final phase -> window (990, C_m = 1100]: team 14 (T = 1085) is a positive
    assert dict(zip(TEAM_IDS, y[2])) == {11: 0.0, 12: 0.0, 13: 0.0, 14: 1.0, 15: 0.0}
    assert bool(rec["cut_terminal"][2]) and not bool(rec["cut_terminal"][0])
    assert float(rec["label_window_end_s"][2]) == END_W and float(rec["label_window_end_s"][0]) == 810.0


def test_winner_is_censored_even_with_a_finite_kill_time():
    src = make_source()
    ai = src.account_index()
    kills = {k: v.copy() for k, v in src.kills.items()}
    extra = {"w": [1060.0, 1070.0], "victim": [ai["c1"], ai["c2"]], "killer": [ai["e2"]] * 2, "victim_team": [15, 15],
             "vx": [np.nan] * 2, "vy": [np.nan] * 2, "kx": [np.nan] * 2, "ky": [np.nan] * 2, "rank": [0, 0]}  # fmt: skip
    for k in kills:
        kills[k] = np.concatenate([kills[k], np.asarray(extra[k], dtype=kills[k].dtype)])
    pos = {k: v.copy() for k, v in src.pos.items()}
    for a, t in ((ai["c1"], 1060.0), (ai["c2"], 1070.0)):  # a dead body reports health 0 after the kill
        pos["health"][(pos["acct"] == a) & (pos["w"] > t)] = 0.0
    src2 = replace(src, kills=kills, pos=pos)
    tl = AL.timelines_from_source(src2)
    T = AL.team_elimination_times(src2.team_ids, src2.roster, src2.account_index(), tl)
    assert T[TI[15]] == 1070.0  # all members killed: finite by the kill rule ...
    rec = build(src2)
    assert np.isinf(rec["team_death_s"][TI[15]])  # ... but the winner is censored at C_m
    assert not (rec["y_hazard_v2"][:, TI[15]] > 0).any()
    assert rec["target_mask_v2"][:3, TI[15]].all()  # and stays at risk at every cut
    assert rec["_stats"]["censored_winner_finite_T"] == 1


def test_members_without_kill_event_follow_the_death_type_fallback():
    src = make_source()
    roster = list(src.roster)
    i = [j for j, r in enumerate(roster) if r[1] == "c2"][0]
    roster[i] = (15, "c2", "logout", 700.0)  # c2 left the game (no kill event)
    src2 = replace(src, roster=roster, team_won=np.zeros(5, dtype=bool))
    tl = AL.timelines_from_source(src2)
    T = AL.team_elimination_times(src2.team_ids, src2.roster, src2.account_index(), tl)
    assert T[TI[15]] == np.inf  # c1 is still alive
    # whereas a leaver on an otherwise dead team dies at time_survived + median offset
    roster = [(t, a, ("logout" if a == "b2" else dt), (700.0 if a == "b2" else ts)) for t, a, dt, ts in src.roster]
    src3 = replace(src, roster=roster)
    kills = {k: v.copy() for k, v in src3.kills.items()}
    m = kills["victim"] != src3.account_index()["b2"]
    kills = {k: v[m] for k, v in kills.items()}
    src3 = replace(src3, kills=kills)
    tl = AL.timelines_from_source(src3)
    T = AL.team_elimination_times(src3.team_ids, src3.roster, src3.account_index(), tl)
    assert np.isfinite(T[TI[12]]) and T[TI[12]] == pytest.approx(max(650.0, 700.0 + AL.FALLBACK_OFFSET_S + 0.0), abs=60.0)


def test_comeback_row_keeps_its_query_token_and_only_the_key_mask_follows_the_causal_status():
    rec = build()
    i = TI[13]
    assert bool(rec["target_mask_v2"][0, i]) and not bool(rec["node_mask"][0, i])  # in the risk set, causally dead
    assert bool(rec["query_mask"][0, i])  # the query token stays
    assert rec["_stats"]["comeback_rows"] == 1 and rec["_stats"]["comeback_rows_by_cut"][0] == 1
    # its row carries the genuine features (alive counts follow the causal status: 0 alive at the cut)
    x = rec["X"][0, i].numpy()
    assert x[COL["phase_idx_norm"]] == pytest.approx(1 / 8) and x[COL["team_size_norm"]] == pytest.approx(2 / 4)
    assert x[COL["alive_players_end_norm"]] == 0.0 and x[COL["alive_players_start_norm"]] == pytest.approx(0.5)
    # query = node | target everywhere
    assert np.array_equal(rec["query_mask"].numpy(), rec["node_mask"].numpy() | rec["target_mask_v2"].numpy())


def test_exclusion_only_for_a_truncated_log():
    src = make_source(end_w=None)  # no LogMatchEnd
    rec = build(src)
    assert rec["cut_excluded"] == {1: "no_match_end", 2: "no_match_end", 3: "no_match_end"}
    assert not rec["target_mask_v2"].any() and not rec["cut_valid"].any() and rec["cut_mask"][:3].all()
    assert rec["_stats"]["excluded_rows"] == {1: 5, 2: 3, 3: 2}  # would-be rows (teams with T > b_k)

    def with_gap(lo, hi):
        s = make_source()
        keep = ~((s.pos["w"] >= lo) & (s.pos["w"] < hi))
        return replace(s, pos={k: v[keep] for k, v in s.pos.items()})

    rec = build(with_gap(650.0, 730.0))  # an 80 s hole inside the label window of cut 1 (600, 810]
    assert rec["cut_excluded"] == {1: "position_gap"}
    assert not rec["target_mask_v2"][0].any() and rec["target_mask_v2"][1].any() and rec["cut_valid"][1]
    assert build(with_gap(200.0, 270.0))["cut_excluded"] == {}  # a hole before every window
    assert build(with_gap(650.0, 700.0))["cut_excluded"] == {}  # 50 s + sampling < 60 s: not a truncated log
    # a trailing hole (no position after 1000 s) overlaps the terminal window (990, C_m]
    assert build(with_gap(1000.0, 1200.0))["cut_excluded"] == {3: "position_gap"}


# --------------------------------------------------------------------------------------
# the cut: nothing at or after a_k enters an input
# --------------------------------------------------------------------------------------
def all_outputs(cf: FT.CutFeatures) -> dict:
    out = {"X": cf.X, "node": cf.node_alive, "counts": cf.alive_counts, "pos": cf.team_pos, "zone": cf.zone,
           "ai": cf.team_ai_frac, "zone_valid": np.asarray(cf.zone_valid)}  # fmt: skip
    for key, (i, j, w) in cf.edges.items():
        out[f"e_{key}_i"], out[f"e_{key}_j"], out[f"e_{key}_w"] = i, j, w
    return out


@pytest.mark.parametrize("k", [1, 2, 3])
def test_prefix_replay_on_the_synthetic_match(k):
    src = make_source()
    a = PH.phases_of(src).cut(k)
    full = all_outputs(cut_features(src, k))
    part = all_outputs(FT.compute_cut_features(drop_after(src, a), k, a, map_size=MAP))
    assert full.keys() == part.keys()
    for name in full:
        assert same(full[name], part[name]), f"cut {k}: {name} differs between the full and the truncated source"


def test_replay_test_is_not_vacuous_a_leaky_builder_fails_it(monkeypatch):
    src = make_source()
    a = PH.phases_of(src).cut(1)

    def forgets_the_event_tables(s, c):  # cuts the phase events only (keeps the builder alive) -> leaks everything else
        return replace(s, phase_events={k: v[s.phase_events["w"] < c] for k, v in s.phase_events.items()})

    monkeypatch.setattr(FT, "truncate", forgets_the_event_tables)
    leaky_full = all_outputs(FT.compute_cut_features(src, 1, a, map_size=MAP))
    leaky_part = all_outputs(FT.compute_cut_features(drop_after(src, a), 1, a, map_size=MAP))
    differing = [n for n in leaky_full if not same(leaky_full[n], leaky_part[n])]
    assert "X" in differing  # the damage / item / grid records after the cut leaked into the features
    monkeypatch.undo()
    ok_full = all_outputs(FT.compute_cut_features(src, 1, a, map_size=MAP))
    ok_part = all_outputs(FT.compute_cut_features(drop_after(src, a), 1, a, map_size=MAP))
    assert all(same(ok_full[n], ok_part[n]) for n in ok_full)


def test_records_after_the_cut_are_ignored_records_before_it_are_used():
    base = make_source()
    a = PH.phases_of(base).cut(1)  # 590
    ref = all_outputs(cut_features(base, 1))
    ai = base.account_index()
    # after the cut: damage, kill, item, position sample -> every output unchanged
    late = make_source(
        extra_dmg=[(a + 0.5, "c1", "a1", 500.0), (a + 3.0, "a1", "c1", 100.0)],
        extra_kills=[(a + 1.0, "c1")], extra_items=[(a + 2.0, 15, "Item_Weapon_Kar98k_C", "Weapon", "Main")],
        extra_pos=[{"w": a + 0.2, "g": int(a), "acct": ai["a2"], "team": 11, "health": 100.0, "x": 9e3, "y": 9e3,
                    "z": 5.0, "veh": False, "vspeed": np.nan}],
    )  # fmt: skip
    got = all_outputs(cut_features(late, 1))
    for n in ref:
        assert same(ref[n], got[n]), f"{n} changed although every added record is at or after a_1"
    # exactly at the cut counts as "after" (strict W < a_k)
    at = make_source(extra_dmg=[(a, "c1", "a1", 500.0)], extra_kills=[(a, "c1")])
    got = all_outputs(cut_features(at, 1))
    for n in ref:
        assert same(ref[n], got[n])
    # before the cut: the same records move the damage, alive and loot features
    early = make_source(extra_dmg=[(a - 0.5, "c1", "a1", 500.0)], extra_kills=[(a - 1.0, "c1")],
                        extra_items=[(a - 2.0, 15, "Item_Weapon_Kar98k_C", "Weapon", "Main")])  # fmt: skip
    got = all_outputs(cut_features(early, 1))
    assert not same(ref["X"], got["X"])
    r15, g15 = ref["X"][TI[15]], got["X"][TI[15]]
    assert g15[COL["damage_dealt_norm"]] > r15[COL["damage_dealt_norm"]]
    assert g15[COL["weapon_pickup_count_norm"]] > r15[COL["weapon_pickup_count_norm"]]
    assert got["counts"][TI[15], 2] < ref["counts"][TI[15], 2]  # c1 died before the cut -> the minimum drops


GRID_COLUMNS = [s[0] for s in FT.SPECS if s[1] in (*FT.TRAJ_COLUMNS, *FT.EXPO_COLUMNS, *FT.HAB_TEAM_COLUMNS, *FT.Z_COLUMNS)]


def test_grid_points_are_admissible_only_if_all_samples_they_read_precede_the_cut():
    assert GR.last_grid_time(590.0) == 580.0  # 580 + 5 < 590 but 590 + 5 is not
    assert GR.last_grid_time(595.0) == 580.0  # 590 + 5 is not < 595
    assert GR.last_grid_time(595.5) == 590.0
    src = make_source()
    base = all_outputs(cut_features(src, 1))
    ai = src.account_index()
    gcols = [COL[n] for n in GRID_COLUMNS]
    one = {"health": 100.0, "x": 9e3, "y": 9e3, "z": 5.0, "veh": False, "vspeed": np.nan, "team": 11, "acct": ai["a2"]}
    # a closer sample for the INADMISSIBLE grid point 590 (|589.9 - 590| < every other) leaves every grid-based output alone
    got = all_outputs(cut_features(make_source(extra_pos=[{**one, "w": 589.9, "g": 589}]), 1))
    assert same(base["pos"], got["pos"])
    assert same(base["X"][:, gcols], got["X"][:, gcols])
    assert all(same(base[n], got[n]) for n in base if n.startswith("e_"))
    # (the raw-position state features DO see the record: it precedes the cut)
    assert not same(base["X"], got["X"])
    # ... while a closer sample for the admissible grid point 580 moves the grid
    got = all_outputs(cut_features(make_source(extra_pos=[{**one, "w": 580.05, "g": 580}]), 1))
    assert not same(base["pos"], got["pos"]) and not same(base["X"][:, gcols], got["X"][:, gcols])


def test_grid_runs_to_the_end_of_a_log_longer_than_2000_s():
    events = [(90.0, 0.1), (330.0, 1.0), (600.0, 1.5), (720.0, 2.0), (1500.0, 2.5), (1620.0, 3.0), (2400.0, 3.5), (2500.0, 4.0)]
    src = make_source(length_s=2600.0, end_w=2600.0, phase_events=events,
                      schedule={1: 510.0, 2: 900.0, 3: 900.0, 4: 100.0})  # fmt: skip
    ph = PH.phases_of(src)
    assert ph.cut(3) == 2390.0 and src.end_w == 2600.0
    cf = FT.compute_cut_features(src, 3, ph.cut(3), map_size=MAP)
    assert cf.diag["n_grid"] > 200  # SNAP_T_MAX = 200 snapshots (2000 s) would have truncated it
    gi_last = GR.last_grid_time(2390.0)
    assert gi_last == 2380.0
    # the team position of the cut slot comes from a grid point after 2000 s
    assert np.isfinite(cf.team_pos[TI[15]]).all()
    assert cf.X[TI[15], COL["path_length_norm"]] > 0
    rec = AS.build_match_v2(src, map_size=MAP)
    assert bool(rec["cut_valid"][2]) and float(rec["match_end_s"]) == 2600.0 and float(rec["cut_t"][2]) == 2390.0
    # T beyond 2000 s is a normal event: kill of c-team members after 2000 s -> censoring / labels work
    assert rec["label_window_end_s"][2] == 2600.0


# --------------------------------------------------------------------------------------
# scheduled durations (D3)
# --------------------------------------------------------------------------------------
def test_phase_duration_inputs_use_the_schedule_not_the_observed_duration():
    base = cut_features(make_source(), 1)
    assert np.allclose(base.X[:, COL["phase_duration_10min"]], 510.0 / 600.0)
    # a different schedule moves every duration-based column ...
    s2 = make_source(schedule={**SCHEDULE, 1: 300.0})
    other = cut_features(s2, 1)
    assert np.allclose(other.X[:, COL["phase_duration_10min"]], 300.0 / 600.0)
    assert not np.allclose(base.X[:, COL["avg_speed_norm"]], other.X[:, COL["avg_speed_norm"]])
    # ... and a later b_1 (a longer OBSERVED phase 1, same schedule) leaves the schedule columns untouched
    events = [(w, g) if g < 1.5 else (w + 100.0, g) for w, g in PHASE_EVENTS]
    longer = cut_features(make_source(phase_events=events), 1)
    assert np.allclose(longer.X[:, COL["phase_duration_10min"]], 510.0 / 600.0)
    # the duration columns are exactly the documented ones and every other column does not divide by a duration
    assert set(FT.DURATION_FEATURES) <= set(FT.FEATURE_NAMES) and "phase_duration_10min" in FT.DURATION_FEATURES


def test_migration_deficit_is_distance_minus_path_over_scheduled_duration():
    cf = cut_features(make_source(), 1)
    i = TI[15]
    req = cf.X[i, COL["hab_required_speed_to_safe_mps"]] * 40.0
    obs = cf.X[i, COL["hab_observed_speed_mps"]] * 40.0
    deficit = cf.X[i, COL["hab_migration_deficit_mps"]] * 40.0
    assert deficit == pytest.approx(req - obs, abs=1e-4)
    dist = req * 510.0  # distance from the last alive position to the visible white circle
    path = obs * 510.0
    assert deficit == pytest.approx((dist - path) / 510.0, abs=1e-4)


def test_schedule_table_levels_and_fallback():
    durations = {f"m{i}": {1: 500.0 + i % 3, 2: 200.0} for i in range(30)}
    meta = {f"m{i}": ("c1", "Baltic_Main", "official") for i in range(30)}
    meta.update({f"x{i}": ("c2", "Baltic_Main", "competitive") for i in range(3)})
    durations.update({f"x{i}": {1: 520.0} for i in range(3)})
    table = D2.schedule_table(durations, meta)
    sched, label = D2.resolve_schedule(table, "c1", "Baltic_Main", "official")
    assert sched[1] == 501.0 and sched[2] == 200.0 and label == "cohort_median"
    sched, label = D2.resolve_schedule(table, "c2", "Baltic_Main", "competitive")  # 3 matches: below the support
    assert label.endswith("_fallback") and 1 in sched


def test_blue_zone_options_parsing_and_missing_options():
    assert S.parse_blue_zone_options("[]") == [] and S.parse_blue_zone_options(None) == []
    assert S.parse_blue_zone_options('[{"phaseNum": 1}]') == [{"phaseNum": 1}]
    assert S.parse_blue_zone_options('"[{\\"phaseNum\\": 1}]"') == [{"phaseNum": 1}]  # double encoded
    assert S.schedule_from_options([]) == {}
    assert S.schedule_from_options([{"phaseNum": 2, "warningDuration": 120, "releaseDuration": 90}]) == {2: 210.0}
    assert S.schedule_from_options([{"phaseNum": 2}]) == {}  # incomplete -> caller falls back


# --------------------------------------------------------------------------------------
# schema / contract
# --------------------------------------------------------------------------------------
def test_feature_layout_is_the_v1_layout():
    assert FT.FEATURE_NAMES == V1.FEATURE_NAMES and FT.N_FEATURES == 150
    assert FT.REPLACED_FEATURES == V1.REPLACED_FEATURES
    assert set(FT.EVENT_FEATURES_L2) <= set(FT.FEATURE_NAMES) and "damage_dealt_norm" in FT.EVENT_FEATURES_L2


def test_record_fields_dtypes_and_masks():
    rec = build()
    N = len(TEAM_IDS)
    shapes = {"X": (8, N, 150), "node_mask": (8, N), "query_mask": (8, N), "target_mask_v2": (8, N),
              "y_hazard_v2": (8, N), "team_pos": (8, N, 2), "zone": (8, 8), "team_death_s": (N,),
              "cut_mask": (8,), "cut_valid": (8,), "cut_terminal": (8,), "phase_start_s": (8,),
              "phase_end_s": (8,), "cut_t": (8,), "phase_duration_sched_s": (8,)}  # fmt: skip
    for key, shape in shapes.items():
        assert tuple(rec[key].shape) == shape, key
    assert rec["X"].dtype.is_floating_point and rec["node_mask"].dtype.__str__() == "torch.bool"
    assert np.isfinite(rec["X"].numpy()).all()
    assert "node_mask_v2" not in rec  # its presence would rebuild the alive mask from the final death time (L3)
    assert set(rec["edges"]) == {"prox", "combat", "copress"}
    for idx, w in rec["edges"].values():
        assert idx.shape[1] == 3 and w.shape[0] == idx.shape[0]
        if len(idx):
            assert (idx[:, 1] != idx[:, 2]).all()
    # slot bookkeeping: phase_mask 4 phases, cuts 1..3, final phase slot empty
    assert rec["phase_mask"].numpy().tolist() == [True] * 4 + [False] * 4
    assert rec["cut_mask"].numpy().tolist() == [True] * 3 + [False] * 5
    assert not rec["X"][3].numpy().any() and not rec["node_mask"][3].any()
    assert float(rec["phase_end_s"][0]) == 600.0 and np.isnan(rec["phase_end_s"][3].item())
    assert rec["cut_t"].numpy()[:3].tolist() == [590.0, 800.0, 980.0]
    assert rec["phase_duration_sched_s"].numpy()[:2].tolist() == [510.0, 210.0]


def test_alive_module_public_api_is_stable():
    """P1 imports these (coordinator contract): names and leading parameters must not change."""
    sigs = {
        "build_timelines": ["kill_w", "kill_victim", "pos_w", "pos_acct", "pos_health"],
        "timelines_from_source": ["src"],
        "team_members": ["roster", "account_index"],
        "player_alive_at": ["timelines", "account", "t"],
        "causal_team_alive": ["members", "timelines", "t"],
        "team_alive_at": ["src", "t", "timelines"],
        "alive_count_features": ["members", "timelines", "start", "end"],
        "team_elimination_times": ["team_ids", "roster", "account_index", "timelines"],
        "member_final_death": ["timeline", "death_type", "time_survived", "offset_s"],
        "revive_evidence": ["timelines", "pos", "map_size"],
    }
    for name, params in sigs.items():
        got = list(inspect.signature(getattr(AL, name)).parameters)
        assert got[: len(params)] == params, (name, got)
    assert hasattr(AL.PlayerTimeline, "alive_at") and hasattr(AL.PlayerTimeline, "transitions")


def test_meta_v2_block_documents_the_isgame_semantics_and_every_changed_feature(tmp_path):
    from model_stgcn.v3.cohorts import REQUIRED_COHORTS

    spec = REQUIRED_COHORTS[0]
    meta = D2.build_meta_v2(spec, "test", split_path=tmp_path / "split.json", split_ids=["a", "b"], extra={"n_dumped": 2},
                            schedule_info={"source": "test"})  # fmt: skip
    assert meta["version"] == "v2" and meta["v2"]["version"] == "v2"
    assert "documentation.pubg.com/en/telemetry-objects.html#common" in meta["v2"]["isgame"]["url"]
    assert "x.0" in meta["v2"]["isgame"]["text"] and "x.5" in meta["v2"]["isgame"]["text"]
    ch = meta["v2"]["changed_features"]
    assert set(ch["feature_change_class"]) == set(FT.FEATURE_NAMES) and sum(ch["class_sizes"].values()) == 150
    assert ch["replaced_values_same_names"] == list(FT.REPLACED_FEATURES)
    assert set(ch["scheduled_duration_inputs"]) == set(FT.DURATION_FEATURES)
    assert meta["replaced_features"] == list(FT.REPLACED_FEATURES) and meta["feature_names"] == FT.FEATURE_NAMES
    assert "team_death_source" not in meta and not any(k.startswith("fixb") for k in meta)
    assert "node_mask_v2" in meta["conventions"] and meta["clock"]["W"].startswith("LogMatchStart clock")


def test_aggregate_stats_counts_rows_comeback_rows_exclusions_and_revive_flags():
    recs = [build(), build(make_source(end_w=None))]
    stats = D2.aggregate_stats(recs)
    assert stats["rows_by_cut"][:3] == [5, 3, 2] and stats["positives_by_cut"][:3] == [2, 1, 1]  # the excluded match adds none
    assert stats["comeback_rows"] == 1 and stats["comeback_rows_by_cut"][0] == 1
    assert stats["excluded_match_cuts"] == {"no_match_end": 3} and stats["excluded_rows"] == {"no_match_end": 10}
    assert stats["revive_evidence"]["events"] == 4 and all("_stats" not in r for r in recs)
