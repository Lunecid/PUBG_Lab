"""Sagong et al. (2025) baseline: feature windows, causality, team adaptation, types, Eq. 4 and the fit stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pubg_lab.survival.baselines.sagong2025.features as F
import pubg_lab.survival.baselines.sagong2025.fit as T

T0 = 1000.0  # match start epoch: ep = W + T0
WINDOWS = {1: (100.0, 300.0), 2: (310.0, 500.0)}  # cuts (E - 10): 290 and 490


def _ev(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def make_raw(extra_dmg=(), extra_kill=(), extra_pick=(), extra_pos=()) -> F.MatchRaw:
    team_of = {"a1": 1, "a2": 1, "b1": 2, "c1": 3}
    pos = []
    for w, x in ((100, 0), (110, 30), (120, 60), (130, 90), (295, 1000), (320, 1040), (330, 1080)):
        pos.append(("a1", w + T0, float(x), 0.0))
    for w, x in ((100, 0), (110, 10)):
        pos.append(("a2", w + T0, float(x), 0.0))
    for w in (100, 110, 120, 130):
        pos.append(("b1", w + T0, 50.0, 0.0))
    for w in (100, 110, 120, 130):
        pos.append(("c1", w + T0, 500.0, 500.0))
    pos += [(a, w + T0, x, y) for a, w, x, y in extra_pos]
    dmg = [  # attacker, victim, damage, W
        ("a1", "b1", 40.0, 150), ("a1", "b1", 30.0, 160), ("a2", "b1", 20.0, 170),
        ("a1", "a2", 99.0, 180),  # friendly fire: not an attack on an enemy
        ("b1", "a1", 10.0, 185),
        ("a1", "b1", 1000.0, 292),  # after the cut of phase 1 (290)
        ("a1", "c1", 7.0, 320),  # phase 2
        ("a1", "c1", 500.0, 495),  # after the cut of phase 2 (490)
    ]
    dmg += list(extra_dmg)
    kill = [  # killer, victim, suicide, W
        ("a1", "b1", False, 200),  # enemy kill, helpers: a2 (a1 is the killer)
        ("a1", "a2", False, 250),  # team kill: not counted
        ("b1", "c1", False, 295),  # between the cut of phase 1 and the start of phase 2: in no window
        ("c1", "c1", True, 100),  # suicide
    ]
    kill += list(extra_kill)
    pick = [("a1", "pickup", 120), ("a1", "pickup_carepackage", 130), ("a1", "drop", 140), ("a2", "pickup", 150),
            ("a2", "pickup", 291), ("b1", "pickup_vehicle_trunk", 105)]
    pick += list(extra_pick)
    return F.MatchRaw(
        match_id="m", start_epoch=T0, team_of=team_of,
        pos=_ev(pos, ["acc", "ep", "x", "y"]),
        dmg=_ev([(a, v, d, w + T0) for a, v, d, w in dmg], ["attacker", "victim", "damage", "ep"]),
        kill=_ev([(k, v, s, w + T0) for k, v, s, w in kill], ["killer", "victim", "is_suicide", "ep"]),
        pick=_ev([(a, t, w + T0) for a, t, w in pick], ["acc", "event_type", "ep"]),
    )


def feats(raw=None, windows=None) -> pd.DataFrame:
    return F.compute_match_features(raw or make_raw(), windows or WINDOWS)


def row(df, team, phase):
    return df[(df.team_id == team) & (df.phase == phase)].iloc[0]


def test_guard_equals_causal_cut_guard():
    cc = pytest.importorskip('pubg_lab.data.stdb.temporal.causal_cut')
    assert F.GUARD_S == cc.GUARD_S


def test_team_one_phase_one_values():
    r = row(feats(), 1, 1)
    assert r.attack == 3 and r.damage == 90.0  # friendly fire and the post-cut hit are out
    assert r.kill == 1  # the team kill (a1 -> a2) and the suicide are out
    assert r.assist == 1  # a2 damaged b1 within 60 s before a1 killed it; the killer a1 gets none
    assert r["item"] == 3 and r.carepackage == 1  # 'drop' and the pick-up after the cut are out
    assert r.distance == pytest.approx((90.0 + 10.0) / 2)  # mean over the two observed members
    assert r.density == 1  # b1 within 100 m of a1; the teammate a2 does not count; c1 is far
    assert r.n_members == 2 and r.n_obs == 2


def test_other_teams_phase_one():
    df = feats()
    b = row(df, 2, 1)
    assert b.attack == 1 and b.damage == 10.0 and b.kill == 0 and b.assist == 0 and b["item"] == 1
    assert b.density == 2  # a1 and a2 (different team) within 100 m
    c = row(df, 3, 1)
    assert c.density == 0 and c.attack == 0 and c.kill == 0  # c1's only kill is a suicide / outside the windows


def test_assist_proxy_excludes_damage_by_other_teams_and_solo_helpers():
    raw = make_raw(extra_dmg=[("c1", "b1", 90.0, 190)])
    df = feats(raw)
    assert row(df, 1, 1).assist == 1  # a2 helps teammate a1
    assert row(df, 3, 1).assist == 0  # c1 is on a different team
    raw.team_of["a2"] = 4  # every player now has a distinct team
    assert (feats(raw)["assist"] == 0).all()


def test_steps_belong_to_the_window_of_their_end_sample():
    df = feats()
    # sample W=295 lies in no window: the 910 m step 130 -> 295 is nowhere; the step 295 -> 320 (40 m... x 1000 -> 1040) is phase 2's
    r2 = row(df, 1, 2)
    assert r2.distance == pytest.approx(40.0 + 40.0)  # steps ending at W=320 and W=330, only a1 observed
    assert r2.n_obs == 1


def test_nothing_after_the_cut_is_read():
    base = feats()
    late = make_raw(
        extra_dmg=[("a2", "b1", 5000.0, 291), ("b1", "a1", 5.0, 299)],
        extra_kill=[("a2", "b1", False, 291)],
        extra_pick=[("a1", "pickup_carepackage", 295), ("b1", "pickup", 299)],
        extra_pos=[("a2", 291, 4000.0, 0.0), ("b1", 292, 52.0, 0.0), ("c1", 293, 52.0, 0.0)],
    )
    for team in (1, 2, 3):
        a, b = row(base, team, 1), row(feats(late), team, 1)
        assert np.allclose(a[list(F.FEATURES)].to_numpy(dtype=float), b[list(F.FEATURES)].to_numpy(dtype=float)), team
    # ... and the position at exactly the cut W = E - 10 = 290 IS read
    at_cut = make_raw(extra_pos=[("a2", 290, 4000.0, 0.0)])
    assert row(feats(at_cut), 1, 1).distance > row(base, 1, 1).distance


def test_density_counts_only_other_teams_within_radius():
    w = np.array([100.0, 100.0, 100.0, 100.0])
    x = np.array([0.0, 50.0, 100.0, 101.0])
    y = np.zeros(4)
    acc = np.arange(4)
    team = np.array([0, 1, 1, 2])
    out = F.max_density_by_team(w, x, y, acc, team, 3)
    # P0 (team 0, x=0): P1 (50 m) and P2 (100 m, the radius is inclusive) -> 2; P3 at 101 m is outside
    # P1 (team 1, x=50): P0 and P3 (51 m) -> 2, its teammate P2 (50 m) is not counted
    # P2 (team 1, x=100): P0 (100 m) and P3 (1 m) -> 2;  P3 (team 2, x=101): P1 and P2 -> 2
    assert out.tolist() == [2.0, 2.0, 2.0]
    out = F.max_density_by_team(w, x, y, acc, team, 3, radius_m=60.0)  # P0: only P1; P1: P0 and P3; P3: P1 (51) and P2 (1)
    assert out.tolist() == [1.0, 2.0, 2.0]


def test_density_uses_the_nearest_sample_per_grid_point():
    w = np.array([100.0, 103.0, 100.0])  # two samples of player 0 in the same grid cell: the nearest (100) is used
    x = np.array([0.0, 500.0, 40.0])
    out = F.max_density_by_team(w, x, np.zeros(3), np.array([0, 0, 1]), np.array([0, 0, 1]), 2)
    assert out.tolist() == [1.0, 1.0]


def test_too_short_phase_gives_zeros():
    df = F.compute_match_features(make_raw(), {1: (100.0, 105.0)})  # cut = 95 <= start
    assert (df[list(F.FEATURES)].to_numpy() == 0).all()


def test_solo_team_of_one():
    raw = F.MatchRaw("s", T0, {"p1": 1, "p2": 2},
                     pos=_ev([("p1", 100 + T0, 0.0, 0.0), ("p1", 110 + T0, 3.0, 4.0), ("p2", 100 + T0, 10.0, 0.0), ("p2", 110 + T0, 10.0, 0.0)],
                             ["acc", "ep", "x", "y"]),
                     dmg=_ev([("p1", "p2", 25.0, 150 + T0)], ["attacker", "victim", "damage", "ep"]),
                     kill=_ev([("p1", "p2", False, 160 + T0)], ["killer", "victim", "is_suicide", "ep"]),
                     pick=_ev([("p1", "pickup", 120 + T0)], ["acc", "event_type", "ep"]))
    df = F.compute_match_features(raw, {1: (100.0, 300.0)})
    p1 = row(df, 1, 1)
    assert (p1.attack, p1.kill, p1.damage, p1.assist, p1["item"]) == (1, 1, 25.0, 0, 1)
    assert p1.distance == pytest.approx(5.0) and p1.density == 1 and p1.n_members == 1
    assert row(df, 2, 1).kill == 0


# ----------------------------------------------------------------------------- types, Eq. 4, inputs
def _units(rows):
    df = pd.DataFrame(rows, columns=["split", "phase", "alive_start", "target", "y"] + list(F.FEATURES))
    return T.add_scores(df)


def test_scores_are_unweighted_sums():
    df = _units([("train", 1, True, True, 0, 1, 2, 3, 4, 10, 20, 30, 1)])
    assert df.fight.iloc[0] == 1 + 2 + 3 + 4 and df.active.iloc[0] == 10 + 20 + 30 + 1


def test_means_use_train_alive_units_only_and_types_follow_table_5():
    z = (0, 0, 0, 0, 0, 0, 0, 0)
    df = _units([
        ("train", 1, True, True, 0) + (10, 0, 0, 0, 10, 0, 0, 0),  # F=10 A=10
        ("train", 1, True, True, 1) + (0, 0, 0, 0, 0, 0, 0, 0),  # F=0 A=0
        ("train", 1, False, False, 0) + (1000, 0, 0, 0, 1000, 0, 0, 0),  # not alive at start: ignored in the mean
        ("test", 1, True, True, 0) + (1000, 0, 0, 0, 1000, 0, 0, 0),  # test: ignored in the mean
        ("test", 1, True, True, 0) + (6, 0, 0, 0, 0, 0, 0, 0),  # F=6 > 5, A=0
        ("test", 1, True, True, 0) + (0, 0, 0, 0, 6, 0, 0, 0),  # F=0, A=6 > 5
        ("test", 1, True, True, 0) + (5, 0, 0, 0, 5, 0, 0, 0),  # exactly the mean: '<=' -> Camper
    ])
    means = T.fit_phase_means(df[(df.split == "train") & df.alive_start])
    assert means.loc[1, "mean_fight"] == 5.0 and means.loc[1, "mean_active"] == 5.0
    types = T.assign_types(df, means)
    assert types.tolist() == [1, 4, 1, 1, 2, 3, 4]  # Hunter, Camper, (Hunter, Hunter), Sniper, Traveler, Camper


def test_survival_table_is_the_preceding_transition_rate():
    rows = pd.DataFrame({
        "phase": [1, 1, 1, 1, 2, 2, 3],
        "ptype": [1, 1, 4, 4, 1, 4, 1],
        "y": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    })
    table, info = T.fit_survival_table(rows, n_cuts=4)
    assert all(table[(1, t)] == 1.0 for t in (1, 2, 3, 4))  # first phase: 1.0
    assert table[(2, 1)] == 0.5 and table[(2, 4)] == 1.0  # SR_2[i] = 1 - mean(y_1 | type_1 = i)
    assert table[(2, 2)] == pytest.approx(0.75)  # empty cell -> phase-wide survival of the transition 1 -> 2
    assert table[(3, 1)] == 0.0 and table[(3, 4)] == 0.0  # transition 2 -> 3: both type 1 and 4 rows were eliminated
    assert [2, 2] in info["fallback_cells"]
    assert table[(4, 1)] == 1.0  # from the single phase-3 row


def test_paper_survival_rate_counts_units_alive_at_the_start_of_the_next_phase():
    def u(match, team, phase, ptype, alive=True):
        return {"match_id": match, "team_id": team, "phase": phase, "alive_start": alive, "ptype": ptype}

    units = pd.DataFrame([
        u("m1", "A", 1, 1), u("m1", "B", 1, 1), u("m1", "C", 1, 4), u("m1", "D", 1, 4),  # phase 1: types 1, 1, 4, 4
        u("m1", "A", 2, 1), u("m1", "C", 2, 4),  # B and D died in phase 1
        u("m1", "A", 3, 2),  # C died in phase 2
        u("m2", "E", 1, 1),  # m2 has no phase 2: its units are not counted for SR_2
    ])
    exists = {("m1", 1), ("m1", 2), ("m1", 3), ("m2", 1)}
    table, info = T.fit_survival_table_paper(units, exists, n_cuts=3)
    assert all(table[(1, t)] == 1.0 for t in (1, 2, 3, 4))
    assert table[(2, 1)] == 0.5 and table[(2, 4)] == 0.5  # 1 of 2 type-1 and 1 of 2 type-4 units survive to phase 2
    assert table[(2, 2)] == 0.5 and [2, 2] in info["fallback_cells"]  # empty cell -> overall 2 / 4
    assert info["n_by_cell"]["2:1"] == 2  # E (match without phase 2) is out
    # phase 2 -> 3: A (type 1 at phase 2) is alive at the start of 3, C (type 4 at phase 2) is not
    assert table[(3, 1)] == 1.0 and table[(3, 4)] == 0.0


def test_model_inputs_follow_the_paper_order():
    df = _units([("train", 2, True, True, 0, 1, 2, 3, 4, 5, 6, 7, 1)])
    df["ptype"] = 3
    X = T.model_inputs(df, {(2, 3): 0.8})
    assert X.shape == (1, 12) and len(T.INPUT_NAMES) == 12
    assert X[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 1, 10, 19, 3, 0.8]


# ----------------------------------------------------------------------------- the fit stage end to end
def _synthetic_plan(tmp_path, n_matches=60, seed=0):
    rng = np.random.default_rng(seed)
    base = tmp_path / "toy"
    base.mkdir(parents=True)
    units, feats_ = [], []
    for m in range(n_matches):
        split = "train" if m < 40 else ("val" if m < 50 else "test")
        for team in range(8):
            for k in range(1, 4):
                act = rng.random()
                y = float(rng.random() < 0.2 + 0.5 * (1 - act))
                units.append((split, f"m{m}", "official", team, k, True, True, y, (team + 1) / 9, float(team < 5), 100.0 * k, 100.0 * k + 150))
                feats_.append((f"m{m}", team, k, *[act * 10] * 4, act * 500, act * 5, act * 30, float(act > 0.9), 4.0, 4.0))
    pd.DataFrame(units, columns=["split", "match_id", "match_type", "team_id", "phase", "alive_start", "target", "y",
                                 "final_rank_norm", "top5", "S", "E"]).to_parquet(base / "units.parquet", index=False)
    pd.DataFrame(feats_, columns=["match_id", "team_id", "phase", *F.FEATURES, "n_members", "n_obs"]).to_parquet(base / "features.parquet", index=False)
    wins = [("train" if m < 40 else ("val" if m < 50 else "test"), f"m{m}", k, 100.0 * k, 100.0 * k + 150) for m in range(n_matches) for k in range(1, 5)]
    pd.DataFrame(wins, columns=["split", "match_id", "phase", "S", "E"]).to_parquet(base / "windows.parquet", index=False)
    return tmp_path


@pytest.mark.parametrize("kind", ["lr", "rf", "xgb", "hgb"])
def test_fit_stage_writes_ptr_format(tmp_path, kind):
    pytest.importorskip("pyarrow")
    if kind == "xgb":
        pytest.importorskip("xgboost")
    plan = _synthetic_plan(tmp_path / "plan")
    out = tmp_path / "runs"
    res = T.run_cohort(plan, "toy", out, seed=42, n_jobs=1, models=(kind,))
    name = T.MODELS[kind]
    d = out / "toy" / name / "seed42"
    import pubg_lab.survival.ptr.evaluation.metrics as M

    for split in ("val", "test"):
        df = M.read_predictions(d / f"pred_{split}.csv")
        assert list(df.columns) == M.PRED_COLUMNS and df.p_hazard.between(0, 1).all()
        assert np.allclose(df.rank_score, -df.p_hazard)
        expect = pd.read_parquet(plan / "toy" / "units.parquet")
        expect = expect[(expect.split == split) & expect.target]
        assert len(df) == len(expect) and set(df.match_id) == set(expect.match_id)
    assert res[name]["labels"] == "v2" and res[name]["n_features"] == 12 and "config_hash" in res[name]
    assert res[name]["test"]["hazard"]["auc"] > 0.55  # the toy signal is learnable


def _toy_hab_table(plan, names=("zone_a", "hab_b")):
    """A hab44.parquet for the synthetic plan: one row per risk-set row, columns = a deterministic function of the key."""
    import json

    units = pd.read_parquet(plan / "toy" / "units.parquet")
    units = units[units.target]
    h = units[["split", "match_id", "team_id", "phase", "y"]].copy()
    h["zone_a"] = (h.team_id * 3 + h.phase).astype("float32")
    h["hab_b"] = (h.phase * 0.5).astype("float32")
    h.sample(frac=1.0, random_state=0).to_parquet(plan / "toy" / "hab44.parquet", index=False)  # shuffled on purpose
    (plan / "toy" / "hab44_audit.json").write_text(json.dumps(
        {"names": list(names), "indices": [0, 1], "dump_fingerprint": {}, "equals_hgb_noHab_feature_removal": True}))


def test_attach_hab_joins_by_key_and_checks_labels(tmp_path):
    pytest.importorskip("pyarrow")
    from pubg_lab.survival.baselines.sagong2025.hab import attach_hab

    plan = _synthetic_plan(tmp_path / "plan")
    _toy_hab_table(plan)
    df, prep = T.prepare(plan, "toy")
    rows = {s: df[(df.split == s) & df.target].sort_values(["match_id", "team_id", "phase"], kind="mergesort").reset_index(drop=True)
            for s in ("train", "val", "test")}
    X = {s: T.model_inputs(r, prep["table"]) for s, r in rows.items()}
    XH, names, info = attach_hab(plan, "toy", rows, X)
    assert names == ["zone_a", "hab_b"] and info["n_columns"] == 2
    for s, r in rows.items():
        assert XH[s].shape == (len(r), 12 + 2) and np.array_equal(XH[s][:, :12], X[s])
        assert np.array_equal(XH[s][:, 12], (r.team_id * 3 + r.phase).to_numpy(dtype=float))  # joined by key, not by position
        assert np.array_equal(XH[s][:, 13], (r.phase * 0.5).to_numpy(dtype=float))
    bad = pd.read_parquet(plan / "toy" / "hab44.parquet")
    bad.loc[bad.index[0], "y"] = 1.0 - bad.loc[bad.index[0], "y"]  # one flipped label must be caught
    bad.to_parquet(plan / "toy" / "hab44.parquet", index=False)
    with pytest.raises(AssertionError):
        attach_hab(plan, "toy", rows, X)


def test_hgb_hab_arm_writes_ptr_format(tmp_path):
    pytest.importorskip("pyarrow")
    plan = _synthetic_plan(tmp_path / "plan")
    _toy_hab_table(plan)
    out = tmp_path / "runs"
    res = T.run_cohort(plan, "toy", out, seed=42, n_jobs=1, models=("hgb-hab",))
    r = res["SAGONG-hgb-hab"]
    assert r["n_features"] == 14 and r["n_sagong_inputs"] == 12 and r["input_names"][-2:] == ["zone_a", "hab_b"]
    assert r["habitat_columns"]["n_columns"] == 2 and (out / "toy" / "SAGONG-hgb-hab" / "seed42" / "pred_test.csv").is_file()
    assert r["test"]["hazard"]["auc"] > 0.55


def test_hgb_control_is_the_hgb_cut_classifier():
    hgb = pytest.importorskip('pubg_lab.survival.ptr.models.hgb')
    own = T.make_model("hgb", 42, 2)
    assert own.get_params() == hgb.make_classifier(42).get_params()
    assert own.get_params()["early_stopping"] is False and own.get_params()["random_state"] == 42


def test_hgb_settings_must_equal_the_recorded_hgb_cut_run(tmp_path):
    import json

    pytest.importorskip('pubg_lab.survival.ptr.models.hgb')
    params = T.make_model("hgb", 42, 2).get_params()
    run = tmp_path / "toy" / "HGB-cut" / "seed42"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"classifier": params}))
    T._assert_hgb_settings(params, tmp_path, "toy", 42)  # identical: passes
    (run / "metrics.json").write_text(json.dumps({"classifier": {**params, "max_iter": 100}}))
    with pytest.raises(AssertionError):
        T._assert_hgb_settings(params, tmp_path, "toy", 42)
    (run / "metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        T._assert_hgb_settings(params, tmp_path, "toy", 42)
