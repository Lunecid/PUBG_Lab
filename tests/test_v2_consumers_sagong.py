"""v2 consumers, part 3: the Sagong et al. (2025) re-implementation on the v2 dump (DB-free; parquet tests need pyarrow, run them
with ``/usr/local/bin/python3``).

* indicators: the window is ``[s_k, a_k)``, strictly before the producer's cut; rebuilt from the records before the cut they are
  bit-identical to the full-data build (prefix replay, on the producer's own synthetic match);
* plan: the units / windows follow the v2 dump (``query_mask`` rows of the label-valid cuts, ``alive_cut`` = ``node_mask``, exact ``A``);
  nothing is derived from death times, comeback rows are units, excluded cuts are absent;
* fit: types and the Eq. 4 survival-rate table are fitted on the TRAIN split from the causal alive status; v1 / v2 caches are never mixed.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.baselines.sagong2025.features as F
import pubg_lab.survival.baselines.sagong2025.fit as T
from tests.v2_fixtures import sagong_raw_from_source, synth_record, write_v2_cohort


def _truncate(raw: F.MatchRaw, cut: float) -> F.MatchRaw:
    def f(df):
        return df[df["ep"] < cut].reset_index(drop=True)

    return F.MatchRaw(raw.match_id, raw.start_epoch, raw.team_of, f(raw.pos), f(raw.dmg), f(raw.kill), f(raw.pick), raw.official)


@pytest.fixture(scope="module")
def synth():
    src, rec = synth_record()
    return src, rec, sagong_raw_from_source(src)


def _windows(rec, ks=(1, 2, 3)):
    s, a = rec["phase_start_s"].double().numpy(), rec["cut_t"].double().numpy()
    return {k: (float(s[k - 1]), float(a[k - 1])) for k in ks}


# --------------------------------------------------------------------------- indicators
def test_v2_window_is_strictly_before_the_cut_and_v1_default_is_unchanged():
    t0 = 1000.0
    pos = pd.DataFrame({"acc": ["p1", "p1", "p1"], "ep": [t0 + 10, t0 + 20, t0 + 30], "x": [0.0, 3.0, 7.0], "y": [0.0, 4.0, 4.0]})
    empty = lambda cols: pd.DataFrame({c: [] for c in cols})  # noqa: E731
    kill = pd.DataFrame({"killer": ["p1"], "victim": ["p2"], "is_suicide": [False], "ep": [t0 + 30]})  # exactly at the cut
    dmg = pd.DataFrame({"attacker": ["p1", "p1"], "victim": ["p2", "p2"], "damage": [10.0, 99.0], "ep": [t0 + 25, t0 + 30]})
    raw = F.MatchRaw("m", 0.0, {"p1": 1, "p2": 2}, pos, dmg, kill, empty(["acc", "event_type", "ep"]))
    row = lambda df: df[(df.team_id == 1) & (df.phase == 1)].iloc[0]  # noqa: E731
    v2 = row(F.compute_match_features(raw, {1: (t0, t0 + 30)}, end_is_cut=True))
    assert (v2.attack, v2.damage, v2.kill) == (1, 10.0, 0)  # the damage / kill / position stamped exactly at the cut are not read
    assert v2.distance == pytest.approx(5.0)  # step 2 ends at t0 + 20, step 3 would end at the cut
    v1 = row(F.compute_match_features(raw, {1: (t0, t0 + 40)}))  # v1: [s, e - 10] inclusive -> the same cut, upper end inclusive
    assert (v1.attack, v1.damage, v1.kill) == (2, 109.0, 1) and v1.distance == pytest.approx(9.0)


def test_indicators_are_bit_identical_when_rebuilt_from_the_records_before_each_cut(synth):
    src, rec, raw = synth
    win = _windows(rec)
    full = F.compute_match_features(raw, win, end_is_cut=True)
    assert len(full) == 3 * src.n_teams and (full[list(F.FEATURES)].to_numpy() > 0).any()
    for k, (s, a) in win.items():
        part = F.compute_match_features(_truncate(raw, a), {k: (s, a)}, end_is_cut=True)
        want = full[full.phase == k].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), want, check_exact=True)


def test_late_records_cannot_change_indicators_but_early_ones_do(synth):
    src, rec, raw = synth
    win = _windows(rec, (1,))
    a = win[1][1]
    base = F.compute_match_features(raw, win, end_is_cut=True)
    late = F.MatchRaw(raw.match_id, 0.0, raw.team_of, raw.pos, pd.concat([raw.dmg, pd.DataFrame({"attacker": ["a1"], "victim": ["b1"], "damage": [77.0], "ep": [a + 0.5]})],
                                                                         ignore_index=True), raw.kill, raw.pick)  # fmt: skip
    pd.testing.assert_frame_equal(F.compute_match_features(late, win, end_is_cut=True), base, check_exact=True)
    early = F.MatchRaw(raw.match_id, 0.0, raw.team_of, raw.pos, pd.concat([raw.dmg, pd.DataFrame({"attacker": ["a1"], "victim": ["b1"], "damage": [77.0], "ep": [a - 0.5]})],
                                                                          ignore_index=True), raw.kill, raw.pick)  # fmt: skip
    row = lambda df: df[df.team_id == 11].iloc[0]  # noqa: E731
    assert row(F.compute_match_features(early, win, end_is_cut=True)).damage == row(base).damage + 77.0


def test_db_clock_option_is_validated_and_sql_computes_w():
    import pubg_lab.survival.baselines.sagong2025.db as db

    assert "m.started_at" in db.sql_pos("w") and "m.started_at" not in db.sql_pos("epoch")
    assert db.SQL_POS == db.sql_pos("epoch") and "JOIN stdb.matches" in db.sql_kill("w")
    with pytest.raises(ValueError):
        db.sql_pos("game")


# --------------------------------------------------------------------------- Eq. 4 / types with the causal alive column
def test_survival_table_uses_the_causal_alive_column_and_ignores_comeback_rows_as_present():
    def u(match, team, phase, ptype, alive=True):
        return {"match_id": match, "team_id": team, "phase": phase, "alive_cut": alive, "ptype": ptype}

    units = pd.DataFrame([
        u("m1", "A", 1, 1), u("m1", "B", 1, 1), u("m1", "C", 1, 4, alive=False),  # C: a comeback unit, causally dead at cut 1
        u("m1", "A", 2, 1), u("m1", "C", 2, 4),  # C is back (alive at cut 2); B died
    ])
    exists = {("m1", 1), ("m1", 2)}
    table, info = T.fit_survival_table_paper(units, exists, n_cuts=2)
    assert T.alive_column(units) == "alive_cut"
    assert table[(2, 1)] == 0.5  # A survives, B does not
    assert info["n_by_cell"]["2:4"] == 0  # C was not alive at cut 1, so it is not 'present in phase 1': it cannot be a survivor either


# --------------------------------------------------------------------------- plan / fit on a v2 dump (parquet)
@pytest.fixture()
def v2_plan(tmp_path):
    pytest.importorskip("pyarrow")
    from pubg_lab.survival.baselines.sagong2025.plan import build_plan

    base = write_v2_cohort(tmp_path / "dumps", "c", n=14, seed=5, comeback_prob=0.1, exclude_prob=0.25)
    audit = build_plan(base.parent, "c", tmp_path / "plan")
    return tmp_path, base.parent, audit


def test_plan_units_and_windows_follow_the_dump(v2_plan):
    tmp, dump_dir, audit = v2_plan
    assert audit["version"] == "v2" and audit["meta"]["version"] == "v2" and "causal" in audit["alive_status"]
    splits = D.load_cohort(dump_dir, "c")
    units = pd.read_parquet(tmp / "plan" / "c" / "units.parquet")
    wins = pd.read_parquet(tmp / "plan" / "c" / "windows.parquet")
    assert "alive_cut" in units.columns and "alive_start" not in units.columns and "A" in wins.columns
    want_rows = set()
    n_cb = 0
    for s, d in splits.items():
        for m in d["matches"]:
            valid = m["cut_valid"].numpy()
            for p in range(7):
                if not valid[p]:
                    continue
                for i in np.flatnonzero(m["query_mask"][p].numpy()):
                    want_rows.add((s, str(m["match_id"]), int(m["team_ids"][i]), p + 1))
                n_cb += int((m["target_mask_v2"][p] & ~m["node_mask"][p]).sum())
    got = set(zip(units["split"], units["match_id"], units["team_id"], units["phase"]))
    assert got == want_rows and len(units) == len(want_rows)
    assert int((units["target"] & ~units["alive_cut"]).sum()) == n_cb > 0  # the comeback rows are units
    # alive_cut = node_mask, target = target_mask_v2, y = y_hazard_v2 (read straight from the records)
    by = {(s, str(m["match_id"])): m for s, d in splits.items() for m in d["matches"]}
    for r in units.sample(200, random_state=0).itertuples():
        m = by[(r.split, r.match_id)]
        i = int(np.flatnonzero(m["team_ids"].numpy() == r.team_id)[0])
        assert bool(m["node_mask"][r.phase - 1, i]) == bool(r.alive_cut) and bool(m["target_mask_v2"][r.phase - 1, i]) == bool(r.target)
        assert float(m["y_hazard_v2"][r.phase - 1, i]) == (r.y if r.target else 0.0)
    # exact windows: [S, A) with A = b_k - 10, only the label-valid cuts (no window for an excluded cut)
    assert np.allclose(wins["A"], wins["E"] - 10.0, rtol=0, atol=0)
    n_excluded = sum(len(m["cut_excluded"]) for d in splits.values() for m in d["matches"])
    assert n_excluded > 0
    seen = set(zip(wins["split"], wins["match_id"], wins["phase"]))
    for s, d in splits.items():
        for m in d["matches"]:
            for k in m["cut_excluded"]:
                assert (s, str(m["match_id"]), k) not in seen
    m = splits["train"]["matches"][0]
    r = wins[(wins.split == "train") & (wins.match_id == str(m["match_id"])) & (wins.phase == 1)]
    if len(r):
        assert float(r["S"].iloc[0]) == float(m["phase_start_s"][0]) and float(r["E"].iloc[0]) == float(m["phase_end_s"][0])


def _fake_features(plan_dir, cohort="c", version="v2", sha=None):
    units = pd.read_parquet(plan_dir / cohort / "units.parquet")
    rng = np.random.default_rng(0)
    keys = units[["match_id", "team_id", "phase"]].drop_duplicates().reset_index(drop=True)
    feats = keys.copy()
    for f in F.FEATURES:
        feats[f] = rng.random(len(keys)) * 10
    feats["n_members"] = 4.0
    feats["n_obs"] = 4.0
    feats.to_parquet(plan_dir / cohort / "features.parquet", index=False)
    audit = json.loads((plan_dir / cohort / "plan_audit.json").read_text())
    (plan_dir / cohort / "features_meta.json").write_text(json.dumps({"version": version, "windows_sha1": sha or audit["windows_sha1"],
        "feature_source_sha256": hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()}))


def test_stale_feature_source_is_refused_by_extract_and_fit(v2_plan):
    import pubg_lab.survival.baselines.sagong2025.extract as X
    tmp, _, _ = v2_plan
    base = tmp / "plan" / "c"
    _fake_features(tmp / "plan")
    meta = base / "features_meta.json"
    record = json.loads(meta.read_text())
    wins = pd.read_parquet(base / "windows.parquet")
    X.check_features_meta(base, X._plan_info(base, wins))
    for bad in ("obsolete-code", None):
        record["feature_source_sha256"] = bad
        meta.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="feature source fingerprint"):
            X.check_features_meta(base, X._plan_info(base, wins))
        with pytest.raises(ValueError, match="feature source fingerprint"):
            T.prepare(tmp / "plan", "c")


def test_prepare_fits_on_train_with_the_causal_alive_column(v2_plan):
    tmp, dump_dir, audit = v2_plan
    _fake_features(tmp / "plan")
    df, prep = T.prepare(tmp / "plan", "c")
    info = prep["info"]
    assert info["dump_version"] == "v2" and info["alive_column"] == "alive_cut" and "causal" in info["alive_status"]
    # phase means: train units that are causally alive at the cut only (comeback / dead rows do not enter)
    tr = df[(df.split == "train") & df.alive_cut]
    for k, v in info["phase_means"].items():
        assert v["fight"] == pytest.approx(float(tr[tr.phase == k].fight.mean()))
    # the table is a train-only statistic: scrambling val / test labels cannot change it
    d2 = pd.read_parquet(tmp / "plan" / "c" / "units.parquet")
    d2.loc[d2.split != "train", "y"] = 1.0
    d2.to_parquet(tmp / "plan" / "c" / "units.parquet", index=False)
    _, prep2 = T.prepare(tmp / "plan", "c")
    assert prep2["info"]["survival_rate_table"] == info["survival_rate_table"]


def test_fit_runs_on_v2_and_records_the_version(v2_plan):
    tmp, dump_dir, audit = v2_plan
    _fake_features(tmp / "plan")
    res = T.run_cohort(tmp / "plan", "c", tmp / "runs", seed=42, n_jobs=1, models=("hgb",))
    r = res["SAGONG-hgb"]
    assert r["dump_version"] == "v2" and r["sagong"]["dump_version"] == "v2"
    units = pd.read_parquet(tmp / "plan" / "c" / "units.parquet")
    import pubg_lab.survival.ptr.evaluation.metrics as M

    for split in ("val", "test"):
        pred = M.read_predictions(tmp / "runs" / "c" / "SAGONG-hgb" / "seed42" / f"pred_{split}.csv")
        ref = units[(units.split == split) & units.target]
        assert len(pred) == len(ref)  # the PTR rows: every risk-set row of the dump, the comeback rows included


def test_v1_and_v2_caches_are_never_mixed(v2_plan):
    tmp, dump_dir, audit = v2_plan
    _fake_features(tmp / "plan", version="v1")  # features extracted for a v1 plan
    with pytest.raises(D.DumpVersionError, match="Sagong features"):
        T.prepare(tmp / "plan", "c")
    _fake_features(tmp / "plan", version="v2", sha="0" * 40)  # other windows
    with pytest.raises(ValueError, match="other windows"):
        T.prepare(tmp / "plan", "c")
    # the extract stage refuses to skip a cache of another version
    import pubg_lab.survival.baselines.sagong2025.extract as X

    base = tmp / "plan" / "c"
    wins = pd.read_parquet(base / "windows.parquet")
    with pytest.raises(D.DumpVersionError):
        X.check_features_meta(base, {**X._plan_info(base, wins), "version": "v1"})
    # a habitat table of another dump version
    from pubg_lab.survival.baselines.sagong2025.hab import AUDIT_FILE, HAB_FILE, attach_hab

    pd.DataFrame({"split": [], "match_id": [], "team_id": [], "phase": [], "y": []}).to_parquet(base / HAB_FILE, index=False)
    (base / AUDIT_FILE).write_text(json.dumps({"names": [], "indices": [], "dump_fingerprint": {}}))  # no dump_version = v1
    with pytest.raises(D.DumpVersionError, match="habitat table"):
        attach_hab(tmp / "plan", "c", {}, {}, version="v2")


def test_a_v1_plan_still_works_with_the_alive_start_column(tmp_path):
    """The existing v1 caches carry ``alive_start`` only: the fit code reads them unchanged."""
    pytest.importorskip("pyarrow")
    units = pd.DataFrame({"match_id": ["m"] * 4, "team_id": [1, 2, 1, 2], "phase": [1, 1, 2, 2], "alive_start": [True, True, True, False],
                          "ptype": [1, 4, 1, 4]})  # fmt: skip
    assert T.alive_column(units) == "alive_start"
    table, _ = T.fit_survival_table_paper(units, {("m", 1), ("m", 2)}, n_cuts=2)
    assert table[(2, 1)] == 1.0 and table[(2, 4)] == 0.0
