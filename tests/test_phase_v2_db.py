"""DB-gated tests of the phase-tier dump v2 on real matches (read-only SELECTs; set PG_PASSWORD).

* prefix replay: >= 50 (default 204 = 17 per cohort x 12 cohorts) stratified matches, every cut: the inputs rebuilt from a
  source that the DATABASE truncated at a_k are EXACTLY the inputs built from the full source (the key leak test);
* labels / risk set / censoring / terminal interval / exclusions against an independent SQL + brute-force derivation
  (``tools/verify_dump_match_v2.py``) on stratified matches, incl. real revive / redeploy cases and matches whose API
  duration exceeds 2000 s;
* knocked is not eliminated on real groggy events;
* hand-check of three real matches (a redeploy case, a long-API-duration match, a typical one).

Environment: ``PHASE_V2_REPLAY_PER_COHORT`` (default 17; the paper-citable run uses the default, 50 matches is the minimum).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from model_stgcn.phase_v2 import alive as AL
from model_stgcn.phase_v2 import assemble as AS
from model_stgcn.phase_v2 import replay as RP
from model_stgcn.phase_v2.phases import phases_of
from model_stgcn.phase_v2.source import fetch_match_source
from tests.conftest import requires_db
from tools.verify_dump_match_v2 import print_report, verify_record

PER_COHORT = int(os.environ.get("PHASE_V2_REPLAY_PER_COHORT", "17"))
RESULT_JSON = Path(os.environ.get("PHASE_V2_REPLAY_JSON", "")) if os.environ.get("PHASE_V2_REPLAY_JSON") else None


@pytest.fixture(scope="module")
def cur():
    from stdb.conn import connect

    conn = connect()
    conn.set_session(readonly=True, autocommit=True)
    c = conn.cursor()
    yield c
    conn.close()


def _record(cur, mid):
    src = fetch_match_source(cur, mid, schedule=RP.DEFAULT_SCHEDULE, schedule_source="test")
    return src, AS.build_match_v2(src, map_size=RP.MAP_SIZE[src.map_name])


@requires_db
def test_prefix_replay_on_stratified_real_matches(cur):
    ms = RP.stratified_matches(cur, PER_COHORT)
    assert len({c for c, _ in ms}) == 12 and len(ms) >= 50
    res = RP.run_prefix_replay(cur, ms)
    if RESULT_JSON is not None:
        RESULT_JSON.write_text(json.dumps({k: v for k, v in res.items()}, indent=2) + "\n", encoding="utf-8")
    print(f"\nprefix replay: N={res['n_matches']} matches, {res['n_cuts']} cuts, runtime {res['runtime_s']} s, "
          f"mismatches {len(res['mismatches'])}, mutation {res['mutation_check']}")
    assert res["n_matches"] >= 50 and res["n_cuts"] >= 7 * 40
    assert res["mismatches"] == [], res["mismatches"][:3]
    # the database really dropped records, and the last kept record always precedes the cut
    assert res["rows_seen_truncated"] < res["rows_seen_full"]
    assert res["min_gap_between_last_kept_record_and_cut_s"] > 0
    # a builder that forgets to cut the event tables is detected on every compared cut
    m = res["mutation_check"]
    assert m["cuts_compared"] > 0 and m["cuts_detected_as_leaky"] == m["cuts_compared"]


@requires_db
def test_labels_against_the_independent_sql_derivation_on_stratified_matches(cur):
    ms = RP.stratified_matches(cur, 3, tag="labels")
    bad = []
    n_rows = n_pos = n_marker = 0
    for slug, mid in ms:
        _src, rec = _record(cur, mid)
        for c in verify_record(cur, rec):
            if not c.ok:
                bad.append((slug, mid, c.name, c.detail))
        n_rows += int(rec["target_mask_v2"].sum())
        n_pos += int(((rec["y_hazard_v2"] > 0) & rec["target_mask_v2"]).sum())
    assert not bad, bad[:3]
    assert n_rows > 1000 and 0 < n_pos < n_rows


@requires_db
def test_real_revive_and_redeploy_example_follows_the_causal_rule(cur):
    found = 0
    for slug, mid in RP.stratified_matches(cur, 6, tag="revive"):
        src = fetch_match_source(cur, mid)
        tl = AL.timelines_from_source(src)
        ev = AL.revive_evidence(tl, src.pos, RP.MAP_SIZE[src.map_name])
        if not ev:
            continue
        e = ev[0]
        a = e["acct"]
        k, r = e["kill_w"], e["revive_w"]
        assert AL.player_alive_at(tl, a, k - 0.001)
        assert not AL.player_alive_at(tl, a, k)  # a kill at t counts
        assert not AL.player_alive_at(tl, a, 0.5 * (k + r))  # dead body / waiting: dead
        assert not AL.player_alive_at(tl, a, r - 1e-6)  # revive evidence is not observable before its time
        assert AL.player_alive_at(tl, a, r)  # first healthy sample after the kill
        assert e["gap_s"] > 20.0 and not e["too_soon"]  # never the dead body's own health-0 window
        assert e["z_m"] > 500.0  # redeploy re-enters from the air: not a ground arena
        found += 1
        if found >= 3:
            break
    assert found >= 1


@requires_db
def test_knocked_is_not_eliminated_on_real_groggy_events(cur):
    n = 0
    for slug, mid in RP.stratified_matches(cur, 4, tag="groggy"):
        src = fetch_match_source(cur, mid)
        tl = AL.timelines_from_source(src)
        g = src.groggy
        for w, v in zip(g["w"], g["victim"]):
            tlv = tl.get(int(v))
            if tlv is None or len(tlv.kills) == 0:
                continue
            later = tlv.kills[tlv.kills > w]
            if len(later) == 0 or later[0] - w < 5.0:
                continue
            t = float(w) + 2.0  # knocked at w, killed >= 5 s later: still alive at w + 2 s although down
            if np.any(tlv.kills <= t):
                continue
            assert AL.player_alive_at(tl, int(v), t)
            n += 1
        if n >= 20:
            break
    assert n >= 5  # real knocked-then-killed-later players exist and stay alive until the kill event


@requires_db
def test_matches_with_api_duration_over_2000_s_are_built_without_a_cap(cur):
    cur.execute(
        """SELECT m.match_id FROM stdb.matches m JOIN pubg.matches p ON p.match_id = m.match_id
            WHERE p.duration > 2000 AND p.match_type <> 'airoyale' AND m.map_name IN ('Baltic_Main', 'Tiger_Main')
              AND m.game_mode IN ('squad', 'squad-fpp', 'duo') ORDER BY md5(m.match_id) LIMIT 4"""
    )
    ids = [r[0] for r in cur.fetchall()]
    assert ids
    for mid in ids:
        src, rec = _record(cur, mid)
        assert np.isfinite(rec["X"].numpy()).all() and rec["cut_valid"].any()
        for c in verify_record(cur, rec):
            assert c.ok, (mid, c)
        # the real log is shorter than the API duration (no match of the corpus lasts 2000 s); v2 has no grid cap anyway
        assert src.end_w is not None and src.end_w > 0


# Real matches picked from the v2 dump run (the corpus is frozen): the only two risk-set rows that are causally dead at the cut
# (comeback / redeploy), the longest real log of the kept population, and a match whose API duration exceeds 2000 s.
COMEBACK_MATCHES = {
    "602d79af-02da-4ac0-8581-787fdca34231": (1, 14),  # Erangel TPP squad: cut 1, team 14 (T = 756.6 > b_1 = 600.3)
    "27ce37c2-0df6-452a-b61d-406ae6787034": (4, 2),  # Taego TPP squad: cut 4, team 2 (T = 1174.7 > b_4 = 1170.5)
}
LONGEST_MATCH = "dbffc7f9-fce6-4b09-a701-51cd9a4b6109"  # Erangel FPP squad (competitive), C_m = 1975.4 s


@requires_db
def test_hand_check_comeback_and_long_matches_end_to_end(cur, capsys):
    """Hand-check of real matches: two comeback / redeploy rows, the longest real log, a match whose API duration is
    > 2000 s: every independent SQL / brute-force check passes and the comeback rows keep their query token."""
    cur.execute(
        "SELECT m.match_id FROM stdb.matches m JOIN pubg.matches p ON p.match_id = m.match_id "
        "WHERE p.duration > 2000 AND p.match_type = 'official' AND m.game_mode = 'squad' ORDER BY md5(m.match_id) LIMIT 1"
    )
    api_long = cur.fetchone()[0]
    with capsys.disabled():
        for mid, comeback in {**COMEBACK_MATCHES, LONGEST_MATCH: None, api_long: None}.items():
            src, rec = _record(cur, mid)
            print(f"\n-- hand-check {mid[:8]} cuts={int(rec['cut_mask'].sum())} comeback_rows={rec['_stats']['comeback_rows']} "
                  f"C_m={float(rec['match_end_s']):.1f}s API-duration-long={mid == api_long}")
            assert print_report(mid, verify_record(cur, rec))
            if comeback is not None:
                k, tid = comeback
                i = list(rec["team_ids"].tolist()).index(tid)
                assert bool(rec["target_mask_v2"][k - 1, i]) and not bool(rec["node_mask"][k - 1, i])  # risk set, key off
                assert bool(rec["query_mask"][k - 1, i])  # query token stays
                assert rec["_stats"]["comeback_rows"] == 1
                assert float(rec["team_death_s"][i]) > float(rec["phase_end_s"][k - 1])  # T > b_k
                assert float(rec["X"][k - 1, i].abs().sum()) > 0  # genuine features, not a zeroed token
            else:
                assert rec["cut_valid"].any()
    assert float(_record(cur, LONGEST_MATCH)[1]["match_end_s"]) < 2000.0  # no real log reaches 2000 s


@requires_db
def test_conventions_hold_on_the_built_records(cur):
    for slug, mid in RP.stratified_matches(cur, 2, tag="conv"):
        src, rec = _record(cur, mid)
        ph = phases_of(src)
        assert (rec["query_mask"] == (rec["node_mask"] | rec["target_mask_v2"])).all()
        assert not rec["target_mask_v2"][7].any() and not rec["X"][~rec["cut_mask"]].any()
        for k in range(1, 8):
            if not ph.has_cut(k):
                assert not rec["target_mask_v2"][k - 1].any() and not rec["node_mask"][k - 1].any()
        assert (rec["target_mask_v2"] <= rec["cut_valid"][:, None]).all()
        assert np.isfinite(rec["X"].numpy()).all()
