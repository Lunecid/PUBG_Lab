"""v2 consumers on REAL matches (read-only database + the real v2 dump): prefix replay and agreement with the producer.

Needs ``PG_PASSWORD`` (``set -a; source /Users/lunecid/project/pubg_collector/.env; set +a``; the tests skip without it) and the real
dump ``model_stgcn/dumps/bigcomp2027_phase_v2`` (skipped without it; the synthetic equivalents live in ``test_v2_consumers_*.py``).
Matches: the first two test-split matches of each of the 12 cohorts (24), the two matches of the corpus that contain a comeback row
(Erangel TPP squad val, Taego TPP squad train) and the one match without a ``team_won`` flag (b5134321): 27 matches, 155 cuts, of which
106 are replayed (up to 4 per match: bounds the run time).

For every match and cut ``a_k = b_k - 10 s`` (the producer's exact cut):

* KRAFTON: the causal squad status equals the dump's ``node_mask`` at every cut; the 55 features rebuilt from the records strictly
  before ``a_k`` are BIT-identical to the full-data build; the sample rows / key / risk flags / T are the dump's;
* kf table: every row (features, flags) rebuilt from the truncated records equals the full-data row; tokens = the dump's ``query_mask``;
* Sagong: the eight indicators over ``[s_k, a_k)`` rebuilt from the truncated records are bit-identical to the full-data build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import requires_db

KT = Path(__file__).resolve().parents[1] / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))

torch = pytest.importorskip("torch")

DUMP = Path(__file__).resolve().parents[1] / 'model_stgcn' / "dumps" / "bigcomp2027_phase_v2"
N_PER_COHORT = 2
MAX_CUTS = 4  # cuts replayed per match (the first, the middle ones, the last): bounds the run time
needs_dump = pytest.mark.skipif(not (DUMP / "population.json").is_file(), reason="the v2 dump is not on disk")


def _cohorts():
    from pubg_lab.survival.ptr.data.cohorts import discover_cohorts

    return discover_cohorts(DUMP)


def _comeback_matches():
    """The real matches that contain a comeback row (found through the producer's manifests)."""
    out = []
    import pubg_lab.survival.ptr.data.data as D

    for c in _cohorts():
        man = json.loads((DUMP / c / "dump_manifest.json").read_text())
        for split, info in man["splits"].items():
            if info["stats"]["comeback_rows"]:
                d = D.load_dump(DUMP / c / f"{split}.pt")
                for m in d["matches"]:
                    if D.comeback_mask(m).any():
                        out.append((c, split, m))
    return out


NO_WINNER_MATCH = ("erangel_tpp_squad", "val", "b5134321-aeca-459e-ab84-508e8e87dc16")  # the corpus' one match without a team_won flag


def _no_winner_match():
    import pubg_lab.survival.ptr.data.data as D

    c, split, mid = NO_WINNER_MATCH
    d = D.load_dump(DUMP / c / f"{split}.pt")
    return [(c, split, m) for m in d["matches"] if str(m["match_id"]) == mid]


@pytest.fixture(scope="module")
def real_matches():
    import pubg_lab.survival.ptr.data.data as D

    picked = []
    for c in _cohorts():
        d = D.load_dump(DUMP / c / "test.pt")
        picked += [(c, "test", m) for m in d["matches"][:N_PER_COHORT]]
        del d
    picked += _comeback_matches()
    picked += _no_winner_match()
    seen, uniq = set(), []
    for item in picked:  # a match may appear twice (it can be one of the first two test matches)
        if item[2]["match_id"] not in seen:
            seen.add(item[2]["match_id"])
            uniq.append(item)
    assert len(uniq) >= 20
    return uniq


def _cut_slots(lab, limit=MAX_CUTS):
    slots = [p for p in range(8) if np.isfinite(lab["cut_t"][p]) and lab["query_mask"][p].any()]
    if len(slots) <= limit:
        return slots
    pick = np.unique(np.linspace(0, len(slots) - 1, limit).round().astype(int))
    return [slots[i] for i in pick]


@pytest.fixture(scope="module")
def krafton_inputs(real_matches):
    import projects.krafton_baseline.data.extract as E
    from projects.krafton_baseline.data.labels import compact_match

    out = {}
    for c, split, m in real_matches:
        mid = str(m["match_id"])
        lab = compact_match(m)
        T = E.fetch_batch([mid])
        meta = T["meta"].set_index("match_id")
        TT = {k: E._by_match(v) for k, v in T.items() if k != "meta"}
        row = meta.loc[mid]
        raw, extra = E.build_raw(mid, lab, TT, row, float(row.started_ep))
        out[mid] = (c, lab, raw, extra)
    return out


@needs_dump
def test_the_real_dump_loads_and_the_comeback_rows_are_there(real_matches):
    import pubg_lab.survival.ptr.data.data as D

    cb = [(c, s, m) for c, s, m in real_matches if D.comeback_mask(m).any()]
    assert len(cb) == 2  # the two comeback matches of the corpus (producer's run_summary: 2 comeback rows)
    for _c, _s, m in cb:
        assert D.dump_version({"version": "v2"}) == "v2" and bool((m["query_mask"] & ~m["node_mask"]).any())
        assert (D.effective_alive(m, "v2", False) == m["query_mask"]).all() and (D.key_mask(m) == m["node_mask"]).all()


@requires_db
@needs_dump
def test_krafton_causal_alive_equals_the_dump_node_mask_at_every_cut(krafton_inputs):
    from projects.krafton_baseline.data.features import team_alive_causal

    n_cells = 0
    for mid, (c, lab, raw, extra) in krafton_inputs.items():
        slots = [p for p in range(8) if np.isfinite(lab["cut_t"][p])]
        got = team_alive_causal(raw, lab["cut_t"][slots])
        want = lab["node_mask"][slots].T
        assert np.array_equal(got, want), f"{c}/{mid}: causal status != node_mask at cuts {slots}"
        n_cells += want.size
    assert n_cells > 1000


@requires_db
@needs_dump
def test_krafton_features_replay_bit_for_bit_from_the_records_before_each_cut(krafton_inputs):
    import projects.krafton_baseline.data.extract as E
    from projects.krafton_baseline.data.features import compute_features

    n = 0
    for mid, (c, lab, raw, extra) in krafton_inputs.items():
        slots = _cut_slots(lab)
        ts = lab["cut_t"][slots]
        full = compute_features(raw, ts)
        for k, t in enumerate(ts):
            part = compute_features(E.truncate_raw(raw, float(t)), np.array([t]))
            assert np.array_equal(full["F"][:, k], part["F"][:, 0]), f"{c}/{mid}: features differ at cut {slots[k] + 1}"
            assert np.array_equal(full["POS"][:, k], part["POS"][:, 0], equal_nan=True)
            assert np.array_equal(full["Zt"][k], part["Zt"][0]) and np.array_equal(full["member_alive"][:, k], part["member_alive"][:, 0])
            n += 1
    assert n >= 60


@requires_db
@needs_dump
def test_krafton_samples_are_the_dump_rows_flags_and_T(krafton_inputs):
    import projects.krafton_baseline.data.extract as E

    for mid, (c, lab, raw, extra) in krafton_inputs.items():
        for split in ("val", "train"):
            res, dropped = E.match_samples(mid, lab, raw, extra, split, "every30" if split == "train" else "krafton192")
            if res is None:
                continue
            off = np.concatenate([[0], np.cumsum(res["S_n"])])
            T = lab["team_death_s"]
            t_win = max(2000.0, extra["end_w"] or 0.0, float(T[np.isfinite(T)].max()) + 1.0)
            for i in range(len(res["S_t"])):
                sl = slice(off[i], off[i + 1])
                tid = res["TIDX"][sl].astype(int)
                t = float(res["S_t"][i]) if split == "val" else None
                assert np.array_equal(res["TGT"][sl], np.where(np.isinf(T[tid]), t_win, T[tid]).astype(np.float32))  # the dump's T
                if split == "val":
                    p = int(res["S_cut"][i]) - 1
                    assert np.array_equal(tid, np.flatnonzero(lab["query_mask"][p]))
                    assert np.array_equal(res["KEY"][sl], lab["node_mask"][p, tid])
                    assert np.array_equal(res["RISK"][sl], T[tid] > lab["cut_t"][p])
                    assert (res["RISK"][sl] | ~lab["target"][p, tid]).all()  # every risk-set row is a Cox row
                    assert np.isclose(t, lab["cut_t"][p], rtol=0, atol=1e-3)
                else:  # train times (every 30 s) are exactly representable: keys are the causal status, risk rows T > t
                    assert np.array_equal(res["RISK"][sl], T[tid] > float(res["S_t"][i]))


@requires_db
@needs_dump
def test_kf_table_rows_replay_exactly_and_are_the_dump_tokens(krafton_inputs):
    import projects.krafton_baseline.data.extract as E
    import projects.krafton_baseline.data.kftable as KF

    for mid, (c, lab, raw, extra) in krafton_inputs.items():
        slots = [p for p in _cut_slots(lab) if p < 7]  # the table covers the cut phases 1..7 (slot 7 has tokens but no cut label)
        sub = dict(lab)
        for key in ("node_mask", "query_mask", "target"):
            keep = np.zeros_like(lab[key])
            keep[slots] = lab[key][slots]
            sub[key] = keep
        full = KF.table_rows(mid, sub, raw, extra)
        assert full["n"] == int(sub["query_mask"].sum()) and full["n_risk"] == int(sub["target"].sum())
        assert np.array_equal(full["alive_at_cut"].astype(bool), np.array(
            [lab["node_mask"][ph - 1, list(lab["team_ids"]).index(t)] for ph, t in zip(full["phase"], full["team_id"])]))  # fmt: skip
        for p in slots:
            one = dict(lab)
            for key in ("node_mask", "query_mask", "target"):
                keep = np.zeros_like(lab[key])
                keep[p] = lab[key][p]
                one[key] = keep
            part = KF.table_rows(mid, one, E.truncate_raw(raw, float(lab["cut_t"][p])), extra)
            sel = full["phase"] == p + 1
            assert part["n"] == int(sel.sum()), f"{c}/{mid} cut {p + 1}"
            for key in ("F", "team_id", "in_risk_set", "alive_at_cut", "cut_t"):
                assert np.array_equal(part[key], full[key][sel]), f"{c}/{mid} cut {p + 1}: {key} differs after truncation"
            assert np.array_equal(part["y"], full["y"][sel], equal_nan=True)


@pytest.fixture(scope="module")
def sagong_raws(real_matches):
    from pubg_lab.survival.baselines.sagong2025.db import connect_readonly, fetch_raw_batch

    conn = connect_readonly()
    try:
        cur = conn.cursor()
        ids = [str(m["match_id"]) for _, _, m in real_matches]
        raws = {}
        for i in range(0, len(ids), 6):
            raws.update(fetch_raw_batch(cur, ids[i : i + 6], clock="w"))
        cur.close()
    finally:
        conn.close()
    return raws


@requires_db
@needs_dump
def test_sagong_indicators_replay_bit_for_bit_from_the_records_before_each_cut(real_matches, sagong_raws):
    import pubg_lab.survival.ptr.data.data as D
    import pubg_lab.survival.baselines.sagong2025.features as F

    def trunc(raw, cut):
        f = lambda df: df[df["ep"] < cut].reset_index(drop=True)  # noqa: E731
        return F.MatchRaw(raw.match_id, raw.start_epoch, raw.team_of, f(raw.pos), f(raw.dmg), f(raw.kill), f(raw.pick), raw.official)

    n = 0
    for c, split, m in real_matches:
        mid = str(m["match_id"])
        raw = sagong_raws[mid]
        assert raw.start_epoch == 0.0  # the database computed W
        start, cut = D.boundary_times(m["phase_start_s"]), D.cut_times(m["phase_end_s"])
        valid = [p for p in range(7) if bool(m["cut_valid"][p])]
        win = {p + 1: (float(start[p]), float(cut[p])) for p in valid}
        full = F.compute_match_features(raw, win, end_is_cut=True)
        for p in _slice(valid, MAX_CUTS):
            part = F.compute_match_features(trunc(raw, win[p + 1][1]), {p + 1: win[p + 1]}, end_is_cut=True)
            want = full[full.phase == p + 1].reset_index(drop=True)
            pd.testing.assert_frame_equal(part.reset_index(drop=True), want, check_exact=True)
            n += 1
    assert n >= 60


def _slice(items, limit):
    if len(items) <= limit:
        return items
    pick = np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))
    return [items[i] for i in pick]


@requires_db
@needs_dump
def test_sagong_window_rows_include_the_records_before_the_cut_and_nothing_else(real_matches, sagong_raws):
    """The indicators use records in ``[s_k, a_k)``: re-computed by hand from the raw frames for the number of enemy kills."""
    import pubg_lab.survival.ptr.data.data as D
    import pubg_lab.survival.baselines.sagong2025.features as F

    checked = 0
    for c, split, m in real_matches[:12]:
        raw = sagong_raws[str(m["match_id"])]
        start, cut = D.boundary_times(m["phase_start_s"]), D.cut_times(m["phase_end_s"])
        p = 1
        if not bool(m["cut_valid"][p]):
            continue
        s, a = float(start[p]), float(cut[p])
        df = F.compute_match_features(raw, {p + 1: (s, a)}, end_is_cut=True)
        kl = raw.kill
        team = raw.team_of
        ok = kl["killer"].isin(team) & kl["victim"].isin(team) & ~kl["is_suicide"].fillna(False).astype(bool) & (kl["ep"] >= s) & (kl["ep"] < a)
        k = kl[ok]
        k = k[k["killer"].map(team).to_numpy() != k["victim"].map(team).to_numpy()]
        by_team = k["killer"].map(team).value_counts()
        for t, n in by_team.items():
            assert int(df[df.team_id == t].kill.iloc[0]) == int(n)
        assert int(df.kill.sum()) == int(by_team.sum())
        checked += 1
    assert checked >= 6


@needs_dump
def test_the_match_without_a_winner_flag_keeps_its_rows_and_is_censored_by_the_alive_rule(real_matches):
    """b5134321: no ``team_won`` team; its rank-0 team is alive at the end and censored (T = +inf) by the alive rule.  The loader must
    return the stored rows (a winner is never re-added) and every consumer reads T / masks from the record."""
    import pubg_lab.survival.ptr.data.data as D

    ms = [m for _, _, m in real_matches if str(m["match_id"]) == NO_WINNER_MATCH[2]]
    assert len(ms) == 1
    m = ms[0]
    assert not bool(m["team_won"].any())
    y, target = D.select_labels(m, "v2")
    assert torch.equal(target, m["target_mask_v2"]) and torch.equal(y, m["y_hazard_v2"] * m["target_mask_v2"])
    cens = torch.isposinf(m["team_death_s"])
    assert cens.sum() >= 1  # the team(s) alive at the end
    assert not (y[:, cens] > 0).any()  # censored teams are never positive
    valid = m["cut_valid"].bool()
    assert target[:, cens][valid].all()  # ... and are at risk at every valid cut (T > b_k for all k)
    assert D.winner_censor_summary({"meta": {"version": "v2"}, "matches": [m]})["winner_teams"] == 0
