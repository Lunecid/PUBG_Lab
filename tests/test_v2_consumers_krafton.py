"""v2 consumers, part 2: the KRAFTON chain (features, extract samples, kf table, K1 / K2 / K3 masks), DB-free.

Everything is checked against a REAL v2 record: the producer's own synthetic match (``tests/phase_v2_synth.py``: revive,
knocked-but-alive, a comeback team that is causally dead at the first cut yet at risk, a team that dies in the terminal
window) built by ``phase_v2.assemble.build_match_v2``.  The consumers see the same raw rows through ``tests/v2_fixtures``.

* the causal squad status (``features.team_alive_causal``) equals the producer's ``node_mask`` at every cut, and no future record
  can change it;
* features at a cut are bit-identical when rebuilt from the records strictly before the cut (prefix replay);
* samples: rows / key flags / risk flags / T come from the dump (task rules 1, 2, 5), comeback rows are rows without being keys;
* the kf table rows are the dump's tokens, the label-version guards refuse v1 / v2 mixes (rule 6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

KT = Path(__file__).resolve().parents[1] / "third_party" / "krafton_pubg_winprob"
if str(KT) not in sys.path:
    sys.path.insert(0, str(KT))

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from adaptation import data as KD  # noqa: E402
from adaptation import extract as E  # noqa: E402
from adaptation import kftable as KF  # noqa: E402
from adaptation import ourfeat as OF  # noqa: E402
from adaptation import ourfeat_hazard as H3  # noqa: E402
from adaptation import schema as S  # noqa: E402
from adaptation import train as T1  # noqa: E402
from adaptation.features import compute_features, member_alive_causal, team_alive_causal  # noqa: E402
from adaptation.labels import check_label_version, compact_match, label_version  # noqa: E402
from model_stgcn.ptr import data as D  # noqa: E402
from model_stgcn.ptr import extra as EX  # noqa: E402
from tests.v2_fixtures import krafton_raw_from_source, synth_record  # noqa: E402

CUT_SLOTS = (0, 1, 2)  # cuts 1..3 of the synthetic match: a = 590, 800, 980


@pytest.fixture(scope="module")
def match():
    src, rec = synth_record()
    lab = compact_match(rec)
    return src, rec, lab, krafton_raw_from_source(src, lab)


def _extra(lab, raw):
    return {"flags": {"first_state_w": float(raw["zone"]["t"][0])}, "end_w": lab["match_end_s"]}


def _dump(recs):
    from model_stgcn.phase_v2 import features as FT

    n = len(FT.FEATURE_NAMES)
    meta = {"feature_names": list(FT.FEATURE_NAMES), "feature_groups": {"zone": list(range(13))}, "version": "v2", "cohort": "synthetic",
            "split": "val", "zone_columns": list(D.ZONE_COLUMNS)}  # fmt: skip
    assert n == 150
    return {"meta": meta, "matches": recs}


# --------------------------------------------------------------------------- the compact label
def test_compact_match_v2_carries_the_dump_exactly(match):
    src, rec, lab, raw = match
    assert label_version(lab) == "v2" and "query_mask" in lab
    for key in ("node_mask", "query_mask"):
        assert np.array_equal(lab[key], rec[key].numpy())
    assert np.array_equal(lab["target"], rec["target_mask_v2"].numpy()) and np.array_equal(lab["y"], rec["y_hazard_v2"].numpy())
    assert np.array_equal(lab["team_death_s"], rec["team_death_s"].double().numpy(), equal_nan=True)  # T exactly as stored
    assert np.isposinf(lab["team_death_s"][lab["team_won"]]).all()  # winners censored by the dump
    assert np.array_equal(lab["cut_t"], rec["cut_t"].double().numpy(), equal_nan=True) and lab["match_end_s"] == rec["match_end_s"]
    assert np.array_equal(lab["phase_end_s"], rec["phase_end_s"].double().numpy(), equal_nan=True)


def test_label_version_guard_refuses_a_v1_table_next_to_a_v2_dump(match):
    _, _, lab, _ = match
    rec_v2 = {"cohort": "c", "split": "val", "dump_version": "v2", "matches": [lab]}
    rec_v1 = {"cohort": "c", "split": "val", "matches": [{k: v for k, v in lab.items() if k not in ("query_mask", "node_mask")}]}
    check_label_version(rec_v2, "v2", against="the dump")
    assert label_version(rec_v1) == "v1"
    with pytest.raises(D.DumpVersionError, match="derived from a v1 dump but the dump is a v2 dump"):
        check_label_version(rec_v1, "v2", against="the dump")
    with pytest.raises(D.DumpVersionError):
        check_label_version(rec_v2, "v1", against="a v1 dump")


# --------------------------------------------------------------------------- causal alive status
def test_causal_squad_status_equals_the_producer_node_mask_at_every_cut(match):
    _, rec, lab, raw = match
    cuts = lab["cut_t"][list(CUT_SLOTS)]
    assert np.array_equal(team_alive_causal(raw, cuts), lab["node_mask"][list(CUT_SLOTS)].T)
    # the comeback team (13, index 2) is dead at cut 1 yet in the risk set; the knocked team 14 (index 3) stays alive
    assert not lab["node_mask"][0, 2] and lab["target"][0, 2] and lab["node_mask"][0, 3]


def test_no_future_record_can_change_the_alive_status_or_the_features_at_a_cut():
    src0, rec0 = synth_record()
    lab0 = compact_match(rec0)
    raw0 = krafton_raw_from_source(src0, lab0)
    # a late kill of a1 (team 11) and of the winner's players (after the third cut a = 980): labels and T move, inputs must not
    src1, rec1 = synth_record(extra_kills=[(1000.0, "a1"), (995.0, "c1"), (1000.0, "c2")])
    lab1 = compact_match(rec1)
    raw1 = krafton_raw_from_source(src1, lab1)
    cuts = lab0["cut_t"][list(CUT_SLOTS)]
    assert np.array_equal(team_alive_causal(raw0, cuts), team_alive_causal(raw1, cuts))
    f0, f1 = compute_features(raw0, cuts), compute_features(raw1, cuts)
    assert np.array_equal(f0["F"], f1["F"]) and np.array_equal(f0["POS"], f1["POS"], equal_nan=True)
    # ... while the label side DOES move (the late kill of a1 changes T of team 11)
    assert lab0["team_death_s"][0] == 905.0 and lab1["team_death_s"][0] == 1000.0


def test_member_alive_matches_the_producer_player_timeline_semantics(match):
    """A kill at exactly the cut is invisible at the cut (records < a_k); a healthy sample at the cut does not revive."""
    from model_stgcn.phase_v2 import alive as AL

    kills_w = np.array([100.0, 300.0])
    victim = np.array([0, 0])
    pos_w = np.array([150.0, 400.0, 500.0])
    pos_acct = np.zeros(3, dtype=np.int64)
    pos_health = np.array([100.0, 100.0, 100.0])
    life = AL.build_timelines(kills_w, victim, pos_w, pos_acct, pos_health)
    ts = np.array([99.0, 100.0, 100.0 + 1e-9, 149.0, 150.0, 150.0 + 1e-9, 299.0, 300.0, 300.0 + 1e-9, 400.0, 400.0 + 1e-9])
    got = member_alive_causal(life, 1, ts)[0]
    # alive; alive (the kill at 100 is not < 100); dead; dead; dead (the revive at 150 is not < 150); alive; alive; alive (the
    # second kill at 300 is not < 300); dead (it is < 300 + 1e-9); dead (the revive at 400 is not < 400); alive again
    assert got.tolist() == [True, True, False, False, False, True, True, True, False, False, True]


# --------------------------------------------------------------------------- features: prefix replay, bit for bit
def test_features_are_bit_identical_when_rebuilt_from_the_records_before_the_cut(match):
    _, _, lab, raw = match
    cuts = list(lab["cut_t"][list(CUT_SLOTS)]) + [150.0, 333.3, 555.5, 700.0, 880.5, 1000.0]
    full = compute_features(raw, np.sort(np.asarray(cuts)))
    for k, t in enumerate(np.sort(np.asarray(cuts))):
        part = compute_features(E.truncate_raw(raw, float(t)), np.array([t]))
        assert np.array_equal(full["F"][:, k, :], part["F"][:, 0, :]), f"features differ at t={t}"
        assert np.array_equal(full["POS"][:, k], part["POS"][:, 0], equal_nan=True)
        assert np.array_equal(full["Zt"][k], part["Zt"][0]) and np.array_equal(full["member_alive"][:, k], part["member_alive"][:, 0])


def test_features_at_a_cut_do_not_read_a_record_exactly_at_the_cut(match):
    """Strict ``< t``: a damage event / position sample stamped exactly at the cut is not seen."""
    src, rec, lab, raw = match
    t = float(lab["cut_t"][0])
    raw2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    dm = dict(raw["damage"])
    for key in dm:  # one more damage event of player 0 exactly at the cut
        extra_val = {"attacker_pidx": 0, "victim_pidx": 2, "dmg": 50.0, "t": t, "wclass": "AR"}[key]
        dm[key] = np.concatenate([dm[key], np.array([extra_val], dtype=dm[key].dtype)])
    raw2["damage"] = dm
    a, b = compute_features(raw, np.array([t])), compute_features(raw2, np.array([t]))
    assert np.array_equal(a["F"], b["F"])
    c = compute_features(raw2, np.array([t + 1e-6]))  # one microsecond later it is read
    assert not np.array_equal(a["F"], c["F"])


def test_position_zone_and_kill_records_at_the_cut_are_not_read_either(match):
    """Every read is strict: a position sample / game state / kill stamped exactly at the cut changes nothing at the cut."""
    src, rec, lab, raw = match
    t = float(lab["cut_t"][1])  # 800.0
    base = compute_features(raw, np.array([t]))
    alive0 = team_alive_causal(raw, np.array([t]))

    def with_rows(group, **rows):
        r2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
        g = dict(raw[group])
        n = len(next(iter(rows.values())))
        for key in g:
            g[key] = np.concatenate([g[key], np.asarray(rows[key], dtype=g[key].dtype)]) if key in rows else np.concatenate(
                [g[key], np.zeros(n, dtype=g[key].dtype)])  # fmt: skip
        r2[group] = g
        return r2

    # a healthy position sample of player 0 at the cut far away from everything, a game state at the cut with another circle
    pos = with_rows("pos", pidx=[0], t=[t], health=[100.0], x=[9e5], y=[9e5], z=[0.0], veh=[False])
    zone = with_rows("zone", t=[t], sx=[1.0], sy=[1.0], sr=[5e4], px=[1.0], py=[1.0], pr=[3e4])
    for r2 in (pos, zone):
        f = compute_features(r2, np.array([t]))
        assert np.array_equal(f["F"], base["F"]) and np.array_equal(f["POS"], base["POS"], equal_nan=True) and np.array_equal(f["Zt"], base["Zt"])
    f = compute_features(pos, np.array([t + 1e-6]))
    assert not np.array_equal(f["POS"], base["POS"], equal_nan=True)  # one microsecond later the sample is the latest one
    f = compute_features(zone, np.array([t + 1e-6]))
    assert not np.array_equal(f["Zt"], base["Zt"])
    # a kill of a still-alive player exactly at the cut does not kill it at the cut (life is rebuilt from the records < t)
    from adaptation.features import member_life

    kills = raw["kills"]
    w_raw = np.concatenate([kills["w_raw"], [t]])
    victim = np.concatenate([kills["victim_pidx"], [8]])  # c1 (team 15), alive at the end
    life = member_life(w_raw, victim, raw["pos"]["t"], raw["pos"]["pidx"], raw["pos"]["health"])
    raw3 = dict(raw, life=life)
    assert np.array_equal(team_alive_causal(raw3, np.array([t])), alive0)
    assert not np.array_equal(member_alive_causal(life, len(raw["pid"]), np.array([t + 1e-6]))[8], [True])  # visible right after


def test_squad_sums_do_not_depend_on_how_many_cuts_are_computed_together(match):
    _, _, lab, raw = match
    ts = np.array([590.0, 800.0, 980.0])
    full = compute_features(raw, ts)["F"]
    for k in range(3):
        assert np.array_equal(full[:, k], compute_features(raw, ts[k : k + 1])["F"][:, 0])


# --------------------------------------------------------------------------- samples (K1): rows, flags, T
def test_val_test_samples_are_the_dump_rows_with_key_and_risk_flags(match):
    _, rec, lab, raw = match
    res, dropped = E.match_samples("synthetic", lab, raw, _extra(lab, raw), "val", "krafton192")
    assert dropped == 0 and res is not None
    slots = [p for p in range(8) if lab["target"][p].any()]
    assert res["S_cut"].tolist() == [p + 1 for p in slots] and np.allclose(res["S_t"], lab["cut_t"][slots].astype(np.float32))
    off = np.concatenate([[0], np.cumsum(res["S_n"])])
    T = lab["team_death_s"]
    t_win = max(2000.0, lab["match_end_s"], float(T[np.isfinite(T)].max()) + 1.0)
    for i, p in enumerate(slots):
        sl = slice(off[i], off[i + 1])
        tid = res["TIDX"][sl].astype(int)
        assert np.array_equal(tid, np.flatnonzero(lab["query_mask"][p]))  # rows = the dump's tokens (node_mask | target)
        assert np.array_equal(res["KEY"][sl], lab["node_mask"][p, tid])  # keys = the dump's node_mask
        assert np.array_equal(res["RISK"][sl], T[tid] > lab["cut_t"][p])  # Cox rows: T > a_k
        assert (res["RISK"][sl] | ~lab["target"][p, tid]).all()  # every risk-set row of the dump is a Cox row
        want = np.where(np.isinf(T[tid]), t_win, T[tid]).astype(np.float32)  # T of the dump; censoring at max(2000, C_m, ...)
        assert np.array_equal(res["TGT"][sl], want)
    # the comeback team (index 2) at cut 1: a row, not a key, at risk
    i0 = slots.index(0)
    sl = slice(off[i0], off[i0 + 1])
    j = list(res["TIDX"][sl]).index(2)
    assert not res["KEY"][sl][j] and res["RISK"][sl][j] and lab["target"][0, 2]


def test_train_samples_use_causal_keys_and_dump_T_and_drop_small_samples(match):
    _, rec, lab, raw = match
    res, dropped = E.match_samples("synthetic", lab, raw, _extra(lab, raw), "train", "every30")
    ts = res["S_t"].astype(np.float64)
    T = lab["team_death_s"]
    off = np.concatenate([[0], np.cumsum(res["S_n"])])
    n_dead_key = 0
    for i, t in enumerate(ts):
        sl = slice(off[i], off[i + 1])
        tid = res["TIDX"][sl].astype(int)
        key = team_alive_causal(raw, np.array([t]))[:, 0]
        assert np.array_equal(res["KEY"][sl], key[tid])  # keys: the causal status at t (records < t)
        assert np.array_equal(res["RISK"][sl], T[tid] > t)
        assert set(tid) == set(np.flatnonzero(key | (T > t)))  # rows: keys or risk rows
        assert int((T > t).sum()) >= 2  # fewer than two Cox rows: dropped
        n_dead_key += int((~res["KEY"][sl]).sum())
    assert n_dead_key >= 1  # the comeback team is a non-key row while it is dead (570..652) and still T > t
    assert len(ts) > 0 and dropped >= 0


def test_T_used_in_training_equals_the_dump_T_not_a_rederived_one(match):
    """T comes from the dump's ``team_death_s`` even where the raw rows would say otherwise (the v1 re-derivation is gone)."""
    src, rec, lab, raw = match
    lab2 = dict(lab)
    T2 = lab["team_death_s"].copy()
    T2[0] = 1234.5  # team 11: not what the kill events say
    lab2["team_death_s"] = T2
    res, _ = E.match_samples("synthetic", lab2, raw, _extra(lab2, raw), "train", "every30")
    for i in range(len(res["S_t"])):
        sl = slice(res["S_n"][:i].sum(), res["S_n"][: i + 1].sum())
        rows = res["TIDX"][sl].astype(int)
        if 0 in rows:
            assert res["TGT"][sl][list(rows).index(0)] == np.float32(1234.5)
    # a winner that is not censored in the dump is refused, nothing silently fixes it
    bad = dict(lab)
    bad["team_death_s"] = lab["team_death_s"].copy()
    bad["team_death_s"][lab["team_won"]] = 1090.0
    with pytest.raises(ValueError, match="censors the winners"):
        E.match_samples("synthetic", bad, raw, _extra(bad, raw), "val", "krafton192")


def test_alive_status_disagreeing_with_the_dump_is_an_error_at_a_cut(match):
    _, rec, lab, raw = match
    bad = dict(lab)
    bad["node_mask"] = lab["node_mask"].copy()
    bad["node_mask"][0, 0] = ~bad["node_mask"][0, 0]
    bad["query_mask"] = bad["node_mask"] | lab["target"]
    with pytest.raises(ValueError, match="differs from the dump's node_mask"):
        E.match_samples("synthetic", bad, raw, _extra(bad, raw), "val", "krafton192")


# --------------------------------------------------------------------------- K1: batch masks, backbone, Cox
def _k1_set(match):
    _, rec, lab, raw = match
    res, _ = E.match_samples("synthetic", lab, raw, _extra(lab, raw), "val", "krafton192")
    arrays = {k: res[k] for k in ("X", "POS", "TGT", "TIDX", "S_t", "S_cut", "S_n", "S_zone", "KEY", "RISK")}
    arrays["S_mi"] = np.zeros(len(res["S_t"]), dtype=np.int32)
    arrays["dump_version"] = np.array("v2")
    ss = KD.SampleSet(arrays, np.array(["synthetic"]), 0)
    mean, std = KD.fit_standardizer(ss)
    ss.finalize(mean, std)
    return ss, res


def test_k1_batch_separates_cox_rows_from_attention_keys(match):
    ss, res = _k1_set(match)
    assert ss.dump_version == "v2"
    b = ss.batch(np.arange(len(ss)))
    valid = b["padding_mask"]
    off = np.concatenate([[0], np.cumsum(res["S_n"])])
    for i in range(len(ss)):  # per sample: loss rows = RISK, keys = KEY, padding beyond the stored rows
        n = int(res["S_n"][i])
        assert torch.equal(b["squad_alive_mask"][i, :n], torch.as_tensor(res["RISK"][off[i] : off[i + 1]]))
        assert torch.equal(b["key_mask"][i, :n], torch.as_tensor(res["KEY"][off[i] : off[i + 1]]))
        assert not b["squad_alive_mask"][i, n:].any() and not b["key_mask"][i, n:].any()
    key = T1.key_mask_of(b)
    assert key.sum() < valid.sum()  # the comeback row is stored but is not a key
    assert not (key & ~valid).any() and not (b["squad_alive_mask"] & ~valid).any()
    # a v1 batch (no flags) behaves as before: every stored squad is key and loss row
    ss.KEY = ss.RISK = None
    b1 = ss.batch(np.arange(len(ss)))
    assert torch.equal(T1.key_mask_of(b1), b1["padding_mask"]) and torch.equal(b1["squad_alive_mask"], b1["padding_mask"])


def test_k1_backbone_ignores_comeback_rows_as_keys_and_cox_loss_counts_them_once(match):
    from src.models import TransformerBackbone
    from src.models.heads import get_head
    from src.training.losses import get_loss_fn

    ss, res = _k1_set(match)
    b = ss.batch(np.arange(len(ss)))
    key = T1.key_mask_of(b)
    cb = (b["padding_mask"] & ~key)
    assert cb.any()
    torch.manual_seed(0)
    backbone = TransformerBackbone(input_dim=S.N_INPUT, embed_dim=16, num_heads=2, num_layers=2, dropout=0.0, num_maps=len(S.MAPS)).eval()
    head = get_head(T1.LOSS, embed_dim=16, dropout=0.0).eval()

    def run(batch):
        with torch.no_grad():
            emb = backbone(batch["x_continuous"], batch["positions"], batch["bluezone_info"], batch["whitezone_info"], batch["map_idx"],
                           valid_mask=T1.key_mask_of(batch))  # fmt: skip
            return head(emb)

    base = run(b)
    assert torch.isfinite(base).all()
    pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    bi, si = (int(x[0]) for x in cb.nonzero(as_tuple=True))
    pert["x_continuous"][bi, si] += 25.0
    out = run(pert)
    other = b["padding_mask"].clone()
    other[bi, si] = False
    assert float((out - base).abs()[other].max()) < 1e-6  # nobody attends to the comeback row
    assert float((out - base).abs()[bi, si]) > 0  # the row itself is a query
    crit = get_loss_fn(T1.LOSS)
    valid_loss = b["squad_alive_mask"] & b["padding_mask"]
    assert bool(valid_loss[bi, si])  # a Cox loss row ...
    l0 = crit(base, b["y"], valid_loss)
    y2 = b["y"].clone()
    y2[bi, si] = y2[bi, si] * 0.25
    assert float((crit(base, y2, valid_loss) - l0).abs()) > 0  # ... that really enters the loss
    assert float((crit(base, y2, valid_loss & ~cb) - crit(base, b["y"], valid_loss & ~cb)).abs()) == 0.0


def test_k1_shards_roundtrip_and_refuse_mixed_versions(match, tmp_path):
    _, rec, lab, raw = match
    res, _ = E.match_samples("synthetic", lab, raw, _extra(lab, raw), "val", "krafton192")
    cohort = "paramo_tpp_squad"  # a real cohort name: schema.cohort_map must know it
    base = tmp_path / cohort / "shards" / "val"
    base.mkdir(parents=True)
    arrays = {k: res[k] for k in ("X", "POS", "TGT", "TIDX", "S_t", "S_cut", "S_n", "S_zone", "KEY", "RISK")}
    arrays.update(S_mi=np.zeros(len(res["S_t"]), dtype=np.int32), match_ids=np.array(["synthetic"]), dump_version=np.array("v2"))
    np.savez(base / "shard_00000.npz", **arrays)
    ss = KD.load_split(tmp_path, cohort, "val")
    assert ss.dump_version == "v2" and ss.KEY is not None and ss.RISK.dtype == bool
    # a v1 shard next to a v2 shard
    v1 = {k: v for k, v in arrays.items() if k not in ("KEY", "RISK", "dump_version")}
    np.savez(base / "shard_00001.npz", **v1)
    with pytest.raises(ValueError, match="different dump versions"):
        KD.load_split(tmp_path, cohort, "val")
    with pytest.raises(D.DumpVersionError, match="shard"):
        E.check_shard_version(base / "shard_00001.npz", {"cohort": cohort, "split": "val", "dump_version": "v2"})
    E.check_shard_version(base / "shard_00000.npz", {"cohort": cohort, "split": "val", "dump_version": "v2"})


# --------------------------------------------------------------------------- K2 / K3 on the dump's tokens
def test_k2_samples_are_the_dump_tokens_and_match_the_k1_rows(match):
    _, rec, lab, raw = match
    dump = _dump([rec])
    std = D.Standardizer.fit(dump)
    ss, info = OF.build_samples(dump, std, {}, "val")
    res, _ = E.match_samples("synthetic", lab, raw, _extra(lab, raw), "val", "krafton192")
    assert np.array_equal(ss.TIDX, res["TIDX"]) and np.array_equal(ss.S_n, res["S_n"]) and np.array_equal(ss.S_cut, res["S_cut"])
    assert np.array_equal(ss.KEY, res["KEY"]) and np.array_equal(ss.RISK, res["RISK"]) and np.array_equal(ss.TGT, res["TGT"])  # same T, C_m
    assert ss.dump_version == "v2" and info["n_comeback_rows"] == 1 and info["n_key_rows"] == int(res["KEY"].sum())
    assert OF.t_win_of(rec) == max(2000.0, rec["match_end_s"], 1086.0)
    # the K1 extraction log and K2 must agree on the censoring time
    with pytest.raises(ValueError, match="censoring time"):
        OF.build_samples(dump, std, {"synthetic": 2500.0}, "val")
    assert OF.build_samples(dump, std, {"synthetic": 2000.0}, "val")[1]["t_win_fallback"] == 0


def test_k3_labels_sit_on_the_dump_risk_rows_including_the_comeback_row(match):
    _, rec, lab, raw = match
    dump = _dump([rec])
    std = D.Standardizer.fit(dump)
    base, _ = OF.build_samples(dump, std, {}, "val")
    hs = H3.attach_labels(base, {"synthetic": lab})
    assert int(hs.RISK.sum()) == int(lab["target"].sum())  # every risk-set row (comeback included) carries a label
    b = hs.batch(np.arange(len(hs)))
    key = T1.key_mask_of(b)
    assert (b["padding_mask"] & ~key).any()
    assert ((b["y"] >= 0) & ~key).any()  # a labelled row that is not a key
    assert ((b["y"] >= 0) & ~(b["squad_alive_mask"] & b["padding_mask"])).sum() == 0  # ... and it is a loss row of the BCE trainer
    # the BCE trainer masks: loss rows = labelled rows, attention keys = key rows only
    m = H3.masked_bce(torch.zeros_like(b["y"]), b["y"], b["squad_alive_mask"] & b["padding_mask"])
    assert torch.isfinite(m)


# --------------------------------------------------------------------------- kf table
def test_kf_table_rows_are_the_dump_tokens_with_the_dump_flags(match):
    _, rec, lab, raw = match
    r = KF.table_rows("synthetic", lab, raw, _extra(lab, raw))
    slots = [p for p in range(7) if lab["query_mask"][p].any()]
    assert r["n"] == int(lab["query_mask"][slots].sum())
    assert int(r["in_risk_set"].sum()) == int(lab["target"][slots].sum()) == r["n_risk"]
    got = set(zip(r["phase"].tolist(), r["team_id"].tolist()))
    want = {(p + 1, int(lab["team_ids"][i])) for p in slots for i in np.flatnonzero(lab["query_mask"][p])}
    assert got == want
    # alive_at_cut = the dump's node_mask; the comeback row has alive_at_cut = 0 and in_risk_set = 1
    tid = lab["team_ids"]
    cb = (r["phase"] == 1) & (r["team_id"] == tid[2])
    assert cb.sum() == 1 and r["alive_at_cut"][cb][0] == 0 and r["in_risk_set"][cb][0] == 1
    node = np.array([lab["node_mask"][ph - 1, list(tid).index(t)] for ph, t in zip(r["phase"], r["team_id"])])
    assert np.array_equal(r["alive_at_cut"].astype(bool), node)
    assert np.allclose(r["cut_t"], lab["cut_t"][r["phase"] - 1])
    # features equal compute_features at the producer's cut (no other source)
    F = compute_features(raw, lab["cut_t"][slots])["F"]
    k_of = {p + 1: k for k, p in enumerate(slots)}
    idx = {int(t): i for i, t in enumerate(tid)}
    for n in range(r["n"]):
        assert np.array_equal(r["F"][n], F[idx[int(r["team_id"][n])], k_of[int(r["phase"][n])]])


def test_kf_table_rows_replay_exactly_from_the_records_before_each_cut(match):
    _, rec, lab, raw = match
    full = KF.table_rows("synthetic", lab, raw, _extra(lab, raw))
    for p in (0, 1, 2):
        only = dict(lab)
        for key in ("node_mask", "query_mask", "target"):
            only[key] = np.zeros_like(lab[key])
            only[key][p] = lab[key][p]
        trunc = E.truncate_raw(raw, float(lab["cut_t"][p]))
        part = KF.table_rows("synthetic", only, trunc, _extra(lab, raw))
        sel = full["phase"] == p + 1
        assert part["n"] == int(sel.sum())
        assert np.array_equal(part["F"], full["F"][sel]) and np.array_equal(part["team_id"], full["team_id"][sel])
        assert np.array_equal(part["in_risk_set"], full["in_risk_set"][sel]) and np.array_equal(part["alive_at_cut"], full["alive_at_cut"][sel])


def test_kf_table_refuses_a_causal_status_that_differs_from_the_dump(match):
    _, rec, lab, raw = match
    bad = dict(lab)
    bad["node_mask"] = lab["node_mask"].copy()
    bad["node_mask"][1, 1] = ~bad["node_mask"][1, 1]
    bad["query_mask"] = bad["node_mask"] | lab["target"]
    with pytest.raises(ValueError, match="causal squad status differs from the dump's node_mask"):
        KF.table_rows("synthetic", bad, raw, _extra(bad, raw))


def _table_frame(match):
    _, rec, lab, raw = match
    r = KF.table_rows("synthetic", lab, raw, _extra(lab, raw))
    cols = {"match_id": r["match_id"], "team_id": r["team_id"], "phase": r["phase"], "cut_t": r["cut_t"], "in_risk_set": r["in_risk_set"],
            "y_hazard_v2": r["y"], "alive_at_cut": r["alive_at_cut"]}  # fmt: skip
    df = pd.DataFrame({k: (np.full(r["n"], v, dtype=object) if k == "match_id" else v) for k, v in cols.items()})
    for j, name in enumerate(KF.KF):
        df[name] = r["F"][:, j]
    return df


def _write_table(root, df, version):
    base = Path(root)
    for split in ("train", "val", "test"):
        (base / "c").mkdir(parents=True, exist_ok=True)
        KF.write_table(df, base / "c" / f"{split}.csv.gz")
    if version is not None:
        (base / "meta.json").write_text(json.dumps({"name": f"bigcomp2027_kraftonfeat_{version}", "dump_version": version, "version": version}))


def test_extra_join_accepts_the_v2_table_and_refuses_v1_tables_next_to_a_v2_dump(match, tmp_path):
    _, rec, lab, raw = match
    dump = {"meta": _dump([rec])["meta"], "matches": [rec]}
    df = _table_frame(match)
    _write_table(tmp_path / "v2tab", df, "v2")
    joined = EX.load_extra(tmp_path / "v2tab", "c", {"train": dump, "val": dump, "test": dump})
    j = joined["val"]
    assert j.report["alignment"]["checked"] and j.report["target_rows"] == int(rec["target_mask_v2"].sum())
    tok = j.tokens["synthetic"]
    assert np.isfinite(tok[0, 2]).all()  # the comeback token has features (a dead squad's snapshot values)
    assert EX.table_dump_version(tmp_path / "v2tab") == "v2"
    # a table without meta.json / with a v1 name counts as v1: refused next to a v2 dump, before anything is read
    _write_table(tmp_path / "legacy", df, None)
    with pytest.raises(D.DumpVersionError, match="derived from a v1 dump but the .* dump is a v2 dump"):
        EX.load_extra(tmp_path / "legacy", "c", {"train": dump, "val": dump, "test": dump})
    _write_table(tmp_path / "v1tab", df, "v1")
    with pytest.raises(D.DumpVersionError):
        EX.load_extra(tmp_path / "v1tab", "c", {"train": dump, "val": dump, "test": dump})
    # ... and the other way round: a v2 table next to a v1 dump
    from model_stgcn.ptr.synthetic import make_dump

    v1 = make_dump(2, seed=3)
    with pytest.raises(D.DumpVersionError):
        EX.load_extra(tmp_path / "v2tab", "c", {"train": v1, "val": v1, "test": v1})
    assert str(EX.default_extra_dir("x/dumps/bigcomp2027_phase_v2")).endswith("bigcomp2027_kraftonfeat_v2")


def test_hgb_kraftonfeat_rows_follow_the_v2_table_and_labels(match):
    from adaptation import hgb_kraftonfeat as HK

    _, rec, lab, raw = match
    df = _table_frame(match)
    rows = HK.build_rows(df, {"synthetic": lab})
    assert len(rows["y"]) == int(lab["target"][:7].sum())
    assert (df.loc[df["in_risk_set"] == 1].shape[0]) == len(rows["y"])
    keys = list(zip(rows["keys"]["match_id"], rows["keys"]["team_id"], rows["keys"]["phase"]))
    assert keys == sorted(keys, key=lambda x: (x[0], x[1], x[2]))
    # the comeback risk row is a row of the table (kf features of a dead squad) and of the HGB rows
    assert (1, int(lab["team_ids"][2])) in {(k[2], k[1]) for k in keys}


# --------------------------------------------------------------------------- contract details added with the final dump
def test_member_alive_uses_the_contracts_revive_rule_including_the_stale_health_guard():
    """A healthy sample 1.6 s after a kill is stale (the dead body still reports health); revive evidence is the first healthy
    sample >= 20 s after the kill (``alive.MIN_REVIVE_GAP_S``).  The status is the contract's, not a copy."""
    from adaptation.features import member_life
    from model_stgcn.phase_v2 import alive as AL

    life = member_life(np.array([100.0]), np.array([0]), np.array([101.6, 119.0, 140.0, 200.0]), np.zeros(4, dtype=np.int64),
                       np.array([10.0, 5.0, 100.0, 100.0]))  # fmt: skip
    assert AL.MIN_REVIVE_GAP_S == 20.0 and life[0].revive[0] == 140.0  # the samples at 101.6 s and 119 s are ignored
    got = member_alive_causal(life, 1, np.array([101.7, 125.0, 140.0, 140.0 + 1e-9, 300.0]))[0]
    assert got.tolist() == [False, False, False, True, True]
    # identical to the contract's per-player function for every t (the vectorised loop only calls it)
    ts = np.linspace(90.0, 210.0, 241)
    want = [AL.player_alive_at(life, 0, float(np.nextafter(t, -np.inf))) for t in ts]
    assert member_alive_causal(life, 1, ts)[0].tolist() == want


def test_a_v2_label_with_a_stripped_mask_is_refused_not_rederived_from_T(match):
    from model_stgcn.ptr.data import DumpVersionError

    _, rec, lab, raw = match
    for strip in ("node_mask", "query_mask", "cut_t"):
        bad = {k: v for k, v in lab.items() if k != strip}  # still marked v2 by dump_version
        with pytest.raises(DumpVersionError, match="v2 label without"):
            E.match_samples("synthetic", bad, raw, _extra(lab, raw), "val", "krafton192")
        with pytest.raises(DumpVersionError, match="v2 label without"):
            KF.table_rows("synthetic", bad, raw, _extra(lab, raw))


def test_a_match_without_a_winner_flag_is_censored_by_the_alive_rule_in_every_consumer():
    src, rec = synth_record(no_winner=True)
    lab = compact_match(rec)
    raw = krafton_raw_from_source(src, lab)
    assert not lab["team_won"].any() and np.isposinf(lab["team_death_s"][4])  # team 15: alive at the end -> T = +inf, no flag needed
    res, _ = E.match_samples("synthetic", lab, raw, _extra(lab, raw), "val", "krafton192")
    off = np.concatenate([[0], np.cumsum(res["S_n"])])
    for i in range(len(res["S_t"])):
        sl = slice(off[i], off[i + 1])
        j = list(res["TIDX"][sl]).index(4)
        assert res["RISK"][sl][j] and res["KEY"][sl][j] and res["TGT"][sl][j] == np.float32(max(2000.0, lab["match_end_s"], 1086.0))
        p = int(res["S_cut"][i]) - 1
        assert lab["target"][p, 4] and lab["y"][p, 4] == 0.0  # at risk at every cut, never positive: stored rows, nothing re-added
    dump = _dump([rec])
    assert D.winner_censor_summary(dump)["winner_teams"] == 0
    ss, info = OF.build_samples(dump, D.Standardizer.fit(dump), {}, "val")
    assert np.array_equal(ss.TIDX, res["TIDX"]) and np.array_equal(ss.RISK, res["RISK"])
    r = KF.table_rows("synthetic", lab, raw, _extra(lab, raw))
    assert int(r["in_risk_set"].sum()) == int(lab["target"][:7].sum())


def test_no_input_of_the_krafton_and_sagong_consumers_depends_on_a_phase_duration():
    """The scheduled durations (``schedule_table.json``) feed the dump's duration features only; the 55 KRAFTON features and the
    12 Sagong inputs are counts, distances and ratios of positions, none is a rate or a duration (observed or scheduled)."""
    import re

    from model_stgcn.sagong2025 import features as SF
    from model_stgcn.sagong2025 import fit as SFIT

    banned = re.compile(r"dur|per_min|(?<!survival_)rate|elapsed|speed|frac|time|phase_len|length")  # survival_rate = Eq. 4, not a rate per time
    assert [f for f in S.FEATURES if banned.search(f)] == [] and len(S.FEATURES) == 55
    assert [f for f in (*SF.FEATURES, *SFIT.INPUT_NAMES) if banned.search(f)] == []
