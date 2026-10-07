"""v2 consumers, part 1: loader labels and the key / query mask contract of PTR-Former (DB-free).

Contract (``model_stgcn/dumps/bigcomp2027_phase_v2/CONTRACT.md``, task rules 1-2 and 6):

* the loader takes rows, Y and T from the v2 dump exactly; winners are censored by the dump; no winner is re-added
  (not even at an excluded cut) and nothing is re-derived from death times;
* key mask = ``node_mask`` (causally alive at the cut), token / query mask = ``query_mask`` = ``node_mask | target``; a
  comeback row (risk set, causally dead at the cut) is a query token but never a key for another team or later slot;
* a query token always attends to its own key, no attention row of a query token is empty, no NaN;
* the loss and the metrics read the target rows once;
* v1 dumps still load, and v1 / v2 artefacts cannot be mixed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model_stgcn.ptr import data as D
from model_stgcn.ptr import hgb as H
from model_stgcn.ptr import relations as R
from model_stgcn.ptr.arms import ARMS, model_config
from model_stgcn.ptr.loss import ptr_loss
from model_stgcn.ptr.model import PTRFormer
from model_stgcn.ptr.synthetic import make_dump
from tests.v2_fixtures import comeback_cells, make_v2_dump, write_v2_cohort

KEYS = ("hazard_logit", "rank_score")


def build(arm_name="M11RZ", *, seed=0, n_matches=4, d_model=32, n_layers=2, dump=None, **kw):
    """(model in eval mode, dataset, batch of all matches) on a v2 dump that contains comeback rows."""
    kw.setdefault("comeback_prob", 0.12)
    dump = dump or make_v2_dump(n_matches, seed=seed, **kw)
    sz = D.Standardizer.fit(dump)
    arm = ARMS[arm_name]
    ds = D.PTRDataset(dump, sz, drop_groups=arm.drop_groups, relation_types=R.RELATION_TYPES if arm.relation else None,
                      shuffle_seed=42 if arm.shuffle else None)  # fmt: skip
    cfg = model_config(arm, ds.n_features, n_phases=ds.n_phases, zone_dim=ds.zone_dim, d_model=d_model, n_heads=4, n_layers=n_layers, dropout=0.0)
    torch.manual_seed(seed)
    model = PTRFormer(cfg, init_seed=seed).eval()
    return model, ds, ds.get_batch(list(range(len(dump["matches"]))))


def clone(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def run(model, batch):
    with torch.no_grad():
        return model(batch)


# --------------------------------------------------------------------------- version dispatch
def test_dump_version_dispatch():
    v1, v2 = make_dump(2, seed=1), make_v2_dump(2, seed=1)
    assert D.dump_version(v1) == "v1" and D.dump_version(v1["meta"]) == "v1"
    assert D.dump_version(v2) == "v2" and D.dump_version(v2["meta"]) == "v2"
    D.validate_dump(v1)
    D.validate_dump(v2)
    for bad in ("v3", "3", "garbage"):
        with pytest.raises(D.DumpVersionError, match="unsupported dump version"):
            D.dump_version({"version": bad})


def test_v2_dump_has_one_label_set_and_refuses_pipeline_labels():
    dump = make_v2_dump(2, seed=2)
    with pytest.raises(D.DumpVersionError, match="not defined for a v2 dump"):
        D.validate_dump(dump, labels="pipeline")
    with pytest.raises(D.DumpVersionError):
        D.select_labels(dump["matches"][0], "pipeline")


def test_v1_loader_path_is_unchanged_for_v1_dumps():
    dump = make_dump(3, seed=3)
    assert D.alive_source(dump, "v2") == "node_mask_v2" and not D.is_v2_match(dump["matches"][0])
    m = dump["matches"][0]
    assert D.effective_alive(m, "v2", True).equal(D.event_alive_mask(m))
    assert D.key_mask(m, "v2", True).equal(D.effective_alive(m, "v2", True))
    assert not D.comeback_mask(m).any()
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump), relation_types=R.RELATION_TYPES)
    assert "query_mask" not in ds.get_batch([0, 1])  # v1 batches keep the old schema


@pytest.mark.parametrize(
    "corrupt, exc, text",
    [
        (lambda d: d["matches"][0]["query_mask"].__ior__(~d["matches"][0]["node_mask"] & ~d["matches"][0]["target_mask_v2"]), ValueError, "query_mask must equal"),
        (lambda d: d["matches"][0].__setitem__("node_mask_v2", d["matches"][0]["node_mask"].clone()), D.DumpVersionError, "node_mask_v2"),
        (lambda d: d["matches"][0].pop("query_mask"), ValueError, "missing keys"),
        (lambda d: d["matches"][0].pop("cut_t"), ValueError, "missing keys"),
        (lambda d: d["matches"][0]["team_death_s"].__setitem__(int(d["matches"][0]["team_won"].nonzero()[0, 0]), 123.0), ValueError, "censors the winners"),
        (lambda d: d["matches"][0]["team_death_s"].__setitem__(0, float("-inf")), ValueError, "finite, \\+inf"),
        (lambda d: d["matches"][0]["target_mask_v2"].__setitem__((7, 0), True), ValueError, "last phase slot"),
    ],
)
def test_validate_dump_rejects_bad_v2_records(corrupt, exc, text):
    dump = make_v2_dump(3, seed=4)
    corrupt(dump)
    with pytest.raises(exc, match=text):
        D.validate_dump(dump)


def test_validate_rejects_risk_rows_at_an_excluded_cut():
    dump = make_v2_dump(6, seed=5, exclude_prob=0.5)
    m = next(mm for mm in dump["matches"] if mm["cut_excluded"])
    p = min(m["cut_excluded"]) - 1
    D.validate_dump(dump)
    m["target_mask_v2"][p, 0] = True
    m["query_mask"][p, 0] = True
    with pytest.raises(ValueError, match="excluded / invalid cut"):
        D.validate_dump(dump)


def test_v1_record_with_query_mask_and_mixed_split_versions_are_refused(tmp_path):
    dump = make_dump(2, seed=6)
    dump["matches"][0]["query_mask"] = dump["matches"][0]["node_mask"].clone()
    with pytest.raises(D.DumpVersionError, match="not 'v2'"):
        D.validate_dump(dump)
    base = write_v2_cohort(tmp_path, "mix", n=2)
    torch.save(make_dump(2, seed=7, cohort="mix", split="val"), base / "val.pt")  # a v1 split next to two v2 splits
    with pytest.raises(D.DumpVersionError, match="split train is a v2 dump but split val is a v1 dump"):
        D.load_cohort(tmp_path, "mix")


def test_load_cohort_roundtrip_v2_and_event_alive_is_refused(tmp_path):
    write_v2_cohort(tmp_path, "c", n=3)
    splits = D.load_cohort(tmp_path, "c")
    assert {D.dump_version(v) for v in splits.values()} == {"v2"}
    m = splits["train"]["matches"][0]
    with pytest.raises(D.DumpVersionError, match="L3 look-ahead"):
        D.event_alive_mask(m)  # a mask derived from the final death time would be the leak
    assert D.alive_source(splits["train"], "v2") == D.ALIVE_SOURCE_V2 and not D.uses_event_alive(splits["train"], "v2")


# --------------------------------------------------------------------------- labels: rows, Y and T come from the dump
def test_loader_rows_and_labels_equal_the_dump_exactly_and_no_winner_is_readded():
    dump = make_v2_dump(10, seed=8, exclude_prob=0.3)
    n_excl = 0
    for m in dump["matches"]:
        y, target = D.select_labels(m, "v2")
        assert torch.equal(target, m["target_mask_v2"]) and torch.equal(y, m["y_hazard_v2"] * m["target_mask_v2"])
        won = m["team_won"]
        assert torch.isposinf(m["team_death_s"][won]).all()  # winners are censored by the dump
        assert not (y[:, won] > 0).any()  # a winner is never positive
        for k in m["cut_excluded"]:  # excluded cut: all-False risk row, the winner is NOT put back
            n_excl += 1
            assert not target[k - 1].any()
        assert not target[-1].any()
    assert n_excl > 0
    for s in ("v2",):
        summary = D.winner_censor_summary(dump, s)
        assert summary["target_rows_added"] == 0 and summary["target_rows_removed"] == 0 and summary["positives_removed"] == 0
        assert summary["winner_teams"] == len(dump["matches"]) and summary["winners_finite_death"] == 0
    # the v1 winner override WOULD have re-added the winners at the excluded cuts: that is what the pass-through avoids
    legacy = copy.deepcopy(dump)
    for m in legacy["matches"]:
        for k in D.V2_ONLY_KEYS:  # strip every v2-only key: the record is then read through the v1 rules
            m.pop(k, None)
        m["node_mask_v2"] = m["node_mask"].clone()
    legacy["meta"].pop("version")
    legacy_added = D.winner_censor_summary(legacy, "v2")["target_rows_added"]
    assert legacy_added > 0


def test_hgb_rows_are_exactly_the_dump_risk_rows_in_order():
    dump = make_v2_dump(8, seed=9)
    rows = H.build_rows(dump, "HGB-cut")
    ref_keys, ref_y = [], []
    for m in sorted(dump["matches"], key=lambda mm: str(mm["match_id"])):
        tgt = m["target_mask_v2"].numpy()
        p_idx, i_idx = np.nonzero(tgt)
        tid = m["team_ids"].numpy()
        order = np.lexsort((p_idx, tid[i_idx]))
        ref_keys += [(str(m["match_id"]), int(tid[i_idx[o]]), int(p_idx[o]) + 1) for o in order]
        ref_y += [float(m["y_hazard_v2"][p_idx[o], i_idx[o]]) for o in order]
    got = list(zip(rows.keys["match_id"], rows.keys["team_id"].astype(int), rows.keys["phase"].astype(int)))
    assert got == ref_keys and np.array_equal(rows.y, np.asarray(ref_y, dtype=np.float32))
    assert int(sum(m["target_mask_v2"].sum() for m in dump["matches"])) == len(rows.y)


def test_standardizer_and_summary_count_query_tokens_incl_comebacks():
    dump = make_v2_dump(8, seed=10, comeback_prob=0.15)
    cells = comeback_cells(dump)
    assert cells
    s = D.event_alive_summary(dump)
    n_query = sum(int(m["query_mask"].sum()) for m in dump["matches"])
    n_key = sum(int(m["node_mask"].sum()) for m in dump["matches"])
    assert (s["n_alive"], s["n_node_mask"], s["ghost_removed"]) == (n_query, n_key, 0)
    assert s["comeback_rows"] == len(cells) and sum(s["comeback_rows_by_cut"]) == len(cells)
    assert s["raised"] == n_query - n_key
    # tokens the standardiser sees = query tokens (a comeback token is a model input)
    sz = D.Standardizer.fit(dump)
    rows = np.concatenate([m["X"].numpy()[m["query_mask"].numpy()] for m in dump["matches"]])
    np.testing.assert_allclose(sz.x_mean, np.nanmean(rows.astype(np.float64), axis=0), rtol=1e-9, atol=1e-9)


def test_dataset_items_carry_key_query_and_target_masks():
    dump = make_v2_dump(5, seed=11, comeback_prob=0.15)
    _, ds, batch = build(dump=dump)
    for i, m in enumerate(dump["matches"]):
        it = ds[i]
        assert torch.equal(it["node_mask"], m["node_mask"]) and torch.equal(it["query_mask"], m["query_mask"])
        assert torch.equal(it["target_mask"], m["target_mask_v2"])
        assert (it["query_mask"] | ~it["target_mask"]).all() and (it["query_mask"] | ~it["node_mask"]).all()
    assert (batch["query_mask"] | ~batch["target_mask"]).all() and (batch["query_mask"] | ~batch["node_mask"]).all()
    assert int((batch["query_mask"] & ~batch["node_mask"]).sum()) == len(comeback_cells(dump))  # the only non-key tokens


# --------------------------------------------------------------------------- model: key / query masks
def _first_comeback(batch):
    cb = batch["target_mask"] & ~batch["node_mask"]
    assert cb.any(), "the fixture must contain a comeback row"
    b, p, i = (int(v[0]) for v in cb.nonzero(as_tuple=True))
    return b, p, i


@pytest.mark.parametrize("arm", ["M00", "M10", "M01", "M11", "M11R", "M11RZ"])
def test_comeback_query_is_present_finite_and_its_key_is_masked_for_everyone_else(arm):
    model, ds, batch = build(arm, seed=1)
    out = run(model, batch)
    for k in KEYS:
        assert torch.isfinite(out[k]).all(), k  # no NaN anywhere, comeback rows included
    b, p, i = _first_comeback(batch)
    base = {k: out[k].clone() for k in KEYS}

    pert = clone(batch)
    g = torch.Generator().manual_seed(3)
    pert["x"][b, p, i] = torch.randn(pert["x"].shape[-1], generator=g) * 1e3  # huge garbage in the comeback token
    pert["pos"][b, p, i] = 7.0
    if "rel" in pert:  # every edge that touches the comeback team in this phase, both roles
        pert["rel"][b, p, :, i, :] = 5.0
        pert["rel"][b, p, :, :, i] = 5.0
    out2 = run(model, pert)
    for k in KEYS:
        assert torch.isfinite(out2[k]).all()
        diff = (out2[k] - base[k]).abs()
        other = torch.ones_like(diff, dtype=torch.bool)
        other[b, p, i] = False
        assert float(diff[other].max()) < 1e-5, f"{k}: a comeback token leaked into another team / slot"
    # the query itself is present: its own output reacts to its own token
    assert max(float((out2[k][b, p, i] - base[k][b, p, i]).abs()) for k in KEYS) > 1e-4


def test_comeback_query_reads_the_alive_keys_and_its_own_past_keys():
    """The comeback row sees the board (alive keys of its slot) and its own earlier key slots; changing a key team changes it."""
    model, ds, batch = build("M11R", seed=2)
    b, p, i = _first_comeback(batch)
    base = run(model, batch)
    others = [j for j in range(batch["x"].shape[2]) if j != i and bool(batch["node_mask"][b, p, j])]
    assert others
    pert = clone(batch)
    pert["x"][b, p, others[0]] += 5.0
    out = run(model, pert)
    assert max(float((out[k][b, p, i] - base[k][b, p, i]).abs()) for k in KEYS) > 1e-6  # team axis: the comeback attends to the keys
    if p >= 1 and bool(batch["node_mask"][b, p - 1, i]):  # phase axis: its earlier key slot is visible
        pert = clone(batch)
        pert["x"][b, p - 1, i] += 5.0
        out = run(model, pert)
        assert max(float((out[k][b, p, i] - base[k][b, p, i]).abs()) for k in KEYS) > 1e-6


def test_comeback_token_is_not_a_key_for_later_slots_of_the_same_team():
    model, ds, batch = build("M11", seed=3)
    b, p, i = _first_comeback(batch)
    if p + 1 >= batch["x"].shape[1]:
        pytest.skip("comeback at the last slot")
    base = run(model, batch)
    pert = clone(batch)
    pert["x"][b, p, i] += 50.0
    out = run(model, pert)
    # a later slot of the same team does not see slot p (its key flag is off there), so only (p, i) changes
    assert max(float((out[k][b, p + 1 :, i] - base[k][b, p + 1 :, i]).abs().max()) for k in KEYS) < 1e-5


def test_no_attention_row_of_a_query_token_is_empty():
    """Nobody else is a key in the slot: the comeback query keeps its own key (finite output), others see only themselves."""
    model, ds, batch = build("M11RZ", seed=4)
    b, p, i = _first_comeback(batch)
    lone = clone(batch)
    lone["node_mask"][b, p] = False  # nobody is a key in this slot of this match, the comeback row is still a query
    out = run(model, lone)
    for k in KEYS:
        assert torch.isfinite(out[k]).all()
    # a batch where NO token is a key anywhere in a match and every row has only its own key
    empty = clone(batch)
    empty["node_mask"][b] = False
    out = run(model, empty)
    for k in KEYS:
        assert torch.isfinite(out[k]).all()


def test_without_query_mask_the_model_behaves_as_v1_and_with_equal_masks_it_is_identical():
    model, ds, batch = build("M11RZ", seed=5, comeback_prob=0.0)
    assert not (batch["query_mask"] & ~batch["node_mask"]).any()  # no comeback row: tokens == keys
    # force query == key (drop the comeback / guard difference): the result must equal the v1 path (no query_mask in the batch)
    eq = clone(batch)
    eq["query_mask"] = eq["node_mask"].clone()
    no_q = {k: v for k, v in eq.items() if k != "query_mask"}
    o1, o2 = run(model, eq), run(model, no_q)
    for k in KEYS:
        torch.testing.assert_close(o1[k], o2[k], atol=0.0, rtol=0.0)


def test_loss_is_counted_once_on_target_rows_and_includes_comeback_rows():
    model, ds, batch = build("M11R", seed=6)
    b, p, i = _first_comeback(batch)
    out = run(model, batch)
    total, parts = ptr_loss(out, batch, rank_weight=0.1)
    tm = batch["target_mask"]
    assert int(parts["n_target"]) == int(tm.sum()) == int(sum(int(m["target_mask_v2"].sum()) for m in ds.matches))
    assert bool(tm[b, p, i]) and not bool(batch["node_mask"][b, p, i])  # the comeback row is a loss row ...
    manual = F.binary_cross_entropy_with_logits(out["hazard_logit"][tm], batch["y_hazard"][tm], reduction="sum") / tm.sum()
    torch.testing.assert_close(parts["bce"], manual, atol=1e-6, rtol=1e-6)  # ... counted exactly once
    # a non-target row (a key / query that is not in the risk set) carries no loss: its logit cannot change the loss
    non_target = batch["query_mask"] & ~tm
    assert non_target.any()
    bad = {k: v.clone() for k, v in out.items()}
    bad["hazard_logit"][non_target] = 1e6
    bad["rank_score"][non_target] = 1e6
    t2, _ = ptr_loss(bad, batch, rank_weight=0.1)
    torch.testing.assert_close(t2, total, atol=1e-6, rtol=1e-6)


def test_training_step_runs_on_v2_batches_with_comeback_rows_and_gradients_are_finite():
    model, ds, batch = build("M11RZ", seed=7)
    model.train()
    total, _ = ptr_loss(model(batch), batch, rank_weight=0.1)
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shuffle_permutes_the_token_set_of_a_v2_dump_deterministically():
    dump = make_v2_dump(4, seed=12, comeback_prob=0.2)
    sz = D.Standardizer.fit(dump)
    arm = ARMS["M11R-shuf"]
    ds = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES, shuffle_seed=7)
    a, b = ds[0], ds[0]
    for name in a["edges"]:
        assert torch.equal(a["edges"][name][0], b["edges"][name][0])
    m = dump["matches"][0]
    perms = R.shuffle_permutations(m["match_id"], 7, m["query_mask"].numpy())
    for p in range(perms.shape[0]):  # tokens outside the query mask keep their label
        keep = ~m["query_mask"][p].numpy()
        assert (perms[p][keep] == np.arange(len(keep))[keep]).all()


def test_stage2_player_layer_is_refused_for_v2():
    dump = make_v2_dump(2, seed=13)
    with pytest.raises(D.DumpVersionError, match="player layer"):
        D.PlayerStandardizer.fit(dump, {})


# --------------------------------------------------------------------------- no fallback to team_death_s; a match without a winner flag
_STRIPPED = {  # which entry points must refuse a v2 record that lacks the key (they never fall back to T)
    "query_mask": ("effective_alive", "effective_alive_event", "key_mask", "select_labels", "comeback_mask"),
    "node_mask": ("effective_alive", "effective_alive_event", "key_mask", "select_labels", "comeback_mask"),
    "target_mask_v2": ("select_labels", "comeback_mask"),
    "y_hazard_v2": ("select_labels",),
}
_ENTRY = {
    "effective_alive": lambda m: D.effective_alive(m, "v2", False),
    "effective_alive_event": lambda m: D.effective_alive(m, "v2", True),
    "key_mask": lambda m: D.key_mask(m),
    "select_labels": lambda m: D.select_labels(m, "v2"),
    "comeback_mask": lambda m: D.comeback_mask(m),
}


@pytest.mark.parametrize("strip", sorted(_STRIPPED))
def test_a_v2_record_with_a_stripped_mask_is_refused_not_read_through_the_v1_rules(strip):
    """The v2 contract: ``node_mask_v2`` is intentionally absent and NO path may fall back to recomputing masks from T."""
    dump = make_v2_dump(2, seed=21)
    m = dump["matches"][0]
    assert D.is_v2_match(m)
    m.pop(strip)
    assert D.is_v2_match(m)  # still marked by the other v2-only keys: it is NOT silently treated as a v1 record
    for name in _STRIPPED[strip]:
        with pytest.raises(D.DumpVersionError, match="never rebuilds"):
            _ENTRY[name](m)
    with pytest.raises(ValueError):
        D.validate_dump(dump)  # a v2 dump with such a record does not validate either


def test_one_v2_key_is_enough_to_mark_a_record_and_a_v1_dump_may_not_carry_any():
    for key in D.V2_ONLY_KEYS:
        assert D.is_v2_match({key: 1})
    assert not D.is_v2_match({"cut_t": 1, "cut_wall_s": 1, "node_mask_label": 1, "node_mask": 1})  # v1 records carry these
    v1 = make_dump(2, seed=22)
    v1["matches"][0]["cut_valid"] = torch.ones(8, dtype=torch.bool)
    with pytest.raises(D.DumpVersionError, match="v2-only key"):
        D.validate_dump(v1)


def test_a_match_without_a_winner_flag_keeps_its_stored_rows():
    """One real match (b5134321) has no ``team_won`` team: its alive team is censored by the alive rule (T = +inf).  Nothing is
    re-added or fixed: rows, labels, masks and T are the dump's, the censored team is at risk at every cut and never positive."""
    dump = make_v2_dump(3, seed=23)
    m = dump["matches"][0]
    won = m["team_won"].clone()
    m["team_won"][:] = False  # the producer's record for that match: no winner flag, the alive team keeps T = +inf
    assert torch.isposinf(m["team_death_s"][won]).all()
    D.validate_dump(dump)
    y, target = D.select_labels(m, "v2")
    assert torch.equal(target, m["target_mask_v2"]) and torch.equal(y, m["y_hazard_v2"] * m["target_mask_v2"])
    assert not (y[:, won] > 0).any() and target[:, won][m["cut_valid"]].all()  # at risk at every valid cut, never positive
    s = D.winner_censor_summary(dump)
    assert s["winner_teams"] == len(dump["matches"]) - 1 and s["target_rows_added"] == 0 and s["winners_finite_death"] == 0
    # no winner flag, but a finite T for the team that should be censored: that IS a defect and is reported, not repaired
    m["team_death_s"][won] = 123.0
    D.validate_dump(dump)  # without team_won there is nothing to compare against (the alive rule decides)
    assert torch.equal(D.select_labels(m, "v2")[1], m["target_mask_v2"])


def test_cut_lead_and_exact_cut_times_follow_the_producer():
    from model_stgcn.phase_v2.phases import CUT_LEAD_S as PRODUCER_LEAD

    assert D.CUT_LEAD_S == PRODUCER_LEAD == 10.0
    dump = make_v2_dump(2, seed=24)
    m = dump["matches"][0]
    b = D.boundary_times(m["phase_end_s"])
    assert np.array_equal(b, m["phase_end_s"].double().numpy(), equal_nan=True)  # float64 time fields are used as they are
    assert np.array_equal(D.cut_times(m["phase_end_s"]), b - 10.0, equal_nan=True)
    # a float32 copy (an earlier producer) is rounded to the millisecond: it recovers the exact boundary
    exact = torch.tensor([1234.567, 801.001, float("nan")], dtype=torch.float64)
    f32 = exact.float()
    assert not torch.equal(f32.double()[:2], exact[:2])
    assert np.array_equal(D.boundary_times(f32), exact.numpy(), equal_nan=True)
