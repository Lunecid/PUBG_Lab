"""G1 falsification contracts; fixtures are synthetic, never protected matches."""
from copy import deepcopy
from dataclasses import fields, replace
import sqlite3

import numpy as np
import pytest
import torch

import test_dataset
from pubg_lab.world_model.data.dataset import WorldDataset
from pubg_lab.world_model.features.inputs import ConditionVocabulary, adapt_graph, load_sample, collate_samples
from pubg_lab.world_model.models.encoder import WorldEncoder


@pytest.fixture
def dataset():
    helper = test_dataset.DatasetTests()
    helper.setUp()
    yield WorldDataset(helper.root)
    helper.tearDown()


def model_for(ds):
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    vocab = ConditionVocabulary.from_dataset(ds)
    return WorldEncoder(len(ds.item_vocabulary)+2, vocab.sizes, d=32).eval(), vocab


def prepare(ds, vocab, graph=None):
    if graph is None:
        graph, _ = ds.sample("fit", 0)
    k, ego = ds.bank("fit")["anchors"][0]
    return adapt_graph(graph, ego_id=int(ego), frame_end=int(k),
                       t0=float(ds.bank("fit")["times"][k]), view=ds.view, vocabulary=vocab)


def assert_outputs_close(a, b, permutation=None):
    for key in a:
        expected = a[key]
        if permutation is not None and key in ("player", "player_valid"):
            expected = expected[permutation]
        torch.testing.assert_close(expected, b[key], atol=2e-5, rtol=2e-5)


def test_masked_nan_future_labels_and_unknown_conditions(dataset):
    model, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    a = prepare(dataset, vocab, g)
    altered = deepcopy(g)
    altered["x"][~altered["mask"]] = np.nan
    altered["future_rank"] = np.full(100, -999)
    altered["targets"] = {"mask": np.ones((999, 6))}
    b = prepare(dataset, vocab, altered)
    assert_outputs_close(model(a), model(b))
    assert "targets" not in {f.name for f in fields(a)}
    assert vocab.encode({"patch": "unseen"})[-1] == 1
    assert not a.game_mask[:, 2].any()  # alive-team count is server-only


def test_permutations_preserve_team_and_permute_players(dataset):
    model, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    baseline = model(prepare(dataset, vocab, g))
    rng = np.random.default_rng(941)
    for _ in range(12):
        perm = rng.permutation(len(g["player_ids"]))
        inv = np.argsort(perm)
        h = deepcopy(g)
        h["player_ids"] = h["player_ids"][perm]
        for key in ("x", "mask", "source_time", "mobility", "mobility_mask",
                    "reentry_confidence", "reentry_confidence_mask"):
            h[key] = h[key][:, perm]
        for key in ("member_edges", "located_in", "vehicle_evidence"):
            if len(h[key]):
                h[key][:, 0] = inv[h[key][:, 0].astype(int)]
        assert_outputs_close(baseline, model(prepare(dataset, vocab, h)), perm)


def test_target_members_do_not_depend_on_future_mask(dataset):
    model, vocab = model_for(dataset)
    sample = load_sample(dataset, "fit", 0, vocab)
    before = model(sample.inputs)
    sample.targets["mask"][:] = False
    sample.targets["xy"][:] = 1e12
    assert_outputs_close(before, model(sample.inputs))
    assert len(sample.inputs.target_player_indices) == 5
    assert len(collate_samples([sample, sample]).inputs) == 2


def test_future_source_rejected(dataset):
    _, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    g["source_time"][-1, 0] = 1e9
    with pytest.raises(ValueError, match="future"):
        prepare(dataset, vocab, g)


def test_observable_gap_blocks_old_player_history(dataset):
    model, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    g["mask"][6, 0, :8] = False
    a = prepare(dataset, vocab, g)
    h = deepcopy(g)
    h["x"][:6, 0, 0:2] += 50000
    h["items_sparse"] = h["items_sparse"][h["items_sparse"][:, 0] >= 7]
    # Prior items must also be blocked by current segment.
    assert_outputs_close(model(a), model(prepare(dataset, vocab, h)))


def test_no_spatial_no_team_and_finite_gradient(dataset):
    model, vocab = model_for(dataset)
    for ablation in ("no_spatial", "no_team_node"):
        ds = WorldDataset(dataset.root, ablation=ablation)
        s = load_sample(ds, "fit", 0, vocab)
        out = model(s.inputs)
        assert all(torch.isfinite(v).all() for v in out.values())
        assert out["cell" if ablation == "no_spatial" else "team"].shape[0] == 0
    model.zero_grad()
    output = model(prepare(dataset, vocab))
    output["player"][:, 0].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_zone_sensitivity_and_cpu_exact_repeat(dataset):
    model, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    a = model(prepare(dataset, vocab, g))
    b = model(prepare(dataset, vocab, g))
    assert all(torch.equal(a[k], b[k]) for k in a)
    g["game"][-1, 3] += 300
    c = model(prepare(dataset, vocab, g))
    assert not torch.equal(a["player"], c["player"])


def test_server_opponent_state_reaches_focal_player(dataset):
    model, vocab = model_for(dataset)
    ds = WorldDataset(dataset.root, view="server")
    g, _ = ds.sample("fit", 0)
    a = model(prepare(ds, vocab, g))
    g["x"][-1, 5, 0] += 300
    b = model(prepare(ds, vocab, g))
    assert not torch.allclose(a["player"][0], b["player"][0], atol=1e-6, rtol=1e-6)


def test_private_and_alive_team_changes_do_not_reach_observer(dataset):
    model, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    a = model(prepare(dataset, vocab, g))
    g["x"][:, 1:, 8:] = 99999
    g["game"][:, 2] = 99999
    g["mobility"][:, 1:] = 9
    g["reentry_confidence"][:, 1:] = 99
    assert_outputs_close(a, model(prepare(dataset, vocab, g)))


@pytest.mark.parametrize("members", [1, 2, 5])
def test_variable_team_sizes_and_empty_history(dataset, members):
    model, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    g["player_ids"] = g["player_ids"][:members]
    for key in ("x", "mask", "source_time", "mobility", "mobility_mask",
                "reentry_confidence", "reentry_confidence_mask"):
        g[key] = g[key][:, :members]
    for key in ("member_edges", "located_in", "vehicle_evidence"):
        g[key] = g[key][g[key][:, 0] < members]
    # Remove all history except current frame; includes all-masked attention rows.
    g["mask"][:-1] = False
    a = prepare(dataset, vocab, g)
    out = model(a)
    assert out["player"].shape == (members, 32)
    assert out["team"].shape == (1, 32)
    assert all(torch.isfinite(v).all() for v in out.values())
    assert len(a.target_player_indices) == members


def test_reentry_and_damage_window_reset_use_visible_evidence(dataset):
    _, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    g["x"][6, 0, 5] = 0
    g["x"][7:, 0, 16] = np.arange(6)*10
    g["x"][:, 0, 9:14] = 123
    a = prepare(dataset, vocab, g)
    assert not a.history_valid[:7, 0].any()
    assert not a.mask[7, 0, 3:5].any()
    assert not a.mask[7:10, 0, 10:14].any()
    assert a.mask[10:, 0, 10:14].all()
    assert not (a.items[:, 1] == 0).any()  # equipment predates reentry


def test_bad_observed_value_and_future_item_raise(dataset):
    _, vocab = model_for(dataset)
    g, _ = dataset.sample("fit", 0)
    h = deepcopy(g)
    h["x"][-1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        prepare(dataset, vocab, h)
    g["items_sparse"] = np.vstack([g["items_sparse"], [9999, 0, 2]])
    with pytest.raises(ValueError, match="future"):
        prepare(dataset, vocab, g)


def test_fixed_cpu_attention_path_matches_grad_and_no_grad(dataset):
    model, vocab = model_for(dataset)
    original = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        g = prepare(dataset, vocab)
        a = model(g)
        with torch.no_grad():
            b = model(g)
        assert all(torch.equal(a[k], b[k]) for k in a)
    finally:
        torch.backends.mha.set_fastpath_enabled(original)


def test_own_match_future_spatial_support_is_excluded(dataset):
    _, vocab = model_for(dataset)
    # Fixture has only the focal match in fit: every observed edge must disappear.
    sample = load_sample(dataset, "fit", 0, vocab)
    assert len(sample.inputs.observed) == 0
    assert len(sample.inputs.inferred) > 0
    assert sample.provenance["spatial_prior_policy"] == "fit_minus_current_match"
    # Add the second synthetic match to the fit population, retaining its 99 counts.
    dataset.fit_ids.add("excluded")
    dataset.spatial_cache.clear()
    a = load_sample(dataset, "fit", 0, vocab).inputs
    assert a.observed[:, 3].tolist() == [99]
    # Changing focal-match support, including future movement, must have no effect.
    with sqlite3.connect(dataset.root/"catalog.sqlite") as db:
        db.execute("UPDATE spatial_support SET n=999999 WHERE match_id='fit'")
    dataset.spatial_cache.clear()
    dataset._g1_exclusion_cache.clear()
    b = load_sample(dataset, "fit", 0, vocab).inputs
    for name in ("observed", "inferred", "cells"):
        assert torch.equal(getattr(a, name), getattr(b, name))
