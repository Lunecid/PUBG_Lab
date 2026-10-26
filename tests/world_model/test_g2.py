import math
import sqlite3
from dataclasses import fields

import numpy as np
import pytest
import torch

import test_dataset
from world_model.dataset import WorldDataset
from world_model.indexed_dataset import IndexedWorldDataset
from world_model.inputs import ConditionVocabulary, load_sample
from world_model.pretraining import joint_gmm_nll, masked_mse, corrupt_health, pair_targets, PretrainHeads, PathProbe, pretrain_losses, supervised_targets
from world_model.encoder import WorldEncoder


@pytest.fixture
def ds():
    helper = test_dataset.DatasetTests()
    helper.setUp()
    dataset = WorldDataset(helper.root)
    # Align the temporary catalog to its bank, as required by the real V3 contract.
    with sqlite3.connect(helper.root/"catalog.sqlite") as db:
        db.execute("DELETE FROM spatial_support")
        db.executemany("INSERT INTO spatial_support VALUES(?,?,?,?,?,?,?,?)",
                       [("fit", "A__40.2", *map(int, row))
                        for row in dataset.bank("fit")["spatial_edges"]])
    yield dataset
    helper.tearDown()


def test_indexed_loader_is_identical(ds):
    fast = IndexedWorldDataset(ds.root)
    vocab = ConditionVocabulary.from_dataset(ds)
    for view in ("observer", "server"):
        ds.view = fast.view = view
        for anchor in (0, 3, len(ds.bank("fit")["anchors"])-1):
            a, b = load_sample(ds, "fit", anchor, vocab), load_sample(fast, "fit", anchor, vocab)
            for f in fields(a.inputs):
                x, y = getattr(a.inputs, f.name), getattr(b.inputs, f.name)
                if isinstance(x, torch.Tensor):
                    assert torch.equal(x, y), f.name
                else:
                    assert x == y


def test_masked_losses_ignore_nan_and_empty():
    pred = torch.tensor([2., 4.], requires_grad=True)
    y = torch.tensor([0., float("nan")])
    loss = masked_mse(pred, y, torch.tensor([True, False]))
    assert loss.item() == 4.
    loss.backward()
    assert pred.grad.tolist() == [4., 0.]
    assert masked_mse(pred, y, torch.zeros(2, dtype=torch.bool)) is None


def test_joint_gaussian_matches_hand_calculation_and_permutations():
    y = torch.zeros(2, 1, 2, dtype=torch.float64)
    mu = torch.zeros(2, 2, 1, 2, dtype=torch.float64, requires_grad=True)
    logs = torch.zeros_like(mu)
    weights = torch.zeros(2, dtype=torch.float64)
    mask = torch.ones(2, 1, dtype=torch.bool)
    loss = joint_gmm_nll(mu, logs, weights, y, mask)
    assert loss.item() == pytest.approx(.5*math.log(2*math.pi))
    loss.backward()
    assert torch.isfinite(mu.grad).all()
    rng = np.random.default_rng(37)
    mu = torch.from_numpy(rng.normal(size=(2, 4, 3, 2)))
    logs = torch.from_numpy(rng.normal(size=mu.shape)*.1)
    y = torch.from_numpy(rng.normal(size=(2, 3, 2)))
    mask = torch.tensor([[True, False, True], [True, True, False]])
    pi = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    expected = joint_gmm_nll(mu, logs, pi, y, mask)
    torch.testing.assert_close(expected, joint_gmm_nll(mu.flip(0), logs.flip(0), pi, y.flip(0), mask.flip(0)))
    torch.testing.assert_close(expected, joint_gmm_nll(mu[:, [2, 0, 3, 1]], logs[:, [2, 0, 3, 1]],
                                                     pi[[2, 0, 3, 1]], y, mask))
    y[~mask] = float("nan")
    torch.testing.assert_close(expected, joint_gmm_nll(mu, logs, pi, y, mask))


def test_shared_component_not_independent_player_mixtures():
    # Joint components put both actors at -10 OR both at +10; mixed target is unlikely.
    mu = torch.tensor([[[[-10., 0.]], [[10., 0.]]],
                       [[[-10., 0.]], [[10., 0.]]]])
    y = torch.tensor([[[-10., 0.]], [[10., 0.]]])
    joint = joint_gmm_nll(mu, torch.zeros_like(mu), torch.zeros(2), y, torch.ones(2, 1, dtype=torch.bool))
    independent = sum(joint_gmm_nll(mu[i:i+1], torch.zeros_like(mu[i:i+1]), torch.zeros(2),
                                   y[i:i+1], torch.ones(1, 1, dtype=torch.bool)) for i in range(2))/2
    assert joint > independent+40


def test_health_corruption_removes_all_copies(ds):
    vocab = ConditionVocabulary.from_dataset(ds)
    sample = load_sample(ds, "fit", 0, vocab)
    a, target, selected = corrupt_health(sample.inputs, seed=123)
    assert selected.any()
    assert not a.mask[:, selected, 5].any()
    assert torch.count_nonzero(a.x[:, selected, 5]) == 0
    original = sample.inputs.x.clone()
    sample.inputs.x[:, selected, 5] = .123
    b, _, again = corrupt_health(sample.inputs, seed=123)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.mask, b.mask)
    assert torch.equal(selected, again)
    sample.inputs.x.copy_(original)


def test_pair_geometry_and_solo_are_not_fake_zero():
    xy = torch.tensor([[[0., 0.], [0., 0.]], [[3., 4.], [6., 8.]]])
    distance, valid, pairs = pair_targets(xy, torch.tensor([[True, True], [True, False]]))
    assert distance[0, 0] == 5
    assert valid.tolist() == [[True, False]]
    distance, valid, pairs = pair_targets(xy[:1], torch.ones(1, 2, dtype=torch.bool))
    assert not len(distance)


def test_all_pretraining_heads_backpropagate_and_probe_can_fit(ds):
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(20261020)
    vocab = ConditionVocabulary.from_dataset(ds)
    sample = load_sample(ds, "fit", 0, vocab)
    encoder = WorldEncoder(len(ds.item_vocabulary)+2, vocab.sizes)
    heads = PretrainHeads()
    total, parts, support = pretrain_losses(encoder, heads, sample, 123)
    assert total is not None and torch.isfinite(total)
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert heads.motion.weight.grad.abs().sum() > 0
    assert heads.health.weight.grad.abs().sum() > 0
    with torch.no_grad():
        latent = encoder(sample.inputs)
    probe = PathProbe()
    opt = torch.optim.AdamW(probe.parameters(), lr=.003, foreach=False)
    target = supervised_targets(sample)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = masked_mse(probe(latent, sample.inputs), target["delta"], target["mask"][..., None])
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]*.25


def test_vectorized_audit_horizons_match_loader(ds):
    from world_model.preprocess import targets_at
    b = ds.bank("fit")
    for k, ego in b["anchors"][::7]:
        expected = targets_at(b, int(k), int(ego))
        q = int(k)+np.arange(1, 7)
        qc = np.minimum(q, len(b["times"])-1)
        members = expected["members"]
        valid = (b["target_valid"][qc][:, members] & (q < len(b["times"]))[:, None]
                 & (b["target_episode"][qc][:, members] == b["episode"][k, members])
                 & (b["target_segment"][qc][:, members] == b["segment"][k, members])
                 & b["valid"][k, members]).T
        assert np.array_equal(valid, expected["mask"])


def test_metric_units_oracle_and_missing_labels(ds):
    from tools.train_world_model_g2 import evaluate_prediction
    s = load_sample(ds, "fit", 0, ConditionVocabulary.from_dataset(ds))
    y = supervised_targets(s)
    exact, _, _, _ = evaluate_prediction(s, y["delta"])
    assert exact["ADE"] < 1e-4
    # A common +3m,+4m translation has 5m path error but preserves pair geometry.
    shifted, _, _, _ = evaluate_prediction(s, y["delta"]+torch.tensor([.03, .04]))
    assert shifted["ADE"] == pytest.approx(5., abs=1e-4)
    assert shifted["FDE"] == pytest.approx(5., abs=1e-4)
    if shifted["pair_MAE"] is not None:
        assert shifted["pair_MAE"] < 1e-4
    s.targets["mask"][:] = False
    empty, _, _, _ = evaluate_prediction(s, y["delta"])
    assert empty["ADE"] is None and empty["FDE"] is None and empty["pair_MAE"] is None
