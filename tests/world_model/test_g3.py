import numpy as np
import pytest
import torch
from dataclasses import fields
import test_dataset
from world_model.cohorts import classify_match, movement_events, label_support, load_at, CohortDataset, candidate_windows
from world_model.inputs import ConditionVocabulary
from world_model.indexed_dataset import IndexedWorldDataset
from world_model.historical_prior import HistoricalPrior
import sqlite3
import json


def test_classification_does_not_pool_ai_or_conflicting_maps():
    base = dict(map="Baltic_Main", game_mode="squad", br_rule_candidate="1")
    assert classify_match({**base, "match_type": "official"}) == "normal_br"
    assert classify_match({**base, "match_type": "competitive"}) == "ranked_br"
    assert classify_match({**base, "match_type": "airoyale"}) == "airoyale"
    assert classify_match({**base, "match_type": "tutorialatoz"}) == "unresolved"
    assert classify_match({**base, "map": "Tutorial_Main", "match_type": "tutorialatoz"}) == "tutorial"
    assert classify_match({**base, "map": "Range_Main", "match_type": "trainingroom"}) == "training"
    assert classify_match({**base, "match_type": "mystery"}) == "unresolved"


def test_emergency_runs_and_direct_inferred_are_separate():
    b = {"times": np.arange(7)*10, "mobility": np.array([[0], [4], [4], [0], [4], [4], [0]]),
         "valid": np.ones((7, 1), bool), "team": np.array([3]),
         "episode": np.zeros((7, 1), int), "source_time": np.arange(7)[:, None]*10}
    events = movement_events(b, {"issues": [[0, 20, 20, "reentry_direct", "evidence"],
                                           [0, 50, 50, "reentry_inferred", "inference"]]})
    assert [e["kind"] for e in events].count("emergency_pickup") == 2
    assert [e["kind"] for e in events].count("reentry_direct") == 1
    assert [e["kind"] for e in events].count("reentry_inferred") == 1


def test_future_episode_boundary_masks_paths_and_zero_is_not_missing():
    b = {"times": np.arange(20)*10, "team": np.array([1]), "known": np.ones((20, 1), bool),
         "valid": np.ones((20, 1), bool), "episode": np.zeros((20, 1), int),
         "segment": np.zeros((20, 1), int), "target_valid": np.ones((20, 1), bool),
         "target_episode": np.zeros((20, 1), int), "target_segment": np.zeros((20, 1), int),
         "target_mode": np.ones((20, 1), int)}
    assert label_support(b, 12, 0)["ego_first30"]
    b["target_episode"][14:] = 1
    assert not label_support(b, 12, 0)["ego_first30"]
    assert label_support(b, 12, 0)["ego_points"] == 1
    # A subsequent return must not rescue the earlier return's missing labels.
    b["episode"][14:] = 1
    event = dict(actor=0, start_s=120, end_s=120, event_episode=0)
    assert all(c["frame"] < 14 for c in candidate_windows(b, event))


def test_event_metadata_cannot_enter_encoder_and_overlay_is_restored():
    helper = test_dataset.DatasetTests(); helper.setUp()
    try:
        ds = IndexedWorldDataset(helper.root, role="validation")
        mid = ds.match_ids[0]; b = ds.bank(mid)
        original = b["anchors"].copy()
        vocab = ConditionVocabulary.from_dataset(ds)
        a = load_at(ds, mid, 13, 0, vocab)
        # No event manifest is accepted by load_at. Future labels may change,
        # encoder input must not change.
        b["target"][14:] += 10000
        c = load_at(ds, mid, 13, 0, vocab)
        for f in fields(a.inputs):
            x, y = getattr(a.inputs, f.name), getattr(c.inputs, f.name)
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y), f.name
            else:
                assert x == y
        assert np.array_equal(ds.bank(mid)["anchors"], original)
        with pytest.raises(PermissionError):
            load_at(ds, "protected", 13, 0, vocab)
    finally:
        helper.tearDown()


def test_historical_prior_rejects_time_map_self_and_fold_leakage(tmp_path):
    path = tmp_path/"prior.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE donors(match_id TEXT,role TEXT,cohort TEXT,map TEXT,patch TEXT,start_ts REAL,end_ts REAL);
        CREATE TABLE contributions(match_id TEXT,map TEXT,sx INTEGER,sy INTEGER,dx INTEGER,dy INTEGER,mode INTEGER,n INTEGER);
        """)
        donors = [("past", 10), ("late", 100), ("self", 10), ("inner_heldout", 10)]
        for mid, end in donors:
            db.execute("INSERT INTO donors VALUES(?,?,?,?,?,?,?)", (mid, "train", "normal_br", "Baltic_Main", "40.2", 0, end))
            db.execute("INSERT INTO contributions VALUES(?,?,?,?,?,?,?,?)", (mid, "Baltic_Main", 0, 0, 1, 0, 1, 7))
    prior = HistoricalPrior(path, {"past", "late", "self"})
    kwargs = dict(map_name="Baltic_Main", target_mid="self", target_start=50,
                  target_patch="41.1", cells=[(0, 0), (1, 0)])
    r = prior.query(**kwargs)
    assert json.loads(json.dumps(prior.query(**{**kwargs, "cells": np.array(kwargs["cells"])}))) == r
    assert len(r) == 1 and r[0]["count"] == 7 and r[0]["donor_matches"] == 1
    assert r[0]["known_change"] == 1 and not r[0]["geometry_verified"]
    assert not prior.query(**{**kwargs, "map_name": "Tiger_Main"})
    assert not prior.query(**{**kwargs, "target_start": 10})  # strict <
    prior.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE contributions SET n=999999 WHERE match_id='late'")
    prior = HistoricalPrior(path, {"past", "late", "self"})
    assert prior.query(**kwargs) == r
    prior.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE donors SET role='validation' WHERE match_id='late'")
    with pytest.raises(PermissionError):
        HistoricalPrior(path, {"past"})


def test_cohort_cannot_promote_validation_into_fit():
    helper = test_dataset.DatasetTests(); helper.setUp()
    try:
        full = json.loads((helper.root/"ablations/full.json").read_text())
        path = helper.root/"cohort.json"
        valid = {**full, "cohort": "primary_br"}
        path.write_text(json.dumps(valid))
        ds = CohortDataset(helper.root, path, "validation")
        assert ds.fit_ids == set(full["train_match_ids"])
        invalid = {**valid, "train_match_ids": full["validation_match_ids"]}
        path.write_text(json.dumps(invalid))
        with pytest.raises(PermissionError):
            CohortDataset(helper.root, path)
    finally:
        helper.tearDown()
