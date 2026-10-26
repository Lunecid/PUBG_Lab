"""PTR data layer: schema, standardization, feature-block removal, collate, relations, shuffle.

All DB-free: inputs come from ``model_stgcn.ptr.synthetic``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.models.relations as R
from pubg_lab.survival.ptr.data.synthetic import group_columns, make_dump

K = {name: i for i, name in enumerate(R.RELATION_TYPES)}


def event_alive_ref(m):
    """Independent re-implementation of the ghost-fix rule: alive at the start of slot k iff the last-kill
    death time is after the start of the phase, start(0) = 0 and start(k) = phase_end_s[k-1] (NaN = unknown)."""
    T = m["team_death_s"].double().numpy()
    end = m["phase_end_s"].double().numpy()
    start = m["phase_start_s"].double().numpy() if "phase_start_s" in m else np.r_[0.0, end[:-1]]
    with np.errstate(invalid="ignore"):
        alive = (~np.isnan(T))[None, :] & (T[None, :] > start[:, None])
    return torch.from_numpy(alive & m["phase_mask"].numpy()[:, None])


def alive_v2(m, dump=None):
    """Effective alive tokens under the default (v2) labels of a synthetic dump: the event-based mask for the
    default (new-format) dumps, node_mask raised over the v2 targets for legacy ones."""
    if "node_mask_v2" in m:  # new-format synthetic dumps carry the dump script's own node_mask_v2
        return event_alive_ref(m)
    return m["node_mask"] | m["target_mask_v2"]


# --------------------------------------------------------------------------- schema
def test_synthetic_dump_matches_schema():
    dump = make_dump(4, n_features=150, seed=1)
    D.validate_dump(dump)
    m = dump["matches"][0]
    P, N, F = m["X"].shape
    assert (P, F) == (8, 150)
    assert len(dump["meta"]["feature_names"]) == 150
    assert m["node_mask"].dtype == torch.bool and m["target_mask"].dtype == torch.bool
    assert m["zone"].shape == (8, 8) and m["zone_valid"].shape == (8,)
    assert m["team_pos"].shape == (P, N, 2)
    assert set(m["edges"]) == {"prox", "combat", "copress"}
    for idx, w in m["edges"].values():
        assert idx.dtype == torch.int64 and idx.shape[1] == 3 and w.shape == (idx.shape[0],)
    assert {"zone", "habitat"} <= set(dump["meta"]["feature_groups"])
    # target is a subset of alive-at-start, as in the label table (pipeline labels only)
    assert not (m["target_mask"] & ~m["node_mask"]).any()
    # v2.3 fields: event-based labels next to the pipeline ones, AI share, match type, death time
    assert m["target_mask_v2"].dtype == torch.bool and m["target_mask_v2"].shape == (P, N)
    assert m["y_hazard_v2"].dtype == torch.float32 and m["y_hazard_v2"].shape == (P, N)
    assert m["team_ai_frac"].dtype == torch.float32 and m["team_ai_frac"].shape == (N,)
    assert m["team_death_s"].shape == (N,) and isinstance(m["match_type"], str)
    assert {mm["match_type"] for mm in dump["matches"]} == {"official", "competitive"}


def test_group_columns_partition_all_features():
    for f in (4, 24, 150):
        groups = group_columns(f)
        cols = sorted(c for v in groups.values() for c in v)
        assert cols == list(range(f))
        assert all(len(v) >= 1 for v in groups.values())


def test_synthetic_v2_labels_follow_the_event_definition_of_the_design_doc():
    """Pins the fixture to design doc 4.2: target(k) = T > phase_end_s(k), y(k) = T <= phase_end_s(k+1),
    T = inf (NaN here) for the winner; slot P-1 is never a target."""
    legacy = make_dump(3, seed=40, event_death=False)["matches"]  # legacy format: NaN marks the winner
    assert all(torch.isnan(m["team_death_s"][m["team_won"]]).all() for m in legacy)
    for m in make_dump(10, seed=40)["matches"]:
        P = m["X"].shape[0]
        T = m["team_death_s"].numpy().astype(np.float64)
        T = np.where(np.isnan(T), np.inf, T)
        end = m["phase_end_s"].numpy().astype(np.float64)
        want = T[None, :] > end[:, None]
        want[P - 1] = False
        assert (want == m["target_mask_v2"].numpy()).all()
        y = np.zeros(want.shape)
        y[: P - 1] = T[None, :] <= end[1:, None]
        assert (np.where(want, y, 0.0) == m["y_hazard_v2"].numpy()).all()
        won = m["team_won"]
        assert won.sum() == 1 and torch.isinf(m["team_death_s"][won]).all()  # event format: +inf = alive at the end
        assert (m["final_rank_norm"][won] == 1.0 / len(won)).all()  # rank 1 = winner = smallest value


def test_synthetic_pipeline_labels_differ_from_v2_unless_noise_is_zero():
    noisy, clean = make_dump(8, seed=41), make_dump(8, seed=41, pipeline_noise=0.0)
    assert any(not torch.equal(m["target_mask"], m["target_mask_v2"]) for m in noisy["matches"])
    assert any(not torch.equal(m["y_hazard"], m["y_hazard_v2"]) for m in noisy["matches"])
    for m in clean["matches"]:
        assert torch.equal(m["target_mask"], m["target_mask_v2"]) and torch.equal(m["y_hazard"], m["y_hazard_v2"])
        assert torch.equal(m["node_mask"], m["node_mask"] | m["target_mask_v2"])
    # v2 targets that sit on a token the pipeline node_mask does not cover exist in the noisy dump
    assert sum(int((m["target_mask_v2"] & ~m["node_mask"]).sum()) for m in noisy["matches"]) > 0
    # ... while the pipeline labels keep their own invariant
    assert all(not (m["target_mask"] & ~m["node_mask"]).any() for m in noisy["matches"])


@pytest.mark.parametrize(
    "corrupt,labels",
    [
        (lambda d: d["matches"][0].__setitem__("X", d["matches"][0]["X"][:, :, :-1]), "v2"),
        (lambda d: d["matches"][0]["target_mask"].__ior__(~d["matches"][0]["node_mask"]), "pipeline"),
        (lambda d: d["matches"][0]["edges"].__setitem__(
            "prox", (torch.tensor([[0, 0, 999]]), torch.tensor([0.5]))
        ), "v2"),
        (lambda d: d["matches"][0].pop("zone"), "v2"),
        (lambda d: d["meta"].pop("feature_groups"), "v2"),
        # v2.3: loud failures instead of silent skips
        (lambda d: d["matches"][0].pop("team_ai_frac"), "v2"),
        (lambda d: d["matches"][0].pop("team_ai_frac"), "pipeline"),
        (lambda d: d["matches"][0].__setitem__("team_ai_frac", d["matches"][0]["team_ai_frac"][:-1]), "v2"),
        (lambda d: d["matches"][0].__setitem__("team_ai_frac", d["matches"][0]["team_ai_frac"] + 2.0), "v2"),
        (lambda d: d["matches"][0].__setitem__("team_ai_frac", d["matches"][0]["team_ai_frac"] - 1.5), "v2"),
        (lambda d: d["matches"][0].pop("match_type"), "v2"),
        (lambda d: d["matches"][0].__setitem__("match_type", None), "v2"),
        (lambda d: d["matches"][0].__setitem__("match_type", ""), "v2"),
        (lambda d: d["matches"][0].pop("target_mask_v2"), "v2"),
        (lambda d: d["matches"][0].pop("y_hazard_v2"), "v2"),
        (lambda d: d["matches"][0].__setitem__("y_hazard_v2", d["matches"][0]["y_hazard_v2"] + 0.5), "v2"),
        (lambda d: d["matches"][0].pop("target_mask"), "pipeline"),
        # the last phase slot can never be a target (evaluation rows are the cuts 1..7)
        (lambda d: d["matches"][0]["target_mask_v2"].__setitem__((-1, 0), True), "v2"),
        (lambda d: (d["matches"][0]["target_mask"].__setitem__((-1, 0), True),
                    d["matches"][0]["node_mask"].__setitem__((-1, 0), True)), "pipeline"),
    ],
)
def test_validate_dump_rejects_bad_schema(corrupt, labels):
    dump = make_dump(2, seed=2)
    corrupt(dump)
    with pytest.raises(ValueError):
        D.validate_dump(dump, labels=labels)


def test_validate_dump_error_names_the_missing_field_and_the_match():
    dump = make_dump(2, seed=2)
    dump["matches"][1].pop("team_ai_frac")
    with pytest.raises(ValueError, match=r"team_ai_frac") as exc:
        D.validate_dump(dump)
    assert dump["matches"][1]["match_id"] in str(exc.value)
    legacy = make_dump(2, seed=2, v2_fields=False)
    with pytest.raises(ValueError, match=r"y_hazard_v2|target_mask_v2"):
        D.validate_dump(legacy)  # default labels = v2
    D.validate_dump(legacy, labels="pipeline")  # a legacy dump is still fine for the pipeline labels
    with pytest.raises(ValueError, match="labels"):
        D.validate_dump(make_dump(2), labels="v3")


def test_load_cohort_roundtrip(tmp_path):
    from pubg_lab.survival.ptr.data.synthetic import write_cohort_dumps

    root = write_cohort_dumps(tmp_path, "c1", sizes={"train": 3, "val": 2, "test": 2}, seed=3)
    splits = D.load_cohort(tmp_path, "c1")
    assert set(splits) == {"train", "val", "test"}
    assert [len(splits[s]["matches"]) for s in ("train", "val", "test")] == [3, 2, 2]
    assert splits["val"]["meta"]["split"] == "val"
    assert root == tmp_path / "c1"
    with pytest.raises(FileNotFoundError, match="c1"):  # the message lists the cohorts that do exist
        D.load_cohort(tmp_path, "missing_cohort")


# --------------------------------------------------------------------------- standardization
def test_standardizer_uses_alive_tokens_only():
    train = make_dump(6, seed=3, dead_garbage=True)
    rows = np.concatenate(
        [m["X"].numpy()[alive_v2(m).numpy()] for m in train["matches"]]
    ).astype(np.float64)
    sz = D.Standardizer.fit(train)
    np.testing.assert_allclose(sz.x_mean, rows.mean(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sz.x_std, rows.std(0), rtol=1e-6, atol=1e-7)

    # garbage at dead tokens must not move the statistics
    for m in train["matches"]:
        m["X"][~alive_v2(m)] = 1e6
    sz2 = D.Standardizer.fit(train)
    np.testing.assert_allclose(sz2.x_mean, sz.x_mean)
    np.testing.assert_allclose(sz2.x_std, sz.x_std)
    # the pipeline labels define "alive" by the pipeline node_mask alone: other tokens, other statistics
    pipe = D.Standardizer.fit(train, labels="pipeline")
    assert not np.allclose(pipe.x_mean, sz.x_mean)


def test_standardizer_is_nan_aware_and_nan_becomes_zero():
    train = make_dump(6, seed=4, nan_frac=0.1)
    alive_rows = np.concatenate(
        [m["X"].numpy()[alive_v2(m).numpy()] for m in train["matches"]]
    ).astype(np.float64)
    assert np.isnan(alive_rows).any()
    sz = D.Standardizer.fit(train)
    np.testing.assert_allclose(sz.x_mean, np.nanmean(alive_rows, 0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sz.x_std, np.nanstd(alive_rows, 0), rtol=1e-6, atol=1e-7)

    ds = D.PTRDataset(train, sz, pos_in_token=False)
    F = train["matches"][0]["X"].shape[-1]
    for i in range(len(ds)):
        item = ds[i]
        assert torch.isfinite(item["x"]).all()
        nan_pos = torch.isnan(train["matches"][i]["X"])
        assert (item["x"][..., :F][nan_pos] == 0).all()  # the last column is team_ai_frac


def test_standardizer_scales_with_train_stats_and_floors_constant_columns():
    train = make_dump(4, seed=5)
    for m in train["matches"]:
        m["X"][..., 5] = 3.0  # constant column -> std floor, no inf/NaN
    sz = D.Standardizer.fit(train)
    assert sz.x_std[5] == 1.0 and np.isfinite(sz.x_std).all() and (sz.x_std > 0).all()
    ds = D.PTRDataset(train, sz, pos_in_token=False)
    item = ds[0]
    alive = item["node_mask"]
    assert torch.allclose(item["x"][..., 5][alive], torch.zeros(int(alive.sum())), atol=1e-6)
    # alive tokens of the fit set are (approximately) zero-mean / unit-std after scaling
    allx = torch.cat([ds[i]["x"][ds[i]["node_mask"]] for i in range(len(ds))])
    cols = [c for c in range(allx.shape[1]) if c != 5]
    assert allx[:, cols].mean(0).abs().max() < 1e-4
    assert (allx[:, cols].std(0, unbiased=False) - 1).abs().max() < 1e-3


def test_standardizer_state_dict_roundtrip():
    sz = D.Standardizer.fit(make_dump(3, seed=6))
    sz2 = D.Standardizer.from_state_dict(sz.state_dict())
    for name in ("x_mean", "x_std", "pos_mean", "pos_std", "zone_mean", "zone_std", "ai_mean", "ai_std"):
        np.testing.assert_array_equal(getattr(sz, name), getattr(sz2, name))


# --------------------------------------------------------------------------- feature blocks
def test_nohab_drops_exactly_group_columns():
    dump = make_dump(2, n_features=150, seed=7)
    groups = dump["meta"]["feature_groups"]
    keep = D.keep_columns(dump["meta"], ("zone", "habitat"))
    dropped = sorted(set(range(150)) - set(keep.tolist()))
    assert dropped == sorted(set(groups["zone"]) | set(groups["habitat"]))
    assert len(dropped) > 0 and (np.diff(keep) > 0).all()
    assert D.keep_columns(dump["meta"], ()).tolist() == list(range(150))
    with pytest.raises(KeyError):
        D.keep_columns(dump["meta"], ("not_a_group",))

    sz = D.Standardizer.fit(dump)
    full = D.PTRDataset(dump, sz)
    nohab = D.PTRDataset(dump, sz, drop_groups=("zone", "habitat"))
    assert full.n_features == 150 + 2 + 1 and nohab.n_features == len(keep) + 2 + 1  # + team_pos x/y + ai share
    torch.testing.assert_close(nohab[0]["x"][..., : len(keep)], full[0]["x"][..., keep])
    assert nohab.feature_names[: len(keep)] == [dump["meta"]["feature_names"][c] for c in keep]


# --------------------------------------------------------------------------- collate
def test_collate_pads_within_batch_and_masks_padding():
    dump = make_dump(3, team_range=(5, 9), seed=8)
    sz = D.Standardizer.fit(dump)
    ds = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES)
    batch = ds.get_batch([0, 1, 2])
    ns = [m["X"].shape[1] for m in dump["matches"]]
    nmax = max(ns)
    assert len(set(ns)) > 1, "test needs matches of different sizes"
    assert batch["x"].shape == (3, 8, nmax, ds.n_features)
    assert batch["rel"].shape == (3, 8, len(R.RELATION_TYPES), nmax, nmax)
    assert batch["match_ids"] == [m["match_id"] for m in dump["matches"]]
    for b, n in enumerate(ns):
        assert not batch["node_mask"][b, :, n:].any()
        assert not batch["target_mask"][b, :, n:].any()
        assert (batch["y_hazard"][b, :, n:] == 0).all()
        assert (batch["team_ids"][b, n:] == -1).all()
        assert batch["team_ids"][b, :n].tolist() == dump["matches"][b]["team_ids"].tolist()
        assert (batch["rel"][b, :, :, n:, :] == 0).all()
        assert (batch["rel"][b, :, :, :, n:] == 0).all()
        m = dump["matches"][b]
        torch.testing.assert_close(batch["node_mask"][b, :, :n], alive_v2(m))  # raised over the v2 targets
        torch.testing.assert_close(batch["target_mask"][b, :, :n], m["target_mask_v2"])
    assert batch["match_types"] == [m["match_type"] for m in dump["matches"]]
    # team_ai_frac is the last token column, constant over phases (a team property)
    ai_col = batch["x"][..., -1]
    assert (ai_col == ai_col[:, :1]).all()
    # labels are zeroed outside the target mask and everything float is finite
    assert (batch["y_hazard"][~batch["target_mask"]] == 0).all()
    for key, val in batch.items():
        if torch.is_tensor(val) and val.is_floating_point():
            assert torch.isfinite(val).all(), key
    assert batch["zone_valid"].dtype == torch.bool and batch["zone_valid"].shape == (3, 8)


def test_collate_sanitises_missing_positions_and_zone():
    dump = make_dump(2, seed=9, zone_missing_prob=0.0)
    dump["matches"][0]["team_pos"][1, 0] = float("nan")
    dump["matches"][0]["zone"][2] = float("nan")
    dump["matches"][1]["y_hazard_v2"][~dump["matches"][1]["target_mask_v2"]] = float("nan")
    sz = D.Standardizer.fit(dump)
    batch = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES).get_batch([0, 1])
    assert not batch["pos_valid"][0, 1, 0]
    assert not batch["zone_valid"][0, 2]
    assert torch.isfinite(batch["pos"]).all() and torch.isfinite(batch["zone"]).all()
    assert torch.isfinite(batch["zone_std"]).all() and torch.isfinite(batch["y_hazard"]).all()


def test_zone_layout_defaults_and_meta_override():
    assert D.zone_layout({}) == {name: i for i, name in enumerate(D.ZONE_COLUMNS)}
    names = list(reversed(D.ZONE_COLUMNS))
    assert D.zone_layout({"zone_columns": names})["cur_cx"] == 7
    with pytest.raises(ValueError):
        D.zone_layout({"zone_columns": ["a", "b"]})


def test_top10_is_ignored():
    """Design doc v2.1: the top-10 label is not used - not required, not carried into batches."""
    dump = make_dump(2, seed=14)
    assert "top10" in dump["matches"][0]  # the dump schema still carries it
    for m in dump["matches"]:
        m.pop("top10")
    D.validate_dump(dump)  # a dump without it is valid
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump))
    assert "top10" not in ds[0] and "top10" not in ds.get_batch([0, 1])


def test_dataset_without_relations_has_no_rel_tensor():
    dump = make_dump(2, seed=10)
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump))
    assert "rel" not in ds.get_batch([0, 1])


# --------------------------------------------------------------------------- relations
def _toy_edges():
    return {
        "prox": (
            torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 2], [0, 2, 0], [1, 0, 1], [1, 1, 0]]),
            torch.tensor([0.9, 0.9, 0.5, 0.5, 0.7, 0.7]),
        ),
        "combat": (torch.tensor([[0, 1, 0]]), torch.tensor([30.0])),  # phase 0: team 1 hit team 0
        "copress": (torch.tensor([[0, 0, 2], [0, 2, 0]]), torch.tensor([0.4, 0.4])),
    }


def test_escalation_zeroes_prox_when_combat_exists_in_either_direction():
    dense = R.densify_edges_batch([_toy_edges()], 2, 4)
    rel = R.build_relation_tensor(dense)
    assert rel.shape == (1, 2, 4, 4, 4)
    prox = rel[0, :, K["prox"]]
    assert prox[0, 0, 1] == 0 and prox[0, 1, 0] == 0  # escalated pair, both directions
    assert prox[0, 0, 2] == pytest.approx(0.5) and prox[0, 2, 0] == pytest.approx(0.5)
    assert prox[1, 0, 1] == pytest.approx(0.7)  # no combat at phase 1
    out, inn = rel[0, :, K["combat_out"]], rel[0, :, K["combat_in"]]
    assert out[0, 1, 0] == pytest.approx(np.log1p(30.0)) and out[0, 0, 1] == 0
    assert inn[0, 0, 1] == pytest.approx(np.log1p(30.0)) and inn[0, 1, 0] == 0
    cop = rel[0, :, K["copress"]]
    assert cop[0, 0, 2] == pytest.approx(0.4) and cop[0, 2, 0] == pytest.approx(0.4)


def test_combat_edge_with_zero_weight_still_escalates():
    edges = _toy_edges()
    edges["combat"] = (torch.tensor([[0, 0, 1]]), torch.tensor([0.0]))
    rel = R.build_relation_tensor(R.densify_edges_batch([edges], 2, 4))
    assert rel[0, 0, K["prox"], 0, 1] == 0 and rel[0, 0, K["prox"], 1, 0] == 0


def test_escalation_only_when_combat_channels_are_present():
    dense = R.densify_edges_batch([_toy_edges()], 2, 4)
    rel = R.build_relation_tensor(dense, ("prox", "copress"))
    assert rel.shape[2] == 2
    assert rel[0, 0, 0, 0, 1] == pytest.approx(0.9)  # prox untouched: no combat information leaks
    rel = R.build_relation_tensor(dense, ("prox", "combat_out"))
    assert rel[0, 0, 0, 0, 1] == 0


def test_symmetric_types_symmetrised_combat_stays_directed():
    one = R.densify_edges_batch([{"prox": (torch.tensor([[0, 0, 1]]), torch.tensor([0.8]))}], 1, 3)
    two = R.densify_edges_batch(
        [{"prox": (torch.tensor([[0, 0, 1], [0, 1, 0]]), torch.tensor([0.8, 0.8]))}], 1, 3
    )
    torch.testing.assert_close(one["prox"], two["prox"])
    assert one["prox"][0, 0, 0, 1] == one["prox"][0, 0, 1, 0] == pytest.approx(0.8)
    c = R.densify_edges_batch(
        [{"combat": (torch.tensor([[0, 0, 1], [0, 0, 1]]), torch.tensor([5.0, 7.0]))}], 1, 3
    )
    assert c["combat"][0, 0, 0, 1] == 7.0  # duplicates collapse with max
    assert c["combat"][0, 0, 1, 0] == 0
    assert c["combat_exists"][0, 0, 0, 1] and not c["combat_exists"][0, 0, 1, 0]
    # diagonal is removed, empty / missing edge types are fine
    d = R.densify_edges_batch([{"prox": (torch.tensor([[0, 1, 1]]), torch.tensor([0.5]))}], 1, 3)
    assert d["prox"].sum() == 0 and d["combat"].sum() == 0 and not d["combat_exists"].any()
    e = R.densify_edges_batch(
        [{"prox": (torch.zeros((0, 3), dtype=torch.int64), torch.zeros(0))}], 1, 3
    )
    assert e["prox"].sum() == 0


def test_edge_weights_are_sanitised():
    edges = {"copress": (torch.tensor([[0, 0, 1]]), torch.tensor([float("nan")]))}
    rel = R.build_relation_tensor(R.densify_edges_batch([edges], 1, 2))
    assert torch.isfinite(rel).all() and rel.sum() == 0


# --------------------------------------------------------------------------- shuffle
def _signature(edges, n_phases, n_nodes):
    """Per phase: sorted multiset of per-node (prox, combat_out, combat_in, copress) degrees."""
    deg = np.zeros((n_phases, n_nodes, 4), dtype=np.int64)
    for name, col in (("prox", 0), ("copress", 3)):
        idx, _ = edges[name]
        for p, i, _j in idx.tolist():
            deg[p, i, col] += 1  # edges stored in both directions -> per-node undirected degree
    idx, _ = edges["combat"]
    for p, i, j in idx.tolist():
        deg[p, i, 1] += 1
        deg[p, j, 2] += 1
    return [sorted(map(tuple, deg[p].tolist())) for p in range(n_phases)]


def test_shuffle_preserves_degree_multiset_and_stays_within_match_and_phase():
    dump = make_dump(5, team_range=(6, 12), seed=11)
    moved = 0
    for m in dump["matches"]:
        P, N, _ = m["X"].shape
        alive_np = alive_v2(m).numpy()  # the alive tokens the dataset uses (v2 labels): where edges live
        perms = R.shuffle_permutations(m["match_id"], 42, alive_np)
        assert perms.shape == (P, N)
        for p in range(P):
            alive = alive_np[p]
            assert sorted(perms[p][alive].tolist()) == np.flatnonzero(alive).tolist()
            assert (perms[p][~alive] == np.flatnonzero(~alive)).all()  # dead teams never move
        shuffled = R.shuffle_edges(m["edges"], perms)
        for name, (idx0, w0) in m["edges"].items():
            idx1, w1 = shuffled[name]
            assert idx1.shape == idx0.shape
            assert torch.equal(idx1[:, 0], idx0[:, 0])  # same phase for every edge
            assert torch.equal(w1, w0)  # weights untouched
            assert (idx1[:, 1:] >= 0).all() and (idx1[:, 1:] < N).all()  # same match (same N)
            assert alive_v2(m)[idx1[:, 0], idx1[:, 1]].all()  # endpoints alive at that phase
            assert alive_v2(m)[idx1[:, 0], idx1[:, 2]].all()
            moved += int((idx1 != idx0).any())
        # the *same* permutation is applied to every relation type: the joint degree multiset holds
        assert _signature(shuffled, P, N) == _signature(m["edges"], P, N)
    assert moved > 0, "shuffle should actually change the edge endpoints"


def test_shuffle_applies_one_permutation_to_all_relation_matrices():
    dump = make_dump(3, team_range=(8, 12), seed=12)
    sz = D.Standardizer.fit(dump)
    plain = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES)
    shuf = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES, shuffle_seed=7)
    a, b = plain.get_batch([0, 1, 2]), shuf.get_batch([0, 1, 2])
    assert not torch.equal(a["rel"], b["rel"])
    for bi in range(3):
        # per-(match, phase, type) multiset of weights is preserved, including prox after
        # escalation: that only holds if prox and combat were relabelled consistently.
        for p in range(8):
            for r in range(len(R.RELATION_TYPES)):
                x = torch.sort(a["rel"][bi, p, r].flatten()).values
                y = torch.sort(b["rel"][bi, p, r].flatten()).values
                torch.testing.assert_close(x, y)
    # nothing leaks outside the match block
    ns = [m["X"].shape[1] for m in dump["matches"]]
    for bi, n in enumerate(ns):
        assert (b["rel"][bi, :, :, n:, :] == 0).all() and (b["rel"][bi, :, :, :, n:] == 0).all()


def test_shuffle_is_fixed_per_match_phase_seed():
    alive = np.ones((8, 10), dtype=bool)
    a = R.shuffle_permutations("match-1", 42, alive)
    assert (a == R.shuffle_permutations("match-1", 42, alive)).all()
    assert (a != R.shuffle_permutations("match-1", 43, alive)).any()
    assert (a != R.shuffle_permutations("match-2", 42, alive)).any()
    assert any((a[0] != a[p]).any() for p in range(1, 8))  # not one permutation reused everywhere

    dump = make_dump(2, seed=13)
    sz = D.Standardizer.fit(dump)
    ds = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES, shuffle_seed=3)
    e1, e2 = ds[0]["edges"], ds[0]["edges"]
    for name in e1:
        assert torch.equal(e1[name][0], e2[name][0])  # identical on every access (every epoch)


# --------------------------------------------------------------------------- v2.2: diagonal, combat modes
def test_relation_diagonal_is_zero_even_when_the_dump_lists_self_loops():
    dump = make_dump(2, seed=15)
    m = dump["matches"][0]
    for name, (idx, w) in list(m["edges"].items()):
        loops = torch.tensor([[0, 2, 2], [1, 3, 3]])
        m["edges"][name] = (torch.cat([idx, loops]), torch.cat([w, torch.tensor([0.9, 0.9])]))
    D.validate_dump(dump)
    rel = D.PTRDataset(dump, D.Standardizer.fit(dump), relation_types=R.RELATION_TYPES).get_batch([0, 1])["rel"]
    assert (torch.diagonal(rel, dim1=-2, dim2=-1) == 0).all()
    assert rel.abs().sum() > 0


def test_combat_weight_modes_give_the_same_channels_from_either_dump_convention():
    """The real dump stores ln(1 + intensity) (stdb/phase_edge_etl.py:166); a raw-intensity dump must give
    the same model input, so the channel never depends on how the dump script stores the weight."""
    raw = make_dump(3, seed=16, combat_weights="raw")
    log = make_dump(3, seed=16, combat_weights="log")
    mode_raw, max_raw = D.infer_combat_weights(raw)
    mode_log, max_log = D.infer_combat_weights(log)
    assert (mode_raw, mode_log) == ("raw", "log")
    assert max_raw > D.COMBAT_LOG_MAX >= max_log > 0
    a = D.PTRDataset(raw, D.Standardizer.fit(raw), relation_types=R.RELATION_TYPES, combat_weights="raw")
    b = D.PTRDataset(log, D.Standardizer.fit(log), relation_types=R.RELATION_TYPES, combat_weights="log")
    ra, rb = a.get_batch([0, 1, 2])["rel"], b.get_batch([0, 1, 2])["rel"]
    torch.testing.assert_close(ra, rb, atol=1e-5, rtol=1e-5)
    # the wrong mode double-compresses an already-log weight
    wrong = D.PTRDataset(log, D.Standardizer.fit(log), relation_types=R.RELATION_TYPES, combat_weights="raw")
    rw = wrong.get_batch([0, 1, 2])["rel"]
    assert not torch.allclose(ra, rw)
    k = R.RELATION_TYPES.index("combat_out")
    assert rw[:, :, k].max() < ra[:, :, k].max()
    # prox / copress are untouched by the mode
    for name in ("prox", "copress"):
        j = R.RELATION_TYPES.index(name)
        torch.testing.assert_close(ra[:, :, j], rw[:, :, j])
    with pytest.raises(ValueError):
        D.PTRDataset(log, D.Standardizer.fit(log), combat_weights="exp")
    with pytest.raises(ValueError):
        R.build_relation_tensor(R.densify_edges_batch([_toy_edges()], 2, 4), combat_weights="exp")


def test_combat_channels_are_log1p_of_intensity_for_both_modes():
    edges = {"combat": (torch.tensor([[0, 1, 0]]), torch.tensor([30.0]))}
    dense = R.densify_edges_batch([edges], 1, 3)
    k = {n: i for i, n in enumerate(R.RELATION_TYPES)}
    raw = R.build_relation_tensor(dense, combat_weights="raw")[0, 0]
    assert raw[k["combat_out"], 1, 0] == pytest.approx(np.log1p(30.0))
    assert raw[k["combat_in"], 0, 1] == pytest.approx(np.log1p(30.0))
    log = R.build_relation_tensor(dense, combat_weights="log")[0, 0]  # already ln(1 + intensity)
    assert log[k["combat_out"], 1, 0] == pytest.approx(30.0)
    assert log[k["combat_in"], 0, 1] == pytest.approx(30.0)
    assert D.infer_combat_weights([{"edges": {}}]) == ("log", 0.0)  # no combat edges: nothing to transform


# --------------------------------------------------------------------------- v2.2: noHab keeps alive counts
def test_nohab_keeps_alive_count_variables_that_sit_inside_the_groups():
    dump = make_dump(3, n_features=24, seed=17, alive_count_names=True)
    meta = dump["meta"]
    names, g = meta["feature_names"], meta["feature_groups"]
    zone, hab = g["zone"], g["habitat"]
    alive_cols = {zone[0], zone[1], hab[0]}
    assert {names[c] for c in alive_cols} == set(D.ALIVE_COUNT_FEATURES)

    plain = D.removed_columns(meta, ("zone", "habitat"))  # no exception list: the exact groups
    assert sorted(plain.tolist()) == sorted(set(zone) | set(hab))
    removed = D.removed_columns(meta, ("zone", "habitat"), D.ALIVE_COUNT_FEATURES)
    assert set(removed.tolist()) == (set(zone) | set(hab)) - alive_cols == {zone[2], hab[1], hab[2]}
    keep = D.keep_columns(meta, ("zone", "habitat"), D.ALIVE_COUNT_FEATURES)
    assert alive_cols <= set(keep.tolist()) and set(keep.tolist()).isdisjoint(removed.tolist())
    assert len(keep) + len(removed) == 24
    assert D.keep_columns(meta, ("zone", "habitat"), ("no_such_feature",)).tolist() == D.keep_columns(meta, ("zone", "habitat")).tolist()

    rep = D.feature_removal_report(meta, ("zone", "habitat"), D.ALIVE_COUNT_FEATURES)
    assert rep["removed_columns"] == ["safe_radius_start_norm", "hab_safe_area_per_alive_team", "hab_loss_ratio"]
    assert rep["removed_indices"] == sorted(removed.tolist()) and rep["n_removed"] == 3
    assert set(rep["kept_by_exception"]) == set(D.ALIVE_COUNT_FEATURES)  # the spatial ratio is NOT kept
    assert rep["drop_groups"] == ["zone", "habitat"]

    sz = D.Standardizer.fit(dump)
    ds = D.PTRDataset(dump, sz, drop_groups=("zone", "habitat"), keep_names=D.ALIVE_COUNT_FEATURES)
    assert ds.removal == rep and ds.n_features == 24 - 3 + 2 + 1
    assert set(D.ALIVE_COUNT_FEATURES) <= set(ds.feature_names)
    assert "hab_safe_area_per_alive_team" not in ds.feature_names
    assert D.PTRDataset(dump, sz).removal["n_removed"] == 0  # nothing dropped: empty report


def test_nohab_on_the_real_feature_layout_never_removes_the_alive_count_variables():
    """Guards the removal against drift of the real 150-column layout (read-only import of the dump script).

    The dump script's groups may or may not contain the alive-count variables (they did in the first version
    and no longer do); either way noHab must remove exactly the spatial zone / habitat columns."""
    try:
        import pubg_lab.survival.data.dump_phase_tier as real

        groups, names = real.build_feature_groups(), list(real.FEATURE_NAMES)
    except Exception as exc:  # the dump script is owned by another workstream
        pytest.skip(f"real feature layout unavailable: {exc}")
    meta = {"feature_names": names, "feature_groups": groups}
    inside = set(groups["zone"]) | set(groups["habitat"])
    spared = {names[c] for c in inside if names[c] in D.ALIVE_COUNT_FEATURES}  # inside the groups, kept anyway
    rep = D.feature_removal_report(meta, ("zone", "habitat"), D.ALIVE_COUNT_FEATURES)
    assert not set(rep["removed_columns"]) & set(D.ALIVE_COUNT_FEATURES)  # never removed
    assert set(rep["kept_by_exception"]) == spared
    assert rep["n_removed"] == len(inside) - len(spared) == len(rep["removed_columns"])
    assert "hab_safe_area_per_alive_team" in rep["removed_columns"]  # a spatial ratio, not a count
    assert len(names) == 150 and set(D.ALIVE_COUNT_FEATURES) <= set(names)
    keep = D.keep_columns(meta, ("zone", "habitat"), D.ALIVE_COUNT_FEATURES)
    assert {names[c] for c in keep} >= set(D.ALIVE_COUNT_FEATURES)
    assert len(keep) == 150 - rep["n_removed"]


# =========================================================================== v2.3: labels, AI share, match type
def test_default_labels_are_v2_and_every_entry_point_agrees():
    assert D.DEFAULT_LABELS == "v2" and D.LABEL_CHOICES == ("v2", "pipeline")
    assert D.LABEL_FIELDS == {"v2": ("y_hazard_v2", "target_mask_v2"), "pipeline": ("y_hazard", "target_mask")}
    dump = make_dump(3, seed=50)
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump))
    assert ds.labels == "v2"
    with pytest.raises(ValueError, match="labels"):
        D.PTRDataset(dump, D.Standardizer.fit(dump), labels="truth")
    with pytest.raises(ValueError, match="labels"):
        D.select_labels(dump["matches"][0], "truth")


def test_label_choice_selects_different_rows_when_v2_and_pipeline_differ():
    """The risk set (target rows), the labels and the alive tokens all follow the one label choice."""
    dump = make_dump(10, seed=51)  # pipeline_noise > 0: the two label sets differ
    sz = D.Standardizer.fit(dump)
    v2 = D.PTRDataset(dump, sz, labels="v2")
    pipe = D.PTRDataset(dump, sz, labels="pipeline")
    n_diff_rows = n_diff_y = 0
    for i, m in enumerate(dump["matches"]):
        a, b = v2[i], pipe[i]
        assert torch.equal(a["target_mask"], m["target_mask_v2"])
        assert torch.equal(b["target_mask"], m["target_mask"])
        zero = torch.zeros(())
        assert torch.equal(a["y_hazard"], torch.where(m["target_mask_v2"], m["y_hazard_v2"], zero))
        assert torch.equal(b["y_hazard"], torch.where(m["target_mask"], m["y_hazard"], zero))
        n_diff_rows += int((a["target_mask"] != b["target_mask"]).sum())
        both = a["target_mask"] & b["target_mask"]
        n_diff_y += int((a["y_hazard"][both] != b["y_hazard"][both]).sum())
    assert n_diff_rows > 0, "v2 and pipeline must select different risk-set rows on this dump"
    assert n_diff_y > 0, "v2 and pipeline must disagree on some labels of shared rows"
    # batches carry the same selection into the loss
    ba, bb = v2.get_batch([0, 1, 2]), pipe.get_batch([0, 1, 2])
    assert not torch.equal(ba["target_mask"], bb["target_mask"])
    # identical label sets (pipeline_noise=0) -> identical selections
    clean = make_dump(5, seed=51, pipeline_noise=0.0)
    csz = D.Standardizer.fit(clean)
    c2, cp = D.PTRDataset(clean, csz, labels="v2"), D.PTRDataset(clean, csz, labels="pipeline")
    for i in range(5):
        assert torch.equal(c2[i]["target_mask"], cp[i]["target_mask"])
        assert torch.equal(c2[i]["y_hazard"], cp[i]["y_hazard"])


def test_effective_alive_mask_is_raised_over_the_selected_targets():
    """A v2 target sitting on a token the pipeline node_mask does not cover would get a constant output
    from a dead token: the loader raises node_mask over the selected risk set (the dump script does the same
    for the pipeline labels)."""
    dump = make_dump(10, seed=52, event_death=False)  # legacy dump: the fallback mask
    gap = sum(int((m["target_mask_v2"] & ~m["node_mask"]).sum()) for m in dump["matches"])
    assert gap > 0, "test needs v2 targets outside the pipeline node_mask"
    sz = D.Standardizer.fit(dump)
    v2, pipe = D.PTRDataset(dump, sz, labels="v2"), D.PTRDataset(dump, sz, labels="pipeline")
    assert not D.uses_event_alive(dump, "v2") and v2.alive_source == "node_mask|target_v2"
    for i, m in enumerate(dump["matches"]):
        a = v2[i]
        assert not (a["target_mask"] & ~a["node_mask"]).any()
        assert torch.equal(a["node_mask"], m["node_mask"] | m["target_mask_v2"])
        assert torch.equal(pipe[i]["node_mask"], m["node_mask"])  # pipeline: already a superset
        assert D.effective_alive(m, "v2", event=False).equal(a["node_mask"])
    batch = v2.get_batch(list(range(len(v2))))
    assert not (batch["target_mask"] & ~batch["node_mask"]).any()


def test_selecting_v2_on_a_legacy_dump_fails_loudly_pipeline_still_works():
    legacy = make_dump(3, seed=53, v2_fields=False)
    with pytest.raises(ValueError, match=r"target_mask_v2|y_hazard_v2"):
        D.select_labels(legacy["matches"][0], "v2")
    with pytest.raises(ValueError, match=r"target_mask_v2|y_hazard_v2"):
        D.Standardizer.fit(legacy)
    with pytest.raises(ValueError, match=r"target_mask_v2|y_hazard_v2"):
        D.PTRDataset(legacy, D.Standardizer.fit(legacy, labels="pipeline"))
    sz = D.Standardizer.fit(legacy, labels="pipeline")
    ds = D.PTRDataset(legacy, sz, labels="pipeline")
    assert ds.get_batch([0, 1])["target_mask"].any()


def test_load_cohort_validates_the_selected_labels(tmp_path):
    from pubg_lab.survival.ptr.data.synthetic import write_cohort_dumps

    write_cohort_dumps(tmp_path, 'archive/legacy_snapshot', sizes={"train": 3, "val": 2, "test": 2}, seed=54, v2_fields=False)
    with pytest.raises(ValueError, match=r"target_mask_v2|y_hazard_v2"):
        D.load_cohort(tmp_path, 'archive/legacy_snapshot')  # default v2
    assert set(D.load_cohort(tmp_path, 'archive/legacy_snapshot', labels="pipeline")) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="labels"):
        D.load_cohort(tmp_path, 'archive/legacy_snapshot', labels="v3")


def test_team_ai_frac_is_appended_to_every_token_and_broadcast_over_phases():
    dump = make_dump(5, seed=55, ai_nan_frac=0.25)
    assert any(torch.isnan(m["team_ai_frac"]).any() for m in dump["matches"])
    sz = D.Standardizer.fit(dump)
    ds = D.PTRDataset(dump, sz)
    assert D.AI_FEATURE_NAME == "team_ai_frac"
    assert ds.feature_names[-3:] == ["team_pos_x", "team_pos_y", "team_ai_frac"]
    assert ds.n_features == 24 + 3
    # statistics: alive tokens (effective mask), NaN-aware, float64
    vals = np.concatenate(
        [np.broadcast_to(m["team_ai_frac"].numpy()[None, :], alive_v2(m).shape)[alive_v2(m).numpy()] for m in dump["matches"]]
    ).astype(np.float64)
    assert np.isnan(vals).any()
    np.testing.assert_allclose(sz.ai_mean, [np.nanmean(vals)], rtol=1e-6)
    np.testing.assert_allclose(sz.ai_std, [np.nanstd(vals)], rtol=1e-6)
    for i, m in enumerate(dump["matches"]):
        col = ds[i]["x"][..., -1]  # [P, N]
        assert (col == col[:1]).all(), "a team property: identical in every phase slot"
        ai = m["team_ai_frac"].numpy().astype(np.float64)
        want = np.nan_to_num((ai - sz.ai_mean[0]) / sz.ai_std[0], nan=0.0)
        np.testing.assert_allclose(col[0].numpy(), want, rtol=1e-5, atol=1e-6)
        assert (col[0][torch.isnan(m["team_ai_frac"])] == 0).all()  # NaN -> 0 after scaling
        assert torch.isfinite(col).all()
    # without the position columns it is still the last column
    ds2 = D.PTRDataset(dump, sz, pos_in_token=False)
    assert ds2.n_features == 24 + 1 and ds2.feature_names[-1] == "team_ai_frac"
    torch.testing.assert_close(ds2[0]["x"][..., -1], ds[0]["x"][..., -1])
    # it is part of every arm's input (also noHab: it belongs to no removed group)
    nohab = D.PTRDataset(dump, sz, drop_groups=("zone", "habitat"))
    assert nohab.feature_names[-1] == "team_ai_frac" and nohab.n_features == len(nohab.keep) + 3


def test_team_ai_frac_constant_or_all_missing_does_not_break_scaling():
    dump = make_dump(3, seed=56)
    for m in dump["matches"]:
        m["team_ai_frac"][:] = 0.5  # constant -> std floor 1, standardised value exactly 0
    sz = D.Standardizer.fit(dump)
    assert sz.ai_std[0] == 1.0 and sz.ai_mean[0] == pytest.approx(0.5)
    assert (D.PTRDataset(dump, sz)[0]["x"][..., -1] == 0).all()
    for m in dump["matches"]:
        m["team_ai_frac"][:] = float("nan")  # no team has participants: stats stay finite, tokens are 0
    sz = D.Standardizer.fit(dump)
    assert np.isfinite(sz.ai_mean).all() and sz.ai_std[0] == 1.0
    assert (D.PTRDataset(dump, sz)[0]["x"][..., -1] == 0).all()


def test_match_type_is_carried_into_every_batch():
    dump = make_dump(4, seed=57)
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump))
    assert ds[1]["match_type"] == "competitive"
    assert ds.get_batch([0, 1, 2, 3])["match_types"] == ["official", "competitive", "official", "competitive"]


def test_match_type_guard_enforces_the_registered_population():
    """official + competitive only (airoyale is excluded upstream; a leak must be loud, not silent)."""
    assert D.DEFAULT_MATCH_TYPES == ("official", "competitive")
    dump = make_dump(4, seed=58, match_types=("official", "competitive", "airoyale"))
    assert D.match_type_counts(dump) == {"official": 2, "competitive": 1, "airoyale": 1}
    with pytest.raises(ValueError, match="airoyale"):
        D.check_match_types(dump)
    D.check_match_types(dump, allowed=("official", "competitive", "airoyale"))
    D.check_match_types(dump, allowed=None)  # guard switched off
    D.check_match_types(make_dump(4, seed=58))


def test_load_cohort_applies_the_match_type_guard(tmp_path):
    from pubg_lab.survival.ptr.data.synthetic import write_cohort_dumps

    write_cohort_dumps(
        tmp_path, "leaky", sizes={"train": 4, "val": 3, "test": 3}, seed=59,
        match_types=("official", "airoyale"),
    )  # fmt: skip
    with pytest.raises(ValueError, match=r"airoyale.*leaky/(train|val|test)|leaky/(train|val|test).*airoyale"):
        D.load_cohort(tmp_path, "leaky")
    ok = D.load_cohort(tmp_path, "leaky", match_types=("official", "airoyale"))
    assert set(ok) == {"train", "val", "test"}
    assert set(D.load_cohort(tmp_path, "leaky", match_types=None)) == {"train", "val", "test"}


# =========================================================================== ghost fix (event-based alive mask)
def tiny_event_match():
    """Hand-made match: 4 teams, P = 5 slots; deaths 50 / 150 / +inf / NaN (unknown)."""
    inf, nan = float("inf"), float("nan")
    return {
        "team_death_s": torch.tensor([50.0, 150.0, inf, nan]),
        "phase_end_s": torch.tensor([100.0, 200.0, 300.0, nan, nan]),
        "phase_mask": torch.tensor([True, True, True, False, False]),
    }


def test_event_alive_mask_follows_team_death_s_and_phase_end_s():
    m = tiny_event_match()
    alive = D.event_alive_mask(m)
    assert alive.dtype == torch.bool and alive.shape == (5, 4)
    # start(0) = 0, start(k) = phase_end_s[k-1]: T > start
    assert alive[0].tolist() == [True, True, True, False]  # T > 0; the NaN (unknown) team is never alive
    assert alive[1].tolist() == [False, True, True, False]  # T > 100
    assert alive[2].tolist() == [False, False, True, False]  # T > 200
    assert not alive[3].any() and not alive[4].any()  # phases the match does not have
    # without phase_mask, NaN phase ends still keep absent slots empty
    bare = {k: v for k, v in m.items() if k != "phase_mask"}
    assert D.event_alive_mask(bare).equal(alive)
    # a present phase 1 keeps every known team alive, an absent first phase keeps none
    m2 = dict(m, phase_mask=torch.tensor([False, True, True, False, False]))
    assert not D.event_alive_mask(m2)[0].any()
    for mm in make_dump(8, seed=70)["matches"]:
        assert torch.equal(D.event_alive_mask(mm), event_alive_ref(mm))  # vs the independent re-implementation
        assert D.event_alive_mask(mm)[:, mm["team_won"]].all()  # the winner (+inf) is alive in every phase


def test_event_alive_equals_the_risk_set_shifted_by_one_cut():
    """alive at the start of slot k+1 <=> T > phase_end_s[k] <=> v2 target at cut k: a team is alive at a phase
    start iff it was at risk at the cut before (this is why the loader uses phase_end_s[k-1], not phase_start_s)."""
    for m in make_dump(8, seed=71)["matches"]:
        m = {k: v for k, v in m.items() if k != "phase_start_s"}  # the fallback rule (dumps without phase_start_s)
        alive, target = D.event_alive_mask(m), m["target_mask_v2"]
        P = alive.shape[0]
        assert torch.equal(alive[1:P], target[: P - 1])
        assert not (target & ~alive).any()  # every v2 target sits on an alive token


@pytest.mark.parametrize(
    "marker,expect",
    [("field", True), ("flag", True), ("both", True)],
)
def test_event_marker_detection(marker, expect):
    dump = make_dump(3, seed=72, event_marker=marker)
    assert D.uses_event_alive(dump, "v2") is expect
    assert D.uses_event_alive(dump, "pipeline") is False  # the pipeline labels never use it
    assert ("node_mask_v2" in dump["matches"][0]) == (marker in ("field", "both"))
    assert (dump["meta"].get("team_death_source") == "last_kill_event") == (marker in ("flag", "both"))


def test_legacy_dump_keeps_the_fallback_even_though_it_has_team_death_s():
    legacy = make_dump(3, seed=73, event_death=False)
    assert "team_death_s" in legacy["matches"][0] and "node_mask_v2" not in legacy["matches"][0]
    assert not D.uses_event_alive(legacy, "v2")  # time_survived-style death times are NOT event based
    assert D.alive_source(legacy, "v2") == "node_mask|target_v2"
    assert D.alive_source(legacy, "pipeline") == "node_mask|target_pipeline"
    new = make_dump(3, seed=73)
    assert D.alive_source(new, "v2") == "node_mask_v2" and D.alive_source(new, "pipeline") == "node_mask|target_pipeline"
    other = make_dump(3, seed=73, event_death=False)
    other["meta"]["team_death_source"] = "time_survived"  # a different declared source is not event based either
    assert not D.uses_event_alive(other, "v2")


def test_ghost_tokens_are_excluded_from_tokens_and_keys_under_event_based_v2():
    dump = make_dump(10, seed=74)
    ghosts = sum(int((m["node_mask"] & ~event_alive_ref(m)).sum()) for m in dump["matches"])
    assert ghosts > 0, "test needs ghost tokens (node_mask True although the events say dead)"
    sz = D.Standardizer.fit(dump)
    v2 = D.PTRDataset(dump, sz, labels="v2")
    pipe = D.PTRDataset(dump, sz, labels="pipeline")
    assert v2.alive_source == "node_mask_v2" and pipe.alive_source == "node_mask|target_pipeline"
    kept_by_pipeline = 0
    for i, m in enumerate(dump["matches"]):
        a = v2[i]["node_mask"]
        assert torch.equal(a, event_alive_ref(m))  # exactly the event mask, ghosts are gone
        assert not (a & ~event_alive_ref(m)).any()
        assert not (v2[i]["target_mask"] & ~a).any()  # every v2 target sits on an alive token
        ghost = m["node_mask"] & ~event_alive_ref(m)
        assert not (a & ghost).any()
        kept_by_pipeline += int((pipe[i]["node_mask"] & ghost).sum())  # the pipeline labels keep their ghosts
    assert kept_by_pipeline == ghosts
    # a ghost token no longer reaches the model: the batch masks (= tokens and attention keys) exclude it
    batch = v2.get_batch(list(range(len(v2))))
    for b, m in enumerate(dump["matches"]):
        n = m["X"].shape[1]
        assert torch.equal(batch["node_mask"][b, :, :n], event_alive_ref(m))
    # legacy dump of the same matches (no event death times): the fallback keeps the ghosts, by design
    legacy = make_dump(10, seed=74, event_death=False)
    lv2 = D.PTRDataset(legacy, D.Standardizer.fit(legacy), labels="v2")
    kept = sum(int((lv2[i]["node_mask"] & ~event_alive_ref_legacy(m)).sum()) for i, m in enumerate(legacy["matches"]))
    assert kept > 0


def event_alive_ref_legacy(m):
    """Truth alive tokens of a legacy synthetic match: slot p alive iff the team has not died before the phase."""
    T = m["team_death_s"].double().numpy()
    end = m["phase_end_s"].double().numpy()
    start = np.r_[0.0, end[:-1]]
    T = np.where(np.isnan(T), np.inf, T)  # legacy format: NaN = the winner
    return torch.from_numpy(T[None, :] > start[:, None])


@pytest.mark.parametrize("marker", ["field", "flag"])
def test_v2_targets_must_be_inside_the_event_alive_mask(marker):
    dump = make_dump(4, seed=75, event_marker=marker)
    D.validate_dump(dump)
    m = dump["matches"][0]
    i = int(m["target_mask_v2"].any(0).nonzero()[0])  # a team that is a v2 target somewhere
    m["team_death_s"][i] = 0.0  # ... but is "dead" before every phase in the death times
    with pytest.raises(ValueError, match=r"target_mask_v2.*(subset|alive)"):
        D.validate_dump(dump)
    D.validate_dump(dump, labels="pipeline")  # the pipeline labels do not depend on the death times
    with pytest.raises(ValueError, match="alive"):
        D.PTRDataset(dump, D.Standardizer.fit(make_dump(2, seed=75)))


def test_event_based_dump_requires_death_times_and_phase_ends_and_sane_values():
    for field in ("team_death_s", "phase_end_s"):
        dump = make_dump(3, seed=76)
        dump["matches"][1].pop(field)
        with pytest.raises(ValueError, match=field):
            D.validate_dump(dump)
        D.validate_dump(dump, labels="pipeline")  # only the v2 alive mask needs them
    dump = make_dump(3, seed=76)
    dump["matches"][0]["team_death_s"][0] = float("-inf")
    with pytest.raises(ValueError, match="team_death_s"):
        D.validate_dump(dump)
    dump = make_dump(3, seed=76)
    dump["matches"][0]["team_death_s"] = dump["matches"][0]["team_death_s"][:-1]
    with pytest.raises(ValueError, match="team_death_s"):
        D.validate_dump(dump)
    # a legacy dump does not need them for the pipeline labels, and not even for v2 (fallback)
    legacy = make_dump(3, seed=76, event_death=False)
    for m in legacy["matches"]:
        m.pop("team_death_s")
        m.pop("phase_end_s")
    D.validate_dump(legacy)


def test_unknown_death_time_means_not_alive_and_not_a_target():
    dump = make_dump(4, seed=77)
    m = dump["matches"][0]
    i = int((~m["target_mask_v2"].any(0)).nonzero()[0]) if (~m["target_mask_v2"].any(0)).any() else None
    if i is None:  # pick the team with the fewest targets and clear them
        i = int(m["target_mask_v2"].sum(0).argmin())
        m["target_mask_v2"][:, i] = False
        m["y_hazard_v2"][:, i] = 0.0
    m["team_death_s"][i] = float("nan")  # unknown death time (no member information)
    assert not D.event_alive_mask(m)[:, i].any()
    D.validate_dump(dump)
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump))
    assert not ds[0]["node_mask"][:, i].any()


def test_event_alive_audit_against_the_dump_node_mask_v2():
    """With phase_start_s in the dump the loader's mask IS the dump's node_mask_v2; the phase_end_s[k-1] fallback
    (dumps without phase_start_s) is the looser rule: it can only add tokens (deaths inside the grid gap)."""
    dump = make_dump(12, seed=78)
    tot = {"agree": 0, "only_loader": 0, "only_dump": 0}
    for m in dump["matches"]:
        a = D.event_alive_audit(m)
        assert a is not None
        for k in tot:
            tot[k] += a[k]
        assert a["n_loader"] == int(D.event_alive_mask(m).sum()) and a["n_dump"] == int(m["node_mask_v2"].sum())
        assert D.event_alive_rule(m) == "phase_start_s"
    assert tot["only_loader"] == 0 and tot["only_dump"] == 0 and tot["agree"] > 0
    summary = D.event_alive_summary(dump)
    assert summary["start_rule"] == "phase_start_s" and summary["only_loader"] == 0 and summary["only_dump"] == 0
    assert summary["ghost_removed"] > 0 and summary["n_alive"] == sum(int(D.event_alive_mask(m).sum()) for m in dump["matches"])
    # fallback: no phase_start_s
    loose = make_dump(12, seed=78)
    for m in loose["matches"]:
        del m["phase_start_s"]
    s2 = D.event_alive_summary(loose)
    assert s2["start_rule"] == "phase_end_s[k-1]" and s2["only_loader"] > 0 and s2["only_dump"] == 0
    flag_only = make_dump(2, seed=78, event_marker="flag")["matches"][0]
    assert D.event_alive_audit(flag_only) is None  # no node_mask_v2 field to compare with
    legacy_summary = D.event_alive_summary(make_dump(5, seed=78, event_death=False), labels="v2")
    assert legacy_summary["ghost_removed"] == 0 and legacy_summary["raised"] > 0 and "start_rule" not in legacy_summary


def test_standardizer_fits_on_the_event_alive_tokens():
    train = make_dump(6, seed=79)
    rows = np.concatenate([m["X"].numpy()[event_alive_ref(m).numpy()] for m in train["matches"]]).astype(np.float64)
    sz = D.Standardizer.fit(train)
    np.testing.assert_allclose(sz.x_mean, rows.mean(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sz.x_std, rows.std(0), rtol=1e-6, atol=1e-7)
    for m in train["matches"]:  # ghost tokens carry garbage in the synthetic data: they must not move the fit
        m["X"][m["node_mask"] & ~event_alive_ref(m)] = 1e6
    sz2 = D.Standardizer.fit(train)
    np.testing.assert_allclose(sz2.x_mean, sz.x_mean)
    np.testing.assert_allclose(sz2.x_std, sz.x_std)
    assert not np.allclose(D.Standardizer.fit(train, labels="pipeline").x_mean, sz.x_mean)  # pipeline sees ghosts


def test_winner_censoring_under_v2_only():
    """team_won => death = +inf: y = 0 at every cut, at risk wherever phase k+1 exists, alive in every slot."""
    dump = make_dump(6, seed=91)
    m = dump["matches"][0]
    w = int(m["team_won"].nonzero()[0])
    P = m["phase_end_s"].shape[0]
    cut_ok = torch.zeros(P, dtype=torch.bool)
    cut_ok[: P - 1] = m["phase_mask"][: P - 1] & m["phase_mask"][1:]
    # corrupt the winner as the real dumps do: a finite last-kill time inside the match and a v2 positive
    m["team_death_s"][w] = float(m["phase_end_s"][2]) - 1.0
    m["target_mask_v2"][:, w] = False
    m["y_hazard_v2"][:, w] = 0.0
    m["target_mask_v2"][:3, w] = True
    m["y_hazard_v2"][2, w] = 1.0
    y, t = D.select_labels(m, "v2")
    assert torch.equal(t[:, w], cut_ok) and not y[:, w].any()
    assert D.event_alive_mask(m)[:, w][m["phase_mask"]].all()  # immortal in every present slot
    _, t_pipe = D.select_labels(m, "pipeline")  # pipeline labels untouched
    assert torch.equal(t_pipe, m["target_mask"].bool())
    others = torch.arange(len(m["team_won"])) != w
    raw_t = m["target_mask_v2"].bool()
    assert torch.equal(t[:, others], raw_t[:, others])  # other teams unchanged
    D.validate_dump(dump)
    s = D.winner_censor_summary(dump, "v2")
    assert s["applied"] and s["winners_finite_death"] >= 1 and s["positives_removed"] >= 1 and s["target_rows_added"] >= 1
    assert D.winner_censor_summary(dump, "pipeline") == {**s, "applied": False, "winner_teams": 0, "winners_finite_death": 0,
        "positives_removed": 0, "target_rows_added": 0, "target_rows_removed": 0}  # fmt: skip
    ds = D.PTRDataset(dump, D.Standardizer.fit(dump))
    item = ds[0]
    assert not item["y_hazard"][:, w].any() and bool(item["target_mask"][:, w].equal(cut_ok))
