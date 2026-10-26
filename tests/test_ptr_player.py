"""Stage 2 (player layer): the player dump contract and its join, relation tensors, the player attention / pooling of the
model (dead-player invariance, -tm removes only the teammate bias, pooling masks) and the training CLI on synthetic
dumps.  DB-free, CPU."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.evaluation.metrics as M
import pubg_lab.survival.ptr.models.player as PL
import pubg_lab.survival.ptr.models.relations as R
import pubg_lab.survival.ptr.training.train as T
from pubg_lab.survival.ptr.models.arms import get_arm, model_config
from pubg_lab.survival.ptr.models.model import PTRConfig, PTRFormer, TeamPool, _Init
from pubg_lab.survival.ptr.data.synthetic import make_dump, make_player_dump, write_cohort_dumps, write_player_dumps

FULL = PL.PLAYER_RELATION_TYPES
TM = tuple(t for t in FULL if t != "teammate")


# --------------------------------------------------------------------------- helpers
def joined(players):
    return {str(pm["match_id"]): pm for pm in players["matches"]}


def build_dataset(dump, players, types=FULL, mode="log", **kw):
    pj = joined(players)
    sz, psz = D.Standardizer.fit(dump), D.PlayerStandardizer.fit(dump, pj)
    return D.PTRDataset(
        dump, sz, relation_types=R.RELATION_TYPES, combat_weights="log", players=pj, player_standardizer=psz,
        player_relation_types=types, player_combat_weights=mode, **kw,
    )  # fmt: skip


def make_model(ds, types=FULL, layers=1, seed=3, d=32, zone=True):
    cfg = PTRConfig(
        in_features=ds.n_features, n_phases=ds.n_phases, zone_dim=ds.zone_dim, d_model=d, n_heads=4, n_layers=2, dropout=0.0,
        relation=True, zone=zone, player=True, player_features=ds.n_player_features, player_relation_types=types,
        player_layers=layers,
    )  # fmt: skip
    return PTRFormer(cfg, init_seed=seed).eval()


@pytest.fixture(scope="module")
def world():
    dump = make_dump(6, seed=21, team_range=(5, 8))
    players = make_player_dump(dump, seed=22, signal=0.0)
    return dump, players


def out(model, batch):
    with torch.no_grad():
        return model(batch)


# --------------------------------------------------------------------------- the dump contract and the join
def test_synthetic_player_dump_follows_the_contract(world):
    dump, players = world
    assert players["meta"]["player_feature_names"] == [f"player_{k}" for k in range(6)]
    for m, pm in zip(dump["matches"], players["matches"]):
        assert pm["match_id"] == m["match_id"]
        PL.validate_player_match(pm, len(m["team_ids"]), 8, 6)
        alive = D.effective_alive(m, "v2", True)
        pa = PL.alive_players(alive, pm)
        assert torch.equal(pa, pm["player_alive_cut"] & alive[:, pm["player_team_idx"]])  # cut flag AND the team is alive
        for p in range(8):
            for i in range(len(m["team_ids"])):
                has = bool(pa[p, pm["player_team_idx"] == i].any())
                assert not has or bool(alive[p, i])  # no dead team has a live player (a live team may have none at the cut)
    # the two alive flags differ in both directions, as in the real dump: that is what the causal flag is for
    node = torch.cat([pm["player_node_mask"].flatten() for pm in players["matches"]])
    cut = torch.cat([pm["player_alive_cut"].flatten() for pm in players["matches"]])
    assert bool((node & ~cut).any()) and bool((~node & cut).any())


def test_validation_catches_malformed_player_matches(world):
    dump, players = world
    m, pm = dump["matches"][0], players["matches"][0]
    n, P = len(m["team_ids"]), pm["Xp"].shape[1]
    ok = lambda x: PL.validate_player_match(x, n, 8, 6)  # noqa: E731
    assert ok(pm) == []
    assert ok({**pm, "player_ids": [f"account.{k}" for k in range(P)]}) == []  # the real dump stores account-id strings
    with pytest.raises(ValueError, match="missing fields"):
        ok({k: v for k, v in pm.items() if k != "player_is_ai"})
    with pytest.raises(ValueError, match="Xp must be"):
        ok({**pm, "Xp": pm["Xp"][:, :, :4]})
    with pytest.raises(ValueError, match="Xp must be"):
        ok({**pm, "Xp": pm["Xp"][:7]})
    with pytest.raises(ValueError, match="player_team_idx must index"):
        ok({**pm, "player_team_idx": pm["player_team_idx"] + n})
    with pytest.raises(ValueError, match="player_alive_cut must be"):
        ok({**pm, "player_alive_cut": pm["player_alive_cut"][:, :-1]})
    # the causal flag is mandatory: a dump with only the event-based node mask is refused, never silently used
    with pytest.raises(ValueError, match=r"no 'player_alive_cut'.*never used"):
        ok({k: v for k, v in pm.items() if k != "player_alive_cut"})
    with pytest.raises(ValueError, match="player_ids must have"):
        ok({**pm, "player_ids": pm["player_ids"][:-1]})
    bad_edges = {**pm["edges_player"], "combat": (torch.tensor([[0, 0, P]]), torch.tensor([1.0]))}
    with pytest.raises(ValueError, match="index outside"):
        ok({**pm, "edges_player": bad_edges})
    with pytest.raises(ValueError, match=r"idx \[E, 3\]"):
        ok({**pm, "edges_player": {"teammate": (torch.zeros(3, 2, dtype=torch.long), torch.ones(3))}})
    extra = {**pm, "edges_player": {**pm["edges_player"], "zone_object": (torch.zeros(0, 3, dtype=torch.long), torch.zeros(0))}}
    assert ok(extra) == ["zone_object"]  # unknown edge types are ignored, and reported
    # -1 = a player without a team is allowed (never alive)
    assert ok({**pm, "player_team_idx": torch.full_like(pm["player_team_idx"], -1)}) == []


def write_split_dumps(tmp_path, n_train=5):
    phase_root, player_root = tmp_path / "phase", tmp_path / "players"
    write_cohort_dumps(phase_root, "c", sizes={"train": n_train, "val": 3, "test": 3}, seed=4)
    write_player_dumps(phase_root, "c", player_root, seed=5, signal=0.0)
    return phase_root, player_root


def load_phase(phase_root):
    return {s: D.load_dump(phase_root / "c" / f"{s}.pt") for s in ("train", "val", "test")}


def test_join_by_match_id_and_incomplete_dumps(tmp_path):
    phase_root, player_root = write_split_dumps(tmp_path)
    phase = load_phase(phase_root)
    players, info = PL.load_player_cohort(player_root, "c", phase)
    assert {s: len(v) for s, v in players.items()} == {"train": 5, "val": 3, "test": 3}
    assert info["feature_names"] == [f"player_{k}" for k in range(6)] and info["ignored_edge_types"] == []
    got = info["per_split"]["train"]
    assert (got["joined"], got["player_matches_in_dump"], got["ignored"]) == (5, 5, 0) and set(got["edge_endpoints_not_live"]) == {"teammate", "enemy_prox", "combat"}
    # a subsample (or the population filter) of the phase dump: the extra player matches are ignored
    sub, _ = D.subsample_dump(phase["train"], 2, 1)
    players2, info2 = PL.load_player_cohort(player_root, "c", {**phase, "train": sub})
    assert len(players2["train"]) == 2 and info2["per_split"]["train"]["ignored"] == 3
    assert set(players2["train"]) == {str(m["match_id"]) for m in sub["matches"]}
    # the order of the player matches does not matter; a missing match fails with the count and the ids
    dump = torch.load(player_root / "c" / "train.pt", weights_only=False)
    dump["matches"] = dump["matches"][::-1][:-1]
    torch.save(dump, player_root / "c" / "train.pt")
    with pytest.raises(ValueError, match=r"1 of 5 matches .* no player match"):
        PL.load_player_cohort(player_root, "c", phase)
    with pytest.raises(FileNotFoundError, match="player dump for cohort nope"):
        PL.load_player_cohort(player_root, "nope", phase)
    # feature names must agree across splits; meta must carry them
    phase_root, player_root = write_split_dumps(tmp_path / "again")
    val = torch.load(player_root / "c" / "val.pt", weights_only=False)
    val["meta"]["player_feature_names"] = ["a"] * 6
    torch.save(val, player_root / "c" / "val.pt")
    with pytest.raises(ValueError, match="differ from the other splits"):
        PL.load_player_cohort(player_root, "c", load_phase(phase_root))
    val["meta"].pop("player_feature_names")
    torch.save(val, player_root / "c" / "val.pt")
    with pytest.raises(ValueError, match="player_feature_names"):
        PL.load_player_cohort(player_root, "c", load_phase(phase_root))


def test_default_player_dump_dir_sits_next_to_the_phase_dump():
    assert PL.default_player_dump_dir("model_stgcn/dumps/bigcomp2027_phase_v1").as_posix() == "model_stgcn/dumps/bigcomp2027_player_v1"


# --------------------------------------------------------------------------- alive players, standardiser
def test_a_player_of_a_dead_team_is_never_alive(world):
    dump, players = world
    m, pm = dump["matches"][0], players["matches"][0]
    team_alive = D.effective_alive(m, "v2", True).clone()
    pm2 = {**pm, "player_alive_cut": torch.ones_like(pm["player_alive_cut"])}  # the dump claims everybody is alive at the cut
    pa = PL.alive_players(team_alive, pm2)
    assert torch.equal(pa, team_alive[:, pm["player_team_idx"]])
    team_alive[:, 0] = False  # team 0 dead everywhere
    assert not PL.alive_players(team_alive, pm2)[:, pm["player_team_idx"] == 0].any()
    nameless = {**pm2, "player_team_idx": torch.full_like(pm["player_team_idx"], -1)}  # -1: no team
    assert not PL.alive_players(D.effective_alive(m, "v2", True), nameless).any()


def test_player_standardiser_uses_alive_players_only_and_nan_becomes_zero(world):
    dump, players = world
    pj = joined(players)
    psz = D.PlayerStandardizer.fit(dump, pj)
    rows = []
    for m in dump["matches"]:
        pm = pj[str(m["match_id"])]
        rows.append(pm["Xp"].numpy()[PL.alive_players(D.effective_alive(m, "v2", True), pm).numpy()])
    rows = np.concatenate(rows).astype(np.float64)
    np.testing.assert_allclose(psz.xp_mean, np.nanmean(rows, 0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(psz.xp_std, np.nanstd(rows, 0), rtol=1e-6, atol=1e-7)  # garbage at dead players did not enter
    z = psz.transform(torch.tensor([[[float("nan")] * 6, [1e9] * 6]]))
    assert torch.isfinite(z).all() and (z[0, 0] == 0).all()
    again = D.PlayerStandardizer.from_state_dict(psz.state_dict())
    np.testing.assert_allclose(again.xp_mean, psz.xp_mean)


# --------------------------------------------------------------------------- relation tensors
def dense_of(edges, n=5, phases=2):
    return PL.densify_player_edges_batch([edges], phases, n)


def test_player_relation_channels_symmetry_direction_and_escalation():
    e = lambda idx, w: (torch.tensor(idx, dtype=torch.long), torch.tensor(w, dtype=torch.float32))  # noqa: E731
    edges = {
        "teammate": e([[0, 0, 1], [0, 2, 3]], [0.9, 0.5]),  # stored once
        "enemy_prox": e([[0, 0, 2], [0, 1, 3], [0, 0, 2], [1, 3, 4], [1, 0, 4]], [0.4, 0.6, 0.7, 0.2, 0.35]),  # a duplicate collapses with max
        "combat": e([[0, 1, 3], [1, 4, 3]], [2.0, 1.0]),  # attacker 1 -> victim 3 in slot 0
    }
    dense = dense_of(edges)
    assert dense["teammate"][0, 0, 1, 0] == dense["teammate"][0, 0, 0, 1] == pytest.approx(0.9)  # symmetrised
    assert dense["enemy_prox"][0, 0, 0, 2] == pytest.approx(0.7)
    assert bool(dense["combat_exists"][0, 0, 1, 3]) and not bool(dense["combat_exists"][0, 0, 3, 1])  # directed
    rel = PL.build_player_relation_tensor(dense, FULL, "log")
    assert rel.shape == (1, 2, 4, 5, 5)
    tm, prox, out, inn = (rel[0, :, k] for k in range(4))
    assert tm[0, 0, 1] == tm[0, 1, 0] == pytest.approx(0.9) and tm[0, 2, 3] == pytest.approx(0.5)
    assert out[0, 1, 3] == pytest.approx(2.0) and out[0, 3, 1] == 0.0  # combat_out: attacker -> victim, 'log' = as dumped
    assert inn[0, 3, 1] == pytest.approx(2.0) and inn[0, 1, 3] == 0.0  # combat_in is its transpose
    assert prox[0, 1, 3] == prox[0, 3, 1] == 0.0  # escalation: the combat pair loses its enemy_prox (both directions)
    assert prox[0, 0, 2] == pytest.approx(0.7) and prox[1, 0, 4] == pytest.approx(0.35)  # other pairs / other slots keep it
    assert prox[1, 3, 4] == 0.0  # slot 1: the pair (3, 4) has a combat edge (4 -> 3) in that slot only
    raw = PL.build_player_relation_tensor(dense, FULL, "raw")
    assert raw[0, 0, 2, 1, 3] == pytest.approx(np.log1p(2.0))  # raw intensity -> log1p
    # escalation only happens when a combat channel is requested (no leak of the combat edge set into a model without it)
    no_combat = PL.build_player_relation_tensor(dense, ("teammate", "enemy_prox"), "log")
    assert no_combat[0, 0, 1, 1, 3] == pytest.approx(0.6)
    # -tm: exactly the channels 1..3 of the full tensor
    tm_only = PL.build_player_relation_tensor(dense, TM, "log")
    assert tm_only.shape[2] == 3 and torch.equal(tm_only, rel[:, :, 1:])
    with pytest.raises(ValueError):
        PL.build_player_relation_tensor(dense, ("teammate", "zone"), "log")
    with pytest.raises(ValueError):
        PL.build_player_relation_tensor(dense, FULL, "linear")
    # self loops, negative and NaN weights are dropped / zeroed
    messy = {"teammate": e([[0, 1, 1], [0, 0, 2], [0, 2, 3]], [5.0, -1.0, float("nan")])}
    d2 = dense_of(messy)
    assert d2["teammate"].sum() == 0.0


def test_player_combat_weight_mode_is_inferred_from_the_weights():
    mk = lambda w: {"edges_player": {"combat": (torch.zeros(len(w), 3, dtype=torch.long), torch.tensor(w))}}  # noqa: E731
    assert PL.infer_player_combat_weights([mk([1.0, 6.5])]) == ("log", pytest.approx(6.5))
    assert PL.infer_player_combat_weights([mk([1.0]), mk([250.0])])[0] == "raw"
    assert PL.infer_player_combat_weights([{"edges_player": {}}]) == ("log", 0.0)


# --------------------------------------------------------------------------- dataset and batches
def test_dataset_items_and_padded_batches_carry_the_players(world):
    dump, players = world
    ds = build_dataset(dump, players)
    assert ds.n_player_features == 6
    item = ds[0]
    pm = players["matches"][0]
    P = pm["Xp"].shape[1]
    assert item["xp"].shape == (8, P, 6) and item["p_alive"].shape == (8, P) and item["p_team"].shape == (P,)
    assert not item["xp"][~item["p_alive"]].any() and torch.isfinite(item["xp"]).all()  # dead players zeroed, NaN gone
    assert torch.equal(item["p_alive"], PL.alive_players(item["node_mask"], pm))
    batch = ds.get_batch([0, 1, 2])
    Pmax = max(players["matches"][i]["Xp"].shape[1] for i in range(3))
    assert batch["xp"].shape == (3, 8, Pmax, 6) and batch["rel_p"].shape == (3, 8, 4, Pmax, Pmax)
    for b in range(3):
        P_b = players["matches"][b]["Xp"].shape[1]
        assert not batch["p_alive"][b, :, P_b:].any() and not batch["xp"][b, :, P_b:].any()  # padding is dead
        assert not batch["rel_p"][b, :, :, P_b:].any() and not batch["rel_p"][b, :, :, :, P_b:].any()
        assert (batch["p_team"][b, :P_b] == players["matches"][b]["player_team_idx"]).all()
    # the team-level batch is untouched by the player entries
    plain = D.PTRDataset(dump, D.Standardizer.fit(dump), relation_types=R.RELATION_TYPES, combat_weights="log")
    ref = plain.get_batch([0, 1, 2])
    assert all(torch.equal(batch[k], ref[k]) for k in ("x", "node_mask", "target_mask", "y_hazard", "rel"))
    assert "xp" not in ref and "rel_p" not in ref
    # -tm batches have the three remaining channels
    assert build_dataset(dump, players, TM).get_batch([0])["rel_p"].shape[2] == 3
    with pytest.raises(ValueError, match="player_standardizer"):
        D.PTRDataset(dump, D.Standardizer.fit(dump), players=joined(players))


# --------------------------------------------------------------------------- the model: pooling
def test_team_pool_masks_dead_players_and_foreign_teams():
    torch.manual_seed(0)
    pool = TeamPool(8, _Init(1)).eval()
    B, P, M, N = 2, 3, 6, 3
    h = torch.randn(B, P, M, 8)
    team = torch.tensor([[0, 0, 1, 1, 2, 2]] * B)
    alive = torch.ones(B, P, M, dtype=torch.bool)
    alive[0, 0, 4:] = False  # team 2 has no alive player in (0, 0)
    alive[0, 1, 0] = False  # one of team 0's two players dead in (0, 1)
    with torch.no_grad():
        pooled, w = pool(h, alive, team, N)
    assert pooled.shape == (B, P, N, 8) and w.shape == (B, P, N, M) and torch.isfinite(pooled).all()
    assert (pooled[0, 0, 2] == 0).all() and (w[0, 0, 2] == 0).all()  # empty team: exactly zero
    assert w[0, 1, 0, 0] == 0 and w[0, 1, 0, 1] == pytest.approx(1.0)  # dead player: weight 0, the other one gets everything
    for b in range(B):
        for p in range(P):
            for t in range(N):
                members = (team[b] == t) & alive[b, p]
                assert w[b, p, t][~members].abs().sum() == 0  # only the team's own alive players
                if members.any():
                    assert w[b, p, t].sum() == pytest.approx(1.0)
    # another team's player (or a dead player) cannot change a team's pooled vector
    h2 = h.clone()
    h2[0, 0, 2:4] += 5.0  # team 1's players
    h2[0, 0, 4:] += 5.0  # dead players of team 2
    with torch.no_grad():
        pooled2, _ = pool(h2, alive, team, N)
    assert torch.equal(pooled2[0, 0, 0], pooled[0, 0, 0]) and torch.equal(pooled2[0, 0, 2], pooled[0, 0, 2])
    assert not torch.equal(pooled2[0, 0, 1], pooled[0, 0, 1])


# --------------------------------------------------------------------------- the model: invariances
def player_batch(world, types=FULL, idx=(0, 1)):
    dump, players = world
    ds = build_dataset(dump, players, types)
    return ds, ds.get_batch(list(idx))


def test_forward_is_finite_with_garbage_at_dead_players_and_dead_teams(world):
    ds, batch = player_batch(world)
    model = make_model(ds)
    ref = out(model, batch)
    assert torch.isfinite(ref["hazard_logit"]).all() and torch.isfinite(ref["rank_score"]).all()
    # garbage (NaN / huge) in every dead player slot, and in the edges of dead players: no change at all
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    dead = ~b2["p_alive"]
    b2["xp"][dead] = float("nan")
    b2["p_ai"] = b2["p_ai"].clone()
    dead_pair = ~(b2["p_alive"].unsqueeze(-1) & b2["p_alive"].unsqueeze(-2))  # [B,P,M,M]
    b2["rel_p"] = torch.where(dead_pair.unsqueeze(2), torch.full_like(b2["rel_p"], 1e6), b2["rel_p"])
    got = out(model, b2)
    assert torch.equal(got["hazard_logit"], ref["hazard_logit"]) and torch.equal(got["rank_score"], ref["rank_score"])
    # dead players must not carry gradient either
    model.train()
    batch["xp"].requires_grad_(True)
    loss = model(batch)["hazard_logit"].sum()
    loss.backward()
    assert batch["xp"].grad[~batch["p_alive"]].abs().sum() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_a_dead_team_cannot_reach_an_alive_one_through_the_players(world):
    ds, batch = player_batch(world)
    model = make_model(ds)
    ref = out(model, batch)
    # a REAL team (not padding) that is dead at some slot and has players: give those players the dump's alive flag and loud features
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    found = None
    for b in range(batch["node_mask"].shape[0]):
        n = int(batch["n_teams"][b])
        for p in range(batch["node_mask"].shape[1]):
            for t in range(n):
                members = b2["p_team"][b] == t
                if not bool(batch["node_mask"][b, p, t]) and bool(members.any()) and bool(batch["p_alive"][b, p].any()):
                    found = (b, p, t, members)
                    break
            if found:
                break
        if found:
            break
    assert found is not None
    b, p, t, members = found
    b2["p_alive"][b, p, members] = True  # the dump says alive, but the team is dead at this slot
    b2["xp"][b, p, members] = 50.0
    got = out(model, b2)
    assert torch.equal(got["hazard_logit"], ref["hazard_logit"])
    # (the model re-applies the team mask itself: without it these loud tokens would be attended)
    live_before = int((batch["p_alive"] & torch.gather(batch["node_mask"], 2, batch["p_team"][:, None, :].expand_as(batch["p_alive"]))).sum())
    assert live_before == int(batch["p_alive"].sum())  # the dataset already gave only live tokens


def test_the_player_attention_reaches_all_alive_players_including_enemies(world):
    ds, batch = player_batch(world)
    model = make_model(ds)
    ref = out(model, batch)["hazard_logit"]
    b = 0
    team = batch["p_team"][b]
    p = int(batch["p_alive"][b].any(-1).nonzero()[0])  # first slot with players
    alive = batch["p_alive"][b, p]
    teams_alive = sorted({int(t) for t in team[alive]})
    assert len(teams_alive) >= 2
    own, enemy = teams_alive[0], teams_alive[1]
    enemy_players = ((team == enemy) & alive).nonzero().flatten()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["xp"][b, p, enemy_players[0]] += 3.0
    got = out(model, b2)["hazard_logit"]
    # the own team's token at this slot changes although only an enemy player was touched (all-alive attention) ...
    assert not torch.allclose(got[b, p, own], ref[b, p, own], atol=1e-6)
    # ... and without player blocks (pooling only) it does not: players only reach their own team then
    flat = make_model(ds, layers=0)
    r0, g0 = out(flat, batch)["hazard_logit"], out(flat, b2)["hazard_logit"]
    # the team axis still mixes teams from the phase slot of the change on, so compare the pooled tokens directly
    xt = batch["x"]
    pooled = lambda bb: flat.pool_players(bb, torch.where(bb["node_mask"].unsqueeze(-1), xt, 0.0), bb["node_mask"].bool(), flat.phase_emb.weight[: xt.shape[1]])  # noqa: E731
    pa, pb = pooled(batch), pooled(b2)
    assert torch.equal(pa[b, p, own], pb[b, p, own]) and not torch.equal(pa[b, p, enemy], pb[b, p, enemy])
    assert r0.shape == g0.shape


def test_player_features_of_a_later_phase_cannot_change_earlier_outputs(world):
    ds, batch = player_batch(world)
    model = make_model(ds)
    ref = out(model, batch)["hazard_logit"]
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    q = 4
    b2["xp"][:, q:] += 2.0
    got = out(model, b2)["hazard_logit"]
    assert torch.equal(got[:, :q], ref[:, :q]) and not torch.equal(got[:, q:], ref[:, q:])


def test_relation_biases_act_on_the_player_attention_and_vanish_with_zero_weights(world):
    ds, batch = player_batch(world)
    model = make_model(ds)
    ref = out(model, batch)["hazard_logit"]
    for blk in model.player_blocks:  # biases are learned parameters: zero them -> the edges can no longer matter
        blk.rel_b.data.zero_()
    zero_ref = out(model, batch)["hazard_logit"]
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["rel_p"] = b2["rel_p"] * 7.0 + 0.3
    assert torch.equal(out(model, b2)["hazard_logit"], zero_ref)  # no bias -> edges have no effect
    assert not torch.allclose(zero_ref, ref, atol=1e-6)  # (and the nonzero biases did matter)
    for blk in model.player_blocks:
        blk.rel_b.data.normal_(0, 1.0)
    assert not torch.equal(out(model, b2)["hazard_logit"], out(model, batch)["hazard_logit"])  # with bias they do


def test_tm_removes_only_the_teammate_bias(world):
    dump, players = world
    ds_full, ds_tm = build_dataset(dump, players, FULL), build_dataset(dump, players, TM)
    full, tm = make_model(ds_full, FULL), make_model(ds_tm, TM)
    assert full.player_blocks[0].rel_b.shape[0] == 4 and tm.player_blocks[0].rel_b.shape[0] == 3
    # same weights everywhere except the bias table: channels 1..3 of the full model are the -tm model's channels
    sd = {k: v.clone() for k, v in full.state_dict().items()}
    for k in list(sd):
        if k.endswith("rel_b") and k.startswith("player_blocks"):
            sd[k][0] = 0.0  # teammate bias off
    tm_sd = {k: (v[1:].clone() if k.endswith("rel_b") and k.startswith("player_blocks") else v.clone()) for k, v in sd.items()}
    full.load_state_dict(sd)
    tm.load_state_dict(tm_sd)
    bf, bt = ds_full.get_batch([0, 1, 2]), ds_tm.get_batch([0, 1, 2])
    assert torch.equal(bf["rel_p"][:, :, 1:], bt["rel_p"])  # the remaining channels are literally the same tensors
    of, ot = out(full, bf), out(tm, bt)
    assert torch.equal(of["hazard_logit"], ot["hazard_logit"]) and torch.equal(of["rank_score"], ot["rank_score"])
    # a nonzero teammate bias is what makes the full model different ...
    for blk in full.player_blocks:
        blk.rel_b.data[0] = 1.5
    assert not torch.equal(out(full, bf)["hazard_logit"], ot["hazard_logit"])
    # ... and the -tm model is not given the teammate edges at all: whatever the dump holds for them cannot matter
    other = {**players, "matches": [{**pm, "edges_player": {**pm["edges_player"], "teammate": (torch.zeros(0, 3, dtype=torch.long), torch.zeros(0))}}
                                    for pm in players["matches"]]}  # the teammate edges deleted from the dump
    bt2 = build_dataset(dump, other, TM).get_batch([0, 1, 2])
    assert torch.equal(bt2["rel_p"], bt["rel_p"]) and torch.equal(out(tm, bt2)["hazard_logit"], ot["hazard_logit"])
    assert not torch.equal(build_dataset(dump, other, FULL).get_batch([0, 1, 2])["rel_p"], bf["rel_p"])  # the full arm sees the change
    # the arm registry says the same
    assert get_arm("M11RZ-P-tm").player_relation_types == TM and get_arm("M11RZ-P").player_relation_types == FULL


def test_m11rz_p_nests_m11rz_through_the_shared_init_streams(world):
    dump, players = world
    ds = build_dataset(dump, players)
    batch = ds.get_batch([0, 1])
    p_model = make_model(ds, seed=7)
    base_cfg = PTRConfig(in_features=ds.n_features, n_phases=8, zone_dim=ds.zone_dim, d_model=32, n_heads=4, n_layers=2, dropout=0.0,
                         relation=True, zone=True)  # fmt: skip
    base = PTRFormer(base_cfg, init_seed=7).eval()
    shared = {k: v for k, v in base.state_dict().items()}
    for k, v in p_model.state_dict().items():
        if k in shared:
            assert torch.equal(v, shared[k]), k  # every M11RZ component has the same initial weights in M11RZ-P
    assert set(shared) < set(p_model.state_dict())
    assert any(k.startswith("player_") or k.startswith("team_pool") for k in set(p_model.state_dict()) - set(shared))
    # with the player contribution switched off the two models are the same function
    p_model.pool_players = lambda b, x, alive, phase: torch.zeros(*x.shape[:3], 32)
    assert torch.equal(out(p_model, batch)["hazard_logit"], out(base, batch)["hazard_logit"])


def test_missing_player_tensors_fail_with_a_clear_message(world):
    dump, players = world
    ds = build_dataset(dump, players)
    model = make_model(ds)
    plain = D.PTRDataset(dump, D.Standardizer.fit(dump), relation_types=R.RELATION_TYPES, combat_weights="log").get_batch([0])
    with pytest.raises(ValueError, match="player"):
        model(plain)
    bad = ds.get_batch([0])
    bad["rel_p"] = bad["rel_p"][:, :, :2]
    with pytest.raises(ValueError, match="rel_p"):
        model(bad)
    with pytest.raises(ValueError):
        PTRConfig(in_features=4, player=True, player_features=0, player_relation_types=FULL)
    with pytest.raises(ValueError):
        PTRConfig(in_features=4, player=True, player_features=3, player_relation_types=("teammate", "x"))
    assert model_config(get_arm("M11RZ-P"), 10, player_features=6).player_relation_types == FULL


# --------------------------------------------------------------------------- the training CLI
SMALL = ["--d-model", "32", "--n-heads", "4", "--n-layers", "2"]


@pytest.fixture(scope="module")
def stage2_dumps(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage2")
    write_cohort_dumps(root / "phase", "c", sizes={"train": 12, "val": 5, "test": 5}, seed=6)
    write_player_dumps(root / "phase", "c", root / "players", seed=7, signal=0.0)
    return root


def cli(root, out_dir, arm, *extra, epochs=1, seed=42):
    return ["--dump-dir", str(root / "phase"), "--cohort", "c", "--arm", arm, "--seed", str(seed), "--out-dir", str(out_dir),
            "--device", "cpu", "--max-epochs", str(epochs), "--player-dump-dir", str(root / "players"), *SMALL, *extra]  # fmt: skip


@pytest.mark.parametrize("arm", ["M11RZ-P", "M11RZ-P-tm"])
def test_stage_two_arms_train_end_to_end_and_share_rows_with_m11rz(tmp_path, stage2_dumps, arm):
    out_dir = tmp_path / "o"
    assert T.main(cli(stage2_dumps, out_dir, arm)) == 0
    assert T.main(cli(stage2_dumps, out_dir, "M11RZ")) == 0
    met = json.loads((out_dir / "c" / arm / "seed42" / "metrics.json").read_text())
    pl = met["data"]["player"]
    assert pl["feature_names"] == [f"player_{k}" for k in range(6)] and pl["layers"] == 1 and pl["combat_weights"] == "log"
    assert pl["relation_types"] == list(get_arm(arm).player_relation_types)
    assert pl["join"]["train"]["joined"] == 12 and pl["per_split"]["train"]["n_alive_player_tokens"] > 0
    assert set(pl["per_split"]["train"]["n_edges"]) == {"teammate", "enemy_prox", "combat"}
    assert met["config"]["player"] is True and met["config"]["player_features"] == 6 and met["arm_spec"]["player"] is True
    assert met["n_params"] > json.loads((out_dir / "c" / "M11RZ" / "seed42" / "metrics.json").read_text())["n_params"]
    a, b = (out_dir / "c" / x / "seed42" / "pred_test.csv" for x in (arm, "M11RZ"))
    M.align_predictions(M.read_predictions(a), M.read_predictions(b))  # identical rows: the paired bootstrap applies
    ckpt = torch.load(out_dir / "c" / arm / "seed42" / "best.pt", weights_only=False)
    assert ckpt["player_standardizer"] is not None and "xp_mean" in ckpt["player_standardizer"]
    assert T.main(cli(stage2_dumps, out_dir, arm)) == 0  # re-running into the same directory is fine


def test_player_arms_hash_the_player_dump_and_other_arms_do_not(tmp_path, stage2_dumps):
    h = lambda arm, *extra: T.config_hash(T.parse_args(cli(stage2_dumps, tmp_path / "o", arm, *extra)))  # noqa: E731
    base = h("M11RZ-P")
    assert base != h("M11RZ-P-tm") and base != h("M11RZ-P", "--player-layers", "2")
    spec = T.config_spec(T.parse_args(cli(stage2_dumps, tmp_path / "o", "M11RZ-P")))
    assert spec["player_layers"] == 1 and set(spec["player_dump"]) == {"train", "val", "test"}
    m11rz = h("M11RZ")
    assert "player_dump" not in T.config_spec(T.parse_args(cli(stage2_dumps, tmp_path / "o", "M11RZ")))
    assert m11rz == h("M11RZ", "--player-layers", "3")  # arms without players do not care about the player options
    assert m11rz == T.config_hash(T.parse_args(cli(stage2_dumps, tmp_path / "o", "M11RZ", "--player-dump-dir", str(tmp_path / "x"))))
    # re-dumping the players invalidates a player arm's hash (file mtime), not a team-level arm's
    os.utime(stage2_dumps / "players" / "c" / "train.pt", ns=(1, 1))
    assert h("M11RZ-P") != base and h("M11RZ") == m11rz


def test_the_default_player_dump_dir_is_next_to_the_phase_dump(tmp_path):
    root = tmp_path / "dumps"
    write_cohort_dumps(root / "bigcomp2027_phase_v1", "c", sizes={"train": 6, "val": 3, "test": 3}, seed=2)
    write_player_dumps(root / "bigcomp2027_phase_v1", "c", root / "bigcomp2027_player_v1", seed=3)
    argv = ["--dump-dir", str(root / "bigcomp2027_phase_v1"), "--cohort", "c", "--arm", "M11RZ-P-tm", "--out-dir", str(tmp_path / "o"),
            "--max-epochs", "1", *SMALL]  # fmt: skip
    assert T.main(argv) == 0
    assert json.loads((tmp_path / "o" / "c" / "M11RZ-P-tm" / "seed42" / "metrics.json").read_text())["data"]["player"]["dump_dir"].endswith("bigcomp2027_player_v1")


def test_subsample_and_player_arms_combine(tmp_path, stage2_dumps):
    assert T.main(cli(stage2_dumps, tmp_path / "o", "M11RZ-P", "--train-subsample", "5", "--subsample-seed", "3")) == 0
    met = json.loads((tmp_path / "o" / "c" / "M11RZ-P" / "sub5_ss3_seed42" / "metrics.json").read_text())
    assert met["data"]["player"]["join"]["train"]["joined"] == 5 and met["n_matches"]["train"] == 5


def test_an_incomplete_player_dump_fails_before_training(tmp_path, stage2_dumps):
    root = tmp_path / "broken"
    write_cohort_dumps(root / "phase", "c", sizes={"train": 6, "val": 3, "test": 3}, seed=6)
    write_player_dumps(root / "phase", "c", root / "players", seed=7)
    dump = torch.load(root / "players" / "c" / "test.pt", weights_only=False)
    dump["matches"] = dump["matches"][:2]
    torch.save(dump, root / "players" / "c" / "test.pt")
    with pytest.raises(ValueError, match="player dump is incomplete"):
        T.main(cli(root, tmp_path / "o", "M11RZ-P"))
    assert not (tmp_path / "o" / "c" / "M11RZ-P" / "seed42" / "metrics.json").exists()


def test_a_cue_that_only_the_players_carry_is_learned_by_the_player_arm(tmp_path):
    root = tmp_path / "signal"
    write_cohort_dumps(root / "phase", "c", sizes={"train": 40, "val": 10, "test": 14}, seed=8, signal=0.0, lag_signal=0.0)
    write_player_dumps(root / "phase", "c", root / "players", seed=9, signal=3.0)  # Xp[..., 0] ~ the team's hazard label
    aucs = {}
    for arm in ("M11RZ", "M11RZ-P"):
        argv = ["--dump-dir", str(root / "phase"), "--cohort", "c", "--arm", arm, "--seed", "1", "--out-dir", str(tmp_path / "o"),
                "--max-epochs", "10", "--player-dump-dir", str(root / "players"), "--d-model", "32", "--n-heads", "4", "--n-layers", "2",
                "--patience", "10"]  # fmt: skip
        T.main(argv)
        aucs[arm] = json.loads((tmp_path / "o" / "c" / arm / "seed1" / "metrics.json").read_text())["test"]["hazard"]["auc"]
    assert aucs["M11RZ-P"] > 0.85 > aucs["M11RZ"] + 0.1, aucs  # the team-level model cannot see the cue, the player layer can


def test_the_check_cli_reports_the_join_and_the_failures(tmp_path, stage2_dumps, capsys):
    argv = ["--player-dir", str(stage2_dumps / "players"), "--dump-dir", str(stage2_dumps / "phase")]
    assert PL.main(argv) == 0
    text = capsys.readouterr().out
    assert text.startswith("c: ok, 12 train matches, 6 features") and "without an alive player" in text and "teammate" in text
    rep = PL.check_cohort(stage2_dumps / "players", stage2_dumps / "phase", "c")
    tr = rep["splits"]["train"]
    assert tr["matches"] == 12 and tr["players_in_dead_teams"] >= 0 and tr["teams_without_players"] >= 0 and tr["alive_player_tokens"] > 0
    assert tr["node_mask_only_tokens"] > 0 and tr["edge_endpoints_not_live"]["teammate"][1] == 0 and tr["edge_endpoints_not_live"]["enemy_prox"][1] == 0
    assert tr["nan_cells"] == 0 and set(tr["weights"]) == {"teammate", "enemy_prox", "combat"}
    # a cohort the player dump does not have fails (exit code 1, the reason on stderr), the others are still checked
    assert PL.main([*argv, "--cohorts", "c,nope"]) == 1
    assert "nope: FAILED" in capsys.readouterr().err


# --------------------------------------------------------------------------- the causal alive flag, not the event-based node mask
def test_the_event_based_player_node_mask_is_never_used(world):
    dump, players = world
    base = build_dataset(dump, players)
    ref = base.get_batch([0, 1, 2])
    model = make_model(base)
    out_ref = out(model, ref)
    rng = np.random.default_rng(0)
    variants = {
        "all alive": lambda pm: torch.ones_like(pm["player_node_mask"]),
        "all dead": lambda pm: torch.zeros_like(pm["player_node_mask"]),
        "random": lambda pm: torch.from_numpy(rng.random(tuple(pm["player_node_mask"].shape)) < 0.5),
        "inverted": lambda pm: ~pm["player_node_mask"],
    }
    for name, make in variants.items():
        other = {**players, "matches": [{**pm, "player_node_mask": make(pm)} for pm in players["matches"]]}
        got = build_dataset(dump, other).get_batch([0, 1, 2])
        for key in ("xp", "p_alive", "p_team", "p_ai", "rel_p"):
            assert torch.equal(got[key], ref[key]), (name, key)
        assert torch.equal(out(model, got)["hazard_logit"], out_ref["hazard_logit"]), name
    gone = {**players, "matches": [{k: v for k, v in pm.items() if k != "player_node_mask"} for pm in players["matches"]]}
    assert torch.equal(build_dataset(dump, gone).get_batch([0, 1, 2])["p_alive"], ref["p_alive"])  # the field may be absent
    # tokens follow alive_cut: flipping a player's cut flag changes its token and the outputs
    m0 = {**players["matches"][0]}
    m0["player_alive_cut"] = ~m0["player_alive_cut"]
    flipped = {**players, "matches": [m0, *players["matches"][1:]]}
    assert not torch.equal(build_dataset(dump, flipped).get_batch([0])["p_alive"], ref["p_alive"][:1, :, : m0["Xp"].shape[1]])


def test_a_dump_without_the_cut_flag_is_refused_before_anything_is_trained(tmp_path):
    phase_root, player_root = write_split_dumps(tmp_path)
    dump = torch.load(player_root / "c" / "train.pt", weights_only=False)
    for pm in dump["matches"]:
        del pm["player_alive_cut"]  # only player_node_mask is left
    torch.save(dump, player_root / "c" / "train.pt")
    with pytest.raises(ValueError, match="player_alive_cut"):
        PL.load_player_cohort(player_root, "c", load_phase(phase_root))


def test_teammate_and_enemy_edges_must_end_on_live_tokens_combat_edges_are_only_reported(tmp_path):
    phase_root, player_root = write_split_dumps(tmp_path)
    phase = load_phase(phase_root)
    players, info = PL.load_player_cohort(player_root, "c", phase)
    rep = info["per_split"]["train"]["edge_endpoints_not_live"]
    assert rep["teammate"][0] > 0 and rep["teammate"][1] == 0 and rep["enemy_prox"][1] == 0
    assert rep["combat"][0] > 0 and rep["combat"][1] > 0  # victims killed inside the window: dead at the cut, by construction
    # an enemy_prox edge on a token that is dead at the cut is a broken dump: refused, naming the match and the type
    dump = torch.load(player_root / "c" / "train.pt", weights_only=False)
    pm = dump["matches"][0]
    dead = (~pm["player_alive_cut"]).nonzero()[0]
    other = (pm["player_alive_cut"][int(dead[0])]).nonzero()[0, 0]
    edge = torch.tensor([[int(dead[0]), int(dead[1]), int(other)]])
    pm["edges_player"]["enemy_prox"] = (torch.cat([pm["edges_player"]["enemy_prox"][0], edge]),
                                        torch.cat([pm["edges_player"]["enemy_prox"][1], torch.tensor([0.5])]))  # fmt: skip
    torch.save(dump, player_root / "c" / "train.pt")
    with pytest.raises(ValueError, match=r"not a live token.*enemy_prox") as err:
        PL.load_player_cohort(player_root, "c", phase)
    assert str(pm["match_id"]) in str(err.value)
    # the unit-level assertion, including the dead-team case (alive at the cut but the team is dead)
    m = phase["train"]["matches"][1]
    pm1 = players["train"][str(m["match_id"])]
    live = PL.alive_players(D.effective_alive(m, "v2", True), pm1)
    assert PL.assert_edges_on_live_tokens(pm1, live)["teammate"][1] == 0
    with pytest.raises(ValueError, match="teammate"):
        PL.assert_edges_on_live_tokens(pm1, torch.zeros_like(live))
    assert PL.assert_edges_on_live_tokens(pm1, torch.zeros_like(live), strict_types=())["combat"][1] == PL.edge_endpoint_report(pm1, live)["combat"][1] + int(
        (PL.edge_endpoint_report(pm1, torch.zeros_like(live))["combat"][0] - PL.edge_endpoint_report(pm1, live)["combat"][1]))  # fmt: skip


def test_the_endpoint_counts_reach_metrics_json(tmp_path, stage2_dumps):
    assert T.main(cli(stage2_dumps, tmp_path / "o", "M11RZ-P-tm")) == 0
    met = json.loads((tmp_path / "o" / "c" / "M11RZ-P-tm" / "seed42" / "metrics.json").read_text())
    rep = met["data"]["player"]["join"]["train"]["edge_endpoints_not_live"]
    assert rep["teammate"][1] == 0 and rep["enemy_prox"][1] == 0 and rep["combat"][1] > 0


# --------------------------------------------------------------------------- finer checks of the player layer's wiring
def test_padding_does_not_change_a_matchs_outputs_dead_keys_are_never_attended(world):
    ds, _ = player_batch(world)
    model = make_model(ds)
    n_players = [pm["Xp"].shape[1] for pm in world[1]["matches"]]
    assert len(set(n_players)) > 2  # the matches differ in size, so batching pads players (and teams)
    n0 = len(world[0]["matches"][0]["team_ids"])
    solo = out(model, ds.get_batch([0]))["hazard_logit"][0, :, :n0]
    for other in (1, 2, 3, 4, 5):
        pair = out(model, ds.get_batch([0, other]))["hazard_logit"][0, :, :n0]
        assert torch.allclose(solo, pair, atol=2e-5), other  # extra dead players / teams change nothing
    big = max(range(6), key=lambda i: n_players[i])
    rev = out(model, ds.get_batch([big, 0]))["hazard_logit"][1, :, :n0]
    assert torch.allclose(solo, rev, atol=2e-5)


def test_the_relation_bias_diagonal_is_zero_even_if_a_batch_carries_self_loops(world):
    ds, batch = player_batch(world)
    model = make_model(ds)
    for blk in model.player_blocks:
        blk.rel_b.data.normal_(0, 1.0)
    ref = out(model, batch)["hazard_logit"]
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    eye = torch.eye(b2["rel_p"].shape[-1], dtype=torch.bool)
    b2["rel_p"] = torch.where(eye, torch.full_like(b2["rel_p"], 5.0), b2["rel_p"])  # self loops with a large weight
    assert torch.equal(out(model, b2)["hazard_logit"], ref)


def test_the_player_tokens_are_player_features_team_features_and_the_ai_flag(world):
    """The input of ``player_mlp`` is [Xp | the team's token features | player_is_ai] at live players and 0 elsewhere; the
    team token gets the pooled players added exactly once (hooks on the real forward pass)."""
    ds, batch = player_batch(world)
    model = make_model(ds)
    seen = {}
    h1 = model.player_mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("mlp_in", a[0].detach().clone()))
    h2 = model.player_blocks[0].register_forward_pre_hook(lambda m, a: seen.__setitem__("block_in", a[0].detach().clone()))
    emb_calls = []
    h3 = model.emb_drop.register_forward_pre_hook(lambda m, a: emb_calls.append(a[0].detach().clone()))
    out(model, batch)
    for h in (h1, h2, h3):
        h.remove()
    x = torch.where(batch["node_mask"].unsqueeze(-1), batch["x"], torch.zeros_like(batch["x"]))
    B, P, M, _ = batch["xp"].shape
    team = batch["p_team"]
    live = batch["p_alive"] & torch.gather(batch["node_mask"], 2, team[:, None, :].expand(B, P, M))
    expect = torch.cat([batch["xp"], torch.stack([x[b][:, team[b]] for b in range(B)]), batch["p_ai"][:, None, :, None].expand(B, P, M, 1)], dim=-1)
    expect = torch.where(live.unsqueeze(-1), expect, torch.zeros_like(expect))
    assert torch.equal(seen["mlp_in"], expect)  # features, the player's TEAM's features, the AI flag; zero at dead players
    assert (batch["p_ai"] > 0).any() and (expect[..., -1][live] > 0).any() and expect[..., batch["xp"].shape[-1] : -1].abs().sum() > 0
    phase = model.phase_emb.weight[:P]
    with torch.no_grad():
        want_block_in = torch.where(live.unsqueeze(-1), model.player_mlp(expect) + phase[None, :, None, :], torch.zeros(()))
    assert torch.allclose(seen["block_in"], want_block_in, atol=1e-6)  # phase embedding added, dead players zeroed
    team_tokens = emb_calls[1]  # call 0: the player tokens, call 1: the team token that enters the blocks, call 2: the zone token
    with torch.no_grad():
        pooled = model.pool_players(batch, x, batch["node_mask"].bool(), phase)
        want = model.token_mlp(x) + phase[None, :, None, :] + pooled
    assert torch.allclose(team_tokens, want, atol=1e-6) and float(pooled.abs().sum()) > 0


def test_player_and_team_blocks_do_not_share_their_init_streams(world):
    ds, _ = player_batch(world)
    model = make_model(ds, seed=5)
    team_rel = model.blocks[0].rel_b.detach()
    player_rel = model.player_blocks[0].rel_b.detach()
    assert team_rel.shape == player_rel.shape and not torch.equal(team_rel, player_rel)
    assert not torch.equal(model.blocks[0].attn_t.q.weight, model.player_blocks[0].attn.q.weight)
    two = make_model(ds, layers=2, seed=5)
    assert not torch.equal(two.player_blocks[0].rel_b, two.player_blocks[1].rel_b)


def test_players_without_a_team_are_never_alive_and_cannot_break_the_gather(world):
    dump, players = world
    pm = players["matches"][0]
    team = pm["player_team_idx"].clone()
    team[-1] = -1
    team[0] = -1
    other = {**players, "matches": [{**pm, "player_team_idx": team}, *players["matches"][1:]]}
    ds = build_dataset(dump, other)
    item = ds[0]
    assert int(item["p_team"].min()) >= 0 and not item["p_alive"][:, -1].any() and not item["p_alive"][:, 0].any()
    model = make_model(ds)
    res = out(model, ds.get_batch([0, 1]))
    assert torch.isfinite(res["hazard_logit"]).all()


def test_stage_two_is_defined_for_the_v2_labels_only(tmp_path, stage2_dumps):
    """The player flag and edges are aligned to the event-based team mask of the v2 labels.  Under the pipeline labels the
    team alive mask differs (ghost tokens), the endpoint assertion of the join notices, and the CLI refuses up front."""
    phase = {s: D.load_dump(stage2_dumps / "phase" / "c" / f"{s}.pt", "pipeline") for s in ("train", "val", "test")}
    PL.load_player_cohort(stage2_dumps / "players", "c", phase, labels="v2")  # fine
    with pytest.raises(ValueError, match="not a live token"):
        PL.load_player_cohort(stage2_dumps / "players", "c", phase, labels="pipeline")  # the label choice is honoured
    for arm in ("M11RZ-P", "M11RZ-P-tm"):
        with pytest.raises(ValueError, match="defined for --labels v2"):
            T.main(cli(stage2_dumps, tmp_path / "o", arm, "--labels", "pipeline"))
    assert not (tmp_path / "o" / "c").exists()
