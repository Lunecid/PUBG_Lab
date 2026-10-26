"""PTR-Former invariances (DB-free, CPU, synthetic batches).

The properties tested here are the contract of the model:

* dead-team invariance        - features of teams that are not alive never reach alive outputs
* phase causality             - phase p+1 cannot change outputs at phases <= p; no information
                                crosses phases at all when the phase axis is off
* team-axis isolation         - with the team axis off another team cannot change my output
* relation bias gating        - edges matter iff the relation bias is on
* zone object gating          - zone features matter iff the zone object is on
* NaN safety                  - fully masked rows / NaN at dead tokens never produce NaN
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.models.relations as R
from pubg_lab.survival.ptr.models.arms import ARMS, model_config
from pubg_lab.survival.ptr.models.model import MultiHeadAttention, PTRConfig, PTRFormer, masked_attention
from pubg_lab.survival.ptr.data.synthetic import make_dump

KEYS = ("hazard_logit", "rank_score")
NEURAL_ARMS = ["M00", "M10", "M01", "M11", "M11R", "M11RZ"]


def build(arm_name, *, seed=0, n_matches=3, d_model=32, n_layers=2, **dump_kw):
    """(model in eval mode, collated batch that always carries the relation tensor)."""
    dump = make_dump(n_matches, n_features=24, seed=seed, team_range=(6, 10), **dump_kw)
    sz = D.Standardizer.fit(dump)
    arm = ARMS[arm_name]
    ds = D.PTRDataset(
        dump,
        sz,
        drop_groups=arm.drop_groups,
        relation_types=R.RELATION_TYPES,
        shuffle_seed=42 if arm.shuffle else None,
    )
    cfg = model_config(
        arm,
        ds.n_features,
        n_phases=ds.n_phases,
        zone_dim=ds.zone_dim,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        dropout=0.0,
    )
    torch.manual_seed(seed)
    model = PTRFormer(cfg, init_seed=seed).eval()
    return model, ds.get_batch(list(range(n_matches)))


def clone(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def run(model, batch):
    with torch.no_grad():
        return model(batch)


def assert_same(a, b, mask=None, atol=1e-5):
    for k in KEYS:
        x, y = (a[k], b[k]) if mask is None else (a[k][mask], b[k][mask])
        torch.testing.assert_close(x, y, atol=atol, rtol=1e-5)


def max_diff(a, b, mask):
    return max(float((a[k][mask] - b[k][mask]).abs().max()) for k in KEYS)


def dead_pairs(alive):
    return alive.logical_not().unsqueeze(-1) | alive.logical_not().unsqueeze(-2)  # [B,P,N,N]


# --------------------------------------------------------------------------- dead-team invariance
@pytest.mark.parametrize("arm", NEURAL_ARMS)
@pytest.mark.parametrize("fill", ["noise", "nan"])
def test_dead_team_invariance(arm, fill):
    model, batch = build(arm, seed=1)
    base = run(model, batch)
    alive = batch["node_mask"]
    dead = ~alive
    assert dead.any() and alive.any()

    pert = clone(batch)
    if fill == "noise":
        g = torch.Generator().manual_seed(5)
        pert["x"][dead] = torch.randn(int(dead.sum()), pert["x"].shape[-1], generator=g) * 100
        pert["pos"][dead] = 9.0
        rel_fill = 7.0
    else:
        pert["x"][dead] = float("nan")
        pert["pos"][dead] = float("nan")
        rel_fill = float("nan")
    pair = dead_pairs(alive).unsqueeze(2).expand_as(pert["rel"])
    pert["rel"][pair] = rel_fill  # every edge touching a dead team, in both roles

    out = run(model, pert)
    assert_same(out, base, mask=alive)
    for k in KEYS:
        assert torch.isfinite(out[k]).all(), k


def test_dead_team_invariance_with_non_monotone_alive_pattern():
    """Alive-ness is not assumed monotone: a team may be absent at p and present at p+1."""
    model, batch = build("M11RZ", seed=3, n_matches=2)
    g = torch.Generator().manual_seed(0)
    batch["node_mask"] = torch.rand(batch["node_mask"].shape, generator=g) < 0.7
    batch["target_mask"] &= batch["node_mask"]
    base = run(model, batch)
    alive = batch["node_mask"]
    pert = clone(batch)
    pert["x"][~alive] = 1e3
    pert["rel"][dead_pairs(alive).unsqueeze(2).expand_as(pert["rel"])] = 3.0
    assert_same(run(model, pert), base, mask=alive)


# --------------------------------------------------------------------------- phase causality
@pytest.mark.parametrize("arm", NEURAL_ARMS)
def test_phase_causality(arm):
    model, batch = build(arm, seed=2)
    base = run(model, batch)
    n_phases = batch["x"].shape[1]
    for p in range(n_phases - 1):
        pert = clone(batch)
        fut = slice(p + 1, None)
        pert["x"][:, fut] = torch.randn_like(pert["x"][:, fut]) * 10
        pert["pos"][:, fut] += 1.0
        pert["rel"][:, fut] = torch.rand_like(pert["rel"][:, fut]) * 3
        pert["zone"][:, fut] += 0.3
        pert["zone_std"][:, fut] += 2.0
        pert["node_mask"][:, fut] = ~pert["node_mask"][:, fut]  # even the alive pattern changes
        out = run(model, pert)
        for k in KEYS:
            torch.testing.assert_close(out[k][:, : p + 1], base[k][:, : p + 1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "arm,flows", [("M00", False), ("M01", False), ("M10", True), ("M11", True), ("M11RZ", True)]
)
def test_phase_axis_moves_information_forward_only_when_on(arm, flows):
    model, batch = build(arm, seed=4)
    base = run(model, batch)
    p = 2
    both = batch["node_mask"][:, p] & batch["node_mask"][:, p + 1]  # alive at p and p+1
    assert both.any()
    pert = clone(batch)
    pert["x"][:, p] += 1.0
    out = run(model, pert)
    delta = max(float((out[k][:, p + 1][both] - base[k][:, p + 1][both]).abs().max()) for k in KEYS)
    assert (delta > 1e-6) == flows
    # outputs at earlier phases never move
    for k in KEYS:
        torch.testing.assert_close(out[k][:, :p], base[k][:, :p], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- team axis
@pytest.mark.parametrize(
    "arm,mixes", [("M00", False), ("M10", False), ("M01", True), ("M11", True), ("M11R", True)]
)
def test_team_axis_off_means_other_teams_cannot_affect_a_team(arm, mixes):
    model, batch = build(arm, seed=6)
    base = run(model, batch)
    alive = batch["node_mask"]
    b, p = 0, 3
    j = int(alive[b, p].nonzero().flatten()[0])
    pert = clone(batch)
    pert["x"][b, p, j] += 5.0
    pert["pos"][b, p, j] += 0.5
    pert["rel"][b, p, :, j, :] += 1.0
    pert["rel"][b, p, :, :, j] += 1.0
    out = run(model, pert)

    others = alive.clone()
    others[b, :, j] = False  # every other team of match b ...
    others[b, :p] = False  # ... at the perturbed phase and later (earlier phases cannot move)
    shift = max_diff(out, base, others)
    assert (shift > 1e-6) == mixes
    # other matches in the batch never move, whatever the arm
    rest = alive.clone()
    rest[b] = False
    assert_same(out, base, mask=rest)
    # team j itself reacts to its own change
    self_mask = torch.zeros_like(alive)
    self_mask[b, p, j] = True
    assert max_diff(out, base, self_mask) > 1e-6


# --------------------------------------------------------------------------- relation bias
@pytest.mark.parametrize("arm", ["M00", "M10", "M01", "M11"])
def test_relation_bias_off_means_edges_have_no_effect(arm):
    model, batch = build(arm, seed=7)
    assert not any("rel_b" in n for n, _ in model.named_parameters())
    base = run(model, batch)
    for rel in (torch.zeros_like(batch["rel"]), torch.rand_like(batch["rel"]) * 5):
        pert = clone(batch)
        pert["rel"] = rel
        assert_same(run(model, pert), base, atol=0.0)
    pert = clone(batch)
    pert.pop("rel")  # the model must not even require edges
    assert_same(run(model, pert), base, atol=0.0)


def test_relation_bias_on_means_edges_have_an_effect():
    model, batch = build("M11R", seed=7)
    assert any("rel_b" in n for n, _ in model.named_parameters())
    base = run(model, batch)
    alive = batch["node_mask"]
    none = clone(batch)
    none["rel"] = torch.zeros_like(batch["rel"])
    assert max_diff(run(model, none), base, alive) > 1e-5
    for r, name in enumerate(R.RELATION_TYPES):  # every relation channel is wired in
        pert = clone(batch)
        pert["rel"][:, :, r] = 0.0
        assert max_diff(run(model, pert), base, alive) > 1e-7, name
    # direction matters: transposing the relation matrices changes the output
    tr = clone(batch)
    tr["rel"] = batch["rel"].transpose(-1, -2).contiguous()
    assert max_diff(run(model, tr), base, alive) > 1e-6


def test_relation_bias_is_added_per_block_and_head_with_query_key_orientation():
    cfg = PTRConfig(in_features=6, n_phases=3, d_model=8, n_heads=2, n_layers=1, relation=True)
    model = PTRFormer(cfg, init_seed=0)
    blk = model.blocks[0]
    with torch.no_grad():
        blk.rel_b.copy_(torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2) / 10)
    rel = torch.rand(2, 3, 4, 5, 5)
    bias = blk.relation_bias(rel)
    assert bias.shape == (2, 3, 2, 5, 5)  # [B,P,H,query,key]
    expect = torch.einsum("bprij,rh->bphij", rel, blk.rel_b)
    torch.testing.assert_close(bias, expect)
    # i is the query axis, j the key axis
    manual = sum(blk.rel_b[r, 1].detach() * rel[1, 2, r, 3, 4] for r in range(4))
    assert float(bias[1, 2, 1, 3, 4].detach()) == pytest.approx(float(manual), rel=1e-5)


# --------------------------------------------------------------------------- zone object
def test_zone_off_means_zone_features_have_no_effect():
    model, batch = build("M11R", seed=8)
    assert not any("zone" in n for n, _ in model.named_parameters())
    base = run(model, batch)
    pert = clone(batch)
    pert["zone"] = pert["zone"] + 3.0
    pert["zone_std"] = pert["zone_std"] * -5.0
    pert["zone_valid"] = ~pert["zone_valid"]
    pert["pos"] = pert["pos"] + 1.0  # raw positions only feed the zone geometry
    assert_same(run(model, pert), base, atol=0.0)


def test_zone_on_means_zone_features_have_an_effect():
    model, batch = build("M11RZ", seed=8)
    assert any("zone" in n for n, _ in model.named_parameters())
    base = run(model, batch)
    alive = batch["node_mask"]
    pert = clone(batch)
    pert["zone_std"] += 1.0
    assert max_diff(run(model, pert), base, alive) > 1e-6
    pert = clone(batch)
    pert["pos"] += 0.2  # team position relative to the zone feeds the team<->zone bias
    assert max_diff(run(model, pert), base, alive) > 1e-6
    pert = clone(batch)
    pert["zone"][:, :, :6] += 0.1  # raw zone geometry (current/next centre and radius)
    assert max_diff(run(model, pert), base, alive) > 1e-6


def test_invalid_zone_phase_masks_the_zone_token():
    model, batch = build("M11RZ", seed=9)
    batch["zone_valid"][:, 3] = False
    base = run(model, batch)
    pert = clone(batch)
    pert["zone"][:, 3] += 5.0
    pert["zone_std"][:, 3] += 5.0
    pert["pos"][:, 3] += 1.0
    pert["zone"][:, 3, 0] = float("nan")
    assert_same(run(model, pert), base, atol=1e-6)
    # a valid phase still reacts
    pert = clone(batch)
    pert["zone_std"][:, 4] += 5.0
    assert max_diff(run(model, pert), base, batch["node_mask"]) > 1e-6


def test_zone_requires_team_axis_and_relation_requires_team_axis():
    base = dict(in_features=6, n_phases=4, d_model=8, n_heads=2, n_layers=1)
    with pytest.raises(ValueError):
        PTRConfig(**base, team_axis=False, zone=True)
    with pytest.raises(ValueError):
        PTRConfig(**base, team_axis=False, relation=True)
    with pytest.raises(ValueError):
        PTRConfig(**{**base, "d_model": 10, "n_heads": 4})  # d_model must divide by heads
    with pytest.raises(ValueError):
        PTRConfig(**base, relation=True, relation_types=("prox", "nope"))
    PTRConfig(**base, team_axis=True, zone=True, relation=True)


# --------------------------------------------------------------------------- NaN safety
def test_fully_masked_rows_produce_no_nan_and_finite_gradients():
    model, batch = build("M11RZ", seed=10)
    batch["node_mask"][0, 2] = False  # every team dead in one phase
    batch["node_mask"][1] = False  # a whole match with nobody alive
    batch["target_mask"] &= batch["node_mask"]
    batch["zone_valid"][2, 5] = False
    out = run(model, batch)
    for k in KEYS:
        assert torch.isfinite(out[k]).all(), k

    from pubg_lab.survival.ptr.training.loss import ptr_loss

    model.train()
    out = model(batch)
    loss, _ = ptr_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    for name, prm in model.named_parameters():
        assert prm.grad is None or torch.isfinite(prm.grad).all(), name


def test_single_team_match_and_all_dead_batch_are_finite():
    model, batch = build("M11RZ", seed=11, n_matches=2)
    batch["node_mask"][:] = False
    batch["target_mask"][:] = False
    out = run(model, batch)
    assert all(torch.isfinite(out[k]).all() for k in KEYS)
    batch["node_mask"][:, :, 0] = True  # one lonely alive team per phase
    out = run(model, batch)
    assert all(torch.isfinite(out[k]).all() for k in KEYS)


# --------------------------------------------------------------------------- attention core
def _naive_attention(q, k, v, mask, bias):
    scale = q.shape[-1] ** -0.5
    out = torch.zeros_like(q)
    for idx in np.ndindex(*q.shape[:-2]):  # every (..., head)
        for a in range(q.shape[-2]):
            allowed = mask[idx][a] if mask.dim() == q.dim() else mask[idx[:-1]][a]
            if not allowed.any():
                continue
            logits = (q[idx][a] @ k[idx].T) * scale + (bias[idx][a] if bias is not None else 0)
            logits = logits.masked_fill(~allowed, float("-inf"))
            out[idx][a] = torch.softmax(logits, -1) @ v[idx]
    return out


def test_masked_attention_matches_naive_reference_incl_empty_rows():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 3, 4, 8, generator=g)  # [batch, head, Lq, dh]
    k = torch.randn(2, 3, 5, 8, generator=g)
    v = torch.randn(2, 3, 5, 8, generator=g)
    bias = torch.randn(2, 3, 4, 5, generator=g)
    mask = torch.rand(2, 1, 4, 5, generator=g) < 0.6
    mask[0, 0, 1] = False  # a row with no allowed key
    mask[1, 0, 3] = False
    out = masked_attention(q, k, v, mask, bias)
    ref = _naive_attention(q, k, v, mask.expand(2, 3, 4, 5), bias)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out).all()
    assert (out[0, :, 1] == 0).all() and (out[1, :, 3] == 0).all()  # empty rows give zeros
    # without bias
    out2 = masked_attention(q, k, v, mask, None)
    torch.testing.assert_close(out2, _naive_attention(q, k, v, mask.expand(2, 3, 4, 5), None), atol=1e-5, rtol=1e-5)


def test_multihead_attention_bias_is_added_to_logits():
    """With q = k = 0 the weights are softmax(bias): bias[i, j] is (query i, key j)."""
    d = 4
    mha = MultiHeadAttention(d_model=d, n_heads=1, dropout=0.0)
    with torch.no_grad():
        for lin in (mha.q, mha.k):
            lin.weight.zero_()
            lin.bias.zero_()
        for lin in (mha.v, mha.o):
            lin.weight.copy_(torch.eye(d))
            lin.bias.zero_()
    x = torch.randn(1, 3, d)
    bias = torch.tensor([[[[0.0, 5.0, -5.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]]])  # [1,H=1,3,3]
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    out = mha(x, x, mask, bias)
    expect = torch.softmax(bias[0, 0], -1) @ x[0]
    torch.testing.assert_close(out[0], expect, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- init parity / wiring
def test_shared_components_share_initialisation_across_arms():
    def make(arm_name):
        arm = ARMS[arm_name]
        cfg = model_config(arm, 20, n_phases=8, zone_dim=8, d_model=32, n_heads=4, n_layers=2)
        return PTRFormer(cfg, init_seed=123)

    m00, m11, m11rz = make("M00"), make("M11"), make("M11RZ")
    for other in (m11, m11rz):
        for name in ("token_mlp.0.weight", "token_mlp.2.bias", "blocks.0.ffn.0.weight",
                     "blocks.1.ffn.3.weight", "hazard_head.weight", "rank_head.weight",
                     "blocks.0.attn_t.q.weight"):  # fmt: skip
            torch.testing.assert_close(dict(m00.named_parameters())[name],
                                       dict(other.named_parameters())[name])
    # M00 has no phase attention parameters at all (identity), M11 does
    assert not any(".attn_p." in n for n, _ in m00.named_parameters())
    assert any(".attn_p." in n for n, _ in m11.named_parameters())
    # a different init seed gives different weights
    other = PTRFormer(model_config(ARMS["M00"], 20, n_phases=8, zone_dim=8, d_model=32, n_heads=4), init_seed=124)
    assert not torch.equal(dict(m00.named_parameters())["token_mlp.0.weight"],
                           dict(other.named_parameters())["token_mlp.0.weight"])


def test_eval_forward_is_deterministic_and_train_mode_uses_dropout():
    model, batch = build("M11RZ", seed=12)
    a, b = run(model, batch), run(model, batch)
    assert_same(a, b, atol=0.0)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.5
    torch.manual_seed(0)
    t1 = model(batch)["hazard_logit"]
    t2 = model(batch)["hazard_logit"]
    assert not torch.equal(t1, t2)


def test_forward_validates_relation_channels_and_phase_count():
    model, batch = build("M11R", seed=13)
    bad = clone(batch)
    bad["rel"] = bad["rel"][:, :, :2]
    with pytest.raises(ValueError):
        model(bad)
    bad = clone(batch)
    bad["rel"] = bad["rel"].new_zeros(0)
    with pytest.raises(ValueError):
        model(bad)
    bad = clone(batch)
    bad.pop("rel")
    with pytest.raises(ValueError):  # relation bias on but no edges supplied
        model(bad)

    small = PTRFormer(
        PTRConfig(in_features=batch["x"].shape[-1], n_phases=4, d_model=16, n_heads=2, n_layers=1)
    ).eval()
    with pytest.raises(ValueError):  # batch has 8 phases, model knows 4
        small(batch)


# =========================================================================== stronger structural tests
# Value-perturbation tests cannot see whether dead tokens are *attended to* (their values are
# sanitised to exact zeros, which still carry non-zero keys/values through the biases).  The tests
# below pin that down: an explicit loop-based reference of the whole forward pass, padding
# invariance, "zone token fully masked == zone off", and NaN-safe gradients.
def _ref_mha(mha, q_vec, k_mat, bias):
    """One query token against an explicit key set.  q_vec [d]; k_mat [m, d]; bias [H, m] | None."""
    H, dh = mha.n_heads, mha.d_head
    q = mha.q(q_vec).view(H, dh)
    K = mha.k(k_mat).view(-1, H, dh)
    V = mha.v(k_mat).view(-1, H, dh)
    logits = torch.einsum("hd,mhd->hm", q, K) / dh**0.5
    if bias is not None:
        logits = logits + bias
    w = torch.softmax(logits, dim=-1)
    return mha.o(torch.einsum("hm,mhd->hd", w, V).reshape(-1))


@torch.no_grad()
def reference_forward(model, batch):
    """Spec-level re-implementation with explicit per-token key sets (slow, obviously correct)."""
    cfg = model.cfg
    x, alive = batch["x"], batch["node_mask"]
    B, P, N, _ = x.shape
    H = cfg.n_heads
    h = torch.zeros(B, P, N, cfg.d_model)
    for b, p, i in alive.nonzero().tolist():
        h[b, p, i] = model.token_mlp(x[b, p, i]) + model.phase_emb.weight[p]
    z = {}
    if cfg.zone:
        for b in range(B):
            for p in range(P):
                if batch["zone_valid"][b, p]:
                    z[(b, p)] = model.zone_mlp(batch["zone_std"][b, p]) + model.phase_emb.weight[p]

    def geo(b, p, i):  # (pos - centre)/radius and dist/radius for the current and the next zone
        zone, pos = batch["zone"][b, p], batch["pos"][b, p, i]
        feats = []
        for cx, r in ((0, 2), (3, 5)):
            rel = (pos - zone[cx : cx + 2]) / max(float(zone[r]), 1e-6)
            feats += [rel.clamp(-50, 50), rel.norm().clamp(max=50).reshape(1)]
        return torch.cat(feats)

    for blk in model.blocks:
        if cfg.phase_axis:  # per team, over its own alive phases p' <= p
            new = h.clone()
            for b, p, i in alive.nonzero().tolist():
                ks = [q for q in range(p + 1) if alive[b, q, i]]
                keys = blk.ln_p(torch.stack([h[b, q, i] for q in ks]))
                new[b, p, i] = h[b, p, i] + _ref_mha(blk.attn_p, blk.ln_p(h[b, p, i]), keys, None)
            h = new
        new_h, new_z = h.clone(), dict(z)
        for b in range(B):
            for p in range(P):
                teams = [i for i in range(N) if alive[b, p, i]]
                has_zone = (b, p) in z
                toks = {i: blk.ln_t(h[b, p, i]) for i in teams}
                if has_zone:
                    toks["z"] = blk.ln_t(z[(b, p)])
                everyone = teams + (["z"] if has_zone else [])

                def bias_of(q, k):
                    bias = torch.zeros(H)
                    if blk.rel_b is not None and q != "z" and k != "z":
                        rel = batch["rel"][b, p, :, q, k]
                        bias = bias + (blk.rel_b * rel[:, None]).sum(0)
                    if blk.zone_bias is not None and (q == "z") != (k == "z"):
                        team = k if q == "z" else q
                        if batch["pos_valid"][b, p, team]:  # missing position -> no zone bias
                            bias = bias + blk.zone_bias(geo(b, p, team))
                    return bias

                for q in everyone:
                    keys = everyone if cfg.team_axis else [q]
                    kmat = torch.stack([toks[k] for k in keys])
                    bias = torch.stack([bias_of(q, k) for k in keys], dim=1)
                    out = _ref_mha(blk.attn_t, toks[q], kmat, bias)
                    if q == "z":
                        new_z[(b, p)] = z[(b, p)] + out
                    else:
                        new_h[b, p, q] = h[b, p, q] + out
        h, z = new_h, new_z
        for b, p, i in alive.nonzero().tolist():
            h[b, p, i] = h[b, p, i] + blk.ffn(blk.ln_f(h[b, p, i]))
        for key in z:
            z[key] = z[key] + blk.ffn(blk.ln_f(z[key]))
    out = model.final_ln(h)
    return {
        "hazard_logit": model.hazard_head(out).squeeze(-1),
        "rank_score": model.rank_head(out).squeeze(-1),
    }


@pytest.mark.parametrize("arm", ["M00", "M10", "M01", "M11", "M11R", "M11R-shuf", "M11RZ", "M11RZ-noHab"])
def test_forward_matches_loop_based_reference(arm):
    """Whole-model equivalence incl. non-monotone alive patterns and invalid zone phases."""
    model, batch = build(arm, seed=21, n_matches=2, d_model=16, n_layers=2)
    g = torch.Generator().manual_seed(1)
    real = torch.arange(batch["x"].shape[2])[None, :] < batch["n_teams"][:, None]  # not padding
    batch["node_mask"] = (torch.rand(batch["node_mask"].shape, generator=g) < 0.7) & real[:, None, :]
    batch["target_mask"] &= batch["node_mask"]
    batch["zone_valid"][:, [1, 4]] = False
    batch["zone_valid"][1, 6] = False
    batch["pos_valid"][0, 3, :3] = False  # teams without a recorded position at one phase
    batch["pos"][0, 3, :3] = 0.0
    with torch.no_grad():  # make every learnable bias matter
        for blk in model.blocks:
            if blk.rel_b is not None:
                blk.rel_b.normal_(0, 1.0)
    got, want = run(model, batch), reference_forward(model, batch)
    for k in KEYS:
        torch.testing.assert_close(got[k], want[k], atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("arm", NEURAL_ARMS)
def test_padding_and_batching_do_not_change_a_matchs_outputs(arm):
    """Dead (padded) tokens have exactly-zero values but non-zero keys: they must not be attended."""
    model, batch = build(arm, seed=22, n_matches=3)
    n_teams = batch["n_teams"].tolist()
    assert len(set(n_teams)) > 1
    full = run(model, batch)
    dump = make_dump(3, n_features=24, seed=22, team_range=(6, 10))
    sz = D.Standardizer.fit(dump)
    arm_spec = ARMS[arm]
    ds = D.PTRDataset(dump, sz, relation_types=R.RELATION_TYPES, drop_groups=arm_spec.drop_groups)
    for k, n in enumerate(n_teams):
        alone = run(model, ds.get_batch([k]))  # padded to the match's own size only
        for key in KEYS:
            torch.testing.assert_close(
                alone[key][0, :, :n], full[key][k, :, :n], atol=1e-5, rtol=1e-5
            )
    # and padding a batch further with extra dead teams changes nothing either
    wide = clone(batch)
    pad = 5
    for key in ("x", "pos"):
        wide[key] = torch.nn.functional.pad(wide[key], (0, 0, 0, pad))
    for key in ("node_mask", "target_mask", "pos_valid"):
        wide[key] = torch.nn.functional.pad(wide[key], (0, pad))
    wide["rel"] = torch.nn.functional.pad(wide["rel"], (0, pad, 0, pad))
    out = run(model, wide)
    for key in KEYS:
        torch.testing.assert_close(out[key][:, :, : batch["x"].shape[2]], full[key], atol=1e-5, rtol=1e-5)


def test_zone_token_fully_masked_equals_the_zone_off_model():
    """Same init seed => shared weights are identical; with zone_valid all False the extra zone
    parameters must be unreachable, so PTR-Former reduces exactly to M11R."""
    rz, batch = build("M11RZ", seed=23)
    r, _ = build("M11R", seed=23)
    shared = dict(r.named_parameters())
    for name, prm in rz.named_parameters():
        if name in shared:
            torch.testing.assert_close(prm, shared[name])  # identical shared initialisation
    batch["zone_valid"][:] = False
    torch.testing.assert_close(run(rz, batch)["hazard_logit"], run(r, batch)["hazard_logit"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(run(rz, batch)["rank_score"], run(r, batch)["rank_score"], atol=1e-6, rtol=1e-5)
    # and the unreachable parameters really are irrelevant
    before = run(rz, batch)["hazard_logit"]
    with torch.no_grad():
        for name, prm in rz.named_parameters():
            if "zone" in name:
                prm.add_(3.0)
    torch.testing.assert_close(run(rz, batch)["hazard_logit"], before, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ["M11R", "M11RZ"])
def test_gradients_stay_finite_with_nan_at_dead_inputs_and_invalid_zone(arm):
    from pubg_lab.survival.ptr.training.loss import ptr_loss

    model, batch = build(arm, seed=24)
    model.train()
    alive = batch["node_mask"]
    batch["x"][~alive] = float("nan")
    batch["pos"][~alive] = float("nan")
    batch["rel"][dead_pairs(alive).unsqueeze(2).expand_as(batch["rel"])] = float("nan")
    batch["zone"][:, 2] = float("nan")
    batch["zone_std"][:, 2] = float("nan")
    batch["zone_valid"][:, 2] = False
    out = model(batch)
    loss, _ = ptr_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    for name, prm in model.named_parameters():
        assert prm.grad is None or torch.isfinite(prm.grad).all(), name


# =========================================================================== v2.2 details
def test_zone_mlp_is_per_block_and_team_zone_bias_is_symmetric():
    model, _ = build("M11RZ", seed=25)
    b0, b1 = model.blocks[0], model.blocks[1]
    assert b0.zone_bias is not b1.zone_bias  # one zone MLP per block ...
    assert not torch.equal(b0.zone_bias[0].weight, b1.zone_bias[0].weight)  # ... with its own weights
    linears = [m for m in b0.zone_bias if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(6, 16), (16, model.cfg.n_heads)]  # 6 -> 16 -> H

    B, P, N, H = 2, 3, 5, model.cfg.n_heads
    g = torch.Generator().manual_seed(0)
    geo = torch.randn(B, P, N, 6, generator=g)
    geo_ok = torch.ones(B, P, N, dtype=torch.bool)
    geo_ok[0, 1, 2] = False  # a team without a usable position / zone
    rel = torch.randn(B, P, len(R.RELATION_TYPES), N, N, generator=g)
    bias = b0._team_bias(rel, geo, geo_ok, N)
    assert bias.shape == (B, P, H, N + 1, N + 1)  # N teams + the zone token, query x key
    torch.testing.assert_close(bias[..., N, :N], bias[..., :N, N])  # zone->team == team->zone (symmetric)
    assert (bias[..., N, N] == 0).all()  # zone attends to itself without bias
    torch.testing.assert_close(bias[..., :N, :N], b0.relation_bias(rel))  # team-team block is the relation bias
    assert (bias[0, 1, :, 2, N] == 0).all() and (bias[0, 1, :, N, 2] == 0).all()  # invalid geometry -> no bias
    assert bias[0, 0, :, 1, N].abs().sum() > 0  # valid geometry -> the MLP output


def test_relation_bias_diagonal_is_zero_inside_the_model():
    model, batch = build("M11R", seed=26)
    base = run(model, batch)
    pert = clone(batch)
    eye = torch.eye(batch["rel"].shape[-1], dtype=torch.bool)
    pert["rel"] = pert["rel"] + 5.0 * eye  # a non-zero diagonal must not matter
    assert_same(run(model, pert), base, atol=0.0)


def test_self_attention_is_allowed_and_a_lone_alive_team_reduces_to_the_self_only_model():
    """M01 (team axis on) and M00 (self only) share every weight (same init seed).  With exactly one alive
    team per phase the team-axis keys are {itself}: the outputs must coincide.  That fails if a team could
    not attend to itself, or if dead teams were attended to."""
    m00, batch = build("M00", seed=27)
    m01, _ = build("M01", seed=27)
    for name, prm in m00.named_parameters():
        torch.testing.assert_close(prm, dict(m01.named_parameters())[name])
    lone = torch.zeros_like(batch["node_mask"])
    lone[:, :, 0] = True
    batch["node_mask"] = lone
    batch["target_mask"] &= lone
    a, b = run(m00, batch), run(m01, batch)
    assert_same(a, b, atol=1e-6)
    # sanity: with several alive teams the two models do differ
    batch["node_mask"][:, :, :3] = True
    assert max_diff(run(m00, batch), run(m01, batch), batch["node_mask"]) > 1e-6


# =========================================================================== ghost fix, end to end
def _ghost_setup(seed=80, arm="M11RZ"):
    dump = make_dump(4, n_features=24, seed=seed, team_range=(6, 10), dead_garbage=False)
    sz = D.Standardizer.fit(dump)
    cfg = model_config(ARMS[arm], 24 + 3, n_phases=8, zone_dim=8, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    torch.manual_seed(seed)
    return dump, sz, PTRFormer(cfg, init_seed=seed).eval()


def _ghost_mask(m):
    T = m["team_death_s"].double()
    end = m["phase_end_s"].double()
    start = torch.cat([torch.zeros(1, dtype=torch.float64), end[:-1]])
    return m["node_mask"] & ~(T[None, :] > start[:, None])  # pipeline-alive, event-dead


def test_ghost_teams_cannot_influence_any_prediction_under_event_based_v2():
    """Changing everything about the ghost tokens (features, position) must leave every output unchanged when the
    ghost fix is active: they are neither tokens nor attention keys.  Control: with the pipeline alive mask the
    same change does move the outputs of the teams that share a phase with a ghost."""
    dump, sz, model = _ghost_setup()
    ghosts = [_ghost_mask(m) for m in dump["matches"]]
    assert sum(int(g.sum()) for g in ghosts) > 0
    changed = copy.deepcopy(dump)
    for m, g in zip(changed["matches"], ghosts):
        m["X"][g] += 7.0
        m["team_pos"][g] = 0.123
    idx = list(range(len(dump["matches"])))
    outs = {}
    for name, d in (("base", dump), ("changed", changed)):
        ds = D.PTRDataset(d, sz, relation_types=R.RELATION_TYPES, labels="v2")
        assert ds.alive_source == "node_mask_v2"
        outs[name] = run(model, ds.get_batch(idx)), ds.get_batch(idx)["node_mask"]
    assert torch.equal(outs["base"][1], outs["changed"][1])
    alive = outs["base"][1]
    assert_same(outs["changed"][0], outs["base"][0], mask=alive, atol=0.0)

    pipe = {}
    for name, d in (("base", dump), ("changed", changed)):
        ds = D.PTRDataset(d, sz, relation_types=R.RELATION_TYPES, labels="pipeline")
        pipe[name] = run(model, ds.get_batch(idx)), ds.get_batch(idx)["node_mask"]
    ghost_phase_mates = pipe["base"][1] & ~torch.stack(  # alive teams of phases that contain a ghost token
        [torch.nn.functional.pad(g, (0, pipe["base"][1].shape[2] - g.shape[1])) for g in ghosts]
    )
    assert max_diff(pipe["changed"][0], pipe["base"][0], ghost_phase_mates) > 1e-6


def test_ghost_teams_do_not_enter_the_zone_token_or_the_phase_axis_either():
    """Same property for the arms that differ in how a ghost could leak: zone token, phase axis, relation bias."""
    for arm in ("M10", "M01", "M11R", "M11RZ"):
        dump, sz, model = _ghost_setup(seed=81, arm=arm)
        ghosts = [_ghost_mask(m) for m in dump["matches"]]
        changed = copy.deepcopy(dump)
        for m, g in zip(changed["matches"], ghosts):
            m["X"][g] = torch.randn_like(m["X"][g]) * 20
            m["team_pos"][g] = torch.rand_like(m["team_pos"][g])
        idx = list(range(len(dump["matches"])))
        batches = [D.PTRDataset(d, sz, relation_types=R.RELATION_TYPES).get_batch(idx) for d in (dump, changed)]
        a, b = (run(model, x) for x in batches)
        assert_same(a, b, mask=batches[0]["node_mask"], atol=0.0)
