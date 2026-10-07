"""PTR-Former: phase-axis, team-axis and relation-biased attention over team tokens.

Token grid ``h[B, P, N, d]`` (phase slot p, team i).  ``L`` pre-LN blocks, each::

    h += PhaseAxisAttn(LN(h))   per team, over phases p' <= p, keys masked by node_mask  (off: identity)
    h += TeamAxisAttn(LN(h))    per phase, over alive teams (+ one zone token)           (off: self only)
    h += FFN(LN(h))

Team-axis logits carry an additive relation bias (Graphormer style)::

    logit[i, j] = q_i . k_j / sqrt(d_h)  +  sum_r b[r, h] * W_r[i, j]

with ``b`` learned per block and head and ``W_r`` the relation matrices of
:mod:`model_stgcn.ptr.relations` (prox, combat_out, combat_in, copress).  The optional zone token
(one per phase) joins the team-axis set; the team<->zone bias per head is a small MLP of the team's
position relative to the current and next zone circles: ``(pos - centre) / radius`` and
``dist / radius`` for both.  ``zone_valid == False`` masks the token.

Masking contract (all verified in ``tests/test_ptr_model.py`` and ``tests/test_v2_masks_labels.py``):

* the relation bias diagonal is 0 (a team still attends to itself through the q.k term);
* ``node_mask`` is the KEY mask on both axes (v1: alive at phase start; v2: teams causally alive at the cut a_k).  The
  optional ``query_mask`` (batches of a v2 dump) is the TOKEN mask: ``node_mask | target`` -- it adds the *comeback*
  rows, risk-set teams that are causally dead at the cut.  A query token always attends to its own key in addition to
  the ``node_mask`` keys (team axis and, with the same rule, the phase axis: the slot's own position is always allowed,
  past slots count only where their own ``node_mask`` is set), so no attention row of a query token is ever empty,
  while no other team (and no later slot of the same team) can attend to a non-key token.  Without ``query_mask`` the
  model behaves exactly as before (tokens = keys).
* tokens outside the token mask are forced to exactly zero after every sub-layer and every raw input is sanitised with
  ``torch.where`` (never ``* mask``) so NaN / huge values at dead positions cannot reach a live output, not even through
  gradients.
* Rows with no valid key return zeros (never NaN).
* The hazard head reads the final token ``(p, i)``; the rank head is a separate linear score
  (higher = worse expected final placement, see :mod:`model_stgcn.ptr.loss`).  The loss and the metrics read the
  ``target_mask`` rows only (a subset of the token mask): a comeback row is scored exactly once.

Stage 2 (``cfg.player``, the M11RZ-P arms; design doc 3 / 4.1).  Before the blocks above, every alive player token of a
phase slot -- ``[Xp (player features) | the team's token features | player_is_ai]`` -- passes ``player_layers``
:class:`PlayerBlock` s (relation-biased attention over ALL alive players of the match in that slot, FFN) and is pooled
into its team by :class:`TeamPool` (attention pooling over the team's alive players); the pooled vector is added to the
team's stage-1 token embedding, so the team token = ``token_mlp(x) + phase + pool(players)`` and M11RZ-P nests M11RZ
(shared components have the same init streams).  The player attention bias is ``sum_r b[r, h] W_r[i, j]`` over the
channels ``player_relation_types`` (teammate, enemy_prox, combat_out, combat_in; the ``-tm`` arm has no teammate
channel).  A player is alive only if the dump says so AND its team is alive at that slot; dead players are zeroed and
masked everywhere (never ``* mask``), and a team without an alive player gets a zero pooled vector.
"""

from __future__ import annotations

import contextlib
import zlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_stgcn.ptr.player import check_player_relation_types
from model_stgcn.ptr.relations import RELATION_TYPES, check_relation_types

GEO_DIM = 6  # (pos - c)/r (2) + dist/r (1), for the current and the next zone circle
GEO_CLAMP = 50.0  # overflow guard for dist/radius when a circle has (almost) shrunk to a point
REL_INIT_STD = 0.1


@dataclass(frozen=True)
class PTRConfig:
    in_features: int
    n_phases: int = 8
    zone_dim: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    phase_axis: bool = True
    team_axis: bool = True
    relation: bool = False
    relation_types: tuple[str, ...] = RELATION_TYPES
    zone: bool = False
    zone_hidden: int = 16
    player: bool = False  # stage 2: player tokens, player attention, pooling to the team tokens
    player_features: int = 0  # columns of Xp
    player_relation_types: tuple[str, ...] = ()
    player_layers: int = 1  # PlayerBlocks before the pooling (0: pool the player embeddings directly)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")
        if self.relation and not self.team_axis:
            raise ValueError("relation bias lives on the team axis: relation=True needs team_axis=True")
        if self.zone and not self.team_axis:
            raise ValueError("the zone token joins the team-axis set: zone=True needs team_axis=True")
        object.__setattr__(self, "relation_types", check_relation_types(self.relation_types))
        if self.player:
            if self.player_features < 1:
                raise ValueError("player=True needs player_features >= 1 (columns of Xp)")
            if self.player_layers < 0:
                raise ValueError("player_layers must be >= 0")
            object.__setattr__(self, "player_relation_types", check_player_relation_types(self.player_relation_types))


class _Init:
    """Per-component RNG scopes: the same ``init_seed`` gives shared components the same weights in
    every arm, whichever optional components (phase attention, relation bias, zone) exist."""

    def __init__(self, seed: int | None) -> None:
        self.seed = seed

    @contextlib.contextmanager
    def scope(self, tag: str):
        if self.seed is None:
            yield
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((int(self.seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2**31 - 1))
            yield


def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with an additive logit bias and a boolean key mask.

    ``q [..., H, Lq, dh]``, ``k/v [..., H, Lk, dh]``, ``mask`` (True = may attend) and ``bias
    [..., H, Lq, Lk]`` broadcast against the logits.  Query rows without any allowed key give 0.
    """
    logits = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if bias is not None:
        logits = logits + bias
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 2)
    attn = torch.softmax(logits, dim=-1)
    attn = attn * mask.any(dim=-1, keepdim=True).to(attn.dtype)  # empty rows -> exactly zero
    attn = F.dropout(attn, dropout_p, training)
    return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head, self.dropout = n_heads, d_model // n_heads, float(dropout)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, xq, xkv, mask, bias=None):
        H, dh = self.n_heads, self.d_head
        q = self.q(xq).unflatten(-1, (H, dh)).transpose(-2, -3)
        k = self.k(xkv).unflatten(-1, (H, dh)).transpose(-2, -3)
        v = self.v(xkv).unflatten(-1, (H, dh)).transpose(-2, -3)
        out = masked_attention(q, k, v, mask, bias, self.dropout, self.training)
        return self.o(out.transpose(-2, -3).flatten(-2))


def zone_geometry(
    pos: torch.Tensor, zone: torch.Tensor, pos_ok: torch.Tensor, zone_ok: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Team position relative to the current / next zone circle.

    ``pos [B,P,N,2]``, ``zone [B,P,>=6]`` in canonical column order (cur cx, cy, r, next cx, cy, r),
    ``pos_ok [B,P,N]``, ``zone_ok [B,P]``.  Returns ``geo [B,P,N,6]`` (zeros where invalid) and the
    validity mask ``[B,P,N]``.
    """
    ok = pos_ok & zone_ok.unsqueeze(-1)
    pos = torch.where(ok.unsqueeze(-1), pos, torch.zeros_like(pos))
    zone = torch.where(zone_ok.unsqueeze(-1), zone, torch.zeros_like(zone))
    feats = []
    for cx, r in ((0, 2), (3, 5)):
        centre = zone[..., cx : cx + 2].unsqueeze(2)  # [B,P,1,2]
        radius = zone[..., r].clamp(min=1e-6)[:, :, None, None]  # [B,P,1,1]
        rel = (pos - centre) / radius
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        feats += [rel.clamp(-GEO_CLAMP, GEO_CLAMP), dist.clamp(max=GEO_CLAMP)]
    geo = torch.cat(feats, dim=-1)
    return torch.where(ok.unsqueeze(-1), geo, torch.zeros_like(geo)), ok


class PTRBlock(nn.Module):
    def __init__(self, cfg: PTRConfig, layer: int, init: _Init) -> None:
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.cfg = cfg
        if cfg.phase_axis:
            with init.scope(f"b{layer}.phase"):
                self.ln_p = nn.LayerNorm(d)
                self.attn_p = MultiHeadAttention(d, H, cfg.dropout)
        else:
            self.ln_p = self.attn_p = None
        with init.scope(f"b{layer}.team"):
            self.ln_t = nn.LayerNorm(d)
            self.attn_t = MultiHeadAttention(d, H, cfg.dropout)
        if cfg.relation:
            with init.scope(f"b{layer}.rel"):
                self.rel_b = nn.Parameter(torch.randn(len(cfg.relation_types), H) * REL_INIT_STD)
        else:
            self.register_parameter("rel_b", None)
        if cfg.zone:
            with init.scope(f"b{layer}.zone"):
                self.zone_bias = nn.Sequential(
                    nn.Linear(GEO_DIM, cfg.zone_hidden), nn.GELU(), nn.Linear(cfg.zone_hidden, H)
                )
        else:
            self.zone_bias = None
        with init.scope(f"b{layer}.ffn"):
            self.ln_f = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, cfg.ffn_mult * d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.ffn_mult * d, d),
            )
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal", torch.tril(torch.ones(cfg.n_phases, cfg.n_phases, dtype=torch.bool)), persistent=False
        )

    def relation_bias(self, rel: torch.Tensor) -> torch.Tensor:
        """``rel [B,P,R,N,N]`` (query i, key j) -> additive logit bias ``[B,P,H,N,N]``."""
        return torch.einsum("bprij,rh->bphij", rel, self.rel_b)

    def _team_bias(self, rel, geo, geo_ok, n_teams):
        bias = None
        if self.rel_b is not None:
            bias = self.relation_bias(rel)
        if self.zone_bias is not None:
            zb = self.zone_bias(geo) * geo_ok.unsqueeze(-1).to(geo.dtype)  # [B,P,N,H]
            zb = zb.permute(0, 1, 3, 2)  # [B,P,H,N]
            if bias is None:
                bias = zb.new_zeros(*zb.shape, n_teams)
            top = torch.cat([bias, zb.unsqueeze(-1)], dim=-1)  # keys: teams + zone
            bottom = torch.cat([zb.unsqueeze(-2), zb.new_zeros(*zb.shape[:-1], 1, 1)], dim=-1)
            bias = torch.cat([top, bottom], dim=-2)  # queries: teams + zone
        return bias

    def forward(self, h, z, alive, rel, geo, geo_ok, zone_ok, query=None):
        """``alive`` is the KEY mask ``[B,P,N]``; ``query`` the token mask (default: the key mask, the v1 behaviour).  A query
        token always attends to its own key (team axis) / its own slot (phase axis) in addition to the key mask."""
        key = alive
        own = query is not None and query is not key
        query = key if query is None else query
        n_teams = h.shape[2]
        P = h.shape[1]
        a = query.unsqueeze(-1)
        zero = torch.zeros((), dtype=h.dtype, device=h.device)

        if self.attn_p is not None:  # phase axis: per team, causal over phases, key slots only (+ the query's own slot)
            t = self.ln_p(h).transpose(1, 2)  # [B,N,P,d]
            mask = self.causal[:P, :P][None, None, None] & key.transpose(1, 2)[:, :, None, None, :]
            if own:
                eye_p = torch.eye(P, dtype=torch.bool, device=h.device)
                mask = mask | (eye_p & query.transpose(1, 2)[:, :, None, :, None])
            upd = self.attn_p(t, t, mask).transpose(1, 2)
            h = h + torch.where(a, self.drop(upd), zero)

        toks = self.ln_t(h)  # team axis: per phase over the key teams (+ zone token) and the query's own key
        key_alive, qry_alive = key, query
        if z is not None:
            toks = torch.cat([toks, self.ln_t(z)], dim=2)
            key_alive = torch.cat([key, zone_ok.unsqueeze(-1)], dim=2)
            qry_alive = torch.cat([query, zone_ok.unsqueeze(-1)], dim=2)
        mask = key_alive[:, :, None, None, :]
        m = toks.shape[2]
        if own:
            eye = torch.eye(m, dtype=torch.bool, device=h.device)
            mask = mask | (eye & qry_alive[:, :, None, :, None])
        if not self.cfg.team_axis:  # self only: each token may only look at itself
            mask = mask & torch.eye(m, dtype=torch.bool, device=h.device)
        upd = self.attn_t(toks, toks, mask, self._team_bias(rel, geo, geo_ok, n_teams))
        h = h + torch.where(a, self.drop(upd[:, :, :n_teams]), zero)
        if z is not None:
            zmask = zone_ok[:, :, None, None]
            z = z + torch.where(zmask, self.drop(upd[:, :, n_teams:]), zero)

        h = h + torch.where(a, self.drop(self.ffn(self.ln_f(h))), zero)
        if z is not None:
            z = z + torch.where(zone_ok[:, :, None, None], self.drop(self.ffn(self.ln_f(z))), zero)
        return h, z


class PlayerBlock(nn.Module):
    """Pre-LN block of the player layer: attention over ALL alive players of one phase slot with the additive relation
    bias ``sum_r b[r, h] W_r[i, j]`` (``b`` per block and head), then an FFN.  Dead players neither attend nor are
    attended to and stay exactly zero."""

    def __init__(self, cfg: PTRConfig, layer: int, init: _Init) -> None:
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        with init.scope(f"p{layer}.attn"):
            self.ln = nn.LayerNorm(d)
            self.attn = MultiHeadAttention(d, H, cfg.dropout)
        with init.scope(f"p{layer}.rel"):
            self.rel_b = nn.Parameter(torch.randn(len(cfg.player_relation_types), H) * REL_INIT_STD)
        with init.scope(f"p{layer}.ffn"):
            self.ln_f = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, cfg.ffn_mult * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_mult * d, d)
            )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, alive: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """``h [B,P,M,d]`` (phase slot P, player M), ``alive [B,P,M]``, ``rel [B,P,R,M,M]`` (diagonal 0, dead pairs 0)."""
        a = alive.unsqueeze(-1)
        zero = torch.zeros((), dtype=h.dtype, device=h.device)
        bias = torch.einsum("bprij,rh->bphij", rel, self.rel_b)
        t = self.ln(h)
        upd = self.attn(t, t, alive[:, :, None, None, :], bias)
        h = h + torch.where(a, self.drop(upd), zero)
        return h + torch.where(a, self.drop(self.ffn(self.ln_f(h))), zero)


class TeamPool(nn.Module):
    """Attention pooling of the alive players of a team into one vector per (phase slot, team).

    ``score = w . LN(h)``, a softmax over the team's alive players, values ``V LN(h)``; a team without an alive player
    (or a padded team) gets exactly zero.  Players only ever contribute to their own team."""

    def __init__(self, d_model: int, init: _Init) -> None:
        super().__init__()
        with init.scope("player_pool"):
            self.ln = nn.LayerNorm(d_model)
            self.score = nn.Linear(d_model, 1)
            self.value = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, alive: torch.Tensor, team: torch.Tensor, n_teams: int):
        """``h [B,P,M,d]``, ``alive [B,P,M]``, ``team [B,M]`` (team index of every player) -> ``([B,P,N,d], weights [B,P,N,M])``."""
        t = self.ln(h)
        s = self.score(t).squeeze(-1)
        member = team[:, None, :] == torch.arange(n_teams, device=h.device)[None, :, None]  # [B,N,M]
        ok = member[:, None] & alive[:, :, None, :]  # [B,P,N,M]
        logits = s[:, :, None, :].masked_fill(~ok, torch.finfo(s.dtype).min / 2)
        attn = torch.softmax(logits, dim=-1) * ok.any(dim=-1, keepdim=True).to(s.dtype)
        return torch.einsum("bpnm,bpmd->bpnd", attn, self.value(t)), attn


class PTRFormer(nn.Module):
    def __init__(self, cfg: PTRConfig, init_seed: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        init = _Init(init_seed)
        d = cfg.d_model
        with init.scope("token"):
            self.token_mlp = nn.Sequential(nn.Linear(cfg.in_features, d), nn.GELU(), nn.Linear(d, d))
        with init.scope("phase_emb"):
            self.phase_emb = nn.Embedding(cfg.n_phases, d)
            nn.init.normal_(self.phase_emb.weight, std=0.02)
        if cfg.zone:
            with init.scope("zone_token"):
                self.zone_mlp = nn.Sequential(nn.Linear(cfg.zone_dim, d), nn.GELU(), nn.Linear(d, d))
        if cfg.player:
            with init.scope("player_token"):
                self.player_mlp = nn.Sequential(
                    nn.Linear(cfg.player_features + cfg.in_features + 1, d), nn.GELU(), nn.Linear(d, d)
                )
            self.player_blocks = nn.ModuleList(PlayerBlock(cfg, layer, init) for layer in range(cfg.player_layers))
            self.team_pool = TeamPool(d, init)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(PTRBlock(cfg, layer, init) for layer in range(cfg.n_layers))
        with init.scope("heads"):
            self.final_ln = nn.LayerNorm(d)
            self.hazard_head = nn.Linear(d, 1)
            self.rank_head = nn.Linear(d, 1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _relations(self, batch, alive, query=None):
        """Relation channels ``[B,P,R,N,N]`` (query i, key j); a pair counts only if i is a token and j a KEY (v1: both alive)."""
        rel = batch.get("rel")
        B, P, N = alive.shape
        want = (B, P, len(self.cfg.relation_types), N, N)
        if rel is None or tuple(rel.shape) != want:
            got = None if rel is None else tuple(rel.shape)
            raise ValueError(f"relation bias needs batch['rel'] of shape {want}, got {got}")
        eye = torch.eye(N, dtype=torch.bool, device=alive.device)  # relation bias diagonal = 0
        query = alive if query is None else query
        pair_alive = (query.unsqueeze(-1) & alive.unsqueeze(-2) & ~eye).unsqueeze(2)
        return torch.where(pair_alive, rel, torch.zeros_like(rel))

    def _player_relations(self, batch, p_alive):
        rel = batch.get("rel_p")
        B, P, M = p_alive.shape
        want = (B, P, len(self.cfg.player_relation_types), M, M)
        if rel is None or tuple(rel.shape) != want:
            got = None if rel is None else tuple(rel.shape)
            raise ValueError(f"the player attention needs batch['rel_p'] of shape {want}, got {got}")
        eye = torch.eye(M, dtype=torch.bool, device=p_alive.device)  # relation bias diagonal = 0
        pair_alive = (p_alive.unsqueeze(-1) & p_alive.unsqueeze(-2) & ~eye).unsqueeze(2)
        return torch.where(pair_alive, rel, torch.zeros_like(rel))

    def pool_players(self, batch: dict, x: torch.Tensor, alive: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Stage 2: player tokens -> player attention -> attention pooling; returns ``[B, P, N, d]``.

        ``x`` is the (dead-token zeroed) team token features, ``alive`` the team alive mask, ``phase`` the phase embeddings."""
        cfg = self.cfg
        for key in ("xp", "p_alive", "p_team", "p_ai", "rel_p"):
            if key not in batch:
                raise ValueError(f"arm with a player layer needs batch['{key}'] (build the dataset with the player dump)")
        xp, team = batch["xp"], batch["p_team"].long()
        B, P, M, _ = xp.shape
        N = x.shape[2]
        # a player is alive iff the dump says so and its team is alive at the slot (dead teams cannot leak in)
        p_alive = batch["p_alive"].bool() & torch.gather(alive, 2, team[:, None, :].expand(B, P, M))
        xt = torch.gather(x, 2, team[:, None, :, None].expand(B, P, M, x.shape[-1]))
        ai = batch["p_ai"][:, None, :, None].expand(B, P, M, 1).to(xp.dtype)
        tok = torch.cat([xp, xt, ai], dim=-1)
        tok = torch.where(p_alive.unsqueeze(-1), tok, torch.zeros_like(tok))
        hp = self.player_mlp(tok) + phase[None, :, None, :]
        hp = torch.where(p_alive.unsqueeze(-1), self.emb_drop(hp), torch.zeros_like(hp))
        rel = self._player_relations(batch, p_alive)
        for block in self.player_blocks:
            hp = block(hp, p_alive, rel)
        pooled, _ = self.team_pool(hp, p_alive, team, N)
        return pooled

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        x, key = batch["x"], batch["node_mask"].bool()  # node_mask = the attention KEY mask
        qm = batch.get("query_mask")  # v2 batches: the token mask (node_mask | target); absent -> tokens = keys (v1)
        query = key if qm is None else (qm.bool() | key)
        B, P, N, _ = x.shape
        if P > cfg.n_phases:
            raise ValueError(f"batch has {P} phase slots, the model was built for {cfg.n_phases}")
        x = torch.where(query.unsqueeze(-1), x, torch.zeros_like(x))
        phase = self.phase_emb.weight[:P]
        h = self.token_mlp(x) + phase[None, :, None, :]
        if cfg.player:
            h = h + self.pool_players(batch, x, query, phase)
        h = self.emb_drop(h)
        h = torch.where(query.unsqueeze(-1), h, torch.zeros_like(h))

        rel = self._relations(batch, key, None if qm is None else query) if cfg.relation else None
        z = geo = geo_ok = zone_ok = None
        if cfg.zone:
            zone_ok = batch["zone_valid"].bool()
            zs = torch.where(zone_ok.unsqueeze(-1), batch["zone_std"], torch.zeros_like(batch["zone_std"]))
            z = self.emb_drop(self.zone_mlp(zs) + phase[None])
            z = torch.where(zone_ok.unsqueeze(-1), z, torch.zeros_like(z)).unsqueeze(2)  # [B,P,1,d]
            pos_ok = batch.get("pos_valid")
            pos_ok = query if pos_ok is None else query & pos_ok.bool()
            geo, geo_ok = zone_geometry(batch["pos"], batch["zone"], pos_ok, zone_ok)
        for block in self.blocks:
            h, z = block(h, z, key, rel, geo, geo_ok, zone_ok, query=None if qm is None else query)
        out = self.final_ln(h)
        return {
            "hazard_logit": self.hazard_head(out).squeeze(-1),
            "rank_score": self.rank_head(out).squeeze(-1),
        }
