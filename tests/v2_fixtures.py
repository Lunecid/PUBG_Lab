"""Synthetic dumps in the **v2 record format** (``model_stgcn/dumps/bigcomp2027_phase_v2/CONTRACT.md``), DB-free.

Built on :func:`model_stgcn.ptr.synthetic.make_dump` (features, edges, rank / death times) and re-labelled with the v2
rules, so every record carries the keys of the v2 dump and a ``meta['version'] == 'v2'``::

    s_k = 90 + 200 (k - 1)      b_k = s_{k+1} = 90 + 200 k   (k = 1..P-1)      a_k = b_k - 10        phase P is final: no cut
    T   = synthetic death time (+inf for the winner)          C_m = match_end_s (after every finite T)
    risk set   T > b_k            y = 1  iff  b_k < T <= b_{k+1}   ((b_k, C_m] for the last cut)
    node_mask  causally alive at a_k:  T > a_k   (so a team that dies in the last 10 s before b_k is a KEY but not a risk row)
    comeback   a few risk-set rows are set causally dead at the cut (``node_mask`` False, ``target`` True): the row keeps its
               query token (``query_mask``), only its key flag is off.  ``comeback_prob`` controls how many.
    excluded   a few (match, cut) pairs are excluded as the producer does for a truncated log: ``cut_valid`` False and an
               all-False risk row; the winners must not be re-added there.
"""

from __future__ import annotations

import numpy as np
import torch

from model_stgcn.ptr.synthetic import make_dump

N_PHASES = 8


def _bounds(P: int):
    s = np.array([90.0 + 200.0 * p for p in range(P)])
    b = np.full(P, np.nan)
    b[: P - 1] = s[1:]
    return s, b


def to_v2_match(m: dict, rng: np.random.Generator, *, comeback_prob: float = 0.06, exclude_prob: float = 0.12) -> dict:
    P, N = m["X"].shape[0], m["X"].shape[1]
    s, b = _bounds(P)
    T = m["team_death_s"].double().numpy().copy()  # +inf for the winner (event_death format)
    won = m["team_won"].numpy().astype(bool)
    T[won] = np.inf
    finite = T[np.isfinite(T)]
    c_m = float(finite.max() + 5.0) if finite.size else 1e4
    s32, b32 = s.astype(np.float32), b.astype(np.float32)
    T32 = T.astype(np.float32)
    Td = T32.astype(np.float64)
    bd = b32.astype(np.float64)
    cut = bd - 10.0
    known = ~np.isnan(Td)
    node = np.zeros((P, N), dtype=bool)
    target = np.zeros((P, N), dtype=bool)
    y = np.zeros((P, N), dtype=np.float32)
    cut_valid = np.zeros(P, dtype=bool)
    excluded: dict[int, str] = {}
    for p in range(P - 1):
        node[p] = known & (Td > cut[p])
        risk = known & (Td > bd[p])
        if rng.random() < exclude_prob:  # an excluded cut: nodes exist, no risk-set row (winners included)
            excluded[p + 1] = "position_gap"
            continue
        cut_valid[p] = True
        e = bd[p + 1] if p + 1 < P - 1 else c_m
        target[p] = risk
        y[p] = (risk & np.isfinite(Td) & (Td <= e)).astype(np.float32)
    comeback = target & node & (rng.random((P, N)) < comeback_prob) & ~won[None, :]
    node = node & ~comeback  # causally dead at the cut although T > b_k
    query = node | target
    cut_mask = np.zeros(P, dtype=bool)
    cut_mask[: P - 1] = True
    out = dict(m)
    for k in ("node_mask_v2", "node_mask_label"):
        out.pop(k, None)
    out.update(
        node_mask=torch.from_numpy(node),
        query_mask=torch.from_numpy(query),
        target_mask_v2=torch.from_numpy(target),
        target_mask=torch.from_numpy(target.copy()),
        y_hazard_v2=torch.from_numpy(y),
        y_hazard=torch.from_numpy(y.copy()),
        team_death_s=torch.from_numpy(T32),
        phase_start_s=torch.from_numpy(s32),
        phase_end_s=torch.from_numpy(b32),
        cut_t=torch.from_numpy(np.where(np.isnan(b32), np.nan, (cut)).astype(np.float32)),
        cut_wall_s=torch.from_numpy(np.where(np.isnan(b32), np.nan, (cut)).astype(np.float32)),
        phase_mask=torch.ones(P, dtype=torch.bool),
        cut_mask=torch.from_numpy(cut_mask),
        cut_valid=torch.from_numpy(cut_valid),
        cut_excluded=dict(excluded),
        cut_terminal=torch.from_numpy(np.arange(P) == P - 2),
        match_end_s=c_m,
        last_phase_idx=P,
        clock_delta_s=0.0,
        clock_delta_median_s=0.0,
    )
    return out


def make_v2_dump(n_matches: int = 6, *, seed: int = 0, n_features: int = 24, cohort: str = "synthetic_v2", split: str = "train",
                 comeback_prob: float = 0.06, exclude_prob: float = 0.12, team_range=(6, 11), **kw) -> dict:
    """A v2-format dump (see the module docstring); ``kw`` goes to ``make_dump`` (e.g. ``alive_count_names``)."""
    base = make_dump(n_matches, n_features=n_features, seed=seed, cohort=cohort, split=split, team_range=team_range,
                     event_death=True, event_marker="field", **kw)
    rng = np.random.default_rng(10_000 + seed)
    matches = [to_v2_match(m, rng, comeback_prob=comeback_prob, exclude_prob=exclude_prob) for m in base["matches"]]
    meta = dict(base["meta"])
    meta.pop("team_death_source", None)
    meta.update(version="v2", v2={"version": "v2"}, design_version="v2.3+phase_dump_v2", n_phases=N_PHASES, n_features=n_features)
    return {"meta": meta, "matches": matches}


def comeback_cells(dump: dict) -> list[tuple[int, int, int]]:
    """``(match index, slot, team index)`` of every comeback row (risk-set row that is not a key)."""
    out = []
    for i, m in enumerate(dump["matches"]):
        cb = (m["target_mask_v2"] & ~m["node_mask"]).numpy()
        out += [(i, int(p), int(t)) for p, t in zip(*np.nonzero(cb))]
    return out


def write_v2_cohort(root, cohort: str = "synthetic_v2", n: int = 6, seed: int = 0, **kw):
    """``<root>/<cohort>/{train,val,test}.pt`` of v2 dumps (one cohort)."""
    from pathlib import Path

    base = Path(root) / cohort
    base.mkdir(parents=True, exist_ok=True)
    for j, split in enumerate(("train", "val", "test")):
        torch.save(make_v2_dump(n, seed=seed + j, cohort=cohort, split=split, **kw), base / f"{split}.pt")
    return base


# ---------------------------------------------------------------------------------------------------------------------
# a real v2 record + the consumers' raw inputs built from ONE synthetic MatchSource (tests/phase_v2_synth.py, the producer's
# own fixture: revive, knocked-but-alive, a comeback team, a team that dies inside the terminal window)
# ---------------------------------------------------------------------------------------------------------------------
def synth_record(*, no_winner: bool = False, **source_kw):
    """``(MatchSource, v2 dump record)`` of the producer's synthetic match, built by ``phase_v2.assemble.build_match_v2``.

    ``no_winner=True`` reproduces the one real match without a ``team_won`` flag (b5134321...): its team that is alive at the end
    keeps ``T = +inf`` through the alive rule alone."""
    from dataclasses import replace

    from model_stgcn.phase_v2.assemble import build_match_v2
    from tests.phase_v2_synth import make_source

    src = make_source(**source_kw)
    if no_winner:
        src = replace(src, team_won=np.zeros_like(src.team_won))
    return src, build_match_v2(src, map_size=8160.0)


def krafton_raw_from_source(src, lab: dict, *, drop_after: float | None = None) -> dict:
    """The KRAFTON ``raw`` dict (adaptation.extract.build_raw layout, v2: ``life`` instead of ``d``) of a ``MatchSource``."""
    from adaptation.features import member_life

    roster = sorted(src.roster, key=lambda r: (int(r[0]), str(r[1])))
    pid = np.array([str(r[1]) for r in roster])
    pid2idx = {p: i for i, p in enumerate(pid)}
    tidx = {int(t): i for i, t in enumerate(lab["team_ids"])}
    team = np.array([tidx[int(r[0])] for r in roster], dtype=np.int64)
    a2p = np.array([pid2idx.get(a, -1) for a in src.accounts], dtype=np.int64)
    N = len(lab["team_ids"])

    def idx(a):
        a = np.asarray(a, dtype=np.int64)
        out = np.full(len(a), -1, dtype=np.int64)
        ok = a >= 0
        out[ok] = a2p[a[ok]]
        return out

    k, g, d, p = src.kills, src.groggy, src.damage, src.pos
    first_member = {int(t): int(np.flatnonzero(team == i)[0]) for t, i in tidx.items() if (team == i).any()}

    def team_member(t):
        return np.array([first_member.get(int(x), -1) for x in t], dtype=np.int64)

    pick, use = src.items["pickup"], src.items["use"]
    slot = np.full((N, 4), -1, dtype=np.int64)
    for s in range(N):
        mem = np.flatnonzero(team == s)[:4]
        slot[s, : len(mem)] = mem
    raw = {
        "pid": pid, "team": team, "is_ai": np.zeros(len(pid), dtype=bool), "N": N, "slot": slot,
        "kills": {"killer_pidx": idx(k["killer"]), "finisher_pidx": np.full(len(k["w"]), -1, dtype=np.int64),
                  "t": np.maximum(k["w"], 0.0), "victim_pidx": idx(k["victim"]), "w_raw": k["w"].copy()},
        "groggy": {"attacker_pidx": idx(g["attacker"]), "victim_pidx": idx(g["victim"]), "t": np.maximum(g["w"], 0.0)},
        "damage": {"attacker_pidx": idx(d["attacker"]), "victim_pidx": idx(d["victim"]), "dmg": d["damage"].copy(),
                   "t": np.maximum(d["w"], 0.0), "wclass": np.array(["AR"] * len(d["w"]), dtype=object)},
        "pos": {"pidx": idx(p["acct"]), "t": p["w"].copy(), "health": p["health"].copy(), "x": p["x"] * 100.0, "y": p["y"] * 100.0,
                "z": p["z"] * 100.0, "veh": p["veh"].copy()},
        "zone": {"t": src.gs["w"].copy(), "sx": src.gs["safe_x"] * 100.0, "sy": src.gs["safe_y"] * 100.0, "sr": src.gs["safe_r"] * 100.0,
                 "px": src.gs["poison_x"] * 100.0, "py": src.gs["poison_y"] * 100.0, "pr": src.gs["poison_r"] * 100.0},
        "pick": {"pidx": team_member(pick["team"]), "t": np.maximum(pick["w"], 0.0), "kind": np.zeros(len(pick["w"]), dtype=np.int8),
                 "item_id": np.asarray(pick["item"], dtype=str), "stack": pick["stack"].astype(np.float64),
                 "removed": np.zeros(len(pick["w"]), dtype=bool), "parachute": np.zeros(len(pick["w"]), dtype=bool)},
        "equip": {"pidx": np.zeros(0, dtype=np.int64), "t": np.zeros(0), "item_id": np.zeros(0, dtype=str)},
        "use": {"pidx": team_member(use["team"]), "t": np.maximum(use["w"], 0.0), "item_id": np.asarray(use["item"], dtype=str)},
    }  # fmt: skip
    for key in ("pick", "use", "equip"):  # drop events of non-members
        pass
    raw["life"] = member_life(raw["kills"]["w_raw"], raw["kills"]["victim_pidx"], raw["pos"]["t"], raw["pos"]["pidx"], raw["pos"]["health"])
    ok = raw["pos"]["pidx"] >= 0
    raw["pos"] = {key: v[ok] for key, v in raw["pos"].items()}
    return raw


def sagong_raw_from_source(src):
    """The Sagong ``MatchRaw`` (clock = W, ``start_epoch`` 0) of a ``MatchSource``."""
    import pandas as pd

    from model_stgcn.sagong2025.features import MatchRaw

    team_of = {str(r[1]): int(r[0]) for r in src.roster}
    first = {}
    for t, a in ((int(r[0]), str(r[1])) for r in src.roster):
        first.setdefault(t, a)
    acc = np.asarray(src.accounts, dtype=object)

    def name(a):
        a = np.asarray(a, dtype=np.int64)
        return np.where(a >= 0, acc[np.clip(a, 0, len(acc) - 1)], None)

    p, d, k, pk = src.pos, src.damage, src.kills, src.items["pickup"]
    return MatchRaw(
        match_id="synthetic", start_epoch=0.0, team_of=team_of,
        pos=pd.DataFrame({"acc": name(p["acct"]), "ep": p["w"], "x": p["x"], "y": p["y"]}),
        dmg=pd.DataFrame({"attacker": name(d["attacker"]), "victim": name(d["victim"]), "damage": d["damage"], "ep": d["w"]}),
        kill=pd.DataFrame({"killer": name(k["killer"]), "victim": name(k["victim"]), "is_suicide": np.zeros(len(k["w"]), dtype=bool), "ep": k["w"]}),
        pick=pd.DataFrame({"acc": [first.get(int(t)) for t in pk["team"]], "event_type": ["pickup"] * len(pk["w"]), "ep": pk["w"]}),
    )
