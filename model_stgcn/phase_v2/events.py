"""Team-attributed combat events and loot counters from the raw event tables (ports of stdb.etl spatial events,
rq1_feature_etl.build_team_phase_combat / build_team_phase_loot and phase_edge_etl's combat edges)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

KIND_DAMAGE, KIND_GROGGY, KIND_KILL = 0, 1, 2


def account_team_table(src, pos: dict[str, np.ndarray]) -> np.ndarray:
    """Team ID of every account index (-1 unknown): the roster first, then the team of the account's first position
    record for accounts outside the roster (stdb.etl ``_team_lookup_from_positions`` for non-participants)."""
    n = len(src.accounts)
    out = np.full(n, -1, dtype=np.int64)
    idx = src.account_index()
    for team, pid, _dt, _ts in src.roster:
        a = idx.get(pid)
        if a is not None:
            out[a] = int(team)
    if len(pos["w"]):
        unknown = out[pos["acct"]] < 0
        if unknown.any():
            a_u, t_u = pos["acct"][unknown], pos["team"][unknown]
            first = np.unique(a_u, return_index=True)[1]  # first record per account (records are time ordered)
            out[a_u[first]] = t_u[first]
    return out


@dataclass
class TeamEvents:
    """Concatenated kill / damage / groggy rows with team attribution (team ids, -1 unknown)."""

    w: np.ndarray
    kind: np.ndarray  # 0 damage, 1 groggy, 2 kill
    att: np.ndarray  # attacker team id
    vic: np.ndarray  # victim team id
    damage: np.ndarray  # damage (kind 0), else 0
    px: np.ndarray  # event position: victim (x, y) else attacker (x, y), NaN if neither
    py: np.ndarray


def _lookup(acct_team: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), -1, dtype=np.int64)
    ok = idx >= 0
    out[ok] = acct_team[idx[ok]]
    return out


def build_team_events(src, acct_team: np.ndarray) -> TeamEvents:
    parts = []
    k, d, g = src.kills, src.damage, src.groggy
    # kills: attacker team by lookup, victim team from the row (else lookup)
    vt = k["victim_team"].copy()
    miss = vt < 0
    vt[miss] = _lookup(acct_team, k["victim"])[miss]
    kpx = np.where(np.isfinite(k["vx"]), k["vx"], k["kx"])
    kpy = np.where(np.isfinite(k["vx"]), k["vy"], k["ky"])
    parts.append((k["w"], np.full(len(k["w"]), KIND_KILL), _lookup(acct_team, k["killer"]), vt,
                  np.zeros(len(k["w"])), kpx, kpy))  # fmt: skip
    dpx = np.where(np.isfinite(d["vx"]), d["vx"], d["ax"])
    dpy = np.where(np.isfinite(d["vx"]), d["vy"], d["ay"])
    parts.append((d["w"], np.full(len(d["w"]), KIND_DAMAGE), _lookup(acct_team, d["attacker"]),
                  _lookup(acct_team, d["victim"]), np.nan_to_num(d["damage"], nan=0.0), dpx, dpy))  # fmt: skip
    gpx = np.where(np.isfinite(g["vx"]), g["vx"], g["ax"])
    gpy = np.where(np.isfinite(g["vx"]), g["vy"], g["ay"])
    parts.append((g["w"], np.full(len(g["w"]), KIND_GROGGY), _lookup(acct_team, g["attacker"]),
                  _lookup(acct_team, g["victim"]), np.zeros(len(g["w"])), gpx, gpy))  # fmt: skip
    cols = [np.concatenate([p[i] for p in parts]) for i in range(7)]
    order = np.argsort(cols[0], kind="stable")
    cols = [c[order] for c in cols]
    return TeamEvents(w=cols[0], kind=cols[1].astype(np.int64), att=cols[2].astype(np.int64),
                      vic=cols[3].astype(np.int64), damage=cols[4], px=cols[5], py=cols[6])  # fmt: skip


def window(ev: TeamEvents, lo: float, hi: float) -> TeamEvents:
    m = (ev.w >= lo) & (ev.w < hi)
    return TeamEvents(*(getattr(ev, f)[m] for f in ("w", "kind", "att", "vic", "damage", "px", "py")))


# --------------------------------------------------------------------------------------
# combat features (rq1_feature_etl.build_team_phase_combat)
# --------------------------------------------------------------------------------------
COMBAT_COLUMNS = (
    "damage_dealt", "damage_taken", "damage_events_dealt", "damage_events_taken", "groggy_dealt", "groggy_taken",
    "kills_dealt", "kills_taken", "distinct_opponents_damaged", "distinct_opponents_damaged_by", "combat_initiated",
)  # fmt: skip


def combat_features(ev: TeamEvents, team_index: dict[int, int]) -> dict[str, np.ndarray]:
    """Per-team (N,) arrays.  ``combat_initiated`` is 1.0 / 0.0 / NaN (NULL: no damage either way)."""
    N = len(team_index)
    out = {c: np.zeros(N) for c in COMBAT_COLUMNS}
    out["combat_initiated"] = np.full(N, np.nan)
    both = (ev.att >= 0) & (ev.vic >= 0) & (ev.att != ev.vic)
    if not both.any():
        return out
    att_id, vic_id = ev.att[both], ev.vic[both]
    a = np.fromiter((team_index.get(int(t), -1) for t in att_id), dtype=np.int64, count=len(att_id))
    v = np.fromiter((team_index.get(int(t), -1) for t in vic_id), dtype=np.int64, count=len(vic_id))
    kind, dmg, w = ev.kind[both], ev.damage[both], ev.w[both]
    first_dealt = np.full(N, np.inf)
    first_taken = np.full(N, np.inf)
    opp_dealt = [set() for _ in range(N)]
    opp_taken = [set() for _ in range(N)]
    for ai, vi, aid, vid, kd, dm, t in zip(a, v, att_id, vic_id, kind, dmg, w):
        if ai >= 0:  # role dealt of the attacker team
            if kd == KIND_DAMAGE:
                out["damage_dealt"][ai] += dm
                out["damage_events_dealt"][ai] += 1
                opp_dealt[ai].add(int(vid))
                first_dealt[ai] = min(first_dealt[ai], t)
            elif kd == KIND_GROGGY:
                out["groggy_dealt"][ai] += 1
            else:
                out["kills_dealt"][ai] += 1
        if vi >= 0:  # role taken of the victim team
            if kd == KIND_DAMAGE:
                out["damage_taken"][vi] += dm
                out["damage_events_taken"][vi] += 1
                opp_taken[vi].add(int(aid))
                first_taken[vi] = min(first_taken[vi], t)
            elif kd == KIND_GROGGY:
                out["groggy_taken"][vi] += 1
            else:
                out["kills_taken"][vi] += 1
    out["distinct_opponents_damaged"] = np.asarray([len(s) for s in opp_dealt], dtype=np.float64)
    out["distinct_opponents_damaged_by"] = np.asarray([len(s) for s in opp_taken], dtype=np.float64)
    fd_inf, ft_inf = np.isinf(first_dealt), np.isinf(first_taken)
    init = np.full(N, np.nan)
    init[~fd_inf & ft_inf] = 1.0
    init[fd_inf & ~ft_inf] = 0.0
    both_ok = ~fd_inf & ~ft_inf
    init[both_ok] = (first_dealt[both_ok] <= first_taken[both_ok]).astype(np.float64)
    out["combat_initiated"] = init
    return out


def combat_edges(ev: TeamEvents, team_index: dict[int, int]) -> list[tuple[int, int, float]]:
    """Directed attacker -> victim combat edges ``(i, j, weight)``: ln(1 + damage + 100 kills + 50 groggy)
    (phase_edge_etl); only pairs with a positive intensity; teams outside the roster are dropped."""
    both = (ev.att >= 0) & (ev.vic >= 0) & (ev.att != ev.vic)
    acc: dict[tuple[int, int], list[float]] = {}
    for at, vi, kd, dm in zip(ev.att[both], ev.vic[both], ev.kind[both], ev.damage[both]):
        i, j = team_index.get(int(at)), team_index.get(int(vi))
        if i is None or j is None:
            continue
        rec = acc.setdefault((i, j), [0.0, 0.0, 0.0])  # damage_sum, kills, groggy
        if kd == KIND_DAMAGE:
            rec[0] += dm
        elif kd == KIND_KILL:
            rec[1] += 1
        else:
            rec[2] += 1
    out = []
    for (i, j), (dsum, kills, groggy) in sorted(acc.items()):
        inten = max(0.0, dsum) + 100.0 * kills + 50.0 * groggy
        if inten > 0:
            out.append((i, j, float(np.log(1.0 + inten))))
    return out


# --------------------------------------------------------------------------------------
# loot features (rq1_feature_etl.build_team_phase_loot)
# --------------------------------------------------------------------------------------
LOOT_COLUMNS = (
    "pickup_count", "equip_count", "use_count", "unique_item_count", "weapon_pickup_count", "weapon_equip_count",
    "ammo_pickup_count", "armor_pickup_count", "throwable_pickup_count", "heal_use_count", "boost_use_count",
    "carepackage_pickup_count", "lootbox_pickup_count",
)  # fmt: skip


def _like(pattern: str) -> re.Pattern:
    """SQL LIKE (``%`` any string, ``_`` any single character) as a full-match regex."""
    rx = "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern)
    return re.compile(rx, re.DOTALL)


_RX = {
    "weapon": _like("item_weapon_%"),
    "ammo": _like("%ammo%"),
    "armor": _like("%armor%"), "helmet": _like("%helmet%"), "back": _like("%back%"),
    "throw": _like("%throw%"),
    "heal": _like("%heal%"), "firstaid": _like("%firstaid%"), "medkit": _like("%medkit%"), "bandage": _like("%bandage%"),
    "boost": _like("%boost%"), "energydrink": _like("%energydrink%"), "painkiller": _like("%painkiller%"),
    "adrenaline": _like("%adrenaline%"),
}  # fmt: skip


def _flags(item: str | None, cat: str | None, sub: str | None) -> dict[str, bool]:
    il, cl, sl = (item or "").lower(), (cat or "").lower(), (sub or "").lower()
    m = lambda k: _RX[k].fullmatch(il) is not None  # noqa: E731
    return {
        "weapon": m("weapon") or cl == "weapon",
        "ammo": m("ammo") or cl == "ammunition" or sl == "ammo",
        "armor": m("armor") or m("helmet") or m("back") or sl in ("vest", "helmet", "backpack"),
        "throwable": sl == "throwable" or m("throw"),
        "heal": m("heal") or m("firstaid") or m("medkit") or m("bandage"),
        "boost": m("boost") or m("energydrink") or m("painkiller") or m("adrenaline"),
    }


def loot_features(items: dict[str, dict[str, np.ndarray]], lo: float, hi: float, team_index: dict[int, int]) -> dict[str, np.ndarray]:
    """Per-team (N,) counters of the pickup / equip / use events with ``lo <= w < hi``."""
    N = len(team_index)
    out = {c: np.zeros(N) for c in LOOT_COLUMNS}
    uniq: list[set] = [set() for _ in range(N)]
    cache: dict[tuple, dict[str, bool]] = {}
    for kind in ("pickup", "equip", "use"):
        t = items[kind]
        m = (t["w"] >= lo) & (t["w"] < hi)
        if not m.any():
            continue
        for team, item, cat, sub, stack, care, loot in zip(
            t["team"][m], t["item"][m], t["cat"][m], t["sub"][m], t["stack"][m], t["carepackage"][m], t["lootbox"][m]
        ):
            i = team_index.get(int(team))
            if i is None:
                continue
            key = (item, cat, sub)
            fl = cache.get(key)
            if fl is None:
                fl = cache[key] = _flags(item, cat, sub)
            out[f"{kind}_count"][i] += 1
            if item is not None:
                uniq[i].add(item)
            if kind == "pickup":
                if fl["weapon"]:
                    out["weapon_pickup_count"][i] += 1
                if fl["ammo"]:
                    out["ammo_pickup_count"][i] += stack
                if fl["armor"]:
                    out["armor_pickup_count"][i] += 1
                if fl["throwable"]:
                    out["throwable_pickup_count"][i] += 1
                if care:
                    out["carepackage_pickup_count"][i] += 1
                if loot:
                    out["lootbox_pickup_count"][i] += 1
            elif kind == "equip":
                if fl["weapon"]:
                    out["weapon_equip_count"][i] += 1
            else:
                if fl["heal"]:
                    out["heal_use_count"][i] += 1
                if fl["boost"]:
                    out["boost_use_count"][i] += 1
    out["unique_item_count"] = np.asarray([len(s) for s in uniq], dtype=np.float64)
    return out
