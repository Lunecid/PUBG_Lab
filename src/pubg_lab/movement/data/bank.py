"""Causal 10-second state banks, shared by both information views."""

import gzip
import json
from collections import defaultdict
import numpy as np

FEATURES = [
    "x_m",
    "y_m",
    "vx_mps",
    "vy_mps",
    "health",
    "position_age_s",
    "alive_evidence",
    "knock_age_s",
    "combat_damage_3s",
    "combat_damage_10s",
    "combat_damage_30s",
    "environment_damage_10s",
    "other_damage_30s",
    "outgoing_damage_30s",
    "vehicle_recorded",
    "equipment_age_s",
    "damage_age_s",
    "equipment_ambiguous_count",
    "pickup_events_30s",
    "use_events_30s",
    "zero_damage_events_30s",
    "landed",
]
PRIVATE = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
F = len(FEATURES)
MAX_ITEMS = 12


def read_raw(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def rolling(rows, times, value, window):
    if not rows:
        return np.zeros(len(times), np.float32)
    ts = np.array([r["t"] for r in rows])
    v = np.r_[0, np.cumsum([value(r) for r in rows])]
    return (
        v[np.searchsorted(ts, times, side="right")]
        - v[np.searchsorted(ts, times - window, side="right")]
    ).astype(np.float32)


def build_bank(raw, vocab):
    tables = raw["tables"]
    P = len(raw["actors"])
    end = max([r["t"] for r in tables["positions"]] + [0])
    times = np.arange(0, int(max(end, 0) // 10) * 10 + 1, 10, dtype=np.float32)
    T = len(times)
    x = np.zeros((T, P, F), np.float32)
    mask = np.zeros_like(x, bool)
    valid = np.zeros((T, P), bool)
    source = np.full((T, P), -1.0, np.float32)
    target = np.zeros((T, P, 2), np.float32)
    target_valid = valid.copy()
    target_time = source.copy()
    target_vehicle = np.zeros((T, P), np.int8)
    items = np.zeros((T, P, MAX_ITEMS), np.int32)
    vehicle = np.zeros((T, P), np.int32)
    item_seen = valid.copy()
    overflows = 0
    ambiguous = 0
    postdeath = 0
    group = {name: defaultdict(list) for name in tables}
    for name, rows in tables.items():
        for r in rows:
            if "actor" in r:
                group[name][r["actor"]].append(r)
    outgoing = defaultdict(list)
    for r in tables["damage"]:
        if r["attacker"] >= 0:
            outgoing[r["attacker"]].append(r)
    for p in range(P):
        rows = group["positions"][p]
        land = min([r["t"] for r in group["parachute_landing"][p]] + [float("inf")])
        death = min([r["t"] for r in group["kills"][p]] + [float("inf")])
        postdeath += sum(r["t"] > death and (r["hp"] or 0) > 0 for r in rows)
        if rows:
            ts = np.array([r["t"] for r in rows])
            xy = np.array(
                [
                    [
                        r["x"] if r["x"] is not None else np.nan,
                        r["y"] if r["y"] is not None else np.nan,
                    ]
                    for r in rows
                ]
            )
            hp = np.array([r["hp"] if r["hp"] is not None else np.nan for r in rows])
            conflicts = np.array([sum(ts == t) > 1 for t in ts])
            sane = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy < 8160).all(1) & ~conflicts
            dts = np.diff(ts)
            speed = np.divide(
                np.linalg.norm(np.diff(xy, axis=0), axis=1),
                dts,
                out=np.full(len(dts), np.inf),
                where=dts > 0,
            )
            sane[1:] &= (speed <= 120) | (dts > 15)
            ix = np.searchsorted(ts, times, side="right") - 1
            has = ix >= 0
            ix = np.maximum(ix, 0)
            age = times - ts[ix]
            fresh = has & (age <= 15)
            ground = sane[ix] & (ts[ix] >= land) & (times < death) & fresh
            valid[:, p] = ground
            source[:, p] = np.where(has, ts[ix], -1)
            x[:, p, :2] = np.where(ground[:, None], xy[ix], 0)
            mask[:, p, :2] = ground[:, None]
            x[:, p, 4] = np.where(fresh & np.isfinite(hp[ix]) & (times < death), hp[ix], 0)
            mask[:, p, 4] = fresh & np.isfinite(hp[ix]) & (times < death)
            x[:, p, 5] = np.where(has, np.minimum(age, 300), 0)
            mask[:, p, 5] = has
            x[:, p, 6] = np.where(times >= death, 0, np.where(fresh, 1, 0))
            mask[:, p, 6] = fresh | (times >= death)
            prior = np.maximum(ix - 1, 0)
            dt = ts[ix] - ts[prior]
            vm = ground & (ix > 0) & sane[prior] & (ts[prior] >= land) & (dt >= 5) & (dt <= 15)
            vel = np.divide(
                xy[ix] - xy[prior], dt[:, None], out=np.zeros((T, 2)), where=dt[:, None] > 0
            )
            x[:, p, 2:4] = np.where(vm[:, None], vel, 0)
            mask[:, p, 2:4] = vm[:, None]
            cats = np.array(
                [
                    vocab["vehicles"].get(r["vehicle_class"], 1) if r["vehicle_class"] else 0
                    for r in rows
                ]
            )
            on = np.array([bool(r["vehicle_class"] or r["vehicle_type"]) for r in rows])
            vehicle[:, p] = np.where(fresh & (times < death), cats[ix], 0)
            x[:, p, 14] = np.where(fresh & (times < death), on[ix], 0)
            mask[:, p, 14] = fresh & (times < death)
            right = np.clip(np.searchsorted(ts, times), 0, len(ts) - 1)
            left = np.clip(right - 1, 0, len(ts) - 1)
            near = np.where(abs(ts[left] - times) <= abs(ts[right] - times), left, right)
            tv = (
                (abs(ts[near] - times) <= 5)
                & sane[near]
                & (ts[near] >= land)
                & (ts[near] < death)
                & (times < death)
                & (times >= land)
            )
            target[:, p] = np.where(tv[:, None], xy[near], 0)
            target_valid[:, p] = tv
            target_time[:, p] = np.where(tv, ts[near], -1)
            target_vehicle[:, p] = np.where(tv, on[near], 0)
        x[:, p, 21] = times >= land
        mask[:, p, 21] = True
        knocks = group["groggy"][p]
        if knocks:
            kt = np.array([r["t"] for r in knocks])
            ki = np.searchsorted(kt, times, side="right") - 1
            ok = ki >= 0
            x[:, p, 7] = np.where(ok, np.minimum(times - kt[np.maximum(ki, 0)], 300), 0)
            mask[:, p, 7] = ok
        ds = group["damage"][p]
        combat = [
            r
            for r in ds
            if r["type"] in ("Damage_Gun", "Damage_Explosion_Grenade", "Damage_Explosion_Molotov")
            and (r["damage"] or 0) > 0
        ]
        env = [
            r
            for r in ds
            if r["type"]
            in (
                "Damage_BlueZone",
                "Damage_Fall",
                "Damage_VehicleCrashHit",
                "Damage_VehicleHit",
                "Damage_Drown",
            )
            and (r["damage"] or 0) > 0
        ]
        other = [r for r in ds if r not in combat and r not in env and (r["damage"] or 0) > 0]
        for col, rs, w in [
            (8, combat, 3),
            (9, combat, 10),
            (10, combat, 30),
            (11, env, 10),
            (12, other, 30),
            (13, outgoing[p], 30),
        ]:
            x[:, p, col] = rolling(rs, times, lambda r: max(r["damage"] or 0, 0), w)
            mask[:, p, col] = True
        for col, name in [(18, "item_pickup"), (19, "item_use")]:
            x[:, p, col] = rolling(group[name][p], times, lambda r: 1, 30)
            mask[:, p, col] = True
        x[:, p, 20] = rolling(ds, times, lambda r: int(r["damage"] == 0), 30)
        mask[:, p, 20] = True
        if ds:
            dt = np.array([r["t"] for r in ds])
            di = np.searchsorted(dt, times, side="right") - 1
            ok = di >= 0
            x[:, p, 16] = np.where(ok, np.minimum(times - dt[np.maximum(di, 0)], 300), 0)
            mask[:, p, 16] = ok
        eq = [
            r
            for r in group["item_equip"][p]
            if r["category"] == "Weapon" or r["subcategory"] in ("Headgear", "Vest", "Throwable")
        ]
        eqgroups = defaultdict(list)
        for r in eq:
            eqgroups[r["t"], r["item"]].append(r)
        eg = sorted(eqgroups.items())
        ei = 0
        state = {}
        last = None
        for k, t in enumerate(times):
            while ei < len(eg) and eg[ei][0][0] <= t:
                (et, item), rs = eg[ei]
                acts = {r["action"] for r in rs}
                last = et
                if acts == {"equip", "unequip"}:
                    state[item] = None
                elif acts == {"equip"}:
                    state[item] = vocab["items"].get(item, 1)
                elif acts == {"unequip"}:
                    state.pop(item, None)
                ei += 1
            known = sorted(v for v in state.values() if v is not None)
            namb = sum(v is None for v in state.values())
            overflows += max(0, len(known) - MAX_ITEMS)
            ambiguous += int(namb > 0)
            items[k, p, : min(MAX_ITEMS, len(known))] = known[:MAX_ITEMS]
            item_seen[k, p] = last is not None
            if last is not None:
                x[k, p, 15] = min(t - last, 300)
                mask[k, p, 15] = True
                x[k, p, 17] = namb
                mask[k, p, 17] = True
    game = np.zeros((T, 9), np.float32)
    gm = np.zeros_like(game, bool)
    gt = np.full(T, -1.0, np.float32)
    gs = tables["game_states"]
    for k, t in enumerate(times):
        hist = [r for r in gs if r["t"] <= t]
        if hist:
            r = hist[-1]
            gt[k] = r["t"]
            vals = [
                r[n]
                for n in (
                    "phase",
                    "alive_players",
                    "alive_teams",
                    "safe_x",
                    "safe_y",
                    "safe_r",
                    "poison_x",
                    "poison_y",
                    "poison_r",
                )
            ]
            ok = np.array([v is not None for v in vals]) & (t - r["t"] <= 15)
            game[k] = np.where(ok, [v or 0 for v in vals], 0)
            gm[k] = ok
    damage = np.array(
        [
            [
                r["t"],
                r["actor"],
                r["attacker"],
                max(r["damage"] or 0, 0),
                r["attacker_x"] if r["attacker_x"] is not None else np.nan,
                r["attacker_y"] if r["attacker_y"] is not None else np.nan,
            ]
            for r in tables["damage"]
        ],
        np.float32,
    ).reshape(-1, 6)
    return dict(
        times=times,
        x=x,
        mask=mask,
        valid=valid,
        source_time=source,
        target=target,
        target_valid=target_valid,
        target_time=target_time,
        target_vehicle=target_vehicle,
        items=items,
        item_seen=item_seen,
        vehicle=vehicle,
        team=np.array([a["team"] for a in raw["actors"]], np.int16),
        kind=np.array([a["kind"] for a in raw["actors"]], np.int8),
        game=game,
        game_mask=gm,
        game_time=gt,
        damage=damage,
        quality=np.array([overflows, ambiguous, postdeath], np.int64),
    )
