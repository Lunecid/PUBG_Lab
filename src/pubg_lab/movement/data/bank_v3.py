"""As-of multi-life and mobility evidence. No retrospective revival backfilling.
Mode 1 is nonvehicle after landing, NOT certified ground/foot/swim.
"""

from collections import Counter, defaultdict
import numpy as np
from pubg_lab.movement.data.bank import build_bank as legacy_bank

MODE_NAMES = [
    "unknown",
    "nonvehicle_after_landing",
    "land_vehicle",
    "water_vehicle",
    "emergency_pickup",
    "redeploy_aircraft",
    "initial_aircraft",
    "airborne_after_exit",
    "motor_glider",
    "stationary_mount",
]


def mode_of(r):
    text = (str(r.get("vehicle_type") or "") + " " + str(r.get("vehicle_class") or "")).lower()
    if "motorglider" in text:
        return 8
    if "mortar" in text:
        return 9
    if "emergencypickup" in text:
        return 4
    if "redeployaircraft" in text:
        return 5
    if "transportaircraft" in text:
        return 6
    if "boat" in text or "aquarail" in text or "jet_ski" in text:
        return 3
    if r.get("vehicle_type") or r.get("vehicle_class"):
        return 2
    return 0


def actor_states(tables, p):
    events = []
    for name, priority in [("positions", 1), ("parachute_landing", 0), ("kills", 2)]:
        events.extend((r["t"], priority, name, r) for r in tables[name] if r["actor"] == p)
    events.sort(key=lambda e: (e[0], e[1]))
    alive = True
    episode = 0
    start = -1e9
    death = -1e9
    landed = False
    air = False
    confidence = 2
    candidate = None
    last = None
    rows = []
    transitions = []
    stats = Counter()
    for t, _, name, r in events:
        if name == "kills":
            alive = False
            death = t
            landed = False
            air = False
            candidate = None
            last = None
            transitions.append((t, False, episode, start, 0))
            continue
        if name == "parachute_landing":
            landed = True
            air = False
            last = None
            continue
        xyz = np.array(
            [r.get(k, np.nan) if r.get(k) is not None else np.nan for k in ("x", "y", "z")]
        )
        sane = np.isfinite(xyz).all() and (xyz[:2] >= 0).all() and (xyz[:2] < 8160).all()
        m = mode_of(r)
        hp = r.get("hp")
        positive = hp is not None and hp > 0
        if not alive and sane and positive and t > death + 1:
            direct = m == 5
            inferred = (landed and t > death + 10) or (
                candidate is not None
                and 5 <= t - candidate["t"] <= 15
                and t > death + 20
                and np.linalg.norm(xyz[:2] - np.array([candidate["x"], candidate["y"]])) > 5
            )
            if direct or inferred:
                alive = True
                episode += 1
                start = t
                confidence = 2 if direct else 1
                last = None
                candidate = None
                air = direct or not landed
                transitions.append((t, True, episode, start, confidence))
                stats["direct_aircraft_reentries" if direct else "inferred_reentries"] += 1
            else:
                candidate = r
        elif not alive:
            candidate = None
        if m in (2, 3, 9):
            air = False
            landed = True
        if m in (4, 5, 6, 8):
            air = True
            landed = False
        if m == 0:
            m = 7 if air else 1 if landed else 0
        ok = bool(alive and sane and hp is not None and hp >= 0)
        vel = np.zeros(3)
        vm = False
        if (
            ok
            and last is not None
            and last["episode"] == episode
            and last["mode"] == m
            and 5 <= t - last["t"] <= 15
        ):
            vel = (xyz - last["xyz"]) / (t - last["t"])
            limit = 200 if m in (5, 6) else 120 if m in (2, 3, 4, 7, 8) else 30
            vm = bool(np.linalg.norm(vel[:2]) <= limit)
            if not vm:
                stats["velocity_outlier"] += 1
        row = {
            **r,
            "xyz": xyz,
            "episode": episode,
            "episode_start": start,
            "mode": m,
            "confidence": confidence,
            "valid": ok,
            "landed": landed,
            "velocity": vel if vm else np.zeros(3),
            "velocity_valid": vm,
        }
        rows.append(row)
        last = row if ok else None
        stats["valid_post_reentry_positions"] += bool(ok and episode > 0)
        stats["mode_" + MODE_NAMES[m]] += ok
    return rows, transitions, stats


def build_bank(raw, vocab):
    # Legacy helper retains shared combat/game logic. All lifecycle-dependent fields
    # are overwritten, including equipment reset below; old artifact stays intact.
    b = legacy_bank(raw, vocab)
    T, P = b["valid"].shape
    times = b["times"]
    stats = Counter()
    allrows = []
    b.update(
        episode=np.full((T, P), -1, np.int16),
        episode_start=np.full((T, P), -1e9, np.float32),
        mobility=np.zeros((T, P), np.int8),
        mobility_known=np.zeros((T, P), bool),
        reentry_confidence=np.zeros((T, P), np.int8),
        z=np.zeros((T, P), np.float32),
        vz=np.zeros((T, P), np.float32),
        target_episode=np.full((T, P), -1, np.int16),
        target_mode=np.zeros((T, P), np.int8),
        target_interpolated=np.zeros((T, P), bool),
    )
    for p in range(P):
        rows, transitions, st = actor_states(raw["tables"], p)
        stats.update(st)
        allrows.append(rows)
        if not rows:
            continue
        ts = np.array([r["t"] for r in rows])
        duplicates = Counter(ts)
        idx = np.searchsorted(ts, times, side="right") - 1
        life = 0
        life_start = -1e9
        live = True
        conf = 2
        cursor = 0
        for k, t in enumerate(times):
            while cursor < len(transitions) and transitions[cursor][0] <= t:
                _, live, life, life_start, conf = transitions[cursor]
                cursor += 1
            b["episode"][k, p] = life
            b["episode_start"][k, p] = life_start
            b["reentry_confidence"][k, p] = conf
            ix = idx[k]
            r = rows[max(ix, 0)]
            age = t - r["t"]
            fresh = ix >= 0 and 0 <= age <= 15
            ok = bool(
                fresh and live and r["valid"] and r["episode"] == life and duplicates[r["t"]] == 1
            )
            b["valid"][k, p] = ok
            b["source_time"][k, p] = r["t"] if ix >= 0 else -1
            b["x"][k, p, :4] = 0
            b["mask"][k, p, :4] = False
            b["x"][k, p, 4] = r["hp"] if ok else 0
            b["mask"][k, p, 4] = ok
            b["x"][k, p, 5] = min(age, 300) if ix >= 0 else 0
            b["mask"][k, p, 5] = ix >= 0
            b["x"][k, p, 6] = float(live)
            b["mask"][k, p, 6] = bool(not live or ok)
            b["x"][k, p, 14] = bool(ok and r["vehicle_class"] or ok and r["vehicle_type"])
            b["mask"][k, p, 14] = ok
            b["vehicle"][k, p] = (
                vocab["vehicles"].get(r["vehicle_class"], 1) if ok and r["vehicle_class"] else 0
            )
            b["x"][k, p, 21] = bool(ok and r["landed"])
            b["mask"][k, p, 21] = ok
            if ok:
                b["x"][k, p, :2] = r["xyz"][:2]
                b["mask"][k, p, :2] = True
                b["x"][k, p, 2:4] = r["velocity"][:2]
                b["mask"][k, p, 2:4] = r["velocity_valid"]
                b["mobility"][k, p] = r["mode"]
                b["mobility_known"][k, p] = r["mode"] != 0
                b["z"][k, p] = r["xyz"][2]
                b["vz"][k, p] = r["velocity"][2]
            # Label-only interpolation never crosses an episode, mode transition or death.
            right = int(np.searchsorted(ts, t))
            left = right - 1
            candidates = []
            b["target_valid"][k, p] = False
            b["target"][k, p] = 0
            b["target_time"][k, p] = -1
            b["target_vehicle"][k, p] = 0
            if 0 <= left and right < len(rows):
                a, d = rows[left], rows[right]
                gap = d["t"] - a["t"]
                between = any(a["t"] < et <= d["t"] for et, *_ in transitions)
                if (
                    0 < gap <= 15
                    and a["valid"]
                    and d["valid"]
                    and a["episode"] == d["episode"]
                    and a["mode"] == d["mode"]
                    and not between
                    and duplicates[a["t"]] == duplicates[d["t"]] == 1
                ):
                    w = (t - a["t"]) / gap
                    xy = (1 - w) * a["xyz"][:2] + w * d["xyz"][:2]
                    candidates = [(xy, t, a["episode"], a["mode"], True)]
            if not candidates:
                near = min(
                    (j for j in (left, right) if 0 <= j < len(rows)),
                    key=lambda j: abs(ts[j] - t),
                    default=None,
                )
                if near is not None:
                    a = rows[near]
                    if (
                        abs(a["t"] - t) <= 5
                        and a["valid"]
                        and duplicates[a["t"]] == 1
                        and not any(min(a["t"], t) < et <= max(a["t"], t) for et, *_ in transitions)
                    ):
                        candidates = [(a["xyz"][:2], a["t"], a["episode"], a["mode"], False)]
            if candidates and live:
                xy, at, ep, mode, interp = candidates[0]
                if ep == life:
                    b["target"][k, p] = xy
                    b["target_time"][k, p] = at
                    b["target_valid"][k, p] = True
                    b["target_episode"][k, p] = ep
                    b["target_mode"][k, p] = mode
                    b["target_vehicle"][k, p] = mode in (2, 3, 4, 5, 6, 8, 9)
                    b["target_interpolated"][k, p] = interp
        # Rebuild gear with kill resets; never carry dead-life equipment into recall.
        equip = defaultdict(list)
        for r in raw["tables"]["item_equip"]:
            if r["actor"] == p and (
                r["category"] == "Weapon" or r["subcategory"] in ("Headgear", "Vest", "Throwable")
            ):
                equip[r["t"], r["item"]].append(r)
        ev = [(t, 1, item, rs) for (t, item), rs in equip.items()] + [
            (r["t"], 0, None, None) for r in raw["tables"]["kills"] if r["actor"] == p
        ]
        ev.sort(key=lambda e: (e[0], e[1]))
        cursor = 0
        state = {}
        last = None
        for k, t in enumerate(times):
            while cursor < len(ev) and ev[cursor][0] <= t:
                et, typ, item, rs = ev[cursor]
                cursor += 1
                if typ == 0:
                    state = {}
                    last = None
                    continue
                acts = {r["action"] for r in rs}
                last = et
                if len(acts) > 1:
                    state[item] = None
                elif "equip" in acts:
                    state[item] = vocab["items"].get(item, 1)
                else:
                    state.pop(item, None)
            codes = sorted(v for v in state.values() if v is not None)
            b["items"][k, p] = 0
            b["items"][k, p, : min(12, len(codes))] = codes[:12]
            b["item_seen"][k, p] = last is not None
            b["x"][k, p, 15] = min(t - last, 300) if last is not None else 0
            b["mask"][k, p, 15] = last is not None
            b["x"][k, p, 17] = sum(v is None for v in state.values())
            b["mask"][k, p, 17] = last is not None
            if b["episode"][k, p] > 0 and t - b["episode_start"][k, p] < 30:
                b["x"][k, p, 7] = 0
                b["mask"][k, p, 7] = False
    b["quality_v3"] = np.array(
        [
            stats["direct_aircraft_reentries"],
            stats["inferred_reentries"],
            stats["valid_post_reentry_positions"],
        ],
        np.int64,
    )
    return b, dict(stats), allrows
