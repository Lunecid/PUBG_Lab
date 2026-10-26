"""Causal, variable-roster preprocessing. Coordinates are metres, time is seconds.

Design: docs/experiment_design/PUBG_WORLD_MODEL_FULL_CORPUS_V2_2026-10-20.md.
No estimator is fitted here. Observer access is applied before graph aggregation.
Ref: Hu et al. 2020, Heterogeneous Graph Transformer, arXiv:2003.01332:
typed player/team relations motivate the interface, not a reproduced architecture.
"""
from collections import Counter, defaultdict
import math

import numpy as np

FEATURES = [
    "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "health",
    "position_age_s", "alive_evidence", "vehicle_recorded",
    "received_damage_10s", "received_damage_30s", "outgoing_damage_30s",
    "pickup_events_30s", "use_events_30s", "equipment_age_s",
    "equipment_ambiguity", "episode_age_s",
]
GAME_FEATURES = ["phase", "alive_players", "alive_teams", "safe_x", "safe_y",
                 "safe_r", "poison_x", "poison_y", "poison_r"]
MODES = ["unknown", "nonvehicle_after_landing", "land_vehicle", "water_vehicle",
         "emergency_pickup", "redeploy_aircraft", "initial_aircraft",
         "airborne_after_exit_inferred", "motor_glider", "stationary_mount"]
PUBLIC_TEAM_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]


def finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def mode_of(row):
    text = (str(row.get("vehicle_type") or "") + " " +
            str(row.get("vehicle_class") or "")).lower()
    for needle, mode in [("motorglider", 8), ("mortar", 9),
                         ("emergencypickup", 4), ("redeployaircraft", 5),
                         ("transportaircraft", 6), ("boat", 3),
                         ("aquarail", 3), ("jet_ski", 3)]:
        if needle in text:
            return mode
    return 2 if text.strip() else 0


def actor_timeline(tables, actor):
    events = []
    for name, priority in [("landing", 0), ("positions", 1), ("kills", 2)]:
        events.extend((r["t"], priority, r.get("row_id", 0), name, r)
                      for r in tables.get(name, []) if r["actor"] == actor)
    events.sort(key=lambda e: e[:3])
    live, landed, air = True, False, False
    episode, segment, confidence = 0, 0, 2
    episode_start, death = 0., -math.inf
    candidate = last = None
    rows, transitions, issues = [], [], []
    seen_positions = defaultdict(list)
    for _, _, _, name, r in events:
        if name == "positions":
            seen_positions[r["t"]].append(r)
    duplicate_conflicts = set()
    for t, rs in seen_positions.items():
        signatures = {tuple(r.get(k) for k in ("x", "y", "z", "hp",
                                               "vehicle_type", "vehicle_class"))
                      for r in rs}
        if len(signatures) > 1:
            duplicate_conflicts.add(t)
    previous_signature = None
    for t, _, _, name, r in events:
        if name == "kills":
            if live:
                segment += 1
            live, landed, air = False, False, False
            death, last, candidate = t, None, None
            transitions.append((t, live, episode, confidence, episode_start))
            continue
        if name == "landing":
            landed, air = True, False
            continue
        signature = tuple(r.get(k) for k in ("t", "x", "y", "z", "hp",
                                             "vehicle_type", "vehicle_class"))
        if signature == previous_signature:
            continue
        previous_signature = signature
        xyz = np.array([r[k] if finite(r.get(k)) else np.nan
                        for k in ("x", "y", "z")], dtype=np.float64)
        sane = bool(np.isfinite(xyz).all() and not np.all(xyz == 0)
                    and t not in duplicate_conflicts)
        hp = r.get("hp")
        healthy = finite(hp) and 0 <= hp <= 100
        m = mode_of(r)
        if not live and sane and healthy and hp > 0 and t > death + 1:
            direct = m == 5
            inferred = (candidate is not None and 5 <= t-candidate["t"] <= 15
                        and t > death+20
                        and np.linalg.norm(xyz[:2]-candidate["xyz"][:2]) > 5)
            if direct or inferred:
                live, episode, episode_start = True, episode+1, t
                confidence, segment, last = (2 if direct else 1), segment+1, None
                air = direct or not landed
                transitions.append((t, live, episode, confidence, episode_start))
                issues.append((actor, t, t, "reentry_direct" if direct else
                               "reentry_inferred", "evidence" if direct else "inference"))
            candidate = {"t": t, "xyz": xyz}
        elif not live:
            candidate = None
        if m in (2, 3, 9):
            landed, air = True, False
        if m in (4, 5, 6, 8):
            landed, air = False, True
        if m == 0:
            m = 7 if air else 1 if landed else 0
        ok = bool(live and sane and healthy)
        velocity = np.zeros(2)
        velocity_known = False
        if t in duplicate_conflicts:
            issues.append((actor, t, t, "conflicting_position_timestamp", "mask"))
        if not sane or not healthy:
            issues.append((actor, t, t, "invalid_position_or_health", "mask"))
        if sane and (np.any(xyz[:2] < 0) or np.any(xyz[:2] > 15000)):
            issues.append((actor, t, t, "broad_extent_candidate", "review_not_clip"))
        if last is not None and ok:
            dt = t-last["t"]
            if dt > 15:
                segment += 1
                issues.append((actor, last["t"], t, "track_gap", "mask_bridge"))
            elif dt > 0 and last["mode"] == m and last["episode"] == episode:
                velocity = (xyz[:2]-last["xyz"][:2])/dt
                limit = 200 if m in (5, 6) else 120 if m in (2, 3, 4, 7, 8) else 30
                velocity_known = bool(np.linalg.norm(velocity) <= limit)
                if not velocity_known:
                    segment += 1
                    issues.append((actor, last["t"], t, "speed_candidate", "mask_bridge"))
        rr = {**r, "xyz": xyz, "valid": ok, "episode": episode,
              "segment": segment, "confidence": confidence, "mode": m,
              "episode_start": episode_start,
              "velocity": velocity if velocity_known else np.zeros(2),
              "velocity_known": velocity_known}
        rows.append(rr)
        if not ok:
            segment += 1
        last = rr if ok else None
    return rows, transitions, issues


def rolling(rows, times, window, value):
    if not rows:
        return np.zeros(len(times), np.float32)
    rs = sorted(rows, key=lambda r: r["t"])
    ts = np.array([r["t"] for r in rs])
    values = np.r_[0., np.cumsum([value(r) for r in rs])]
    return (values[np.searchsorted(ts, times, side="right")] -
            values[np.searchsorted(ts, times-window, side="right")]).astype(np.float32)


def outcomes(raw):
    """Conservative team targets. Conflicting ranks are masked, never overwritten.

    Last logged deaths alone do not prove terminal elimination. A team elimination
    time is retained only as a candidate, with terminal_time_mask=False.
    """
    teams = raw["teams"]
    N = len(teams)
    ranks = [r.get("rank") for r in teams]
    coherent = (raw["meta"].get("br_rule_candidate") and raw["meta"].get("end_time")
                and N > 1 and all(isinstance(r, int) and 1 <= r <= N for r in ranks)
                and len(set(ranks)) == N
                and sum(r.get("won") is True for r in teams) == 1
                and all(r.get("won") == (r.get("rank") == 1) for r in teams))
    result = []
    for team in teams:
        members = [p for p in raw["actors"] if p["team"] == team["team"]]
        personal = [p.get("rank") for p in members]
        conflict = any(r is not None and r != team.get("rank") for r in personal)
        complete_personal = bool(members) and all(r is not None for r in personal)
        deaths = [r for r in raw["tables"]["kills"]
                  if r["actor"] in {p["actor"] for p in members}]
        death_ranks = [r.get("rank") for r in deaths if r.get("rank") is not None]
        death_conflict = any(r != team.get("rank") for r in death_ranks)
        rank_mask = bool(coherent and complete_personal and not conflict and not death_conflict)
        winner_mask = bool(coherent and complete_personal and
                           all((r == 1) == bool(team["won"]) for r in personal))
        result.append({**team, "rank_mask": rank_mask, "winner_mask": winner_mask,
                       "rank_utility": (N-team["rank"])/(N-1) if rank_mask else None,
                       "participant_rank_conflict": conflict,
                       "kill_rank_conflict": death_conflict,
                       "terminal_time_candidate": max([r["t"] for r in deaths], default=None),
                       "terminal_time_mask": False,
                       "reason": "cross_field_consistent_candidate" if rank_mask else
                                 "missing_or_conflicting_evidence"})
    return result


def build_bank(raw):
    """Build server evidence arrays; graph views must pass through view_at()."""
    end_s = raw["meta"].get("end_s")
    after_end = {}
    if finite(end_s):
        after_end = {name: [r for r in rs if r["t"] > end_s]
                     for name, rs in raw["tables"].items()}
        raw = {**raw, "tables": {name: [r for r in rs if r["t"] <= end_s]
                                for name, rs in raw["tables"].items()}}
    tables = raw["tables"]
    P = len(raw["actors"])
    max_t = max([r["t"] for r in tables["positions"] if finite(r.get("t"))]+[0.])
    times = np.arange(0, int(max(max_t, 0)//10)*10+1, 10, dtype=np.float64)
    T = len(times)
    x = np.zeros((T, P, len(FEATURES)), np.float32)
    mask = np.zeros_like(x, bool)
    bank = {"times": times, "x": x, "mask": mask,
            "valid": np.zeros((T, P), bool), "source_time": np.full((T, P), -1., np.float64),
            "known": np.zeros((T, P), bool),
            "episode": np.zeros((T, P), np.int16),
            "segment": np.zeros((T, P), np.int32),
            "mobility": np.zeros((T, P), np.int8),
            "confidence": np.zeros((T, P), np.int8),
            "target": np.zeros((T, P, 2), np.float32),
            "target_valid": np.zeros((T, P), bool),
            "target_episode": np.full((T, P), -1, np.int16),
            "target_segment": np.full((T, P), -1, np.int32),
            "target_mode": np.zeros((T, P), np.int8),
            "target_source_time": np.full((T, P), -1., np.float64),
            "target_interpolated": np.zeros((T, P), bool),
            "team": np.array([a["team"] for a in raw["actors"]], np.int32),
            "kind_eval_only": np.array([a["kind"] for a in raw["actors"]], np.int8)}
    # Item state is sparse and unbounded, avoiding a fixed 12-item truncation.
    item_vocab = list(dict.fromkeys(r["item"] for r in
                      sorted(tables["equip"], key=lambda r: (r["t"], r.get("row_id", 0)))
                      if r.get("item")))
    item_codes = {v: i for i, v in enumerate(item_vocab)}
    sparse_items, issues, spatial = [], [], Counter()
    for name, rs in after_end.items():
        if rs:
            issues.append((-1, min(r["t"] for r in rs), max(r["t"] for r in rs),
                           "post_match_end_"+name, "excluded_normal_or_unknown_context"))
    grouped = {name: defaultdict(list) for name in tables}
    outgoing_groups = defaultdict(list)
    for name, rs in tables.items():
        for r in rs:
            if "actor" in r:
                grouped[name][r["actor"]].append(r)
            if name == "damage" and r.get("attacker", -1) >= 0:
                outgoing_groups[r["attacker"]].append(r)
    for p in range(P):
        local = {name: groups[p] for name, groups in grouped.items()}
        rows, transitions, pi = actor_timeline(local, p)
        issues.extend(pi)
        if not rows:
            continue
        ts = np.array([r["t"] for r in rows])
        first = min(ts)
        bank["known"][:, p] = times >= first
        idx = np.searchsorted(ts, times, side="right")-1
        live, ep, conf, estart, cursor = True, 0, 2, 0., 0
        for k, t in enumerate(times):
            while cursor < len(transitions) and transitions[cursor][0] <= t:
                _, live, ep, conf, estart = transitions[cursor]
                cursor += 1
            bank["episode"][k, p] = ep
            bank["confidence"][k, p] = conf
            if idx[k] < 0:
                continue
            r = rows[idx[k]]
            age = t-r["t"]
            bank["source_time"][k, p] = r["t"]
            ok = bool(age <= 15 and live and r["valid"] and r["episode"] == ep)
            bank["valid"][k, p] = ok
            bank["segment"][k, p] = r["segment"]
            bank["mobility"][k, p] = r["mode"] if ok else 0
            x[k, p, 6] = age
            mask[k, p, 6] = True
            x[k, p, 7] = float(live)
            mask[k, p, 7] = bool(ok or not live)
            x[k, p, 16] = t-estart
            mask[k, p, 16] = bank["known"][k, p]
            if ok:
                x[k, p, :3] = r["xyz"]
                mask[k, p, :3] = True
                x[k, p, 3:5] = r["velocity"]
                mask[k, p, 3:5] = r["velocity_known"]
                x[k, p, 5] = r["hp"]
                mask[k, p, 5] = True
                x[k, p, 8] = bool(r.get("vehicle_class") or r.get("vehicle_type"))
                mask[k, p, 8] = True
            right = int(np.searchsorted(ts, t))
            left = right-1
            target = None
            if right < len(rows) and ts[right] == t and rows[right]["valid"]:
                target = (rows[right]["xyz"][:2], rows[right], t, False)
            elif left >= 0 and right < len(rows):
                a, b = rows[left], rows[right]
                dt = b["t"]-a["t"]
                if (0 < dt <= 15 and a["valid"] and b["valid"] and
                        a["episode"] == b["episode"] and a["segment"] == b["segment"]
                        and a["mode"] == b["mode"] and
                        not any(a["t"] < z[0] <= b["t"] for z in transitions)):
                    w = (t-a["t"])/dt
                    target = ((1-w)*a["xyz"][:2]+w*b["xyz"][:2], a, b["t"], True)
            if target is not None and live and target[1]["episode"] == ep:
                xy, r, source_t, interp = target
                bank["target"][k, p] = xy
                bank["target_valid"][k, p] = True
                bank["target_episode"][k, p] = r["episode"]
                bank["target_segment"][k, p] = r["segment"]
                bank["target_mode"][k, p] = r["mode"]
                bank["target_source_time"][k, p] = source_t
                bank["target_interpolated"][k, p] = interp
        ds = [r for r in local["damage"] if finite(r["damage"])
              and r["damage"] >= 0]
        outgoing = [r for r in outgoing_groups[p] if finite(r["damage"]) and r["damage"] >= 0]
        for col, rs, window, val in [
            (9, ds, 10, lambda r: r["damage"]), (10, ds, 30, lambda r: r["damage"]),
            (11, outgoing, 30, lambda r: r["damage"]),
            (12, local["pickup"], 30, lambda r: 1),
            (13, local["use"], 30, lambda r: 1),
        ]:
            x[:, p, col] = rolling(rs, times, window, val)
            mask[:, p, col] = bank["known"][:, p]  # logged-event counts, not total perception
        eqevents = defaultdict(list)
        for r in local["equip"]:
            if r["actor"] == p and r.get("item"):
                eqevents[r["t"], r["item"]].append(r)
        ev = [(t, 1, item, rs) for (t, item), rs in eqevents.items()]
        ev += [(r["t"], 0, "", []) for r in local["kills"]]
        ev.sort(key=lambda e: e[:3])
        state, last, j = {}, None, 0
        for k, t in enumerate(times):
            while j < len(ev) and ev[j][0] <= t:
                et, kind, item, rs = ev[j]
                j += 1
                if kind == 0:
                    state, last = {}, None
                    continue
                acts = {str(r["action"]).lower() for r in rs}
                last = et
                if len(acts) > 1:
                    state[item] = None
                elif next(iter(acts)) in ("equip", "logitemequip"):
                    state[item] = item_codes[item]
                elif next(iter(acts)) in ("unequip", "logitemunequip"):
                    state.pop(item, None)
                else:
                    state[item] = None
            if last is not None:
                x[k, p, 14:16] = [t-last, sum(v is None for v in state.values())]
                mask[k, p, 14:16] = True
            sparse_items.extend((k, p, v) for v in sorted(v for v in state.values() if v is not None))
        for a, b in zip(rows[:-1], rows[1:]):
            dt = b["t"]-a["t"]
            if (a["valid"] and b["valid"] and a["episode"] == b["episode"]
                    and a["segment"] == b["segment"] and a["mode"] == b["mode"]
                    and a["mode"] in (1, 2, 3) and 0 < dt <= 15):
                cell = lambda r: tuple(np.floor(r["xyz"][:2]/100).astype(int))
                src, dst = cell(a), cell(b)
                if src != dst:
                    spatial[(*src, *dst, a["mode"])] += 1
    bank["items_sparse"] = np.array(sparse_items, np.int32).reshape(-1, 3)
    game = np.zeros((T, len(GAME_FEATURES)), np.float32)
    gm = np.zeros_like(game, bool)
    gt = np.full(T, -1., np.float64)
    gs = sorted(tables["game"], key=lambda r: (r["t"], r.get("row_id", 0)))
    gts = np.array([r["t"] for r in gs])
    if len(gs):
        gi = np.searchsorted(gts, times, side="right")-1
        for k, t in enumerate(times):
            if gi[k] < 0:
                continue
            r = gs[gi[k]]
            gt[k] = r["t"]
            for f, name in enumerate(GAME_FEATURES):
                if finite(r.get(name)) and t-r["t"] <= 15:
                    game[k, f] = r[name]
                    gm[k, f] = True
    bank.update(game=game, game_mask=gm, game_time=gt)
    # P4 targets refer to logged future events, with match-end right censoring.
    # They are not loaded by view_at() and cannot become encoder inputs.
    horizon = np.array([10, 20, 30, 40, 50, 60])
    future_damage = np.zeros((T, P, 6), np.float32)
    future_death = np.zeros((T, P, 6), bool)
    event_mask = np.zeros((T, P, 6), bool)
    end_s = raw["meta"].get("end_s")
    for p in range(P):
        ds = sorted((r for r in grouped["damage"][p] if finite(r["damage"]) and r["damage"] >= 0),
                    key=lambda r: r["t"])
        dts = np.array([r["t"] for r in ds])
        values = np.r_[0., np.cumsum([r["damage"] for r in ds])]
        deaths = np.array(sorted({r["t"] for r in grouped["kills"][p]}))
        for h, seconds in enumerate(horizon):
            future_damage[:, p, h] = (values[np.searchsorted(dts, times+seconds, side="right")] -
                                     values[np.searchsorted(dts, times, side="right")])
            future_death[:, p, h] = (np.searchsorted(deaths, times+seconds, side="right") >
                                    np.searchsorted(deaths, times, side="right"))
            event_mask[:, p, h] = bank["valid"][:, p] & (
                (times+seconds <= end_s) if finite(end_s) else False)
    bank.update(future_damage=future_damage, future_death=future_death,
                future_event_mask=event_mask)
    # Anchors selected exclusively from as-of evidence, not future-label coverage.
    anchors = [(k, int(p)) for k in range(12, T, 3)
               for p in np.flatnonzero(bank["valid"][k])]
    bank["anchors"] = np.array(anchors, np.int32).reshape(-1, 2)
    bank["spatial_edges"] = np.array([[*k, v] for k, v in sorted(spatial.items())],
                                    np.int64).reshape(-1, 6)
    return bank, {"item_vocabulary_local": item_vocab, "issues": issues,
                  "post_end_rows_excluded": {k:len(v) for k,v in after_end.items()},
                  "outcomes": outcomes(raw), "counts": {
                      "actors": P, "frames": T, "anchors": len(anchors),
                      "valid_states": int(bank["valid"].sum()),
                      "target_states": int(bank["target_valid"].sum()),
                      "issues": len(issues)}}


def view_at(bank, k, ego, view="observer"):
    """Compact causal graph. No kind/rank/future mask is exposed to the encoder.

    Team HUD is a telemetry proxy: teammate kinematics/health only; equipment and
    combat history are private to ego. Damage attacker coordinates are not exposed.
    """
    if view not in ("observer", "server"):
        raise ValueError(view)
    lo = max(0, k-12)
    own = bank["team"] == bank["team"][ego]
    known = bank["known"][k]
    visible = known & (own if view == "observer" else True)
    players = np.flatnonzero(visible)
    if ego not in players:
        raise ValueError("Ego has no as-of evidence")
    x = bank["x"][lo:k+1, players].copy()
    mask = bank["mask"][lo:k+1, players].copy()
    if view == "observer":
        for j, p in enumerate(players):
            if p != ego:
                mask[:, j, :] = False
                mask[:, j, PUBLIC_TEAM_FEATURES] = bank["mask"][lo:k+1, p][:, PUBLIC_TEAM_FEATURES]
    x[~mask] = 0
    teams = np.unique(bank["team"][players])
    team_lookup = {int(t): i for i, t in enumerate(teams)}
    member_edges = np.array([[j, team_lookup[int(bank["team"][p])]]
                            for j, p in enumerate(players)], np.int32)
    # Typed team aggregation sees only already-masked player features.
    team_x = np.zeros((len(teams), 7), np.float32)
    for i, team in enumerate(teams):
        js = np.flatnonzero(bank["team"][players] == team)
        valid = mask[-1, js, 0] & mask[-1, js, 1]
        points = x[-1, js[valid], :2]
        team_x[i, 0:2] = [len(js), len(points)]
        if len(points):
            team_x[i, 2:4] = points.mean(0)
            team_x[i, 4:6] = points.std(0)
        if len(points) > 1:
            distances = np.linalg.norm(points[:, None]-points[None], axis=-1)
            team_x[i, 6] = distances[np.triu_indices(len(points), 1)].mean()
    sparse = bank["items_sparse"]
    item_rows = sparse[(sparse[:, 0] >= lo) & (sparse[:, 0] <= k)]
    allowed = {ego} if view == "observer" else set(players)
    item_rows = item_rows[np.array([p in allowed for p in item_rows[:, 1]], bool)]
    mobility = bank["mobility"][lo:k+1, players].copy()
    mobility_mask = bank["valid"][lo:k+1, players].copy() & (mobility != 0)
    if view == "observer":
        mobility_mask[:, players != ego] = False
    mobility[~mobility_mask] = 0
    vehicles = [(j, int(mobility[-1,j]), float(x[-1,j,6])) for j in range(len(players))
                if mobility_mask[-1,j] and mobility[-1,j] in (2,3,4,5,6,8,9)]
    return {"player_ids": players, "team_ids": teams, "x": x, "mask": mask,
            "member_edges": member_edges, "team_x": team_x,
            "items_sparse": item_rows, "game": bank["game"][lo:k+1].copy(),
            "game_mask": bank["game_mask"][lo:k+1].copy(),
            "mobility_ego": bank["mobility"][lo:k+1, ego].copy(),
            "mobility": mobility, "mobility_mask": mobility_mask,
            "vehicle_evidence": np.array(vehicles,np.float32).reshape(-1,3),
            "game_source_time": bank["game_time"][lo:k+1].copy(),
            "source_time": bank["source_time"][lo:k+1, players].copy()}


def targets_at(bank, k, ego):
    members = np.flatnonzero((bank["team"] == bank["team"][ego]) & bank["known"][k])
    xy = np.zeros((len(members), 6, 2), np.float32)
    mask = np.zeros((len(members), 6), bool)
    modes = np.zeros((len(members), 6), np.int8)
    for h in range(6):
        q = k+h+1
        if q >= len(bank["times"]):
            continue
        xy[:, h] = bank["target"][q, members]
        mask[:, h] = (bank["target_valid"][q, members]
                      & (bank["target_episode"][q, members] == bank["episode"][k, members])
                      & (bank["target_segment"][q, members] == bank["segment"][k, members])
                      & bank["valid"][k, members])
        modes[:, h] = bank["target_mode"][q, members]
    xy[~mask] = 0
    return {"members": members, "xy": xy, "mask": mask, "mode": modes,
            "logged_damage": bank["future_damage"][k, members],
            "logged_death": bank["future_death"][k, members],
            "event_mask": bank["future_event_mask"][k, members]}
