"""Synthetic ``MatchSource`` factory for the phase-tier v2 tests (no database).

Timeline on W (seconds)::

    LogPhaseChange (isGame): 0.1 @90, 1.0 @330, 1.5 @600, 2.0 @720, 2.5 @810, 3.0 @930, 3.5 @990, 4.0 @1050
        => phase 1 = [90, 600), phase 2 = [600, 810), phase 3 = [810, 990), phase 4 (FINAL) from 990
        b_1 = 600, b_2 = 810, b_3 = 990  ->  a_1 = 590, a_2 = 800, a_3 = 980;  C_m (LogMatchEnd) = 1100

Teams (2 players each): 11 (a1 killed 400 and redeployed (healthy again from 502), a2 killed 900, a1 killed 905 -> T = 905),
12 (b1 killed 650, b2 killed 700 -> T = 700; b1 knocked 640-649 = health 0 without kill), 13 (d1 killed 560, d2 killed 570:
causally DEAD at a_1 = 590, d1 redeployed from 652, killed again 800 -> T = 800: the comeback / risk-set-but-dead row),
14 (e1 knocked from 570 -- health 0, no kill -- until killed at 1085, e2 killed 1080 -> T = 1085, in the terminal window;
between 570 and 1085 the team is alive although e1 is down: knocked is not eliminated), 15 (winner: c1, c2 alive at the end).
"""

from __future__ import annotations

import numpy as np

from model_stgcn.phase_v2.source import (
    DAMAGE_DTYPES, GAME_STATE_DTYPES, GROGGY_DTYPES, ITEM_DTYPES, KILL_DTYPES, MatchSource, POS_DTYPES,
    PHASE_EVENT_DTYPES,
)

PHASE_EVENTS = [(90.0, 0.1), (330.0, 1.0), (600.0, 1.5), (720.0, 2.0), (810.0, 2.5), (930.0, 3.0), (990.0, 3.5), (1050.0, 4.0)]
TEAMS = {11: ["a1", "a2"], 12: ["b1", "b2"], 13: ["d1", "d2"], 14: ["e1", "e2"], 15: ["c1", "c2"]}
SCHEDULE = {1: 510.0, 2: 210.0, 3: 180.0, 4: 180.0, 5: 160.0, 6: 120.0, 7: 100.0, 8: 120.0}
END_W = 1100.0
KILLS = {"a1": [400.0, 905.0], "a2": [900.0], "b1": [650.0], "b2": [700.0], "d1": [560.0, 800.0], "d2": [570.0], "e1": [1085.0], "e2": [1080.0]}
REVIVE = {"a1": 502.0, "d1": 652.0}  # first healthy sample after the first kill
KNOCKED = {"b1": (640.0, 649.0), "e1": (570.0, 1e9), "c2": (610.0, 640.0)}  # health 0 without a kill


def _table(dtypes, rows: dict) -> dict:
    return {k: np.asarray(rows.get(k, []), dtype=v) for k, v in dtypes.items()}


def make_source(*, end_w: float | None = END_W, extra_pos=None, extra_dmg=None, extra_kills=None, extra_items=None,
                length_s: float | None = None, schedule=SCHEDULE, phase_events=PHASE_EVENTS) -> MatchSource:
    """Build the synthetic match.  ``length_s`` stretches the final phase (long-match test)."""
    accounts = [a for ps in TEAMS.values() for a in ps]
    aidx = {a: i for i, a in enumerate(accounts)}
    team_of = {a: t for t, ps in TEAMS.items() for a in ps}
    last = float(length_s) if length_s is not None else END_W
    # ---- positions: one sample per player every 10 s, +- jitter, a different phase offset per player
    pos = {k: [] for k in POS_DTYPES}
    for a in accounts:
        off = 0.3 * (aidx[a] + 1)
        t = 100.0 + off
        kills = KILLS.get(a, [])
        while t < last:
            health = 100.0
            dead = False
            for i, kt in enumerate(kills):
                rv = REVIVE.get(a) if i == 0 else None
                if t >= kt and (rv is None or t < rv):
                    dead = True
                    if t > kt + 60.0:  # a dead body reports health 0 for ~60 s, then nothing
                        dead = None
            if dead is None:
                t += 10.0
                continue
            if dead:
                health = 0.0
            ko = KNOCKED.get(a)
            if ko and ko[0] <= t <= ko[1]:
                health = 0.0
            ix = aidx[a]
            base_x = 1000.0 + 150.0 * (ix % 5) + 0.2 * t
            pos["w"].append(t); pos["g"].append(int(t)); pos["acct"].append(ix); pos["team"].append(team_of[a])
            pos["health"].append(health); pos["x"].append(base_x); pos["y"].append(1000.0 + 100.0 * ix + 0.1 * t)
            pos["z"].append(20.0 + ix); pos["veh"].append(False); pos["vspeed"].append(np.nan)
            t += 10.0
    if extra_pos:
        for row in extra_pos:
            for k, v in row.items():
                pos[k].append(v)
    order = np.argsort(np.asarray(pos["w"]), kind="stable")
    pos_t = {k: np.asarray(v, dtype=POS_DTYPES[k])[order] for k, v in pos.items()}
    # ---- game states every 10 s: blue circle shrinks inside each phase, white circle changes per phase
    gs = {k: [] for k in GAME_STATE_DTYPES}
    t = 100.0
    while t < last:
        ph = 1 + sum(1 for (w, g) in phase_events if w <= t and g >= 1.5 and int(np.ceil(g)) + 0 > 1) if False else None
        radius = max(3000.0 - 2.0 * (t - 100.0), 400.0)
        gs["w"].append(t); gs["n_teams"].append(5.0); gs["n_players"].append(10.0)
        gs["safe_x"].append(2000.0); gs["safe_y"].append(2000.0); gs["safe_r"].append(radius)
        gs["poison_x"].append(2000.0); gs["poison_y"].append(2000.0); gs["poison_r"].append(radius * 0.7)
        t += 10.0
    gs_t = {k: np.asarray(v, dtype=GAME_STATE_DTYPES[k]) for k, v in gs.items()}
    # ---- kills
    kr = {k: [] for k in KILL_DTYPES}
    allk = [(w, a) for a, ws in KILLS.items() for w in ws]
    if extra_kills:
        allk += list(extra_kills)
    for w, a in sorted(allk):
        killer = {"a1": "b1", "a2": "b2", "b1": "c1", "b2": "c1", "d1": "c2", "d2": "c1", "e1": "c1", "e2": "c1"}.get(a, "c1")
        kr["w"].append(w); kr["victim"].append(aidx[a]); kr["killer"].append(aidx[killer]); kr["victim_team"].append(team_of[a])
        for c in ("vx", "vy", "kx", "ky"):
            kr[c].append(np.nan)
        kr["rank"].append(0)
    kills_t = {k: np.asarray(v, dtype=KILL_DTYPES[k]) for k, v in kr.items()}
    # ---- damage: team 11 -> 12 @585 (before cut 1), @595 (AFTER cut 1), team 15 -> 14 @200
    dm = {k: [] for k in DAMAGE_DTYPES}
    dmg_rows = [(200.0, "c1", "e1", 40.0), (585.0, "a1", "b1", 30.0), (595.0, "a2", "b2", 70.0), (850.0, "c2", "a2", 20.0)]
    dmg_rows += list(extra_dmg or [])
    for w, att, vic, d in sorted(dmg_rows):
        dm["w"].append(w); dm["attacker"].append(aidx[att]); dm["victim"].append(aidx[vic]); dm["damage"].append(d)
        dm["ax"].append(1000.0); dm["ay"].append(1000.0); dm["vx"].append(1050.0); dm["vy"].append(1050.0)
    dmg_t = {k: np.asarray(v, dtype=DAMAGE_DTYPES[k]) for k, v in dm.items()}
    gr = {k: [] for k in GROGGY_DTYPES}
    for w, att, vic in [(570.0, "c1", "e1"), (640.0, "c1", "b1")]:
        gr["w"].append(w); gr["attacker"].append(aidx[att]); gr["victim"].append(aidx[vic])
        for c in ("ax", "ay", "vx", "vy"):
            gr[c].append(1000.0)
    groggy_t = {k: np.asarray(v, dtype=GROGGY_DTYPES[k]) for k, v in gr.items()}
    # ---- items
    def items(rows):
        out = {k: [] for k in ITEM_DTYPES}
        for w, team, item, cat, sub in rows:
            out["w"].append(w); out["team"].append(team); out["item"].append(item); out["cat"].append(cat)
            out["sub"].append(sub); out["stack"].append(1); out["carepackage"].append(False); out["lootbox"].append(False)
        return {k: np.asarray(v, dtype=ITEM_DTYPES[k]) for k, v in out.items()}
    pick = [(120.0, 11, "Item_Weapon_AK47_C", "Weapon", "Main"), (130.0, 12, "Item_Ammo_556mm_C", "Ammunition", "None"),
            (589.0, 11, "Item_Armor_C_01_Lv3_C", "Equipment", "Vest"), (592.0, 11, "Item_Armor_D_01_Lv2_C", "Equipment", "Vest")]
    pick += list(extra_items or [])
    use = [(300.0, 15, "Item_Heal_FirstAid_C", "Use", "Heal"), (591.0, 15, "Item_Boost_PainKiller_C", "Use", "Boost")]
    src = MatchSource(
        match_id="synthetic", map_name="Baltic_Main", game_mode="squad",
        team_ids=np.asarray(sorted(TEAMS), dtype=np.int64),
        team_num_players=np.full(len(TEAMS), 2.0), team_final_rank=np.asarray([2.0, 4.0, 3.0, 5.0, 1.0]),
        team_won=np.asarray([False, False, False, False, True]),
        roster=[(t, a, None, None) for t, ps in TEAMS.items() for a in ps],
        match_type="official", schedule=dict(schedule), schedule_source="synthetic",
        end_w=end_w if end_w is None else (float(end_w) if length_s is None else float(length_s)),
        phase_events={"w": np.asarray([w for w, _ in phase_events], dtype=np.float64),
                      "is_game": np.asarray([g for _, g in phase_events], dtype=np.float64)},
        gs=gs_t, pos=pos_t, kills=kills_t, damage=dmg_t, groggy=groggy_t,
        items={"pickup": items(pick), "equip": items([]), "use": items(use)},
        accounts=accounts,
    )
    return src
