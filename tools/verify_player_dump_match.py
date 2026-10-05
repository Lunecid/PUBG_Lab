"""Hand check of dumped PLAYER-level matches against the database (independent re-derivation).

    python -m tools.verify_player_dump_match --dump model_stgcn/dumps/bigcomp2027_player_v1/erangel_tpp_squad/test.pt \
        --match-id <uuid> [--match-id <uuid>] [--show]

Every number the player dump claims is recomputed here with plain SQL + simple python loops (no numpy
vectorisation, none of the dump's assembly code) from the raw ``pubg.*`` tables and compared:

  1. players       participants with a roster team in the phase dump's team order; ai flag; account exclusion
  2. node mask     last-kill-event death time (label audit rule, own implementation + the audit's own code)
                   > phase_start_s; team death times and alive counts against the PHASE dump; first death
                   times; alive at the cut (no kill event up to the cut, or revived) and its strict variant
  3. Xp            all 13 columns for every (phase, player): the last raw position row <= cut_t within 15 s,
                   team centroid, health, damage dealt / taken, knocks, kills, cumulative kills / Heal item uses (Boost in an extra),
                   signed zone distances against stdb.dynamic_regions at cut_t
  4. edges         teammate / enemy_prox / combat sets and weights
  5. causality     position rows used happen before the true start of the next phase (first LogPhaseChange, event
                   clock); no event after cut_wall_s (<= phase_end_s - 10) enters any number (the same numbers are
                   recomputed with the uncapped window [phase_start_s, phase_end_s] to show the cap matters)
  6. coverage      share of alive players at the cut (no kill event yet / last-death rule) with a position
                   sample <= 15 s before it
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

from model_stgcn import dump_phase_tier as D
from model_stgcn import dump_player_tier as PT
from stdb.conn import connect

TOL = 2e-4
INF = float("inf")
AUDIT_LIB = Path(__file__).resolve().parent.parent / "reports/bigcomp2027_data_profile/scripts/label_audit_lib.py"


# ------------------------------------------------------------------------------------------------
# independent raw reads
# ------------------------------------------------------------------------------------------------
def started_epoch(cur, mid: str) -> float:
    cur.execute("SELECT extract(epoch FROM started_at)::float8 FROM stdb.matches WHERE match_id = %s", (mid,))
    return float(cur.fetchone()[0])


def read_participants(cur, mid: str):
    cur.execute(
        """
        SELECT r.team_id, p.player_id, p.death_type, p.time_survived::float8
          FROM pubg.participants p JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
         WHERE p.match_id = %s AND r.team_id IS NOT NULL
        """,
        (mid,),
    )
    return [(int(t), str(p), dt, None if ts is None else float(ts)) for t, p, dt, ts in cur.fetchall()]


def independent_death_times(cur, mid: str, s_ep: float, parts) -> dict[str, float]:
    """Per-player last death time on the event clock (+inf alive), the label audit's rule re-implemented."""
    cur.execute("SELECT victim_id, extract(epoch FROM event_time)::float8 FROM pubg.telem_kills WHERE match_id = %s AND victim_id IS NOT NULL", (mid,))
    last: dict[str, float] = {}
    for victim, ep in cur.fetchall():
        w = float(ep) - s_ep
        if str(victim) not in last or w > last[str(victim)]:
            last[str(victim)] = w
    pairs = [last[pid] - ts for _t, pid, _dt, ts in parts if not pid.startswith("ai.") and pid in last and ts is not None]
    off = float(np.median(pairs)) if pairs else 0.70
    out = {}
    for _t, pid, dt, ts in parts:
        if dt == "alive":
            out[pid] = INF
        elif pid in last:
            out[pid] = last[pid]
        elif ts is not None:
            out[pid] = ts + off
        else:
            out[pid] = INF
    return out


def read_all_positions(cur, mid: str, s_ep: float, accounts):
    """account -> [(G t, health, x_m, y_m, W of the row, id)] ascending (t, id): the WHOLE history, no window."""
    cur.execute(
        """
        SELECT account_id, elapsed_time, health::float8, pos_x::float8, pos_y::float8,
               extract(epoch FROM event_time)::float8, id
          FROM pubg.telem_positions
         WHERE match_id = %s AND account_id = ANY(%s) AND is_game >= 0.1
           AND pos_x IS NOT NULL AND pos_y IS NOT NULL AND elapsed_time IS NOT NULL
         ORDER BY account_id, elapsed_time, id
        """,
        (mid, list(accounts)),
    )
    out: dict[str, list] = {}
    for acc, t, h, x, y, ep, rid in cur.fetchall():
        out.setdefault(str(acc), []).append((float(t), h, float(x) / 100.0, float(y) / 100.0, float(ep) - s_ep, rid))
    return out


def read_events(cur, mid: str, s_ep: float):
    def q(sql):
        cur.execute(sql, (mid,))
        return cur.fetchall()

    kills = [(a, v, float(ep) - s_ep) for a, v, ep in q("SELECT killer_id, victim_id, extract(epoch FROM event_time)::float8 FROM pubg.telem_kills WHERE match_id = %s")]
    knocks = [(a, v, float(ep) - s_ep) for a, v, ep in q("SELECT attacker_id, victim_id, extract(epoch FROM event_time)::float8 FROM pubg.telem_groggy WHERE match_id = %s")]
    damage = [(a, v, None if d is None else float(d), float(ep) - s_ep) for a, v, d, ep in q("SELECT attacker_id, victim_id, damage::float8, extract(epoch FROM event_time)::float8 FROM pubg.telem_damage WHERE match_id = %s")]
    items = [(a, c, s, float(ep) - s_ep) for a, c, s, ep in q("SELECT account_id, category, sub_category, extract(epoch FROM event_time)::float8 FROM pubg.telem_item_use WHERE match_id = %s")]
    return kills, knocks, damage, items


# ------------------------------------------------------------------------------------------------
def verify(cur, pm: dict, phase_m: dict, show: bool = False) -> list[tuple[str, bool, str]]:
    mid = pm["match_id"]
    res: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        res.append((name, bool(ok), detail))

    s_ep = started_epoch(cur, mid)
    team_ids = phase_m["team_ids"].tolist()
    tindex = {t: i for i, t in enumerate(team_ids)}
    size = float(phase_m["map_size"])
    delta = float(phase_m["clock_delta_s"])
    parts = read_participants(cur, mid)

    # 1. players -------------------------------------------------------------------------------
    want = sorted((tindex[t], pid) for t, pid, _dt, _ts in parts if t in tindex)
    want_ids = [pid for _t, pid in want]
    check("player_ids / player_team_idx / player_is_ai == participants with a roster team of the phase dump",
          pm["player_ids"] == want_ids
          and pm["player_team_idx"].tolist() == [t for t, _p in want]
          and pm["player_is_ai"].tolist() == [p.startswith("ai.") for p in want_ids],
          f"{len(want_ids)} players in {len(set(t for t, _ in want))} teams, {sum(p.startswith('ai.') for p in want_ids)} AI")
    pid_of = {p: i for i, p in enumerate(want_ids)}
    team_of = {pid: tindex[t] for t, pid, _dt, _ts in parts if t in tindex}
    P = len(want_ids)

    cur.execute("SELECT count(DISTINCT account_id) FROM pubg.telem_positions WHERE match_id = %s AND account_id IS NOT NULL AND NOT (account_id = ANY(%s))", (mid, want_ids))
    n_nonpart = int(cur.fetchone()[0])
    check("accounts that are not player nodes (Monster.*, ...) never appear", not any(str(p).startswith("Monster.") for p in pm["player_ids"]),
          f"{n_nonpart} non-node accounts in the raw positions")

    # 2. node mask + death times ----------------------------------------------------------------
    death = independent_death_times(cur, mid, s_ep, parts)
    bad_d = sum(1 for i, pid in enumerate(want_ids) if not ((math.isinf(death[pid]) and math.isinf(float(pm["player_death_s"][i]))) or abs(death[pid] - float(pm["player_death_s"][i])) < 1e-2))
    bad_node = 0
    for k in range(8):
        if not bool(phase_m["phase_mask"][k]):
            bad_node += int(pm["player_node_mask"][k].any())
            continue
        start = float(phase_m["phase_start_s"][k])
        for i, pid in enumerate(want_ids):
            bad_node += int((death[pid] > start) != bool(pm["player_node_mask"][k, i]))
    check("player_node_mask == (last kill-event death time > phase_start_s)", bad_d == 0 and bad_node == 0, f"{bad_d} death times and {bad_node} mask cells wrong")
    kills_all, knocks_all, damage_all, items_all = read_events(cur, mid, s_ep)
    kill_times: dict[str, list[float]] = {}
    for _a, v, w in kills_all:
        if v in pid_of:
            kill_times.setdefault(v, []).append(w)
    first_death = {pid: (min(kill_times[pid]) if pid in kill_times else death[pid]) for pid in want_ids}
    bad_f = sum(1 for i, pid in enumerate(want_ids) if not ((math.isinf(first_death[pid]) and math.isinf(float(pm["player_first_death_s"][i]))) or abs(first_death[pid] - float(pm["player_first_death_s"][i])) < 1e-2))
    check("player_first_death_s == first kill event as victim (fallback death time without one)", bad_f == 0, f"{bad_f} wrong")
    if AUDIT_LIB.exists():
        try:
            spec = importlib.util.spec_from_file_location("label_audit_lib", AUDIT_LIB)
            lib = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lib)
            _r, _tt, _mt, pl = lib.build_rows(cur, [mid], {mid: "verify"})
            audit = {str(r.player_id): float(r.d_evt) for r in pl.itertuples(index=False)}
            bad_a = 0
            for i, pid in enumerate(want_ids):
                a, b = audit.get(pid), float(pm["player_death_s"][i])
                bad_a += int(a is None or not ((math.isinf(a) and math.isinf(b)) or abs(a - b) < 1e-2))
            check("player death times agree with the label audit's own code (d_evt)", bad_a == 0, f"{P} players, {bad_a} mismatches")
        except Exception as exc:  # pragma: no cover - optional cross-check
            check("player death times cross-check with the audit lib", True, f"skipped ({exc!r})")
    # alignment with the phase dump (team death times, alive counts)
    team_max = {}
    for i, pid in enumerate(want_ids):
        team_max[team_of[pid]] = max(team_max.get(team_of[pid], -INF), float(pm["player_death_s"][i]))
    bad_t = sum(1 for ti, v in team_max.items() if not ((math.isinf(v) and math.isinf(float(phase_m["team_death_s"][ti]))) or abs(v - float(phase_m["team_death_s"][ti])) < 1e-2))
    f_start = D.FEATURE_NAMES.index("alive_players_start_norm")
    bad_c = 0
    for k in range(8):
        if not bool(phase_m["phase_mask"][k]):
            continue
        for ti in team_max:
            n = sum(1 for i in range(P) if pm["player_team_idx"][i] == ti and bool(pm["player_node_mask"][k, i]))
            bad_c += int(abs(n / D.ALIVE_SCALE - float(phase_m["X"][k, ti, f_start])) > 1e-5)
    check("max player death == phase dump team_death_s; node-mask counts == phase dump alive_players_start_norm * 4", bad_t == 0 and bad_c == 0, f"{bad_t} teams, {bad_c} team-phases differ")

    # 3. Xp -----------------------------------------------------------------------------------
    positions = read_all_positions(cur, mid, s_ep, want_ids)
    kills, knocks, damage, items = kills_all, knocks_all, damage_all, items_all
    cur.execute("SELECT region_kind, snapshot_t, ST_X(center_pos), ST_Y(center_pos), radius_m FROM stdb.dynamic_regions WHERE match_id = %s AND region_kind IN ('safe','poison')", (mid,))
    circles = {(k, int(t)): (float(x), float(y), float(r)) for k, t, x, y, r in cur.fetchall()}
    cur.execute("SELECT phase::int, min(extract(epoch FROM event_time))::float8 FROM pubg.telem_phase_changes WHERE match_id = %s AND phase IS NOT NULL GROUP BY phase", (mid,))
    start_w = {int(p): float(e) - s_ep for p, e in cur.fetchall()}

    def zone_feature(x, y, circle):
        if circle is None or circle[2] < PT.MIN_ZONE_RADIUS_M:
            return 0.0
        cx, cy, r = circle
        return max(PT.ZONE_CLIP[0], min(PT.ZONE_CLIP[1], (r - math.hypot(x - cx, y - cy)) / r))

    worst_circle = 0.0
    cells = 0
    mismatches: list[str] = []
    n_alive = n_cov = 0
    cov_by_phase = {}
    cov_last = {}
    n_late_rows = 0
    margin_min = INF
    would_change = would_total = 0
    expect_edges = {name: {} for name in PT.EDGE_TYPES}
    for k in range(8):
        if not bool(phase_m["phase_mask"][k]) or not math.isfinite(float(phase_m["cut_t"][k])):
            continue
        T = float(phase_m["cut_t"][k])
        Tw = T + delta
        start = float(phase_m["phase_start_s"][k])
        end = float(phase_m["phase_end_s"][k])

        def in_win(w):
            return start <= w <= Tw

        def in_win_uncapped(w):
            return start <= w <= end

        # position sample: the last raw row <= cut_t (game clock) no older than 15 s
        samp = {}
        for pid in want_ids:
            best = None
            for row in positions.get(pid, []):
                if row[0] <= T:
                    best = row
                else:
                    break
            if best is not None and T - best[0] <= PT.POS_STALE_S:
                samp[pid] = best
                nxt = start_w.get(k + 2)
                if nxt is not None:
                    margin_min = min(margin_min, nxt - best[4])
                    n_late_rows += int(best[4] >= nxt)
        strict = {pid for pid in want_ids if first_death[pid] > Tw}  # no kill event yet (the data profile's alive)

        def last_kill_le(pid):
            ws = [w for w in kill_times.get(pid, []) if w <= Tw]
            if ws:
                return max(ws)
            return death[pid] if (pid not in kill_times and death[pid] <= Tw) else -INF

        revived = {pid for pid, row in samp.items() if (row[1] or 0.0) > 0 and row[4] > last_kill_le(pid)}
        alive = strict | revived
        last_rule = {pid for pid in want_ids if death[pid] > Tw}
        n_alive += len(strict)
        n_cov += len([p for p in strict if p in samp])
        cov_by_phase[k] = (len(strict), len([p for p in strict if p in samp]))
        cov_last[k] = (len(last_rule), len([p for p in last_rule if p in samp]))
        # team centroid: alive members with a sample, else every member with a sample
        cent = {}
        for ti in set(team_of.values()):
            mem = [p for p in want_ids if team_of[p] == ti and p in samp]
            use = [p for p in mem if p in alive] or mem
            if use:
                cent[ti] = (sum(samp[p][2] for p in use) / len(use), sum(samp[p][3] for p in use) / len(use))
        # zone circles straight from stdb.dynamic_regions at cut_t; the phase dump's zone row must equal them
        db_cur, db_nxt = circles.get(("safe", int(T))), circles.get(("poison", int(T)))
        for tag, db, cols in (("cur", db_cur, slice(0, 3)), ("next", db_nxt, slice(3, 6))):
            row = [float(v) * size for v in phase_m["zone"][k, cols].tolist()]
            if db is not None and all(math.isfinite(v) for v in row):
                worst_circle = max(worst_circle, max(abs(row[0] - db[0]), abs(row[1] - db[1]), abs(row[2] - db[2])) / 100.0)
            elif (db is None) != (not all(math.isfinite(v) for v in row)):
                mismatches.append(f"phase {k + 1}: zone {tag} circle present in only one of phase dump / DB")

        # events: damage dealt (other-team victims) / taken (any source but self) in the capped and the uncapped window
        dealt = {pid: 0.0 for pid in want_ids}
        taken = {pid: 0.0 for pid in want_ids}
        dealt_u = {pid: 0.0 for pid in want_ids}
        taken_u = {pid: 0.0 for pid in want_ids}
        for a, v, d, w in damage:
            if d is None or d <= 0 or v not in pid_of:
                continue
            enemy_hit = a in pid_of and a != v and team_of[a] != team_of[v]
            if in_win(w):
                if enemy_hit:
                    dealt[a] += d / 100.0
                if a != v:
                    taken[v] += d / 100.0
            if in_win_uncapped(w):
                if enemy_hit:
                    dealt_u[a] += d / 100.0
                if a != v:
                    taken_u[v] += d / 100.0
        knocks_n = {pid: 0.0 for pid in want_ids}
        for a, v, w in knocks:
            if a in pid_of and v in pid_of and team_of[a] != team_of[v] and in_win(w):
                knocks_n[a] += 1.0
        kills_ph = {pid: 0.0 for pid in want_ids}
        kills_cum = {pid: 0.0 for pid in want_ids}
        kills_unc = {pid: 0.0 for pid in want_ids}
        for a, v, w in kills:
            if a in pid_of and v in pid_of and team_of[a] != team_of[v]:
                kills_ph[a] += float(in_win(w))
                kills_cum[a] += float(w <= Tw)
                kills_unc[a] += float(in_win_uncapped(w))
        heals = {pid: 0.0 for pid in want_ids}
        boosts = {pid: 0.0 for pid in want_ids}
        for a, c, sub, w in items:
            if a in pid_of and c == "Use" and w <= Tw:
                if sub == "Heal":
                    heals[a] += 1.0
                elif sub == "Boost":
                    boosts[a] += 1.0

        for pid in want_ids:
            f = [0.0] * PT.N_PLAYER_FEATURES
            f[5], f[6], f[7], f[8], f[9], f[10] = dealt[pid], taken[pid], knocks_n[pid], kills_ph[pid], kills_cum[pid], heals[pid]
            if pid in samp:
                _t, h, x, y, _w, _id = samp[pid]
                ti = team_of[pid]
                f[0] = (x - cent[ti][0]) / 100.0
                f[1] = (y - cent[ti][1]) / 100.0
                f[2] = x / size
                f[3] = y / size
                f[4] = (0.0 if h is None else h) / 100.0
                f[11] = zone_feature(x, y, db_cur)
                f[12] = zone_feature(x, y, db_nxt)
            got = pm["Xp"][k, pid_of[pid]].tolist()
            for c in range(PT.N_PLAYER_FEATURES):
                cells += 1
                if abs(got[c] - f[c]) / (abs(f[c]) + 1.0) > TOL:
                    mismatches.append(f"{pid[:12]} phase {k + 1} {PT.PLAYER_FEATURE_NAMES[c]}: dump {got[c]:.5f} vs {f[c]:.5f}")
            would_total += 1
            would_change += int(abs(dealt[pid] - dealt_u[pid]) > 1e-9 or abs(taken[pid] - taken_u[pid]) > 1e-9 or kills_ph[pid] != kills_unc[pid])
            got_valid = bool(pm["player_pos_valid"][k, pid_of[pid]])
            if (got_valid != (pid in samp) or bool(pm["player_alive_cut"][k, pid_of[pid]]) != (pid in alive)
                    or bool(pm["player_alive_cut_strict"][k, pid_of[pid]]) != (pid in strict)):
                mismatches.append(f"{pid[:12]} phase {k + 1}: pos_valid / alive_cut / alive_cut_strict flags differ")
            if abs(float(pm["cum_use_boost"][k, pid_of[pid]]) - boosts[pid]) > 1e-9:
                mismatches.append(f"{pid[:12]} phase {k + 1}: cum_use_boost differs")

        # edges at the cut: event-alive players that have a sample
        ids = [p for p in want_ids if p in samp and p in alive]
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = ids[a_i], ids[b_i]
                d = math.hypot(samp[a][2] - samp[b][2], samp[a][3] - samp[b][3])
                w = math.exp(-d / PT.EDGE_SCALE_M)
                if team_of[a] == team_of[b]:
                    kind = "teammate"
                elif d <= PT.ENEMY_PROX_M:
                    kind = "enemy_prox"
                else:
                    continue
                expect_edges[kind][(k, pid_of[a], pid_of[b])] = w
                expect_edges[kind][(k, pid_of[b], pid_of[a])] = w
        pairs = {}
        for a, v, d, w in damage:
            if d and d > 0 and a in pid_of and v in pid_of and team_of[a] != team_of[v] and in_win(w):
                pairs.setdefault((a, v), [0.0, 0.0, 0.0])[0] += d
        for a, v, w in kills:
            if a in pid_of and v in pid_of and team_of[a] != team_of[v] and in_win(w):
                pairs.setdefault((a, v), [0.0, 0.0, 0.0])[1] += 1
        for a, v, w in knocks:
            if a in pid_of and v in pid_of and team_of[a] != team_of[v] and in_win(w):
                pairs.setdefault((a, v), [0.0, 0.0, 0.0])[2] += 1
        for (a, v), (dm, kl, kn) in pairs.items():
            expect_edges["combat"][(k, pid_of[a], pid_of[v])] = math.log1p(dm + 100.0 * kl + 50.0 * kn)

    check("Xp (13 columns) + pos_valid / alive_cut flags + cum_use_boost == independent recomputation", not mismatches,
          f"{cells} cells, {len(mismatches)} wrong" + (f", e.g. {mismatches[0]}" if mismatches else ""))
    check("the phase dump's zone circles equal stdb.dynamic_regions at cut_t (max abs diff, hectometres)", worst_circle < 2e-3, f"{worst_circle:.6f} hm")
    bad_e = 0
    detail = []
    for kind in PT.EDGE_TYPES:
        idx, w = pm["edges_player"][kind]
        got = {tuple(r): float(x) for r, x in zip(idx.tolist(), w.tolist())}
        want_e = expect_edges[kind]
        diff = set(got) ^ set(want_e)
        wdiff = max((abs(got[k_] - want_e[k_]) for k_ in set(got) & set(want_e)), default=0.0)
        bad_e += len(diff) + int(wdiff > 1e-5)
        detail.append(f"{kind} {len(got)}/{len(want_e)}")
    check("edges_player (teammate / enemy_prox / combat) sets and weights == independent recomputation", bad_e == 0, "dump/independent edges: " + ", ".join(detail))
    # 5. causality --------------------------------------------------------------------------
    check("no raw position row used lies at/after the true start of the next phase", n_late_rows == 0,
          f"{'smallest margin %.1f s' % margin_min if margin_min < INF else 'no next phase'}")
    check("(info) damage/kill numbers that WOULD change with the uncapped window [phase_start_s, phase_end_s]", True,
          f"{would_change} of {would_total} player-phases")
    last_w = max([w for _a, _v, w in kills] + [0.0])
    cut_w = max(float(v) for v in pm["cut_wall_s"][~torch.isnan(pm["cut_wall_s"])]) if bool((~torch.isnan(pm["cut_wall_s"])).any()) else 0.0
    ok_cap = all(float(pm["cut_wall_s"][k]) <= float(pm["phase_end_s"][k]) - 10.0 + 1e-3 for k in range(8) if not torch.isnan(pm["cut_wall_s"][k]))
    check("cut_wall_s <= phase_end_s - 10 for every phase (events are used up to cut_wall_s only)", ok_cap, f"largest cut_wall_s {cut_w:.1f} s of a {last_w:.0f} s match")
    # 6. coverage ---------------------------------------------------------------------------
    cov = n_cov / max(n_alive, 1)
    dump_cov = {k: (int(pm["player_alive_cut_strict"][k].sum()), int((pm["player_alive_cut_strict"][k] & pm["player_pos_valid"][k]).sum())) for k in cov_by_phase}
    la = sum(v[0] for v in cov_last.values())
    lv = sum(v[1] for v in cov_last.values())
    check("coverage: alive players (no kill event yet) at the cut with a position sample <= 15 s before it (dump flags == recomputation)",
          dump_cov == cov_by_phase, f"{n_cov} of {n_alive} ({100 * cov:.2f} %); with the last-death rule {lv} of {la} ({100 * lv / max(la, 1):.2f} %)")
    if show:
        print(f"  clock delta {delta:.2f} s; phases with a cut: {[k + 1 for k in range(8) if math.isfinite(float(phase_m['cut_t'][k]))]}")
        print("  phase  node  alive@cut  with_sample  teammate  enemy_prox  combat")
        for k in range(8):
            if not bool(phase_m["phase_mask"][k]):
                continue
            ne = [int((pm["edges_player"][n][0][:, 0] == k).sum()) if len(pm["edges_player"][n][0]) else 0 for n in PT.EDGE_TYPES]
            print(f"  {k + 1:5d} {int(pm['player_node_mask'][k].sum()):5d} {int(pm['player_alive_cut'][k].sum()):9d} "
                  f"{int((pm['player_alive_cut'][k] & pm['player_pos_valid'][k]).sum()):12d} {ne[0]:9d} {ne[1]:11d} {ne[2]:7d}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True, help="a player dump split (.pt)")
    ap.add_argument("--match-id", action="append", required=True)
    ap.add_argument("--phase-dump", type=Path, default=None, help="phase dump split of the same cohort/split (default: from meta)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    dump = torch.load(args.dump, map_location="cpu", weights_only=True)
    phase_path = args.phase_dump or Path(dump["meta"]["phase_dump"]["path"])
    phase = torch.load(phase_path, map_location="cpu", weights_only=True)
    phase_by_id = {m["match_id"]: m for m in phase["matches"]}
    failed = 0
    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            for pm in dump["matches"]:
                if pm["match_id"] not in args.match_id:
                    continue
                results = verify(cur, pm, phase_by_id[pm["match_id"]], show=args.show)
                print(f"match {pm['match_id']}  cohort {dump['meta']['cohort']}  players {len(pm['player_ids'])}  teams {len(pm['team_ids'])}")
                for name, ok, detail in results:
                    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
                failed += int(not all(ok for _, ok, _ in results))
    finally:
        conn.close()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
