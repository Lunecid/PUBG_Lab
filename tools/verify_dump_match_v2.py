"""Independent re-derivation of one v2 phase-dump record straight from the raw ``pubg.*`` tables (hand-check tool).

Nothing here imports the phase_v2 builder logic: phases, causal alive status, the elimination time T, the risk set, the
labels, the exclusions, the alive-count inputs, a sample of event inputs, the combat edges and the zone vector are all
recomputed with plain SQL aggregates and brute-force loops, then compared with the record (built by
``model_stgcn.phase_v2.assemble.build_match_v2`` or loaded from a v2 dump file).

    python -m tools.verify_dump_match_v2 --match-id <id>            # builds the record with the builder
    python -m tools.verify_dump_match_v2 --match-id <id> --dump-dir model_stgcn/dumps/bigcomp2027_phase_v2
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

W = "extract(epoch FROM (%(t)s - s.event_time))::float8"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _q(cur, sql: str, params=None) -> list[tuple]:
    cur.execute(sql, params or ())
    return cur.fetchall()


def raw_facts(cur, match_id: str) -> dict:
    """Everything the checks need, with W = event_time - LogMatchStart."""
    (map_name,) = _q(cur, "SELECT map_name FROM stdb.matches WHERE match_id = %s", (match_id,))[0]
    ev = _q(
        cur,
        """SELECT ceil(c.is_game)::int, min(extract(epoch FROM (c.event_time - s.event_time)))::float8
             FROM pubg.telem_phase_changes c JOIN pubg.telem_match_start s ON s.match_id = c.match_id
            WHERE c.match_id = %s GROUP BY ceil(c.is_game)""",
        (match_id,),
    )
    first = {int(k): float(w) for k, w in ev if int(k) >= 1}
    end = _q(
        cur,
        """SELECT extract(epoch FROM (e.event_time - s.event_time))::float8 FROM pubg.telem_match_end e
             JOIN pubg.telem_match_start s ON s.match_id = e.match_id WHERE e.match_id = %s""",
        (match_id,),
    )
    roster = _q(
        cur,
        """SELECT r.team_id, p.player_id, p.death_type, p.time_survived::float8
             FROM pubg.participants p JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
            WHERE p.match_id = %s AND r.team_id IS NOT NULL""",
        (match_id,),
    )
    team_rows = _q(cur, "SELECT team_id, num_players, final_rank, won FROM stdb.teams WHERE match_id = %s ORDER BY team_id", (match_id,))
    kills = _q(
        cur,
        """SELECT k.victim_id, extract(epoch FROM (k.event_time - s.event_time))::float8, k.victim_rank
             FROM pubg.telem_kills k JOIN pubg.telem_match_start s ON s.match_id = k.match_id
            WHERE k.match_id = %s AND k.victim_id IS NOT NULL""",
        (match_id,),
    )
    healthy = _q(
        cur,
        """SELECT p.account_id, extract(epoch FROM (p.event_time - s.event_time))::float8
             FROM pubg.telem_positions p JOIN pubg.telem_match_start s ON s.match_id = p.match_id
            WHERE p.match_id = %s AND p.is_game >= 0.1 AND p.account_id IS NOT NULL AND p.team_id IS NOT NULL
              AND p.elapsed_time IS NOT NULL AND p.pos_x IS NOT NULL AND p.pos_y IS NOT NULL AND p.health > 0""",
        (match_id,),
    )
    pos_t = _q(
        cur,
        """SELECT DISTINCT extract(epoch FROM (p.event_time - s.event_time))::float8
             FROM pubg.telem_positions p JOIN pubg.telem_match_start s ON s.match_id = p.match_id
            WHERE p.match_id = %s AND p.is_game >= 0.1 AND p.account_id IS NOT NULL AND p.team_id IS NOT NULL
              AND p.elapsed_time IS NOT NULL AND p.pos_x IS NOT NULL AND p.pos_y IS NOT NULL ORDER BY 1""",
        (match_id,),
    )
    return {
        "map_name": map_name, "first": first, "end": float(end[0][0]) if end and end[0][0] is not None else None,
        "roster": roster, "teams": team_rows, "kills": kills, "healthy": healthy,
        "pos_times": np.asarray([r[0] for r in pos_t], dtype=np.float64),
    }  # fmt: skip


def brute_alive(kill_times: list[float], healthy_times: list[float], t: float) -> bool:
    """Alive at t: no kill <= t, or a healthy sample in (last kill <= t, t]."""
    ks = [k for k in kill_times if k <= t]
    if not ks:
        return True
    last = max(ks)
    return any(last < h <= t for h in healthy_times)


def alive_count_curve(members, kills_by, healthy_by, lo: float, hi: float):
    """(count at lo, count at hi, min, time average) over [lo, hi) by evaluating the definition on every elementary
    segment between consecutive event times (kills and healthy samples of the members) at the segment midpoint."""
    def count(t):
        return sum(brute_alive(kills_by.get(a, []), healthy_by.get(a, []), t) for a in members)

    cuts = {lo, hi}
    for a in members:
        for t in kills_by.get(a, []) + healthy_by.get(a, []):
            if lo < t < hi:
                cuts.add(t)
    pts = sorted(cuts)
    n_lo, n_hi = count(lo), count(hi - 1e-9)
    if not hi > lo:
        return count(hi), count(hi), count(hi), count(hi)
    vals, lens = [], []
    for x, y in zip(pts, pts[1:]):
        vals.append(count(0.5 * (x + y)))
        lens.append(y - x)
    return n_lo, n_hi, min(vals), float(np.dot(vals, lens) / (hi - lo))


def verify_record(cur, rec: dict, *, tol: float = 1e-4) -> list[Check]:
    mid = rec["match_id"]
    f = raw_facts(cur, mid)
    out: list[Check] = []
    team_ids = [int(t) for t in rec["team_ids"].tolist()]
    N = len(team_ids)
    first = f["first"]
    ks_all = sorted(first)
    final = max(ks_all)

    # ---- 1 phases / cuts -----------------------------------------------------------------------------------
    ok = True
    for k in range(1, 9):
        p = k - 1
        has = k in first
        ok &= bool(rec["phase_mask"][p]) == has
        if has:
            ok &= abs(float(rec["phase_start_s"][p]) - first[k]) < 1e-3
            b = first.get(k + 1)
            if b is None:
                ok &= math.isnan(float(rec["phase_end_s"][p])) and not bool(rec["cut_mask"][p])
            else:
                ok &= abs(float(rec["phase_end_s"][p]) - b) < 1e-3
                ok &= abs(float(rec["cut_t"][p]) - (b - 10.0)) < 1e-3 and bool(rec["cut_mask"][p])
    out.append(Check("phase starts / b_k / a_k = b_k - 10 from LogPhaseChange events (ceil(isGame))", bool(ok),
                     f"phases {ks_all}, final phase {final}"))
    out.append(Check("LogMatchEnd C_m", f["end"] is not None and abs(float(rec["match_end_s"]) - f["end"]) < 1e-3, f"C_m = {f['end']}"))

    # ---- 2 causal alive counts, node_mask --------------------------------------------------------------------
    acct_team = {pid: int(t) for t, pid, _d, _s in f["roster"]}
    members = defaultdict(list)
    for pid, t in acct_team.items():
        members[t].append(pid)
    kills_by, rank_kills = defaultdict(list), defaultdict(list)
    for v, w, rk in f["kills"]:
        kills_by[v].append(float(w))
        if rk and v in acct_team:
            rank_kills[acct_team[v]].append(float(w))
    healthy_by = defaultdict(list)
    for a, w in f["healthy"]:
        healthy_by[a].append(float(w))
    bad_counts = bad_mask = n_checked = 0
    col = {n: i for i, n in enumerate(rec["_feature_names"])} if "_feature_names" in rec else None
    names4 = ["alive_players_start_norm", "alive_players_end_norm", "alive_players_min_norm", "alive_players_avg_norm"]
    from model_stgcn.phase_v2.features import FEATURE_NAMES  # column order only

    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for k in range(1, 8):
        if k not in first or (k + 1) not in first:
            continue
        s_k, a_k = first[k], first[k + 1] - 10.0
        for i, tid in enumerate(team_ids):
            mem = members.get(tid, [])
            kb = {a: [t for t in kills_by.get(a, []) if t < a_k] for a in mem}  # records strictly before the cut
            hb = {a: [t for t in healthy_by.get(a, []) if t < a_k] for a in mem}
            c = alive_count_curve(mem, kb, hb, s_k, a_k)
            got = [float(rec["X"][k - 1, i, col[n]]) * 4.0 for n in names4]
            n_checked += 1
            want = list(c) if a_k > s_k else [c[1]] * 4
            if any(abs(g - w) > 5e-4 for g, w in zip(got, want)):
                bad_counts += 1
            if bool(rec["node_mask"][k - 1, i]) != (c[1] > 0):
                bad_mask += 1
    out.append(Check("alive_players_{start,end,min,avg}: brute-force step function over [s_k, a_k) from records < a_k",
                     bad_counts == 0, f"{n_checked} team-cut rows, {bad_counts} mismatches"))
    out.append(Check("node_mask = team causally alive at a_k (>= 1 member alive)", bad_mask == 0, f"{bad_mask} mismatches"))

    # ---- 3 T, censoring, risk set, labels --------------------------------------------------------------------
    offs = []
    for t, pid, dt, ts in f["roster"]:
        if not pid.startswith("ai.") and ts is not None and kills_by.get(pid):
            offs.append(max(kills_by[pid]) - ts)
    off = float(np.median(offs)) if offs else 0.70
    T_ref = {}
    for tid in team_ids:
        fin = []
        for t, pid, dt, ts in f["roster"]:
            if t != tid:
                continue
            ks = kills_by.get(pid, [])
            if not ks:
                fin.append(math.inf if dt in (None, "alive") else (ts + off if ts is not None else math.inf))
            else:
                last = max(ks)
                fin.append(math.inf if any(h > last for h in healthy_by.get(pid, [])) else last)
        T_ref[tid] = max(fin) if fin else math.nan
    won = {int(t): bool(w) for t, _n, _r, w in f["teams"]}
    for t in T_ref:
        if won.get(t):
            T_ref[t] = math.inf
    bad_T = sum(
        1 for i, tid in enumerate(team_ids)
        if not ((math.isnan(T_ref[tid]) and math.isnan(float(rec["team_death_s"][i])))
                or (T_ref[tid] == float(rec["team_death_s"][i])) or abs(T_ref[tid] - float(rec["team_death_s"][i])) < 1e-3)
    )  # fmt: skip
    out.append(Check("T = max over members of the final death (last kill without later healthy sample); winners +inf", bad_T == 0, f"{bad_T} teams differ"))
    game_marker_bad = 0
    n_marker = 0
    for i, tid in enumerate(team_ids):
        T = float(rec["team_death_s"][i])
        if np.isfinite(T) and rank_kills.get(tid):
            n_marker += 1
            game_marker_bad += abs(max(rank_kills[tid]) - T) > 1e-3
    out.append(Check("T equals the time of the team's last victim_rank > 0 kill (the game's own elimination marker)", game_marker_bad == 0,
                     f"{n_marker} finite-T teams with a marked kill, {game_marker_bad} differ"))
    out.append(Check("exactly one winner, censored (T = +inf), never a positive label", sum(won.values()) == 1 and all(
        np.isinf(float(rec["team_death_s"][i])) and not bool((rec["y_hazard_v2"][:7, i] > 0).any()) for i, t in enumerate(team_ids) if won[t]), ""))

    # exclusions: position gap > 60 s overlapping the window / no LogMatchEnd
    pt = f["pos_times"]
    gaps = []
    if f["end"] is not None and len(pt):
        ext = np.append(pt, max(f["end"], pt[-1]))
        gaps = [(float(a), float(b)) for a, b in zip(ext[:-1], ext[1:]) if b - a > 60.0]
    bad_lab = 0
    n_rows = n_pos = 0
    for k in range(1, 8):
        if k not in first or (k + 1) not in first:
            continue
        b = first[k + 1]
        nxt = first.get(k + 2)
        e = f["end"] if nxt is None else nxt
        excluded = f["end"] is None or any(g0 < e and g1 > b for g0, g1 in gaps)
        if excluded != (k in rec["cut_excluded"]) or bool(rec["cut_valid"][k - 1]) == excluded:
            bad_lab += 1
        for i, tid in enumerate(team_ids):
            want_risk = (not excluded) and (not math.isnan(T_ref[tid])) and T_ref[tid] > b
            want_y = float(want_risk and math.isfinite(T_ref[tid]) and T_ref[tid] <= e)
            n_rows += int(want_risk)
            n_pos += int(want_y)
            if bool(rec["target_mask_v2"][k - 1, i]) != want_risk or float(rec["y_hazard_v2"][k - 1, i]) != want_y:
                bad_lab += 1
    out.append(Check("risk set {T > b_k}, label {b_k < T <= b_{k+1}} ((b_k, C_m] when phase k+1 is final), exclusions only for truncated logs",
                     bad_lab == 0, f"{n_rows} risk rows, {n_pos} positives, {bad_lab} mismatches, excluded {rec['cut_excluded']}"))
    out.append(Check("query_mask = node_mask | target_mask_v2", bool((rec["query_mask"] == (rec["node_mask"] | rec["target_mask_v2"])).all()), ""))
    comeback = int((rec["target_mask_v2"] & ~rec["node_mask"]).sum())
    out.append(Check("comeback rows (risk set but causally dead at a_k) keep a query token", bool(
        ((rec["target_mask_v2"] & ~rec["node_mask"]) <= rec["query_mask"]).all()), f"{comeback} such rows"))

    # ---- 4 event inputs by SQL --------------------------------------------------------------------------------
    team_of = {pid: t for pid, t in acct_team.items()}
    bad_ev = n_ev = 0
    wexpr = "extract(epoch FROM (%s.event_time - s.event_time))"
    for k in range(1, 8):
        if k not in first or (k + 1) not in first:
            continue
        s_k, a_k = first[k], first[k + 1] - 10.0
        if not a_k > s_k:
            continue
        dmg = _q(
            cur,
            f"""SELECT d.attacker_id, d.victim_id, coalesce(d.damage, 0)::float8
                  FROM pubg.telem_damage d JOIN pubg.telem_match_start s ON s.match_id = d.match_id
                 WHERE d.match_id = %s AND {wexpr % 'd'} >= %s AND {wexpr % 'd'} < %s""",
            (mid, s_k, a_k),
        )
        kil = _q(
            cur,
            f"""SELECT k.killer_id, k.victim_id FROM pubg.telem_kills k JOIN pubg.telem_match_start s ON s.match_id = k.match_id
                 WHERE k.match_id = %s AND k.victim_id IS NOT NULL AND {wexpr % 'k'} >= %s AND {wexpr % 'k'} < %s""",
            (mid, s_k, a_k),
        )
        pick = _q(
            cur,
            f"""SELECT p.team_id, count(*) FROM pubg.telem_item_pickup p JOIN pubg.telem_match_start s ON s.match_id = p.match_id
                 WHERE p.match_id = %s AND p.team_id IS NOT NULL AND {wexpr % 'p'} >= %s AND {wexpr % 'p'} < %s GROUP BY 1""",
            (mid, s_k, a_k),
        )
        dealt, taken, kills_taken = defaultdict(float), defaultdict(float), defaultdict(int)
        for att, vic, dm in dmg:
            ta, tv = team_of.get(att), team_of.get(vic)
            if ta is not None and tv is not None and ta != tv:
                dealt[ta] += dm
                taken[tv] += dm
        for killer, vic in kil:
            ta, tv = team_of.get(killer), team_of.get(vic)
            if ta is not None and tv is not None and ta != tv:
                kills_taken[tv] += 1
        pick = {int(t): int(c) for t, c in pick}
        for i, tid in enumerate(team_ids):
            n_ev += 1
            x = rec["X"][k - 1, i]
            exp = {"damage_dealt_norm": dealt[tid] / 1000.0, "damage_taken_norm": taken[tid] / 1000.0,
                   "kills_taken_norm": kills_taken[tid] / 10.0, "pickup_count_norm": pick.get(tid, 0) / 100.0}  # fmt: skip
            if any(abs(float(x[col[n]]) - v) > 1e-4 * (1 + abs(v)) for n, v in exp.items()):
                bad_ev += 1
    out.append(Check("event inputs (damage dealt / taken, kills taken, pickups) = SQL sums over [s_k, a_k) with roster teams", bad_ev == 0,
                     f"{n_ev} team-cut rows, {bad_ev} mismatches"))

    # ---- 5 combat edges by SQL --------------------------------------------------------------------------------
    idx, w = rec["edges"]["combat"]
    bad_edge = n_edge = 0
    for k in range(1, 9):
        if k not in first or (k + 1) not in first:
            continue
        s_k, a_k = first[k], first[k + 1] - 10.0
        acc = defaultdict(lambda: [0.0, 0, 0])
        for att, vic, dm in _q(cur, f"""SELECT d.attacker_id, d.victim_id, coalesce(d.damage,0)::float8 FROM pubg.telem_damage d
              JOIN pubg.telem_match_start s ON s.match_id = d.match_id WHERE d.match_id=%s AND {wexpr % 'd'} >= %s AND {wexpr % 'd'} < %s""", (mid, s_k, a_k)):
            ta, tv = team_of.get(att), team_of.get(vic)
            if ta is not None and tv is not None and ta != tv:
                acc[(ta, tv)][0] += dm
        for killer, vic in _q(cur, f"""SELECT k.killer_id, k.victim_id FROM pubg.telem_kills k JOIN pubg.telem_match_start s ON s.match_id=k.match_id
              WHERE k.match_id=%s AND k.victim_id IS NOT NULL AND {wexpr % 'k'} >= %s AND {wexpr % 'k'} < %s""", (mid, s_k, a_k)):
            ta, tv = team_of.get(killer), team_of.get(vic)
            if ta is not None and tv is not None and ta != tv:
                acc[(ta, tv)][1] += 1
        for att, vic in _q(cur, f"""SELECT g.attacker_id, g.victim_id FROM pubg.telem_groggy g JOIN pubg.telem_match_start s ON s.match_id=g.match_id
              WHERE g.match_id=%s AND {wexpr % 'g'} >= %s AND {wexpr % 'g'} < %s""", (mid, s_k, a_k)):
            ta, tv = team_of.get(att), team_of.get(vic)
            if ta is not None and tv is not None and ta != tv:
                acc[(ta, tv)][2] += 1
        want = {(a, b): math.log(1 + max(0.0, d) + 100 * kc + 50 * gc) for (a, b), (d, kc, gc) in acc.items()
                if max(0.0, d) + 100 * kc + 50 * gc > 0}
        got = {}
        sel = idx[:, 0] == k - 1
        for _p, i, j in idx[sel].tolist():
            got[(team_ids[i], team_ids[j])] = float(w[(idx[:, 0] == k - 1) & (idx[:, 1] == i) & (idx[:, 2] == j)][0])
        n_edge += len(want)
        if set(want) != set(got) or any(abs(want[e] - got[e]) > 1e-4 for e in want):
            bad_edge += 1
    out.append(Check("combat edges = SQL aggregate ln(1 + damage + 100 kills + 50 groggy) over [s_k, a_k)", bad_edge == 0, f"{n_edge} edges, {bad_edge} phases differ"))

    # ---- 6 zone, ai share, rank ------------------------------------------------------------------------------
    bad_zone = 0
    size = float(rec["map_size"])
    for k in range(1, 9):
        if k not in first or (k + 1) not in first:
            continue
        a_k = first[k + 1] - 10.0
        row = _q(
            cur,
            """SELECT safe_zone_x::float8, safe_zone_y::float8, safe_zone_radius::float8, poison_zone_x::float8, poison_zone_y::float8,
                      poison_zone_radius::float8 FROM pubg.telem_game_states g JOIN pubg.telem_match_start s ON s.match_id = g.match_id
                WHERE g.match_id = %s AND extract(epoch FROM (g.event_time - s.event_time)) < %s
                  AND g.safe_zone_radius > 0 AND g.poison_zone_radius > 0 ORDER BY g.event_time DESC LIMIT 1""",
            (mid, a_k),
        )
        if row:
            sx, sy, sr, px, py, pr = [v / 100.0 for v in row[0]]
            want = np.array([sx, sy, sr, px, py, pr]) / size
            if not np.allclose(rec["zone"][k - 1, :6].numpy(), want, atol=1e-5):
                bad_zone += 1
    out.append(Check("zone vector = latest game state before a_k / map size", bad_zone == 0, f"{bad_zone} cuts differ"))
    ai_bad = sum(
        1 for i, tid in enumerate(team_ids)
        if members.get(tid) and abs(float(rec["team_ai_frac"][i]) - np.mean([p.startswith("ai.") for p in members[tid]])) > 1e-6
    )  # fmt: skip
    out.append(Check("team_ai_frac = share of ai.* accounts in the roster", ai_bad == 0, f"{ai_bad} teams differ"))
    nteam = len(f["teams"])
    rank_bad = sum(1 for (t, _n, r, _w), x in zip(f["teams"], rec["final_rank_norm"].tolist()) if r is not None and abs(r / nteam - x) > 1e-6)
    out.append(Check("final_rank_norm = rank / n_teams", rank_bad == 0, f"{rank_bad} teams differ"))
    return out


def print_report(match_id: str, checks: list[Check]) -> bool:
    ok = all(c.ok for c in checks)
    print(f"== {match_id}: {'ALL PASS' if ok else 'FAIL'}")
    for c in checks:
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--match-id", action="append", required=True)
    ap.add_argument("--dump-dir", default=None, help="load the record from a v2 dump instead of building it")
    args = ap.parse_args()
    from stdb.conn import connect

    conn = connect()
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    all_ok = True
    for mid in args.match_id:
        if args.dump_dir:
            import glob

            import torch

            rec = None
            for p in glob.glob(f"{args.dump_dir}/*/*.pt"):
                d = torch.load(p, weights_only=False)
                rec = next((m for m in d["matches"] if m["match_id"] == mid), None)
                if rec is not None:
                    break
            if rec is None:
                print(f"{mid}: not in the dump")
                all_ok = False
                continue
        else:
            from model_stgcn.phase_v2.assemble import build_match_v2
            from model_stgcn.phase_v2.replay import DEFAULT_SCHEDULE, MAP_SIZE
            from model_stgcn.phase_v2.source import fetch_match_source

            src = fetch_match_source(cur, mid, schedule=DEFAULT_SCHEDULE)
            rec = build_match_v2(src, map_size=MAP_SIZE[src.map_name])
        all_ok &= print_report(mid, verify_record(cur, rec))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
