"""Hand check of ONE dumped match against the database (independent re-derivation, design v2.3).

    python -m tools.verify_dump_match --match-id <uuid> [--show]      # build the dump entry and check it
    python -m tools.verify_dump_match --dump <train.pt> --index 0     # check an entry of a written dump
    python -m tools.verify_dump_match --dump <test.pt> --match-id <uuid> [--match-id ...]   # ... by match id

Every number the dump claims is recomputed here with plain SQL + small python loops (not with the dump
code path) and compared:

  1. X            re-scaled from stdb.v_team_phase_feature_long + the three V2 feature tables; the four
                  event-based alive-count columns against kill events (independent death model)
  2. pipeline     node_mask_label / target_mask / y_hazard / final_rank_norm / top5 / top10 vs
     labels       stdb.team_phase_labels (must be unchanged)
  3. labels v2    team_death_s / target_mask_v2 / y_hazard_v2 / node_mask_v2 from raw kill events,
                  participants and stdb.match_phase_global bounds; cross-checked with the label audit's own
                  code (reports/bigcomp2027_data_profile/scripts/label_audit_lib.py) when importable
  4. edges        stdb.team_phase_edge_cache rows (counts and weights, directions)
  5. clock        clock_delta_median_s vs an independent game-state median, clock_delta_s >= median (raised
                  in 0.25 s steps for drifting clocks); cut_t == floor((phase_end_s - delta - 10) / 10) * 10,
                  cut_wall_s == cut_t + delta
  6. team_pos     recomputed from RAW telem_positions (nearest sample within +-5 s per player, centroid of the
                  alive members, fallback to the previous grid point inside the window) / map size
  7. zone         stdb.dynamic_regions (safe / poison) at cut_t / map size, phase_no, phase_frac
  8. no look-     the last raw position row the capped grid may read (G <= cut_t + 5) happens BEFORE the true start
     ahead        of phase k+1 (first LogPhaseChange of phase k+1, event clock), and so does the latest game
                  state of the uncapped game-state window; the cap-only rule (no clock shift) and the plain
                  median delta are evaluated for comparison; team_phase_state.position_rows == raw rows in the
                  shifted + capped window; mean_health / waypoints re-derived from raw rows / alive grid rows
  9. audit        team_won / team_ai_frac / team_has_monster_dmg
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

from model_stgcn import dump_phase_tier as D
from stdb.conn import connect
from stdb.queries import PHASE_NODE_FEATURE_SPECS, PHASE_NODE_FEATURE_SPECS_V2

TOL = 1e-4
INF = float("inf")
AUDIT_LIB = Path(__file__).resolve().parent.parent / "reports/bigcomp2027_data_profile/scripts/label_audit_lib.py"


def _scale(value, scale, transform):
    if value is None:
        return 0.0
    if transform == "bool":
        return 1.0 if bool(value) else 0.0
    x = float(value)
    if not np.isfinite(x):
        return 0.0
    if transform == "log1p":
        return float(np.log1p(max(x, 0.0)))
    return x / scale if scale else x


# ------------------------------------------------------------------------------------------------
# independent re-derivations
# ------------------------------------------------------------------------------------------------
def independent_clock(cur, mid: str) -> tuple[float, float]:
    """``(delta, S)``: game-state median of (event epoch - elapsed_time) minus the match start epoch."""
    cur.execute(
        """
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM gs.event_time) - gs.elapsed_time)::float8,
               extract(epoch FROM m.started_at)::float8
          FROM pubg.telem_game_states gs JOIN stdb.matches m ON m.match_id = gs.match_id
         WHERE gs.match_id = %s AND gs.elapsed_time IS NOT NULL
         GROUP BY m.started_at
        """,
        (mid,),
    )
    o_gs, s_ep = cur.fetchone()
    return float(o_gs) - float(s_ep), float(s_ep)


def independent_deaths(cur, mid: str, s_ep: float):
    """Per-team death time T (W clock, +inf = alive at the end) and per-player death times."""
    cur.execute("SELECT victim_id, extract(epoch FROM event_time)::float8 FROM pubg.telem_kills WHERE match_id = %s AND victim_id IS NOT NULL", (mid,))
    last: dict[str, float] = {}
    for victim, ep in cur.fetchall():
        w = float(ep) - s_ep
        if victim not in last or w > last[victim]:
            last[victim] = w
    cur.execute(
        """
        SELECT r.team_id, p.player_id, p.death_type, p.time_survived::float8
          FROM pubg.participants p
          JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
         WHERE p.match_id = %s AND r.team_id IS NOT NULL
        """,
        (mid,),
    )
    parts = cur.fetchall()
    offs = [last[pid] - ts for _t, pid, _d, ts in parts if not str(pid).startswith("ai.") and pid in last and ts is not None]
    off = float(np.median(offs)) if offs else 0.70
    per_player: dict[str, tuple[int, float]] = {}
    for tid, pid, dtype, ts in parts:
        if dtype == "alive":
            d = INF
        elif pid in last:
            d = last[pid]
        elif ts is not None:
            d = float(ts) + off
        else:
            d = INF
        per_player[str(pid)] = (int(tid), d)
    team_t: dict[int, float] = {}
    for tid, d in per_player.values():
        team_t[tid] = max(team_t.get(tid, -INF), d)
    return team_t, per_player, off


def state_of(target: bool, y: float) -> int:
    return 0 if not target else (2 if y > 0.5 else 1)


def raw_positions(cur, mid: str, s_ep: float):
    """Raw rows of the match grouped per team: ``{team_id: [(times, rows)]}`` with one entry per account;
    a row is ``(elapsed_time, health, x_m, y_m, event clock W)``."""
    cur.execute(
        """
        SELECT account_id, team_id, elapsed_time::float8, health::float8, pos_x::float8, pos_y::float8,
               extract(epoch FROM event_time)::float8
          FROM pubg.telem_positions
         WHERE match_id = %s AND account_id IS NOT NULL AND team_id IS NOT NULL
           AND pos_x IS NOT NULL AND elapsed_time IS NOT NULL AND is_game >= 0.1
         ORDER BY account_id, elapsed_time
        """,
        (mid,),
    )
    per_account: dict[tuple[str, int], list[tuple[float, float | None, float, float, float]]] = {}
    for acc, tid, t, h, x, y, ep in cur.fetchall():
        per_account.setdefault((str(acc), int(tid)), []).append((t, h, x / 100.0, y / 100.0, float(ep) - s_ep))
    teams: dict[int, list[tuple[list[float], list]]] = {}
    for (_acc, tid), rows in per_account.items():
        teams.setdefault(tid, []).append(([r[0] for r in rows], rows))
    return teams


def raw_team_centroid(teams, team_id: int, gt: float):
    """Centroid (m) of the alive members' nearest raw samples within +-5 s of grid time ``gt`` (None if no
    alive member has one) -- the stdb.etl resampling rule re-implemented independently."""
    alive = []
    for ts, rows in teams.get(team_id, []):
        i = bisect.bisect_left(ts, gt)
        cands = [j for j in (i - 1, i) if 0 <= j < len(rows)]
        if not cands:
            continue
        best = min(cands, key=lambda j: (abs(ts[j] - gt), j))  # ties -> the earlier sample
        t, h, x, y, _w = rows[best]
        if abs(t - gt) > 5.0:
            continue
        if h is None or h > 0:
            alive.append((x, y))
    if not alive:
        return None
    return (sum(a for a, _ in alive) / len(alive), sum(b for _, b in alive) / len(alive))


def max_w_before(teams, g_limit: float) -> float:
    """Latest event-clock time among raw position rows whose game clock is <= ``g_limit``."""
    best = -INF
    for players in teams.values():
        for ts, rows in players:
            i = bisect.bisect_right(ts, g_limit)
            if i:
                best = max(best, max(r[4] for r in rows[:i]))
    return best


# ------------------------------------------------------------------------------------------------
def verify(cur, m: dict, map_name: str, fixb_since: str | None, show: bool = False) -> list[tuple[str, bool, str]]:
    mid = m["match_id"]
    res: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        res.append((name, bool(ok), detail))

    team_ids = m["team_ids"].tolist()
    tindex = {t: i for i, t in enumerate(team_ids)}
    N = len(team_ids)
    size = float(m["map_size"])
    P = D.N_PHASES

    delta_med, s_ep = independent_clock(cur, mid)
    delta = float(m["clock_delta_s"])  # the effective delta the ETL windows and cut_t use (median, raised for drifting clocks)
    team_t, per_player, off = independent_deaths(cur, mid, s_ep)
    cur.execute("SELECT phase_idx, phase_start_s::float8, phase_end_s::float8 FROM stdb.match_phase_global WHERE match_id = %s ORDER BY phase_idx", (mid,))
    bnd_all = {int(p): (float(a), float(b)) for p, a, b in cur.fetchall()}
    last_phase = max(bnd_all) if bnd_all else None
    bnd = {p: v for p, v in bnd_all.items() if 1 <= p <= P}

    # 1. X ---------------------------------------------------------------------------------------
    specs = list(PHASE_NODE_FEATURE_SPECS) + list(PHASE_NODE_FEATURE_SPECS_V2)
    cur.execute("SELECT * FROM stdb.v_team_phase_feature_long WHERE match_id = %s AND phase_idx BETWEEN 1 AND 8", (mid,))
    cols = [d[0] for d in cur.description]
    rows = {(int(r[cols.index("team_id")]), int(r[cols.index("phase_idx")])): dict(zip(cols, r)) for r in cur.fetchall()}
    for table in ("team_phase_network_features", "team_phase_habitat_features", "team_phase_z_affordance_features"):
        cur.execute(f"SELECT * FROM stdb.{table} WHERE match_id = %s", (mid,))
        c2 = [d[0] for d in cur.description]
        for r in cur.fetchall():
            d = dict(zip(c2, r))
            key = (int(d["team_id"]), int(d["phase_idx"]))
            if key in rows:
                rows[key].update(d)
    replaced = set(D.REPLACED_FEATURES)
    worst = 0.0
    cells = 0
    X = m["X"].numpy()
    for (tid, ph), row in rows.items():
        if tid not in tindex:
            continue
        for f, (name, col, scale, transform) in enumerate(specs):
            if name in replaced:
                continue
            want = _scale(row.get(col), scale, transform)
            got = float(X[ph - 1, tindex[tid], f])
            worst = max(worst, abs(want - got) / (abs(want) + 1.0))
            cells += 1
    check("X (150 - 4 replaced columns) matches the re-scaled DB rows", worst < TOL, f"{cells} cells, worst rel err {worst:.2e}")

    # 1b. replaced alive counts from kill events (independent death model, numeric time average)
    members_by_team: dict[int, list[float]] = {}
    for tid, d in per_player.values():
        members_by_team.setdefault(tid, []).append(d)
    fcol = {name: [s[0] for s in specs].index(name) for name in D.REPLACED_FEATURES}
    bad_alive = n_alive = 0
    pos_based_diff = 0
    for ph, (a, b) in bnd.items():
        ts_grid = np.linspace(a, b, 2001) if b > a else np.array([a])
        for tid, i in tindex.items():
            deaths = np.array(members_by_team.get(tid, []), dtype=np.float64)
            n_s = float((deaths > a).sum())
            n_e = float((deaths > b).sum())
            n_avg = float(np.mean([(deaths > t).sum() for t in ts_grid])) if deaths.size else 0.0
            want = {"alive_players_start_norm": n_s, "alive_players_end_norm": n_e,
                    "alive_players_min_norm": n_e, "alive_players_avg_norm": n_avg}
            for name, value in want.items():
                tol = 2e-3 if name.endswith("avg_norm") else 1e-6
                bad_alive += int(abs(float(X[ph - 1, i, fcol[name]]) - value / 4.0) > tol)
                n_alive += 1
            row = rows.get((tid, ph))
            if row is not None:
                pos_based_diff += int(abs(float(row.get("alive_players_start") or 0.0) - n_s) > 0.5 or abs(float(row.get("alive_players_end") or 0.0) - n_e) > 0.5)
    check("replaced alive_players_* columns == event-based member counts / 4", bad_alive == 0,
          f"{n_alive} cells, {bad_alive} wrong; the old position-based start/end counts differ on {pos_based_diff} team-phases")

    # 2. pipeline labels -------------------------------------------------------------------------
    cur.execute(
        """
        SELECT team_id, cut_phase, alive_at_phase_start, target_defined, hazard_next,
               final_rank_norm, final_top5, final_top10
          FROM stdb.team_phase_labels WHERE match_id = %s AND cut_phase BETWEEN 1 AND 8
        """,
        (mid,),
    )
    lab = cur.fetchall()
    nml, tm, yh = m["node_mask_label"].numpy(), m["target_mask"].numpy(), m["y_hazard"].numpy()
    bad = 0
    pipe_state: dict[tuple[int, int], int] = {}
    for tid, ph, alive0, tdef, haz, frn, t5, t10 in lab:
        if tid not in tindex:
            continue
        i, p = tindex[tid], ph - 1
        want_t = bool(tdef) and ph <= 7
        bad += int(bool(alive0) != bool(nml[p, i]))
        bad += int(want_t != bool(tm[p, i]))
        if want_t:
            bad += int(float(haz) != float(yh[p, i]))
        if ph <= 7:
            pipe_state[(tid, ph)] = state_of(want_t, float(haz or 0))
        if ph == 1:
            bad += int(abs(float(frn) - float(m["final_rank_norm"][i])) > 1e-6)
            bad += int(float(bool(t5)) != float(m["top5"][i]))
            bad += int(float(bool(t10)) != float(m["top10"][i]))
    check("pipeline labels identical to team_phase_labels (node_mask_label / target_mask / y_hazard / rank)", bad == 0, f"{len(lab)} rows, {bad} mismatches")
    check("target_mask is a subset of node_mask", not bool((m["target_mask"] & ~m["node_mask"]).any()))

    # 3. labels v2 -------------------------------------------------------------------------------
    bad_t = bad_state = bad_node = n_rows = 0
    agree = disagree = 0
    v2_state: dict[tuple[int, int], int] = {}
    for tid, i in tindex.items():
        T = team_t.get(tid)
        got_T = float(m["team_death_s"][i])
        if T is None:
            bad_t += int(not math.isnan(got_T))
        else:
            bad_t += int(not ((math.isinf(T) and math.isinf(got_T) and got_T > 0) or abs(T - got_T) < 1e-2))
        for ph, (a, b) in bnd.items():
            want_node = T is not None and T > a
            bad_node += int(want_node != bool(m["node_mask_v2"][ph - 1, i]))
        for k in range(1, P):
            if k not in bnd or (k + 1) not in bnd:
                want_tgt, want_y = False, 0.0
            else:
                e_k = bnd[k][1]
                e_n = INF if (k + 1) == last_phase else bnd[k + 1][1]
                want_tgt = T is not None and T > e_k
                want_y = float(want_tgt and math.isfinite(T) and T <= e_n)
            got_tgt = bool(m["target_mask_v2"][k - 1, i])
            got_y = float(m["y_hazard_v2"][k - 1, i])
            bad_state += int(want_tgt != got_tgt) + int(want_y != got_y)
            n_rows += 1
            v2_state[(tid, k)] = state_of(got_tgt, got_y)
            if (tid, k) in pipe_state and k in bnd and (k + 1) in bnd:
                agree += int(pipe_state[(tid, k)] == v2_state[(tid, k)])
                disagree += int(pipe_state[(tid, k)] != v2_state[(tid, k)])
    check("team_death_s == max over members of the last kill event (+inf alive, W clock)", bad_t == 0, f"{N} teams, {bad_t} wrong; match offset {off:.3f} s")
    check("target_mask_v2 / y_hazard_v2 == independent recomputation", bad_state == 0, f"{n_rows} team-cuts, {bad_state} wrong")
    check("node_mask_v2 == (team_death_s > phase_start_s)", bad_node == 0, f"{bad_node} wrong")
    check("(info) pipeline vs v2 on this match: identical / different team-cuts", True, f"{agree} / {disagree}")
    if AUDIT_LIB.exists():
        try:
            spec = importlib.util.spec_from_file_location("label_audit_lib", AUDIT_LIB)
            lib = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lib)
            r, tt, _mt, _pl = lib.build_rows(cur, [mid], {mid: "verify"})
            bad_a = 0
            for _, row in tt.iterrows():
                i = tindex.get(int(row.team_id))
                if i is None:
                    continue
                got = float(m["team_death_s"][i])
                want = float(row.T_evt)
                bad_a += int(not ((math.isinf(want) and math.isinf(got) and got > 0) or abs(want - got) < 1e-2))
            for _, row in r.iterrows():
                tid, k = int(row.team_id), int(row.k)
                if (tid, k) in v2_state:
                    bad_a += int(int(row.tr_evt) != v2_state[(tid, k)])
            check("labels v2 agree with the label audit's own code (T_evt, tr_evt)", bad_a == 0, f"{len(tt)} teams, {len(r)} team-cuts, {bad_a} mismatches")
        except Exception as exc:  # pragma: no cover - the cross-check is optional
            check("labels v2 cross-check with the audit lib", True, f"skipped ({exc!r})")

    # 4. edges -----------------------------------------------------------------------------------
    cur.execute(
        "SELECT phase_idx, edge_type, src_team_id, dst_team_id, weight FROM stdb.team_phase_edge_cache WHERE match_id = %s AND phase_idx BETWEEN 1 AND 8",
        (mid,),
    )
    db_edges: dict[tuple[str, int, int, int], float] = {}
    for ph, et, s, d, w in cur.fetchall():
        if s in tindex and d in tindex:
            db_edges[(D.EDGE_KEYS[et], ph - 1, tindex[s], tindex[d])] = float(w)
    dump_edges = {}
    for key, (idx, w) in m["edges"].items():
        for (p, i, j), ww in zip(idx.tolist(), w.tolist()):
            dump_edges[(key, p, i, j)] = ww
    same_keys = set(db_edges) == set(dump_edges)
    worst_w = max((abs(db_edges[k] - dump_edges[k]) for k in db_edges.keys() & dump_edges.keys()), default=0.0)
    check("edges equal the edge cache (raw weights, directions)", same_keys and worst_w < 1e-5,
          f"{len(db_edges)} db edges / {len(dump_edges)} dump edges, worst weight diff {worst_w:.2e}")
    combat = [(p, i, j) for (k, p, i, j) in dump_edges if k == "combat"]
    check("combat edges are directed (some pair without its reverse)", (not combat) or any((p, j, i) not in set(combat) for p, i, j in combat))

    # 5. clock + cut -----------------------------------------------------------------------------
    check("clock_delta_median_s == independent game-state median (event epoch - elapsed_time - started_at)",
          abs(float(m["clock_delta_median_s"]) - delta_med) < 1e-3,
          f"dump {float(m['clock_delta_median_s']):.3f} s, independent {delta_med:.3f} s")
    raised_by = delta - delta_med
    check("clock_delta_s >= the median (raised only where the clock drifts, in whole 0.25 s steps)",
          raised_by > -1e-3 and abs(raised_by / 0.25 - round(raised_by / 0.25)) < 1e-2,
          f"used {delta:.3f} s = median + {raised_by:.2f} s")
    ok_cut = True
    for ph, (a, b) in bnd.items():
        cut = math.floor((b - delta - 10.0) / 10.0 + 1e-9) * 10.0
        want = cut if cut >= a - delta - 1e-9 else float("nan")
        got = float(m["cut_t"][ph - 1])
        ok_cut &= (math.isnan(want) and math.isnan(got)) or abs(want - got) < 1e-6
        if not math.isnan(got):
            ok_cut &= abs(float(m["cut_wall_s"][ph - 1]) - (got + delta)) < 1e-2
            ok_cut &= float(m["cut_wall_s"][ph - 1]) <= b - 10.0 + 1e-3
        ok_cut &= abs(float(m["phase_end_s"][ph - 1]) - b) < 1e-3 and abs(float(m["phase_start_s"][ph - 1]) - a) < 1e-3
    check("cut_t == floor((phase_end_s - delta - 10) / 10) * 10; cut_wall_s == cut_t + delta <= phase_end_s - 10", ok_cut)

    # 6. team_pos from RAW positions -------------------------------------------------------------
    teams_raw = raw_positions(cur, mid, s_ep)
    exact = fallback = missing = bad_pos = 0
    for ph, (a, b) in bnd.items():
        t_c = float(m["cut_t"][ph - 1])
        if math.isnan(t_c):
            continue
        t_lo = a - delta
        for tid, i in tindex.items():
            got = m["team_pos"][ph - 1, i].numpy()
            gt, found, n_back = t_c, None, 0
            while gt >= t_lo - 1e-9:
                found = raw_team_centroid(teams_raw, tid, gt)
                if found is not None:
                    break
                gt -= 10.0
                n_back += 1
            if found is None:
                missing += 1
                bad_pos += int(not np.isnan(got).all())
                continue
            exact += int(n_back == 0)
            fallback += int(n_back > 0)
            bad_pos += int(np.abs(got - np.array(found) / size).max() > 2e-4)
    check("team_pos == centroid recomputed from RAW telem_positions at cut_t (or the previous grid point) / map size", bad_pos == 0,
          f"{exact} exact, {fallback} fallback, {missing} NaN, {bad_pos} wrong")

    # 7. zone ------------------------------------------------------------------------------------
    cur.execute(
        "SELECT region_kind, snapshot_t, ST_X(center_pos), ST_Y(center_pos), radius_m FROM stdb.dynamic_regions WHERE match_id = %s AND region_kind IN ('safe','poison')",
        (mid,),
    )
    dr = {(k, int(t)): (float(x), float(y), float(r)) for k, t, x, y, r in cur.fetchall()}
    bad_zone = 0
    for ph, (a, b) in bnd.items():
        t_c = float(m["cut_t"][ph - 1])
        row = m["zone"][ph - 1].numpy()
        if math.isnan(t_c):
            continue
        for off_c, kind in ((0, "safe"), (3, "poison")):
            v = dr.get((kind, int(t_c)))
            if v is None:
                bad_zone += int(not np.isnan(row[off_c : off_c + 3]).all())
            else:
                bad_zone += int(np.abs(row[off_c : off_c + 3] - np.array(v) / size).max() > 1e-4)
        bad_zone += int(abs(row[7] - ph) > 1e-6)
        want_frac = np.clip((t_c + delta - a) / max(b - a, 1.0), 0, 1)
        bad_zone += int(abs(row[6] - want_frac) > 1e-4)
    check("zone == dynamic_regions(safe, poison) at cut_t / map size, phase_no, phase_frac (W clock)", bad_zone == 0, f"{bad_zone} wrong")

    # 8. no look-ahead (raw proof) ---------------------------------------------------------------
    cur.execute(
        "SELECT phase::int, min(extract(epoch FROM event_time))::float8 FROM pubg.telem_phase_changes WHERE match_id = %s AND phase IS NOT NULL GROUP BY phase",
        (mid,),
    )
    start_w = {int(p): float(e) - s_ep for p, e in cur.fetchall()}
    n_checked = leaks = leaks_cap_only = 0
    margins = []
    for ph, (a, b) in bnd.items():
        t_c = float(m["cut_t"][ph - 1])
        nxt = start_w.get(ph + 1)
        if math.isnan(t_c) or nxt is None:
            continue
        n_checked += 1
        w_max = max_w_before(teams_raw, t_c + 5.0)  # latest raw row the grid point at cut_t may read
        margins.append(nxt - w_max)
        leaks += int(w_max >= nxt)
        old_cut = math.floor((b - 10.0) / 10.0 + 1e-9) * 10.0  # cap only, no clock shift
        if old_cut >= a:
            leaks_cap_only += int(max_w_before(teams_raw, old_cut + 5.0) >= nxt)
    check("no raw position row the capped grid may read lies at/after the true start of the next phase", leaks == 0,
          f"{n_checked} phase windows, {leaks} leaks, smallest margin {min(margins):.1f} s" if margins else "no next phase")
    check("(info) the cap-only rule (no clock shift) would have leaked in this many of the same windows", True, f"{leaks_cap_only} of {n_checked}")
    cur.execute(
        "SELECT gs.elapsed_time::float8, extract(epoch FROM gs.event_time)::float8 FROM pubg.telem_game_states gs "
        "WHERE gs.match_id = %s AND gs.elapsed_time IS NOT NULL",
        (mid,),
    )
    states = [(g_, w_ - s_ep) for g_, w_ in cur.fetchall()]
    n_gs = leaks_gs = leaks_gs_median = 0
    for ph, (a, b) in bnd.items():
        nxt = start_w.get(ph + 1)
        if nxt is None:
            continue
        n_gs += 1
        inside = [w_ for g_, w_ in states if g_ <= b - delta]
        leaks_gs += int(bool(inside) and max(inside) >= nxt)
        inside_med = [w_ for g_, w_ in states if g_ <= b - delta_med]
        leaks_gs_median += int(bool(inside_med) and max(inside_med) >= nxt)
    check("no game state inside the uncapped game-state window [., phase_end_s - delta] lies at/after the next phase start", leaks_gs == 0,
          f"{n_gs} phase windows, {leaks_gs} leaks (the plain median would leave {leaks_gs_median})")
    cur.execute(
        """
        SELECT s.team_id, s.phase_idx, s.position_rows,
               (SELECT count(DISTINCT p.elapsed_time)
                  FROM pubg.telem_positions p
                 WHERE p.match_id = s.match_id AND p.team_id = s.team_id
                   AND p.account_id IS NOT NULL AND p.is_game >= 0.1
                   AND p.pos_x IS NOT NULL AND p.pos_y IS NOT NULL AND p.elapsed_time IS NOT NULL
                   AND p.elapsed_time BETWEEN g.phase_start_s - %(d)s AND g.phase_end_s - %(d)s - 10)
          FROM stdb.team_phase_state s
          JOIN stdb.match_phase_global g ON g.match_id = s.match_id AND g.phase_idx = s.phase_idx
         WHERE s.match_id = %(m)s
        """,
        {"m": mid, "d": delta},
    )
    pr_rows = cur.fetchall()
    bad_rows = sum(1 for _t, _p, a, b in pr_rows if int(a) != int(b))
    check("team_phase_state.position_rows == raw position seconds in [start - delta, end - delta - 10]", bad_rows == 0,
          f"{len(pr_rows)} team-phases, {bad_rows} wrong")
    # a feature built from raw positions: mean_health_avg / min_health_min over the shifted + capped window
    cur.execute("SELECT team_id, phase_idx, mean_health_avg, min_health_min FROM stdb.team_phase_state WHERE match_id = %s", (mid,))
    bad_h = n_h = 0
    for tid, ph, mh_avg, mh_min in cur.fetchall():
        if ph not in bnd:
            continue
        a, b = bnd[ph]
        lo, hi = a - delta, b - delta - 10.0
        per_sec: dict[int, list[float]] = {}
        for _ts, rows_ in teams_raw.get(int(tid), []):
            for t, h, _x, _y, _w in rows_:
                if lo <= t <= hi and h is not None and h > 0:
                    per_sec.setdefault(int(round(t)), []).append(h)
        n_h += 1
        if per_sec:
            want_avg = sum(sum(v) / len(v) for v in per_sec.values()) / len(per_sec)
            want_min = min(min(v) for v in per_sec.values())
            bad_h += int(mh_avg is None or abs(float(mh_avg) - want_avg) > 1e-3 * (1 + abs(want_avg))
                         or mh_min is None or abs(float(mh_min) - want_min) > 1e-3 * (1 + abs(want_min)))
        else:
            bad_h += int(mh_avg is not None)
    check("team_phase_state.mean_health_avg / min_health_min == recomputed from raw rows of the shifted + capped window", bad_h == 0,
          f"{n_h} team-phases, {bad_h} wrong")
    cur.execute("SELECT team_id, snapshot_t FROM stdb.team_snapshot_features WHERE match_id = %s", (mid,))
    snaps: dict[int, list[int]] = {}
    for tid, t in cur.fetchall():
        snaps.setdefault(int(tid), []).append(int(t))
    cur.execute("SELECT team_id, phase_idx, waypoints FROM stdb.team_phase_trajectory WHERE match_id = %s", (mid,))
    wrong_wp = n_wp = 0
    for tid, ph, wp in cur.fetchall():
        if ph not in bnd:
            continue
        a, b = bnd[ph]
        want = sum(1 for t in snaps.get(int(tid), []) if a - delta <= t <= b - delta - 10.0)
        n_wp += 1
        wrong_wp += int(int(wp) != want)
    check("team_phase_trajectory.waypoints == alive grid rows in [start - delta, end - delta - 10]", wrong_wp == 0, f"{n_wp} team-phases, {wrong_wp} wrong")
    if fixb_since:
        stale = D.stale_fixb_tables(cur, mid, fixb_since)
        check("derived rows were rebuilt after the one-clock marker", not stale, f"stale tables: {stale}")

    # 9. audit fields ----------------------------------------------------------------------------
    cur.execute("SELECT team_id, won FROM stdb.teams WHERE match_id = %s", (mid,))
    won = {int(t): bool(w) for t, w in cur.fetchall()}
    check("team_won == stdb.teams.won", all(bool(m["team_won"][tindex[t]]) == won[t] for t in tindex if t in won))
    cur.execute(
        """
        SELECT r.team_id, avg((p.player_id LIKE 'ai.%%')::int)::float8
          FROM pubg.participants p JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
         WHERE p.match_id = %s AND r.team_id IS NOT NULL GROUP BY r.team_id
        """,
        (mid,),
    )
    bad_a = 0
    for tid, aif in cur.fetchall():
        i = tindex.get(int(tid))
        if i is not None:
            bad_a += int(abs(float(m["team_ai_frac"][i]) - float(aif)) > 1e-6)
    cur.execute(
        """
        SELECT r.team_id, sum(d.damage)::float8
          FROM pubg.telem_damage d
          JOIN pubg.participants p ON p.match_id = d.match_id AND p.player_id = d.victim_id
          JOIN pubg.rosters r ON r.match_id = p.match_id AND r.roster_id = p.roster_id
         WHERE d.match_id = %s AND d.attacker_id LIKE 'Monster.%%' GROUP BY r.team_id
        """,
        (mid,),
    )
    monster = {int(t): float(v) for t, v in cur.fetchall()}
    for tid, i in tindex.items():
        bad_a += int(abs(float(m["team_has_monster_dmg"][i]) - monster.get(tid, 0.0)) > 1e-3)
    cur.execute("SELECT match_type FROM pubg.matches WHERE match_id = %s", (mid,))
    bad_a += int(cur.fetchone()[0] != m["match_type"])
    check("team_ai_frac / team_has_monster_dmg / match_type equal an independent SQL recomputation", bad_a == 0, f"{bad_a} mismatches")

    if show:
        print(f"  clock delta {delta:.2f} s, last phase {last_phase}, phase bounds (W): " + ", ".join(f"p{p}=({a:.0f},{b:.0f})" for p, (a, b) in bnd.items()))
        print("  team  n  ai   T(W s)   death phase  pipeline states k=1..7 | v2 states k=1..7   (0 outside, 1 neg, 2 pos)")
        for tid, i in tindex.items():
            T = float(m["team_death_s"][i])
            dph = next((p for p, (a, b) in sorted(bnd.items()) if T <= b), None) if math.isfinite(T) else None
            ps = "".join(str(pipe_state.get((tid, k), "-")) for k in range(1, 8))
            vs = "".join(str(v2_state.get((tid, k), "-")) for k in range(1, 8))
            print(f"  {tid:4d} {len(members_by_team.get(tid, [])):2d} {float(m['team_ai_frac'][i]):.2f} {T:8.1f}  {str(dph):>11s}  {ps} | {vs}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--match-id", action="append")
    ap.add_argument("--map-name", default=None, help="needed with --match-id when it cannot be inferred")
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--fixb-since", default=None)
    ap.add_argument("--show", action="store_true", help="print a per-team label table")
    args = ap.parse_args()
    if not args.match_id and not args.dump:
        ap.error("pass --match-id or --dump")

    since = args.fixb_since
    if since is None:
        marker = Path("model_stgcn/etl_runs/fixb_2026-10-05/fixb_since.txt")
        since = marker.read_text(encoding="utf-8").strip() if marker.exists() else None

    failed = 0
    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            entries = []
            if args.dump:
                dump = torch.load(args.dump, map_location="cpu", weights_only=True)
                if args.match_id:  # pick the entries of the written dump by match id
                    wanted = set(args.match_id)
                    entries += [(m, dump["meta"]["map_name"]) for m in dump["matches"] if m["match_id"] in wanted]
                    if len(entries) != len(wanted):
                        print(f"warning: found {len(entries)} of {len(wanted)} requested match ids in {args.dump}")
                else:
                    entries.append((dump["matches"][args.index], dump["meta"]["map_name"]))
            else:
                for mid in args.match_id:
                    cur.execute("SELECT map_name FROM stdb.matches WHERE match_id = %s", (mid,))
                    map_name = args.map_name or cur.fetchone()[0]
                    raw = D.fetch_raw_match(cur, mid, map_name)
                    if raw is None:
                        print(f"match {mid} is not phase-ready")
                        failed += 1
                        continue
                    item = D.assemble_match(raw, map_size_m=D.MAP_SIZE_M[map_name])
                    item.pop("_stats", None)
                    entries.append((item, map_name))
            for m, map_name in entries:
                results = verify(cur, m, map_name, since, show=args.show)
                print(f"match {m['match_id']}  map {map_name}  teams {len(m['team_ids'])}  type {m['match_type']}")
                for name, ok, detail in results:
                    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
                failed += int(not all(ok for _, ok, _ in results))
    finally:
        conn.close()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
