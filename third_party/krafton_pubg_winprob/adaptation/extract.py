"""Extraction of KRAFTON-style squad samples from the raw ``pubg.*`` tables (READ-ONLY database access).

    python -m adaptation.extract --cohorts erangel_tpp_squad --splits train,val,test --workers 3 \\
        [--train-mode krafton192|every30] [--limit-matches 20] [--out-dir model_stgcn/dumps/krafton_winprob_v2]

For every match of a cohort split the script fetches the raw telemetry rows (SELECT only; the connection is a
read-only session), puts them on the unified clock W, and writes shards ``<out>/<cohort>/shards/<split>/*.npz``
with one row per (sample time, squad row):

* train: KRAFTON's sampling schedule (178 uniform-random points over its fixed phase schedule + 14 anchors,
  extracting_pgc_features.py), seeded per match; points before the first game state or with fewer than two
  risk-set squads are dropped (the weighted Cox loss ignores them anyway).  ``--train-mode every30`` = every 30 s.
* val / test: the phase cuts of the dump's v2 risk set, t = a_k = b_k - 10 s on W, for every cut with at least one
  risk-set row.

v2 labels (a ``dump_version == 'v2'`` label table; the dump is the single source of truth, task rules 1-3 and 5):

* W is computed in SQL (``extract(epoch FROM (event_time - stdb.matches.started_at))``, the producer's clock, bitwise);
* every feature at ``t`` reads records strictly before ``t`` and the squad alive status is the causal one of
  ``model_stgcn.phase_v2.alive`` (``raw['life']``); the squad's key flag ``KEY`` is that status (at a cut it must equal the
  dump's ``node_mask`` exactly: ``n_alive_mismatch``);
* event times are the dump's ``T`` (``team_death_s``: final elimination, winners ``+inf``), censoring at
  ``max(2000, C_m = match_end_s, max finite T + 1)`` (KRAFTON's convention with the dump's ``C_m``); nothing is
  re-derived from kill events / ``time_survived`` (the v1 rule and its winner override are gone for v2);
* rows of a sample = the squads that are keys (``KEY``) or at risk (``RISK = T > t``); at a cut they are the dump's
  ``query_mask`` rows, so a *comeback* squad (risk set but causally dead) is a row without being a key.  ``RISK`` rows are
  the Cox loss rows; only ``KEY`` rows are attention keys.  Shards carry the per-row ``KEY`` / ``RISK`` flags.

No credential is read from or written to any file here: the connection parameters come from the environment
(PG_HOST, PG_PORT, PG_DBNAME / PG_DATABASE, PG_USER, PG_PASSWORD) exactly as in reports/bigcomp2027_data_profile.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one numpy thread per worker (resource cap: <= 3 extraction workers)

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptation import schema as S  # noqa: E402
from adaptation.features import compute_features, member_alive_causal, member_death_times, member_life, team_alive_causal, _as_idx  # noqa: E402
from adaptation.labels import check_label_version, load_labels  # noqa: E402

REPO = ROOT.parents[1]
DEFAULT_OUT = REPO / "model_stgcn/dumps/krafton_winprob_v2"
SEED = 42
BATCH = 8  # matches per database round trip

_CONN = None


def get_conn():
    """Read-only session (no autocommit write possible); parameters from the environment only."""
    global _CONN
    if _CONN is None:
        import psycopg2
        import psycopg2.extensions

        psycopg2.extensions.register_type(
            psycopg2.extensions.new_type(
                psycopg2.extensions.DECIMAL.values, "DEC2FLOAT", lambda v, c: float(v) if v is not None else None
            )
        )
        g = lambda *names, default=None: next((os.environ[n] for n in names if os.environ.get(n)), default)  # noqa: E731
        c = psycopg2.connect(
            host=g("PG_HOST", "PGHOST", default="localhost"),
            port=int(g("PG_PORT", "PGPORT", default="5432")),
            dbname=g("PG_DBNAME", "PG_DATABASE", "PGDATABASE", default="pubg_survival"),
            user=g("PG_USER", "PGUSER", default="postgres"),
            password=g("PG_PASSWORD", "PGPASSWORD"),
        )
        c.set_session(readonly=True, autocommit=True)
        cur = c.cursor()
        cur.execute("SET statement_timeout='300s'")
        cur.close()
        _CONN = c
    return _CONN


EQUIP_IDS = sorted(set(S.HELMET_ITEMS) | set(S.VEST_ITEMS) | set(S.BACKPACK_ITEMS))
USE_IDS = sorted(S.AIDS_ITEMS)


def _w(alias: str) -> str:
    """The producer's clock: seconds since LogMatchStart, computed by the database (``stdb.matches`` row ``m``)."""
    return f"extract(epoch FROM ({alias}.event_time - m.started_at))::float8 AS w"


_JOIN = "JOIN stdb.matches m ON m.match_id = {a}.match_id"
SQL = dict(
    meta="""SELECT m.match_id, extract(epoch FROM m.started_at)::float8 AS started_ep, m.duration_s, m.map_name,
                   (SELECT extract(epoch FROM min(e.event_time))::float8 FROM pubg.telem_match_end e WHERE e.match_id = m.match_id) AS end_ep
              FROM stdb.matches m WHERE m.match_id = ANY(%s)""",
    part="""SELECT p.match_id, p.player_id, r.team_id, p.death_type, p.time_survived::float8 AS ts
              FROM pubg.participants p JOIN pubg.rosters r ON r.roster_id = p.roster_id AND r.match_id = p.match_id
             WHERE p.match_id = ANY(%s) AND r.team_id IS NOT NULL ORDER BY p.match_id, r.team_id, p.player_id""",
    kill=f"""SELECT k.match_id, k.killer_id, k.victim_id, k.finisher_id, extract(epoch FROM k.event_time)::float8 AS ep, {_w('k')}
               FROM pubg.telem_kills k {_JOIN.format(a='k')} WHERE k.match_id = ANY(%s)""",
    grg=f"""SELECT g.match_id, g.attacker_id, g.victim_id, extract(epoch FROM g.event_time)::float8 AS ep, {_w('g')}
              FROM pubg.telem_groggy g {_JOIN.format(a='g')} WHERE g.match_id = ANY(%s)""",
    dmg=f"""SELECT d.match_id, d.attacker_id, d.victim_id, d.damage::float8 AS dmg, d.damage_causer, extract(epoch FROM d.event_time)::float8 AS ep, {_w('d')}
              FROM pubg.telem_damage d {_JOIN.format(a='d')} WHERE d.match_id = ANY(%s)""",
    pos=f"""SELECT p.match_id, p.account_id AS acc, p.health::float8 AS health, p.pos_x::float8 AS x, p.pos_y::float8 AS y, p.pos_z::float8 AS z,
                  p.vehicle_type AS veh, extract(epoch FROM p.event_time)::float8 AS ep, {_w('p')}
             FROM pubg.telem_positions p {_JOIN.format(a='p')}
            WHERE p.match_id = ANY(%s) AND p.is_game >= 0.1 AND p.account_id IS NOT NULL AND p.team_id IS NOT NULL
              AND p.elapsed_time IS NOT NULL AND p.pos_x IS NOT NULL AND p.pos_y IS NOT NULL""",
    gs=f"""SELECT s.match_id, extract(epoch FROM s.event_time)::float8 AS ep, {_w('s')}, s.safe_zone_x::float8 AS sx, s.safe_zone_y::float8 AS sy,
                 s.safe_zone_radius::float8 AS sr, s.poison_zone_x::float8 AS px, s.poison_zone_y::float8 AS py, s.poison_zone_radius::float8 AS pr
            FROM pubg.telem_game_states s {_JOIN.format(a='s')} WHERE s.match_id = ANY(%s) AND s.is_game >= 0.1 ORDER BY s.match_id, s.event_time""",
    pick=f"""SELECT i.match_id, i.account_id AS acc, i.event_type, i.item_id, i.sub_category, i.stack_count, extract(epoch FROM i.event_time)::float8 AS ep, {_w('i')}
              FROM pubg.telem_item_pickup i {_JOIN.format(a='i')} WHERE i.match_id = ANY(%s) AND i.event_type IN ('pickup', 'drop') AND i.account_id IS NOT NULL""",
    equip=f"""SELECT i.match_id, i.account_id AS acc, i.item_id, extract(epoch FROM i.event_time)::float8 AS ep, {_w('i')}
               FROM pubg.telem_item_equip i {_JOIN.format(a='i')} WHERE i.match_id = ANY(%s) AND i.event_type = 'equip' AND i.item_id = ANY(%s)""",
    use=f"""SELECT i.match_id, i.account_id AS acc, i.item_id, extract(epoch FROM i.event_time)::float8 AS ep, {_w('i')}
             FROM pubg.telem_item_use i {_JOIN.format(a='i')} WHERE i.match_id = ANY(%s) AND i.item_id = ANY(%s)""",
)
PARAMS = {"equip": EQUIP_IDS, "use": USE_IDS}


def fetch_batch(mids: list[str]) -> dict[str, pd.DataFrame]:
    cur = get_conn().cursor()
    out = {}
    for name, sql in SQL.items():
        params = (mids,) if name not in PARAMS else (mids, PARAMS[name])
        cur.execute(sql, params)
        out[name] = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    cur.close()
    return out


def _by_match(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    return {k: g for k, g in df.groupby("match_id", sort=False)}


_WCLASS: dict[str, str] = {}


def weapon_class(causer) -> str:
    if not isinstance(causer, str):
        return ""
    r = _WCLASS.get(causer)
    if r is None:
        if any(p in causer for p in S.EXCLUDED_CAUSER_PATTERNS):
            r = ""
        else:
            r = S.WEAPON_TO_CLASS.get(causer.split("_")[0].replace("Weap", ""), "")
        _WCLASS[causer] = r
    return r


# --------------------------------------------------------------------------- sampling
def krafton_times(match_id: str, seed: int = SEED) -> np.ndarray:
    """178 uniform-random points over upstream's phase schedule + the 14 boundary anchors (192 sorted points)."""
    rng = np.random.RandomState((zlib.crc32(match_id.encode()) ^ seed) & 0xFFFFFFFF)
    parts = []
    for _ph, (a, b, n) in S.PHASE_CONFIG.items():
        tp = rng.uniform(a, b, n)
        anchors = [t for t in S.PHASE_BOUNDARY_POINTS if a <= t < b]
        parts.append(np.concatenate([tp, np.asarray(anchors, dtype=np.float64)]))
    ts = np.sort(np.concatenate(parts))
    assert len(ts) == 192
    return ts


def every_times(t_end: float, step: float = 30.0) -> np.ndarray:
    return np.arange(step, t_end + 1e-9, step)


# --------------------------------------------------------------------------- per match
def truncate_raw(raw: dict, t: float) -> dict:
    """The raw rows that exist at ``t``: a v2 ``raw`` keeps the records with time ``< t`` (strict) and rebuilds the causal member
    timelines from them; a v1 ``raw`` keeps ``<= t`` (the legacy convention).  Used by the prefix-replay checks: the features at
    ``t`` of the truncated copy must equal those of the full data."""
    v2 = "life" in raw
    out = dict(raw)
    for key in ("kills", "groggy", "damage", "pos", "pick", "equip", "use", "zone"):
        sub = raw[key]
        m = (sub["t"] < t) if v2 else (sub["t"] <= t)
        if v2 and key == "kills":  # the strict mask must use the unclamped event time (the one the timelines use)
            m = sub["w_raw"] < t
        out[key] = {k: v[m] for k, v in sub.items()}
    if v2:
        k = out["kills"]
        out["life"] = member_life(k["w_raw"], k["victim_pidx"], out["pos"]["t"], out["pos"]["pidx"], out["pos"]["health"])
    return out


V2_LABEL_KEYS = ("node_mask", "query_mask", "cut_t", "match_end_s")


def _is_v2(lab: dict) -> bool:
    """A compact label of a v2 dump (``dump_version == 'v2'`` or a v2 key present, see ``adaptation.labels.compact_match``).

    A v2 label that lacks one of :data:`V2_LABEL_KEYS` is refused: nothing is rebuilt from ``team_death_s`` for a v2 dump (the
    dump's masks are the single source of truth; the death-time rule of the v1 path is the look-ahead the v2 dump removes)."""
    v2 = lab.get("dump_version") == "v2" or any(k in lab for k in ("query_mask", "node_mask", "cut_t"))
    if v2:
        from model_stgcn.ptr.data import DumpVersionError

        missing = [k for k in V2_LABEL_KEYS if k not in lab]
        if missing:
            raise DumpVersionError(f"{lab.get('match_id', '?')}: a v2 label without {missing}: the v1 death-time rule is not applied to v2 data")
    return v2


def build_raw(mid: str, lab: dict, T: dict, meta_row, started_ep: float) -> tuple[dict, dict]:
    """Numpy view of the raw rows of one match on the unified clock, plus bookkeeping flags.

    v2 labels: times are the database's ``w`` (the producer's clock, exactly), ``raw['life']`` holds the causal member
    timelines (``raw['d']`` does not exist) and the dump's death times / masks are not rebuilt.  v1 labels: the legacy path
    (death times from kill events / ``time_survived``, compared with the dump at 5 ms)."""
    v2 = _is_v2(lab)
    flags: dict = {}
    part = T["part"].get(mid)
    if part is None or len(part) == 0:
        raise ValueError("no participants")
    team_index = {int(t): i for i, t in enumerate(lab["team_ids"])}
    part = part[part.team_id.isin(team_index)].reset_index(drop=True)
    pid = part.player_id.astype(str).to_numpy()
    P = len(pid)
    pid2idx = {p: i for i, p in enumerate(pid)}
    team = np.array([team_index[int(t)] for t in part.team_id], dtype=np.int64)
    is_ai = np.array([p.startswith("ai.") for p in pid])

    def when(df):  # event times on W: v2 = the database's own w (bitwise the producer's), v1 = epoch - started_at
        if v2:
            return df["w"].to_numpy(dtype=np.float64)
        return df.ep.to_numpy(dtype=np.float64) - started_ep

    kill = T["kill"].get(mid)
    kill = kill if kill is not None else pd.DataFrame(columns=["killer_id", "victim_id", "finisher_id", "ep", "w"])
    kw_raw = when(kill)
    kt = np.maximum(kw_raw, 0.0)
    v_idx = _as_idx(kill.victim_id.tolist(), pid2idx)
    N = len(lab["team_ids"])
    if v2:
        d, off = None, float("nan")
        flags["n_death_mismatch"] = 0  # the dump's T is used as it is: nothing to compare (kept for the QA log schema)
        flags["n_winner_fix"] = 0
        dump_td = lab["team_death_s"]
    else:
        last_kill = np.full(P, np.nan)
        for i, t in zip(v_idx, kt):  # last victim kill event per player (the dump's team_death_s rule)
            if i >= 0 and (np.isnan(last_kill[i]) or t > last_kill[i]):
                last_kill[i] = t
        d, off = member_death_times(part.death_type.tolist(), part.ts.to_numpy(dtype=np.float64), last_kill, is_ai)

        # death-time consistency with the dump (labels v2) and winner handling
        td = np.full(N, np.nan)
        for s in range(N):
            m = d[team == s]
            if m.size:
                td[s] = np.inf if np.isinf(m).any() else m.max()
        dump_td = lab["team_death_s"]
        both = np.isfinite(td) & np.isfinite(dump_td)
        with np.errstate(invalid="ignore"):
            mism = int(((np.abs(td - dump_td) > 5e-3) & both).sum() + (np.isinf(td) != np.isinf(dump_td)).sum() + (np.isnan(td) != np.isnan(dump_td)).sum())
        flags["n_death_mismatch"] = mism
        won = lab["team_won"]
        fix = 0
        for s in np.flatnonzero(won):  # a winner whose members all have finite death times keeps its latest member alive
            m = np.flatnonzero(team == s)
            if m.size and not np.isinf(d[m]).any():
                d[m[np.argmax(d[m])]] = np.inf
                fix += 1
        flags["n_winner_fix"] = fix

    slot = np.full((N, 4), -1, dtype=np.int64)
    for s in range(N):
        mem = np.flatnonzero(team == s)[:4]
        slot[s, : len(mem)] = mem

    def W(df):
        return when(df)

    killer = _as_idx(kill.killer_id.tolist(), pid2idx)
    finisher = _as_idx(kill.finisher_id.tolist(), pid2idx)
    grg = T["grg"].get(mid)
    grg = grg if grg is not None else pd.DataFrame(columns=["attacker_id", "victim_id", "ep", "w"])
    dmg = T["dmg"].get(mid)
    dmg = dmg if dmg is not None else pd.DataFrame(columns=["attacker_id", "victim_id", "dmg", "damage_causer", "ep", "w"])
    dmg_ok = dmg.dmg.notna().to_numpy()
    dmg = dmg[dmg_ok]
    pos = T["pos"].get(mid)
    if pos is None or len(pos) == 0:
        raise ValueError("no position rows")
    pidx = _as_idx(pos.acc.tolist(), pid2idx)
    pm = pidx >= 0
    vt = pos.veh.to_numpy(dtype=object)
    in_veh = np.array([isinstance(v, str) and v != "TransportAircraft" for v in vt])
    gs = T["gs"].get(mid)
    if gs is None or len(gs) == 0:
        raise ValueError("no game states")
    gs = gs.sort_values("w" if v2 else "ep")
    pick = T["pick"].get(mid)
    pick = pick if pick is not None else pd.DataFrame(columns=["acc", "event_type", "item_id", "sub_category", "stack_count", "ep", "w"])
    equip = T["equip"].get(mid)
    equip = equip if equip is not None else pd.DataFrame(columns=["acc", "item_id", "ep", "w"])
    use = T["use"].get(mid)
    use = use if use is not None else pd.DataFrame(columns=["acc", "item_id", "ep", "w"])
    item = pick.item_id.astype(str).to_numpy()

    raw = {
        "pid": pid, "team": team, "is_ai": is_ai, "N": N, "slot": slot,
        "kills": {"killer_pidx": killer, "finisher_pidx": finisher, "t": kt},
        "groggy": {"attacker_pidx": _as_idx(grg.attacker_id.tolist(), pid2idx), "victim_pidx": _as_idx(grg.victim_id.tolist(), pid2idx),
                   "t": np.maximum(W(grg), 0.0)},
        "damage": {"attacker_pidx": _as_idx(dmg.attacker_id.tolist(), pid2idx), "victim_pidx": _as_idx(dmg.victim_id.tolist(), pid2idx),
                   "dmg": dmg.dmg.to_numpy(dtype=np.float64), "t": np.maximum(W(dmg), 0.0),
                   "wclass": np.array([weapon_class(c) for c in dmg.damage_causer.tolist()], dtype=object)},
        "pos": {"pidx": pidx[pm], "t": W(pos)[pm], "health": pos.health.to_numpy(dtype=np.float64)[pm],
                "x": pos.x.to_numpy(dtype=np.float64)[pm], "y": pos.y.to_numpy(dtype=np.float64)[pm],
                "z": pos.z.to_numpy(dtype=np.float64)[pm], "veh": in_veh[pm]},
        "zone": {"t": W(gs), "sx": gs.sx.to_numpy(dtype=np.float64), "sy": gs.sy.to_numpy(dtype=np.float64),
                 "sr": gs.sr.to_numpy(dtype=np.float64), "px": gs.px.to_numpy(dtype=np.float64),
                 "py": gs.py.to_numpy(dtype=np.float64), "pr": gs.pr.to_numpy(dtype=np.float64)},
        "pick": {"pidx": _as_idx(pick.acc.tolist(), pid2idx), "t": np.maximum(W(pick), 0.0),
                 "kind": (pick.event_type.to_numpy() == "drop").astype(np.int8), "item_id": item,
                 "stack": pick.stack_count.fillna(0).to_numpy(dtype=np.float64),
                 "removed": pick.sub_category.isin(S.REMOVED_SUBCATEGORIES).to_numpy(),
                 "parachute": np.array([S.START_PARACHUTE in x for x in item], dtype=bool)},
        "equip": {"pidx": _as_idx(equip.acc.tolist(), pid2idx), "t": np.maximum(W(equip), 0.0), "item_id": equip.item_id.astype(str).to_numpy()},
        "use": {"pidx": _as_idx(use.acc.tolist(), pid2idx), "t": np.maximum(W(use), 0.0), "item_id": use.item_id.astype(str).to_numpy()},
    }  # fmt: skip
    if v2:  # causal member timelines (records on W, the same rule and rows as the dump): kills + healthy position samples
        raw["kills"]["victim_pidx"] = v_idx  # kept (with the unclamped times) so that a truncated copy can rebuild the timelines
        raw["kills"]["w_raw"] = kw_raw
        raw["life"] = member_life(kw_raw, v_idx, raw["pos"]["t"], raw["pos"]["pidx"], raw["pos"]["health"])
    else:
        raw["d"] = d
    end_w = None
    if v2:
        mend = lab.get("match_end_s")
        end_w = None if mend is None or not np.isfinite(mend) else float(mend)  # C_m of the dump
    elif meta_row is not None and meta_row.end_ep is not None and np.isfinite(meta_row.end_ep):
        end_w = float(meta_row.end_ep - started_ep)
    flags["n_players"] = int(P)
    flags["n_pos_rows"] = int(pm.sum())
    flags["first_state_w"] = float(raw["zone"]["t"][0])
    return raw, {"flags": flags, "end_w": end_w, "off": off, "dump_td": dump_td}


def _match_samples_v2(mid: str, lab: dict, raw: dict, extra: dict, split: str, mode: str):
    """v2 samples of one match (see the module docstring): rows = key squads or risk-set squads, per-row ``KEY`` / ``RISK``."""
    T = np.asarray(lab["team_death_s"], dtype=np.float64)  # the dump's T: final elimination, winners / survivors +inf
    won = np.asarray(lab["team_won"], dtype=bool)
    if won.any() and not np.isposinf(T[won]).all():
        raise ValueError(f"{mid}: a v2 dump censors the winners (T = +inf); got a finite / unknown T for a winner")
    node, query, target = lab["node_mask"], lab["query_mask"], lab["target"]
    P = node.shape[0]
    first_state = extra["flags"]["first_state_w"]
    if split == "train":
        if mode == "krafton192":
            ts = krafton_times(mid)
        else:
            end = np.nanmax(np.where(np.isfinite(T), T, np.nan)) if np.isfinite(T).any() else 2000.0
            ts = every_times(float(end))
        cut = np.zeros(len(ts), dtype=np.int8)
        slots: list[int] = []
    else:
        slots = [p for p in range(P) if target[p].any()]
        ts = np.asarray(lab["cut_t"], dtype=np.float64)[slots]  # a_k = b_k - 10 s, exactly the producer's cut
        cut = np.array([p + 1 for p in slots], dtype=np.int8)
    alive_m = member_alive_causal(raw["life"], len(raw["pid"]), ts)  # [P, K] members, records < t only (evaluated once)
    key_all = team_alive_causal(raw, ts, alive_m)  # [N, K]: causally alive squads
    risk_all = T[:, None] > ts[None, :]  # [N, K]: the Cox risk set at t (T > t)
    if split == "train":
        rows_all = key_all | risk_all  # keys + risk rows (a comeback squad is a risk row without being a key)
        keep = (ts >= first_state) & (risk_all.sum(0) >= 2) & (key_all.sum(0) >= 1)
    else:
        mism = int((key_all != node[slots].T).sum())  # the causal status at a cut IS the dump's node_mask
        extra["flags"]["n_alive_mismatch"] = mism
        if mism:
            bad = np.argwhere(key_all != node[slots].T)[:3].tolist()
            raise ValueError(f"{mid}: the causal alive status differs from the dump's node_mask in {mism} (squad, cut) cells, e.g. {bad}")
        key_all = node[slots].T
        rows_all = query[slots].T  # the dump's token set at the cut: node_mask | target (comeback rows included)
        keep = ts >= first_state  # every cut with a risk-set row is evaluated, whatever the sample size
    n_dropped = int((~keep).sum())
    if split != "train" and n_dropped:
        raise ValueError(f"{n_dropped} evaluation cut(s) fall outside the observable range (first state {first_state:.1f}s)")
    ts, cut, key_all, risk_all, rows_all = ts[keep], cut[keep], key_all[:, keep], risk_all[:, keep], rows_all[:, keep]
    if len(ts) == 0:
        return None, n_dropped
    out = compute_features(raw, ts, alive_m[:, keep])
    # target time of the weighted Cox loss: winners / survivors censored at max(2000 s, C_m, latest death + 1 s)
    fin = T[np.isfinite(T)]
    c_m = extra["end_w"] if extra["end_w"] is not None else 0.0
    t_win = max(2000.0, float(c_m), (float(fin.max()) + 1.0) if fin.size else 0.0)
    tgt = np.where(np.isinf(T), t_win, T).astype(np.float32)
    rows_X, rows_P, rows_T, rows_i, rows_key, rows_risk, n_per = [], [], [], [], [], [], []
    for k in range(len(ts)):
        s_idx = np.flatnonzero(rows_all[:, k])
        rows_X.append(out["F"][s_idx, k, :])
        rows_P.append(out["POS"][s_idx, k, :, :])
        rows_T.append(tgt[s_idx])
        rows_i.append(s_idx.astype(np.int16))
        rows_key.append(key_all[s_idx, k])
        rows_risk.append(risk_all[s_idx, k])
        n_per.append(len(s_idx))
    res = {
        "X": np.concatenate(rows_X, axis=0), "POS": np.concatenate(rows_P, axis=0), "TGT": np.concatenate(rows_T),
        "TIDX": np.concatenate(rows_i), "KEY": np.concatenate(rows_key), "RISK": np.concatenate(rows_risk),
        "S_t": ts.astype(np.float32), "S_cut": cut, "S_n": np.asarray(n_per, dtype=np.int16),
        "S_zone": out["Zt"], "t_win": t_win,
    }  # fmt: skip
    res["n_rows_no_pos"] = int((np.isnan(res["POS"][:, :, 0]).all(axis=1)).sum())
    return res, n_dropped


def match_samples(mid: str, lab: dict, raw: dict, extra: dict, split: str, mode: str):
    """Sample times + features for one match; returns the row arrays of its samples."""
    if _is_v2(lab):
        return _match_samples_v2(mid, lab, raw, extra, split, mode)
    team_death = lab["team_death_s"]
    T_eff = np.where(lab["team_won"], np.inf, team_death)  # winner censoring, as model_stgcn.ptr.data
    first_state = extra["flags"]["first_state_w"]
    if split == "train":
        if mode == "krafton192":
            ts = krafton_times(mid)
        else:
            end = np.nanmax(np.where(np.isfinite(T_eff), T_eff, np.nan)) if np.isfinite(T_eff).any() else 2000.0
            ts = every_times(float(end))
        cut = np.zeros(len(ts), dtype=np.int8)
    else:
        slots = [p for p in range(len(lab["phase_end_s"])) if lab["target"][p].any()]
        ts = np.array([lab["phase_end_s"][p] - 10.0 for p in slots], dtype=np.float64)
        cut = np.array([p + 1 for p in slots], dtype=np.int8)
    keep = (ts >= first_state) & ((T_eff[:, None] > ts[None, :]).sum(0) >= 2)
    n_dropped = int((~keep).sum())
    if split != "train" and n_dropped:
        raise ValueError(f"{n_dropped} evaluation cut(s) fall outside the observable range (first state {first_state:.1f}s)")
    ts, cut = ts[keep], cut[keep]
    if len(ts) == 0:
        return None, n_dropped
    out = compute_features(raw, ts)
    # target time of the weighted Cox loss: winners / survivors censored at max(2000 s, match end), their convention
    fin = T_eff[np.isfinite(T_eff)]
    t_win = max(2000.0, (extra["end_w"] or 0.0), (float(fin.max()) + 1.0) if fin.size else 0.0)
    tgt = np.where(np.isinf(T_eff), t_win, T_eff).astype(np.float32)
    alive_sq = T_eff[:, None] > ts[None, :]  # [N, K]
    rows_X, rows_P, rows_T, rows_i, n_per = [], [], [], [], []
    for k in range(len(ts)):
        s_idx = np.flatnonzero(alive_sq[:, k])
        rows_X.append(out["F"][s_idx, k, :])
        rows_P.append(out["POS"][s_idx, k, :, :])
        rows_T.append(tgt[s_idx])
        rows_i.append(s_idx.astype(np.int16))
        n_per.append(len(s_idx))
    res = {
        "X": np.concatenate(rows_X, axis=0), "POS": np.concatenate(rows_P, axis=0), "TGT": np.concatenate(rows_T),
        "TIDX": np.concatenate(rows_i), "S_t": ts.astype(np.float32), "S_cut": cut, "S_n": np.asarray(n_per, dtype=np.int16),
        "S_zone": out["Zt"], "t_win": t_win,
    }  # fmt: skip
    res["n_rows_no_pos"] = int((np.isnan(res["POS"][:, :, 0]).all(axis=1)).sum())
    return res, n_dropped


def run_job(job: dict) -> dict:
    """Worker: fetch a batch of matches, write one shard, return per-match QA stats."""
    t0 = time.time()
    cohort, split, bidx, labs, mode, out_dir = job["cohort"], job["split"], job["bidx"], job["labs"], job["mode"], Path(job["out_dir"])
    shard = out_dir / cohort / "shards" / split / f"shard_{bidx:05d}.npz"
    qa = []
    parts = []
    try:
        mids = [lab["match_id"] for lab in labs]
        T = fetch_batch(mids)
        t_fetch = time.time() - t0
        meta = T["meta"].set_index("match_id")
        TT = {k: _by_match(v) for k, v in T.items() if k != "meta"}
        for mi, lab in enumerate(labs):
            mid = lab["match_id"]
            try:
                if mid not in meta.index or meta.loc[mid].started_ep is None:
                    raise ValueError("no stdb.matches row / started_at")
                row = meta.loc[mid]
                raw, extra = build_raw(mid, lab, TT, row, float(row.started_ep))
                res, n_dropped = match_samples(mid, lab, raw, extra, split, mode)
                if res is None:
                    raise ValueError("no usable sample time")
                parts.append((mid, mi, res))
                qa.append({"match_id": mid, "ok": True, "n_samples": int(len(res["S_t"])), "n_rows": int(len(res["TGT"])), "n_dropped_times": n_dropped,
                           "n_rows_no_pos": res["n_rows_no_pos"], "t_win": res["t_win"], "dump_version": "v2" if _is_v2(lab) else "v1",
                           **extra["flags"]})  # fmt: skip
            except Exception:
                qa.append({"match_id": mid, "ok": False, "error": traceback.format_exc()[-400:]})
        shard.parent.mkdir(parents=True, exist_ok=True)
        if parts:
            arrays = {
                "X": np.concatenate([r["X"] for _, _, r in parts]), "POS": np.concatenate([r["POS"] for _, _, r in parts]),
                "TGT": np.concatenate([r["TGT"] for _, _, r in parts]), "TIDX": np.concatenate([r["TIDX"] for _, _, r in parts]),
                "S_mi": np.concatenate([np.full(len(r["S_t"]), j, dtype=np.int32) for j, (_, _, r) in enumerate(parts)]),
                "S_t": np.concatenate([r["S_t"] for _, _, r in parts]), "S_cut": np.concatenate([r["S_cut"] for _, _, r in parts]),
                "S_n": np.concatenate([r["S_n"] for _, _, r in parts]), "S_zone": np.concatenate([r["S_zone"] for _, _, r in parts]),
                "match_ids": np.array([m for m, _, _ in parts]),
            }  # fmt: skip
            if all("KEY" in r for _, _, r in parts):  # v2 shards: per-row key / risk flags and the producing dump's version
                arrays["KEY"] = np.concatenate([r["KEY"] for _, _, r in parts])
                arrays["RISK"] = np.concatenate([r["RISK"] for _, _, r in parts])
                arrays["dump_version"] = np.array("v2")
            np.savez(str(shard) + ".tmp.npz", **arrays)
            os.replace(str(shard) + ".tmp.npz", shard)
    except Exception:
        qa.append({"match_id": "BATCH", "ok": False, "error": traceback.format_exc()[-600:]})
        t_fetch = float("nan")
    return {"cohort": cohort, "split": split, "bidx": bidx, "qa": qa, "sec": time.time() - t0, "fetch_sec": t_fetch}


def check_shard_version(shard: Path, rec: dict) -> None:
    """A shard that already exists is skipped; it must come from the same dump version as the label table (a v1 shard next to
    v2 labels would silently mix a look-ahead input with the v2 rows)."""
    from model_stgcn.ptr.data import check_same_version

    with np.load(shard) as z:
        have = str(z["dump_version"]) if "dump_version" in z.files else "v1"
    check_same_version(have, str(rec.get("dump_version", "v1")), what=f"shard {shard}", against=f"the label table of {rec.get('cohort')}/{rec.get('split')}")


def _init_worker():
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--train-mode", default="krafton192", choices=["krafton192", "every30"])
    ap.add_argument("--limit-matches", type=int, default=0, help="debug: first N matches of every split")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--redo", action="store_true", help="recompute shards that already exist")
    a = ap.parse_args(argv)
    if a.workers > 3:
        raise SystemExit("resource cap: at most 3 extraction workers")
    out_dir = Path(a.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)  # the QA log below is opened without creating its directory
    jobs = []
    for cohort in [c for c in a.cohorts.split(",") if c]:
        for split in [s for s in a.splits.split(",") if s]:
            rec = load_labels(out_dir, cohort, split)
            labs = rec["matches"][: a.limit_matches] if a.limit_matches else rec["matches"]
            for bi, i in enumerate(range(0, len(labs), BATCH)):
                shard = out_dir / cohort / "shards" / split / f"shard_{bi:05d}.npz"
                if shard.exists() and not a.redo:
                    check_shard_version(shard, rec)
                    continue
                jobs.append({"cohort": cohort, "split": split, "bidx": bi, "labs": labs[i : i + BATCH], "mode": a.train_mode, "out_dir": str(out_dir)})
    print(f"jobs={len(jobs)} workers={a.workers} mode={a.train_mode}", flush=True)
    t0 = time.time()
    log = open(out_dir / "logs" / f"extract_{int(t0)}.jsonl", "w")
    n_ok = n_bad = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(run_job, jobs)):
            for q in r["qa"]:
                q.update({"cohort": r["cohort"], "split": r["split"], "bidx": r["bidx"]})
                log.write(json.dumps(q) + "\n")
                n_ok += bool(q["ok"])
                n_bad += not q["ok"]
            log.flush()
            if (k + 1) % 5 == 0 or k + 1 == len(jobs):
                el = time.time() - t0
                print(f"[{el:7.1f}s] {k + 1}/{len(jobs)} batches  matches ok={n_ok} failed={n_bad}  last {r['cohort']}/{r['split']} {r['sec']:.1f}s (fetch {r['fetch_sec']:.1f}s)", flush=True)
    print(f"done in {time.time() - t0:.1f}s: ok={n_ok} failed={n_bad}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
