"""Stage ``extract``: raw telemetry (DB, read-only SELECT) -> the 8 Sagong in-game features per team and phase.

Reads ``<plan-dir>/<cohort>/windows.parquet`` (phase windows of the PTR dump matches) and writes
``features.parquet`` (match_id, team_id, phase, 8 features, n_members, n_obs), ``qc.parquet`` (per match: event counts
and the whole-match enemy attack / damage / kill totals next to ``participants`` sums), ``event_types.json`` and
``features_meta.json`` (dump version, window fingerprint, clock) into the same directory.  The DB is only read (``readonly``
session, SELECT).  At most ``--workers`` (default 2) processes, matches fetched in small batches; the run pauses while free +
inactive memory is below ``--min-free-gb``.

v2 plan (``windows.parquet`` has the column ``A`` = the producer's exact cut ``a_k``): the clock is computed by the database
(``db.fetch_raw_batch(clock='w')``) and every window is ``[S, A)``: only records strictly before the cut are read
(``features.compute_match_features(..., end_is_cut=True)``).  An existing ``features.parquet`` is only skipped when its
``features_meta.json`` names the same dump version and window fingerprint as the plan (a v1 cache next to a v2 plan raises).
"""

from __future__ import annotations

import json
import hashlib
import multiprocessing as mp
import subprocess
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from model_stgcn.sagong2025.features import compute_match_features, whole_match_totals

_CONN = None


def free_inactive_gb() -> float:
    """macOS free + inactive memory in GB (inf if it cannot be read)."""
    try:
        txt = subprocess.check_output(["vm_stat"], text=True, timeout=10)
        size = int(txt.split("page size of")[1].split("bytes")[0])
        pages = {ln.split(":")[0].strip(): int(ln.split(":")[1].strip().rstrip(".")) for ln in txt.splitlines() if ":" in ln}
        return (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)) * size / 1024**3
    except Exception:
        return float("inf")


def _init_worker() -> None:
    global _CONN
    from model_stgcn.sagong2025.db import connect_readonly

    _CONN = connect_readonly()


def _work(task):
    from model_stgcn.sagong2025.db import fetch_raw_batch

    items, v2 = task
    ids = [m for m, _ in items]
    t0 = time.time()
    cur = _CONN.cursor()
    raws = fetch_raw_batch(cur, ids, clock="w" if v2 else "epoch")
    cur.close()
    t_fetch = time.time() - t0
    feats, qc, ev = [], [], Counter()
    for mid, win in items:
        raw = raws.get(mid)
        if raw is None:
            qc.append({"match_id": mid, "missing": 1})
            continue
        df = compute_match_features(raw, win, end_is_cut=v2)
        df.insert(0, "match_id", mid)
        feats.append(df)
        tot = whole_match_totals(raw)
        tot.update(match_id=mid, missing=0, n_players=len(raw.team_of), n_pos=len(raw.pos), n_dmg=len(raw.dmg),
                   n_kill=len(raw.kill), n_pick=len(raw.pick))
        qc.append(tot)
        ev.update(raw.pick["event_type"].value_counts().to_dict())
    return (pd.concat(feats, ignore_index=True) if feats else None), qc, dict(ev), t_fetch, time.time() - t0


FEATURES_META = "features_meta.json"


def _plan_info(base: Path, wins: pd.DataFrame) -> dict:
    """What the features of this plan depend on: dump version (from ``plan_audit.json``; no audit = v1), the window fingerprint, the clock."""
    from model_stgcn.sagong2025.plan import windows_sha1

    audit = base / "plan_audit.json"
    version = json.loads(audit.read_text()).get("version", "v1") if audit.is_file() else "v1"
    feature_source = Path(__file__).with_name("features.py").read_bytes()
    return {"version": version, "windows_sha1": windows_sha1(wins), "feature_source_sha256": hashlib.sha256(feature_source).hexdigest(), "clock": "w (database)" if "A" in wins.columns else "epoch - started_at",
            "window": "[S, A) strictly before the cut a_k" if "A" in wins.columns else "[S, E - GUARD_S]"}


def check_features_meta(base: Path, plan_info: dict) -> None:
    """Refuse a ``features.parquet`` whose ``features_meta.json`` (absent = v1) names another dump version or other windows."""
    from model_stgcn.ptr.data import check_same_version

    p = base / FEATURES_META
    have = json.loads(p.read_text()) if p.is_file() else {"version": "v1", "windows_sha1": None}
    check_same_version(str(have.get("version", "v1")), str(plan_info["version"]), what=f"the Sagong features under {base}", against="the plan")
    if have.get("windows_sha1") not in (None, plan_info["windows_sha1"]):
        raise ValueError(f"{base}: features.parquet was extracted for other windows than windows.parquet (re-run extract with --force)")
    if plan_info.get("feature_source_sha256") and have.get("feature_source_sha256") != plan_info["feature_source_sha256"]:
        raise ValueError(f"{base}: features.parquet has a stale or missing feature source fingerprint (re-run extract with --force)")


def run_extract(plan_dir, cohort: str, workers: int = 2, batch: int = 6, limit: int | None = None,
                min_free_gb: float = 6.0, force: bool = False) -> dict:
    workers = max(1, min(int(workers), 2))  # resource cap agreed with the coordinator
    base = Path(plan_dir) / cohort
    wins = pd.read_parquet(base / "windows.parquet")
    v2 = "A" in wins.columns  # v2 plan: the windows carry the exact cut a_k
    plan_info = _plan_info(base, wins)
    if (base / "features.parquet").is_file() and not force and limit is None:
        check_features_meta(base, plan_info)  # a stale cache of another dump version / other windows must not be reused
        return {"cohort": cohort, "skipped": True}
    items = []
    for mid, g in wins.sort_values(["match_id", "phase"]).groupby("match_id", sort=True):
        end = g["A"] if v2 else g["E"]  # v2: the window is [S, A); v1: [S, E - GUARD_S]
        items.append((mid, {int(k): (float(s), float(e)) for k, s, e in zip(g["phase"], g["S"], end)}))
    if limit:
        items = items[:limit]
    tasks = [(items[i : i + batch], v2) for i in range(0, len(items), batch)]
    ctx = mp.get_context("spawn")
    feats, qcs, ev, t_fetch, t_total, t0 = [], [], Counter(), 0.0, 0.0, time.time()
    pending, done = [], 0
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        it = iter(tasks)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                while free_inactive_gb() < min_free_gb:  # throttle: do not start work while memory is short
                    time.sleep(20)
                try:
                    pending.append(pool.apply_async(_work, (next(it),)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            r = pending.pop(0).get()
            df, qc, e, tf, tt = r
            if df is not None:
                feats.append(df)
            qcs += qc
            ev.update(e)
            t_fetch += tf
            t_total += tt
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"[{time.time() - t0:7.1f}s] {cohort}: {done}/{len(tasks)} batches", flush=True)
    out = pd.concat(feats, ignore_index=True)
    if limit is None:
        out.to_parquet(base / "features.parquet", index=False)
        pd.DataFrame(qcs).to_parquet(base / "qc.parquet", index=False)
        (base / "event_types.json").write_text(json.dumps(dict(ev), indent=1) + "\n")
        (base / FEATURES_META).write_text(json.dumps(plan_info, indent=1) + "\n")
    return {"cohort": cohort, "n_matches": len(items), "n_rows": int(len(out)), "wall_s": time.time() - t0,
            "fetch_s": t_fetch, "proc_s": t_total - t_fetch, "event_types": dict(ev),
            "missing_matches": int(sum(q.get("missing", 0) for q in qcs))}
