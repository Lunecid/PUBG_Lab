"""Convert raw PUBG matches into experiment-usable STDB/phase/graph data.

Stages:
  base    : raw fetched classic matches -> base STDB + RQ1 phase labels
  spatial : phase labels -> trajectory/exposure phase features
  edge    : phase labels -> phase-level dynamic graph edge cache

The raw conversion target intentionally uses ``classic-crossmap`` from
``stdb.build_available``. It covers solo/duo/squad in both TPP and FPP and
excludes Tutorial_Main and Range_Main, which are not suitable for the current
phase/graph survival experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from stdb.build_available import select_match_ids as select_base_ids
from stdb.etl import ingest_match
from stdb.phase_edge_etl import build_match as build_edge_match
from stdb.phase_edge_etl import select_match_ids as select_edge_ids
from stdb.phase_spatial_feature_etl import build_match as build_spatial_match
from stdb.phase_spatial_feature_etl import select_match_ids as select_spatial_ids
from stdb.rq1_feature_etl import build_match as build_rq1_features
from stdb.conn import connect


DEFAULT_OUT = Path("model_stgcn/etl_runs/raw_to_usable_latest.json")


def _base_one(match_id: str) -> dict[str, Any]:
    started = time.time()
    base = ingest_match(match_id)
    if not base.get("ok"):
        return {
            "match_id": match_id,
            "ok": False,
            "stage": "base",
            "elapsed_s": round(time.time() - started, 3),
            "result": base,
        }
    rq1 = build_rq1_features(match_id)
    if not rq1.get("ok"):
        return {
            "match_id": match_id,
            "ok": False,
            "stage": "rq1",
            "elapsed_s": round(time.time() - started, 3),
            "result": rq1,
        }
    return {
        "match_id": match_id,
        "ok": True,
        "stage": "base",
        "elapsed_s": round(time.time() - started, 3),
        "base": base,
        "rq1": rq1,
    }


def _spatial_one(match_id: str) -> dict[str, Any]:
    started = time.time()
    conn = connect()
    try:
        with conn:
            with conn.cursor() as cur:
                counts = build_spatial_match(cur, match_id)
        return {
            "match_id": match_id,
            "ok": True,
            "stage": "spatial",
            "elapsed_s": round(time.time() - started, 3),
            "counts": counts,
        }
    except Exception as exc:
        conn.rollback()
        return {
            "match_id": match_id,
            "ok": False,
            "stage": "spatial",
            "elapsed_s": round(time.time() - started, 3),
            "error": repr(exc),
        }
    finally:
        conn.close()


def _edge_one(match_id: str) -> dict[str, Any]:
    started = time.time()
    conn = connect()
    try:
        with conn:
            with conn.cursor() as cur:
                counts = build_edge_match(cur, match_id)
        return {
            "match_id": match_id,
            "ok": True,
            "stage": "edge",
            "elapsed_s": round(time.time() - started, 3),
            "counts": counts,
        }
    except Exception as exc:
        conn.rollback()
        return {
            "match_id": match_id,
            "ok": False,
            "stage": "edge",
            "elapsed_s": round(time.time() - started, 3),
            "error": repr(exc),
        }
    finally:
        conn.close()


def _run_pool(stage: str, match_ids: list[str], worker_fn, *, workers: int, progress_every: int) -> dict[str, Any]:
    started = time.time()
    if not match_ids:
        return {"stage": stage, "selected": 0, "ok": 0, "failed": 0, "failures": [], "elapsed_s": 0.0}

    print(f"[{stage}] selected={len(match_ids)} workers={workers}", flush=True)
    ok = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    elapsed_sum = 0.0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, mid) for mid in match_ids]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            elapsed_sum += float(result.get("elapsed_s", 0.0))
            if result.get("ok"):
                ok += 1
            else:
                failed += 1
                failures.append(result)
                print(f"[{stage}] FAILED {result.get('match_id')}: {result.get('error') or result.get('result')}", flush=True)
            if done == 1 or done % progress_every == 0 or done == len(match_ids):
                wall = time.time() - started
                rate = done / wall if wall > 0 else 0.0
                print(
                    f"[{stage}] progress {done}/{len(match_ids)} ok={ok} failed={failed} "
                    f"rate={rate:.2f}/s",
                    flush=True,
                )

    return {
        "stage": stage,
        "selected": len(match_ids),
        "ok": ok,
        "failed": failed,
        "failures": failures[:50],
        "mean_match_elapsed_s": round(elapsed_sum / max(ok + failed, 1), 3),
        "elapsed_s": round(time.time() - started, 3),
    }


def select_ids_for_stage(stage: str, *, limit: int | None, refresh: bool) -> list[str]:
    if stage == "base":
        return select_base_ids(
            "classic-crossmap",
            limit=limit,
            refresh=refresh,
        )
    if stage == "spatial":
        return select_spatial_ids(limit=limit, refresh=refresh, match_id=None)
    if stage == "edge":
        return select_edge_ids(limit=limit, refresh=refresh, match_id=None)
    raise ValueError(f"unknown stage: {stage}")


def worker_for_stage(stage: str):
    if stage == "base":
        return _base_one
    if stage == "spatial":
        return _spatial_one
    if stage == "edge":
        return _edge_one
    raise ValueError(f"unknown stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=["base", "spatial", "edge"], default=["base", "spatial", "edge"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run_started = time.time()
    summary: dict[str, Any] = {
        "stages_requested": args.stages,
        "workers": args.workers,
        "limit": args.limit,
        "refresh": args.refresh,
        "stage_results": [],
    }

    for stage in args.stages:
        ids = select_ids_for_stage(stage, limit=args.limit, refresh=args.refresh)
        result = _run_pool(
            stage,
            ids,
            worker_for_stage(stage),
            workers=args.workers,
            progress_every=max(args.progress_every, 1),
        )
        summary["stage_results"].append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary["elapsed_s"] = round(time.time() - run_started, 3)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if result["failed"]:
            print(f"[{stage}] completed with failures; continuing to next stage", flush=True)

    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
