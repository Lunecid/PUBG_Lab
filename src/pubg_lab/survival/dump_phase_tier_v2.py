"""Leak-free phase-tier dump v2 for the BigComp 2027 PTR study (writes ``bigcomp2027_phase_v2``; v1 is untouched).

    python -m model_stgcn.dump_phase_tier_v2 --cohort erangel_tpp_squad --limit 20 --out-dir /tmp/v2_smoke
    python -m model_stgcn.dump_phase_tier_v2 --all --out-dir model_stgcn/dumps/bigcomp2027_phase_v2 --workers 6

Same population (13,948 matches, 12 cohorts, stored splits) and the same per-cohort schema as
``model_stgcn.dump_phase_tier`` plus ``meta['version'] == 'v2'`` and a ``meta['v2']`` block.  Nothing is read from the
derived ``stdb.*`` phase tables and nothing is written to the database: every input is computed in
:mod:`model_stgcn.phase_v2` from the raw ``pubg.*`` telemetry with explicit ``W < a_k`` filters
(``a_k = b_k - 10 s``, ``b_k`` = first LogPhaseChange with ceil(isGame) = k + 1, W = event_time - LogMatchStart).

What changed with respect to v1 (leaks L1-L3 and defects D1-D3 of the coordinator's list)::

    L1  alive_players_{start,end,min,avg}_norm   causal event-based counts over [phase start, a_k) (were taken at phase_end)
    L2  combat / loot / network / relation inputs only use events with time < a_k (were uncapped until phase_end)
    L3  token / key mask of slot k = teams causally alive at a_k (was: final team death time); risk-set rows that are
        causally dead at a_k keep their row and their query token (``query_mask``), only the key mask follows the
        causal status
    D1  phase bounds from events, so the terminal rows (phase k+1 started but has no grid snapshot) exist
    D2  no cap on the grid: it runs to the end of the log (v1's 2000 s cap = snapshot tier SNAP_T_MAX)
    D3  phase-duration inputs use the SCHEDULED duration (cohort / match-type median schedule of the TRAIN matches:
        blue_zone_options is empty in 100 % of the corpus)
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pubg_lab.survival import dump_phase_tier as V1
from pubg_lab.survival.phase_v2 import features as FT
from pubg_lab.survival.phase_v2.assemble import build_match_v2
from pubg_lab.survival.phase_v2.labels import GAP_S
from pubg_lab.survival.phase_v2.phases import CUT_LEAD_S
from pubg_lab.survival.phase_v2.source import fetch_match_source
from pubg_lab.survival.provenance import git_provenance
from model_stgcn.splits import load_or_create_split
from pubg_lab.survival.v3.cohorts import REQUIRED_COHORTS, CohortSpec, resolve_cohort

logger = logging.getLogger("model_stgcn.dump_phase_tier_v2")

DEFAULT_OUT_DIR = Path("model_stgcn/dumps/bigcomp2027_phase_v2")
SPLITS = V1.SPLITS
SEED = V1.SEED
MAX_WORKERS = 6
MIN_SCHEDULE_SUPPORT = 10
SCHEDULE_PHASES = tuple(range(1, 9))  # phases 1..8 are the dump's phase axis

ISGAME_NOTE = {
    "url": "https://documentation.pubg.com/en/telemetry-objects.html#common",
    "text": "Common.isGame 'represents the phase of the game defined by the status of bluezone and safezone': "
    "0 before lift off, 0.1 on airplane, 0.5 when there is no zone on the map (before the game starts), "
    "1.0 first safezone and bluezone appear, 1.5 first bluezone shrinks, 2.0 second bluezone appears, "
    "2.5 second bluezone shrinks.  So x.0 = the x-th zone is announced (the team waits), x.5 = the x-th "
    "blue zone shrinks.",
    "corpus_check": "A LogPhaseChange row carries the isGame of the state that has just ENDED (LogPlayerPosition.isGame "
    "switches to the next value at that instant) and telem_phase_changes.phase == ceil(isGame) for all "
    "303,800 rows of 21,830 matches; phase start times strictly increase and phase numbers are "
    "contiguous from 1 in every match.  Phase k therefore = states k.0 (wait) + k.5 (shrink); "
    "b_k = first LogPhaseChange with ceil(isGame) = k+1 = the end of the k-th shrink, the instant the "
    "(k+1)-th zone appears; a_k = b_k - 10 s.",
}


# --------------------------------------------------------------------------------------
# scheduled phase durations (D3)
# --------------------------------------------------------------------------------------
def _durations_by_match(cur, ids: list[str]) -> dict[str, dict[int, float]]:
    """``{match_id: {k: b_k - s_k}}`` from the LogPhaseChange events (W clock), phases that have a successor."""
    cur.execute(
        """
        WITH f AS (
            SELECT pc.match_id, ceil(pc.is_game)::int AS k,
                   min(extract(epoch FROM (pc.event_time - ms.event_time)))::float8 AS w
              FROM pubg.telem_phase_changes pc
              JOIN pubg.telem_match_start ms ON ms.match_id = pc.match_id
             WHERE pc.match_id = ANY(%s) AND pc.is_game IS NOT NULL
             GROUP BY pc.match_id, ceil(pc.is_game)
        )
        SELECT a.match_id, a.k, b.w - a.w
          FROM f a JOIN f b ON b.match_id = a.match_id AND b.k = a.k + 1
         WHERE a.k >= 1 AND a.k <= %s
        """,
        (ids, max(SCHEDULE_PHASES)),
    )
    out: dict[str, dict[int, float]] = {}
    for mid, k, d in cur.fetchall():
        out.setdefault(mid, {})[int(k)] = float(d)
    return out


def schedule_table(
    durations: dict[str, dict[int, float]], meta: dict[str, tuple[str, str, str]]
) -> dict[tuple, dict[int, tuple[float, int]]]:
    """Median scheduled duration per (level key, phase): ``(n, median)``.  ``meta[match_id] = (cohort slug, map_name,
    match_type)``.  Levels: cohort x match type, map x match type, match type, everything."""
    pools: dict[tuple, dict[int, list[float]]] = {}
    for mid, per_k in durations.items():
        if mid not in meta:
            continue
        slug, map_name, mtype = meta[mid]
        for key in (("cohort", slug, mtype), ("map", map_name, mtype), ("type", mtype), ("all",)):
            d = pools.setdefault(key, {})
            for k, v in per_k.items():
                d.setdefault(k, []).append(v)
    return {
        key: {k: (round(float(np.median(v)), 3), len(v)) for k, v in d.items()}
        for key, d in pools.items()
    }


def resolve_schedule(
    table: dict[tuple, dict[int, tuple[float, int]]],
    slug: str,
    map_name: str,
    match_type: str | None,
) -> tuple[dict[int, float], str]:
    """The match's schedule ``{k: seconds}`` and the lowest level that had to be used (``cohort`` ... ``all``)."""
    levels = [("cohort", ("cohort", slug, match_type)), ("map", ("map", map_name, match_type)),
              ("type", ("type", match_type)), ("all", ("all",))]  # fmt: skip
    out: dict[int, float] = {}
    worst = 0
    for k in SCHEDULE_PHASES:
        for rank, (_name, key) in enumerate(levels):
            ent = table.get(key, {}).get(k)
            if ent is not None and ent[1] >= MIN_SCHEDULE_SUPPORT:
                out[k] = ent[0]
                worst = max(worst, rank)
                break
    return out, f"{levels[worst][0]}_median" + ("" if worst == 0 else "_fallback")


# --------------------------------------------------------------------------------------
# worker pool
# --------------------------------------------------------------------------------------
_CONN = None


def _init_worker() -> None:
    global _CONN
    from pubg_lab.data.stdb.conn import connect

    _CONN = connect()
    _CONN.set_session(readonly=True, autocommit=True)


def _work(job: tuple[str, str, dict[int, float], str]) -> tuple[str, dict | None, str | None]:
    match_id, map_name, schedule, sched_src = job
    try:
        with _CONN.cursor() as cur:
            src = fetch_match_source(cur, match_id, schedule=schedule, schedule_source=sched_src)
            if src is None:
                return match_id, None, "no telem_match_start / teams"
            if len(src.phase_events["w"]) == 0:
                return match_id, None, "no LogPhaseChange events"
            return match_id, build_match_v2(src, map_size=V1.MAP_SIZE_M[map_name]), None
    except Exception as exc:  # pragma: no cover - reported per match
        import traceback

        return match_id, None, f"error: {exc!r} {traceback.format_exc(limit=4)}"


# --------------------------------------------------------------------------------------
# statistics / meta
# --------------------------------------------------------------------------------------
def aggregate_stats(matches: list[dict]) -> dict[str, Any]:
    """Per-cut rows / positives / comeback rows / exclusions and the revive-evidence audit of a split."""
    cut_rows = np.zeros(7, dtype=np.int64)
    cut_pos = np.zeros(7, dtype=np.int64)
    cut_comeback = np.zeros(7, dtype=np.int64)
    excl: Counter = Counter()
    excl_rows: Counter = Counter()
    excl_by_cut: dict[str, Counter] = {}
    revive = Counter()
    z_hist = np.zeros(len(FT_Z_BINS) - 1, dtype=np.int64)
    flagged: list[dict] = []
    stale_ex: list[dict] = []
    sched = Counter()
    censor_finite = 0
    for m in matches:
        s = m.pop("_stats")
        tgt = m["target_mask_v2"].numpy()
        y = m["y_hazard_v2"].numpy()
        for k in range(7):
            cut_rows[k] += int(tgt[k].sum())
            cut_pos[k] += int(((y[k] > 0.5) & tgt[k]).sum())
        cut_comeback += np.asarray(s["comeback_rows_by_cut"], dtype=np.int64)
        for k, reason in s["excluded"].items():
            excl[reason] += 1
            excl_rows[reason] += s["excluded_rows"][k]
            excl_by_cut.setdefault(reason, Counter())[str(k)] += 1
        r = s["revive"]
        for key in (
            "events",
            "low_altitude",
            "off_map",
            "too_soon",
            "stale_health_samples_ignored",
        ):
            revive[key] += r[key]
        for ex in r["stale_examples"]:
            if len(stale_ex) < 50:
                stale_ex.append({"match_id": m["match_id"], **ex})
        z_hist += np.asarray(r["z_hist"], dtype=np.int64)
        for f in r["flagged"]:
            if len(flagged) < 100:
                flagged.append({"match_id": m["match_id"], **f})
        sched[s["schedule_source"]] += 1
        censor_finite += s["censored_winner_finite_T"]
    return {
        "rows_by_cut": cut_rows.tolist(),
        "positives_by_cut": cut_pos.tolist(),
        "n_targets": int(cut_rows.sum()),
        "n_positives": int(cut_pos.sum()),
        "comeback_rows_by_cut": cut_comeback.tolist(),
        "comeback_rows": int(cut_comeback.sum()),
        "excluded_match_cuts": dict(excl),
        "excluded_rows": dict(excl_rows),
        "excluded_match_cuts_by_cut": {k: dict(v) for k, v in excl_by_cut.items()},
        "revive_evidence": {
            **dict(revive),
            "z_hist": z_hist.tolist(),
            "z_bins": FT_Z_BINS,
            "flagged_examples": flagged,
            "stale_examples": stale_ex,
        },
        "schedule_source": dict(sched),
        "winners_censored_with_finite_T": int(censor_finite),
    }


FT_Z_BINS = [-1e9, 300.0, 500.0, 800.0, 1000.0, 1001.9, 1002.3, 1e9]


GAME_STATE_COLUMNS = (
    "avg_alive_teams", "avg_alive_players", "safe_radius_start_m", "safe_radius_end_m", "safe_radius_delta_m",
    "safe_center_shift_m", "poison_radius_start_m", "poison_radius_end_m", "zone_gap_start_m", "zone_gap_end_m",
    *FT.HAB_GLOBAL,
)  # fmt: skip


def feature_change_classes() -> dict[str, str]:
    """Name -> why the value differs from v1 (first matching class)."""
    spec_col = {s[0]: s[1] for s in FT.SPECS}
    out = {}
    for n in FT.FEATURE_NAMES:
        c = spec_col[n]
        if n in FT.REPLACED_FEATURES:
            out[n] = "alive_causal_L1_L3"
        elif n in FT.DURATION_FEATURES:
            out[n] = "scheduled_duration_D3"
        elif n in FT.EVENT_FEATURES_L2:
            out[n] = "event_inputs_capped_at_a_k_L2"
        elif c in ("phase_idx", "team_size"):
            out[n] = "static_unchanged"
        elif c in GAME_STATE_COLUMNS:
            out[n] = "game_states_before_a_k"
        else:
            out[n] = "positions_grid_before_a_k"
    return out


def changed_feature_groups() -> dict[str, Any]:
    classes = feature_change_classes()
    return {
        "replaced_values_same_names": list(FT.REPLACED_FEATURES),
        "scheduled_duration_inputs": list(FT.DURATION_FEATURES),
        "event_inputs_now_capped_at_a_k": list(FT.EVENT_FEATURES_L2),
        "feature_change_class": classes,
        "class_sizes": dict(Counter(classes.values())),
        "all_other_features": "same definition as v1, recomputed from raw telemetry with every record restricted to "
        "W < a_k; grid on W (nearest sample within +-5 s, admissible if t + 5 < a_k) instead of "
        "the game-clock grid; zone state = nearest game state within +-5 s of the grid point",
        "feature_added_columns": [],
        "n_features": FT.N_FEATURES,
    }


def build_meta_v2(
    spec: CohortSpec,
    split: str,
    *,
    split_path: Path,
    split_ids: list[str],
    extra: dict,
    schedule_info: dict,
) -> dict[str, Any]:
    meta = V1.build_meta(spec, split, split_path=split_path, split_ids=split_ids, extra=extra)
    for key in ("fixb", "fixb_cut_guard_s", "fixb_since", "bump_since"):
        meta.pop(key, None)
    meta["clock"] = {
        "W": "LogMatchStart clock = event_time - telem_match_start.event_time (seconds): the ONE clock of v2.  "
        "Positions, game states, kills, damage, groggy, item events and phase changes all carry an event_time, so "
        "no game-clock offset (delta) is estimated or used (clock_delta_s / clock_delta_median_s are 0.0).",
        "cut": "cut_t[p] = cut_wall_s[p] = a_k = b_k - 10 s on W; NaN for the final phase of a match (no cut).",
    }
    meta["design_version"] = "v2.3+phase_dump_v2"
    meta["single_clock"] = True
    meta["version"] = "v2"
    conv = dict(meta["conventions"])
    conv.update(
        {
            "X": "computed from raw pubg.* telemetry (model_stgcn/phase_v2): per-column fixed-constant scaling as in "
            "stdb.queries._fetch_phase_node_features, NULL/NaN/inf -> 0.0; every record of every source has W < a_k",
            "alive_counts": "start = causally alive members at phase start s_k, end = at a_k, min / avg over [s_k, a_k) of the "
            "step function n(t); a member is alive iff no kill event <= t, or a position sample with "
            "health > 0 follows its last kill and is <= t (revive / redeploy evidence); /4",
            "node_mask": "teams causally alive at a_k (token / attention KEY mask of slot k); false in the final-phase slot",
            "query_mask": "node_mask | target_mask_v2: tokens that exist as queries; differs from node_mask only for "
            "risk-set teams that are causally dead at a_k (all members killed, comeback later)",
            "target_mask / y_hazard": "ALIASES of the v2 labels (the v1 pipeline labels from team_phase_labels are not "
            "recomputed in v2)",
            "labels_v2": "T = final team elimination time = max over members of the final death (last kill event not followed "
            "by a healthy position sample; members without kill: alive unless death_type left -> time_survived + "
            "match offset); winners censored (+inf) at C_m; risk set at cut k = T > b_k; y = b_k < T <= b_{k+1}, "
            "(b_k, C_m] when phase k+1 is the final phase; a (match, cut) is excluded only for a truncated log",
            "team_death_s": "T on W (+inf = censored: winner / alive at the end); label side only, never an input",
            "node_mask_v2": "NOT carried: its presence makes model_stgcn/ptr/data.py rebuild the alive mask from the final "
            "death time (the L3 leak); node_mask is the causal mask",
            "edges": "raw edge weights of the phase-k window [s_k, a_k) (prox/copress both directions, combat attacker -> victim)",
            "team_pos": "centroid / map size of the team's last alive grid point t of [s_k, a_k) with t + 5 < a_k; NaN if none",
            "zone": "[cur_cx, cur_cy, cur_r, next_cx, next_cy, next_r, phase_frac, phase_no]: the latest game state before a_k "
            "(blue = 'safe', white = 'poison') / map size; phase_frac = clip((a_k - s_k) / scheduled duration, 0, 1)",
            "cut_t": "a_k on W (not a grid time)",
            "phase_start_s / phase_end_s": "s_k and b_k on W from LogPhaseChange events (phase_end_s NaN for the final phase)",
            "team_has_monster_dmg": "0.0 (audit field of v1; not recomputed: the 12 cohorts have no Monster.* accounts)",
        }
    )
    meta["conventions"] = conv
    meta["v2"] = {
        "version": "v2",
        "spec": {
            "phase": "k = ceil(isGame) of LogPhaseChange; b_k = W of the first event with ceil(isGame) = k+1 (events, not grid); "
            "no such event -> final phase, no cut",
            "cut": f"a_k = b_k - {CUT_LEAD_S:g} s; every model input uses only records with W < a_k",
            "alive": "see conventions['alive_counts']",
            "risk_set": "T > b_k, k = 1..7",
            "exclusions": f"only for a truncated log: no LogMatchEnd, or a position gap > {GAP_S:g} s overlapping the label window",
        },
        "isgame": ISGAME_NOTE,
        "changed_features": changed_feature_groups(),
        "schedule": schedule_info,
        "leaks_fixed": [
            "L1 alive counts",
            "L2 uncapped combat / loot / network / relation events",
            "L3 final-death masks",
        ],
        "defects_fixed": [
            "D1 terminal rows from events",
            "D2 no 2000 s grid cap",
            "D3 scheduled durations",
        ],
        "d2_note": "No match of the corpus lasts 2000 s: max (LogMatchEnd - LogMatchStart) = 1975.4 s, max game-state "
        "elapsed_time < 2000 s; pubg.matches.duration > 2000 s for 1,528 matches is an API artefact (real log "
        "length < 2000 s).  v2 has no cap anyway (tests use a synthetic 2600 s match).",
        "git_commit": git_provenance().get("git_commit"),
    }
    return meta


# --------------------------------------------------------------------------------------
# cohort driver
# --------------------------------------------------------------------------------------
def dump_cohort_v2(
    spec: CohortSpec,
    out_dir: Path,
    *,
    workers: int,
    population: dict[str, Any],
    kept_ids: dict[str, list[str]],
    sched_table: dict,
    mtype: dict[str, str],
    split_root: Path = V1.SPLIT_ROOT,
    limit: int | None = None,
) -> dict[str, Any]:
    from pubg_lab.data.stdb.queries import list_phase_eligible_match_ids

    t0 = time.time()
    split_path = split_root / spec.slug / "split.json"
    ids = list_phase_eligible_match_ids(spec.map_name, spec.game_mode, None)
    load_or_create_split(ids, split_json=str(split_path), seed=SEED, require_existing=True)
    parts = {s: list(kept_ids[s]) for s in SPLITS}
    if limit:
        parts = {k: v[:limit] for k, v in parts.items()}
    cohort_dir = out_dir / spec.slug
    cohort_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")  # fork + libpq is unsafe on macOS
    manifest: dict[str, Any] = {
        "cohort": spec.slug,
        "version": "v2",
        "git_commit": git_provenance().get("git_commit"),
        "population": population["cohorts"][spec.slug],
        "splits": {},
    }
    with ctx.Pool(processes=max(1, min(workers, MAX_WORKERS)), initializer=_init_worker) as pool:
        for split, split_ids in parts.items():
            jobs = []
            src_levels: Counter = Counter()
            for mid in split_ids:
                sched, label = resolve_schedule(
                    sched_table, spec.slug, spec.map_name, mtype.get(mid)
                )
                src_levels[label] += 1
                jobs.append((mid, spec.map_name, sched, label))
            results = pool.map(_work, jobs, chunksize=4)
            matches, problems = [], {}
            for mid, item, err in results:
                if item is None:
                    problems[mid] = err
                else:
                    matches.append(item)
            if problems:
                logger.error(
                    "%s/%s: %d of %d matches could not be dumped, e.g. %s",
                    spec.slug,
                    split,
                    len(problems),
                    len(split_ids),
                    list(problems.items())[:3],
                )
            stats = aggregate_stats(matches)
            stats["schedule_resolution"] = dict(src_levels)
            meta = build_meta_v2(
                spec,
                split,
                split_path=split_path,
                split_ids=split_ids,
                extra={
                    "n_dumped": len(matches),
                    "skipped_matches": problems,
                    "stats": stats,
                    "population": {**population, "this_cohort": spec.slug},
                },
                schedule_info={
                    "source": "cohort x match-type median of (b_k - s_k) over the TRAIN matches of the kept population; "
                    "blue_zone_options is '[]' in 100% of the corpus so EVERY match uses the fallback",
                    "min_support": MIN_SCHEDULE_SUPPORT,
                    "resolution_levels_used": dict(src_levels),
                    "blue_zone_options_missing_matches": len(split_ids),
                    "cohort_schedule": {
                        mt: {
                            str(k): list(v)
                            for k, v in sched_table.get(("cohort", spec.slug, mt), {}).items()
                        }
                        for mt in sorted({mtype.get(m) for m in split_ids if mtype.get(m)})
                    },
                },
            )
            path = cohort_dir / f"{split}.pt"
            tmp = cohort_dir / f".{split}.pt.tmp"
            torch.save({"meta": meta, "matches": matches}, tmp)
            os.replace(tmp, path)
            manifest["splits"][split] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "n_split_ids": len(split_ids),
                "n_dumped": len(matches),
                "n_skipped": len(problems),
                "stats": stats,
            }
            logger.info(
                "%s/%s: %d/%d matches -> %s (%.1f MB) rows=%d pos=%d comeback=%d excl=%s",
                spec.slug,
                split,
                len(matches),
                len(split_ids),
                path,
                path.stat().st_size / 1e6,
                stats["n_targets"],
                stats["n_positives"],
                stats["comeback_rows"],
                stats["excluded_match_cuts"],
            )
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["splits"].values())
    V1._write_json(cohort_dir / "dump_manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cohort", action="append", help="cohort slug / label / alias (repeatable)")
    g.add_argument(
        "--all", action="store_true", help="all required cohorts that survive the population rules"
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--split-root", type=Path, default=V1.SPLIT_ROOT)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"<= {MAX_WORKERS}")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N matches per split")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.out_dir.resolve() == V1.DEFAULT_OUT_DIR.resolve():
        raise SystemExit("refusing to write into the v1 dump directory")

    all_specs = list(REQUIRED_COHORTS)
    requested = all_specs if args.all else [resolve_cohort(c) for c in args.cohort]
    from pubg_lab.data.stdb.conn import connect

    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            population, kept = V1.compute_population(cur, all_specs, args.split_root)
            sizes = V1.verify_map_sizes(cur, {s.map_name for s in requested})
            ids_all = sorted({m for slug, sp in kept.items() for s in SPLITS for m in sp[s]})
            cur.execute(
                "SELECT match_id, match_type FROM pubg.matches WHERE match_id = ANY(%s)", (ids_all,)
            )
            mtype = dict(cur.fetchall())
            # the schedule is a population constant estimated on the TRAIN matches only: no val / test match (nor its own
            # observed phase durations) enters the schedule of a held-out match
            spec_of = {sp_.slug: sp_ for sp_ in all_specs}
            cohort_of = {m: (slug, spec_of[slug].map_name, mtype.get(m))
                         for slug, sp in kept.items() for m in sp["train"]}  # fmt: skip
            durations = _durations_by_match(cur, sorted(cohort_of))
    finally:
        conn.close()
    sched_table = schedule_table(durations, cohort_of)
    logger.info("map sizes verified: %s", sizes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    V1._write_json(args.out_dir / "population.json", population)
    V1._write_json(
        args.out_dir / "schedule_table.json",
        {
            "|".join(map(str, k)): {str(p): list(v) for p, v in d.items()}
            for k, d in sched_table.items()
        },
    )
    failed = 0
    for spec in requested:
        status = population["cohorts"][spec.slug]["status"]
        if status != "kept":
            logger.warning("skipping %s: %s", spec.slug, status)
            continue
        manifest = dump_cohort_v2(
            spec,
            args.out_dir,
            workers=args.workers,
            population=population,
            kept_ids=kept[spec.slug],
            sched_table=sched_table,
            mtype=mtype,
            split_root=args.split_root,
            limit=args.limit,
        )
        if any(s["n_skipped"] for s in manifest["splits"].values()):
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
