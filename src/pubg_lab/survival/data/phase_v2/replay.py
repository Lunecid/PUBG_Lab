"""Prefix-replay test of the v2 inputs (the key leak test; DB-gated).

For every selected match and every cut ``k`` whose boundary ``b_k`` exists, the model inputs of slot k are rebuilt twice:

    A  from the FULL source (all records of the match),
    B  from a source that the DATABASE truncated at ``a_k = b_k - 10 s``: every temporal query (positions, game states,
       kills, damage, groggy, pickup / equip / use, phase changes) carries ``extract(epoch from event_time - started_at) < a_k``,
       so no record at or after the cut ever reaches Python,

and the two results must be EXACTLY equal: X, the edges of all three types, ``node_mask`` (token / key mask), the alive
counts, the AI share, ``team_pos`` and the zone vector.  A mutation check re-runs the comparison with a builder that
forgets to cut the event tables: it must report differences, otherwise the test could not detect a leak.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

import pubg_lab.survival.data.phase_v2.features as FT
from pubg_lab.survival.data.phase_v2.phases import phases_of
from pubg_lab.survival.data.phase_v2.source import fetch_match_source, temporal_row_counts

DEFAULT_SCHEDULE = {1: 510.0, 2: 210.0, 3: 180.0, 4: 180.0, 5: 160.0, 6: 120.0, 7: 100.0, 8: 120.0}
MAP_SIZE = {"Baltic_Main": 8160.0, "Tiger_Main": 8160.0, "Desert_Main": 8160.0, "Neon_Main": 8160.0,
            "Savage_Main": 4080.0, "Chimera_Main": 3060.0, "Summerland_Main": 2040.0}  # fmt: skip


def stratified_matches(cur, per_cohort: int, tag: str = "prefix-replay") -> list[tuple[str, str]]:
    """``[(cohort slug, match_id)]``: ``per_cohort`` matches of every kept cohort (airoyale and Vikendi excluded as in
    the dump), deterministic md5 order."""
    from pubg_lab.survival.v3.data.cohorts import REQUIRED_COHORTS

    out: list[tuple[str, str]] = []
    for spec in REQUIRED_COHORTS:
        if spec.slug == "vikendi_tpp_squad":
            continue
        cur.execute(
            """
            SELECT m.match_id
              FROM stdb.matches m JOIN pubg.matches p ON p.match_id = m.match_id
             WHERE m.map_name = %s AND m.game_mode = %s AND p.match_type <> 'airoyale'
               AND EXISTS (SELECT 1 FROM pubg.telem_phase_changes c WHERE c.match_id = m.match_id)
             ORDER BY md5(m.match_id || %s) LIMIT %s
            """,
            (spec.map_name, spec.game_mode, tag, per_cohort),
        )
        out += [(spec.slug, r[0]) for r in cur.fetchall()]
    return out


def output_dict(cf: FT.CutFeatures) -> dict[str, np.ndarray]:
    out = {
        "X": cf.X, "node_mask": cf.node_alive, "alive_counts": cf.alive_counts, "team_pos": cf.team_pos,
        "zone": cf.zone, "zone_valid": np.asarray(cf.zone_valid), "team_ai_frac": cf.team_ai_frac,
    }  # fmt: skip
    for key, (i, j, w) in cf.edges.items():
        out[f"edges_{key}_i"], out[f"edges_{key}_j"], out[f"edges_{key}_w"] = i, j, w
    return out


def differing(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    return [n for n in a if not np.array_equal(np.asarray(a[n]), np.asarray(b[n]), equal_nan=True)]


def run_prefix_replay(
    cur,
    matches: list[tuple[str, str]],
    *,
    schedule: dict[int, float] | None = None,
    mutation_matches: int = 3,
    progress=None,
) -> dict[str, Any]:
    """Replay every cut of every match; returns a JSON-able summary (``mismatches`` must be empty)."""
    sched = dict(DEFAULT_SCHEDULE if schedule is None else schedule)
    t0 = time.time()
    res: dict[str, Any] = {
        "n_matches": 0, "n_cuts": 0, "n_cohorts": len({c for c, _ in matches}), "mismatches": [],
        "rows_seen_full": 0, "rows_seen_truncated": 0, "cohort_matches": {}, "cuts_by_k": {},
        "max_record_time_gap_s": float("inf"),
    }  # fmt: skip
    min_gap = float("inf")
    mutation = {"matches": 0, "cuts_compared": 0, "cuts_detected_as_leaky": 0}
    import pubg_lab.survival.data.phase_v2.features as features_mod

    original_truncate = features_mod.truncate
    for n, (slug, mid) in enumerate(matches):
        src = fetch_match_source(cur, mid, schedule=sched, schedule_source="replay")
        if src is None:
            continue
        ph = phases_of(src)
        size = MAP_SIZE[src.map_name]
        res["n_matches"] += 1
        res["cohort_matches"][slug] = res["cohort_matches"].get(slug, 0) + 1
        do_mutation = mutation["matches"] < mutation_matches
        mutation["matches"] += int(do_mutation)
        for k in ph.indices:
            if not ph.has_cut(k):
                continue
            a = ph.cut(k)
            full = output_dict(FT.compute_cut_features(src, k, a, map_size=size))
            part_src = fetch_match_source(
                cur, mid, cut_w=a, schedule=sched, schedule_source="replay"
            )
            gap = a - max(
                (float(t["w"].max()) for t in (part_src.pos, part_src.gs, part_src.kills, part_src.damage,
                                              part_src.groggy, *part_src.items.values()) if len(t["w"])),
                default=-np.inf,
            )  # fmt: skip
            min_gap = min(min_gap, gap)
            part = output_dict(FT.compute_cut_features(part_src, k, a, map_size=size))
            bad = differing(full, part)
            res["n_cuts"] += 1
            res["cuts_by_k"][str(k)] = res["cuts_by_k"].get(str(k), 0) + 1
            res["rows_seen_full"] += sum(temporal_row_counts(src).values())
            res["rows_seen_truncated"] += sum(temporal_row_counts(part_src).values())
            if bad or not gap > 0:
                res["mismatches"].append(
                    {"match_id": mid, "cohort": slug, "k": k, "differing": bad, "gap": gap}
                )
            if do_mutation and len(src.kills["w"]):
                # a builder that forgets to cut the event tables must NOT reproduce the truncated result
                features_mod.truncate = lambda s, c: replace(  # noqa: E731
                    s,
                    phase_events={
                        kk: v[s.phase_events["w"] < c] for kk, v in s.phase_events.items()
                    },
                )
                try:
                    leaky = output_dict(FT.compute_cut_features(src, k, a, map_size=size))
                finally:
                    features_mod.truncate = original_truncate
                mutation["cuts_compared"] += 1
                mutation["cuts_detected_as_leaky"] += int(bool(differing(leaky, part)))
        if progress is not None:
            progress(n + 1, len(matches), res)
    res["max_record_time_gap_s"] = None if not np.isfinite(min_gap) else float(min_gap)
    res["min_gap_between_last_kept_record_and_cut_s"] = res.pop("max_record_time_gap_s")
    res["mutation_check"] = mutation
    res["runtime_s"] = round(time.time() - t0, 1)
    return res
