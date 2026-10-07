"""One match -> the v2 dump record (same schema as model_stgcn/dump_phase_tier.py v1 plus the v2 additions).

Per cut k = 1..8 whose boundary ``b_k`` exists, :func:`model_stgcn.phase_v2.features.compute_cut_features` sees only
records before ``a_k = b_k - 10 s``; the final phase of a match has no cut (its slot stays empty: X = 0,
``node_mask`` False).  Labels, the risk set and the elimination time ``T`` (:mod:`labels`, :mod:`alive`) use the
whole log and never feed an input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from model_stgcn.phase_v2 import alive as AL
from model_stgcn.phase_v2 import features as FT
from model_stgcn.phase_v2 import labels as LB
from model_stgcn.phase_v2.phases import phases_of
from model_stgcn.phase_v2.source import MatchSource

N_PHASES = 8
EDGE_OUT_KEYS = FT.EDGE_KEYS  # ('prox', 'combat', 'copress')
Z_BINS = [-1e9, 300.0, 500.0, 800.0, 1000.0, 1001.9, 1002.3, 1e9]  # altitude histogram of revive evidence


def _f32(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def build_match_v2(src: MatchSource, *, map_size: float, cuts: list[int] | None = None) -> dict[str, Any]:
    """Assemble the dump record of a match.  ``cuts`` restricts the cuts that are computed (tests)."""
    N, P, F = src.n_teams, N_PHASES, FT.N_FEATURES
    ph = phases_of(src)
    X = np.zeros((P, N, F), dtype=np.float32)
    node = np.zeros((P, N), dtype=bool)
    phase_mask = np.zeros(P, dtype=bool)
    phase_start = np.full(P, np.nan, dtype=np.float64)  # time fields are float64: a_k must compare exactly with W
    phase_end = np.full(P, np.nan, dtype=np.float64)
    cut_t = np.full(P, np.nan, dtype=np.float64)
    cut_mask = np.zeros(P, dtype=bool)
    sched_s = np.full(P, np.nan, dtype=np.float32)
    team_pos = np.full((P, N, 2), np.nan, dtype=np.float32)
    zone = np.full((P, len(FT.ZONE_COLUMNS)), np.nan, dtype=np.float32)
    zone_valid = np.zeros(P, dtype=bool)
    buckets: dict[str, list[tuple[int, int, int, float]]] = {key: [] for key in EDGE_OUT_KEYS}
    per_cut: dict[int, FT.CutFeatures] = {}
    ai = FT.team_ai_share(src)

    for k in range(1, P + 1):
        if not ph.exists(k):
            continue
        phase_mask[k - 1] = True
        phase_start[k - 1] = ph.start(k)
        if k in src.schedule:
            sched_s[k - 1] = float(src.schedule[k])
        zone[k - 1, 7] = float(k)
        b = ph.boundary(k)
        if b is None:
            continue  # final phase: no cut
        phase_end[k - 1] = b
        if cuts is not None and k not in cuts:
            continue
        a_k = ph.cut(k)
        cf = FT.compute_cut_features(src, k, a_k, map_size=map_size)
        per_cut[k] = cf
        X[k - 1] = cf.X
        node[k - 1] = cf.node_alive
        cut_mask[k - 1] = True
        cut_t[k - 1] = a_k
        team_pos[k - 1] = cf.team_pos
        zone[k - 1] = cf.zone
        zone_valid[k - 1] = cf.zone_valid
        for key, (ii, jj, ww) in cf.edges.items():
            for i, j, w in zip(ii.tolist(), jj.tolist(), ww.tolist()):
                if i != j:
                    buckets[key].append((k - 1, i, j, float(w)))
    edges: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, rows in buckets.items():
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        if rows:
            edges[key] = (torch.tensor([r[:3] for r in rows], dtype=torch.int64),
                          torch.tensor([r[3] for r in rows], dtype=torch.float32))  # fmt: skip
        else:
            edges[key] = (torch.zeros((0, 3), dtype=torch.int64), torch.zeros((0,), dtype=torch.float32))

    # ---- labels (whole log) ------------------------------------------------------------------------------
    acct_idx = src.account_index()
    tl = AL.build_timelines(src.kills["w"], src.kills["victim"], src.pos["w"], src.pos["acct"], src.pos["health"])
    T = AL.team_elimination_times(src.team_ids, src.roster, acct_idx, tl)
    lab = LB.build_labels(T, src.team_won, ph, src.end_w, src.pos["w"])
    target = lab.risk.copy()
    y = lab.y.copy()
    if cuts is not None:
        for k in range(1, P):
            if k not in cuts:
                target[k - 1] = False
                y[k - 1] = 0.0
    n_teams = max(N, 1)
    rank = src.team_final_rank
    rank_norm = np.where(np.isfinite(rank), rank / n_teams, 0.0)
    top5 = np.where(np.isfinite(rank), (rank <= 5).astype(np.float64), 0.0)
    top10 = np.where(np.isfinite(rank), (rank <= 10).astype(np.float64), 0.0)
    query = node | target
    win_end = np.full(P, np.nan, dtype=np.float64)
    terminal = np.zeros(P, dtype=bool)
    for k, (_b, e, term) in lab.window.items():
        win_end[k - 1] = e
        terminal[k - 1] = term
    cut_valid = np.zeros(P, dtype=bool)  # label-valid landmark cuts 1..7: the cut exists and the log is not truncated
    for k in range(1, P):
        cut_valid[k - 1] = bool(cut_mask[k - 1]) and ph.has_cut(k) and (k not in lab.excluded)
    comeback = target & ~node  # in the risk set but causally dead at the cut: query token stays, key mask off
    rev = AL.revive_evidence(tl, src.pos, map_size)
    flagged = [r for r in rev if r["low_altitude"] or r["off_map"] or r["too_soon"]]
    stale = AL.stale_health_samples(src.kills["w"], src.kills["victim"], src.pos["w"], src.pos["acct"], src.pos["health"])

    return {
        "match_id": src.match_id,
        "team_ids": torch.tensor(src.team_ids, dtype=torch.int64),
        "X": torch.from_numpy(X),
        "node_mask": torch.from_numpy(node),
        "target_mask": torch.from_numpy(target.copy()),  # v2 labels (the v1 pipeline labels are not recomputed)
        "y_hazard": torch.from_numpy(y.copy()),
        "final_rank_norm": _f32(rank_norm),
        "top5": _f32(top5),
        "top10": _f32(top10),
        "edges": edges,
        "team_pos": torch.from_numpy(team_pos),
        "zone": torch.from_numpy(zone),
        "zone_valid": torch.from_numpy(zone_valid),
        "map_size": float(map_size),
        "phase_mask": torch.from_numpy(phase_mask),
        "phase_start_s": torch.from_numpy(phase_start),
        "phase_end_s": torch.from_numpy(phase_end),
        "cut_t": torch.from_numpy(cut_t),
        "cut_mask": torch.from_numpy(cut_mask),
        "cut_valid": torch.from_numpy(cut_valid),
        "cut_excluded": {int(k): str(v) for k, v in lab.excluded.items()},
        "cut_terminal": torch.from_numpy(terminal),
        "label_window_end_s": torch.from_numpy(win_end),
        "match_end_s": float("nan") if src.end_w is None else float(src.end_w),
        "phase_duration_sched_s": torch.from_numpy(sched_s),
        "cut_wall_s": torch.from_numpy(cut_t.copy()),  # W == the one clock: identical to cut_t
        "clock_delta_s": 0.0,
        "clock_delta_median_s": 0.0,
        "node_mask_label": torch.from_numpy(node.copy()),
        "last_phase_idx": -1 if ph.last_phase is None else int(ph.last_phase),
        "y_hazard_v2": torch.from_numpy(y),
        "target_mask_v2": torch.from_numpy(target),
        "query_mask": torch.from_numpy(query),
        "team_death_s": torch.from_numpy(np.ascontiguousarray(lab.team_death, dtype=np.float64)),
        "team_won": torch.from_numpy(np.asarray(src.team_won, dtype=bool)),
        "team_ai_frac": _f32(ai),
        "team_has_monster_dmg": torch.zeros(N, dtype=torch.float32),
        "match_type": src.match_type,
        "_stats": {
            "n_targets": int(target.sum()),
            "n_positives": int(((y > 0.5) & target).sum()),
            "comeback_rows": int(comeback.sum()),
            "comeback_rows_by_cut": [int(comeback[k].sum()) for k in range(P - 1)],
            "excluded": {int(k): v for k, v in lab.excluded.items()},
            "excluded_rows": {int(k): int(v) for k, v in lab.excluded_rows.items()},
            "censored_winner_finite_T": int(lab.n_censored_by_win),
            "schedule_source": src.schedule_source,
            "n_cuts": len(per_cut),
            "revive": {
                "events": len(rev),
                "low_altitude": int(sum(r["low_altitude"] for r in rev)),
                "off_map": int(sum(r["off_map"] for r in rev)),
                "too_soon": int(sum(r["too_soon"] for r in rev)),
                "stale_health_samples_ignored": len(stale),
                "stale_examples": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in stale[:3]],
                "z_hist": np.histogram([r["z_m"] for r in rev], bins=Z_BINS)[0].astype(int).tolist(),
                "gap_min_s": float(min((r["gap_s"] for r in rev), default=np.nan)),
                "flagged": [
                    {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in flagged[:5]
                ],
            },
        },
    }
