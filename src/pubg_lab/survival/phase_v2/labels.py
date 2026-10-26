"""Landmark risk set and hazard labels of the v2 dump (labels only; they may use the whole log).

For cut k = 1..7 (a cut exists iff phase k+1 exists, i.e. b_k exists):

    risk set     teams with T > b_k                                (T = final elimination time, winners +inf)
    label        Y = 1  iff  b_k < T <= b_{k+1}
                 if phase k+1 is the final phase (no b_{k+1}) the window is (b_k, C_m]
                 winners / teams alive at the end are censored at C_m: never Y = 1
    exclusions   a (match, cut) is excluded ONLY if the log is truncated:
                 * no LogMatchEnd (C_m unknown)                      -> reason ``no_match_end``
                 * a position-data gap > 60 s overlapping the window -> reason ``position_gap``
                   (gaps are measured between consecutive position records of ALL players on W, closed by C_m)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pubg_lab.survival.phase_v2.phases import Phases

N_PHASES = 8
MAX_CUT = 7
GAP_S = 60.0
REASON_NO_END = "no_match_end"
REASON_GAP = "position_gap"


@dataclass
class LabelResult:
    team_death: np.ndarray  # (N,) float64, +inf censored / NaN unknown
    risk: np.ndarray  # (N_PHASES, N) bool, row k-1 = cut k (rows 7 stays False)
    y: np.ndarray  # (N_PHASES, N) float32
    excluded: dict[int, str] = field(default_factory=dict)  # cut k -> reason
    excluded_rows: dict[int, int] = field(
        default_factory=dict
    )  # cut k -> would-be risk-set rows (teams, T > b_k)
    window: dict[int, tuple[float, float | None, bool]] = field(
        default_factory=dict
    )  # k -> (b_k, e_k, terminal)
    n_censored_by_win: int = 0


def position_gaps(
    pos_w: np.ndarray, end_w: float | None, gap_s: float = GAP_S
) -> list[tuple[float, float]]:
    """``[(start, end)]`` of the intervals between consecutive position records (all players, sorted unique W)
    longer than ``gap_s``; the interval after the last record is closed by ``end_w`` when given."""
    t = np.unique(np.asarray(pos_w, dtype=np.float64))
    if len(t) == 0:
        return []
    ts = t if end_w is None else np.append(t, max(float(end_w), float(t[-1])))
    d = np.diff(ts)
    idx = np.nonzero(d > gap_s)[0]
    return [(float(ts[i]), float(ts[i + 1])) for i in idx]


def build_labels(
    team_death: np.ndarray,
    team_won: np.ndarray,
    phases: Phases,
    end_w: float | None,
    pos_w: np.ndarray,
) -> LabelResult:
    """Risk set / labels / exclusions (see the module docstring).  ``team_death`` is T from
    :func:`model_stgcn.phase_v2.alive.team_elimination_times`; winners are censored (+inf) here."""
    N = len(team_death)
    T = np.asarray(team_death, dtype=np.float64).copy()
    won = np.asarray(team_won, dtype=bool)
    n_win_fin = int((won & np.isfinite(T)).sum())
    T[won] = np.inf  # winners are censored at C_m whatever the kill events say
    res = LabelResult(
        team_death=T,
        risk=np.zeros((N_PHASES, N), dtype=bool),
        y=np.zeros((N_PHASES, N), dtype=np.float32),
        n_censored_by_win=n_win_fin,
    )
    gaps = position_gaps(pos_w, end_w) if end_w is not None else []
    known = ~np.isnan(T)
    for k in range(1, MAX_CUT + 1):
        if not phases.has_cut(k):
            continue
        b = phases.boundary(k)
        nxt = phases.boundary(k + 1)
        terminal = nxt is None
        at_risk = known & (T > b)
        if end_w is None:
            res.excluded[k] = REASON_NO_END
            res.excluded_rows[k] = int(at_risk.sum())
            continue
        e = end_w if terminal else nxt
        res.window[k] = (float(b), float(e), terminal)
        if any(g0 < e and g1 > b for g0, g1 in gaps):
            res.excluded[k] = REASON_GAP
            res.excluded_rows[k] = int(at_risk.sum())
            continue
        res.risk[k - 1] = at_risk
        res.y[k - 1] = (at_risk & np.isfinite(T) & (T <= e)).astype(np.float32)
    return res
