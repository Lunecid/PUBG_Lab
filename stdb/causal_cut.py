"""Causal-cut guard for phase-tier features and edges (Fix B + single clock, 2026-10-05).

Two independent problems are closed here.

1. Resampling look-ahead (Fix B).  ``stdb.etl`` resamples raw telemetry onto a 10 s grid: grid time
   ``t`` takes the raw sample nearest to ``t`` inside ``[t - SNAPSHOT_HALF_WIN, t + SNAPSHOT_HALF_WIN]``.
   The last grid point of a phase can therefore read raw data up to 5 s after the phase ended.  Every
   phase-tier grid / position window is capped at ``phase_end - SNAPSHOT_INTERVAL``.

2. Two clocks (design v2.3, label audit 2026-10-05).  ``stdb.v_match_phase_bounds`` carries
   ``phase_start_s`` / ``phase_end_s`` on the LogMatchStart clock W (``event_time - started_at``): the
   phase-change rows have ``elapsed_time = NULL`` (96 / 96 sampled matches, 249k / 249k rows in the
   cohorts), so ``stdb.etl`` tags the grid with W boundary numbers.  The grid itself, the raw
   position windows and the game states are on the game clock G (``elapsed_time``), which lags W:
   ``W = G + delta`` with a per-match ``delta`` (median 7-8 s, p95 20 s, max 42-75 s).  Comparing G
   times with W boundary numbers lets a "capped" window still end after the true phase boundary
   (about 25-30 % of (match, phase) windows).  Estimate ``delta`` per match exactly as the audit does
   (median over the match's game states of ``event_time - started_at - elapsed_time``), express
   the boundaries on G, and cap THEN::

       window on G  =  [phase_start_s - delta,  phase_end_s - delta - SNAPSHOT_INTERVAL]

3. Clock drift.  In some matches the offset ``W - G`` is not constant: the game clock runs slower than
   the event clock and the offset grows steadily (up to ~45 s at the end of a match, 0.9 s at its
   start).  A single median ``delta`` then ends the window too late in the late phases.  In a 1,500-match
   sample the median alone left 0.12 % of the grid windows (0.3 % of the matches) and 1.5 % of the
   game-state windows (5.5 % of the matches) reading past the true start of the next phase (first
   LogPhaseChange of phase k+1 on W), almost all of them in matches with ``delta`` above ~15 s.
   :func:`fetch_clock_delta` therefore returns the audit's median, RAISED by the smallest multiple of
   0.25 s that closes the look-ahead of every phase of that match (:func:`safe_clock_delta`): the
   last raw row a capped grid point may read (G <= cut + SNAPSHOT_HALF_WIN, event-clock time interpolated
   from the game-state offsets) must end ``GRID_MARGIN_S`` before the next phase starts, and the
   latest game state inside the uncapped game-state window (its own exact W time) must lie before it.
   For ~94 % of the matches the raised value equals the median (nothing to close).  The audit's median
   stays available as :attr:`ClockDelta.median_s` (and in the dump as ``clock_delta_median_s``).

Scope: grid joins (``stdb.moving_objects``, ``stdb.dynamic_regions``, ``stdb.team_proximity``,
``stdb.team_snapshot_features``) and the raw position window of ``stdb.team_phase_state``.  Game
states and kill / damage / item events carry exact timestamps (no resampling): they get the clock
shift but not the cap.  Events are already on W and are compared with the W bounds unchanged.

This module is DB-free except for :func:`fetch_clock_delta` / :func:`fetch_clock_delta_detail`.  The
ETL modules interpolate the window columns into their SQL and ``tests/test_causal_cut.py`` lints every
ETL source so that no uncapped, clock-mixed grid join can reappear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from stdb.etl import SNAPSHOT_HALF_WIN, SNAPSHOT_INTERVAL

# Seconds subtracted from the (clock-aligned) phase end before any grid / position join.
GUARD_S: int = SNAPSHOT_INTERVAL

# Names of the columns the ETL ``bounds`` CTEs expose (all on the game clock G).
WIN_START_COLUMN = "win_start_g"  # phase_start_s - delta
WIN_END_COLUMN = "win_end_g"  # phase_end_s - delta                (no cap)
CUT_COLUMN = "phase_cut_s"  # phase_end_s - delta - GUARD_S      (the capped end)

# delta = median over the match's game states of  (event_time - started_at) - elapsed_time
CLOCK_DELTA_SQL = """
SELECT coalesce(percentile_cont(0.5) WITHIN GROUP (
           ORDER BY extract(epoch FROM (gs.event_time - m.started_at)) - gs.elapsed_time), 0.0)::float8
  FROM pubg.telem_game_states gs
  JOIN stdb.matches m ON m.match_id = gs.match_id
 WHERE gs.match_id = %s
   AND gs.elapsed_time IS NOT NULL
   AND m.started_at IS NOT NULL
"""


# the game-state offsets W - G of a match (G ascending), the W time of the first LogPhaseChange of every
# phase, and the phase bounds the ETLs use (W numbers)
CLOCK_PAIRS_SQL = """
SELECT gs.elapsed_time::float8,
       (extract(epoch FROM (gs.event_time - m.started_at)) - gs.elapsed_time)::float8
  FROM pubg.telem_game_states gs
  JOIN stdb.matches m ON m.match_id = gs.match_id
 WHERE gs.match_id = %s AND gs.elapsed_time IS NOT NULL AND m.started_at IS NOT NULL
 ORDER BY gs.elapsed_time
"""
PHASE_START_W_SQL = """
SELECT pc.phase::int, min(extract(epoch FROM (pc.event_time - m.started_at)))::float8
  FROM pubg.telem_phase_changes pc
  JOIN stdb.matches m ON m.match_id = pc.match_id
 WHERE pc.match_id = %s AND pc.phase IS NOT NULL AND m.started_at IS NOT NULL
 GROUP BY pc.phase
"""
PHASE_BOUNDS_SQL = """
SELECT phase_idx, phase_start_s::float8, phase_end_s::float8
  FROM stdb.v_match_phase_bounds
 WHERE match_id = %s
"""

# the last raw row a capped grid point reads must end this long before the next phase starts (seconds)
GRID_MARGIN_S: float = 1.0
# the latest game state inside the uncapped game-state window (exact W time) must lie this far before it
STATE_MARGIN_S: float = 0.05
RAISE_STEP_S: float = 0.25
MAX_RAISE_S: float = 120.0


@dataclass(frozen=True)
class ClockDelta:
    """Per-match clock lag (W = G + delta) in seconds."""

    median_s: float  # the audit's estimate: median of (event_time - started_at - elapsed_time)
    safe_s: float  # ``median_s`` raised just enough to close the drift look-ahead (>= median_s)
    n_pairs: int  # game states behind the estimate
    n_phases_checked: int = 0  # phases with a known next-phase start
    saturated: bool = False  # True if even ``median + MAX_RAISE_S`` did not satisfy the margins

    @property
    def raised(self) -> bool:
        return self.safe_s > self.median_s + 1e-9


def _dedupe_pairs(g, off):
    """Strictly increasing G with the mean offset of duplicated game times (np.interp needs it)."""
    g = np.asarray(g, dtype=np.float64)
    off = np.asarray(off, dtype=np.float64)
    keep = np.isfinite(g) & np.isfinite(off)
    g, off = g[keep], off[keep]
    if g.size == 0:
        return g, off
    order = np.argsort(g, kind="stable")
    g, off = g[order], off[order]
    uniq, inv = np.unique(g, return_inverse=True)
    if uniq.size == g.size:
        return g, off
    sums = np.bincount(inv, weights=off)
    counts = np.bincount(inv)
    return uniq, sums / counts


def window_margins(delta_s, g, off, bounds, next_start_w):
    """``[(phase_idx, grid_margin_s | None, state_margin_s)]`` of every phase whose next phase has a known start.

    ``grid_margin_s`` = next-phase start minus the event-clock time of the last raw row the capped grid
    point of the phase may read (``cut + SNAPSHOT_HALF_WIN`` on G, offset interpolated); ``None`` if the
    phase is too short for a grid point.  ``state_margin_s`` = next-phase start minus the exact W time
    of the latest game state with ``G <= phase_end_s - delta`` (``+inf`` if there is none).
    ``g`` must be strictly increasing (see :func:`_dedupe_pairs`).
    """
    out = []
    for ph in sorted(bounds):
        nxt = next_start_w.get(ph + 1)
        if nxt is None:
            continue
        start, end = bounds[ph]
        t = last_allowed_grid_point(start, end, delta_s)
        grid_margin = None
        if t is not None:
            horizon = t + SNAPSHOT_HALF_WIN
            grid_margin = float(nxt - (horizon + np.interp(horizon, g, off)))
        win_end = end - delta_s
        i = int(np.searchsorted(g, win_end, side="right")) - 1
        state_margin = float("inf") if i < 0 else float(nxt - (g[i] + off[i]))
        out.append((int(ph), grid_margin, state_margin))
    return out


def safe_clock_delta(
    g,
    off,
    bounds,
    next_start_w,
    *,
    grid_margin_s: float = GRID_MARGIN_S,
    state_margin_s: float = STATE_MARGIN_S,
    step_s: float = RAISE_STEP_S,
    max_raise_s: float = MAX_RAISE_S,
) -> ClockDelta:
    """The audit's median offset, raised in ``step_s`` steps until no phase window reads past the next phase.

    ``g`` / ``off``: game-state pairs (game clock, W - G); ``bounds``: ``{phase_idx: (start_s, end_s)}`` on W;
    ``next_start_w``: ``{phase_idx: W time of its first LogPhaseChange}``.
    """
    g_arr = np.asarray(g, dtype=np.float64)
    off_arr = np.asarray(off, dtype=np.float64)
    finite = np.isfinite(g_arr) & np.isfinite(off_arr)
    if not finite.any():
        return ClockDelta(0.0, 0.0, 0)
    median = float(np.median(off_arr[finite]))
    gg, oo = _dedupe_pairs(g_arr, off_arr)
    n_checked = 0
    for i in range(int(max_raise_s / step_s) + 1):
        delta = median + i * step_s
        margins = window_margins(delta, gg, oo, bounds, next_start_w)
        n_checked = len(margins)
        ok = all(
            (grid is None or grid >= grid_margin_s) and state >= state_margin_s for _ph, grid, state in margins
        )
        if ok:
            return ClockDelta(median, delta, int(finite.sum()), n_checked)
    return ClockDelta(median, median + max_raise_s, int(finite.sum()), n_checked, saturated=True)


_CLOCK_CACHE: dict[str, ClockDelta] = {}
_CLOCK_CACHE_MAX = 512


def clear_clock_cache() -> None:
    _CLOCK_CACHE.clear()


def fetch_clock_delta_detail(cur, match_id: str, *, use_cache: bool = True) -> ClockDelta:
    """:class:`ClockDelta` of a match (audit median + the drift-safe value); zeros if it cannot be estimated.

    The result is cached per process (telemetry and the phase tags never change while an ETL runs);
    ``use_cache=False`` forces a fresh read (and replaces the cached value), e.g. after rewriting
    ``stdb.dynamic_regions`` tags.
    """
    if use_cache and match_id in _CLOCK_CACHE:
        return _CLOCK_CACHE[match_id]
    cur.execute(CLOCK_PAIRS_SQL, (match_id,))
    pairs = cur.fetchall()
    if not pairs:
        result = ClockDelta(0.0, 0.0, 0)
    else:
        g = [float(a) for a, _ in pairs]
        off = [float(b) for _, b in pairs]
        cur.execute(PHASE_BOUNDS_SQL, (match_id,))
        bounds = {int(p): (float(a), float(b)) for p, a, b in cur.fetchall()}
        cur.execute(PHASE_START_W_SQL, (match_id,))
        next_start_w = {int(p): float(w) for p, w in cur.fetchall()}
        result = safe_clock_delta(g, off, bounds, next_start_w)
    if len(_CLOCK_CACHE) >= _CLOCK_CACHE_MAX and match_id not in _CLOCK_CACHE:
        _CLOCK_CACHE.pop(next(iter(_CLOCK_CACHE)))
    _CLOCK_CACHE[match_id] = result  # a forced re-read replaces the cached value
    return result


def fetch_clock_delta(cur, match_id: str) -> float:
    """Per-match clock lag ``delta`` in seconds (W = G + delta) used by every phase-tier window.

    The audit's median, raised where the clock drifts (see the module docstring); 0.0 if it cannot be
    estimated.  ``fetch_clock_delta_detail(...).median_s`` is the plain median.
    """
    value = fetch_clock_delta_detail(cur, match_id).safe_s
    return value if math.isfinite(value) else 0.0


def sql_delta_literal(delta_s: float) -> str:
    """Parenthesised numeric SQL literal for ``delta_s`` (rounded to 1 ms)."""
    return f"({float(delta_s):.3f})"


def sql_window_columns(
    delta_expr: str, *, start_col: str = "phase_start_s", end_col: str = "phase_end_s"
) -> str:
    """Select-list fragment adding the clock-aligned window columns to a phase-bounds CTE.

    ``delta_expr`` is a SQL scalar: :func:`sql_delta_literal` output or ``%(delta_s)s``.
    """
    return (
        f"{start_col} - {delta_expr} AS {WIN_START_COLUMN},\n"
        f"       {end_col} - {delta_expr} AS {WIN_END_COLUMN},\n"
        f"       {end_col} - {delta_expr} - {GUARD_S} AS {CUT_COLUMN}"
    )


# ---- python mirrors (tests, dump script) ------------------------------------------------
def window_g(phase_start_s: float, phase_end_s: float, delta_s: float = 0.0) -> tuple[float, float, float]:
    """``(win_start_g, win_end_g, phase_cut_s)`` of a phase on the game clock."""
    return (
        float(phase_start_s) - delta_s,
        float(phase_end_s) - delta_s,
        float(phase_end_s) - delta_s - GUARD_S,
    )


def phase_cut_s(phase_end_s: float, delta_s: float = 0.0) -> float:
    """Latest game-clock time a phase-tier grid / position window may read."""
    return float(phase_end_s) - delta_s - GUARD_S


def sql_cut(end_expr: str = "phase_end_s") -> str:
    """SQL fragment ``<end_expr> - GUARD_S`` (kept for single-clock call sites)."""
    return f"{end_expr} - {GUARD_S}"


def last_allowed_grid_point(
    phase_start_s: float,
    phase_end_s: float,
    delta_s: float = 0.0,
    interval: int = SNAPSHOT_INTERVAL,
) -> float | None:
    """Last grid time ``t`` (a multiple of ``interval``) with ``win_start_g <= t <= phase_cut_s``.

    ``None`` if the phase is too short for one.  ``delta_s = 0`` reproduces the pure Fix B cap.
    """
    start_g, _end_g, cut = window_g(phase_start_s, phase_end_s, delta_s)
    t = math.floor(cut / interval + 1e-9) * interval
    if t < start_g - 1e-9:
        return None
    return float(t)


def raw_horizon_s(grid_t: float) -> float:
    """Latest raw time (game clock) the resampler may have used for grid point ``grid_t``."""
    return float(grid_t) + SNAPSHOT_HALF_WIN


def cap_is_causal(phase_end_s: float, delta_s: float = 0.0, true_boundary_g: float | None = None) -> bool:
    """True iff the capped grid cannot reach raw data at / after the phase boundary.

    ``true_boundary_g`` is the real boundary on the game clock; by default it is derived from the
    stored (W-valued) ``phase_end_s``: ``phase_end_s - delta``.
    """
    boundary = float(phase_end_s) - delta_s if true_boundary_g is None else float(true_boundary_g)
    last = last_allowed_grid_point(phase_end_s - 10_000, phase_end_s, delta_s)
    return last is not None and raw_horizon_s(last) < boundary
