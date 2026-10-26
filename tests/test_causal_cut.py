"""Fix B + single clock (2026-10-05): phase-tier windows are aligned on ONE clock, then capped.

Background (see stdb/causal_cut.py): the phase bounds ``phase_start_s`` / ``phase_end_s`` are numbers on the
LogMatchStart clock W (``event_time - started_at``), the 10 s grid, the raw positions and the game states are
on the game clock G (``elapsed_time``), which lags W by a per-match ``delta`` (median 7-8 s, max 42-75 s).
The window on G is ``[phase_start_s - delta, phase_end_s - delta - 10]`` (design v2.3).

DB-free part
    * the cap function / SQL fragment excludes the last grid point of a phase;
    * a faithful re-implementation of the ``stdb.etl`` nearest-sample resampler plus a drifting game clock
      shows (i) the leak of the old cap-only rule (about a quarter of the (match, phase) windows read raw
      samples after the true phase boundary) and (ii) that the one-clock window never does;
    * a drifting game clock (offset W - G growing up to ~45 s over a match, as in the real data) leaves a
      residual look-ahead with the plain median ``delta``; ``safe_clock_delta`` raises it just enough and the
      simulated resampler / game states then never read past the true phase boundary;
    * a source / rendered-SQL tripwire: every ETL builder keeps its grid / position windows on the shifted,
      capped game-clock window and no clock-mixed or uncapped join can come back.

DB-gated part (``requires_db``)
    * for sampled matches, perturbing the grid rows between the capped end and the next phase window of every
      phase (``moving_objects``, ``team_proximity``, ``dynamic_regions``, ``team_snapshot_features``) leaves
      EVERY Fix B feature / edge row unchanged, i.e. no row of phase k reads a grid snapshot later than
      ``phase_end_s - delta - 10`` (game clock); a negative control (the same builder with the cap undone)
      must change, so the test cannot pass vacuously;
    * ``team_phase_state.position_rows`` equals an independent count of the raw position seconds inside
      the shifted, capped window;
    * the per-match delta equals the audit's definition (the median), raised only where the clock drifts.

The DB-gated test rebuilds everything inside one transaction that is rolled back; the base grid rows (empty
in the working database) are staged in session-local temp tables by ``stdb.base_stage``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import pubg_lab.data.stdb.temporal.causal_cut as cc
from pubg_lab.data.stdb.pipelines.etl import SNAPSHOT_HALF_WIN, SNAPSHOT_INTERVAL
from tests.conftest import requires_db

REPO_ROOT = Path(__file__).resolve().parent.parent
STDB = REPO_ROOT / "src/pubg_lab/data/stdb"

# phase-tier ETL modules named in the design (docs/experiment_design section 6) + the other phase-tier
# consumers of grid snapshots / raw position windows found by grep.
CUT_ETL_FILES = (
    "phase_spatial_feature_etl.py",
    "habitat_feature_etl.py",
    "phase_edge_etl.py",
    "z_affordance_feature_etl.py",
    "f0_feature_etl.py",
    "rq1_feature_etl.py",
    "patch_combat_zone_pass.py",
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def resample_nearest(raw_times: list[float], grid: list[int]) -> dict[int, float]:
    """The stdb.etl rule: nearest raw sample within +-SNAPSHOT_HALF_WIN (see
    ``_ingest_moving_objects``); grid points without a sample in range are dropped."""
    pts = sorted(raw_times)
    out: dict[int, float] = {}
    i = 0
    for gt in grid:
        while i + 1 < len(pts) and abs(pts[i + 1] - gt) < abs(pts[i] - gt):
            i += 1
        if abs(pts[i] - gt) > SNAPSHOT_HALF_WIN:
            continue
        out[gt] = pts[i]
    return out


class _Recorder:
    """Stand-in cursor that records every executed statement."""

    rowcount = 0

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql, params=None):  # noqa: D401 - cursor API
        self.statements.append(sql)

    def fetchone(self):
        return (0,)

    def fetchall(self):  # no game states -> delta 0.0
        return []


def _recorded_sql(builder, *args) -> str:
    rec = _Recorder()
    builder(rec, *args)
    return "\n".join(rec.statements)


# an uncapped grid join: bounds on the grid's clock compared with the stored (W) end or the uncapped end
UNCAPPED_GRID_JOIN = re.compile(
    r"snapshot_t\s+BETWEEN\s+(\w+)\.\w+\s+AND\s+\1\.(?:phase_end_s|win_end_g)\b", re.IGNORECASE
)
# a clock-mixed window: grid / position time compared with the W-valued phase_start_s
CLOCK_MIXED_JOIN = re.compile(
    r"(?:snapshot_t|elapsed_time)\s+BETWEEN\s+(\w+)\.phase_start_s\b", re.IGNORECASE
)
CAPPED_GRID_JOIN = re.compile(
    r"(?:snapshot_t|elapsed_time)\s+BETWEEN\s+(\w+)\.win_start_g\s+AND\s+\1\.phase_cut_s\b", re.IGNORECASE
)


# --------------------------------------------------------------------------------------
# (a) DB-free: the cap function excludes the last grid point
# --------------------------------------------------------------------------------------
def test_guard_equals_snapshot_interval():
    assert cc.GUARD_S == SNAPSHOT_INTERVAL == 10
    assert cc.sql_cut("pb.phase_end_s") == "pb.phase_end_s - 10"
    assert cc.sql_cut() == "phase_end_s - 10"


def test_window_columns_shift_both_ends_then_cap():
    cols = " ".join(cc.sql_window_columns("%(delta_s)s").split())
    assert "phase_start_s - %(delta_s)s AS win_start_g" in cols
    assert "phase_end_s - %(delta_s)s AS win_end_g" in cols
    assert "phase_end_s - %(delta_s)s - 10 AS phase_cut_s" in cols
    assert cc.sql_delta_literal(8.2749) == "(8.275)"
    assert cc.window_g(100.0, 590.0, 8.5) == (91.5, 581.5, 571.5)


def test_cap_excludes_last_grid_point_of_the_phase():
    phase_start, phase_end = 90.0, 590.0
    grid = list(range(int(phase_start), int(phase_end) + 1, SNAPSHOT_INTERVAL))  # 90..590
    cap = cc.phase_cut_s(phase_end)
    kept = [t for t in grid if t <= cap]
    assert cap == 580.0
    assert grid[-1] == 590 and 590 not in kept
    assert kept[-1] == 580 and len(kept) == len(grid) - 1
    assert cc.last_allowed_grid_point(phase_start, phase_end) == 580.0


def test_last_allowed_grid_point_edge_cases():
    assert cc.last_allowed_grid_point(100.0, 110.0) == 100.0  # two-point phase keeps its first point
    assert cc.last_allowed_grid_point(100.0, 100.0) is None  # a one-point phase has none
    for n_points in range(2, 60):
        start, end = 30.0, 30.0 + SNAPSHOT_INTERVAL * (n_points - 1)
        last = cc.last_allowed_grid_point(start, end)
        assert last == end - SNAPSHOT_INTERVAL < end


def test_one_clock_cap_with_a_clock_lag():
    """delta = 8.5: the game-clock window of phase [100, 590] is [91.5, 571.5]; grid points are multiples of 10."""
    assert cc.last_allowed_grid_point(100.0, 590.0, 8.5) == 570.0
    assert cc.last_allowed_grid_point(100.0, 590.0, 0.0) == 580.0
    assert cc.last_allowed_grid_point(100.0, 110.0, 25.0) is None  # a large lag swallows a short phase
    assert cc.last_allowed_grid_point(100.0, 190.0, 8.5) == 170.0


@pytest.mark.parametrize("phase_end", [90.0, 110.0, 590.0, 1550.0])
def test_capped_grid_cannot_reach_raw_samples_at_or_after_phase_end(phase_end):
    assert cc.cap_is_causal(phase_end)
    assert cc.raw_horizon_s(cc.phase_cut_s(phase_end)) == phase_end - SNAPSHOT_HALF_WIN
    assert cc.raw_horizon_s(phase_end) > phase_end  # the last (uncapped) grid point reaches past it


def test_resampler_leak_that_the_cap_closes():
    """Raw samples at 570, 580 and 594.2 with the phase boundary at 593.4:
    grid 590 (= phase_end_s) is served by the post-boundary sample 594.2."""
    boundary = 593.4
    phase_end_s = 590  # last grid point tagged with phase k (590 < int(593.4) = 593)
    raw = [570.0, 580.0, 594.2]
    grid = list(range(570, 600, SNAPSHOT_INTERVAL))
    sampled = resample_nearest(raw, grid)
    assert sampled[590] == 594.2 > boundary, "fixture must reproduce the post-boundary read"

    cap = cc.phase_cut_s(phase_end_s)
    capped = {t: r for t, r in sampled.items() if t <= cap}
    assert 590 not in capped
    assert all(r < boundary for r in capped.values())
    assert max(capped) == 580


# ---- the one-clock property on a simulated match with a drifting game clock ------------------
def _simulate_match(rng: np.random.Generator, n_phases: int = 8, drift_range=(0.004, 0.017)):
    """A match whose game clock G lags the wall clock W by a drift growing from ~0.3 s to ~28 s.

    Phase changes happen near multiples of 10 s on W (like the real zone timers).  The stored bounds follow
    stdb.etl: grid time t (a multiple of 10 on G) is tagged with phase k iff t < int(W boundary of k+1).
    Returns the offset function, the W boundaries, the stored phase bounds and the per-match delta (median
    of W - G over the game-state events, as in the audit).
    """
    duration = 1700.0
    # seconds of lag gained per second of play; the observed envelope is ~2 s at the match start growing
    # to <= ~28 s after ~1640 s (label audit / probe: 0.017 s/s at most)
    drift = rng.uniform(*drift_range)
    lag0 = rng.uniform(0.3, 4.0)

    def off(g):  # W - G at game time g
        return lag0 + drift * g

    w_bounds = [90.0 + 0.5 + rng.uniform(-0.4, 0.4)]  # phase 1 starts
    for _ in range(n_phases):
        w_bounds.append(w_bounds[-1] + 10 * rng.integers(10, 31) + rng.uniform(-0.4, 0.4))
    game_states_g = np.arange(10.0, duration, 10.0)
    delta = float(np.median(off(game_states_g)))
    grid = np.arange(0, int(duration) + 10, 10)
    phase_end_s = {}
    phase_start_s = {}
    for k in range(1, n_phases):
        tagged = [t for t in grid if int(w_bounds[k - 1]) <= t < int(w_bounds[k])]
        if len(tagged) >= 2:
            phase_start_s[k], phase_end_s[k] = float(tagged[0]), float(tagged[-1])
    return off, w_bounds, phase_start_s, phase_end_s, delta


def _true_boundary_g(off, w_boundary: float) -> float:
    """G time at which the wall clock reads ``w_boundary`` (monotone drift)."""
    g = w_boundary
    for _ in range(50):
        g = w_boundary - off(g)
    return g


def test_single_clock_window_removes_the_leak_that_the_cap_alone_leaves():
    rng = np.random.default_rng(3)
    leaks_cap_only = leaks_one_clock = windows = 0
    for _ in range(300):
        off, w_bounds, start_s, end_s, delta = _simulate_match(rng)
        for k, e in end_s.items():
            windows += 1
            g_true = _true_boundary_g(off, w_bounds[k])  # true end of phase k on the game clock
            t_cap_only = cc.last_allowed_grid_point(start_s[k] - 10_000, e, 0.0)
            t_one_clock = cc.last_allowed_grid_point(start_s[k] - 10_000, e, delta)
            leaks_cap_only += int(cc.raw_horizon_s(t_cap_only) >= g_true)
            leaks_one_clock += int(cc.raw_horizon_s(t_one_clock) >= g_true)
    assert windows > 1500
    assert leaks_cap_only / windows > 0.10, "fixture must reproduce the clock-mixing leak of the cap alone"
    # the constant per-match delta leaves a margin of 15 s + the grid rounding; only a drift steeper than
    # anything seen in the data could leak, so the residual must be negligible (real matches: DB-gated test)
    assert leaks_one_clock / windows < 0.005, f"{leaks_one_clock} / {windows} windows still leak"
    assert leaks_one_clock * 20 < leaks_cap_only, "the single clock must remove >95 % of the cap-only leak"


def _next_starts(w_bounds, end_s):
    """phase k + 1 starts at the W boundary k of the simulated match."""
    return {k + 1: w_bounds[k] for k in end_s if k < len(w_bounds)}


def test_safe_clock_delta_equals_the_median_without_drift():
    g = np.arange(10.0, 1700.0, 10.0)
    off = 8.0 + np.random.default_rng(1).normal(0.0, 0.15, g.size)
    bounds = {1: (90.0, 590.0), 2: (600.0, 790.0), 3: (800.0, 960.0)}
    nxt = {2: 591.1, 3: 791.1, 4: 961.1}
    res = cc.safe_clock_delta(g, off, bounds, nxt)
    assert res.safe_s == res.median_s == pytest.approx(float(np.median(off))) and not res.raised
    assert res.n_phases_checked == 3 and not res.saturated and res.n_pairs == g.size


def test_safe_clock_delta_closes_the_drift_leak_that_the_median_leaves():
    """Drifts up to 0.04 s/s (the real maximum is ~0.034): the plain median leaks, the raised delta never."""
    rng = np.random.default_rng(5)
    windows = raised = 0
    leaks_grid_median = leaks_state_median = leaks_grid_safe = leaks_state_safe = 0
    for _ in range(400):
        off, w_bounds, start_s, end_s, _median_sim = _simulate_match(rng, drift_range=(0.0, 0.04))
        g = np.arange(10.0, 1700.0, 10.0)
        offs = off(g)
        bounds = {k: (start_s[k], end_s[k]) for k in end_s}
        res = cc.safe_clock_delta(g, offs, bounds, _next_starts(w_bounds, end_s))
        assert res.safe_s >= res.median_s and not res.saturated
        raised += int(res.raised)
        for k, e in end_s.items():
            windows += 1
            g_true = _true_boundary_g(off, w_bounds[k])  # end of phase k on the game clock
            for delta, grid_leaks, state_leaks in ((res.median_s, "median", None), (res.safe_s, "safe", None)):
                t = cc.last_allowed_grid_point(start_s[k] - 10_000, e, delta)
                grid_leak = int(cc.raw_horizon_s(t) >= g_true)
                last_state = g[g <= e - delta]
                state_leak = int(last_state.size > 0 and last_state[-1] + off(last_state[-1]) >= w_bounds[k])
                if grid_leaks == "median":
                    leaks_grid_median += grid_leak
                    leaks_state_median += state_leak
                else:
                    leaks_grid_safe += grid_leak
                    leaks_state_safe += state_leak
    assert windows > 2000
    assert leaks_grid_median + leaks_state_median > 0 and leaks_state_median > 0, "fixture must reproduce the drift leak"
    assert raised > 20
    assert leaks_grid_safe == 0 and leaks_state_safe == 0, (leaks_grid_safe, leaks_state_safe)


def test_safe_clock_delta_steps_and_saturation():
    g = np.arange(10.0, 1700.0, 10.0)
    off = 0.9 + 0.034 * g  # the worst real match: 0.9 s at the start, ~45 s at the end
    bounds = {4: (990.0, 1170.0)}
    nxt = {5: 1171.1}
    res = cc.safe_clock_delta(g, off, bounds, nxt)
    steps = (res.safe_s - res.median_s) / cc.RAISE_STEP_S
    assert res.raised and abs(steps - round(steps)) < 1e-9, "raised in whole RAISE_STEP_S steps"
    margins = cc.window_margins(res.safe_s, g, off, bounds, nxt)
    assert margins and all(m_state >= cc.STATE_MARGIN_S and (m_grid is None or m_grid >= cc.GRID_MARGIN_S)
                           for _ph, m_grid, m_state in margins)
    # one step less is not enough (the search stops at the first value that satisfies the margins)
    worse = cc.window_margins(res.safe_s - cc.RAISE_STEP_S, g, off, bounds, nxt)
    assert any(m_state < cc.STATE_MARGIN_S or (m_grid is not None and m_grid < cc.GRID_MARGIN_S)
               for _ph, m_grid, m_state in worse)
    # an impossible request saturates instead of looping forever
    sat = cc.safe_clock_delta(g, off, {4: (990.0, 1170.0)}, {5: 100.0}, max_raise_s=5.0)
    assert sat.saturated and sat.safe_s == pytest.approx(sat.median_s + 5.0)


def test_window_margins_edge_cases():
    g = np.array([100.0, 110.0, 120.0])
    off = np.array([8.0, 8.0, 8.0])
    # too short for a grid point and no game state before the window end
    assert cc.window_margins(8.0, g, off, {1: (90.0, 90.0)}, {2: 200.0}) == [(1, None, float("inf"))]
    # window end on G = 112 -> latest game state G = 110, W = 118 -> state margin 12;
    # cut = floor((120 - 8 - 10) / 10) * 10 = 100, horizon 105, W = 113 -> grid margin 17
    ((ph, grid, state),) = cc.window_margins(8.0, g, off, {1: (60.0, 120.0)}, {2: 130.0})
    assert (ph, grid, state) == (1, pytest.approx(17.0), pytest.approx(12.0))
    # the last phase of a match (no next start) is not constrained
    assert cc.window_margins(8.0, g, off, {3: (10.0, 50.0)}, {2: 60.0}) == []


class _ClockCursor:
    """Fake cursor answering the three queries of fetch_clock_delta_detail."""

    def __init__(self, pairs, bounds, starts):
        self.pairs, self.bounds, self.starts = pairs, bounds, starts
        self.statements: list[str] = []
        self._rows: list = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        text = " ".join(sql.split())
        if "FROM pubg.telem_game_states" in text:
            self._rows = self.pairs
        elif "FROM stdb.v_match_phase_bounds" in text:
            self._rows = self.bounds
        elif "FROM pubg.telem_phase_changes" in text:
            self._rows = self.starts
        else:  # pragma: no cover
            raise AssertionError(sql)

    def fetchall(self):
        return list(self._rows)


def test_fetch_clock_delta_returns_the_raised_value_and_caches_per_match():
    cc.clear_clock_cache()
    g = [float(t) for t in range(10, 1700, 10)]
    off = [0.9 + 0.034 * t for t in g]
    cur = _ClockCursor(list(zip(g, off)), [(4, 990.0, 1170.0)], [(5, 1171.1)])
    detail = cc.fetch_clock_delta_detail(cur, "m-drift")
    assert detail.raised and detail.median_s == pytest.approx(float(np.median(off)))
    assert cc.fetch_clock_delta(cur, "m-drift") == detail.safe_s
    assert len(cur.statements) == 3, "the second call must come from the cache"
    cc.fetch_clock_delta_detail(cur, "m-drift", use_cache=False)
    assert len(cur.statements) == 6
    # no game states -> 0.0 (as before the drift handling)
    empty = _ClockCursor([], [], [])
    assert cc.fetch_clock_delta(empty, "m-none") == 0.0
    cc.clear_clock_cache()


def test_resampler_horizon_bound_holds_for_every_alignment():
    """Random sparse raw streams and every boundary alignment: capped grid points only read
    raw samples before the boundary, while the uncapped last grid point sometimes does not."""
    import random

    rng = random.Random(0)
    uncapped_leaks = 0
    for trial in range(400):
        boundary = 600.0 + rng.uniform(0.0, 9.9)
        tagged = [t for t in range(480, 700, SNAPSHOT_INTERVAL) if t < int(boundary)]
        phase_end_s = tagged[-1]
        raw, t = [], 470.0
        while t < 700.0:  # sparse stream (>= 3 s apart) that straddles the boundary
            raw.append(t)
            t += rng.uniform(3.0, 9.0)
        raw.append(boundary + rng.uniform(0.0, 4.9))  # a guaranteed post-boundary sample
        sampled = resample_nearest(raw, tagged)
        cap = cc.phase_cut_s(phase_end_s)
        for gt, r in sampled.items():
            if gt <= cap:
                assert r < boundary, (trial, boundary, gt, r)
        if phase_end_s in sampled and sampled[phase_end_s] >= boundary:
            uncapped_leaks += 1
    assert uncapped_leaks > 0, "fixture never exercised the leak the cap closes"


# --------------------------------------------------------------------------------------
# (a') DB-free: tripwire on every ETL's windows
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("fname", CUT_ETL_FILES)
def test_etl_source_imports_the_guard(fname):
    src = (STDB / "pipelines" / fname).read_text(encoding="utf-8")
    assert 'pubg_lab.data.stdb.temporal.causal_cut' in src, f"{fname} must use the shared Fix B / single-clock guard"
    assert "fetch_clock_delta" in src, f"{fname} must estimate the per-match clock lag"


def test_no_uncapped_or_clock_mixed_window_anywhere_in_stdb_sources():
    offenders = []
    for path in sorted(STDB.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for rx, what in ((UNCAPPED_GRID_JOIN, "uncapped grid join"), (CLOCK_MIXED_JOIN, "clock-mixed window")):
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.name}:{line}: {what}: {m.group(0)!r}")
    assert not offenders, "windows that would read past the cut or mix the two clocks:\n" + "\n".join(offenders)


def _rendered_sql() -> dict[str, str]:
    import pubg_lab.data.stdb.pipelines.f0_feature_etl as f0_feature_etl; import pubg_lab.data.stdb.pipelines.habitat_feature_etl as habitat_feature_etl; import pubg_lab.data.stdb.pipelines.patch_combat_zone_pass as patch_combat_zone_pass; import pubg_lab.data.stdb.pipelines.phase_edge_etl as phase_edge_etl; import pubg_lab.data.stdb.pipelines.phase_spatial_feature_etl as phase_spatial_feature_etl; import pubg_lab.data.stdb.pipelines.rq1_feature_etl as rq1_feature_etl; import pubg_lab.data.stdb.pipelines.z_affordance_feature_etl as z_affordance_feature_etl

    return {
        "spatial": _recorded_sql(phase_spatial_feature_etl.build_match, "m"),
        "edge": _recorded_sql(phase_edge_etl.build_match, "m"),
        "habitat": habitat_feature_etl.BUILD_SQL,
        "z_affordance": z_affordance_feature_etl.BUILD_SQL,
        "f0": f0_feature_etl.BUILD_SQL,
        "patch_combat": patch_combat_zone_pass.PATCH_SQL,
        "rq1_global": _recorded_sql(rq1_feature_etl.build_match_phase_global, "m"),
        "rq1_state": _recorded_sql(rq1_feature_etl.build_team_phase_state, "m"),
    }


# minimum number of capped, shifted windows each builder must keep (exact counts would be brittle)
EXPECTED_CAPPED_JOINS = {
    "spatial": 2,       # trajectory + exposure snapshots
    "edge": 2,          # proximity + co-pressure
    "habitat": 2,       # team snapshots + cut-time white circle
    "z_affordance": 2,  # snapshot features + contact pairs
    "f0": 3,            # team centroid + safe / poison zone centres
    "patch_combat": 2,
    "rq1_global": 4,    # safe / poison start and end
    "rq1_state": 1,     # raw position window of team_phase_state
}


def test_rendered_sql_has_shifted_capped_windows_and_no_uncapped_or_clock_mixed_ones():
    for name, sql in _rendered_sql().items():
        assert not UNCAPPED_GRID_JOIN.search(sql), f"{name}: uncapped grid join in rendered SQL"
        assert not CLOCK_MIXED_JOIN.search(sql), f"{name}: grid / position time compared with phase_start_s"
        assert len(CAPPED_GRID_JOIN.findall(sql)) >= EXPECTED_CAPPED_JOINS[name], (
            f"{name}: expected >= {EXPECTED_CAPPED_JOINS[name]} shifted, capped windows"
        )
        assert re.search(r"phase_end_s\s*-\s*(\(\d+\.\d+\)|%\(delta_s\)s)\s*-\s*10\s+AS\s+phase_cut_s", sql), (
            f"{name}: the shifted cap expression is missing"
        )


def test_rq1_game_state_window_is_shifted_not_capped():
    sql = _rendered_sql()["rq1_global"]
    assert re.search(r"gs\.elapsed_time\s+BETWEEN\s+b\.win_start_g\s+AND\s+b\.win_end_g", sql)
    assert len(re.findall(r"dr\.snapshot_t\s+BETWEEN\s+b\.win_start_g\s+AND\s+b\.phase_cut_s", sql)) == 4


def test_f0_zone_centre_reads_use_the_capped_window():
    sql = _rendered_sql()["f0"]
    assert len(re.findall(r"dr\.snapshot_t\s+BETWEEN\s+b\.win_start_g\s+AND\s+b\.phase_cut_s", sql)) == 2


def test_spatial_event_buckets_are_moved_onto_the_grid_clock():
    sql = _rendered_sql()["spatial"]
    assert re.search(r"floor\(\(e\.event_t - \(0\.000\)\) / %s\)", sql), "combat buckets must use event_t - delta"
    # timing features compare game-clock times with the game-clock window start
    assert "g.win_start_g" in sql and not re.search(r"-\s*g\.phase_start_s", sql)


def test_network_etl_has_no_direct_grid_join():
    """Network features aggregate stdb.team_phase_edge_cache only; their causal guarantee
    is inherited from the capped edge cache (stdb/network_feature_etl.py docstring)."""
    from pubg_lab.data.stdb.pipelines.network_feature_etl import BUILD_SQL

    sql = BUILD_SQL.lower()
    for grid_table in (
        "moving_objects",
        "team_proximity",
        "dynamic_regions",
        "team_snapshot_features",
    ):
        assert grid_table not in sql, f"network features must not read {grid_table} directly"
    assert "team_phase_edge_cache" in sql


def test_clock_delta_sql_follows_the_audit_definition():
    sql = " ".join(cc.CLOCK_DELTA_SQL.split())
    assert "percentile_cont(0.5)" in sql
    assert "extract(epoch FROM (gs.event_time - m.started_at)) - gs.elapsed_time" in sql
    assert "pubg.telem_game_states" in sql and "gs.elapsed_time IS NOT NULL" in sql


# --------------------------------------------------------------------------------------
# (b) DB-gated: no feature row reads a grid snapshot later than phase_end_s - delta - 10
# --------------------------------------------------------------------------------------
def _sample_match_ids(cur, n: int = 3) -> list[str]:
    cur.execute(
        """
        SELECT m.match_id
          FROM stdb.matches m
         WHERE m.map_name = 'Baltic_Main' AND m.game_mode = 'squad'
           AND m.num_teams BETWEEN 20 AND 30
           AND EXISTS (SELECT 1 FROM stdb.team_phase_edge_cache e WHERE e.match_id = m.match_id)
           AND EXISTS (SELECT 1 FROM stdb.team_snapshot_features s WHERE s.match_id = m.match_id)
         ORDER BY m.match_id
         LIMIT %s
        """,
        (n,),
    )
    return [r[0] for r in cur.fetchall()]


FIXB_TABLES = (
    "stdb.match_phase_global",
    "stdb.team_phase_state",
    "stdb.team_phase_trajectory",
    "stdb.team_phase_exposure",
    "stdb.team_phase_edge_cache",
    "stdb.team_phase_habitat_features",
    "stdb.team_phase_network_features",
    "stdb.team_phase_z_affordance_features",
    "stdb.team_phase_f0_extra",
)


def _rebuild_fixb_chain(cur, mid: str) -> None:
    """Same dependency order as tools/run_fixb_phase_etl.py."""
    import pubg_lab.data.stdb.pipelines.f0_feature_etl as f0_feature_etl; import pubg_lab.data.stdb.pipelines.habitat_feature_etl as habitat_feature_etl; import pubg_lab.data.stdb.pipelines.network_feature_etl as network_feature_etl; import pubg_lab.data.stdb.pipelines.phase_edge_etl as phase_edge_etl; import pubg_lab.data.stdb.pipelines.phase_spatial_feature_etl as phase_spatial_feature_etl; import pubg_lab.data.stdb.pipelines.rq1_feature_etl as rq1_feature_etl; import pubg_lab.data.stdb.pipelines.z_affordance_feature_etl as z_affordance_feature_etl

    rq1_feature_etl.build_match_phase_global(cur, mid)
    rq1_feature_etl.build_team_phase_state(cur, mid)
    phase_spatial_feature_etl.build_match(cur, mid)
    phase_edge_etl.build_match(cur, mid)
    habitat_feature_etl.build_match(cur, mid)
    network_feature_etl.build_match(cur, mid)
    z_affordance_feature_etl.build_match(cur, mid)
    f0_feature_etl.build_match(cur, mid)


def _snapshot_fixb_outputs(cur, mid: str) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for table in FIXB_TABLES:
        cur.execute(f"SELECT * FROM {table} WHERE match_id = %s", (mid,))
        cols = [d[0] for d in cur.description]
        drop = {i for i, c in enumerate(cols) if c in {"created_at", "blue_zone_options"}}
        rows = [tuple(v for i, v in enumerate(r) if i not in drop) for r in cur.fetchall()]
        out[table] = sorted(rows, key=lambda r: tuple("" if v is None else str(v) for v in r))
    return out


def _forbidden_intervals(cur, mid: str, delta_s: float) -> list[tuple[float, float]]:
    """Per phase: the grid times strictly between the capped end of its window and the start of the next
    phase's window.  No builder may read these rows (open interval of width 20 s on the game clock)."""
    cur.execute(
        """
        SELECT phase_idx, phase_start_s, phase_end_s
          FROM stdb.v_match_phase_bounds
         WHERE match_id = %s AND phase_idx > 0
         ORDER BY phase_idx
        """,
        (mid,),
    )
    out = []
    for _p, _start, end in cur.fetchall():
        cut = cc.phase_cut_s(float(end), delta_s)
        out.append((cut, cut + 2 * SNAPSHOT_INTERVAL))
    return out


def _perturb_forbidden_rows(cur, mid: str, intervals: list[tuple[float, float]]) -> dict[str, int]:
    """Rewrite every grid row with cut < snapshot_t < cut + 20 of some phase."""
    lo = [a for a, _ in intervals]
    hi = [b for _, b in intervals]
    where = (
        "match_id = %(m)s AND EXISTS (SELECT 1 FROM unnest(%(lo)s::float8[], %(hi)s::float8[]) AS w(a, b) "
        "WHERE snapshot_t > w.a AND snapshot_t < w.b)"
    )
    args = {"m": mid, "lo": lo, "hi": hi}
    n = {}
    cur.execute(
        f"UPDATE stdb.moving_objects SET pos = ST_Translate(pos, 733.0, -311.0, 25.0) WHERE {where}", args
    )
    n["moving_objects"] = cur.rowcount
    cur.execute(f"UPDATE stdb.team_proximity SET distance_m = distance_m * 0.4 + 7.0 WHERE {where}", args)
    n["team_proximity"] = cur.rowcount
    cur.execute(
        f"""
        UPDATE stdb.dynamic_regions
           SET center_pos = ST_Translate(center_pos, 411.0, 207.0, 0.0),
               region     = ST_Translate(region, 411.0, 207.0),
               radius_m   = radius_m * 0.9
         WHERE {where}
        """,
        args,
    )
    n["dynamic_regions"] = cur.rowcount
    cur.execute(
        f"UPDATE stdb.team_snapshot_features SET pos_z = pos_z + 55.0, pos_x = pos_x + 700.0 WHERE {where}", args
    )
    n["team_snapshot_features"] = cur.rowcount
    return n


@pytest.fixture()
def fixb_cursor():
    """A cursor whose ``stdb.moving_objects`` / ``spatial_events`` / ``team_proximity`` point at
    session-local temp tables (the shared ones are empty and stay untouched); everything else is
    the real database inside a transaction that is rolled back."""
    from pubg_lab.data.stdb.pipelines.base_stage import TempBaseCursor, create_temp_base_tables
    from pubg_lab.data.stdb.connection.conn import connect

    conn = connect()
    try:
        create_temp_base_tables(conn)
        with conn.cursor() as raw:
            yield TempBaseCursor(raw)
    finally:
        conn.rollback()
        conn.close()


@requires_db
def test_clock_delta_matches_the_audit_definition(fixb_cursor):
    cur = fixb_cursor
    mids = _sample_match_ids(cur, 2)
    if not mids:
        pytest.skip("no sample match")
    for mid in mids:
        cur.execute(
            """
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM gs.event_time) - gs.elapsed_time)
                   - extract(epoch FROM m.started_at)
              FROM pubg.telem_game_states gs, stdb.matches m
             WHERE gs.match_id = %s AND m.match_id = %s AND gs.elapsed_time IS NOT NULL
             GROUP BY m.started_at
            """,
            (mid, mid),
        )
        audit = float(cur.fetchone()[0])  # the audit's delta = O_gs - S
        detail = cc.fetch_clock_delta_detail(cur, mid, use_cache=False)
        assert abs(detail.median_s - audit) < 1e-3
        assert detail.safe_s >= detail.median_s - 1e-9 and detail.safe_s - detail.median_s <= cc.MAX_RAISE_S
        assert cc.fetch_clock_delta(cur, mid) == detail.safe_s
        assert 0.0 < audit < 120.0


@requires_db
def test_no_feature_row_reads_a_grid_snapshot_after_the_cut(fixb_cursor):
    """Perturb every grid row between the capped end and the next window; no Fix B row may change."""
    from pubg_lab.data.stdb.pipelines.base_stage import analyze_temp_base_tables, reset_temp_base_tables, stage_base_rows
    from pubg_lab.data.stdb.pipelines.habitat_feature_etl import BUILD_SQL as HABITAT_SQL
    from pubg_lab.data.stdb.pipelines.habitat_feature_etl import EDGE_PRESSURE_M

    cur = fixb_cursor
    mids = _sample_match_ids(cur, 3)
    if not mids:
        pytest.skip("no sample Baltic_Main squad match with edge cache + snapshot features")

    control_changed = 0
    for mid in mids:
        reset_temp_base_tables(cur)
        staged = stage_base_rows(cur, mid)
        assert staged["team_rows"] > 0 and staged["proximity_rows"] > 0, staged
        analyze_temp_base_tables(cur)
        _rebuild_fixb_chain(cur, mid)
        before = _snapshot_fixb_outputs(cur, mid)
        assert all(before[t] for t in FIXB_TABLES if "z_affordance" not in t), (
            "fixture match must populate every Fix B table"
        )

        delta = cc.fetch_clock_delta(cur, mid)
        intervals = _forbidden_intervals(cur, mid, delta)
        assert intervals, "match has no phase bounds"
        touched = _perturb_forbidden_rows(cur, mid, intervals)
        assert touched["moving_objects"] > 0 and touched["team_proximity"] > 0, touched
        assert touched["dynamic_regions"] > 0, touched

        _rebuild_fixb_chain(cur, mid)
        after = _snapshot_fixb_outputs(cur, mid)
        for table in FIXB_TABLES:
            assert before[table] == after[table], (
                f"{mid}: {table} changed after perturbing grid rows beyond phase_end_s - delta - 10 "
                "(a feature row read a grid snapshot after the cut)"
            )

        # negative control: the habitat builder with the cap undone must see the perturbation
        cur.execute("DELETE FROM stdb.team_phase_habitat_features WHERE match_id = %s", (mid,))
        uncapped = HABITAT_SQL.replace("b.phase_cut_s", "b.win_end_g")
        assert uncapped != HABITAT_SQL
        cur.execute(uncapped, {"match_id": mid, "edge_pressure_m": EDGE_PRESSURE_M, "delta_s": delta})
        cur.execute("SELECT * FROM stdb.team_phase_habitat_features WHERE match_id = %s", (mid,))
        cols = [d[0] for d in cur.description]
        keep = [i for i, c in enumerate(cols) if c != "created_at"]
        uncapped_rows = sorted(tuple(r[i] for i in keep) for r in cur.fetchall())
        capped_rows = after["stdb.team_phase_habitat_features"]
        if sorted(map(str, uncapped_rows)) != sorted(map(str, capped_rows)):
            control_changed += 1

    assert control_changed > 0, (
        "negative control failed: the uncapped habitat builder was insensitive to the "
        "perturbation, so the test above proves nothing"
    )


@requires_db
def test_state_position_window_is_shifted_and_capped(fixb_cursor):
    """team_phase_state.position_rows == independent count of raw position seconds in the window."""
    from pubg_lab.data.stdb.pipelines.rq1_feature_etl import build_team_phase_state

    cur = fixb_cursor
    mids = _sample_match_ids(cur, 2)
    if not mids:
        pytest.skip("no sample match")
    for mid in mids:
        build_team_phase_state(cur, mid)
        delta = cc.fetch_clock_delta(cur, mid)
        cur.execute(
            """
            SELECT s.team_id, s.phase_idx, s.position_rows,
                   (SELECT count(DISTINCT p.elapsed_time)
                      FROM pubg.telem_positions p
                     WHERE p.match_id = s.match_id AND p.team_id = s.team_id
                       AND p.account_id IS NOT NULL AND p.is_game >= 0.1
                       AND p.pos_x IS NOT NULL AND p.pos_y IS NOT NULL AND p.elapsed_time IS NOT NULL
                       AND p.elapsed_time BETWEEN b.phase_start_s - %(d)s AND b.phase_end_s - %(d)s - 10)
              FROM stdb.team_phase_state s
              JOIN stdb.v_match_phase_bounds b ON b.match_id = s.match_id AND b.phase_idx = s.phase_idx
             WHERE s.match_id = %(m)s
            """,
            {"m": mid, "d": delta},
        )
        rows = cur.fetchall()
        assert rows
        bad = [(t, p, a, b) for t, p, a, b in rows if int(a) != int(b)]
        assert not bad, f"{mid}: position_rows differ from the independent window count: {bad[:5]}"


@requires_db
def test_raised_delta_closes_the_look_ahead_on_real_drifting_matches():
    """EXACT check from raw rows: with the drift-aware delta no capped grid point reads a raw position row, and
    the uncapped game-state window holds no game state, at/after the first LogPhaseChange of the next phase;
    the plain median delta of the same matches does leak (so the test cannot pass vacuously)."""
    import json

    from pubg_lab.data.stdb.connection.conn import connect
    from projects.battle_royale.evaluation.contracts.verify_dump_match import max_w_before, raw_positions

    split = json.loads(
        (REPO_ROOT / "model_stgcn/checkpoints/br_om_ladder_required_v1/miramar_desert_tpp_squad/split.json").read_text()
    )
    candidates = split["splits"]["test"] + split["splits"]["val"] + split["splits"]["train"][:300]
    conn = connect()
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            raised = []
            for mid in candidates:
                detail = cc.fetch_clock_delta_detail(cur, mid, use_cache=False)
                if detail.raised and detail.safe_s - detail.median_s >= 2.0:
                    raised.append((mid, detail))
                if len(raised) == 2:
                    break
            if not raised:
                pytest.skip("no raised-delta match among the first candidates of the Miramar cohort")
            leaks_median = 0
            for mid, detail in raised:
                cur.execute("SELECT extract(epoch FROM started_at)::float8 FROM stdb.matches WHERE match_id = %s", (mid,))
                s_ep = cur.fetchone()[0]
                teams = raw_positions(cur, mid, s_ep)
                cur.execute("SELECT phase_idx, phase_start_s::float8, phase_end_s::float8 FROM stdb.v_match_phase_bounds WHERE match_id = %s", (mid,))
                bounds = {int(p): (a, b) for p, a, b in cur.fetchall()}
                cur.execute(
                    "SELECT phase::int, min(extract(epoch FROM event_time))::float8 FROM pubg.telem_phase_changes "
                    "WHERE match_id = %s AND phase IS NOT NULL GROUP BY phase", (mid,))
                start_w = {int(p): float(e) - s_ep for p, e in cur.fetchall()}
                cur.execute("SELECT elapsed_time::float8, extract(epoch FROM event_time)::float8 FROM pubg.telem_game_states "
                            "WHERE match_id = %s AND elapsed_time IS NOT NULL", (mid,))
                states = [(g, w - s_ep) for g, w in cur.fetchall()]
                n_checked = 0
                for ph, (a, b) in bounds.items():
                    nxt = start_w.get(ph + 1)
                    if nxt is None:
                        continue
                    n_checked += 1
                    for delta, strict in ((detail.safe_s, True), (detail.median_s, False)):
                        cut = cc.last_allowed_grid_point(a, b, delta)
                        grid_leak = cut is not None and max_w_before(teams, cut + SNAPSHOT_HALF_WIN) >= nxt
                        inside = [w for g, w in states if g <= b - delta]
                        state_leak = bool(inside) and max(inside) >= nxt
                        if strict:
                            assert not grid_leak, f"{mid} phase {ph}: raw position row at/after the next phase start"
                            assert not state_leak, f"{mid} phase {ph}: game state at/after the next phase start"
                        else:
                            leaks_median += int(grid_leak or state_leak)
                assert n_checked >= 3
            assert leaks_median > 0, "the plain median must leak on these matches (otherwise nothing was tested)"
    finally:
        conn.close()
