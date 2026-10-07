"""The 10 s team grid on the LogMatchStart clock W (v2 replacement of stdb.moving_objects / team_proximity).

stdb.etl resamples raw positions onto a 10 s grid: grid time ``t`` takes, per player, the raw sample NEAREST to
``t`` inside ``[t - 5, t + 5]`` (an earlier sample wins an exact tie); the team row is the plain mean of the
alive players' (x, y, z) (all players if none is alive), a team is alive iff some sampled player is.
The v1 grid lived on the game clock G (``elapsed_time``) and could read up to 5 s past its cap; here the grid
lives on W and a grid point is ADMISSIBLE for a cut ``a`` only if every sample it may read is earlier than the
cut: ``t + HALF_WIN < a``.  Zone state at a grid point = the game state record nearest to it within +-5 s (same
admissibility).  No cap on the number of grid points: the grid runs as long as the (truncated) log does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTERVAL = 10.0
HALF_WIN = 5.0
_US = 1_000_000


def last_grid_time(cut_w: float, interval: float = INTERVAL, half: float = HALF_WIN) -> float:
    """Largest multiple of ``interval`` with ``t + half < cut_w`` (strict)."""
    m = int(np.floor((cut_w - half) / interval))
    while m * interval + half >= cut_w:
        m -= 1
    return m * interval


def first_grid_time(start_w: float, interval: float = INTERVAL) -> float:
    """Smallest multiple of ``interval`` that is >= ``start_w``."""
    return float(np.ceil(start_w / interval - 1e-12) * interval)


@dataclass
class TeamGrid:
    g: np.ndarray  # (G,) grid times
    present: np.ndarray  # (G, N) bool: some player of the team was sampled at g
    alive: np.ndarray  # (G, N) bool: some sampled player is alive (health NaN or > 0)
    cx: np.ndarray  # (G, N) float64, NaN where the team is not present
    cy: np.ndarray
    cz: np.ndarray

    @property
    def n_grid(self) -> int:
        return int(len(self.g))


def nearest_sample(
    key_sorted: np.ndarray, key_query: np.ndarray, acct_sorted: np.ndarray, acct_query: np.ndarray, half_us: int
) -> np.ndarray:
    """Index into the sorted sample table of the nearest same-account sample within ``half_us`` (-1: none).
    ``key = acct * K + w_us``; ties go to the earlier sample."""
    n = len(key_sorted)
    if n == 0 or len(key_query) == 0:
        return np.full(len(key_query), -1, dtype=np.int64)
    idx = np.searchsorted(key_sorted, key_query, side="left")
    hi = np.clip(idx, 0, n - 1)
    lo = np.clip(idx - 1, 0, n - 1)
    d_hi = np.where(acct_sorted[hi] == acct_query, np.abs(key_sorted[hi] - key_query), np.iinfo(np.int64).max)
    d_lo = np.where(acct_sorted[lo] == acct_query, np.abs(key_sorted[lo] - key_query), np.iinfo(np.int64).max)
    use_lo = d_lo <= d_hi  # earlier sample wins a tie
    best = np.where(use_lo, lo, hi)
    d = np.where(use_lo, d_lo, d_hi)
    return np.where(d <= half_us, best, -1).astype(np.int64)


def build_team_grid(
    pos: dict[str, np.ndarray],
    team_index: dict[int, int],
    g_first: float,
    g_last: float,
    *,
    interval: float = INTERVAL,
    half: float = HALF_WIN,
) -> TeamGrid:
    """Grid rows ``g_first, g_first + interval, ..., <= g_last`` from the (already truncated) position table."""
    N = len(team_index)
    if not g_last >= g_first or len(pos["w"]) == 0:
        z = np.zeros((0, N))
        return TeamGrid(np.zeros(0), z.astype(bool), z.astype(bool), z, z.copy(), z.copy())
    n_g = int(np.floor((g_last - g_first) / interval + 1e-9)) + 1
    g = g_first + interval * np.arange(n_g)
    w_us = np.rint(pos["w"] * _US).astype(np.int64)
    acct = pos["acct"]
    BIG = np.int64(10**12)
    key = acct.astype(np.int64) * BIG + w_us
    order = np.argsort(key, kind="stable")
    key_s, acct_s = key[order], acct[order]
    accts = np.unique(acct)
    qa = np.repeat(accts, n_g)
    qg = np.tile(np.rint(g * _US).astype(np.int64), len(accts))
    qkey = qa.astype(np.int64) * BIG + qg
    pick = nearest_sample(key_s, qkey, acct_s, qa, int(round(half * _US)))
    ok = pick >= 0
    sidx = order[pick[ok]]  # row of the original table
    gi = np.tile(np.arange(n_g), len(accts))[ok]
    team_row = pos["team"][sidx]
    tmap = np.fromiter((team_index.get(int(t), -1) for t in team_row), dtype=np.int64, count=len(team_row))
    keep = tmap >= 0
    sidx, gi, ti = sidx[keep], gi[keep], tmap[keep]
    h = pos["health"][sidx]
    alive = ~np.isfinite(h) | (h > 0)  # NULL health counts as alive (stdb.etl)
    cell = gi * N + ti
    size = n_g * N

    def bc(weights=None, sel=None):
        if sel is None:
            return np.bincount(cell, weights=weights, minlength=size)
        return np.bincount(cell[sel], weights=None if weights is None else weights[sel], minlength=size)

    cnt_all = bc()
    cnt_alive = bc(sel=alive)
    out = {}
    for name in ("x", "y", "z"):
        v = pos[name][sidx]
        s_all = bc(v)
        s_alive = bc(v, sel=alive)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(cnt_alive > 0, s_alive / cnt_alive, np.where(cnt_all > 0, s_all / cnt_all, np.nan))
    shape = (n_g, N)
    return TeamGrid(
        g=g,
        present=(cnt_all > 0).reshape(shape),
        alive=(cnt_alive > 0).reshape(shape),
        cx=out["x"].reshape(shape),
        cy=out["y"].reshape(shape),
        cz=out["z"].reshape(shape),
    )


# --------------------------------------------------------------------------------------
# zone state
# --------------------------------------------------------------------------------------
def zone_at_grid(
    gs: dict[str, np.ndarray], g: np.ndarray, kind: str, half: float = HALF_WIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zone centre / radius (``kind`` 'safe' = current blue zone, 'poison' = announced white circle) at every grid
    time: the game state with a VALID zone of that kind nearest to the grid time within ``half`` seconds (an
    earlier state wins a tie); NaN where there is none."""
    G = len(g)
    cx, cy, cr = (np.full(G, np.nan) for _ in range(3))
    valid = np.isfinite(gs[f"{kind}_r"])
    if G == 0 or not valid.any():
        return cx, cy, cr
    w = gs["w"][valid]
    order = np.argsort(w, kind="stable")
    w = w[order]
    xs, ys, rs = (gs[f"{kind}_{c}"][valid][order] for c in ("x", "y", "r"))
    idx = np.searchsorted(w, g, side="left")
    hi = np.clip(idx, 0, len(w) - 1)
    lo = np.clip(idx - 1, 0, len(w) - 1)
    d_hi, d_lo = np.abs(w[hi] - g), np.abs(w[lo] - g)
    use_lo = d_lo <= d_hi
    best = np.where(use_lo, lo, hi)
    d = np.where(use_lo, d_lo, d_hi)
    ok = d <= half + 1e-9
    cx[ok], cy[ok], cr[ok] = xs[best[ok]], ys[best[ok]], rs[best[ok]]
    return cx, cy, cr


def distance_cube(grid: TeamGrid) -> np.ndarray:
    """``D[g, i, j]`` 3-D distance between the centroids of teams i and j (both alive at g); NaN elsewhere and on
    the diagonal (stdb.team_proximity: both teams alive, no self pairs)."""
    G, N = grid.alive.shape
    if G == 0:
        return np.zeros((0, N, N))
    x = np.where(grid.alive, grid.cx, np.nan)
    y = np.where(grid.alive, grid.cy, np.nan)
    z = np.where(grid.alive, grid.cz, np.nan)
    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    dz = z[:, :, None] - z[:, None, :]
    D = np.sqrt(dx * dx + dy * dy + dz * dz)
    idx = np.arange(N)
    D[:, idx, idx] = np.nan
    return D
