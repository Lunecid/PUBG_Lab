"""Causal alive status of players and teams, and the final team elimination time (labels / risk set only).

Causal player alive status at time t (spec rule 3)
    * alive iff no kill event of the player is <= t;
    * after a kill the player is alive again ONLY if a later position sample with health > 0 exists at a time
      <= t (revive / redeploy evidence observable by t);
    * a team is alive at t iff at least one member is alive.
  Knocked (groggy) players are alive: only kill events end a life.

Final team elimination time T (labels and risk set only, NEVER an input)
    the time of the kill that leaves the team with no member alive and no later member activity, i.e.
    ``T = max over members of the member's FINAL death``; a member's final death is its last kill event unless a
    position sample with health > 0 follows it (then the member is alive at the end -> T = +inf, censored at
    the match end C_m).  Members without any kill event are alive at the end, except participants whose
    ``death_type`` says they left (logout and the like), who die at ``time_survived`` + the match's median
    (kill event - time_survived) over humans (the label audit's fallback, label side only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INF = float("inf")
FALLBACK_OFFSET_S = 0.70  # label audit: median (kill event - time_survived) of human players
# A dead body keeps reporting positions for ~60 s and a sample logged right after the kill can still carry the last
# health value (1 case in 66,741 kill -> healthy-sample pairs: solo match 8df7e412, sample 1.6 s after the kill with
# health 10).  Real revive / redeploy evidence appears >= 34 s after the kill (observed minimum over the corpus), so a
# healthy sample counts as revive evidence only if it is at least this many seconds after the kill.
MIN_REVIVE_GAP_S = 20.0


@dataclass
class PlayerTimeline:
    """Alive/dead alternation of one account.  ``kills`` ascending; ``revive[i]`` = time of the first healthy
    position sample after ``kills[i]`` provided it comes before the next kill (else +inf)."""

    kills: np.ndarray
    revive: np.ndarray

    def alive_at(self, t: float) -> bool:
        if len(self.kills) == 0:
            return True
        i = int(np.searchsorted(self.kills, t, side="right")) - 1  # last kill <= t
        if i < 0:
            return True
        return bool(self.revive[i] <= t)

    def transitions(self) -> list[tuple[float, int]]:
        """``[(time, +1 | -1)]``: -1 at a kill that ends a life, +1 at the revive evidence that starts one."""
        out: list[tuple[float, int]] = []
        alive = True
        for i, k in enumerate(self.kills):
            if alive:
                out.append((float(k), -1))
                alive = False
            r = float(self.revive[i])
            if np.isfinite(r):
                out.append((r, +1))
                alive = True
        return out


def build_timelines(
    kill_w: np.ndarray,
    kill_victim: np.ndarray,
    pos_w: np.ndarray,
    pos_acct: np.ndarray,
    pos_health: np.ndarray,
    min_revive_gap_s: float = MIN_REVIVE_GAP_S,
) -> dict[int, PlayerTimeline]:
    """One :class:`PlayerTimeline` per account that has a kill event (all other accounts are always alive).

    ``pos_w`` need not be sorted.  A healthy sample is ``health > 0`` (NaN counts as not healthy); it is revive
    evidence for a kill only if it is at least ``min_revive_gap_s`` after that kill (stale-health guard, see
    :data:`MIN_REVIVE_GAP_S`; ``0.0`` gives the literal "any later healthy sample" rule)."""
    out: dict[int, PlayerTimeline] = {}
    if len(kill_w) == 0:
        return out
    healthy = np.isfinite(pos_health) & (pos_health > 0)
    hw, ha = pos_w[healthy], pos_acct[healthy]
    order = np.lexsort((hw, ha))
    hw, ha = hw[order], ha[order]
    bounds_lo = np.searchsorted(ha, np.unique(kill_victim), side="left")
    bounds_hi = np.searchsorted(ha, np.unique(kill_victim), side="right")
    span = {int(a): (int(lo), int(hi)) for a, lo, hi in zip(np.unique(kill_victim), bounds_lo, bounds_hi)}
    korder = np.lexsort((kill_w, kill_victim))
    kw, kv = kill_w[korder], kill_victim[korder]
    for a in np.unique(kv):
        a = int(a)
        ks = np.sort(kw[kv == a])
        lo, hi = span[a]
        hs = hw[lo:hi]  # ascending
        rev = np.full(len(ks), INF)
        for i, k in enumerate(ks):
            if min_revive_gap_s > 0:
                j = int(np.searchsorted(hs, k + min_revive_gap_s, side="left"))  # first healthy sample >= kill + gap
            else:
                j = int(np.searchsorted(hs, k, side="right"))  # first healthy sample strictly after the kill
            if j < len(hs):
                nxt = ks[i + 1] if i + 1 < len(ks) else INF
                if hs[j] < nxt:
                    rev[i] = float(hs[j])
        out[a] = PlayerTimeline(ks.astype(np.float64), rev)
    return out


def team_members(roster: list[tuple], account_index: dict[str, int]) -> dict[int, list[int]]:
    """team id -> list of account indices (roster order); roster rows are ``(team, player_id, ...)``."""
    out: dict[int, list[int]] = {}
    for row in roster:
        a = account_index.get(row[1])
        if a is not None:
            out.setdefault(int(row[0]), []).append(a)
    return out


def alive_count_features(
    members: list[int],
    timelines: dict[int, PlayerTimeline],
    start: float,
    end: float,
) -> tuple[float, float, float, float]:
    """``(n_start, n_end, n_min, n_avg)`` of the team's causally alive members on ``[start, end)``.

    ``n_start`` is the count at ``start``, ``n_end`` the count at ``end`` (the cut: only records before it exist,
    so it is the count after the last visible event), ``n_min`` the minimum and ``n_avg`` the time average of the
    step function over the window.  An empty window (``end <= start``) gives the count at ``end`` four times."""
    n0 = float(len(members))
    events: list[tuple[float, int]] = []
    for a in members:
        tl = timelines.get(a)
        if tl is not None:
            events += tl.transitions()
    events.sort()
    times = np.asarray([e[0] for e in events], dtype=np.float64)
    deltas = np.asarray([e[1] for e in events], dtype=np.float64)
    cum = np.cumsum(deltas)

    def count_at(t: float) -> float:
        i = int(np.searchsorted(times, t, side="right"))  # transitions <= t
        return n0 + (float(cum[i - 1]) if i > 0 else 0.0)

    n_start = count_at(start)
    n_end = count_at(end)
    if not end > start:
        return n_end, n_end, n_end, n_end
    lo = int(np.searchsorted(times, start, side="right"))
    hi = int(np.searchsorted(times, end, side="left"))  # transitions in (start, end)
    pts = np.concatenate(([start], times[lo:hi], [end]))
    vals = np.concatenate(([n_start], [count_at(t) for t in times[lo:hi]]))
    # value on [pts[j], pts[j+1]) is vals[j]; several transitions at one instant collapse (zero length)
    seg = np.diff(pts)
    n_min = float(vals.min()) if len(vals) else n_start
    n_avg = float(np.sum(vals * seg) / (end - start))
    return n_start, n_end, n_min, n_avg


def causal_team_alive(
    members: list[int], timelines: dict[int, PlayerTimeline], t: float
) -> bool:
    return any(timelines[a].alive_at(t) if a in timelines else True for a in members)


# --------------------------------------------------------------------------------------
# final elimination (label side)
# --------------------------------------------------------------------------------------
def member_final_death(
    timeline: PlayerTimeline | None, death_type: str | None, time_survived: float | None, offset_s: float
) -> float:
    """Final death time of one member (``+inf`` = alive at the end of the log)."""
    if timeline is None or len(timeline.kills) == 0:
        if death_type is None or death_type == "alive":
            return INF
        return INF if time_survived is None else float(time_survived) + offset_s
    # last life: alive again after the last kill iff revive evidence exists
    if np.isfinite(timeline.revive[-1]):
        return INF
    return float(timeline.kills[-1])


def human_offset(
    roster: list[tuple], account_index: dict[str, int], timelines: dict[int, PlayerTimeline]
) -> float:
    """Match median of (last kill - time_survived) over human members with both (label audit)."""
    diffs = []
    for _t, pid, _dt, ts in roster:
        if str(pid).startswith("ai.") or ts is None:
            continue
        a = account_index.get(pid)
        tl = timelines.get(a) if a is not None else None
        if tl is not None and len(tl.kills):
            diffs.append(float(tl.kills[-1]) - float(ts))
    return float(np.median(diffs)) if diffs else FALLBACK_OFFSET_S


def team_elimination_times(
    team_ids: np.ndarray,
    roster: list[tuple],
    account_index: dict[str, int],
    timelines: dict[int, PlayerTimeline],
) -> np.ndarray:
    """T per team (float64): max over members of the final death; ``+inf`` while any member is alive at the end;
    NaN for a team without roster members."""
    off = human_offset(roster, account_index, timelines)
    per_team: dict[int, list[float]] = {}
    for t, pid, dt, ts in roster:
        a = account_index.get(pid)
        d = member_final_death(timelines.get(a) if a is not None else None, dt, ts, off)
        per_team.setdefault(int(t), []).append(d)
    out = np.full(len(team_ids), np.nan, dtype=np.float64)
    for i, tid in enumerate(team_ids):
        v = per_team.get(int(tid))
        if v:
            out[i] = max(v)
    return out


# --------------------------------------------------------------------------------------
# revive / redeploy evidence audit (coordinator addendum b: comeback arena check)
# --------------------------------------------------------------------------------------
ARENA_Z_M = 500.0  # a redeployed player re-enters from the air (z ~ 1002 m); below this the sample is "on the ground"
ARENA_OUTSIDE_M = 500.0  # first healthy sample farther than this outside the square map


def revive_evidence(
    timelines: dict[int, PlayerTimeline],
    pos: dict[str, np.ndarray],
    map_size: float,
) -> list[dict]:
    """One record per revive event (the first healthy sample after a kill): the gap to the kill, its altitude and
    position, and the arena-like flags.  A comeback arena would show healthy samples that are not parachuting
    (``low_altitude``), lie off the map (``off_map``) or follow the kill within the 60 s the dead body keeps
    reporting health 0 (``too_soon``)."""
    out: list[dict] = []
    if not timelines:
        return out
    w, acct = pos["w"], pos["acct"]
    for a, tl in timelines.items():
        for kt, rv in zip(tl.kills, tl.revive):
            if not np.isfinite(rv):
                continue
            rows = np.nonzero((acct == a) & (w == rv))[0]
            if len(rows) == 0:
                continue
            j = int(rows[0])
            x, y, z = float(pos["x"][j]), float(pos["y"][j]), float(pos["z"][j])
            off = max(-x, x - map_size, -y, y - map_size, 0.0)
            gap = float(rv - kt)
            out.append({
                "acct": int(a), "kill_w": float(kt), "revive_w": float(rv), "gap_s": gap, "z_m": z, "x_m": x, "y_m": y,
                "off_map_m": float(off),
                "low_altitude": z < ARENA_Z_M, "off_map": off > ARENA_OUTSIDE_M, "too_soon": gap < 20.0,
            })  # fmt: skip
    return out


# --------------------------------------------------------------------------------------
# convenience wrappers for consumers (additive; the functions above keep their signatures)
# --------------------------------------------------------------------------------------
def timelines_from_source(src) -> dict[int, PlayerTimeline]:
    """:func:`build_timelines` on a :class:`model_stgcn.phase_v2.source.MatchSource` (account index -> timeline)."""
    return build_timelines(src.kills["w"], src.kills["victim"], src.pos["w"], src.pos["acct"], src.pos["health"])


def player_alive_at(timelines: dict[int, PlayerTimeline], account: int, t: float) -> bool:
    """Causal alive status of one account at ``t`` (accounts without a kill event are always alive)."""
    tl = timelines.get(int(account))
    return True if tl is None else tl.alive_at(t)


def team_alive_at(src, t: float, timelines: dict[int, PlayerTimeline] | None = None) -> np.ndarray:
    """``[N]`` bool (team order of ``src.team_ids``): the team has >= 1 causally alive member at ``t``.

    Uses records ``<= t`` only.  Hand it a source truncated at the cut (``truncate(src, a_k)``) to get the status the
    dump uses at that cut; on a full source with ``t = a_k`` the result is identical because every record that would
    change the answer is later than ``t``."""
    tl = timelines_from_source(src) if timelines is None else timelines
    members = team_members(src.roster, src.account_index())
    return np.asarray(
        [causal_team_alive(members.get(int(tid), []), tl, t) for tid in src.team_ids], dtype=bool
    )


def stale_health_samples(
    kill_w: np.ndarray,
    kill_victim: np.ndarray,
    pos_w: np.ndarray,
    pos_acct: np.ndarray,
    pos_health: np.ndarray,
    min_revive_gap_s: float = MIN_REVIVE_GAP_S,
) -> list[dict]:
    """Kills that are followed by a healthy position sample within ``min_revive_gap_s`` (ignored by the guard of
    :func:`build_timelines`): ``[{acct, kill_w, sample_w, health}]``."""
    out: list[dict] = []
    if len(kill_w) == 0:
        return out
    healthy = np.isfinite(pos_health) & (pos_health > 0)
    for kw, v in zip(kill_w, kill_victim):
        m = healthy & (pos_acct == v) & (pos_w > kw) & (pos_w < kw + min_revive_gap_s)
        if m.any():
            j = int(np.nonzero(m)[0][0])
            out.append({"acct": int(v), "kill_w": float(kw), "sample_w": float(pos_w[j]), "health": float(pos_health[j])})
    return out
