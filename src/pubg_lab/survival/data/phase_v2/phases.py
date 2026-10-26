"""Phase boundaries from LogPhaseChange EVENTS (not from the 10 s grid) on the LogMatchStart clock W.

PUBG telemetry (https://documentation.pubg.com/en/telemetry-objects.html#common, ``Common.isGame``):
``isGame`` "represents the phase of the game defined by the status of bluezone and safezone":
0 before lift off, 0.1 on the airplane, 0.5 no zone on the map, 1.0 first safezone and bluezone appear,
1.5 first bluezone shrinks, 2.0 second bluezone appears, 2.5 second bluezone shrinks, ...
(x.0 = the x-th zone is announced and the team waits, x.5 = the x-th blue zone shrinks).

The ``isGame`` of a LogPhaseChange row is the value of the state that has just ENDED (positions switch their own
``isGame`` to the next value at that very instant, checked on the corpus), and the analysis phase is
``k = ceil(isGame)`` (v1 convention, ``telem_phase_changes.phase``).  Hence

* phase k starts at the first LogPhaseChange with ceil(isGame) = k (state k.0, zone k appears),
* b_k = time of the first LogPhaseChange with ceil(isGame) = k + 1 = the end of the k-th shrink, the instant the
  (k+1)-th zone appears (the start of phase k + 1),
* if no such event exists phase k is the match's final phase and there is no cut at k.

The feature cut of phase k is ``a_k = b_k - CUT_LEAD_S`` (10 s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CUT_LEAD_S = 10.0
MAX_PHASE = 64  # sanity bound for ceil(isGame)


@dataclass(frozen=True)
class Phases:
    """``first_w[k]`` = W time of the first LogPhaseChange with ceil(isGame) == k (k >= 1); missing -> absent."""

    first_w: dict[int, float]

    @property
    def indices(self) -> list[int]:
        return sorted(self.first_w)

    @property
    def last_phase(self) -> int | None:
        return max(self.first_w) if self.first_w else None

    def exists(self, k: int) -> bool:
        return k in self.first_w

    def start(self, k: int) -> float:
        """s_k: the first LogPhaseChange of phase k."""
        return self.first_w[k]

    def boundary(self, k: int) -> float | None:
        """b_k: first LogPhaseChange of phase k+1, None if phase k is the final phase."""
        return self.first_w.get(k + 1)

    def cut(self, k: int) -> float | None:
        """a_k = b_k - 10 s, None if phase k is the final phase (no cut)."""
        b = self.boundary(k)
        return None if b is None else b - CUT_LEAD_S

    def has_cut(self, k: int) -> bool:
        return self.exists(k) and self.boundary(k) is not None


def phases_from_events(w: np.ndarray, is_game: np.ndarray) -> Phases:
    """Phase table from the LogPhaseChange rows (``w`` seconds on W, ``is_game`` float)."""
    first: dict[int, float] = {}
    for t, g in zip(np.asarray(w, dtype=np.float64), np.asarray(is_game, dtype=np.float64)):
        if not (math.isfinite(t) and math.isfinite(g)):
            continue
        k = int(math.ceil(g - 1e-9))
        if k < 1 or k > MAX_PHASE:
            continue
        if k not in first or t < first[k]:
            first[k] = float(t)
    return Phases(first)


def phases_of(src) -> Phases:
    return phases_from_events(src.phase_events["w"], src.phase_events["is_game"])


def monotone(phases: Phases) -> bool:
    """True iff the phase start times strictly increase with the phase number."""
    ks = phases.indices
    return all(phases.first_w[a] < phases.first_w[b] for a, b in zip(ks, ks[1:]))
