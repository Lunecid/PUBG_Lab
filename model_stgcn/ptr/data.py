"""PTR dump loading, standardisation, dataset and collate (DB-free).

Dump schema (one ``torch.save`` dict per cohort split, ``<dump-dir>/<cohort>/{train,val,test}.pt``)::

    meta    feature_names (F), feature_groups {name: cols} incl. 'zone' and 'habitat',
            cohort, split, [zone_columns]
    matches list of dicts: match_id, match_type (str), team_ids[N], X[P,N,F], node_mask[P,N] (alive at
            phase start), final_rank_norm[N], top5[N], [top10[N] - ignored], team_ai_frac[N],
            edges{prox|combat|copress: (idx[E,3]=(p,i,j), w[E])}, team_pos[P,N,2], zone[P,8], zone_valid[P],
            labels, two sets: pipeline = target_mask[P,N] + y_hazard[P,N] (team_phase_labels) and
            v2 = target_mask_v2[P,N] + y_hazard_v2[P,N] (event based, design doc v2.3),
            [team_death_s[N] - carried by the dump, not used here]

Label choice (``labels='v2'`` by default, ``'pipeline'`` for the old ones): ONE choice drives the risk set (the
target rows), the hazard labels, the alive tokens, the standardisation statistics, the HGB rows and therefore
every metric.  Everything reads the labels through :func:`select_labels` / :func:`effective_alive`.

Alive tokens (= the tokens and the attention keys of the model).  Under ``labels='v2'`` on an event-based dump
(ghost fix, design doc v2.3 / label audit) a team is alive at the start of slot ``k`` iff its last-kill death time
is later than the phase start, ``team_death_s > phase_start(k)`` with ``phase_start(0) = 0`` and
``phase_start(k) = phase_end_s[k-1]`` (:func:`event_alive_mask`; NaN = unknown = never alive).  That is exactly "was
at risk at cut k-1" (``node_v2[k] == target_v2[k-1]``) and it drops the *ghost* teams that the position-based
``node_mask`` still reads as alive.  ``target_mask_v2`` must lie inside that mask (checked).  A dump is
event-based when ``meta['team_death_source'] == 'last_kill_event'`` or when its matches carry the dump script's
own ``node_mask_v2`` (that field uses ``phase_start_s[k]``, which is 10 s later than ``phase_end_s[k-1]``; it is
only used as a marker and for the audit in :func:`event_alive_summary`).  For old dumps (no marker) and for
``labels='pipeline'`` the mask stays ``node_mask | target_mask(choice)``: the dump script raises ``node_mask`` over
the pipeline targets (alive at the END of a phase = alive at its start) and the loader raises it over the v2
targets the same way, otherwise such a row would be scored from a dead token.  :func:`alive_source` names the
rule in use; the training / HGB CLIs log it and write it to ``metrics.json``.

Dump versions (``meta['version']``, :func:`dump_version`)
------------------------------------------------------------
``meta['version'] == 'v2'`` marks the leak-free dump of :mod:`model_stgcn.dump_phase_tier_v2` (record contract:
``model_stgcn/dumps/bigcomp2027_phase_v2/CONTRACT.md``).  A dump without that key is a v1 dump and keeps every rule
above unchanged.  For v2 the dump is the single source of truth and the loader derives NOTHING from death times:

* risk set / labels: ``target_mask_v2`` / ``y_hazard_v2`` exactly as stored (``select_labels`` is a pass-through: no
  winner override, no ``_cut_exists``; winners are censored in the dump itself, ``team_death_s == +inf``, and an
  excluded (match, cut) has an all-False row that is never re-filled);
* key mask (:func:`key_mask`): ``node_mask``, the teams causally alive at the cut ``a_k = b_k - 10 s``; only these
  tokens are attention keys (both axes of the model);
* query / token mask (:func:`effective_alive`, ``query_mask``): ``node_mask | target_mask_v2``.  It differs from the
  key mask only for *comeback* rows, risk-set teams that are causally dead at ``a_k`` (``T > b_k``): they keep a
  query token and a loss row but are never a key for another team;
* loss and metrics: ``target_mask_v2`` rows only, each counted once (``target`` is a subset of ``query``).

Mixing a v2 dump with anything derived from a v1 dump (kf tables, plan / label caches) raises
:class:`DumpVersionError`.

Team-level token features: the normalised team centre (``pos_in_token``) and ``team_ai_frac`` (share of the
roster that are AI accounts, broadcast over phases like the position).  ``team_ai_frac`` and ``match_type`` are
**required**: a dump without them fails loudly (no silent skip).  ``match_type`` is carried to the prediction
CSV and must be one of :data:`DEFAULT_MATCH_TYPES` unless the caller widens the list (airoyale is excluded by
the registered population).

Assumptions that are not pinned down by the schema (change them here, in one place):

* ``zone`` columns, in this order: current centre x/y, current radius, next centre x/y, next radius,
  phase-elapsed ratio, phase number (:data:`ZONE_COLUMNS`).  ``meta['zone_columns']`` (a list of
  those names) overrides the order if the dump script uses another one.
* ``zone`` centre/radius and ``team_pos`` share one coordinate system (only ratios are used).
* ``final_rank_norm = final_rank / n_teams`` with rank 1 = winner (small = good placement).

Standardisation: mean/std are fitted on **train** tokens with ``node_mask`` True only (NaN-aware,
float64, Chan merge), constant columns get ``std = 1``, NaN / inf become 0 after scaling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from model_stgcn.ptr import player as PL
from model_stgcn.ptr import relations as R

# a_k = b_k - CUT_LEAD_S (the producer's cut lead; tests/test_v2_masks_labels.py asserts it equals model_stgcn.phase_v2.phases.CUT_LEAD_S)
CUT_LEAD_S = 10.0

# ---- dump versions: ``meta['version']`` (absent = v1)
VERSION_KEY = "version"
DUMP_V1, DUMP_V2 = "v1", "v2"
DUMP_VERSIONS: tuple[str, ...] = (DUMP_V1, DUMP_V2)
ALIVE_SOURCE_V2 = "node_mask+query_mask_v2"
# v2 record keys beyond ``_MATCH_KEYS`` that the loader needs; ``node_mask_v2`` must NOT be present (v1 leak marker)
V2_MATCH_KEYS: tuple[str, ...] = (
    "query_mask", "target_mask_v2", "y_hazard_v2", "team_death_s", "team_won", "phase_mask", "phase_start_s", "phase_end_s", "cut_t",
)  # fmt: skip


class DumpVersionError(ValueError):
    """A v2 dump was combined with something derived from a v1 dump (or the other way round), or its records do not
    follow the contract of its ``meta['version']``."""


def dump_version(dump_or_meta) -> str:
    """``'v2'`` iff ``meta['version'] == 'v2'`` (the leak-free dump); a dump without the key is ``'v1'``.

    Accepts a dump (``{'meta', 'matches'}``) or a meta dict; anything else (for example a ``'v3'`` written by a later
    producer) raises :class:`DumpVersionError` instead of being read with the wrong rules."""
    meta = dump_or_meta
    if isinstance(dump_or_meta, Mapping) and "matches" in dump_or_meta:
        meta = dump_or_meta.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    value = meta.get(VERSION_KEY)
    text = str(value).strip().lower() if value is not None else None
    if text is None or text in ("v1", "1"):
        return DUMP_V1
    if text in ("v2", "2"):
        return DUMP_V2
    raise DumpVersionError(f"unsupported dump version meta['{VERSION_KEY}'] = {value!r}; this loader reads {DUMP_VERSIONS}")


def check_same_version(have: str, want: str, *, what: str, against: str) -> None:
    """Raise :class:`DumpVersionError` unless ``have == want`` (``what`` names the artefact, ``against`` the dump)."""
    if have != want:
        raise DumpVersionError(
            f"{what} was derived from a {have} dump but {against} is a {want} dump: mixing them silently re-introduces the "
            f"look-ahead of the v1 pipeline (or drops the v2 risk-set rows). Rebuild {what} from the {want} dump."
        )


# keys that only a v2 record carries (a v1 record never has any of them; ``cut_t`` / ``cut_wall_s`` exist in both and are not listed)
V2_ONLY_KEYS: tuple[str, ...] = (
    "query_mask", "cut_valid", "cut_mask", "cut_excluded", "cut_terminal", "match_end_s", "label_window_end_s", "phase_duration_sched_s",
)  # fmt: skip


def is_v2_match(m: Mapping) -> bool:
    """A record is a v2 record iff it carries any v2-only key (:data:`V2_ONLY_KEYS`; :func:`validate_dump` enforces that a v1
    dump has none and that a v2 dump has all of :data:`V2_MATCH_KEYS`).  Deciding on the *presence of any* v2 key (not just
    ``query_mask``) means a v2 record whose mask was stripped is refused by the functions below instead of being read through
    the v1 rules (which derive masks and winners from ``team_death_s``)."""
    return any(k in m for k in V2_ONLY_KEYS)


def _require_v2_keys(m: Mapping, keys: Sequence[str], what: str) -> None:
    missing = [k for k in keys if k not in m]
    if missing:
        raise DumpVersionError(
            f"match {m.get('match_id', '?')}: a v2 record without {missing} ({what}): the loader never rebuilds a mask, a risk set or a "
            "winner from team_death_s for a v2 dump (that rule is the L3 look-ahead the v2 dump removes)"
        )


def boundary_times(phase_end_s) -> np.ndarray:
    """``b_k`` per slot as float64 (NaN where the phase has no boundary).

    The final v2 dump stores the time fields as float64 and they are returned as they are (compare ``w < cut_t`` exactly).  A
    float32 copy (an earlier producer, resolution 0.12 ms near 1500 s) is rounded to the millisecond: every telemetry timestamp
    is a whole number of milliseconds, so that recovers the exact float64 boundary."""
    arr = torch.as_tensor(phase_end_s)
    b = arr.double().numpy().copy()
    if arr.dtype == torch.float32:
        b = np.round(b, 3)
    return b


def cut_times(phase_end_s) -> np.ndarray:
    """``a_k = b_k - 10 s`` per slot, float64, exactly the producer's cut time (NaN without a boundary).  Every input that
    is built outside the dump reads the records with ``w < a_k`` (strict) only."""
    return boundary_times(phase_end_s) - float(CUT_LEAD_S)


ZONE_COLUMNS: tuple[str, ...] = (
    "cur_cx", "cur_cy", "cur_r", "next_cx", "next_cy", "next_r", "phase_frac", "phase_no",
)  # fmt: skip
SPLITS: tuple[str, ...] = ("train", "val", "test")
POS_FEATURE_NAMES: tuple[str, ...] = ("team_pos_x", "team_pos_y")
AI_FEATURE_NAME = "team_ai_frac"
STD_FLOOR = 1e-8
# label sets: (hazard label field, risk-set field).  v2 = event based labels (design doc v2.3), the default.
LABEL_CHOICES: tuple[str, ...] = ("v2", "pipeline")
DEFAULT_LABELS = "v2"
LABEL_FIELDS: dict[str, tuple[str, str]] = {
    "v2": ("y_hazard_v2", "target_mask_v2"),
    "pipeline": ("y_hazard", "target_mask"),
}
# registered population (design doc v2.3): official + competitive; airoyale is excluded upstream
DEFAULT_MATCH_TYPES: tuple[str, ...] = ("official", "competitive")
# ghost fix: markers of an event-based dump (death times = last kill event, +inf = alive at the end)
TEAM_DEATH_SOURCE_KEY = "team_death_source"
EVENT_DEATH_SOURCES: tuple[str, ...] = ("last_kill_event",)
EVENT_ALIVE_FIELD = "node_mask_v2"  # the dump script's own mask: marker + audit, not the mask the loader uses
# Alive-count variables are not spatial features, so the noHab arms keep them even though two of them sit in
# the dump's 'zone' group and one in the 'habitat' group (design doc v2.2).  ``hab_safe_area_per_alive_team``
# is a safe-area ratio (spatial) and is removed.  Names that a dump does not have are simply ignored.
ALIVE_COUNT_FEATURES: tuple[str, ...] = (
    "avg_alive_teams_norm", "avg_alive_players_norm", "hab_alive_teams",
)  # fmt: skip
# max |combat weight| above which the dump cannot hold ln(1 + intensity): intensities of a team pair in one
# phase reach >= 100 as soon as a kill is involved (ln(101) = 4.6), while ln(1 + intensity) stays below ~12
COMBAT_LOG_MAX = 30.0
# Extra (union) features, design doc 4.6 (model_stgcn.ptr.extra): ONLY the ``kf_*`` columns of the extra table may enter the token /
# row features.  The table also carries the keys and the producer's flag / label columns (``y_hazard_v2`` IS the label);
# they are never features, and ``check_extra_names`` is the assertion that keeps it so wherever features are assembled.
EXTRA_PREFIX = "kf_"
EXTRA_FORBIDDEN: tuple[str, ...] = ("cut_t", "in_risk_set", "alive_at_cut", "y_hazard_v2", "match_id", "team_id", "phase")

# ``top10`` is part of the dump but unused (design doc v2.1): neither required nor carried.  The label fields
# depend on the label choice (LABEL_FIELDS); everything else is required for every choice.
_MATCH_KEYS = (
    "match_id", "team_ids", "X", "node_mask", "final_rank_norm", "top5", "edges", "team_pos", "zone",
    "zone_valid", "team_ai_frac", "match_type",
)  # fmt: skip


# --------------------------------------------------------------------------- small helpers
def _t(x, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(x)
    return t if dtype is None else t.to(dtype)


def _bool(x) -> torch.Tensor:
    return _t(x).bool()


def check_extra_names(names: Iterable[str]) -> list[str]:
    """Guard of the union arms: every extra feature must be a ``kf_*`` column, none may be a key / flag / label column of the
    extra table (``cut_t``, ``in_risk_set``, ``alive_at_cut``, ``y_hazard_v2``: the label), with or without the prefix, and
    the names are unique and not empty.  Raises ``ValueError``; returns the names as a list."""
    names = [str(n) for n in names]
    bad = [n for n in names if not n.startswith(EXTRA_PREFIX) or n[len(EXTRA_PREFIX):] in EXTRA_FORBIDDEN]
    if bad:
        raise ValueError(
            f"non-kf column(s) would enter the features: {bad}; only {EXTRA_PREFIX}* columns may, never {EXTRA_FORBIDDEN}"
        )
    if not names:
        raise ValueError("no extra features")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate extra feature names: {sorted({n for n in names if names.count(n) > 1})}")
    return names


def check_labels(labels: str) -> str:
    if labels not in LABEL_FIELDS:
        raise ValueError(f"labels must be one of {LABEL_CHOICES}, got {labels!r}")
    return labels


def _cut_exists(m: Mapping, target: torch.Tensor) -> torch.Tensor:
    """``[P]`` bool: cut slot ``p`` exists iff phases ``p`` and ``p+1`` (dump slots) exist (``phase_mask``); without
    ``phase_mask`` the slots where any team is at risk."""
    P = target.shape[0]
    if "phase_mask" in m:
        pm = _bool(m["phase_mask"])
        ok = torch.zeros(P, dtype=torch.bool)
        ok[: P - 1] = pm[: P - 1] & pm[1:P]
        return ok
    return target.any(dim=1)


def winners(m: Mapping) -> torch.Tensor:
    """``[N]`` bool, ``team_won`` (all False when the match carries no such field)."""
    if "team_won" in m:
        return _bool(m["team_won"])
    n = len(m["team_death_s"]) if "team_death_s" in m else len(m["team_ids"])
    return torch.zeros(n, dtype=torch.bool)


def select_labels(m: Mapping, labels: str = DEFAULT_LABELS) -> tuple[torch.Tensor, torch.Tensor]:
    """``(y [P,N] float32 zero outside the risk set, target [P,N] bool)`` of the chosen label set.

    Winner censoring (labels v2 only): a team with ``team_won`` is treated as ``team_death_s = +inf`` whatever its
    last kill event says (a winner can carry a finite time from a kill event after its last teammate fell): it is
    at risk at every cut whose phase ``k+1`` exists and never has a positive label.

    A record of a **v2 dump** (any v2-only key present, :func:`is_v2_match`) is returned exactly as stored: the producer already censored the
    winners (``team_death_s == +inf``), set the risk row of every excluded cut to all-False and wrote ``y`` on the risk
    rows; nothing is re-derived here (no winner override, no cut-existence rule)."""
    if is_v2_match(m):
        return _select_labels_v2(m, labels)
    y_key, t_key = LABEL_FIELDS[check_labels(labels)]
    missing = [k for k in (y_key, t_key) if k not in m]
    if missing:
        raise ValueError(
            f"match {m.get('match_id', '?')}: labels={labels!r} needs {missing}, which the dump does not carry"
            + ("" if labels == "pipeline" else " (use labels='pipeline' for a dump written before design doc v2.3)")
        )
    target = _bool(m[t_key])
    y = torch.where(target, torch.nan_to_num(_t(m[y_key], torch.float32)), torch.zeros(()))
    if labels == "v2":
        won = winners(m)
        if won.any():
            target = target.clone()
            target[:, won] = _cut_exists(m, target)[:, None]
            y = y.clone()
            y[:, won] = 0.0
    return y, target


def _select_labels_v2(m: Mapping, labels: str) -> tuple[torch.Tensor, torch.Tensor]:
    """v2 record: ``(y, target)`` = ``(y_hazard_v2, target_mask_v2)`` as stored (pass-through, see :func:`select_labels`)."""
    if check_labels(labels) != "v2":
        raise DumpVersionError(
            f"match {m.get('match_id', '?')}: a v2 dump carries ONE label set (the leak-free v2 labels; its 'target_mask' / "
            f"'y_hazard' are aliases of them), labels={labels!r} is not defined for it"
        )
    _require_v2_keys(m, ("y_hazard_v2", "target_mask_v2", "query_mask", "node_mask"), "labels and masks are read as stored")
    target = _bool(m["target_mask_v2"])
    y = torch.where(target, torch.nan_to_num(_t(m["y_hazard_v2"], torch.float32)), torch.zeros(()))
    return y, target


def winner_censor_summary(dump: Mapping, labels: str = DEFAULT_LABELS) -> dict:
    """What the winner censoring changed over a dump (zeros for pipeline labels and for v2 dumps)."""
    out = {"applied": labels == "v2", "winner_teams": 0, "winners_finite_death": 0, "positives_removed": 0,
           "target_rows_added": 0, "target_rows_removed": 0}  # fmt: skip
    if dump_version(dump) == DUMP_V2:  # the producer censors the winners; the loader changes no row
        out["applied"] = False
        out["source"] = "v2 dump: winners censored by the producer (team_death_s = +inf), no row added or removed by the loader"
        for m in dump["matches"]:
            won = winners(m)
            out["winner_teams"] += int(won.sum())
            out["winners_finite_death"] += int((won & ~torch.isinf(_t(m["team_death_s"], torch.float64))).sum())
        return out
    if labels != "v2":
        return out
    for m in dump["matches"]:
        if "team_won" not in m:
            continue
        won = winners(m)
        out["winner_teams"] += int(won.sum())
        raw_t, raw_y = _bool(m["target_mask_v2"]), _t(m["y_hazard_v2"], torch.float32)
        fin = won & ~torch.isinf(_t(m["team_death_s"], torch.float64))
        out["winners_finite_death"] += int(fin.sum())
        y, t = select_labels(m, labels)
        out["positives_removed"] += int(((raw_y > 0.5) & raw_t & won[None, :]).sum() - ((y > 0.5) & t & won[None, :]).sum())
        out["target_rows_added"] += int((t & ~raw_t).sum())
        out["target_rows_removed"] += int((~t & raw_t).sum())
    return out


def uses_event_alive(dump: Mapping, labels: str = DEFAULT_LABELS) -> bool:
    """True iff the alive tokens come from the event-based death times (labels v2 on an event-based v1 dump).

    Always False for a v2 dump: its masks come from the records (``node_mask`` / ``query_mask``), never from ``T``."""
    if check_labels(labels) != "v2":
        return False
    if dump_version(dump) == DUMP_V2:
        return False
    meta = dump.get("meta") or {}
    if meta.get(TEAM_DEATH_SOURCE_KEY) in EVENT_DEATH_SOURCES:
        return True
    matches = dump.get("matches") or []
    return bool(matches) and EVENT_ALIVE_FIELD in matches[0]


def alive_source(dump: Mapping, labels: str = DEFAULT_LABELS) -> str:
    """Name of the alive-token rule in use: ``node_mask+query_mask_v2`` (v2 dump) | ``node_mask_v2`` |
    ``node_mask|target_v2`` | ``node_mask|target_pipeline`` (v1)."""
    if dump_version(dump) == DUMP_V2:
        return ALIVE_SOURCE_V2
    if uses_event_alive(dump, labels):
        return "node_mask_v2"
    return "node_mask|target_v2" if labels == "v2" else "node_mask|target_pipeline"


def event_alive_mask(m: Mapping) -> torch.Tensor:
    """``[P, N]`` bool: ``team_death_s > phase_start(k)`` (the v1 dump's ``node_mask_v2`` rule).

    ``phase_start(k)`` is the dump's own ``phase_start_s[k]`` when the match carries it; otherwise (older event dumps)
    ``phase_start(0) = 0`` and ``phase_start(k) = phase_end_s[k-1]``, which is up to one grid gap earlier, so that
    fallback keeps a few more teams.  NaN death times (unknown) are never alive; slots of absent phases (``phase_mask``
    False, NaN phase end or NaN start) are empty.

    Refuses a v2 record: its masks are stored (``node_mask`` = causally alive at the cut, ``query_mask``), and a mask
    derived from the final death time is exactly the look-ahead the v2 dump removes."""
    if is_v2_match(m):
        raise DumpVersionError(
            f"match {m.get('match_id', '?')}: a v2 record carries node_mask / query_mask; deriving the alive mask from "
            "team_death_s (the final death time) would re-introduce the L3 look-ahead"
        )
    death = _t(m["team_death_s"], torch.float64)
    death = torch.where(winners(m), torch.full_like(death, float("inf")), death)  # winner censoring (labels v2)
    end = _t(m["phase_end_s"], torch.float64)
    if "phase_start_s" in m:
        start = _t(m["phase_start_s"], torch.float64)
    else:
        start = torch.cat([torch.zeros(1, dtype=torch.float64), end[:-1]])
    alive = ~torch.isnan(death)[None, :] & (death[None, :] > start[:, None])  # comparisons with NaN are False
    alive &= (torch.isfinite(end) & torch.isfinite(start))[:, None]  # a slot exists only if its phase has bounds
    if "phase_mask" in m:
        alive &= _bool(m["phase_mask"])[:, None]
    return alive


def event_alive_rule(m: Mapping) -> str:
    """Which phase-start rule :func:`event_alive_mask` applies to this match."""
    return "phase_start_s" if "phase_start_s" in m else "phase_end_s[k-1]"


def effective_alive(m: Mapping, labels: str, event: bool) -> torch.Tensor:
    """Alive tokens ``[P, N]`` of a match: the event-based mask (``event=True``, labels v2 on an event-based dump) or
    ``node_mask`` raised over the risk set of the chosen label set.

    A v2 record returns its stored ``query_mask`` (= ``node_mask | target_mask_v2``): every row that must exist as a
    token, comeback rows included (``event`` is ignored).  Use :func:`key_mask` for the attention keys."""
    if is_v2_match(m):
        check_labels(labels)
        _require_v2_keys(m, ("query_mask", "node_mask"), "the token mask is read as stored")
        return _bool(m["query_mask"])
    if event and check_labels(labels) == "v2":
        return event_alive_mask(m)
    return _bool(m["node_mask"]) | select_labels(m, labels)[1]


def key_mask(m: Mapping, labels: str = DEFAULT_LABELS, event: bool = False) -> torch.Tensor:
    """Attention-key mask ``[P, N]`` of a match.  v2 record: ``node_mask``, the teams causally alive at the cut (a comeback
    row is a query but never a key).  v1 record: the same mask as :func:`effective_alive` (every token is a key)."""
    if is_v2_match(m):
        check_labels(labels)
        _require_v2_keys(m, ("query_mask", "node_mask"), "the key mask is read as stored")
        return _bool(m["node_mask"])
    return effective_alive(m, labels, event)


def comeback_mask(m: Mapping) -> torch.Tensor:
    """``target & ~node_mask`` ``[P, N]``: risk-set rows that are causally dead at the cut (v2 records; all False for v1)."""
    if not is_v2_match(m):
        return torch.zeros_like(_bool(m["node_mask"]))
    _require_v2_keys(m, ("target_mask_v2", "node_mask", "query_mask"), "comeback rows are read from the stored masks")
    return _bool(m["target_mask_v2"]) & ~_bool(m["node_mask"])


def event_alive_audit(m: Mapping) -> dict | None:
    """Compare the loader's event mask with the dump's own ``node_mask_v2`` (None if the match has no such field)."""
    if is_v2_match(m) or EVENT_ALIVE_FIELD not in m:
        return None
    mine, theirs = event_alive_mask(m), _bool(m[EVENT_ALIVE_FIELD])
    return {
        "n_loader": int(mine.sum()),
        "n_dump": int(theirs.sum()),
        "agree": int((mine & theirs).sum()),
        "only_loader": int((mine & ~theirs).sum()),
        "only_dump": int((~mine & theirs).sum()),
    }


def event_alive_summary(dump: Mapping, labels: str = DEFAULT_LABELS) -> dict:
    """Token counts of the alive rule in use over a whole dump (what was added / removed w.r.t. ``node_mask``).

    v2 dump: ``n_alive`` = query tokens, ``n_node_mask`` = key tokens, ``raised`` = tokens that are not keys (the
    comeback rows), ``ghost_removed`` = 0 (nothing is derived from death times), plus ``comeback_rows`` per cut."""
    if dump_version(dump) == DUMP_V2:
        P = int(_t(dump["matches"][0]["X"]).shape[0]) if dump["matches"] else 0
        out = {"source": alive_source(dump, labels), "n_alive": 0, "n_node_mask": 0, "ghost_removed": 0, "raised": 0,
               "n_query": 0, "n_key": 0, "comeback_rows": 0, "comeback_rows_by_cut": [0] * max(P - 1, 0)}  # fmt: skip
        for m in dump["matches"]:
            query, key = effective_alive(m, labels, False), key_mask(m, labels)
            comeback = comeback_mask(m)
            out["n_alive"] += int(query.sum())
            out["n_query"] += int(query.sum())
            out["n_node_mask"] += int(key.sum())
            out["n_key"] += int(key.sum())
            out["raised"] += int((query & ~key).sum())
            out["comeback_rows"] += int(comeback.sum())
            for p in range(len(out["comeback_rows_by_cut"])):
                out["comeback_rows_by_cut"][p] += int(comeback[p].sum())
        return out
    event = uses_event_alive(dump, labels)
    out = {"source": alive_source(dump, labels), "n_alive": 0, "n_node_mask": 0, "ghost_removed": 0, "raised": 0}
    if event:
        rules = sorted({event_alive_rule(m) for m in dump["matches"]})
        out["start_rule"] = rules[0] if len(rules) == 1 else "mixed"
    audit = {"only_loader": 0, "only_dump": 0, "agree": 0}
    audited = False
    for m in dump["matches"]:
        alive, node = effective_alive(m, labels, event), _bool(m["node_mask"])
        out["n_alive"] += int(alive.sum())
        out["n_node_mask"] += int(node.sum())
        out["ghost_removed"] += int((node & ~alive).sum())  # read as alive by the position-based table only
        out["raised"] += int((alive & ~node).sum())  # alive here, missing from node_mask
        a = event_alive_audit(m) if event else None
        if a is not None:
            audited = True
            for k in audit:
                audit[k] += a[k]
    if audited:
        out.update(audit)
    return out


def subsample_dump(dump: Mapping, n: int | None, seed: int) -> tuple[Mapping, dict | None]:
    """A dump whose ``matches`` are a random subset of ``n`` matches (learning-curve runs; design doc 4.4).

    One seed defines one random permutation of the matches and a subsample of size ``n`` is its first ``n``
    entries, so the subsamples of one seed are *nested* (300 inside 600 inside 1200) and the neural and HGB runs
    with the same ``(n, seed)`` train on exactly the same matches.  The kept matches keep their dump order.
    ``n=None`` or ``n`` >= the number of matches returns the dump itself.  The returned info dict (None for no
    subsampling) goes into ``metrics.json``; ``match_ids_sha1`` identifies the subset."""
    if n is None:
        return dump, None
    if n < 1:
        raise ValueError(f"train subsample must be >= 1 match, got {n}")
    matches = dump["matches"]
    total = len(matches)
    if n >= total:
        idx = np.arange(total)
    else:
        idx = np.sort(np.random.default_rng(int(seed)).permutation(total)[:n])
    ids = sorted(str(matches[i]["match_id"]) for i in idx)
    info = {
        "n_requested": int(n), "n_used": int(len(idx)), "n_full": int(total), "seed": int(seed),
        "match_ids_sha1": hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest(),
    }  # fmt: skip
    if len(idx) == total:
        return dump, info
    out = dict(dump)
    out["matches"] = [matches[i] for i in idx]
    return out, info


def zone_layout(meta: Mapping) -> dict[str, int]:
    """Map canonical zone column names to their column index in the dump's ``zone`` tensor."""
    names = meta.get("zone_columns") if meta else None
    if names is None:
        return {name: i for i, name in enumerate(ZONE_COLUMNS)}
    names = list(names)
    if sorted(names) != sorted(ZONE_COLUMNS):
        raise ValueError(f"meta['zone_columns'] must be a permutation of {ZONE_COLUMNS}, got {names}")
    return {name: names.index(name) for name in ZONE_COLUMNS}


def _zone_order(meta: Mapping) -> np.ndarray:
    layout = zone_layout(meta)
    return np.array([layout[name] for name in ZONE_COLUMNS], dtype=np.int64)


# --------------------------------------------------------------------------- validation / IO
def validate_dump(dump: Mapping, labels: str = DEFAULT_LABELS) -> None:
    """Raise ``ValueError`` if the dump violates the schema or the invariants of the chosen label set.

    Dispatches on ``meta['version']`` (:func:`dump_version`): a v2 dump is validated against the v2 record contract
    (stored ``query_mask`` / ``node_mask``, censored winners, no rows at excluded cuts) and needs ``labels='v2'``."""
    check_labels(labels)
    meta = dump.get("meta") if isinstance(dump, Mapping) else None
    if not isinstance(meta, Mapping):
        raise ValueError("dump has no 'meta' dict")
    version = dump_version(meta)
    if version == DUMP_V2 and labels != "v2":
        raise DumpVersionError(
            f"labels={labels!r} is not defined for a v2 dump (one leak-free label set, the pipeline labels are not stored)"
        )
    event = isinstance(dump, Mapping) and uses_event_alive(dump, labels)
    for key in ("feature_names", "feature_groups"):
        if key not in meta:
            raise ValueError(f"meta is missing '{key}'")
    n_feat = len(meta["feature_names"])
    for name, cols in meta["feature_groups"].items():
        cols = np.asarray(cols, dtype=np.int64)
        if cols.size and (cols.min() < 0 or cols.max() >= n_feat):
            raise ValueError(f"feature group '{name}' has columns outside [0, {n_feat})")
    zone_layout(meta)
    matches = dump.get("matches")
    if not isinstance(matches, (list, tuple)) or not matches:
        raise ValueError("dump has no matches")
    n_phases = None
    for k, m in enumerate(matches):
        n_phases = _validate_match(m, n_feat, len(ZONE_COLUMNS), k, n_phases, labels, event, version)


def _validate_v2_record(m: Mapping, where: str, P: int, N: int) -> None:
    """Invariants of one v2 record that are not shared with v1 (see ``CONTRACT.md``): nothing is re-derived, the stored masks
    and labels are only checked against each other."""
    if EVENT_ALIVE_FIELD in m:
        raise DumpVersionError(
            f"{where}: a v2 record must not carry '{EVENT_ALIVE_FIELD}' (its presence marks a v1 event dump whose alive mask "
            "is derived from the final death time)"
        )
    node, query, target = _bool(m["node_mask"]), _bool(m["query_mask"]), _bool(m["target_mask_v2"])
    if not torch.equal(query, node | target):
        raise ValueError(f"{where}: query_mask must equal node_mask | target_mask_v2 ({int((query ^ (node | target)).sum())} cells differ)")
    for key, shape in (("query_mask", (P, N)), ("team_death_s", (N,)), ("team_won", (N,)), ("phase_mask", (P,)),
                       ("phase_start_s", (P,)), ("phase_end_s", (P,)), ("cut_t", (P,))):  # fmt: skip
        if tuple(_t(m[key]).shape) != shape:
            raise ValueError(f"{where}: {key} must have shape {shape}, got {tuple(_t(m[key]).shape)}")
    T = _t(m["team_death_s"], torch.float64)
    if bool((T == float("-inf")).any()):
        raise ValueError(f"{where}: team_death_s must be finite, +inf (censored) or NaN (unknown)")
    won = _bool(m["team_won"])
    if won.any() and not bool(torch.isposinf(T[won]).all()):
        raise ValueError(f"{where}: a v2 dump censors the winners (team_death_s = +inf); a winner has a finite or unknown T")
    if bool((target & torch.isnan(T)[None, :]).any()):
        raise ValueError(f"{where}: risk-set rows of teams with unknown death time (NaN)")
    if won.any() and bool(((_t(m["y_hazard_v2"], torch.float32) > 0.5) & target & won[None, :]).any()):
        raise ValueError(f"{where}: a censored winner has a positive label")
    pm = _bool(m["phase_mask"])
    rows = target.any(1)
    has_cut = torch.zeros(P, dtype=torch.bool)
    has_cut[: P - 1] = pm[: P - 1] & pm[1:P] & torch.isfinite(_t(m["phase_end_s"], torch.float64))[: P - 1]
    if bool((rows & ~has_cut).any()):
        raise ValueError(f"{where}: risk-set rows at a slot without a cut (phase k+1 missing or no boundary b_k)")
    if "cut_valid" in m:
        cv = _bool(m["cut_valid"])
        if tuple(cv.shape) != (P,):
            raise ValueError(f"{where}: cut_valid must have shape ({P},), got {tuple(cv.shape)}")
        if bool((rows & ~cv).any()):
            raise ValueError(f"{where}: risk-set rows at an excluded / invalid cut (cut_valid is False)")


def _validate_match(
    m: Mapping, n_feat: int, n_zone: int, k: int, n_phases: int | None, labels: str, event: bool = False,
    version: str = DUMP_V1,
) -> int:
    where = f"match #{k} ({m.get('match_id', '?') if isinstance(m, Mapping) else '?'})"
    v2 = version == DUMP_V2
    y_key, t_key = LABEL_FIELDS["v2" if v2 else labels]
    if not v2 and is_v2_match(m):
        raise DumpVersionError(
            f"{where}: carries v2-only key(s) {[k for k in V2_ONLY_KEYS if k in m]} but meta['version'] is not 'v2'"
        )
    event_keys = ("team_death_s", "phase_end_s") if event else ()
    v2_keys = V2_MATCH_KEYS if v2 else ()
    missing = [key for key in (*_MATCH_KEYS, y_key, t_key, *event_keys, *v2_keys) if key not in m]
    if missing:
        raise ValueError(f"{where}: missing keys {missing} (labels={labels!r})")
    mt = m["match_type"]
    if not isinstance(mt, str) or not mt:
        raise ValueError(f"{where}: match_type must be a non-empty string, got {mt!r}")
    X = _t(m["X"])
    if X.dim() != 3 or X.shape[2] != n_feat:
        raise ValueError(f"{where}: X must be [P, N, {n_feat}], got {tuple(X.shape)}")
    P, N, _ = X.shape
    if n_phases is not None and P != n_phases:
        raise ValueError(f"{where}: {P} phases, earlier matches have {n_phases}")
    expect = {
        "team_ids": (N,), "node_mask": (P, N), t_key: (P, N), y_key: (P, N),
        "final_rank_norm": (N,), "top5": (N,), "team_pos": (P, N, 2),
        "zone": (P, n_zone), "zone_valid": (P,), "team_ai_frac": (N,),
    }  # fmt: skip
    if event:
        expect["team_death_s"] = (N,)
        expect["phase_end_s"] = (P,)
    for key, shape in expect.items():
        if tuple(_t(m[key]).shape) != shape:
            raise ValueError(f"{where}: {key} must have shape {shape}, got {tuple(_t(m[key]).shape)}")
    ai = _t(m["team_ai_frac"], torch.float32)
    ai = ai[torch.isfinite(ai)]
    if ai.numel() and (float(ai.min()) < 0.0 or float(ai.max()) > 1.0):
        raise ValueError(f"{where}: team_ai_frac must lie in [0, 1] (NaN = no participants)")
    node, target = _bool(m["node_mask"]), _bool(m[t_key])
    # the dump script raises node_mask over the pipeline targets, so that invariant is checked for them;
    # the v2 risk set may sit on uncovered tokens: the loader raises node_mask over it (effective_alive)
    if labels == "pipeline" and (target & ~node).any():
        raise ValueError(f"{where}: target_mask must be a subset of node_mask (alive at phase start)")
    if target[-1].any():  # the hazard of the last slot would need a phase after it (evaluation rows: cuts 1..7)
        raise ValueError(f"{where}: {t_key} marks targets in the last phase slot; cut phases are 1..{P - 1}")
    if v2:
        _validate_v2_record(m, where, P, N)
    if event:  # ghost fix: the v2 risk set must sit on tokens that the death times call alive
        if bool((_t(m["team_death_s"], torch.float64) == float("-inf")).any()):
            raise ValueError(f"{where}: team_death_s must be finite, +inf (alive at the end) or NaN (unknown)")
        outside = target & ~event_alive_mask(m)
        if outside.any():
            raise ValueError(
                f"{where}: {t_key} must be a subset of the event-based alive mask (team_death_s > phase start): "
                f"{int(outside.sum())} target rows are on dead tokens"
            )
    if target.any():
        y = _t(m[y_key], torch.float32)[target]
        if not torch.isin(y, torch.tensor([0.0, 1.0])).all():
            raise ValueError(f"{where}: {y_key} must be 0/1 on target rows")
        cols = target.any(0)
        for key in ("final_rank_norm", "top5"):
            if not torch.isfinite(_t(m[key], torch.float32)[cols]).all():
                raise ValueError(f"{where}: {key} must be finite for every team that is a target")
    edges = m["edges"]
    if not isinstance(edges, Mapping):
        raise ValueError(f"{where}: edges must be a dict")
    for name, item in edges.items():
        idx, w = _t(item[0]), _t(item[1])
        if idx.numel() == 0 and w.numel() == 0:
            continue
        if idx.dim() != 2 or idx.shape[1] != 3 or w.shape != (idx.shape[0],):
            raise ValueError(f"{where}: edges['{name}'] must be (idx[E,3], w[E])")
        if (idx < 0).any() or (idx[:, 0] >= P).any() or (idx[:, 1:] >= N).any():
            raise ValueError(f"{where}: edges['{name}'] has indices outside P={P}, N={N}")
    return P


def load_dump(path: str | Path, labels: str = DEFAULT_LABELS) -> dict:
    """``torch.load`` one split (safe loader first, trusted-pickle fallback) and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        dump = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # dumps written with numpy scalars need the full unpickler (local, trusted)
        dump = torch.load(path, map_location="cpu", weights_only=False)
    validate_dump(dump, labels)
    return dump


def match_type_counts(dump: Mapping) -> dict[str, int]:
    """Number of matches per ``match_type`` in a dump."""
    counts: dict[str, int] = {}
    for m in dump["matches"]:
        counts[str(m["match_type"])] = counts.get(str(m["match_type"]), 0) + 1
    return counts


def check_match_types(
    dump: Mapping, allowed: Iterable[str] | None = DEFAULT_MATCH_TYPES, where: str = ""
) -> None:
    """Raise ``ValueError`` if the dump holds matches outside the registered population (None = no check)."""
    if allowed is None:
        return
    allowed = tuple(allowed)
    unexpected = {t: n for t, n in match_type_counts(dump).items() if t not in allowed}
    if unexpected:
        raise ValueError(
            f"{where + ': ' if where else ''}match_type {unexpected} is outside the registered population "
            f"{allowed} (airoyale is excluded by design doc v2.3; widen the list explicitly to include it)"
        )


def load_cohort(
    dump_dir: str | Path,
    cohort: str,
    splits: Sequence[str] = SPLITS,
    labels: str = DEFAULT_LABELS,
    match_types: Iterable[str] | None = DEFAULT_MATCH_TYPES,
) -> dict[str, dict]:
    """Load ``<dump_dir>/<cohort>/<split>.pt`` for every split; meta must agree across splits.

    ``labels`` selects which label set is validated; ``match_types`` is the allowed population (None = any).
    """
    base = Path(dump_dir) / cohort
    if not base.is_dir():
        from model_stgcn.ptr.cohorts import discover_cohorts  # lazy: cohorts imports this module

        have = discover_cohorts(dump_dir) if Path(dump_dir).is_dir() else []
        raise FileNotFoundError(f"cohort '{cohort}' not found in {dump_dir}; available cohorts: {have}")
    out = {s: load_dump(base / f"{s}.pt", labels) for s in splits}
    for s in splits:
        check_match_types(out[s], match_types, f"{cohort}/{s}")
    ref = out[splits[0]]["meta"]
    for s in splits[1:]:
        meta = out[s]["meta"]
        if dump_version(meta) != dump_version(ref):
            raise DumpVersionError(
                f"{cohort}: split {splits[0]} is a {dump_version(ref)} dump but split {s} is a {dump_version(meta)} dump"
            )
        if list(meta["feature_names"]) != list(ref["feature_names"]):
            raise ValueError(f"{cohort}: feature_names differ between {splits[0]} and {s}")
        a = {k: list(map(int, v)) for k, v in meta["feature_groups"].items()}
        b = {k: list(map(int, v)) for k, v in ref["feature_groups"].items()}
        if a != b:
            raise ValueError(f"{cohort}: feature_groups differ between {splits[0]} and {s}")
    return out


def removed_columns(
    meta: Mapping, drop_groups: Iterable[str] = (), keep_names: Iterable[str] = ()
) -> np.ndarray:
    """Sorted column indices removed with the named feature groups, except columns named in ``keep_names``."""
    groups = meta["feature_groups"]
    names = list(meta["feature_names"])
    keep = set(keep_names)
    drop: set[int] = set()
    for name in drop_groups:
        if name not in groups:
            raise KeyError(f"unknown feature group '{name}'; available: {sorted(groups)}")
        drop |= {int(c) for c in groups[name]}
    return np.array(sorted(c for c in drop if names[c] not in keep), dtype=np.int64)


def keep_columns(
    meta: Mapping, drop_groups: Iterable[str] = (), keep_names: Iterable[str] = ()
) -> np.ndarray:
    """Sorted column indices that survive removing the named feature groups (``keep_names`` survive too)."""
    gone = set(removed_columns(meta, drop_groups, keep_names).tolist())
    return np.array([c for c in range(len(meta["feature_names"])) if c not in gone], dtype=np.int64)


def feature_removal_report(
    meta: Mapping, drop_groups: Iterable[str] = (), keep_names: Iterable[str] = ()
) -> dict:
    """What a group removal takes out, by name (written to ``metrics.json`` / the run log)."""
    drop_groups, keep_names = list(drop_groups), list(keep_names)
    names = list(meta["feature_names"])
    removed = removed_columns(meta, drop_groups, keep_names)
    in_groups = {int(c) for g in drop_groups for c in meta["feature_groups"][g]}
    spared = sorted(in_groups - set(removed.tolist()))
    return {
        "drop_groups": drop_groups,
        "keep_names": keep_names,
        "n_removed": int(removed.size),
        "removed_columns": [names[int(c)] for c in removed],
        "removed_indices": [int(c) for c in removed],
        "kept_by_exception": [names[c] for c in spared],
    }


def infer_combat_weights(dump_or_matches: Mapping | Sequence[Mapping]) -> tuple[str, float]:
    """``("log", max_w)`` when every combat weight is <= :data:`COMBAT_LOG_MAX` (the dump already holds
    ``ln(1 + intensity)``), else ``("raw", max_w)``.  Fit on the train split and reuse for the others."""
    matches = dump_or_matches["matches"] if isinstance(dump_or_matches, Mapping) else dump_or_matches
    top = 0.0
    for m in matches:
        item = m["edges"].get("combat") if isinstance(m["edges"], Mapping) else None
        if item is None:
            continue
        w = _t(item[1], torch.float32)
        w = w[torch.isfinite(w)]
        if w.numel():
            top = max(top, float(w.max()))
    return ("log" if top <= COMBAT_LOG_MAX else "raw"), top


# --------------------------------------------------------------------------- standardiser
class _Moments:
    """Streaming NaN-aware per-column mean / M2 (Chan et al. parallel merge, float64)."""

    def __init__(self, dim: int) -> None:
        self.n = np.zeros(dim)
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, block: np.ndarray) -> None:
        if block.shape[0] == 0:
            return
        block = np.asarray(block, dtype=np.float64)
        valid = np.isfinite(block)
        nb = valid.sum(0).astype(np.float64)
        if not nb.any():
            return
        safe = np.where(valid, block, 0.0)
        mean_b = np.divide(safe.sum(0), nb, out=np.zeros_like(nb), where=nb > 0)
        m2_b = (np.where(valid, block - mean_b, 0.0) ** 2).sum(0)
        tot = self.n + nb
        frac = np.divide(nb, tot, out=np.zeros_like(nb), where=tot > 0)
        delta = mean_b - self.mean
        self.mean = self.mean + delta * frac
        self.m2 = self.m2 + m2_b + delta**2 * self.n * frac
        self.n = tot

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(np.divide(self.m2, self.n, out=np.zeros_like(self.m2), where=self.n > 0))
        std = np.where(np.isfinite(std) & (std > STD_FLOOR), std, 1.0)
        return self.mean.copy(), std


class Standardizer:
    """Train-only z-scoring for X (all F columns), team positions and zone features."""

    FIELDS = ("x_mean", "x_std", "pos_mean", "pos_std", "zone_mean", "zone_std", "ai_mean", "ai_std")

    def __init__(self, **arrays: np.ndarray) -> None:
        for name in self.FIELDS:
            setattr(self, name, np.asarray(arrays[name], dtype=np.float64))

    @classmethod
    def fit(
        cls,
        dump_or_matches: Mapping | Sequence[Mapping],
        labels: str = DEFAULT_LABELS,
        event: bool | None = None,
    ) -> "Standardizer":
        """Statistics over the alive tokens (``effective_alive`` of the label choice) of the given matches.

        Given a dump the alive rule follows :func:`uses_event_alive`; for a bare list of matches pass ``event``.  On a v2
        dump the tokens are the stored ``query_mask`` rows (comeback tokens included: they are model inputs)."""
        if isinstance(dump_or_matches, Mapping):
            matches = dump_or_matches["matches"]
            order = _zone_order(dump_or_matches.get("meta", {}))
            event = uses_event_alive(dump_or_matches, labels) if event is None else event
        else:
            matches, order = dump_or_matches, np.arange(len(ZONE_COLUMNS))
            event = bool(event)
        xs = _Moments(int(_t(matches[0]["X"]).shape[-1]))
        pos, zone, ai = _Moments(2), _Moments(len(ZONE_COLUMNS)), _Moments(1)
        for m in matches:
            alive = effective_alive(m, labels, event).numpy()
            xs.update(_t(m["X"]).numpy()[alive])
            pos.update(_t(m["team_pos"]).numpy()[alive])
            share = _t(m["team_ai_frac"], torch.float32).numpy()  # [N], a team property: one value per token
            ai.update(np.broadcast_to(share[None, :], alive.shape)[alive][:, None])
            z = _t(m["zone"]).numpy()[:, order]
            zone.update(z[_bool(m["zone_valid"]).numpy()])
        out = {}
        for prefix, mom in (("x", xs), ("pos", pos), ("zone", zone), ("ai", ai)):
            out[f"{prefix}_mean"], out[f"{prefix}_std"] = mom.finalize()
        return cls(**out)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in self.FIELDS}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "Standardizer":
        return cls(**{name: np.asarray(state[name]) for name in cls.FIELDS})

    @staticmethod
    def _scale(arr, mean, std) -> torch.Tensor:
        z = (np.asarray(arr, dtype=np.float64) - mean) / std
        return torch.from_numpy(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

    def transform_x(self, X, keep: np.ndarray | None = None) -> torch.Tensor:
        X = _t(X).numpy()
        if keep is None:
            return self._scale(X, self.x_mean, self.x_std)
        return self._scale(X[..., keep], self.x_mean[keep], self.x_std[keep])

    def transform_pos(self, pos) -> torch.Tensor:
        return self._scale(_t(pos).numpy(), self.pos_mean, self.pos_std)

    def transform_zone(self, zone_canonical) -> torch.Tensor:
        return self._scale(_t(zone_canonical).numpy(), self.zone_mean, self.zone_std)

    def transform_ai(self, team_ai_frac) -> torch.Tensor:
        """``[N]`` AI shares -> ``[N]`` z-scores (NaN -> 0)."""
        return self._scale(_t(team_ai_frac, torch.float32).numpy(), self.ai_mean[0], self.ai_std[0])


def _refuse_player_layer_on_v2(dump: Mapping) -> None:
    if dump_version(dump) == DUMP_V2:
        raise DumpVersionError(
            "the stage-2 player layer (M11RZ-P arms) is not defined for a v2 dump: the player dump is derived from the v1 "
            "phase dump (event-based alive flags, v1 phase bounds) and has not been rebuilt for v2"
        )


class PlayerStandardizer:
    """Train-only z-scoring of the player features ``Xp`` over the alive player tokens (stage 2)."""

    FIELDS = ("xp_mean", "xp_std")

    def __init__(self, **arrays: np.ndarray) -> None:
        for name in self.FIELDS:
            setattr(self, name, np.asarray(arrays[name], dtype=np.float64))

    @classmethod
    def fit(
        cls, phase_dump: Mapping, players: Mapping[str, Mapping], labels: str = DEFAULT_LABELS, event: bool | None = None
    ) -> "PlayerStandardizer":
        """Statistics over the alive players (:func:`player.alive_players` under the label choice's team alive rule)."""
        _refuse_player_layer_on_v2(phase_dump)
        event = uses_event_alive(phase_dump, labels) if event is None else event
        matches = phase_dump["matches"]
        first = players[str(matches[0]["match_id"])]
        mom = _Moments(int(_t(first["Xp"]).shape[-1]))
        for m in matches:
            pm = players[str(m["match_id"])]
            p_alive = PL.alive_players(effective_alive(m, labels, event), pm).numpy()
            mom.update(_t(pm["Xp"]).numpy()[p_alive])
        mean, std = mom.finalize()
        return cls(xp_mean=mean, xp_std=std)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in self.FIELDS}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "PlayerStandardizer":
        return cls(**{name: np.asarray(state[name]) for name in cls.FIELDS})

    def transform(self, xp) -> torch.Tensor:
        """``[P, M, Fp]`` raw -> z-scores, NaN / inf -> 0 (after scaling, like the team features)."""
        return Standardizer._scale(_t(xp).numpy(), self.xp_mean, self.xp_std)


# --------------------------------------------------------------------------- dataset / collate
class PTRDataset:
    """One item per match (unpadded); :meth:`get_batch` pads within the batch.

    ``labels`` picks the label set ('v2' default, 'pipeline'): the risk set, the hazard labels and the alive
    tokens all follow it (``alive_source`` names the rule: the event-based mask for v2 on an event-based dump,
    else ``node_mask`` raised over that risk set).  On a **v2 dump** (``meta['version'] == 'v2'``) the labels are the
    stored ones and every item carries two masks: ``query_mask`` (``node_mask | target``: the tokens, the rows that
    exist, shuffled and standardised) and ``node_mask`` (the attention KEYS: teams causally alive at the cut; a comeback
    row is a query without being a key); ``target_mask`` (the loss / metric rows) is a subset of ``query_mask``.  For a v1
    dump both masks are the alive tokens.  ``relation_types=None`` skips edge
    handling altogether (no ``rel`` tensor).  With ``shuffle_seed`` the relation matrices are relabelled by a
    fixed per-(match, phase, seed) permutation of the alive teams (the M11R-shuf control), identically on
    every access.  Token features = kept ``X`` columns, [team centre x/y], ``team_ai_frac``, then (union arms only) the extra
    ``kf_*`` columns of ``extra`` (a :class:`model_stgcn.ptr.extra.ExtraSplit`, standardised by ``extra_standardizer``: z-scored
    on train, NaN -> 0), joined by cut phase: slot ``p`` carries the values of cut ``p + 1``.
    """

    def __init__(
        self,
        dump: Mapping,
        standardizer: Standardizer,
        *,
        drop_groups: Iterable[str] = (),
        keep_names: Iterable[str] = (),
        pos_in_token: bool = True,
        relation_types: Sequence[str] | None = None,
        combat_weights: str = "raw",
        shuffle_seed: int | None = None,
        labels: str = DEFAULT_LABELS,
        validate: bool = True,
        players: Mapping[str, Mapping] | None = None,
        player_standardizer: PlayerStandardizer | None = None,
        player_relation_types: Sequence[str] | None = None,
        player_combat_weights: str = "raw",
        extra=None,
        extra_standardizer=None,
    ) -> None:
        self.labels = check_labels(labels)
        if validate:
            validate_dump(dump, self.labels)
        self.version = dump_version(dump)
        if players is not None:
            _refuse_player_layer_on_v2(dump)
        self.event_alive = uses_event_alive(dump, self.labels)
        self.alive_source = alive_source(dump, self.labels)
        if combat_weights not in R.COMBAT_WEIGHT_MODES:
            raise ValueError(f"combat_weights must be one of {R.COMBAT_WEIGHT_MODES}, got '{combat_weights}'")
        self.meta = dump["meta"]
        self.matches = dump["matches"]
        self.standardizer = standardizer
        self.combat_weights = combat_weights
        self.drop_groups = tuple(drop_groups)
        self.keep_names = tuple(keep_names)
        self.keep = keep_columns(self.meta, self.drop_groups, self.keep_names)
        self.removal = feature_removal_report(self.meta, self.drop_groups, self.keep_names)
        self.pos_in_token = bool(pos_in_token)
        self.relation_types = None if relation_types is None else R.check_relation_types(relation_types)
        self.shuffle_seed = shuffle_seed
        self.zone_order = _zone_order(self.meta)
        names = [self.meta["feature_names"][int(c)] for c in self.keep]
        self.extra = extra
        self.extra_standardizer = extra_standardizer
        self.extra_names: list[str] = []
        if extra is not None:
            self.extra_names = check_extra_names(extra.names)  # nothing but kf_* columns enters X
            if extra_standardizer is None:
                raise ValueError("extra tokens given without an extra_standardizer")
            lacking = [str(m["match_id"]) for m in self.matches if str(m["match_id"]) not in extra.tokens]
            if lacking:
                raise ValueError(f"{len(lacking)} match(es) have no extra tokens (first: {lacking[0]})")
        self.feature_names = (
            names + (list(POS_FEATURE_NAMES) if self.pos_in_token else []) + [AI_FEATURE_NAME] + self.extra_names
        )
        self.n_features = len(self.feature_names)
        self.n_phases = int(_t(self.matches[0]["X"]).shape[0])
        self.zone_dim = len(ZONE_COLUMNS)
        # stage 2: the joined player matches (match_id -> player match); None for the team-level arms
        self.players = players
        self.player_standardizer = player_standardizer
        if player_combat_weights not in R.COMBAT_WEIGHT_MODES:
            raise ValueError(f"player_combat_weights must be one of {R.COMBAT_WEIGHT_MODES}, got '{player_combat_weights}'")
        self.player_combat_weights = player_combat_weights
        self.player_relation_types = None if player_relation_types is None else PL.check_player_relation_types(player_relation_types)
        self.n_player_features = 0
        if players is not None:
            if player_standardizer is None:
                raise ValueError("players given without a player_standardizer")
            self.n_player_features = int(_t(players[str(self.matches[0]["match_id"])]["Xp"]).shape[-1])

    def __len__(self) -> int:
        return len(self.matches)

    def __getitem__(self, i: int) -> dict:
        m = self.matches[i]
        sz = self.standardizer
        y, target = select_labels(m, self.labels)
        alive = effective_alive(m, self.labels, self.event_alive)  # tokens: event mask / raised node_mask / v2 query_mask
        key = key_mask(m, self.labels, self.event_alive)  # attention keys: v2 node_mask (causally alive at the cut), else = tokens
        x = sz.transform_x(m["X"], self.keep)
        pos_raw = _t(m["team_pos"], torch.float32)
        pos_valid = torch.isfinite(pos_raw).all(-1)
        if self.pos_in_token:
            x = torch.cat([x, sz.transform_pos(pos_raw)], dim=-1)
        ai = sz.transform_ai(m["team_ai_frac"])  # [N]: the team's AI share, repeated in every phase slot
        x = torch.cat([x, ai[None, :, None].expand(x.shape[0], -1, 1)], dim=-1)
        if self.extra is not None:  # slot p carries the kf values of cut p + 1; the last slot has no cut (NaN -> 0)
            x = torch.cat([x, self.extra_standardizer.transform(self.extra.tokens[str(m["match_id"])])], dim=-1)
        if x.shape[-1] != self.n_features:
            raise AssertionError(f"token features have {x.shape[-1]} columns, expected {self.n_features} ({self.feature_names})")
        zone_raw = _t(m["zone"], torch.float32)[:, torch.from_numpy(self.zone_order)]
        zone_ok = _bool(m["zone_valid"]) & torch.isfinite(zone_raw).all(-1)
        zero = torch.zeros(())
        item = {
            "match_id": str(m["match_id"]),
            "match_type": str(m["match_type"]),
            "team_ids": _t(m["team_ids"], torch.int64),
            "x": x,
            "node_mask": key,  # attention KEY mask (v1: the alive tokens; v2: causally alive at the cut)
            "target_mask": target,
            "y_hazard": y,
            "final_rank_norm": torch.nan_to_num(_t(m["final_rank_norm"], torch.float32)),
            "top5": torch.nan_to_num(_t(m["top5"], torch.float32)),
            "pos": torch.where(pos_valid.unsqueeze(-1), pos_raw, zero),
            "pos_valid": pos_valid,
            "zone": torch.where(zone_ok.unsqueeze(-1), zone_raw, zero),
            "zone_std": torch.where(zone_ok.unsqueeze(-1), sz.transform_zone(zone_raw), zero),
            "zone_valid": zone_ok,
        }
        if self.version == DUMP_V2:  # v1 batches carry no query_mask: the model then reads node_mask for keys AND tokens
            item["query_mask"] = alive  # tokens / queries: node_mask | target_mask (comeback rows are queries, not keys)
        if self.relation_types is not None:
            edges = m["edges"]
            if self.shuffle_seed is not None:
                perms = R.shuffle_permutations(m["match_id"], self.shuffle_seed, alive.numpy())
                edges = R.shuffle_edges(edges, perms)
            item["edges"] = edges
        if self.players is not None:
            pm = self.players[str(m["match_id"])]
            p_alive = PL.alive_players(alive, pm)
            xp = self.player_standardizer.transform(pm["Xp"])
            item.update(
                xp=torch.where(p_alive.unsqueeze(-1), xp, zero),
                p_alive=p_alive,
                p_team=_t(pm["player_team_idx"], torch.int64).clamp(min=0),
                p_ai=torch.nan_to_num(_t(pm["player_is_ai"], torch.float32)),
                p_edges=pm["edges_player"],
            )
        return item

    def get_batch(self, indices: Sequence[int]) -> dict:
        return collate_batch(
            [self[int(i)] for i in indices], self.relation_types, self.combat_weights,
            self.player_relation_types, self.player_combat_weights,
        )  # fmt: skip


def collate_batch(
    items: Sequence[dict],
    relation_types: Sequence[str] | None = None,
    combat_weights: str = "raw",
    player_relation_types: Sequence[str] | None = None,
    player_combat_weights: str = "raw",
) -> dict:
    """Pad teams (and, in stage 2, players) to the largest match in the batch; padding is dead (all masks False, labels 0).

    ``node_mask`` is the attention key mask, ``query_mask`` (carried by the items of a v2 dump) the token mask:
    ``target_mask`` and ``node_mask`` are subsets of ``query_mask``.  Without ``query_mask`` the model reads ``node_mask``
    for keys and tokens alike (v1)."""
    if not items:
        raise ValueError("empty batch")
    B = len(items)
    P, _, F = items[0]["x"].shape
    n_max = max(int(it["x"].shape[1]) for it in items)
    if any(int(it["x"].shape[0]) != P for it in items):
        raise ValueError("matches in one batch must share the number of phase slots")
    Z = items[0]["zone"].shape[-1]
    batch = {
        "x": torch.zeros(B, P, n_max, F),
        "node_mask": torch.zeros(B, P, n_max, dtype=torch.bool),
        "target_mask": torch.zeros(B, P, n_max, dtype=torch.bool),
        "y_hazard": torch.zeros(B, P, n_max),
        "final_rank_norm": torch.zeros(B, n_max),
        "top5": torch.zeros(B, n_max),
        "team_ids": torch.full((B, n_max), -1, dtype=torch.int64),
        "pos": torch.zeros(B, P, n_max, 2),
        "pos_valid": torch.zeros(B, P, n_max, dtype=torch.bool),
        "zone": torch.zeros(B, P, Z),
        "zone_std": torch.zeros(B, P, Z),
        "zone_valid": torch.zeros(B, P, dtype=torch.bool),
    }
    team_keys = ("final_rank_norm", "top5", "team_ids")
    token_keys = ("x", "node_mask", "target_mask", "y_hazard", "pos", "pos_valid")
    if "query_mask" in items[0]:  # tokens / queries; the model reads node_mask as the key mask and this as the token mask
        batch["query_mask"] = torch.zeros(B, P, n_max, dtype=torch.bool)
        token_keys = token_keys + ("query_mask",)
    for b, it in enumerate(items):
        n = int(it["x"].shape[1])
        for key in token_keys:
            batch[key][b, :, :n] = it[key]
        for key in team_keys:
            batch[key][b, :n] = it[key]
        for key in ("zone", "zone_std", "zone_valid"):
            batch[key][b] = it[key]
    batch["n_teams"] = torch.tensor([int(it["x"].shape[1]) for it in items], dtype=torch.int64)
    batch["match_ids"] = [it["match_id"] for it in items]
    batch["match_types"] = [it["match_type"] for it in items]
    if relation_types is not None:
        dense = R.densify_edges_batch([it.get("edges") for it in items], P, n_max)
        batch["rel"] = R.build_relation_tensor(dense, relation_types, combat_weights)
    if "xp" in items[0]:
        m_max = max(int(it["xp"].shape[1]) for it in items)
        Fp = int(items[0]["xp"].shape[-1])
        batch["xp"] = torch.zeros(B, P, m_max, Fp)
        batch["p_alive"] = torch.zeros(B, P, m_max, dtype=torch.bool)
        batch["p_team"] = torch.zeros(B, m_max, dtype=torch.int64)
        batch["p_ai"] = torch.zeros(B, m_max)
        for b, it in enumerate(items):
            m_b = int(it["xp"].shape[1])
            batch["xp"][b, :, :m_b] = it["xp"]
            batch["p_alive"][b, :, :m_b] = it["p_alive"]
            batch["p_team"][b, :m_b] = it["p_team"]
            batch["p_ai"][b, :m_b] = it["p_ai"]
        if player_relation_types is not None:
            dense_p = PL.densify_player_edges_batch([it.get("p_edges") for it in items], P, m_max)
            batch["rel_p"] = PL.build_player_relation_tensor(dense_p, player_relation_types, player_combat_weights)
    return batch
