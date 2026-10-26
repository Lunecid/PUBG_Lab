"""Compact label tables of the PTR dumps (model_stgcn/dumps/bigcomp2027_phase_v2) for the KRAFTON baseline.

The baseline uses the dumps' cohorts, splits, death times (``team_death_s``), winner flags, phase boundaries and the v2
risk set / hazard labels (``select_labels`` of model_stgcn.ptr.data is called, not re-implemented, so that the rows are
the PTR rows by construction).  Only small per-match arrays are kept; the big feature tensors of the dump are dropped,
so workers and the training process never load them.

    python -m adaptation.labels --dump-dir model_stgcn/dumps/bigcomp2027_phase_v2 --out-dir <out> [--cohorts a,b]

v2 dump (``meta['version'] == 'v2'``): the dump is the single source of truth.  The compact match carries, exactly as
stored, ``team_death_s`` (T, winners ``+inf``), the risk set ``target`` and ``y``, the key mask ``node_mask`` (causally
alive at the cut), the token mask ``query_mask`` (``node_mask | target``), the exact float64 cut times ``cut_t``
(``a_k = b_k - 10``, :func:`model_stgcn.ptr.data.cut_times`) and ``match_end_s`` (``C_m``); nothing is re-derived from
death times.  Every label pickle records ``dump_version`` so that a consumer can refuse a v1 table next to a v2 dump.

The dump is read through the PTR loader's own validation (:func:`model_stgcn.ptr.data.validate_dump`) and every record must follow
``meta['version']`` (:func:`check_record_versions`): ``select_labels`` decides v1 vs v2 per record, so a record of a v2 dump that carries no
v2-only key would silently take the v1 branch (winner override, ``_cut_exists``, masks from T) -- ``DumpVersionError`` instead.
"""

from __future__ import annotations

# Project code is separate from the immutable vendored reference.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))
from pubg_lab.common.paths import REPO_ROOT

import argparse
import pickle
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

REPO = REPO_ROOT
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pubg_lab.survival.ptr.data.data import (
    DEFAULT_MATCH_TYPES,
    DUMP_V1,
    DUMP_V2,
    DumpVersionError,
    boundary_times,
    check_match_types,
    check_same_version,
    cut_times,
    dump_version,
    is_v2_match,
    select_labels,
    validate_dump,
)

SPLITS = ("train", "val", "test")


def _np(x, dtype=None) -> np.ndarray:
    a = torch.as_tensor(x).numpy() if not isinstance(x, np.ndarray) else x
    return a if dtype is None else a.astype(dtype)


def compact_match(m: dict) -> dict:
    y, target = select_labels(
        m, "v2"
    )  # v2 dump: the stored risk set / labels; v1 dump: v2 labels + winner censoring
    out = {
        "match_id": str(m["match_id"]),
        "match_type": str(m["match_type"]),
        "team_ids": _np(m["team_ids"], np.int64),
        "team_death_s": _np(m["team_death_s"], np.float64),
        "team_won": _np(m["team_won"], bool),
        "phase_end_s": _np(m["phase_end_s"], np.float64),
        "phase_start_s": _np(m["phase_start_s"], np.float64),
        "phase_mask": _np(m["phase_mask"], bool),
        "target": _np(target, bool),
        "y": _np(y, np.float32),
        "final_rank_norm": _np(m["final_rank_norm"], np.float32),
        "top5": _np(m["top5"], np.float32),
        "last_phase_idx": None if m.get("last_phase_idx") is None else int(m["last_phase_idx"]),
        "clock_delta_s": float(m.get("clock_delta_s", 0.0)),
    }
    if is_v2_match(m):  # masks, exact cut times and the censoring time straight from the v2 record
        out.update(
            dump_version=DUMP_V2,  # marker: extract / kftable refuse a half-stripped v2 label instead of re-deriving from T
            phase_end_s=boundary_times(
                m["phase_end_s"]
            ),  # float32 in the dump -> exact float64 b_k (ms resolution)
            phase_start_s=boundary_times(m["phase_start_s"]),
            node_mask=_np(m["node_mask"], bool),
            query_mask=_np(m["query_mask"], bool),
            cut_t=cut_times(m["phase_end_s"]),
            match_end_s=float(m["match_end_s"])
            if m.get("match_end_s") is not None
            else float("nan"),
            cut_valid=_np(m["cut_valid"], bool) if "cut_valid" in m else None,
        )
    return out


def label_version(rec_or_lab: dict) -> str:
    """``'v2'`` for a label record / compact match of a v2 dump, else ``'v1'`` (old pickles carry no ``dump_version``)."""
    if "dump_version" in rec_or_lab:
        return str(rec_or_lab["dump_version"])
    return DUMP_V2 if "query_mask" in rec_or_lab else DUMP_V1


def check_label_version(rec: dict, want: str, against: str) -> None:
    """Raise ``DumpVersionError`` unless the label pickle ``rec`` was built from a ``want`` dump."""
    check_same_version(
        label_version(rec),
        want,
        what=f"the label table {rec.get('cohort', '?')}/{rec.get('split', '?')}",
        against=against,
    )


def check_record_versions(dump: dict, where: str = "") -> str:
    """Raise ``DumpVersionError`` unless ``is_v2_match(m) == (dump_version(meta) == 'v2')`` for EVERY record ``m``; returns the dump version.

    ``compact_match`` -> ``select_labels`` picks the v1 or the v2 branch per record from the record's own keys.  A v2 dump whose record lacks
    every v2-only key would therefore be read through the v1 rules (winner override, ``_cut_exists``, masks re-derived from T: the look-ahead
    the v2 dump removes), and a v1 dump with a v2-only key through the v2 rules.  Run before :func:`validate_dump`, whose own error for the
    first case is a plain ``ValueError`` (missing keys)."""
    version = dump_version(dump)  # DumpVersionError for an unsupported meta['version']
    matches = dump.get("matches") if isinstance(dump, Mapping) else None
    if not isinstance(matches, (list, tuple)):
        return version  # not a dump: validate_dump says so
    want_v2 = version == DUMP_V2
    bad = [
        str(m.get("match_id", f"#{i}"))
        for i, m in enumerate(matches)
        if isinstance(m, Mapping) and is_v2_match(m) != want_v2
    ]
    if bad:
        what = (
            "have no v2-only key, so select_labels would read them through the v1 rules (winner override, _cut_exists, masks and risk set from T)"
            if want_v2
            else "have a v2-only key, so they would be read through the v2 rules"
        )
        raise DumpVersionError(
            f"{where + ': ' if where else ''}meta['version'] is {version!r} but {len(bad)} of {len(matches)} record(s) {what} "
            f"(V2_ONLY_KEYS of model_stgcn.ptr.data); e.g. match {', '.join(bad[:3])}"
        )
    return version


def load_validated_dump(path: Path | str, where: str = "") -> dict:
    """``torch.load`` one split of a dump and refuse what the PTR loader (``model_stgcn.ptr.data.load_dump``) would refuse: every record
    must follow ``meta['version']`` (:func:`check_record_versions`, ``DumpVersionError``), then ``validate_dump`` (labels ``'v2'``)."""
    dump = torch.load(Path(path), map_location="cpu", weights_only=False)
    check_record_versions(dump, where or str(path))
    validate_dump(dump)
    return dump


def build_cohort(dump_dir: Path, cohort: str, out_dir: Path) -> dict:
    base = dump_dir / cohort
    info = {"cohort": cohort, "splits": {}}
    for split in SPLITS:
        t0 = time.time()
        dump = load_validated_dump(base / f"{split}.pt", f"{cohort}/{split}")
        check_match_types(dump, DEFAULT_MATCH_TYPES, f"{cohort}/{split}")
        meta = dump["meta"]
        mats = [compact_match(m) for m in dump["matches"]]
        rec = {
            "cohort": cohort, "split": split, "map_name": meta.get("map_name"), "game_mode": meta.get("game_mode"),
            "split_ids_sha1": meta.get("split_ids_sha1"), "matches": mats,
            "dump_version": dump_version(meta), "dump_git_commit": meta.get("git_commit"), "dump_created_at": meta.get("created_at"),
        }  # fmt: skip
        out = out_dir / cohort
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"labels_{split}.pkl", "wb") as f:
            pickle.dump(rec, f, protocol=4)
        n_rows = int(sum(m["target"].sum() for m in mats))
        info["splits"][split] = {
            "n_matches": len(mats),
            "n_target_rows": n_rows,
            "sec": round(time.time() - t0, 1),
        }
        print(
            f"{cohort}/{split}: {len(mats)} matches, {n_rows} risk-set rows, {time.time() - t0:.1f}s",
            flush=True,
        )
        del dump
    return info


def load_labels(out_dir: Path | str, cohort: str, split: str) -> dict:
    with open(Path(out_dir) / cohort / f"labels_{split}.pkl", "rb") as f:
        return pickle.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=str(REPO / "model_stgcn/dumps/bigcomp2027_phase_v2"))
    ap.add_argument("--out-dir", default=str(REPO / "model_stgcn/dumps/krafton_winprob_v2"))
    ap.add_argument("--cohorts", default="")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    dump_dir, out_dir = Path(a.dump_dir), Path(a.out_dir)
    from pubg_lab.survival.ptr.data.cohorts import discover_cohorts  # read-only

    cohorts = [c for c in a.cohorts.split(",") if c] or list(discover_cohorts(dump_dir))
    for c in cohorts:
        build_cohort(dump_dir, c, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
