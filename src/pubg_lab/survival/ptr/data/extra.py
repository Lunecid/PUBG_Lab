"""Extra team-level features of the UNION arms (design doc 4.6, post hoc / exploratory): the join of the ``kf_*`` features
that the KRAFTON agent reconstructs (KRAFTON's input representation) onto our own tokens (neural) and rows (HGB).

Table, one per cohort and split, written outside this package::

    <extra-dir>/<cohort>/{train,val,test}.csv.gz      (.csv too; .parquet only where pyarrow / fastparquet is installed)
    columns  match_id, team_id, phase, kf_*  [, in_risk_set, y_hazard_v2]        every other column is ignored

``phase`` is the CUT phase ``k``, the ``phase`` of the prediction CSVs: the cut at the end of dump slot ``p = k - 1`` (the
hazard of the next phase).  The ``kf_*`` values of ``(match, team, k)`` become the extra token features of slot ``k - 1``
for the neural arms ("broadcast to the phase slots by cut k"), appended after the kept ``X`` columns, the team centre and
``team_ai_frac``, and the extra columns of the HGB row ``(match, team, k)``.  Both read the same join (:func:`join_split`).

Coverage is asserted: every alive token of a cut slot (``data.effective_alive`` of the label choice at the slots
``0 .. P - 2``; it contains every target row, since a team at risk at cut ``p`` is alive at the start of slot ``p``) needs a
row, otherwise :class:`ExtraCoverageError` lists the gaps.  On a v2 dump the tokens are the stored ``query_mask`` rows
(``node_mask | target``, comeback rows included) and the table must have been built from that same dump version:
``<extra-dir>/meta.json`` records ``dump_version`` and :func:`load_extra` raises :class:`model_stgcn.ptr.data.DumpVersionError`
for a v1 table next to a v2 dump (or the other way round); a table without ``meta.json`` counts as v1.  The last slot has no cut, so no row: its tokens (counted in
``report['last_slot_alive_tokens']``) are NaN, i.e. 0 after standardisation, and cannot influence a prediction (the phase
axis is causal and that slot has no targets).  Rows nobody needs (dead teams, other matches) are ignored and counted.
NaN values are allowed (HGB reads them natively, the neural arms zero-fill after z-scoring); +-inf becomes NaN and is counted.

When the table carries its producer's label columns (``in_risk_set``, ``y_hazard_v2``) and the labels are v2, the join is also
checked against OUR risk set and hazard labels (:class:`ExtraAlignmentError`): a table that is one phase off, or that was
built on another dump, cannot pass.  The check is skipped (and says so in the report) without those columns.

Standardisation (:class:`ExtraStandardizer`): per ``kf`` column over the covered alive tokens of the TRAIN split (NaN-aware,
float64; a constant column gets ``std = 1``), NaN / inf -> 0 after scaling.  The feature set is the sorted list of the table's
``kf_*`` columns; :func:`names_sha1` identifies it (written to ``metrics.json`` and the config hash).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import pubg_lab.survival.ptr.data.data as D
import pubg_lab.survival.ptr.evaluation.metrics as M

EXTRA_PREFIX = D.EXTRA_PREFIX
KEY_COLUMNS: tuple[str, ...] = ("match_id", "team_id", "phase")
ALIGN_COLUMNS: tuple[str, ...] = (
    "in_risk_set",
    "y_hazard_v2",
)  # optional, ONLY for the alignment check, never features
TABLE_SUFFIXES: tuple[str, ...] = (
    ".csv.gz",
    ".csv",
    ".parquet",
)  # preference order: readable without optional packages first
MAX_LISTED_GAPS = 20
_PARQUET_HINT = (
    "pandas needs pyarrow or fastparquet to read parquet and neither is installed here; "
    "write the table as <split>.csv.gz instead"
)


class ExtraCoverageError(ValueError):
    """Alive tokens of the cut slots without a row in the table; ``.gaps`` lists them (match_id, team_id, phase, kind)."""

    def __init__(self, message: str, gaps: pd.DataFrame) -> None:
        super().__init__(message)
        self.gaps = gaps


class ExtraAlignmentError(ValueError):
    """The table's own risk set / labels disagree with ours; ``.problems`` lists the rows."""

    def __init__(self, message: str, problems: pd.DataFrame) -> None:
        super().__init__(message)
        self.problems = problems


# --------------------------------------------------------------------------- reading
DEFAULT_EXTRA_DIRNAME = "bigcomp2027_kraftonfeat_v1"
META_FILE = "meta.json"


def default_extra_dir(dump_dir: str | Path) -> Path:
    """The kf tables next to the phase dump directory: ``<dumps>/bigcomp2027_kraftonfeat_v2`` for a dump directory named
    ``*_v2`` (the v2 dump), else ``<dumps>/bigcomp2027_kraftonfeat_v1``.  (:func:`load_extra` checks the version anyway.)"""
    version = "v2" if Path(dump_dir).name.endswith("_v2") else "v1"
    return Path(dump_dir).parent / DEFAULT_EXTRA_DIRNAME.replace("_v1", f"_{version}")


def table_dump_version(extra_dir: str | Path) -> str:
    """``dump_version`` recorded in ``<extra_dir>/meta.json`` by ``adaptation.kftable``; a root without it is a v1 table."""
    path = Path(extra_dir) / META_FILE
    if not path.is_file():
        return D.DUMP_V1
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    value = (
        meta.get("dump_version") if meta.get("dump_version") is not None else meta.get("version")
    )
    if (
        value is None
    ):  # a legacy root names its origin only in ``name`` (``bigcomp2027_kraftonfeat_v1``)
        tokens = re.findall(r"(?:^|[_\-.])(v\d+)(?=$|[_\-.])", str(meta.get("name", "")))
        value = tokens[-1] if tokens else D.DUMP_V1
    return D.dump_version({D.VERSION_KEY: value})


def check_table_version(extra_dir: str | Path, dumps: Mapping[str, Mapping]) -> str:
    """Raise :class:`model_stgcn.ptr.data.DumpVersionError` unless the table root was built from the version of every dump."""
    have = table_dump_version(extra_dir)
    for split, dump in dumps.items():
        D.check_same_version(
            have,
            D.dump_version(dump),
            what=f"the extra-feature tables under {extra_dir}",
            against=f"the {split} dump",
        )
    return have


def table_path(extra_dir: str | Path, cohort: str, split: str) -> Path:
    """``<extra-dir>/<cohort>/<split>`` + the first suffix of :data:`TABLE_SUFFIXES` that exists."""
    base = Path(extra_dir) / cohort
    for suffix in TABLE_SUFFIXES:
        path = base / f"{split}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no extra-feature table for {cohort}/{split}: looked for "
        + ", ".join(str(base / f"{split}{s}") for s in TABLE_SUFFIXES)
    )


def _wanted(column: str) -> bool:
    return column in KEY_COLUMNS or column in ALIGN_COLUMNS or str(column).startswith(EXTRA_PREFIX)


def _read_raw(path: Path) -> pd.DataFrame:
    name = path.name
    if name.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(f"cannot read {path}: {_PARQUET_HINT} ({exc})") from exc
    if name.endswith((".csv.gz", ".csv")):
        # float_precision='round_trip': exact parsing of the table's 9-significant-digit floats; cut_t / alive_at_cut are never read
        return pd.read_csv(
            path,
            usecols=_wanted,
            dtype={"match_id": str},
            low_memory=False,
            float_precision="round_trip",
        )
    raise ValueError(f"unsupported extra-feature table {path} (use .csv.gz, .csv or .parquet)")


def _whole_numbers(col: pd.Series, name: str, where: str) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    bad = ~np.isfinite(num) | (num != np.floor(num))
    if bad.any():
        raise ValueError(
            f"{where}: column {name} must hold whole numbers; {int(bad.sum())} bad value(s), first {col[bad].iloc[0]!r}"
        )
    return num.astype(np.int64)


def _numbers(col: pd.Series, name: str, where: str) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    bad = num.isna() & col.notna()
    if bad.any():
        raise ValueError(
            f"{where}: feature column {name} is not numeric (first bad value {col[bad].iloc[0]!r})"
        )
    return num.astype(np.float32)


def read_table_frame(table: pd.DataFrame, where: str = "table") -> pd.DataFrame:
    """Validate and normalise a table (idempotent): ``match_id`` text, whole-number ``team_id`` / ``phase``, float32 ``kf_*``
    columns sorted by name, the optional :data:`ALIGN_COLUMNS` after them; every other column is dropped.  Duplicate keys,
    missing key columns, no ``kf_*`` column or a non-numeric one raise ``ValueError``."""
    missing = [c for c in KEY_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"{where}: missing key column(s) {missing}; the table needs {list(KEY_COLUMNS)}"
        )
    names = sorted(c for c in table.columns if str(c).startswith(EXTRA_PREFIX))
    if not names:
        raise ValueError(f"{where}: no '{EXTRA_PREFIX}*' feature columns")
    D.check_extra_names(names)
    out = pd.DataFrame({"match_id": table["match_id"].astype(str).to_numpy()})
    for key in ("team_id", "phase"):
        out[key] = _whole_numbers(table[key].reset_index(drop=True), key, where).to_numpy()
    for name in names:
        out[name] = _numbers(table[name].reset_index(drop=True), name, where).to_numpy()
    for name in ALIGN_COLUMNS:
        if name in table.columns:
            out[name] = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=np.float64)
    dup = out.duplicated(list(KEY_COLUMNS), keep=False)
    if dup.any():
        first = out.loc[dup, list(KEY_COLUMNS)].head(3).to_dict("records")
        raise ValueError(
            f"{where}: {int(dup.sum())} rows share a (match_id, team_id, phase) key (duplicate keys), e.g. {first}"
        )
    return out[[*KEY_COLUMNS, *names, *[c for c in ALIGN_COLUMNS if c in out.columns]]]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one table file (see :func:`read_table_frame` for what comes back)."""
    path = Path(path)
    return read_table_frame(_read_raw(path), where=str(path))


def table_feature_names(path: str | Path) -> list[str]:
    """Sorted ``kf_*`` column names of a table file, from its header only."""
    path = Path(path)
    if path.name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            columns = pq.read_schema(path).names
        except ImportError as exc:
            raise RuntimeError(f"cannot read {path}: {_PARQUET_HINT} ({exc})") from exc
    else:
        columns = list(pd.read_csv(path, nrows=0).columns)
    return sorted(c for c in columns if str(c).startswith(EXTRA_PREFIX))


def names_sha1(names: Sequence[str]) -> str:
    """Hash of the ordered feature-name list (the extra-feature set)."""
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()


def extra_spec(extra_dir: str | Path | None, cohort: str, splits: Sequence[str] = D.SPLITS) -> dict:
    """What the config hash of a union run takes from the tables: the feature-set hash (train header) and the size / mtime
    of the table files.  Total: a missing directory, file or unreadable header gives ``None`` entries (the run itself
    fails later, loudly), so that a plan can be made before the tables exist."""
    files: dict[str, list[int] | None] = {}
    names: list[str] | None = None
    for split in splits:
        try:
            path = table_path(extra_dir, cohort, split)
            st = path.stat()
            files[split] = [st.st_size, st.st_mtime_ns]
            if split == "train":
                names = table_feature_names(path)
        except Exception:  # noqa: BLE001 - see the docstring: a missing / unreadable table must not break planning
            files[split] = files.get(split)
    try:
        version = table_dump_version(extra_dir) if extra_dir is not None else None
    except Exception:  # noqa: BLE001 - total, like the rest of this function
        version = None
    return {
        "names_sha1": None if not names else names_sha1(names),
        "n_features": None if not names else len(names),
        "files": files,
        "dump_version": version,
    }


# --------------------------------------------------------------------------- the join
@dataclass
class ExtraSplit:
    """One split's join: ``tokens[match_id]`` is ``float32 [P, N, K]`` (NaN where there is no value), ``names`` the K feature
    names (sorted), ``report`` the counts that go to ``metrics.json``."""

    names: list[str]
    tokens: dict[str, np.ndarray]
    report: dict = field(default_factory=dict)


def _listing(frame: pd.DataFrame, limit: int = MAX_LISTED_GAPS) -> str:
    lines = ["  " + "  ".join(str(c) for c in frame.columns)]
    lines += [
        "  " + "  ".join(str(v) for v in row) for row in frame.head(limit).itertuples(index=False)
    ]
    if len(frame) > limit:
        lines.append(f"  ... and {len(frame) - limit} more")
    return "\n".join(lines)


def _alignment(
    needed: pd.DataFrame, merged: pd.DataFrame, tab: pd.DataFrame, labels: str, where: str
) -> dict:
    """Compare the table's own risk-set flag / hazard label with ours (labels v2, table with the label columns)."""
    if labels != "v2":
        return {"checked": False, "reason": f"labels={labels}"}
    if not all(c in tab.columns for c in ALIGN_COLUMNS):
        return {"checked": False, "reason": "no label columns"}
    ours = needed["target"].to_numpy(dtype=bool)
    risk = merged["in_risk_set"].to_numpy(dtype=np.float64)
    bad_risk = ~(risk == ours.astype(np.float64))  # NaN flags count as disagreeing
    y_tab, y_ours = (
        merged["y_hazard_v2"].to_numpy(dtype=np.float64),
        needed["y"].to_numpy(dtype=np.float64),
    )
    bad_y = ours & ~(np.abs(y_tab - y_ours) < 1e-6)
    risk_rows = tab[(tab["in_risk_set"] == 1) & tab["match_id"].isin(set(needed["match_id"]))]
    stray = risk_rows.merge(
        needed[list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="left", indicator=True
    )
    stray = stray[stray["_merge"] == "left_only"]
    problems = []
    for mask, what, ours_col, table_col in (
        (bad_risk, "in_risk_set differs from our risk set", ours.astype(int), risk),
        (bad_y, "y_hazard_v2 differs from our hazard label", y_ours, y_tab),
    ):
        if mask.any():
            problems.append(
                pd.DataFrame(
                    {
                        **{k: needed[k].to_numpy()[mask] for k in KEY_COLUMNS},
                        "problem": what,
                        "table": table_col[mask],
                        "ours": ours_col[mask],
                    }
                )
            )
    if len(stray):
        problems.append(
            pd.DataFrame(
                {
                    **{k: stray[k].to_numpy() for k in KEY_COLUMNS},
                    "problem": "table risk row that is not in our risk set (no alive token of ours)",
                    "table": 1.0,
                    "ours": np.nan,
                }
            )
        )
    if problems:
        bad = pd.concat(problems, ignore_index=True)
        raise ExtraAlignmentError(
            f"extra features ({where}): the table disagrees with our labels on {len(bad)} rows "
            f"(in_risk_set: {int(bad_risk.sum())}, y_hazard_v2: {int(bad_y.sum())}, table risk rows not in our risk set: "
            f"{len(stray)}); is the cut phase off by one, or is the table built on another dump?\n"
            + _listing(bad),
            bad,
        )
    return {
        "checked": True,
        "rows_compared": int(len(needed)),
        "target_rows_compared": int(ours.sum()),
        "risk_set_mismatch": 0,
        "label_mismatch": 0,
        "table_risk_rows_not_ours": 0,
    }


def join_split(
    dump: Mapping,
    table: pd.DataFrame,
    labels: str = D.DEFAULT_LABELS,
    *,
    split: str = "",
    cohort: str = "",
    file: str | None = None,
) -> ExtraSplit:
    """Join a table onto the matches of one split dump (see the module docstring for the contract).

    Raises :class:`ExtraCoverageError` when an alive token of a cut slot has no row and :class:`ExtraAlignmentError` when
    the table's own labels disagree with ours."""
    D.check_labels(labels)
    where = "/".join(x for x in (cohort, split) if x) or "table"
    tab = read_table_frame(table, where)
    names = D.check_extra_names(
        c for c in tab.columns if str(c).startswith(EXTRA_PREFIX)
    )  # only kf_* columns are features
    K = len(names)
    event = D.uses_event_alive(dump, labels)
    ids: list[str] = []
    shapes: list[tuple[int, int]] = []
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("team", "p", "i", "target", "y")}
    last_slot = 0
    for m in dump["matches"]:
        mid = str(m["match_id"])
        alive = D.effective_alive(m, labels, event).numpy()
        y, target = D.select_labels(m, labels)
        P, N = alive.shape
        p_idx, i_idx = np.nonzero(alive[: P - 1])  # the cut slots; the last slot has no cut
        ids.append(mid)
        shapes.append((P, N))
        last_slot += int(alive[P - 1].sum())
        parts["team"].append(np.asarray(m["team_ids"], dtype=np.int64)[i_idx])
        parts["p"].append(p_idx.astype(np.int64))
        parts["i"].append(i_idx.astype(np.int64))
        parts["target"].append(target.numpy()[p_idx, i_idx])
        parts["y"].append(y.numpy()[p_idx, i_idx].astype(np.float64))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where}: the dump holds duplicate match ids; the join needs unique ones")
    counts = np.array([len(a) for a in parts["p"]], dtype=np.int64)
    cat = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for k, v in parts.items()}
    needed = pd.DataFrame(
        {
            "match_id": np.repeat(np.array(ids, dtype=object), counts),
            "team_id": cat["team"],
            "phase": cat["p"] + M.PHASE_OFFSET,  # cut phase k = slot p + 1
            "target": cat["target"].astype(bool),
            "y": cat["y"],
        }
    )
    merged = needed.merge(tab, on=list(KEY_COLUMNS), how="left", indicator=True, sort=False)
    if len(merged) != len(
        needed
    ):  # the keys of the table are unique, so a left merge is one to one
        raise AssertionError(
            f"{where}: the join changed the number of rows ({len(needed)} -> {len(merged)})"
        )
    gone = (merged["_merge"] == "left_only").to_numpy()
    if gone.any():
        gaps = needed.loc[gone, [*KEY_COLUMNS]].copy()
        gaps["kind"] = np.where(needed.loc[gone, "target"].to_numpy(), "target row", "alive token")
        gaps = gaps.reset_index(drop=True)
        n_target = int((gaps["kind"] == "target row").sum())
        raise ExtraCoverageError(
            f"extra features ({where}): {len(gaps)} of {len(needed)} alive tokens of the cut slots have no row in the table "
            f"({n_target} target rows, {len(gaps) - n_target} other alive tokens)\n"
            + _listing(gaps),
            gaps,
        )
    alignment = _alignment(needed, merged, tab, labels, where)
    values = merged[names].to_numpy(dtype=np.float32, copy=True)
    infinite = np.isinf(values)
    values[infinite] = np.nan
    tokens: dict[str, np.ndarray] = {}
    offset = 0
    for mid, (P, N), p_idx, i_idx, n in zip(ids, shapes, parts["p"], parts["i"], counts):
        tok = np.full((P, N, K), np.nan, dtype=np.float32)
        tok[p_idx, i_idx] = values[offset : offset + n]
        tokens[mid] = tok
        offset += int(n)
    in_dump = tab["match_id"].isin(set(ids)).to_numpy()
    report = {
        "split": split,
        "file": file,
        "n_matches": len(ids),
        "table_rows": int(len(tab)),
        "table_rows_used": int(len(needed)),
        "table_rows_unused": int(len(tab) - len(needed)),
        "table_rows_other_matches": int((~in_dump).sum()),
        "alive_tokens_cut_slots": int(len(needed)),
        "target_rows": int(needed["target"].sum()),
        "last_slot_alive_tokens": int(last_slot),
        "cells": int(values.size),
        "nan_cells": int(np.isnan(values).sum()),
        "inf_cells": int(infinite.sum()),
        "alignment": alignment,
    }
    return ExtraSplit(names, tokens, report)


def load_extra(
    extra_dir: str | Path, cohort: str, dumps: Mapping[str, Mapping], labels: str = D.DEFAULT_LABELS
) -> dict[str, ExtraSplit]:
    """Read and join the tables of every split of ``dumps`` (``{split: dump}``); all splits must carry the same ``kf_*`` set.
    Any gap or label disagreement raises before anything is trained or fitted."""
    check_table_version(
        extra_dir, dumps
    )  # a v1 table next to a v2 dump (or vice versa) is refused before anything is read
    tables: dict[str, tuple[Path, pd.DataFrame]] = {}
    for split in dumps:
        path = table_path(extra_dir, cohort, split)
        tables[split] = (path, read_table(path))
    ref_split, ref_names = None, None
    for split, (_, tab) in tables.items():
        names = [c for c in tab.columns if str(c).startswith(EXTRA_PREFIX)]
        if ref_names is None:
            ref_split, ref_names = split, names
        elif names != ref_names:
            raise ValueError(
                f"{cohort}: the kf_ feature sets of {ref_split} and {split} differ "
                f"(only in {ref_split}: {sorted(set(ref_names) - set(names))}, only in {split}: {sorted(set(names) - set(ref_names))})"
            )
    return {
        split: join_split(dumps[split], tab, labels, split=split, cohort=cohort, file=str(path))
        for split, (path, tab) in tables.items()
    }


# --------------------------------------------------------------------------- standardiser
class ExtraStandardizer:
    """Train-only z-scoring of the extra token features over the covered alive tokens of the cut slots (NaN-aware)."""

    FIELDS = ("mean", "std", "n_train", "constant")

    def __init__(self, **arrays: np.ndarray) -> None:
        self.mean = np.asarray(arrays["mean"], dtype=np.float64)
        self.std = np.asarray(arrays["std"], dtype=np.float64)
        self.n_train = np.asarray(
            arrays["n_train"], dtype=np.float64
        )  # non-missing values behind each column's statistics
        self.constant = np.asarray(
            arrays["constant"], dtype=bool
        )  # zero variance (or no value): std was set to 1

    @classmethod
    def fit(
        cls,
        extra: ExtraSplit,
        dump: Mapping,
        labels: str = D.DEFAULT_LABELS,
        event: bool | None = None,
    ) -> "ExtraStandardizer":
        """Statistics over ``extra`` (the join of ``dump``): the alive tokens of the cut slots, missing cells skipped."""
        event = D.uses_event_alive(dump, labels) if event is None else event
        mom = D._Moments(len(extra.names))
        for m in dump["matches"]:
            alive = D.effective_alive(m, labels, event).numpy()
            mom.update(extra.tokens[str(m["match_id"])][:-1][alive[:-1]])
        mean, std = mom.finalize()
        var = np.divide(mom.m2, mom.n, out=np.zeros_like(mom.m2), where=mom.n > 0)
        return cls(mean=mean, std=std, n_train=mom.n, constant=~(np.sqrt(var) > D.STD_FLOOR))

    def constant_columns(self, names: Sequence[str]) -> list[str]:
        return [n for n, c in zip(names, self.constant) if c]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: np.array(getattr(self, name)) for name in self.FIELDS}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "ExtraStandardizer":
        return cls(**{name: np.asarray(state[name]) for name in cls.FIELDS})

    def transform(self, tokens) -> torch.Tensor:
        """``[..., K]`` raw -> z-scores; missing values (NaN / inf) become 0 after scaling, like the other features."""
        return D.Standardizer._scale(tokens, self.mean, self.std)


def describe(
    extra_dir: str | Path, cohort: str, joined: Mapping[str, ExtraSplit], std: ExtraStandardizer
) -> dict:
    """The ``extra_features`` block of ``metrics.json``: the feature set and its hash, the files, the per-split join counts
    and what the standardiser saw."""
    names = joined["train"].names
    return {
        "dir": str(extra_dir),
        "cohort": cohort,
        "n_features": len(names),
        "names": list(names),
        "names_sha1": names_sha1(names),
        "files": {split: j.report.get("file") for split, j in joined.items()},
        "join": {split: j.report for split, j in joined.items()},
        "standardizer": {
            "constant_train_columns": std.constant_columns(names),
            "n_values_min": int(std.n_train.min()) if std.n_train.size else 0,
            "n_values_max": int(std.n_train.max()) if std.n_train.size else 0,
        },
    }
