"""The roots of the v2 re-run in ONE place, and the guard against mixing v1 artifacts with a v2 dump.

Roots (BigComp 2027 re-run on the leak-free v2 dump; manifest ``docs/experiment_design/BIGCOMP2027_RERUN_MANIFEST_v2_2026-10-06.md``, D.4)::

    dump               model_stgcn/dumps/bigcomp2027_phase_v2          phase-tier dump (written by ``dump_phase_tier_v2``)
    kf table           model_stgcn/dumps/bigcomp2027_kraftonfeat_v2    persisted KRAFTON feature table (``adaptation.kftable``)
    KRAFTON data       model_stgcn/dumps/krafton_winprob_v2            labels, shards, QA logs (``adaptation.labels`` / ``extract``)
    Sagong cache       model_stgcn/dumps/bigcomp2027_sagong_v2         ``model_stgcn.sagong2025 plan / extract``
    runs               model_stgcn/checkpoints/bigcomp2027_ptr_v2      every arm: ``<cohort>/<arm>/seed<s>/``  (+ ``_lc``, ``_lc_steps``)
    results            reports/bigcomp2027_results_v2                  aggregator, supplement, ecological model, logs, paper numbers
    KRAFTON reports    reports/krafton_baseline_v2                     ``h0_report`` / ``k3_contrasts`` / ``phase_auc`` ...

Every entry point of the re-run takes these as explicit options; the constants below are only their defaults.  Nothing else in the
code base spells a ``_v2`` path (``tools/rerun_v2.py`` builds all of them from :class:`RerunPaths`).

Version guard.  A *dump* carries its version in ``meta['version']`` (``dump_manifest.json`` repeats it per cohort); the legacy dump has
no such key and counts as ``v1``.  Every *artifact root* (runs, kf table, KRAFTON data, Sagong cache, results) is a function of one dump,
and reading or writing it next to another dump is the silent failure that this module refuses:

* :func:`dump_version`            ``v1`` / ``v2`` of a dump directory (cohorts that disagree raise :class:`VersionMixError`);
* :func:`artifact_version`        the version of a root, with the evidence: an explicit stamp (``.rerun_stamp.json``, written by
  :func:`claim_root`), a ``meta.json`` / ``summary.json`` that names its origin, or the ``_v<N>`` token of the directory name;
* :func:`check_same_version`      raises :class:`VersionMixError` when a root's version differs from the dump's (a root whose version
  cannot be told is fine when it is empty or does not exist yet: it is claimed by the first writer);
* :func:`claim_root`              writes the stamp of a root the caller is about to write into; with a v2 dump it refuses a non-empty
  root that carries no proof of being v2 (a v1 root that was renamed, copied or left over).

``dump_meta_sha1`` is the "dump meta hash" of the run manifest (population + per-cohort manifests + split file sizes, optionally the
``meta`` of every validation split).  ``validate_v2_dump`` is the gate's meta check.

    python -m model_stgcn.ptr.paths roots                       # the v2 roots as JSON
    python -m model_stgcn.ptr.paths version DUMP_DIR
    python -m model_stgcn.ptr.paths validate DUMP_DIR           # exit 1 on errors
    python -m model_stgcn.ptr.paths stamp ROOT --dump-dir DUMP_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "v2"
LEGACY_VERSION = "v1"  # a dump without ``meta['version']``
SPLITS: tuple[str, ...] = ("train", "val", "test")
STAMP_NAME = ".rerun_stamp.json"

DUMPS = REPO_ROOT / "model_stgcn" / "dumps"
CHECKPOINTS = REPO_ROOT / "model_stgcn" / "checkpoints"
REPORTS = REPO_ROOT / "reports"

DUMP_DIR = DUMPS / "bigcomp2027_phase_v2"
KF_DIR = DUMPS / "bigcomp2027_kraftonfeat_v2"
KRAFTON_DATA_DIR = DUMPS / "krafton_winprob_v2"
SAGONG_DIR = DUMPS / "bigcomp2027_sagong_v2"
RUNS_DIR = CHECKPOINTS / "bigcomp2027_ptr_v2"
RUNS_LC_DIR = CHECKPOINTS / "bigcomp2027_ptr_v2_lc"
RUNS_LC_STEPS_DIR = CHECKPOINTS / "bigcomp2027_ptr_v2_lc_steps"
RESULTS_DIR = REPORTS / "bigcomp2027_results_v2"
KRAFTON_REPORTS_DIR = REPORTS / "krafton_baseline_v2"
PAPER_DIR = REPO_ROOT / "paper" / "bigcomp2027"

# the record keys a v2 dump match must carry / must not carry (``dump_phase_tier_v2`` conventions)
V2_REQUIRED_MATCH_KEYS: tuple[str, ...] = (
    "match_id", "team_ids", "X", "node_mask", "query_mask", "target_mask_v2", "y_hazard_v2", "team_death_s", "phase_start_s",
    "phase_end_s", "cut_t", "cut_valid", "cut_mask",
)  # fmt: skip  (dump_phase_tier_v2 CONTRACT.md, section 1)
V2_FORBIDDEN_MATCH_KEYS: tuple[str, ...] = ("node_mask_v2",)  # its presence switches the loader to the v1 final-death mask (leak L3)
V2_N_PHASES = 8
V2_EXPECTED_FEATURES = 150
V2_CUT_LEAD_S = 10.0


class VersionMixError(RuntimeError):
    """A v1 artifact next to a v2 dump (or the reverse), or a root that cannot be proven to belong to the dump."""


# =========================================================================== the roots
@dataclass(frozen=True)
class RerunPaths:
    """Every root of one re-run.  ``RerunPaths()`` is the v2 default; :meth:`under` gives the same layout below another directory
    (tests, dry runs); individual roots are replaced with :meth:`replace`."""

    dump: Path = DUMP_DIR
    kf: Path = KF_DIR
    krafton_data: Path = KRAFTON_DATA_DIR
    sagong: Path = SAGONG_DIR
    runs: Path = RUNS_DIR
    runs_lc: Path = RUNS_LC_DIR
    runs_lc_steps: Path = RUNS_LC_STEPS_DIR
    results: Path = RESULTS_DIR
    krafton_reports: Path = KRAFTON_REPORTS_DIR
    paper: Path = PAPER_DIR

    def __post_init__(self) -> None:
        for f in fields(self):
            object.__setattr__(self, f.name, Path(getattr(self, f.name)))

    @classmethod
    def under(cls, base: str | Path, version: str = VERSION) -> "RerunPaths":
        b = Path(base)
        return cls(
            dump=b / "dumps" / f"bigcomp2027_phase_{version}", kf=b / "dumps" / f"bigcomp2027_kraftonfeat_{version}",
            krafton_data=b / "dumps" / f"krafton_winprob_{version}", sagong=b / "dumps" / f"bigcomp2027_sagong_{version}",
            runs=b / "checkpoints" / f"bigcomp2027_ptr_{version}", runs_lc=b / "checkpoints" / f"bigcomp2027_ptr_{version}_lc",
            runs_lc_steps=b / "checkpoints" / f"bigcomp2027_ptr_{version}_lc_steps", results=b / "reports" / f"bigcomp2027_results_{version}",
            krafton_reports=b / "reports" / f"krafton_baseline_{version}", paper=b / "paper" / "bigcomp2027",
        )  # fmt: skip

    def replace(self, **kw) -> "RerunPaths":
        return replace(self, **kw)

    # ---- derived locations (always below a root above)
    @property
    def ecological(self) -> Path:
        return self.results / "ecological_model"

    @property
    def supplement(self) -> Path:
        return self.results / "supplement"

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    @property
    def run_manifest(self) -> Path:
        return self.results / "run_manifest.json"

    @property
    def run_state(self) -> Path:
        return self.results / "run_state.json"

    @property
    def paper_numbers(self) -> Path:
        return self.results / "paper_numbers.json"

    @property
    def paper_tables(self) -> Path:
        return self.paper / "tables"

    def artifact_roots(self) -> dict[str, Path]:
        """The roots that are a function of the dump (everything but the dump itself and the paper directory)."""
        return {
            "kf_table": self.kf, "krafton_data": self.krafton_data, "sagong_cache": self.sagong, "runs": self.runs,
            "runs_lc": self.runs_lc, "runs_lc_steps": self.runs_lc_steps, "results": self.results,
            "krafton_reports": self.krafton_reports,
        }  # fmt: skip

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}


# =========================================================================== reading versions
_TOKEN = re.compile(r"(?:^|[_\-.])(v\d+)(?=$|[_\-.])")


def name_version(path: str | Path) -> str | None:
    """``v<N>`` token of the LAST path component (``bigcomp2027_ptr_v1_lc`` -> ``v1``); None if there is none or it is ambiguous."""
    found = set(_TOKEN.findall(Path(str(path)).name))
    return next(iter(found)) if len(found) == 1 else None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _manifest_version(manifest) -> str | None:
    if not isinstance(manifest, dict):
        return None
    v = manifest.get("version")
    return str(v) if v else None


def dump_version(dump_dir: str | Path, *, deep: bool = False) -> str:
    """``v2`` when the dump says so (``dump_manifest.json['version']`` of every cohort, and with ``deep`` ``meta['version']`` of the
    validation split of every cohort), ``v1`` for a dump that names no version (the legacy format), ``VersionMixError`` if the cohorts
    disagree.  A directory that does not exist (yet) takes the ``_v<N>`` token of its name, else ``unknown``."""
    root = Path(dump_dir)
    if not root.is_dir():
        return name_version(root) or "unknown"
    seen: dict[str, str] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if not any((d / f"{s}.pt").is_file() for s in SPLITS) and not (d / "dump_manifest.json").is_file():
            continue
        seen[d.name] = _manifest_version(_read_json(d / "dump_manifest.json")) or LEGACY_VERSION
        if deep and (d / "val.pt").is_file():
            meta = _load_meta(d / "val.pt")
            from_meta = str(meta.get("version")) if meta.get("version") else LEGACY_VERSION
            if from_meta != seen[d.name]:
                raise VersionMixError(f"{d}: dump_manifest.json says {seen[d.name]!r}, meta['version'] of val.pt says {from_meta!r}")
    if not seen:
        return name_version(root) or "unknown"
    versions = sorted(set(seen.values()))
    if len(versions) > 1:
        by = {v: sorted(c for c, x in seen.items() if x == v)[:4] for v in versions}
        raise VersionMixError(f"dump directory {root} mixes dump versions {versions} across cohorts: {by}")
    return versions[0]


def _load_meta(path: Path) -> dict:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    return dict(d.get("meta") or {})


# files inside the cohort directories of an artifact root that name the dump version they were made from (written by the producers of the v2 pipeline):
# the Sagong plan / habitat audits (``version`` / ``dump_version``), the run metrics (``dump_version``, also under ``data``)
_CONTENT_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (("plan_audit.json", ("version", "dump_version")), ("hab44_audit.json", ("dump_version", "version")))


def _content_version(root: Path) -> tuple[str | None, str]:
    """The dump version that the CONTENT of a root records (the first few cohort directories: audits and run metrics), else ``(None, 'none')``."""
    try:
        cohorts = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_")) and d.name != "logs")[:3]
    except OSError:
        return None, "none"
    for d in cohorts:
        for fname, keys in _CONTENT_FILES:
            doc = _read_json(d / fname)
            if isinstance(doc, dict):
                for k in keys:
                    if doc.get(k):
                        return str(doc[k]), f"{d.name}/{fname}:{k}"
        try:
            metrics = sorted(d.glob("*/seed*/metrics.json"))[:3]
        except OSError:
            metrics = []
        for m in metrics:
            doc = _read_json(m)
            if isinstance(doc, dict):
                data = doc.get("data")
                v = doc.get("dump_version") or (data.get("dump_version") if isinstance(data, dict) else None)
                if v:
                    return str(v), f"{d.name}/{m.parent.parent.name}/{m.parent.name}/metrics.json:dump_version"
    return None, "none"


def artifact_version(root: str | Path) -> tuple[str | None, str]:
    """``(version, evidence)`` of an artifact root; ``(None, 'none')`` if nothing tells.  Evidence, strongest first: ``stamp``
    (``.rerun_stamp.json``), ``meta.dump_version`` / ``meta.version`` / ``meta.name`` (the root's ``meta.json``: the kf table), the content of the cohort
    directories (audits and run metrics that record ``dump_version``), ``summary.runs_dir`` (an aggregator output), ``name`` (the ``_v<N>`` token of the
    directory name)."""
    p = Path(root)
    stamp = _read_json(p / STAMP_NAME)
    if isinstance(stamp, dict) and stamp.get("version"):
        return str(stamp["version"]), "stamp"
    meta = _read_json(p / "meta.json")
    if isinstance(meta, dict):
        if meta.get("dump_version"):
            return str(meta["dump_version"]), "meta.dump_version"
        if meta.get("version"):
            return str(meta["version"]), "meta.version"
        v = name_version(str(meta.get("name", "")))
        if v:
            return v, "meta.name"
    cv, cwhy = _content_version(p)
    if cv:
        return cv, cwhy
    summary = _read_json(p / "summary.json")
    if isinstance(summary, dict) and summary.get("runs_dir"):
        v = name_version(str(summary["runs_dir"]))
        if v:
            return v, "summary.runs_dir"
    v = name_version(p)
    return (v, "name") if v else (None, "none")


def _non_empty(root: Path) -> bool:
    try:
        return root.is_dir() and any(not c.name.startswith(".") for c in root.iterdir())
    except OSError:
        return False


def check_same_version(dump_dir: str | Path, roots: Mapping[str, str | Path | None], *, deep: bool = False) -> str:
    """Refuse a v1 artifact root next to a v2 dump (or the reverse): returns the dump's version, raises :class:`VersionMixError`.

    ``roots`` maps a role (``runs``, ``kf_table``, ...) to a path; ``None`` entries are skipped.  A root whose version cannot be told
    passes only when it is empty or absent, or when the dump is the legacy v1 (nothing to protect there)."""
    dv = dump_version(dump_dir, deep=deep)
    if dv == "unknown":
        return dv
    bad, unproven = [], []
    for role, path in roots.items():
        if path is None:
            continue
        v, why = artifact_version(path)
        if v is not None and v != dv:
            bad.append(f"{role} {path} is {v} ({why})")
        elif v is None and dv == VERSION and _non_empty(Path(path)):
            unproven.append(f"{role} {path}")
    if bad:
        raise VersionMixError(
            f"refusing to mix versions: the dump {dump_dir} is {dv} but " + "; ".join(bad)
            + ".  Every artifact of a re-run must come from the same dump (use the v2 roots of model_stgcn.ptr.paths)."
        )
    if unproven:
        raise VersionMixError(
            f"the dump {dump_dir} is {dv} but {', '.join(unproven)} hold(s) files and carry no version evidence (no {STAMP_NAME}, no meta, "
            f"no _v<N> in the name): it could be a v1 root that was renamed or copied.  Stamp it only if you are sure it belongs to this dump: "
            f"python -m model_stgcn.ptr.paths stamp <root> --dump-dir {dump_dir}"
        )
    return dv


def kf_dir_for(dump_dir: str | Path) -> Path:
    """The kf-table root that belongs to a dump: ``bigcomp2027_kraftonfeat_<version>`` NEXT TO the dump directory (the v2 dump
    ``.../bigcomp2027_phase_v2`` gives ``.../bigcomp2027_kraftonfeat_v2``; a legacy dump gives ``_v1``)."""
    v = dump_version(dump_dir)
    return Path(dump_dir).parent / f"bigcomp2027_kraftonfeat_{VERSION if v == 'unknown' else v}"


# =========================================================================== stamps
def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_stamp(root: str | Path, *, version: str, dump_dir: str | Path | None = None, dump_meta_sha1: str | None = None, role: str = "") -> Path:
    """(Re)write ``<root>/.rerun_stamp.json`` atomically."""
    r = Path(root)
    r.mkdir(parents=True, exist_ok=True)
    body = {"version": version, "role": role, "dump_dir": str(dump_dir) if dump_dir is not None else None, "dump_meta_sha1": dump_meta_sha1,
            "created_utc": _utc()}  # fmt: skip
    tmp = r / (STAMP_NAME + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, r / STAMP_NAME)
    return r / STAMP_NAME


def claim_root(root: str | Path, dump_dir: str | Path, *, role: str = "", dump_meta: bool = True) -> str:
    """Stamp a root the caller is about to write into with the version of the dump (v2 only: a legacy v1 dump is never stamped, its roots
    stay byte-identical, and a v2 run later refuses them as unproven).  Raises :class:`VersionMixError` when the root already belongs to
    another version, or (v2 dump) holds files but cannot be proven to be v2.  Returns the dump's version.

    An existing stamp of the same version is kept (its ``created_utc`` stays); a changed dump meta hash is NOT an error here, because
    every run carries a config hash over the dump files and goes stale by itself, but it is reported by :func:`stamp_status`."""
    dv = dump_version(dump_dir)
    if dv == "unknown":
        raise VersionMixError(f"cannot tell the version of the dump {dump_dir}: it does not exist and its name has no _v<N> token")
    check_same_version(dump_dir, {role or "root": root})
    if dv != VERSION:
        return dv
    existing, why = artifact_version(root)
    if existing == dv and why == "stamp":
        return dv
    write_stamp(root, version=dv, dump_dir=dump_dir, dump_meta_sha1=dump_meta_sha1(dump_dir) if dump_meta and Path(dump_dir).is_dir() else None, role=role)
    return dv


def stamp_status(root: str | Path, dump_dir: str | Path) -> dict:
    """``{'stamp': dict | None, 'dump_meta_changed': bool | None}`` (None: no stamp / no recorded hash)."""
    stamp = _read_json(Path(root) / STAMP_NAME)
    if not isinstance(stamp, dict):
        return {"stamp": None, "dump_meta_changed": None}
    recorded = stamp.get("dump_meta_sha1")
    now = dump_meta_sha1(dump_dir) if Path(dump_dir).is_dir() else None
    return {"stamp": stamp, "dump_meta_changed": None if not recorded or not now else recorded != now}


# =========================================================================== the dump meta hash and the gate's validation
def _complete_cohorts(root: Path) -> list[Path]:
    return [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith((".", "_")) and all((d / f"{s}.pt").is_file() for s in SPLITS)]


def dump_meta_sha1(dump_dir: str | Path, *, deep: bool = False) -> str:
    """The dump meta hash: SHA-1 over ``population.json``, every cohort's ``dump_manifest.json`` and the byte sizes of its split files
    (and, with ``deep``, the canonical JSON of the ``meta`` of every validation split).  Cheap without ``deep`` (no tensor is read)."""
    root = Path(dump_dir)
    h = hashlib.sha1()
    pop = root / "population.json"
    h.update(b"population\0" + (pop.read_bytes() if pop.is_file() else b"") + b"\0")
    for d in _complete_cohorts(root):
        man = d / "dump_manifest.json"
        h.update(d.name.encode() + b"\0" + (man.read_bytes() if man.is_file() else b"") + b"\0")
        for s in SPLITS:
            h.update(f"{s}:{(d / f'{s}.pt').stat().st_size}".encode() + b"\0")
        if deep:
            from model_stgcn.ptr.util import to_jsonable

            h.update(json.dumps(to_jsonable(_load_meta(d / "val.pt")), sort_keys=True, default=str).encode() + b"\0")
    return h.hexdigest()


PREFIX_REPLAY_FILE = "prefix_replay.json"  # written next to population.json by the dump builder's replay test: n_matches, n_cuts, mismatches, min gap


def validate_v2_dump(dump_dir: str | Path, *, deep: bool = True, max_matches: int = 25, require_prefix_replay: bool = False) -> dict:
    """The gate's check that a dump is a valid v2 dump: ``{'errors': [...], 'warnings': [...], 'cohorts': [...], 'dump_meta_sha1': ...}``.

    Errors: no dump / no ``population.json`` / no complete cohort; a version other than ``v2`` in a manifest or a ``meta``; cohorts of the
    population that are missing (or extra); matches that failed to dump; ``n_phases`` other than 8; ``feature_names`` of a length that
    differs from ``n_features``; records without the v2 keys or WITH ``node_mask_v2`` (leak L3 marker); and, for the first ``max_matches``
    matches of every validation split, the causal invariants ``cut_t = phase_end_s - 10``, ``node_mask <= query_mask``,
    ``target_mask_v2 <= query_mask``, no target in the final slot and ``team_death_s = +inf`` for winners.  Warnings: a feature count other
    than 150, a population count that differs from the dumped matches.

    ``prefix_replay.json`` (the dump builder's no-look-ahead replay: the inputs rebuilt from a source that the DATABASE truncated at a_k): when present its
    ``mismatches`` must be empty (an error otherwise) and its summary is returned under ``prefix_replay``; with ``require_prefix_replay`` (the gate) a missing
    file is an error too.  The mutation check (a builder that forgets to cut the event tables must be caught) is a warning if it caught fewer cuts than it compared."""
    root = Path(dump_dir)
    errors: list[str] = []
    warnings: list[str] = []
    out: dict = {"dump_dir": str(root), "errors": errors, "warnings": warnings, "cohorts": [], "dump_meta_sha1": None}
    if not root.is_dir():
        errors.append(f"dump directory {root} does not exist")
        return out
    cohorts = _complete_cohorts(root)
    out["cohorts"] = [d.name for d in cohorts]
    partial = {d.name: [s for s in SPLITS if not (d / f"{s}.pt").is_file()] for d in sorted(root.iterdir())
               if d.is_dir() and not d.name.startswith((".", "_")) and any((d / f"{s}.pt").is_file() for s in SPLITS) and not all((d / f"{s}.pt").is_file() for s in SPLITS)}  # fmt: skip
    if partial:  # `model_stgcn.ptr.cohorts --strict`: a cohort with a missing split is a dump in progress or a failed split
        errors.append(f"incomplete cohorts (missing split files): {partial}")
    if not cohorts:
        errors.append(f"no complete cohort (train/val/test .pt) in {root}")
        return out
    pop_path = root / "population.json"
    pop = _read_json(pop_path)
    if not isinstance(pop, dict):
        errors.append("population.json is missing or unreadable")
        kept: dict = {}
    else:
        kept = {c: i for c, i in (pop.get("cohorts") or {}).items() if i.get("status") == "kept"}
        missing = sorted(set(kept) - set(out["cohorts"]))
        extra = sorted(set(out["cohorts"]) - set(kept))
        if missing:
            errors.append(f"cohorts kept by population.json but not (completely) dumped: {missing}")
        if extra:
            errors.append(f"cohorts dumped but not kept by population.json: {extra}")
    for d in cohorts:
        man = _read_json(d / "dump_manifest.json")
        if not isinstance(man, dict):
            errors.append(f"{d.name}: dump_manifest.json is missing or unreadable")
            continue
        if _manifest_version(man) != VERSION:
            errors.append(f"{d.name}: dump_manifest.json version is {_manifest_version(man)!r}, expected {VERSION!r}")
        for s in SPLITS:
            sp = (man.get("splits") or {}).get(s) or {}
            if int(sp.get("n_skipped") or 0) > 0:
                errors.append(f"{d.name}/{s}: {sp['n_skipped']} matches could not be dumped (n_skipped)")
            if d.name in kept and int(sp.get("n_dumped", -1)) != int(kept[d.name]["after"][s]):
                warnings.append(f"{d.name}/{s}: population.json keeps {kept[d.name]['after'][s]} matches, the manifest dumped {sp.get('n_dumped')}")
        if deep:
            _validate_cohort_meta(d, errors, warnings, max_matches)
    _check_prefix_replay(root, out, errors, warnings, require_prefix_replay)
    try:
        out["dump_meta_sha1"] = dump_meta_sha1(root, deep=deep)
    except Exception as exc:  # unreadable meta: already an error above
        warnings.append(f"dump meta hash not computed: {exc}")
    return out


def _check_prefix_replay(root: Path, out: dict, errors: list[str], warnings: list[str], required: bool) -> None:
    path = root / PREFIX_REPLAY_FILE
    rep = _read_json(path) if path.is_file() else None
    out["prefix_replay"] = None
    if not isinstance(rep, dict):
        if required:
            errors.append(f"{PREFIX_REPLAY_FILE} is missing or unreadable: the no-look-ahead replay of the dump builder was not recorded")
        return
    mism = rep.get("mismatches")
    n_mism = len(mism) if isinstance(mism, (list, tuple, dict)) else (None if mism is None else int(mism))
    gap = rep.get("min_gap_between_last_kept_record_and_cut_s")
    mut = rep.get("mutation_check") or {}
    out["prefix_replay"] = {"n_matches": rep.get("n_matches"), "n_cuts": rep.get("n_cuts"), "n_mismatches": n_mism, "min_gap_s": gap, "mutation_check": mut}
    if n_mism is None:
        errors.append(f"{PREFIX_REPLAY_FILE} has no 'mismatches'")
    elif n_mism != 0:
        errors.append(f"{PREFIX_REPLAY_FILE}: {n_mism} cut(s) differ between the full and the database-truncated source (a look-ahead leak)")
    if gap is not None and not float(gap) > 0:
        errors.append(f"{PREFIX_REPLAY_FILE}: min gap between the last kept record and the cut is {gap} s (a record at or after the cut was read)")
    if mut and int(mut.get("cuts_detected_as_leaky", 0)) < int(mut.get("cuts_compared", 0)):
        warnings.append(f"{PREFIX_REPLAY_FILE}: the mutation check caught {mut.get('cuts_detected_as_leaky')} of {mut.get('cuts_compared')} cuts: the replay may not detect every leak")


def _validate_cohort_meta(d: Path, errors: list[str], warnings: list[str], max_matches: int) -> None:
    """The ``meta`` and the first ``max_matches`` records of the validation split of one cohort (train / test share the meta by
    construction, and the loader checks that at run time)."""
    import numpy as np
    import torch

    dump = torch.load(d / "val.pt", map_location="cpu", weights_only=False)
    meta = dict(dump.get("meta") or {})
    tag = f"{d.name}/val"
    if meta.get("version") != VERSION:
        errors.append(f"{tag}: meta['version'] is {meta.get('version')!r}, expected {VERSION!r}")
    if int(meta.get("n_phases", -1)) != V2_N_PHASES:
        errors.append(f"{tag}: meta['n_phases'] is {meta.get('n_phases')!r}, expected {V2_N_PHASES}")
    fn = list(meta.get("feature_names") or [])
    if len(fn) != int(meta.get("n_features", len(fn))):
        errors.append(f"{tag}: {len(fn)} feature_names but meta['n_features'] = {meta.get('n_features')}")
    if len(fn) != V2_EXPECTED_FEATURES:
        warnings.append(f"{tag}: {len(fn)} features (the paper plan says {V2_EXPECTED_FEATURES})")
    if not meta.get("single_clock", True):
        errors.append(f"{tag}: meta['single_clock'] is false")
    for m in (dump.get("matches") or [])[:max_matches]:
        mid = m.get("match_id", "?")
        miss = [k for k in V2_REQUIRED_MATCH_KEYS if k not in m]
        if miss:
            errors.append(f"{tag} match {mid}: missing keys {miss}")
            break
        bad = [k for k in V2_FORBIDDEN_MATCH_KEYS if k in m]
        if bad:
            errors.append(f"{tag} match {mid}: carries {bad}: the loader would rebuild the v1 final-death alive mask (leak L3)")
            break
        node, query, target = (np.asarray(m[k]).astype(bool) for k in ("node_mask", "query_mask", "target_mask_v2"))
        if (node & ~query).any():
            errors.append(f"{tag} match {mid}: node_mask is not inside query_mask")
        if (target & ~query).any():
            errors.append(f"{tag} match {mid}: target_mask_v2 is not inside query_mask")
        end, cut = np.asarray(m["phase_end_s"], dtype=np.float64), np.asarray(m["cut_t"], dtype=np.float64)
        ok = np.isfinite(end) & np.isfinite(cut)
        if ok.any() and not np.allclose(cut[ok], end[ok] - V2_CUT_LEAD_S, atol=1e-6, rtol=0):
            errors.append(f"{tag} match {mid}: cut_t differs from phase_end_s - {V2_CUT_LEAD_S:g}")
        if target[-1].any():
            errors.append(f"{tag} match {mid}: a risk-set row in the final phase slot")
        valid = np.asarray(m["cut_valid"]).astype(bool)
        if (target & ~valid[:, None]).any():
            errors.append(f"{tag} match {mid}: risk-set rows at a cut that is not valid (excluded or without b_k)")
        if "team_won" in m:
            won = np.asarray(m["team_won"]).astype(bool)
            death = np.asarray(m["team_death_s"], dtype=np.float64)
            if won.any() and not np.isposinf(death[won]).all():
                errors.append(f"{tag} match {mid}: a winner has a finite team_death_s (winners must be censored at +inf)")


# =========================================================================== command line
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m model_stgcn.ptr.paths", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("roots", help="print the v2 roots as JSON")
    p = sub.add_parser("version", help="print the version of a dump directory")
    p.add_argument("dump_dir")
    p = sub.add_parser("validate", help="the gate's meta check of a v2 dump (exit 1 on errors)")
    p.add_argument("dump_dir")
    p.add_argument("--shallow", action="store_true", help="manifests only, no tensor is read")
    p.add_argument("--require-prefix-replay", action="store_true", help=f"an error if {PREFIX_REPLAY_FILE} is missing (it is checked when present)")
    p = sub.add_parser("stamp", help="stamp an artifact root with the version of a dump")
    p.add_argument("root")
    p.add_argument("--dump-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "roots":
        print(json.dumps(RerunPaths().as_dict(), indent=2))
    elif a.cmd == "version":
        print(dump_version(a.dump_dir))
    elif a.cmd == "validate":
        res = validate_v2_dump(a.dump_dir, deep=not a.shallow, require_prefix_replay=a.require_prefix_replay)
        print(json.dumps(res, indent=2))
        return 1 if res["errors"] else 0
    elif a.cmd == "stamp":
        dv = dump_version(a.dump_dir)
        write_stamp(a.root, version=dv, dump_dir=a.dump_dir, dump_meta_sha1=dump_meta_sha1(a.dump_dir), role="manual")
        print(f"stamped {a.root} as {dv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
