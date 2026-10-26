"""Cohort discovery from the dump directory (design doc v2.3).

The dump directory is the registry of what may be run: ``<dump-dir>/<cohort>/{train,val,test}.pt``.  The dump
script leaves out airoyale matches, Vikendi and cohorts with fewer than 200 matches, so those cohorts simply
are not there; nothing in the training code hard-codes the 13 pre-registered names any more.

    python -m model_stgcn.ptr.cohorts --dump-dir model_stgcn/dumps/bigcomp2027_phase_v1          # one name per line
    python -m model_stgcn.ptr.cohorts --dump-dir ... --json                                      # + split sizes
    for c in $(python -m model_stgcn.ptr.cohorts --dump-dir ...); do ...; done

A directory counts as a cohort when it holds at least one split file; it is *complete* when it holds all of
``train.pt`` / ``val.pt`` / ``test.pt``.  Incomplete cohorts (a dump in progress or a failed split) are never
listed as runnable: they are reported on stderr (``--strict`` turns that into a non-zero exit, and
``discover_cohorts(strict=True)`` into a ``ValueError``).  Directories without split files (logs, scratch) and
hidden directories are ignored.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pubg_lab.survival.ptr.data import SPLITS


def _cohort_dirs(dump_dir: str | Path):
    root = Path(dump_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dump directory {root} does not exist")
    return [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith((".", "_"))]


def scan_cohorts(dump_dir: str | Path, splits: Sequence[str] = SPLITS) -> dict[str, list[str]]:
    """``{cohort: [missing splits]}`` for every directory that holds at least one split file."""
    out: dict[str, list[str]] = {}
    for d in _cohort_dirs(dump_dir):
        present = [s for s in splits if (d / f"{s}.pt").is_file()]
        if present:
            out[d.name] = [s for s in splits if s not in present]
    return out


def discover_cohorts(
    dump_dir: str | Path, splits: Sequence[str] = SPLITS, *, strict: bool = False
) -> list[str]:
    """Sorted names of the complete cohorts in ``dump_dir``; ``strict`` raises on incomplete ones."""
    scan = scan_cohorts(dump_dir, splits)
    partial = {name: missing for name, missing in scan.items() if missing}
    if partial and strict:
        raise ValueError(f"incomplete cohorts in {dump_dir} (missing split files): {partial}")
    return sorted(name for name, missing in scan.items() if not missing)


def cohort_report(dump_dir: str | Path, splits: Sequence[str] = SPLITS) -> list[dict]:
    """One row per discovered cohort: completeness and, when ``dump_manifest.json`` exists, matches per split."""
    rows = []
    for name, missing in scan_cohorts(dump_dir, splits).items():
        manifest = Path(dump_dir) / name / "dump_manifest.json"
        n_matches = None
        if manifest.is_file():
            try:
                info = json.loads(manifest.read_text(encoding="utf-8"))
                n_matches = {
                    s: info["splits"][s]["n_dumped"] for s in splits if s in info.get("splits", {})
                }
            except (ValueError, KeyError, TypeError):
                n_matches = None
        rows.append(
            {
                "cohort": name,
                "complete": not missing,
                "missing_splits": missing,
                "n_matches": n_matches,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m model_stgcn.ptr.cohorts", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument(
        "--json", action="store_true", help="print a JSON list with completeness and split sizes"
    )
    ap.add_argument(
        "--strict", action="store_true", help="exit 1 if any cohort directory is incomplete"
    )
    args = ap.parse_args(argv)
    report = cohort_report(args.dump_dir)
    partial = [r for r in report if not r["complete"]]
    for r in partial:
        print(
            f"warning: cohort {r['cohort']} is incomplete (missing {r['missing_splits']}); not listed",
            file=sys.stderr,
        )
    complete = [r for r in report if r["complete"]]
    if args.json:
        print(json.dumps(complete, indent=2))
    else:
        for r in complete:
            print(r["cohort"])
    return 1 if (partial and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
