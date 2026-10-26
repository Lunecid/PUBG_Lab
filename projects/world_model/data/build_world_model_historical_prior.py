"""Build independent historical evidence from explicitly approved primary fit IDs."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import csv
import sqlite3
import argparse
import time
from pathlib import Path

sys.path.insert(0, str((REPO_ROOT)))
from pubg_lab.world_model.data.cohorts import PRIMARY, classify_match
from pubg_lab.world_model.features.historical_prior import timestamp
from projects.world_model.common.resources import ResourceSampler, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((root / "ablations/full.json").read_text())
    cohort = json.loads((Path(a.cohorts) / "primary_br_manifest.json").read_text())
    ids = set(cohort["train_match_ids"])
    assert ids <= set(plan["train_match_ids"])
    meta = {
        r["match_id"]: r
        for r in csv.DictReader((root / "matches.csv").open())
        if r["match_id"] in ids
    }
    db = sqlite3.connect(out / "historical_prior.sqlite")
    db.executescript("""
      CREATE TABLE donors(match_id TEXT PRIMARY KEY, role TEXT, cohort TEXT, map TEXT, patch TEXT,
         start_ts REAL,end_ts REAL);
      CREATE TABLE contributions(match_id TEXT,map TEXT,sx INTEGER,sy INTEGER,dx INTEGER,dy INTEGER,
         mode INTEGER,n INTEGER,PRIMARY KEY(match_id,sx,sy,dx,dy,mode));
    """)
    excluded = []
    for mid in sorted(ids):
        r = meta[mid]
        assert r["role"] == "train" and classify_match(r) in PRIMARY and r["patch"] == "40.2"
        try:
            start, end = timestamp(r["start_time"]), timestamp(r["end_time"])
            assert end > start
        except (ValueError, AttributeError, AssertionError):
            excluded.append(mid)
            continue
        db.execute(
            "INSERT INTO donors VALUES(?,?,?,?,?,?,?)",
            (mid, r["role"], classify_match(r), r["map"], r["patch"], start, end),
        )
    ids -= set(excluded)
    source = sqlite3.connect(f"file:{root / 'catalog.sqlite'}?mode=ro", uri=True)
    n = 0
    started = time.monotonic()
    with ResourceSampler() as memory:
        for mid, sx, sy, dx, dy, mode, count in source.execute(
            "SELECT match_id,sx,sy,dx,dy,mode,n FROM spatial_support WHERE mode IN (1,2,3) AND match_id IN ("
            + ",".join("?" for _ in ids)
            + ")",
            sorted(ids),
        ):
            db.execute(
                "INSERT INTO contributions VALUES(?,?,?,?,?,?,?,?)",
                (mid, meta[mid]["map"], sx, sy, dx, dy, mode, count),
            )
            n += 1
        db.execute("CREATE INDEX by_map_cell ON contributions(map,sx,sy)")
        db.commit()
        summary = {
            "donor_matches": len(ids),
            "excluded_missing_end": excluded,
            "contribution_rows": n,
            "actual_fit_ids": sorted(ids),
            "source_manifest_sha256": sha(Path(a.cohorts) / "primary_br_manifest.json"),
            "source_catalog_sha256": sha(root / "catalog.sqlite"),
            "target_data_used_for_fit": False,
            "protected_details_read": False,
            "geometry_verified": False,
            "elapsed_s": time.monotonic() - started,
            "max_uss_bytes": memory.max_uss,
        }
        if memory.over_limit or memory.errors:
            raise MemoryError("USS ceiling/measurement")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    db.close()
    source.close()
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("actual_fit_ids", "excluded_missing_end")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
