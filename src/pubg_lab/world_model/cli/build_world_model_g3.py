"""G3 read-only full allowed corpus classification and rare-event index."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import csv
import json
import sqlite3
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import psutil
from pubg_lab.world_model.cohorts import classify_match, movement_events, candidate_windows, PRIMARY
from pubg_lab.world_model.cli.verify_world_model_encoder import ResourceSampler, sha


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((root / "ablations/full.json").read_text())
    role_ids = {r: set(plan[r + "_match_ids"]) for r in ("train", "validation")}
    assert not role_ids["train"] & role_ids["validation"]
    allowed = role_ids["train"] | role_ids["validation"]
    metadata = {
        r["match_id"]: r
        for r in csv.DictReader((root / "matches.csv").open())
        if r["match_id"] in allowed
    }
    assert set(metadata) == allowed
    assert all(metadata[mid]["role"] == role for role, ids in role_ids.items() for mid in ids)
    competitors = [
        p.info["name"]
        for p in psutil.process_iter(["name"])
        if any(k in (p.info["name"] or "").lower() for k in ("strata", "llama"))
    ]
    assert not competitors, competitors
    db = sqlite3.connect(out / "cohort_events.sqlite")
    db.executescript("""
    CREATE TABLE matches(match_id TEXT PRIMARY KEY, role TEXT, cohort TEXT, map TEXT, patch TEXT,
      match_type TEXT, game_mode TEXT, created_at TEXT, end_time TEXT, bank_sha256 TEXT, meta_sha256 TEXT);
    CREATE TABLE events(event_id TEXT PRIMARY KEY, match_id TEXT, actor INTEGER, team INTEGER,
      kind TEXT, start_s REAL, end_s REAL, evidence TEXT, source_time REAL,event_episode INTEGER,
      supported30 INTEGER, supported60 INTEGER, coarse_supported30 INTEGER);
    CREATE TABLE candidates(event_id TEXT, frame INTEGER, stage TEXT, ego_points INTEGER,
      ego_first30 INTEGER, ego_full60 INTEGER, joint_full60 INTEGER, focal_members INTEGER,
      team_points INTEGER, rare_mode_points INTEGER, PRIMARY KEY(event_id,frame));
    CREATE TABLE selected(match_id TEXT, frame INTEGER, actor INTEGER, role TEXT, cohort TEXT,
      PRIMARY KEY(match_id,frame,actor));
    CREATE TABLE selected_events(match_id TEXT, frame INTEGER, actor INTEGER, event_id TEXT,
      PRIMARY KEY(match_id,frame,actor,event_id));
    CREATE INDEX events_by_match ON events(match_id);
    CREATE INDEX events_by_kind ON events(kind);
    """)
    start = time.monotonic()
    manifests = defaultdict(lambda: {"train_match_ids": [], "validation_match_ids": []})
    counts = Counter()
    with ResourceSampler() as memory:
        for index, mid in enumerate(sorted(allowed)):
            row = metadata[mid]
            role = row["role"]
            cohort = classify_match(row)
            manifests[cohort][role + "_match_ids"].append(mid)
            if cohort in PRIMARY:
                manifests["primary_br"][role + "_match_ids"].append(mid)
            bp, mp = root / "banks" / f"{mid}.npz", root / "meta" / f"{mid}.json"
            meta = json.loads(mp.read_text())
            with np.load(bp, allow_pickle=False) as f:
                bank = dict(f)
            assert sha(bp) == meta["bank_sha256"]
            db.execute(
                "INSERT INTO matches VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    role,
                    cohort,
                    row["map"],
                    row["patch"],
                    row["match_type"],
                    row["game_mode"],
                    row["created_at"],
                    row["end_time"],
                    meta["bank_sha256"],
                    sha(mp),
                ),
            )
            coarse = set(map(tuple, bank["anchors"].tolist()))
            for number, event in enumerate(movement_events(bank, meta)):
                eid = hashlib.sha256(
                    f"{mid}:{number}:{event['actor']}:{event['start_s']}:{event['kind']}".encode()
                ).hexdigest()[:24]
                candidates = list(candidate_windows(bank, event))
                support = [c for c in candidates if c["stage"] != "pre" and c["ego_first30"]]
                s60 = any(c["stage"] != "pre" and c["ego_full60"] for c in candidates)
                coarse30 = any((c["frame"], event["actor"]) in coarse for c in support)
                db.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        eid,
                        mid,
                        event["actor"],
                        event["team"],
                        event["kind"],
                        event["start_s"],
                        event["end_s"],
                        event["evidence"],
                        event["source_time"],
                        event["event_episode"],
                        int(bool(support)),
                        int(s60),
                        int(coarse30),
                    ),
                )
                db.executemany(
                    "INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            eid,
                            c["frame"],
                            c["stage"],
                            c["ego_points"],
                            int(c["ego_first30"]),
                            int(c["ego_full60"]),
                            int(c["joint_full60"]),
                            c["focal_members"],
                            c["team_points"],
                            c["rare_mode_points"],
                        )
                        for c in candidates
                    ],
                )
                if support:
                    c = min(support, key=lambda c: c["frame"])
                    key = (mid, c["frame"], event["actor"])
                    db.execute(
                        "INSERT OR IGNORE INTO selected VALUES(?,?,?,?,?)", (*key, role, cohort)
                    )
                    db.execute("INSERT INTO selected_events VALUES(?,?,?,?)", (*key, eid))
                counts["events"] += 1
                counts["candidates"] += len(candidates)
            if (index + 1) % 250 == 0:
                db.commit()
                print(
                    json.dumps(
                        {
                            "matches": index + 1,
                            **counts,
                            "elapsed_s": round(time.monotonic() - start),
                        }
                    ),
                    flush=True,
                )
            if memory.over_limit or memory.errors:
                raise RuntimeError("USS measurement/ceiling")
        db.commit()
        tables = {}
        for name, query in {
            "cohorts": "SELECT role,cohort,COUNT(*) AS matches FROM matches GROUP BY role,cohort",
            "rare_support": """SELECT m.role,m.cohort,e.kind,COUNT(*) AS actor_events,
                COUNT(DISTINCT m.match_id) AS matches,SUM(supported30) AS supported30_events,
                COUNT(DISTINCT CASE WHEN supported30 THEN m.match_id END) AS supported30_matches,
                SUM(supported60) AS supported60_events,
                SUM(supported30 AND NOT coarse_supported30) AS gained30_events
                FROM events e JOIN matches m USING(match_id) GROUP BY m.role,m.cohort,e.kind""",
            "selected_counts": "SELECT role,cohort,COUNT(*) AS samples,COUNT(DISTINCT match_id) AS matches FROM selected GROUP BY role,cohort",
        }.items():
            cur = db.execute(query)
            columns = [c[0] for c in cur.description]
            tables[name] = [dict(zip(columns, r)) for r in cur]
            with (out / f"{name}.csv").open("w") as h:
                w = csv.DictWriter(h, fieldnames=columns)
                w.writeheader()
                w.writerows(tables[name])
        for cohort, manifest in manifests.items():
            (out / f"{cohort}_manifest.json").write_text(
                json.dumps(
                    {
                        **manifest,
                        "source_plan_sha256": sha(root / "ablations/full.json"),
                        "cohort": cohort,
                        "fit_rule": "Only this manifest's train IDs may fit priors/vocabulary/pretraining",
                    },
                    indent=2,
                )
            )
        cur = db.execute(
            "SELECT match_id,frame,actor,role,cohort FROM selected ORDER BY role,match_id,frame,actor"
        )
        with (out / "rare_samples.jsonl").open("w") as h:
            for mid, frame, actor, role, cohort in cur:
                eids = [
                    r[0]
                    for r in db.execute(
                        "SELECT event_id FROM selected_events WHERE match_id=? AND frame=? AND actor=? ORDER BY event_id",
                        (mid, frame, actor),
                    )
                ]
                h.write(
                    json.dumps(
                        dict(
                            match_id=mid,
                            frame=frame,
                            actor=actor,
                            role=role,
                            cohort=cohort,
                            event_ids=eids,
                            selection_uses_future_labels=True,
                        )
                    )
                    + "\n"
                )
        summary = {
            "status": "PASS",
            "source_manifest_sha256": sha(root / "ablations/full.json"),
            "matches": len(allowed),
            **counts,
            **tables,
            "elapsed_s": time.monotonic() - start,
            "max_uss_bytes": memory.max_uss,
            "protected_details_read": False,
            "training_started": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        snapshot = out / "source"
        snapshot.mkdir()
        for path in [
            Path(__file__),
            Path("src/pubg_lab/world_model/cohorts.py"),
            Path("docs/experiment_design/WORLD_MODEL_G3_COHORT_PRIOR_2026-10-20.md"),
        ]:
            (snapshot / path.name).write_bytes(path.read_bytes())
        db.close()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "matches": len(allowed),
                    **counts,
                    "elapsed_s": summary["elapsed_s"],
                    "max_uss_bytes": memory.max_uss,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
