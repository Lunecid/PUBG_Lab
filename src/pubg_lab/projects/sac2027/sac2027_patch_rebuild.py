#!/usr/bin/env python3
"""Rebuild causal features from raw, read-only telemetry under a frozen patch split."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
import pandas as pd
from pubg_lab.survival import dump_phase_tier_v2 as B
from model_stgcn.splits import write_split_manifest
from pubg_lab.survival.v3.cohorts import resolve_cohort

ROOT = REPO / "reports/sac2027_patch_v14"
PATCH = REPO / "reports/bigcomp2027_results_v2/data_period/per_match_patch.csv"
SPLIT_PATCH = {"train": "40.2", "val": "41.1", "test": "41.2"}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def prepare():
    frame = pd.read_csv(PATCH, dtype={"patch": str})
    assert len(frame) == 13948 and frame.match_id.is_unique
    assert set(frame.patch) == set(SPLIT_PATCH.values())
    frame["previous_split"] = frame["split"]
    frame["split"] = frame.patch.map({p: s for s, p in SPLIT_PATCH.items()})
    time_ranges = frame.groupby("split").created_utc.agg(["min", "max"])
    assert time_ranges.loc["train", "max"] < time_ranges.loc["val", "min"]
    assert time_ranges.loc["val", "max"] < time_ranges.loc["test", "min"]
    ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / "split_manifest.csv", index=False)
    kept, pop = {}, {"design": "patch-temporal-2026-10-08", "cohorts": {}}
    counts = []
    for slug, group in frame.groupby("cohort", sort=True):
        kept[slug] = {s: sorted(group.loc[group.split == s, "match_id"]) for s in B.SPLITS}
        n = {s: len(ids) for s, ids in kept[slug].items()}
        assert all(n.values())
        role = "main" if n["train"] >= 50 else "exploratory_low_training_support"
        pop["cohorts"][slug] = {"status": "kept", "after": n, "analysis_role": role}
        counts.append({"cohort": slug, **n, "analysis_role": role})
        write_split_manifest(
            ROOT / "splits" / slug / "split.json",
            ids=sorted(group.match_id),
            train_ids=kept[slug]["train"],
            val_ids=kept[slug]["val"],
            test_ids=kept[slug]["test"],
            seed=42,
            val_frac=n["val"] / len(group),
            test_frac=n["test"] / len(group),
            metadata={"assignment": "patch, not random", "patches": SPLIT_PATCH},
        )
    pd.DataFrame(counts).to_csv(ROOT / "cohort_counts.csv", index=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_training",
        "patch_csv_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        "patches": SPLIT_PATCH,
        "n_matches": len(frame),
        "counts": frame.split.value_counts().to_dict(),
        "schedule_estimation": "40.2 train matches only; original hierarchical median and minimum support 10",
        "feature_build": "all causal inputs rebuilt from raw telemetry; original v2 labels and feature definitions",
        "preprocessing": "fit scaling, imputation and learned transforms on train only",
        "selection": "validation only; never test for checkpoint, hyperparameter or probability calibration selection",
        "test_exposure": "retrospective temporal re-evaluation; these matches appeared in previous analyses",
        "low_support_rule": "train < 50 matches: run but report separately; main pooling excludes these cohorts",
        "boundary_uncertainty": "5 validation matches inside announced 41.1 maintenance; retain existing assignment and report",
        "tuning": "16 configurations per architecture on Erangel TPP squad and Taego TPP squad; matched inputs, BCE target and search budget",
        "inference": "paired match bootstrap; report per-cohort results, human-team subset and between-cohort heterogeneity",
        "scope": "rerun paper main models, decomposition controls, ablations, seeds and RQ1; no reuse of fitted artifacts",
    }
    path = ROOT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        protocol["created_utc"] = old["created_utc"]
        assert protocol == old, "protocol changed; use a separately versioned design"
    write(path, protocol)
    return frame, kept, pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frame, kept, pop = prepare()
    print(frame.groupby(["split", "patch"]).size().to_string(), flush=True)
    if args.prepare_only:
        return
    from pubg_lab.data.stdb.conn import connect

    conn = connect()
    conn.set_session(readonly=True, autocommit=True)
    meta = {
        r.match_id: (r.cohort, r.map_name, r.match_type)
        for r in frame.itertuples()
        if r.split == "train"
    }
    with conn.cursor() as cur:
        durations = B._durations_by_match(cur, sorted(meta))
    conn.close()
    assert set(durations) <= set(meta)
    schedule = B.schedule_table(durations, meta)
    out = ROOT / "dumps/phase_v2"
    write(out / "population.json", pop)
    write(
        out / "schedule_table.json",
        {
            "|".join(map(str, k)): {str(p): list(v) for p, v in d.items()}
            for k, d in schedule.items()
        },
    )
    write(
        out / "schedule_provenance.json",
        {"train_match_ids": sorted(meta), "n_with_durations": len(durations), "patch": "40.2"},
    )
    mtype = dict(zip(frame.match_id, frame.match_type))
    for slug in kept:
        completed = out / slug / "dump_manifest.json"
        if completed.exists():
            man = json.loads(completed.read_text())
            if all(
                man["splits"][s]["n_dumped"] == len(kept[slug][s])
                and not man["splits"][s]["n_skipped"]
                for s in B.SPLITS
            ):
                print("already rebuilt", slug, flush=True)
                continue
        result = B.dump_cohort_v2(
            resolve_cohort(slug),
            out,
            workers=args.workers,
            population=pop,
            kept_ids=kept[slug],
            sched_table=schedule,
            mtype=mtype,
            split_root=ROOT / "splits",
        )
        assert not any(x["n_skipped"] for x in result["splits"].values()), result
    write(
        ROOT / "rebuild_complete.json",
        {"finished_utc": datetime.now(timezone.utc).isoformat(), "n_matches": len(frame)},
    )


if __name__ == "__main__":
    main()
