"""All primary selected labels + per-match/kind two-view adapter verification."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[3]))

from pubg_lab.common.paths import REPO_ROOT
import os

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import sqlite3
import gzip
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import fields

sys.path.insert(0, str((REPO_ROOT)))
import numpy as np
import torch
from pubg_lab.world_model.cohorts import CohortDataset, load_at, label_support
from pubg_lab.world_model.inputs import ConditionVocabulary
from pubg_lab.world_model.preprocess import mode_of
from pubg_lab.world_model.cli.verify_world_model_encoder import ResourceSampler, tensor_hash, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    root, idx, out = Path(a.root), Path(a.index), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    db = sqlite3.connect(f"file:{idx / 'cohort_events.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    counts, raw_checks = defaultdict(Counter), []
    matches_by_stratum = defaultdict(set)
    examples = []
    selected_count = 0
    views = 0
    start = time.monotonic()
    with ResourceSampler() as memory, (out / "primary_rare_samples.jsonl").open("w") as handle:
        for role in ("train", "validation"):
            ds = CohortDataset(root, idx / "primary_br_manifest.json", role)
            vocab = ConditionVocabulary.from_dataset(ds)
            mids = [
                r[0]
                for r in db.execute(
                    """SELECT DISTINCT s.match_id FROM selected s
                JOIN matches m USING(match_id) WHERE s.role=? AND s.cohort IN ('normal_br','ranked_br')
                ORDER BY m.map,s.match_id""",
                    (role,),
                )
            ]
            for index, mid in enumerate(mids):
                b = ds.bank(mid)
                raw = None
                inspected = set()
                rows = list(
                    db.execute(
                        """SELECT s.frame,s.actor,e.event_id,e.kind,e.start_s,e.end_s,e.source_time,e.event_episode
                    FROM selected s JOIN selected_events se USING(match_id,frame,actor)
                    JOIN events e USING(event_id) WHERE s.match_id=? ORDER BY s.frame,s.actor,e.event_id""",
                        (mid,),
                    )
                )
                samples = defaultdict(list)
                for row in rows:
                    k, actor = row["frame"], row["actor"]
                    support = label_support(b, k, actor)
                    assert support["ego_first30"]
                    assert b["episode"][k, actor] == row["event_episode"]
                    key = (role, row["kind"])
                    counts[key]["supported_event_links"] += 1
                    matches_by_stratum[key].add(mid)
                    q = k + np.arange(1, 7)
                    qc = np.minimum(q, len(b["times"]) - 1)
                    valid = (
                        b["target_valid"][qc, actor]
                        & (q < len(b["times"]))
                        & (b["target_episode"][qc, actor] == b["episode"][k, actor])
                        & (b["target_segment"][qc, actor] == b["segment"][k, actor])
                    )
                    mode = 4 if row["kind"] == "emergency_pickup" else 5
                    mode_points = int(((b["target_mode"][qc, actor] == mode) & valid).sum())
                    counts[key]["matching_air_mode_points"] += mode_points
                    counts[key]["samples_with_matching_air_mode"] += int(mode_points > 0)
                    counts[key]["during_event_samples"] += int(b["times"][k] <= row["end_s"])
                    counts[key]["post_event_samples"] += int(b["times"][k] > row["end_s"])
                    counts[key]["full60_samples"] += int(support["ego_full60"])
                    samples[k, actor].append(
                        dict(
                            event_id=row["event_id"],
                            kind=row["kind"],
                            matching_air_mode_points=mode_points,
                            stage="during" if b["times"][k] <= row["end_s"] else "post",
                        )
                    )
                    if row["kind"] not in inspected:
                        inspected.add(row["kind"])
                        hashes = {}
                        for view in ("observer", "server"):
                            ds.view = view
                            sample = load_at(ds, mid, k, actor, vocab)
                            for f in fields(sample.inputs):
                                value = getattr(sample.inputs, f.name)
                                if isinstance(value, torch.Tensor):
                                    assert torch.isfinite(value).all(), f.name
                            if view == "observer":
                                g = sample.inputs
                                others = torch.arange(len(g.actor_ids)) != g.ego_local_index
                                assert not g.mask[:, others, 8:].any()
                            hashes[view] = tensor_hash(
                                (f.name, getattr(sample.inputs, f.name))
                                for f in fields(sample.inputs)
                                if isinstance(getattr(sample.inputs, f.name), torch.Tensor)
                            )
                            views += 1
                        if (
                            sum(x["role"] == role and x["kind"] == row["kind"] for x in examples)
                            < 16
                        ):
                            examples.append(
                                dict(
                                    role=role,
                                    kind=row["kind"],
                                    match_id=mid,
                                    frame=k,
                                    actor=actor,
                                    input_hashes=hashes,
                                )
                            )
                    # Independently check recorded direct evidence in original cached rows.
                    if (
                        row["kind"] != "reentry_inferred"
                        and sum(x["role"] == role and x["kind"] == row["kind"] for x in raw_checks)
                        < 12
                    ):
                        if raw is None:
                            raw = json.load(gzip.open(root / "raw" / f"{mid}.json.gz", "rt"))
                        found = [
                            r
                            for r in raw["tables"]["positions"]
                            if r["actor"] == actor
                            and abs(r["t"] - row["source_time"]) < 1e-5
                            and mode_of(r) == mode
                        ]
                        assert found, "Direct source position/mode missing"
                        prior_death = any(
                            r["actor"] == actor and r["t"] < row["start_s"]
                            for r in raw["tables"]["kills"]
                        )
                        if mode == 5:
                            assert prior_death, "Direct reentry without prior death"
                        raw_checks.append(
                            dict(
                                role=role,
                                kind=row["kind"],
                                match_id=mid,
                                actor=actor,
                                source_time=row["source_time"],
                                source_rows=len(found),
                                prior_death=prior_death,
                                raw_sha256=sha(root / "raw" / f"{mid}.json.gz"),
                            )
                        )
                for (k, actor), events in samples.items():
                    selected_count += 1
                    handle.write(
                        json.dumps(
                            dict(
                                match_id=mid,
                                role=role,
                                frame=k,
                                actor=actor,
                                events=events,
                                metadata_not_model_input=True,
                            )
                        )
                        + "\n"
                    )
                if (index + 1) % 250 == 0:
                    print(
                        json.dumps(
                            dict(
                                role=role,
                                matches=index + 1,
                                samples=selected_count,
                                adapter_views=views,
                                elapsed_s=round(time.monotonic() - start),
                            )
                        ),
                        flush=True,
                    )
                if memory.errors or memory.over_limit:
                    raise MemoryError("USS measurement/ceiling")
        summary = dict(
            status="PASS",
            primary_samples=selected_count,
            adapter_views=views,
            scope="All selected primary labels; first selected anchor per match/kind both views",
            strata=[
                dict(role=k[0], kind=k[1], matches=len(matches_by_stratum[k]), **v)
                for k, v in sorted(counts.items())
            ],
            raw_checks=raw_checks,
            examples=examples,
            elapsed_s=time.monotonic() - start,
            max_uss_bytes=memory.max_uss,
            protected_details_read=False,
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            json.dumps({k: v for k, v in summary.items() if k not in ("raw_checks", "examples")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
