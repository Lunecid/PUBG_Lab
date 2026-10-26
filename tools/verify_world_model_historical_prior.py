"""Read-only historical side-channel demonstration on fixed current-state crops."""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, json, sqlite3, argparse, hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from world_model.cohorts import CohortDataset, load_at
from world_model.inputs import ConditionVocabulary
from world_model.historical_prior import HistoricalPrior, timestamp
from tools.verify_world_model_encoder import ResourceSampler, sha


def main():
    ap = argparse.ArgumentParser()
    for x in ("root", "index", "prior", "output"):
        ap.add_argument("--"+x, required=True)
    a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    index = Path(a.index)
    db = sqlite3.connect(f"file:{index/'cohort_events.sqlite'}?mode=ro", uri=True)
    prior_meta = json.loads((Path(a.prior)/"summary.json").read_text())
    prior = HistoricalPrior(Path(a.prior)/"historical_prior.sqlite", prior_meta["actual_fit_ids"])
    torch.set_num_threads(1)
    records = []
    with ResourceSampler() as memory:
        for role in ("train", "validation"):
            ds = CohortDataset(a.root, index/"primary_br_manifest.json", role)
            vocab = ConditionVocabulary.from_dataset(ds)
            rows = db.execute("""SELECT match_id,frame,actor FROM selected WHERE role=?
                AND cohort IN ('normal_br','ranked_br') ORDER BY match_id,frame,actor""", (role,)).fetchall()
            rows.sort(key=lambda r: hashlib.sha256(str(r).encode()).hexdigest())
            # Exactly 32 distinct matches per role; selection uses no performance.
            chosen, seen = [], set()
            for r in rows:
                if r[0] not in seen:
                    chosen.append(r); seen.add(r[0])
                if len(chosen) == 32:
                    break
            for mid, k, actor in chosen:
                sample = load_at(ds, mid, k, actor, vocab)
                g = sample.inputs
                cells = np.floor(g.cells[:, -2:].numpy()*100).astype(int)
                meta = ds.metadata[mid]
                edges = prior.query(map_name=meta["map"], target_mid=mid,
                                    target_start=timestamp(meta["start_time"]),
                                    target_patch=meta["patch"], cells=cells)
                assert all(e["age_days"] > 0 and not e["geometry_verified"] for e in edges)
                if role == "validation":
                    assert len(g.observed) == 0
                records.append(dict(role=role, match_id=mid, frame=k, actor=actor, map=meta["map"],
                                    exact_patch_edges=len(g.observed), hypothetical_edges=len(g.inferred),
                                    cells=cells.tolist(), historical_edges=edges))
        summary = {
            "status": "PASS", "records": len(records), "protected_details_read": False,
            "model_training": False, "fitted_on_validation": False,
            "roles": {role: {
                "sampled_matches": sum(r["role"] == role for r in records),
                "with_historical_edges": sum(r["role"] == role and bool(r["historical_edges"]) for r in records),
                "historical_edges": sum(len(r["historical_edges"]) for r in records if r["role"] == role),
                "exact_patch_edges": sum(r["exact_patch_edges"] for r in records if r["role"] == role),
            } for role in ("train", "validation")},
            "prior_sha256": sha(Path(a.prior)/"historical_prior.sqlite"),
            "max_uss_bytes": memory.max_uss}
        (out/"examples.json").write_text(json.dumps(records, indent=2))
        (out/"summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    prior.close(); db.close()


if __name__ == "__main__":
    main()
