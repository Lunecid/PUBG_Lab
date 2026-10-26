"""Supplementary short-flight tier; never relabel as 30s-complete trajectories."""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys,json,sqlite3,argparse,time
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.verify_world_model_encoder import ResourceSampler


def main():
    p=argparse.ArgumentParser()
    for k in ("root","index","output"): p.add_argument("--"+k,required=True)
    a=p.parse_args(); root,idx,out=Path(a.root),Path(a.index),Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(f"file:{idx/'cohort_events.sqlite'}?mode=ro",uri=True)
    db.row_factory=sqlite3.Row
    counts=defaultdict(Counter); matches=defaultdict(set); records=[]
    started=time.monotonic()
    with ResourceSampler() as memory:
        mids=db.execute("""SELECT DISTINCT m.match_id,m.role FROM matches m JOIN events e USING(match_id)
          WHERE m.cohort IN ('normal_br','ranked_br') AND e.kind IN ('reentry_direct','emergency_pickup')
          ORDER BY m.role,m.match_id""").fetchall()
        for n,(mid,role) in enumerate(mids):
            with np.load(root/"banks"/f"{mid}.npz",allow_pickle=False) as f: b=dict(f)
            events=db.execute("SELECT * FROM events WHERE match_id=? AND kind IN ('reentry_direct','emergency_pickup')",(mid,)).fetchall()
            for e in events:
                actor=e["actor"]; mode=5 if e["kind"]=="reentry_direct" else 4
                cs=db.execute("SELECT frame FROM candidates WHERE event_id=? AND stage!='pre' ORDER BY frame",(e["event_id"],)).fetchall()
                best=None
                for (k,) in cs:
                    q=k+np.arange(1,7); qc=np.minimum(q,len(b["times"])-1)
                    valid=(b["target_valid"][qc,actor] & (q<len(b["times"]))
                           & (b["target_episode"][qc,actor]==b["episode"][k,actor])
                           & (b["target_segment"][qc,actor]==b["segment"][k,actor]))
                    matching=valid & (b["target_mode"][qc,actor]==mode)
                    score=(int(matching.sum()),int(valid[:3].all()),-k)
                    if score[0] and (best is None or score>best[0]):
                        best=(score,k,valid,matching)
                if best is None: continue
                score,k,valid,matching=best
                key=(role,e["kind"])
                counts[key]["actor_events"]+=1; matches[key].add(mid)
                counts[key]["matching_mode_points"]+=int(matching.sum())
                counts[key]["first30_complete"]+=int(valid[:3].all())
                counts[key]["first20_complete"]+=int(valid[:2].all())
                counts[key]["full60_complete"]+=int(valid.all())
                records.append(dict(match_id=mid,role=role,actor=actor,frame=k,event_id=e["event_id"],
                                    kind=e["kind"],tier="short_transport_partial",
                                    valid_horizons_s=(np.flatnonzero(valid)*10+10).tolist(),
                                    recorded_transport_horizons_s=(np.flatnonzero(matching)*10+10).tolist(),
                                    selected_using_future_labels=True,metadata_not_model_input=True))
            if (n+1)%1000==0: print(json.dumps({"matches":n+1,"selected":len(records)}),flush=True)
            if memory.errors or memory.over_limit: raise MemoryError("USS ceiling/measurement")
        with (out/"samples.jsonl").open("w") as h:
            for r in records: h.write(json.dumps(r)+"\n")
        summary=dict(status="PASS",strata=[dict(role=k[0],kind=k[1],matches=len(matches[k]),**v) for k,v in sorted(counts.items())],
                     selected_event_links=len(records),
                     unique_samples=len({(r["match_id"],r["frame"],r["actor"]) for r in records}),
                     elapsed_s=time.monotonic()-started,max_uss_bytes=memory.max_uss,
                     qualification="Separate partial-label tier; 30s completeness gate not relaxed for main tier",
                     protected_details_read=False,training_started=False)
        (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
