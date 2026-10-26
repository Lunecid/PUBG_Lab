"""Merge tiers without duplicate samples, recheck every short label, smoke inputs."""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys,json,sqlite3,argparse
from pathlib import Path
from collections import defaultdict,Counter
from dataclasses import fields
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from world_model.cohorts import CohortDataset,load_at
from world_model.inputs import ConditionVocabulary
from tools.verify_world_model_encoder import ResourceSampler,sha


def main():
    p=argparse.ArgumentParser()
    for k in ("root","index","short","output"):p.add_argument("--"+k,required=True)
    a=p.parse_args();idx,out=Path(a.index),Path(a.output);out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(f"file:{idx/'cohort_events.sqlite'}?mode=ro",uri=True)
    pool={}
    for mid,k,actor,role,cohort in db.execute("SELECT * FROM selected WHERE cohort IN ('normal_br','ranked_br') ORDER BY role,match_id,frame,actor"):
        events=[r[0] for r in db.execute("SELECT event_id FROM selected_events WHERE match_id=? AND frame=? AND actor=? ORDER BY event_id",(mid,k,actor))]
        pool[mid,k,actor]=dict(match_id=mid,frame=k,actor=actor,role=role,tiers=["first30"],event_ids=events)
    short=defaultdict(list)
    for line in (Path(a.short)/"samples.jsonl").open():
        r=json.loads(line);short[r["role"],r["match_id"]].append(r)
    checked=Counter();views=0;unique_short=set()
    torch.set_num_threads(1)
    with ResourceSampler() as memory:
        for role in ("train","validation"):
            ds=CohortDataset(a.root,idx/"primary_br_manifest.json",role)
            vocab=ConditionVocabulary.from_dataset(ds)
            for (rrole,mid),records in sorted(short.items()):
                if rrole!=role:continue
                b=ds.bank(mid)
                for r in records:
                    k,actor=r["frame"],r["actor"]
                    assert b["known"][k,actor] and b["valid"][k,actor] and k>=12
                    event=db.execute("SELECT event_episode FROM events WHERE event_id=?",(r["event_id"],)).fetchone()
                    assert int(b["episode"][k,actor])==event[0]
                    q=k+np.arange(1,7);qc=np.minimum(q,len(b["times"])-1)
                    valid=(b["target_valid"][qc,actor] & (q<len(b["times"]))
                           & (b["target_episode"][qc,actor]==b["episode"][k,actor])
                           & (b["target_segment"][qc,actor]==b["segment"][k,actor]))
                    mode=5 if r["kind"]=="reentry_direct" else 4
                    actual=(np.flatnonzero(valid & (b["target_mode"][qc,actor]==mode))*10+10).tolist()
                    assert actual and actual==r["recorded_transport_horizons_s"]
                    assert (np.flatnonzero(valid)*10+10).tolist()==r["valid_horizons_s"]
                    bucket=role,r["kind"]
                    if checked[bucket]<16:
                        for view in ("observer","server"):
                            ds.view=view;s=load_at(ds,mid,k,actor,vocab)
                            assert all(torch.isfinite(getattr(s.inputs,f.name)).all()
                                       for f in fields(s.inputs) if isinstance(getattr(s.inputs,f.name),torch.Tensor))
                            views+=1
                    checked[bucket]+=1
                    key=mid,k,actor;unique_short.add(key)
                    if key not in pool:
                        pool[key]=dict(match_id=mid,frame=k,actor=actor,role=role,tiers=[],event_ids=[])
                    pool[key]["tiers"]=sorted(set(pool[key]["tiers"]+["short_transport"]))
                    pool[key]["event_ids"]=sorted(set(pool[key]["event_ids"]+[r["event_id"]]))
                if memory.errors or memory.over_limit:raise MemoryError("USS ceiling/measurement")
        with (out/"samples.jsonl").open("w") as h:
            for key in sorted(pool):
                h.write(json.dumps(pool[key])+"\n")
        summary=dict(status="PASS",unique_samples=len(pool),unique_short_samples=len(unique_short),
                     role_counts=dict(Counter(r["role"] for r in pool.values())),
                     both_tiers=sum(len(r["tiers"])==2 for r in pool.values()),
                     short_event_links_checked=sum(checked.values()),short_adapter_views=views,
                     input_fields=["match_id","frame","actor","role"],
                     metadata_only=["tiers","event_ids"],protected_details_read=False,training_started=False,
                     source_manifest_sha256=sha(idx/"primary_br_manifest.json"),max_uss_bytes=memory.max_uss)
        (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
