#!/usr/bin/env python3
"""Validate the new split, all rebuilt labels, and raw-prefix replay with train-only schedules."""
import argparse
import json
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
import torch
from tools.sac2027_patch_rebuild import ROOT, SPLIT_PATCH, write
from model_stgcn import dump_phase_tier_v2 as B
from model_stgcn.ptr.paths import validate_v2_dump
from model_stgcn.phase_v2 import replay as RP

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--finalize-existing",action="store_true",help="finalize existing full-data audit and replay, without rerunning DB reads")
    args=ap.parse_args()
    dump=ROOT/"dumps/phase_v2"
    if args.finalize_existing:
        audit=json.loads((ROOT/"split_label_validation.json").read_text())
        assert audit["n_matches"]==13948 and audit["labels_identical"] and not audit["gate"]["errors"]
        totals=json.loads((dump/"prefix_replay.json").read_text())
        finalize(dump,totals)
        return
    manifest=pd.read_csv(ROOT/"split_manifest.csv",dtype={"patch":str})
    expected=manifest.set_index("match_id")
    gate=validate_v2_dump(dump,deep=True,max_matches=100000,require_prefix_replay=False)
    assert not gate["errors"],gate
    labels=("team_ids","target_mask_v2","y_hazard_v2","node_mask","query_mask","cut_t","phase_start_s","phase_end_s","team_death_s")
    n=0; changed=0; seen=set()
    for cohort,group in manifest.groupby("cohort"):
        old={}
        for split in SPLIT_PATCH:
            source=torch.load(REPO/"model_stgcn/dumps/bigcomp2027_phase_v2"/cohort/f"{split}.pt",weights_only=False)
            old.update({m["match_id"]:m for m in source["matches"]})
        for split,patch in SPLIT_PATCH.items():
            data=torch.load(dump/cohort/f"{split}.pt",weights_only=False)
            for m in data["matches"]:
                mid=m["match_id"]
                assert mid not in seen;seen.add(mid)
                assert expected.loc[mid,"patch"]==patch
                for key in labels:
                    assert np.array_equal(np.asarray(m[key]),np.asarray(old[mid][key]),equal_nan=True),(mid,key)
                assert torch.isfinite(m["X"]).all()
                changed+=int(not torch.equal(m["X"],old[mid]["X"]))
                n+=1
        print("checked",cohort,"cumulative matches",n,flush=True)
    assert seen==set(manifest.match_id) and n==13948
    provenance=json.loads((dump/"schedule_provenance.json").read_text())
    assert set(provenance["train_match_ids"])==set(manifest.loc[manifest.split=="train","match_id"])
    write(ROOT/"split_label_validation.json",dict(n_matches=n,labels_identical=True,n_matches_with_changed_X=changed,gate=gate))
    table={tuple(k.split("|")):{int(p):tuple(v) for p,v in per.items()}
           for k,per in json.loads((dump/"schedule_table.json").read_text()).items()}
    selected=manifest.sort_values("match_id").groupby(["cohort","split"],sort=True).head(2)
    from stdb.conn import connect
    conn=connect();conn.set_session(readonly=True,autocommit=True)
    totals=dict(n_matches=0,n_cuts=0,mismatches=[],rows_seen_full=0,rows_seen_truncated=0,
        min_gap_between_last_kept_record_and_cut_s=float("inf"),mutation_check=dict(cuts_compared=0,cuts_detected_as_leaky=0))
    with conn.cursor() as cur:
        for row in selected.itertuples():
            sched,_=B.resolve_schedule(table,row.cohort,row.map_name,row.match_type)
            res=RP.run_prefix_replay(cur,[(row.cohort,row.match_id)],schedule=sched,mutation_matches=1)
            for key in ("n_matches","n_cuts","rows_seen_full","rows_seen_truncated"):totals[key]+=res[key]
            totals["mismatches"].extend(res["mismatches"])
            totals["min_gap_between_last_kept_record_and_cut_s"]=min(totals["min_gap_between_last_kept_record_and_cut_s"],res["min_gap_between_last_kept_record_and_cut_s"])
            for key in totals["mutation_check"]:totals["mutation_check"][key]+=res["mutation_check"][key]
            print("prefix replay",totals["n_matches"],"/",len(selected),flush=True)
    conn.close()
    totals["selected_match_ids"]=selected.match_id.tolist()
    totals["schedule_source"]="new 40.2-only hierarchical medians; per-match resolved schedule"
    write(dump/"prefix_replay.json",totals)
    finalize(dump,totals)

def finalize(dump,totals):
    assert totals["n_matches"]==72 and not totals["mismatches"],totals
    # Removing a safeguard need not alter every real-data cut. The synthetic
    # negative control guarantees post-cut signal and is checked separately.
    assert totals["mutation_check"]["cuts_detected_as_leaky"]>0
    totals["mutation_check"]["interpretation"]="Empirical sensitivity, not a per-cut requirement: some cuts are unchanged by this particular mutation. Report all tested cuts."
    write(dump/"prefix_replay.json",totals)
    final=validate_v2_dump(dump,deep=True,require_prefix_replay=True)
    write(ROOT/"validation_complete.json",final)
    assert not final["errors"],final
    print("ALL PATCH VALIDATION CHECKS PASSED",flush=True)

if __name__=="__main__":main()
