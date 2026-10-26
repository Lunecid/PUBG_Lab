#!/usr/bin/env python3
"""Matched test contrasts of frozen controls; average seed metrics, not ensembles."""
import argparse
import hashlib
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
from tools.sac2027_fair_tune import BASE,DUMP,M,save
from model_stgcn.ptr import data as D,hgb as H
from model_stgcn.ptr.supplement import RiskSetAUC

SEEDS=(42,13,2026)
METRICS=("auc","auc_within_phase","nll","brier")

def definitions(suite,cohort):
    b=BASE; paths={}; pairs=[]
    if suite=="tuning":
        paths={a:[b/"tuning"/cohort/a/f"selected_seed{s}"/"pred_test.csv" for s in SEEDS] for a in ("gbdt","krafton","ptr")}
        pairs=[("ptr","gbdt"),("ptr","krafton"),("krafton","gbdt")]
    elif suite=="graph":
        paths={"gatv2":[b/"tuning"/cohort/"gatv2"/f"selected_seed{s}"/"pred_test.csv" for s in SEEDS],
               "gatv2_shuffled":[b/"graph_controls"/cohort/"shuffled"/f"seed{s}"/"pred_test.csv" for s in SEEDS],
               "gbdt":[b/"tuning"/cohort/"gbdt"/f"selected_seed{s}"/"pred_test.csv" for s in SEEDS]}
        for a in ("krafton","ptr"):
            paths[a]=[b/"population_controls"/cohort/a/"mixed_cpu"/f"seed{s}"/"pred_test.csv" for s in SEEDS]
        pairs=[("gatv2",a) for a in ("gbdt","krafton","ptr","gatv2_shuffled")]
    elif suite=="objective":
        paths={a:[b/"objective_controls"/cohort/a/f"seed{s}"/"pred_test.csv" for s in SEEDS]
               for a in ("weighted_full","weighted_truncated","cox_full","cox_truncated","bce","cox_truncated_bce")}
        pairs=[("weighted_truncated","weighted_full"),("cox_truncated","cox_full"),("bce","weighted_full"),
               ("bce","weighted_truncated"),("bce","cox_truncated"),("cox_full","weighted_full"),
               ("bce","cox_truncated_bce"),("cox_truncated_bce","cox_truncated")]
    elif suite=="history":
        paths={a:[b/"history_controls"/cohort/a/f"seed{s}"/"pred_test.csv" for s in SEEDS]
               for a in ("krafton_current_padded","krafton_history_summary","ptr_ordered_prefix","ptr_shuffled_prefix")}
        pairs=[("krafton_history_summary","krafton_current_padded"),("ptr_ordered_prefix","ptr_shuffled_prefix")]
    elif suite=="population":
        for a in ("gbdt","krafton","ptr"):
            paths[a+"_mixed"]=([b/"tuning"/cohort/a/f"selected_seed{s}"/"pred_test.csv" for s in SEEDS] if a=="gbdt" else
                                [b/"population_controls"/cohort/a/"mixed_cpu"/f"seed{s}"/"pred_test.csv" for s in SEEDS])
            for c in ("no_ai_input","human_loss"):
                paths[a+"_"+c]=[b/"population_controls"/cohort/a/c/f"seed{s}"/"pred_test.csv" for s in SEEDS]
                pairs.append((a+"_"+c,a+"_mixed"))
    elif suite=="main_population":
        arms=("HGB-cut","M11R","KRAFTON-tf","KRAFTON-tf-h0","SAGONG-xgb","KRAFTON-tf-ourfeat","KRAFTON-tf-ourfeat-bce","HGB-union")
        paths={a:[b/"checkpoints/ptr_v2"/cohort/a/"seed42/pred_test.csv"] for a in arms}
        pairs=[("M11R","HGB-cut"),("M11R","KRAFTON-tf-h0"),("HGB-cut","KRAFTON-tf-h0"),
               ("M11R","SAGONG-xgb"),("KRAFTON-tf-ourfeat-bce","KRAFTON-tf-ourfeat")]
    return paths,pairs

def calibration(frame):
    f=frame.copy();f["err"]=f.p_hazard-f.y
    phase=f.groupby("phase").err.agg(["mean","size"])
    f["bin"]=np.minimum((f.p_hazard*10).astype(int),9)
    bins=f.groupby(["phase","bin"]).err.agg(["mean","size"])
    return dict(phase_calibration_error=float((phase["mean"].abs()*phase["size"]).sum()/len(f)),
                within_phase_ece=float((bins["mean"].abs()*bins["size"]).sum()/len(f)))

def analyze(suite,cohort,population,paths,pairs,meta):
    frames={};groups={};provenance=[]
    for name,files in paths.items():
        groups[name]=[]
        for seed_index,path in enumerate(files):
            frame=M.read_predictions(path)
            joined=frame.merge(meta,on=M.KEY_COLUMNS,validate="one_to_one",how="outer",indicator=True,suffixes=("","_check"))
            assert joined._merge.eq("both").all() and np.array_equal(joined.y,joined.y_check)
            if population=="human":frame=joined.loc[joined.ai==0,M.PRED_COLUMNS].reset_index(drop=True)
            key=f"{name}:seed{SEEDS[seed_index]}";frames[key]=frame;groups[name].append(key)
            provenance.append(dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    seed=int(hashlib.sha256(f"20261008|{suite}|{cohort}|{population}".encode()).hexdigest()[:8],16)
    _,point,samples=M.bootstrap_arms(frames,metrics=METRICS,n_boot=2000,seed=seed)
    ref=next(iter(frames.values()));nmatch=ref.match_id.nunique()
    counts=M.resample_counts(nmatch,2000,seed)
    for key,frame in frames.items():
        r=RiskSetAUC(frame);point[key]["auc_within_match_phase"]=r.evaluate(np.ones(nmatch))
        samples[key]["auc_within_match_phase"]=r.evaluate(counts)
        point[key].update(calibration(frame))
    means={name:{m:float(np.mean([point[k][m] for k in keys])) for m in point[keys[0]]} for name,keys in groups.items()}
    contrast=[]
    for plus,minus in pairs:
        for metric in (*METRICS,"auc_within_match_phase"):
            estimate=means[plus][metric]-means[minus][metric]
            draws=np.mean([samples[k][metric] for k in groups[plus]],axis=0)-np.mean([samples[k][metric] for k in groups[minus]],axis=0)
            valid=draws[np.isfinite(draws)];lo,hi=np.quantile(valid,[.025,.975]);lo90,hi90=np.quantile(valid,[.05,.95])
            contrast.append(dict(cohort=cohort,population=population,contrast=plus+"_minus_"+minus,metric=metric,
                                 estimate=estimate,se=float(valid.std(ddof=1)),ci_low=float(lo),ci_high=float(hi),
                                 ci90_low=float(lo90),ci90_high=float(hi90)))
    arms=[dict(cohort=cohort,population=population,arm=name,n_rows=len(ref),n_matches=nmatch,
               n_seeds=len(groups[name]),**values) for name,values in means.items()]
    return arms,contrast,provenance,point

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--suite",required=True,choices=("tuning","objective","history","population","main_population","graph"));args=ap.parse_args()
    out=BASE/"control_evaluation"/args.suite;out.mkdir(parents=True,exist_ok=True)
    cohorts=pd.read_csv(BASE/"cohort_counts.csv")
    cohorts=cohorts.loc[cohorts.analysis_role=="main","cohort"].tolist() if args.suite=="main_population" else ["erangel_tpp_squad","taego_tpp_squad"]
    if args.suite=="population":
        from tools.sac2027_followup_run import wait_for
        wait_for([BASE/"population_controls"/c/a/"mixed_cpu"/f"seed{s}"/"metrics.json"
                  for c in cohorts for a in ("krafton","ptr") for s in SEEDS])
    arm_rows=[];contrast_rows=[];provenance=[]
    for c in cohorts:
        rows=H.build_rows(D.load_dump(DUMP/c/"test.pt"),"HGB-cut","v2")
        meta=rows.keys.copy();meta["ai"]=rows.X[:,rows.feature_names.index("team_ai_frac")];meta["y"]=rows.y
        paths,pairs=definitions(args.suite,c)
        for pop in ("all","human"):
            print("START",args.suite,c,pop,flush=True)
            arms,contrasts,files,seed_points=analyze(args.suite,c,pop,paths,pairs,meta)
            arm_rows.extend(arms);contrast_rows.extend(contrasts);provenance.extend(files)
            save(out/f"{c}_{pop}_seed_metrics.json",seed_points)
            pd.DataFrame(arm_rows).to_csv(out/"arm_metrics.csv",index=False)
            pd.DataFrame(contrast_rows).to_csv(out/"contrasts.csv",index=False)
            print("DONE",args.suite,c,pop,flush=True)
    pooled=[]
    if args.suite=="main_population":
        for (population,contrast,metric),g in pd.DataFrame(contrast_rows).groupby(["population","contrast","metric"]):
            pooled.append(dict(population=population,contrast=contrast,metric=metric,**M.dersimonian_laird(g.estimate,g.se)))
    save(out/"pooled.json",pooled)
    save(out/"complete.json",dict(suite=args.suite,files=provenance,cohorts=cohorts,n_boot=2000,
                                  note="Metrics averaged across training seeds. Match bootstrap conditions on these seeds and does not estimate seed-population uncertainty. All contrasts exploratory. No test-driven model selection."))

if __name__=="__main__":main()
