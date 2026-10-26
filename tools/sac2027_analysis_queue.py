#!/usr/bin/env python3
"""Run analyses only when every prespecified producer has finished."""
import argparse
import subprocess
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tools.sac2027_followup_run import wait_for,BASE
from tools.sac2027_evaluate_controls import definitions
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument("suite",choices=("primary","tuning","objective","history","population","main_population","graph"));a=ap.parse_args()
    counts=pd.read_csv(BASE/"cohort_counts.csv")
    main_cohorts=counts.loc[counts.analysis_role=="main","cohort"].tolist()
    if a.suite=="primary":
        wait_for([BASE/"main_complete.json",BASE/"sagong_corrected_complete.json"])
        # Reuse exactly the bootstrap estimates produced by the main matrix;
        # only the declared low-support cohort is omitted from primary pooling.
        from model_stgcn.ptr import aggregate as A
        source=BASE/"results_v2";out=BASE/"primary_results_v2";out.mkdir(parents=True,exist_ok=True)
        pc=pd.read_csv(source/"contrasts_per_cohort.csv");pc=pc[pc.cohort.isin(main_cohorts)]
        arms=pd.read_csv(source/"arm_metrics_per_cohort.csv");arms=arms[arms.cohort.isin(main_cohorts)]
        pc.to_csv(out/"contrasts_per_cohort.csv",index=False);arms.to_csv(out/"arm_metrics_per_cohort.csv",index=False)
        A.pool_contrasts(pc,len(main_cohorts)).to_csv(out/"contrasts_pooled.csv",index=False)
        A.arm_means(arms).to_csv(out/"arm_metrics_mean.csv",index=False)
        subprocess.run([sys.executable,"-m","model_stgcn.ptr.supplement","--runs-dir",str(BASE/"checkpoints/ptr_v2"),
                        "--results-dir",str(out),"--out-dir",str(out/"supplement"),"--cohorts",",".join(main_cohorts),
                        "--workers","2","--n-boot","2000"],check=True,cwd=REPO)
        (out/"complete.json").write_text('{"completed":true,"primary_cohorts":11,"excluded":"karakin_tpp_squad; train n=6"}\n')
    else:
        cohorts=main_cohorts if a.suite=="main_population" else ["erangel_tpp_squad","taego_tpp_squad"]
        paths=[]
        for c in cohorts:
            defs,_=definitions(a.suite,c)
            paths.extend(p.parent/"metrics.json" for files in defs.values() for p in files)
        if a.suite=="main_population":paths.append(BASE/"sagong_corrected_complete.json")
        wait_for(paths)
        subprocess.run([sys.executable,"tools/sac2027_evaluate_controls.py","--suite",a.suite],check=True,cwd=REPO)
    print("COMPLETE",a.suite,flush=True)

if __name__=="__main__":main()
