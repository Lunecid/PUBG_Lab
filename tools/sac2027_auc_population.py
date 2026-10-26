#!/usr/bin/env python3
"""Exact descriptive decomposition of mixed-population AUC into four pair types."""
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tools.sac2027_followup_run import BASE,wait_for
from tools.sac2027_evaluate_controls import definitions
from model_stgcn.ptr import data as D,hgb as H,metrics as M

def main():
    wait_for([BASE/'control_evaluation/main_population/complete.json'])
    counts=pd.read_csv(BASE/'cohort_counts.csv');records=[];checks=[]
    for cohort in counts.loc[counts.analysis_role=='main','cohort']:
        rows=H.build_rows(D.load_dump(BASE/'dumps/phase_v2'/cohort/'test.pt'),'HGB-cut','v2')
        meta=rows.keys.copy();meta['ai']=rows.X[:,rows.feature_names.index('team_ai_frac')]
        paths,_=definitions('main_population',cohort)
        for arm,files in paths.items():
            f=M.read_predictions(files[0]).merge(meta,on=M.KEY_COLUMNS,validate='one_to_one')
            y=f.y.to_numpy();score=f.p_hazard.to_numpy();ai=f.ai.to_numpy()
            denominator=int((y==1).sum())*int((y==0).sum());total=weight_sum=0.
            for pos_group in (0,1):
                for neg_group in (0,1):
                    ps=score[(y==1)&(ai==pos_group)];ns=score[(y==0)&(ai==neg_group)]
                    weight=len(ps)*len(ns)/denominator
                    auc=float(roc_auc_score(np.r_[np.ones(len(ps)),np.zeros(len(ns))],np.r_[ps,ns])) if weight else np.nan
                    contribution=weight*auc if weight else 0.
                    total+=contribution;weight_sum+=weight
                    records.append(dict(cohort=cohort,arm=arm,positive_group='ai' if pos_group else 'human',
                                        negative_group='ai' if neg_group else 'human',n_positive=len(ps),n_negative=len(ns),
                                        pair_weight=weight,pair_auc=auc,weighted_contribution=contribution))
            point=float(roc_auc_score(y,score))
            assert abs(weight_sum-1)<1e-12 and abs(point-total)<1e-12
            checks.append(dict(cohort=cohort,arm=arm,auc=point,reconstructed_auc=total,absolute_error=abs(point-total)))
    out=BASE/'population_auc_decomposition';out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out/'pair_types.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'reconstruction_checks.csv',index=False)
    (out/'README.txt').write_text('Exact pair decomposition, not a causal attribution of model advantage. The mixed AUC weights human-human, human-AI, AI-human and AI-AI positive-negative pairs. No models are selected or retrained here. All 11 primary cohorts, reference seed 42.\n')
    print('Exact AUC reconstruction passed for',len(checks),'cohort/model combinations.',flush=True)

if __name__=='__main__':main()
