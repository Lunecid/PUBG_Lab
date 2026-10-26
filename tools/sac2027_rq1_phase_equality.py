#!/usr/bin/env python3
"""AI-adjusted phase-band equality tests using full clustered covariance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sac2027_rq1_controls import BASE,BASE_TERMS,DUMP,OUT,R,E,H,D,pd

def main():
    cohorts=pd.read_csv(BASE/"cohort_counts.csv")
    cohorts=cohorts.loc[cohorts.analysis_role=="main","cohort"]
    interactions=[];contrasts=[]
    with R.ecomodel_patched(terms=BASE_TERMS+("ai_share",),sources={"ai_share":"team_ai_frac"},construct={"ai_share":"control"}):
        for cohort in cohorts:
            rows=H.build_rows(D.load_dump(DUMP/cohort/"train.pt"),"HGB-cut","v2")
            inputs=E.make_rq1_inputs(rows);model=E.fit_rq1("phase",inputs,rows.y)
            for scale in ("train_sd","group_sd"):
                tables=E.rq1_tables(model,cohort,scale=scale)
                interactions.append(tables["interactions"]);contrasts.append(tables["contrasts"])
            print(cohort,flush=True)
        inter=pd.concat(interactions,ignore_index=True);con=pd.concat(contrasts,ignore_index=True)
        inter.to_csv(OUT/"ai_adjusted_interactions.csv",index=False)
        con.to_csv(OUT/"ai_adjusted_band_contrasts.csv",index=False)
        E.rq1_equality(inter,con).to_csv(OUT/"ai_adjusted_phase_equality.csv",index=False)

if __name__=="__main__":main()
