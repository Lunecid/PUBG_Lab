#!/usr/bin/env python3
"""Completion and selection audit for the fixed v14 experiment plan."""
import hashlib
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
from tools.sac2027_fair_tune import BASE,OUT,M,save
from tools.sac2027_evaluate_controls import definitions

def main():
    required=['main_complete.json','sagong_corrected_complete.json','neural_search_complete.json',
              'gbdt_search_complete.json','history_complete.json','population_ptr_complete.json',
              'population_krafton_complete.json','mapping_complete.json','primary_results_v2/complete.json',
              'native_evaluation/complete.json','graph_complete.json']
    suites=('tuning','objective','history','population','main_population','graph')
    required.extend('control_evaluation/'+s+'/complete.json' for s in suites)
    missing=[p for p in required if not (BASE/p).is_file()]
    assert not missing,missing
    extension=json.loads((BASE/'graph_model_selection/protocol_extension.json').read_text())
    for name,digest in extension['files'].items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,name
    source=json.loads((BASE/'graph_model_selection/source_check.json').read_text())
    assert source['byte_identical'] and source['installed_sha256']==source['upstream_sha256']
    spec=json.loads((OUT/'search_space.json').read_text());selection=[]
    for cohort in ('erangel_tpp_squad','taego_tpp_squad'):
        for arch in ('gbdt','krafton','ptr','gatv2'):
            root=OUT/cohort/arch
            configs=spec['gbdt_trials' if arch=='gbdt' else 'neural_trials']
            trials=[]
            for i,config in enumerate(configs):
                path=root/f'trial{i:02d}'/'trial.json';d=json.loads(path.read_text())
                assert d['setting']==config and np.isfinite(d['val_nll'])
                if arch!='gbdt':
                    assert len(d['history'])<=12
                    assert abs(d['val_nll']-min(r['val_nll'] for r in d['history']))<1e-6
                trials.append((d['val_nll'],i))
            chosen=json.loads((root/'selection.json').read_text())
            assert chosen['chosen_trial']==min(trials)[1] and chosen['n_trials']==16
            for seed in (42,13,2026):
                selected=json.loads((root/f'selected_seed{seed}'/'metrics.json').read_text())
                assert selected['selection']['setting']==chosen['setting'] and selected['seed']==seed
            selection.append(dict(cohort=cohort,architecture=arch,**chosen))
    controls={'objective_controls':30,'history_controls':24,'population_controls':48,'graph_controls':6}
    for name,count in controls.items():
        assert len(list((BASE/name).rglob('metrics.json')))==count,(name,count)
    for status in (BASE/'checkpoints/ptr_v2').glob('run_matrix_status*.csv'):
        frame=pd.read_csv(status)
        assert frame.status.isin(['done','skipped_done']).all(),str(status)
    # Repair a reporting-only field from the persisted *raw* probabilities.
    # Earlier running processes computed this field after calibration. No
    # training, checkpoint selection or final comparison uses this metadata.
    corrections=[]
    for path in (BASE/'objective_controls').glob('*/*/seed*/metrics.json'):
        d=json.loads(path.read_text())
        for split in ('val','test'):
            raw=M.read_predictions(path.parent/f'pred_{split}_raw.csv')
            suite=M.MetricSuite(raw)
            value=suite.evaluate(np.ones(suite.n_matches),('auc_within_phase',))['auc_within_phase']
            before=d['metrics'][split]['within_phase_auc_raw']
            d['metrics'][split]['within_phase_auc_raw']=value
            corrections.append(dict(path=str(path.relative_to(REPO)),split=split,previous=before,raw_csv_value=value))
        d['raw_auc_metadata_source']='pred_*_raw.csv; corrected after execution, predictions and model selection unchanged'
        save(path,d)
    native=json.loads((BASE/'native_evaluation/producer_check.json').read_text())
    assert len(native)==11 and max(x['max_abs_delta'] for x in native)<2e-6
    hashes={}
    for suite in suites:
        j=json.loads((BASE/'control_evaluation'/suite/'complete.json').read_text())
        for rec in j['files']:
            path=REPO/rec['path']
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==rec['sha256'],str(path)
            hashes[rec['path']]=digest
    save(BASE/'final_verification.json',dict(required_complete=required,selections=selection,control_counts=controls,
         prediction_hashes=hashes,native_producer_max_delta=max(x['max_abs_delta'] for x in native),raw_auc_metadata_corrections=corrections,
         note='All choices above are checked against validation results and the frozen search space; no test-driven reselection.'))
    print('Completion, selection, producer parity and prediction provenance verified.',flush=True)

if __name__=='__main__':main()
