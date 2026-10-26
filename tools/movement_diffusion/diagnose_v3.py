"""Locked-plan no-zone retraining comparison, sensitivity, and exact replay."""
import json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset_v3 import TrajectoryDataset,collate,DEFAULT
from movement_diffusion.model_v3 import JointMotion
from movement_diffusion.evaluation import metrics,match_macro
from tools.movement_diffusion.experiment_v3 import hide_zone
OUT=ROOT/'reports/heterograph_diffusion_experiment_v3'
def read(n):return json.loads((OUT/n).read_text())
def write(n,d):(OUT/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);ids=np.load(OUT/'evaluation_indices.npy');meta=read('evaluation_samples.json');pred=dict(np.load(OUT/'predictions.npz'));spec=read('experiment_spec.json');groups=defaultdict(list)
 for i,m in enumerate(meta):groups[m['match_id']].append(i)
 rng=np.random.default_rng(20261018);chosen={int(rng.choice(v)) for v in groups.values()};checks=[];sensitivity={}
 for view in ['observer','server']:
  ds=TrajectoryDataset(split='validation',view=view);models={}
  for kind in ['regression','diffusion']:
   for seed in [41,42,43]:
    key=f'{view}_{kind}_{seed}';ck=torch.load(OUT/f'{key}.pt',map_location='cpu',weights_only=False);assert ck['steps']==2000 and ck['source_sha256']==spec['source_sha256'];assert ck['plan_sha256']==spec['plans'][str(seed)]['sha256'];m=JointMotion(len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4,kind);m.load_state_dict(ck['model']);m.eval();models[key]=(seed,m);sensitivity[key]=[];s=ds[int(ids[0])];b=collate([s]);torch.manual_seed(seed*100000);replay=m.sample(b['conditions'],20)[0].numpy();np.testing.assert_array_equal(replay,pred[key][0]);checks.append({'name':key,'steps':2000,'parameters':ck['parameters'],'exact_first_validation_sample_replay':True,'checkpoint_sha256':hashlib.sha256((OUT/f'{key}.pt').read_bytes()).hexdigest()})
  for number in sorted(chosen):
   s=ds[int(ids[number])];c=collate([s])['conditions'];masked=hide_zone(c);p=s['conditions']['current_presence']
   for key,(seed,m) in models.items():
    torch.manual_seed(seed*100000+number);without=m.sample(masked,20)[0].numpy();withzone=pred[key][number];shift=float(np.linalg.norm(withzone.mean(0)[p]-without.mean(0)[p],axis=-1).mean()*1000);ma=metrics(withzone,s['targets'],p);mb=metrics(without,s['targets'],p);sensitivity[key].append({'sample_id':s['meta']['sample_id'],'meanpath_shift_m':shift,'masked_minus_full_ego_ADE_m':mb.get('ego_meanpathADE_m',0)-ma.get('ego_meanpathADE_m',0)})
  print('SENSITIVITY',view,flush=True)
 write('condition_sensitivity.json',{'method':'One error-independent randomly selected window per188 development matches; paired same initial noise; direct geometry and graph zone numerical values masked, topology retained. Distribution shift diagnostic, not causal importance.','models':{k:{'mean_shift_m':float(np.mean([x['meanpath_shift_m'] for x in r])),'masked_minus_full_ego_ADE_m':float(np.mean([x['masked_minus_full_ego_ADE_m'] for x in r])),'records':r} for k,r in sensitivity.items()}})
 ds=TrajectoryDataset(split='validation',view='observer');ablation={}
 for kind in ['regression','diffusion']:
  key=f'observer_{kind}_41_nozone';ck=torch.load(OUT/f'{key}.pt',map_location='cpu',weights_only=False);assert ck['nozone'] and ck['steps']==2000;m=JointMotion(len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4,kind);m.load_state_dict(ck['model']);m.eval();records=[]
  for number,index in enumerate(ids):
   s=ds[int(index)];c=hide_zone(collate([s])['conditions']);torch.manual_seed(4100000+number);prediction=m.sample(c,20)[0].numpy();records.append({'sample_id':s['meta']['sample_id'],'match_id':s['meta']['match_id'],'metrics':metrics(prediction,s['targets'],s['conditions']['current_presence'])})
  ablation[kind]={'records':records,'match_macro':match_macro(records)};checks.append({'name':key,'steps':2000,'parameters':ck['parameters'],'plan_sha256':ck['plan_sha256'],'checkpoint_sha256':hashlib.sha256((OUT/f'{key}.pt').read_bytes()).hexdigest()});print('ABLATION_EVALUATED',kind,flush=True)
 write('zone_ablation.json',{'design':'Pre-specified one-seed41 retraining ablation: same model, plan, optimizer and budget; no direct circle geometry, no numeric graph circle features, no radial-distance auxiliary loss. Structural zone nodes and time remain. Exploratory, one seed.','models':ablation})
 target=dict(np.load(OUT/'evaluation_targets.npz'));v=target['trajectory_valid'];clock=target['time_error_s'];write('final_verification.json',{'completed_models':len(checks),'main_models':12,'ablation_models':2,'steps_per_model':2000,'main_training_window_presentations':12*2000*8,'ablation_training_window_presentations':2*2000*8,'evaluation_windows':len(ids),'evaluation_matches':len(groups),'test_predictions':0,'checkpoint_checks':checks,'target_mean_abs_clock_error_s':float(abs(clock[v]).mean()),'label_exact_time_fraction':float((abs(clock[v])<.001).mean()),'valid_target_points':int(v.sum()),'complete_joint_60s_windows':int(target['team_valid'].all(1).sum()),'full_training_rerun_performed':False})
 print('DIAGNOSTICS_COMPLETE',flush=True)
if __name__=='__main__':main()
