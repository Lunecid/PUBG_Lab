"""V4 leakage/exposure audit: only train/validation banks are opened."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset_v3 import TrajectoryDataset,DEFAULT
OUT=ROOT/'reports/heterograph_diffusion_experiment_v4';OLD=ROOT/'reports/heterograph_diffusion_experiment_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=list(csv.DictReader((DEFAULT/'matches.csv').open()));splits={s:{r['match_id'] for r in rows if r['split']==s} for s in ['train','validation','test']};by={int(r['match_index']):r for r in rows};assert [len(splits[s]) for s in splits]==[657,188,94]
 assert not any(splits[a]&splits[b] for a,b in [('train','validation'),('train','test'),('validation','test')]);assert not (DEFAULT/'test_index.npy').exists()
 for folder in ['banks','spatial_match']:assert not any((DEFAULT/folder/(mid+'.npz')).exists() for mid in splits['test'])
 try:TrajectoryDataset(split='test')
 except PermissionError:pass
 else:raise AssertionError('test loader opened')
 for file in ['evaluation_indices.npy','transport_audit_plan.json','evaluation_samples.json','evaluation_targets.npz','visualization_cases.json']:assert sha(OUT/file)==sha(OLD/file)
 oldaudit=json.loads((OLD/'leakage_audit.json').read_text());checked=0
 oldspec=json.loads((OLD/'experiment_spec.json').read_text())
 for source,digest in oldspec['source_sha256'].items():assert sha(ROOT/source)==digest
 for mid,digest in oldaudit['bank_sha256'].items():
  assert mid in splits['train']|splits['validation'];assert sha(DEFAULT/'banks'/(mid+'.npz'))==digest;checked+=1
 idx=np.load(DEFAULT/'train_index.npy');assert {by[int(i)]['match_id'] for i in np.unique(idx[:,0])}==splits['train'];md=np.load(OUT/'train_sampling_metadata.npz');coverage={}
 for seed in [41,42,43]:
  plan=np.load(OUT/f'training_indices_seed{seed}.npy');assert plan.shape==(2000,8);assert (idx[plan,4]>0).all();ids=plan.ravel();rng=np.random.default_rng(seed+20000);residual_ids=ids[rng.integers(len(ids),size=(4000,32)).ravel()];coverage[str(seed)]={}
  for stage,selected in [('regression',ids),('residual',residual_ids)]:
   masks={'pickup':md['mode'][selected]==4,'redeploy':md['mode'][selected]==5,'reentry_episode':md['episode'][selected]>0};coverage[str(seed)][stage]={g:{'draws':int(mask.sum()),'unique_windows':len(np.unique(selected[mask])),'matches':len(np.unique(idx[selected[mask],0]))} for g,mask in masks.items()}
 doc={'split_counts':{s:len(v) for s,v in splits.items()},'test_loading_blocked':True,'test_banks':0,'test_predictions':0,'historical_v2_test_training_exposure':True,'all_v3_train_validation_banks_sha256_unchanged':checked,'primary_and_transport_plans_targets_and_cases_byte_identical':True,'inputs_asof':'V3 reconstruction and graph inputs unchanged; V3 real-prefix audit and V4 tests cover future target mutation/private feature exclusion. No new raw-event reconstruction.', 'residual_normalization':'Only fit-cache residuals, separately per view/seed. No validation or test normalization. Frozen mean receives no residual gradients.','future_masks':'Used only in supervised losses/evaluation and fit-label eligibility; never conditioning. Complete-current-member knots required for residual center definition.','rare_draws':'Uniform match then window, replacement. More presentations are not new independent trajectories.','coverage':coverage}
 (OUT/'leakage_audit.json').write_text(json.dumps(doc,indent=2));print(json.dumps({k:v for k,v in doc.items() if k!='coverage'},indent=2))
 tp=json.loads((OUT/'transport_audit_plan.json').read_text())['plan'];selected=[]
 for group in ['emergency_pickup','redeploy_aircraft']:
  ids=[i for i,x in enumerate(tp) if x['group']==group];number=ids[len(ids)//2];selected.append({'transport_position':number,**tp[number],'selection':'Middle position in frozen V3 transport plan within mode; fixed before V4 predictions, independent of errors.'})
 path=OUT/'visual_transport_plan.json'
 if path.exists():assert json.loads(path.read_text())==selected
 else:path.write_text(json.dumps(selected,indent=2))
if __name__=='__main__':main()
