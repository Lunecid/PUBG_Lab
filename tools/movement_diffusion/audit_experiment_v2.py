"""Leakage audit independent of training; all outputs local, no DB mutation."""
import copy,csv,hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset import TrajectoryDataset,DEFAULT
from movement_diffusion.bank import build_bank,read_raw
OUT=ROOT/'reports/heterograph_diffusion_experiment_v2'
def main():
 start=time.time();ds=TrajectoryDataset();val=TrajectoryDataset(split='validation');checks={};rows=list(ds.rows.values());trainids={r['match_id'] for r in rows if r['split']=='train'};valids={r['match_id'] for r in rows if r['split']=='validation'}
 assert len(trainids)==751 and len(valids)==188 and trainids.isdisjoint(valids);checks['match_split']={'train':len(trainids),'validation':len(valids),'intersection':0}
 expected=json.loads((DEFAULT/'bank_sha256.json').read_text());hashtest=0;valid_frames=0;label_error=0.;label_n=0;asof_n=0;postdeath=0;ambiguous=0
 for row in rows:
  path=DEFAULT/'banks'/f"{row['match_id']}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[row['match_id']];hashtest+=1
  with np.load(path) as b:
   tt=b['times'][:,None];has=b['source_time']>=0
   assert (b['source_time'][has]<=np.broadcast_to(tt,has.shape)[has]).all();assert (b['game_time'][b['game_time']>=0]<=b['times'][b['game_time']>=0]).all();asof_n+=int(has.sum())
   assert np.isfinite(b['x']).all() and not b['x'][~b['mask']].any()
   err=np.abs(b['target_time']-tt)[b['target_valid']];assert (err<=5).all();label_error+=float(err.sum());label_n+=len(err);valid_frames+=int(b['valid'].sum());postdeath+=int(b['quality'][2]);ambiguous+=int(((b['x'][:,:,17]>0)&b['valid']).sum())
 checks['bank_integrity']={'banks_sha256_matched':hashtest,'position_asof_checks':asof_n,'valid_ground_frames':valid_frames,'future_label_mean_abs_clock_error_s':label_error/label_n,'post_first_death_positive_rows_excluded':postdeath,'equipment_ambiguous_valid_frames':ambiguous}
 totals=defaultdict(lambda:[0,0]);files={p.stem for p in (DEFAULT/'spatial_match').glob('*.npz')};assert files==trainids
 for mid in sorted(trainids):
  with np.load(DEFAULT/'spatial_match'/f'{mid}.npz') as b:
   for a,c,mode,count in b['edges']:
    key=(int(a),int(c),int(mode));totals[key][0]+=int(count);totals[key][1]+=1
 with np.load(DEFAULT/'spatial_prior.npz') as b:
  actual={tuple(map(int,r[:3])):list(map(int,r[3:])) for r in b['edges']};assert actual==dict(totals)
 checks['spatial_prior']={'training_match_files':len(files),'validation_files':len(files&valids),'all_aggregate_edge_counts_and_support_exact':True,'edges':len(totals),'own_match_subtraction':'covered by contract test'}
 prefix_checks=[]
 for mi in np.unique(ds.index[:,0])[np.linspace(0,750,5).astype(int)]:
  mid=ds.rows[int(mi)]['match_id'];raw=read_raw(DEFAULT/'raw'/f'{mid}.json.gz');full=build_bank(raw,ds.builder.vocab)
  for t in [90,300,600]:
   pre=copy.deepcopy(raw);pre['tables']={k:[r for r in events if r['t']<=t] for k,events in raw['tables'].items()};short=build_bank(pre,ds.builder.vocab);n=len(short['times'])
   for key in ('x','mask','valid','source_time','items','item_seen','vehicle','game','game_mask','game_time'):np.testing.assert_array_equal(short[key],full[key][:n])
   prefix_checks.append({'match_id':mid,'cutoff_s':t})
 checks['causal_prefix_reconstruction']=prefix_checks
 plan_checks={};union=set()
 for path in sorted(OUT.glob('training_indices_seed*.npy')):
  plan=np.load(path);mids={ds.rows[int(i)]['match_id'] for i in np.unique(ds.index[plan.flatten(),0])};assert not mids&valids;union.update(mids);plan_checks[path.name]={'matches':len(mids),'validation_overlap':0,'draws':int(plan.size),'sha256':hashlib.sha256(plan.tobytes()).hexdigest()}
 checks['training_plans']=plan_checks;checks['union_training_matches']=len(union)
 indices=np.load(OUT/'evaluation_indices.npy');mids={val.rows[int(i)]['match_id'] for i in np.unique(val.index[indices,0])};assert mids==valids;checks['evaluation_plan']={'windows':len(indices),'matches':len(mids),'training_overlap':len(mids&trainids),'all_eligible_have_some_future_label':bool((val.index[indices,4]>0).all()),'complete_joint60s_windows':int(val.index[indices,5].sum()),'humans':int((val.index[indices,3]==1).sum()),'bots':int((val.index[indices,3]==2).sum())}
 paired=0
 for index in indices[::100]:
  a=val[int(index)];server=TrajectoryDataset(split='validation',view='server')[int(index)]
  assert a['meta']['sample_id']==server['meta']['sample_id']
  for key in a['targets']:np.testing.assert_array_equal(a['targets'][key],server['targets'][key])
  for key in ('initial_xy','velocity','position_age_s','current_presence'):np.testing.assert_array_equal(a['conditions'][key],server['conditions'][key])
  paired+=1
 checks['paired_view_spot_checks']=paired
 report={'status':'IMPLEMENTATION_CHECKS_PASSED_WITH_SEMANTIC_LIMITATIONS','checks':checks,'remaining_limitations':[
  'Observer is a conservative telemetry/HUD proxy, not verified actual on-screen perception. Safe/poison circle announcement timing is not independently verified.',
  'Server means all exported telemetry participants, not engine-complete world state; all first-death-censored positions omit recall episodes.',
  'Match-disjoint, not account-disjoint or chronological. Pseudonymized source has no persistent account IDs for overlap audit.',
  'Missing future coordinates are excluded from loss/evaluation; population is conditional on recorded future survival/availability. Missing noisy latent coordinates are completed by zero CV residual during training, a modeling assumption.',
  'Targets are nearest telemetry points within plus/minus 5 seconds, not continuous true positions at exact requested times. Vehicle errors include sampling-clock noise.',
  'Terrain/land proxies use historical assets. Observed transitions establish recorded usage, not unobstructed geometry; unobserved/inferred links do not establish impossibility or safety.',
  'Evaluation validation matches were used in prior data-quality inspection, not checkpoint selection in this experiment. No untouched final test set.',
  'Three finite-budget seeds do not establish convergence; graph/zone benefits require retrained ablations and stronger learned deterministic baselines.'
 ],'elapsed_s':time.time()-start}
 (OUT/'leakage_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
