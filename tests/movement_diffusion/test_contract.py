import copy
from pathlib import Path
import numpy as np
from movement_diffusion.bank import build_bank,read_raw,PRIVATE
from movement_diffusion.dataset import TrajectoryDataset,NODE_TYPES,BASE_REL
ROOT=Path(__file__).resolve().parents[2]/'data/processed/heterograph_diffusion_v1'

def datasets():return TrajectoryDataset(ROOT,'train','observer'),TrajectoryDataset(ROOT,'train','server')
def graph_equal(a,b):
 for x,y in zip(a,b):
  for key in ('x','feature_mask','node_type','category','edge_index','edge_type','edge_attr','edge_mask','target_nodes'):np.testing.assert_array_equal(x[key],y[key])

def test_paired_targets_and_no_future_masks_in_conditions():
 obs,srv=datasets()
 for i in [0,len(obs)//2,len(obs)-1]:
  a,b=obs[i],srv[i];assert a['meta']['sample_id']==b['meta']['sample_id']
  for key in a['targets']:np.testing.assert_array_equal(a['targets'][key],b['targets'][key])
  assert not any('future' in key or 'target' in key for key in a['conditions'])
  for g in a['conditions']['graphs']:
   assert g['time_s']<=a['meta']['anchor_s'];assert g['edge_index'].max()<len(g['x']);assert np.isfinite(g['x']).all()

def test_causal_prefix_state_reconstruction():
 ds,_=datasets();r=ds.rows[int(ds.index[0,0])];raw=read_raw(ROOT/'raw'/f"{r['match_id']}.json.gz");full=build_bank(raw,ds.builder.vocab)
 for t in [90,300,600]:
  prefix=copy.deepcopy(raw);prefix['tables']={k:[r for r in rows if r['t']<=t] for k,rows in raw['tables'].items()};short=build_bank(prefix,ds.builder.vocab);n=len(short['times'])
  for key in ('x','mask','valid','source_time','items','item_seen','vehicle','game','game_mask','game_time'):np.testing.assert_array_equal(short[key],full[key][:n])

def test_hidden_state_noninterference():
 ds,_=datasets();mi,k,ego,*_=ds.index[len(ds)//2];r=ds.rows[int(mi)];b=ds.bank(r['match_id']);origin=b['x'][k,ego,:2];slots=np.array([ego,-1,-1,-1]);args=(int(k),int(ego),origin,r['match_id'],True)
 baseline=[ds.builder.graph(b,*args,'observer',slots)];oracle=ds.builder.graph(b,*args,'server',slots);changed={key:value.copy() for key,value in b.items()};others=np.arange(len(b['team']))!=ego
 changed['items'][:,others]=1;changed['vehicle'][:,others]=1
 for col in PRIVATE:changed['x'][:,others,col]=123
 changed['x'][:,:,13]=321;changed['game'][:,2]=99;changed['damage'][:,4:6]=4000
 hidden_enemy=others&(b['team']!=b['team'][ego]);changed['x'][:,hidden_enemy,:2]=6500
 after=[ds.builder.graph(changed,*args,'observer',slots)];graph_equal(baseline,after);serverafter=ds.builder.graph(changed,*args,'server',slots);assert not np.array_equal(oracle['x'],serverafter['x']) or not np.array_equal(oracle['category'],serverafter['category'])

def test_train_only_prior_and_leave_one_match_out():
 ds,_=datasets();r=ds.rows[int(ds.index[0,0])];sub=ds.builder.spatial.local(r['match_id'],True)
 assert sub
 for (a,b,m),count in list(sub.items())[:30]:
  original=next(e for e in ds.builder.spatial.out[a] if e[0]==b and e[1]==m);remaining=next((e for e in ds.builder.spatial.edges(a,sub) if e[0]==b and e[1]==m),None)
  if remaining:assert remaining[2]==original[2]-count and remaining[3]==original[3]-1
  else:assert original[2]==count
 val=TrajectoryDataset(ROOT,'validation');assert set(ds.rows[int(x)]['match_id'] for x in np.unique(ds.index[:,0])).isdisjoint(val.rows[int(x)]['match_id'] for x in np.unique(val.index[:,0]))

def test_fixed_team_targets_and_target_clock():
 ds,_=datasets()
 for i in [0,len(ds)//3,len(ds)*2//3]:
  s=ds[i];p=s['conditions']['current_presence'];target=s['targets'];valid=target['trajectory_valid'];np.testing.assert_array_equal(target['team_valid'],valid[p].all(0));np.testing.assert_allclose(target['team_center'][target['team_valid']],target['trajectory'][p].mean(0)[target['team_valid']]);assert (abs(target['time_error_s'][valid])<=5).all()

def test_equipment_ties_are_order_invariant():
 ds,_=datasets();r=ds.rows[int(ds.index[0,0])];raw=read_raw(ROOT/'raw'/f"{r['match_id']}.json.gz");a=build_bank(raw,ds.builder.vocab);raw['tables']['item_equip'].reverse();b=build_bank(raw,ds.builder.vocab);np.testing.assert_array_equal(a['items'],b['items']);np.testing.assert_array_equal(a['x'][:,:,17],b['x'][:,:,17])

def test_perfect_joint_predictions_have_zero_error():
 from movement_diffusion.evaluation import metrics
 ds,_=datasets();s=ds[len(ds)//3];pred=np.repeat(s['targets']['trajectory'][None],3,axis=0);m=metrics(pred,s['targets'],s['conditions']['current_presence']);assert m and all(abs(v)<1e-3 for v in m.values())

def test_future_target_changes_do_not_change_conditions():
 ds,_=datasets();row=ds.index[len(ds)//3];mi,k,ego,*_=row;mid=ds.rows[int(mi)]['match_id'];bank=ds.bank(mid);a=ds[len(ds)//3]
 originals={key:bank[key].copy() for key in ('target','target_valid','target_time','target_vehicle')}
 try:
  bank['target'][k+1:]+=1000;bank['target_valid'][k+1:]=False;bank['target_time'][k+1:]=-1;bank['target_vehicle'][k+1:]=1;b=ds[len(ds)//3];graph_equal(a['conditions']['graphs'],b['conditions']['graphs']);np.testing.assert_array_equal(a['conditions']['current_presence'],b['conditions']['current_presence'])
 finally:
  bank.update(originals)
