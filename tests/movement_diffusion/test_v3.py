import copy
import numpy as np
import pytest
import torch
from pubg_lab.movement.data.bank_v3 import actor_states, build_bank
from pubg_lab.movement.data.dataset_v3 import TrajectoryDataset, collate
from pubg_lab.movement.models.model_v3 import JointMotion

def fixture():
 tables={k:[] for k in ['positions','item_equip','item_pickup','item_use','damage','groggy','kills','parachute_landing','game_states']}
 for t in range(0,121,10):
  plane=40<=t<=60
  tables['positions'].append(dict(actor=0,t=t,x=100+t,y=100,z=1000 if plane else 10,hp=100,vehicle_type='TransportAircraft' if plane else None,vehicle_class='RedeployAircraft_Tiger_C' if plane else None))
 tables['kills']=[dict(actor=0,t=25)];tables['parachute_landing']=[dict(actor=0,t=0),dict(actor=0,t=80)]
 tables['item_equip']=[dict(actor=0,t=5,item='gun',category='Weapon',subcategory='Main',action='equip')]
 return dict(actors=[dict(team=0,kind=1)],tables=tables)

def test_reentry_asof_no_backfill_and_gear_reset():
 raw=fixture();b,_,_=build_bank(raw,{'items':{'gun':2},'vehicles':{'RedeployAircraft_Tiger_C':2}})
 assert not b['valid'][3,0] and b['episode'][3,0]==0
 assert b['valid'][4,0] and b['episode'][4,0]==1 and b['mobility'][4,0]==5
 assert b['mobility'][7,0]==7 and b['mobility'][8,0]==1
 assert b['items'][2,0].any() and not b['items'][4,0].any()
 assert not b['mask'][4,0,2] # never velocity across the life boundary

def test_prefix_input_invariance():
 raw=fixture();full=build_bank(raw,{'items':{},'vehicles':{}})[0]
 for cut in [30,50,90]:
  prefix=copy.deepcopy(raw)
  prefix['tables']={k:[r for r in rows if r['t']<=cut] for k,rows in raw['tables'].items()}
  b=build_bank(prefix,{'items':{},'vehicles':{}})[0]
  for key in ['x','mask','valid','items','episode','mobility','reentry_confidence','z','vz']:
   np.testing.assert_array_equal(b[key],full[key][:len(b['times'])])

def test_no_revival_from_single_positive_sample():
 r=fixture();r['tables']['positions']=r['tables']['positions'][:4]
 states,transitions,_=actor_states(r['tables'],0)
 assert not states[-1]['valid'] and not any(alive for _,alive,*_ in transitions)

def test_emergency_exit_is_airborne_until_landing():
 r=fixture();r['tables']['kills']=[];r['tables']['positions'][4].update(vehicle_type='EmergencyPickup',vehicle_class='BP_EmergencyPickupVehicle_C');r['tables']['positions']=r['tables']['positions'][:5]+[dict(r['tables']['positions'][7])]
 states,_,_=actor_states(r['tables'],0)
 assert states[-2]['mode']==4 and states[-1]['mode']==7

def test_test_split_locked():
 with pytest.raises(PermissionError):TrajectoryDataset(split='test')

def test_real_sample_masks_and_geometry():
 ds=TrajectoryDataset();sample=ds[0];mi,k,ego,*_=ds.index[0];b=ds.bank(ds.rows[int(mi)]['match_id']);assert sample['conditions']['zone_geometry'].shape==(4,14)
 for g in sample['conditions']['graphs']:assert np.isfinite(g['x']).all()
 assert not sample['conditions']['mobility_state'][1:,:7].any() # teammate private modes withheld

def test_models_shapes_gradients_and_condition_usage():
 torch.set_num_threads(2);ds=TrajectoryDataset();sample=ds[200];batch=collate([sample]);c,t=batch['conditions'],batch['targets'];categories=len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4
 for kind in ['regression','diffusion']:
  torch.manual_seed(4);m=JointMotion(categories,kind);loss,_=m.loss(c,t);assert torch.isfinite(loss);loss.backward();assert m.zone[0].weight.grad is not None
  pred=m.sample(c,3);assert pred.shape==(1,3 if kind=='diffusion' else 1,4,6,2);assert torch.isfinite(pred).all()
  # Linear midpoint structure is structural, not a soft postprocessing clamp.
  np.testing.assert_allclose(pred[0,0,:,0].numpy(),((c['initial_xy'][0]+c['velocity'][0]*c['position_age_s'][0,:,None]+pred[0,0,:,1])/2).numpy(),atol=1e-6)

def test_future_target_mutation_leaves_conditions():
 ds=TrajectoryDataset();a=ds[100];mi,k,ego,*_=ds.index[100];b=ds.bank(ds.rows[int(mi)]['match_id']);saved=b['target'].copy();b['target']+=1234
 try:
  changed=ds[100]
  for name in ['zone_geometry','mobility_state','motion_history','initial_xy']:np.testing.assert_array_equal(a['conditions'][name],changed['conditions'][name])
 finally:b['target'][:]=saved

def test_aircraft_and_stationary_mount_not_ground_vehicle():
 from pubg_lab.movement.data.bank_v3 import mode_of
 assert mode_of({'vehicle_class':'BP_Motorglider_C','vehicle_type':'FlyingVehicle'})==8
 assert mode_of({'vehicle_class':'MortarPawn_C','vehicle_type':'Vehicle'})==9
 assert mode_of({'vehicle_class':'BP_Rubber_boat_C','vehicle_type':None})==3

def test_observer_private_enemy_and_teammate_state_isolation():
 ds=TrajectoryDataset(view='observer');index=100;before=ds[index];mi,k,ego,*_=ds.index[index];b=ds.bank(ds.rows[int(mi)]['match_id']);others=np.arange(len(b['team']))!=ego;enemy=b['team']!=b['team'][ego];saved={key:b[key].copy() for key in ['x','items','item_seen','vehicle','mobility','reentry_confidence','z','vz']}
 try:
  b['x'][:,enemy,:2]+=1;b['x'][:,others,14]=1;b['items'][:,others]=1;b['item_seen'][:,others]=True;b['vehicle'][:,others]=1;b['mobility'][:,others]=8;b['reentry_confidence'][:,others]=1;b['z'][:,others]+=200;b['vz'][:,others]+=20
  after=ds[index]
  for a,d in zip(before['conditions']['graphs'],after['conditions']['graphs']):
   for name in ['x','feature_mask','edge_index','edge_attr','category']:np.testing.assert_array_equal(a[name],d[name])
  for name in ['zone_geometry','mobility_state','motion_history']:np.testing.assert_array_equal(before['conditions'][name],after['conditions'][name])
 finally:
  for key,a in saved.items():b[key][:]=a

def test_curve_anchor_accounts_for_position_age_without_future():
 m=JointMotion(150,'regression');c={'initial_xy':torch.zeros(1,4,2),'velocity':torch.ones(1,4,2)*.01,'position_age_s':torch.ones(1,4)*5};knots=torch.zeros(1,4,3,2)
 torch.testing.assert_close(m.curve(knots,c),torch.full((1,4,6,2),.05))

def test_residual_couples_members_and_remains_equivariant():
 torch.manual_seed(1);m=JointMotion(150,'diffusion');y=torch.randn(1,4,3,2);h=torch.randn(1,4,110);presence=torch.tensor([[True,True,True,False]]);step=torch.tensor([10]);a=m.denoise(y,step,h,presence);changed=y.clone();changed[:,1]+=2;d=m.denoise(changed,step,h,presence);assert not torch.allclose(a[:,0],d[:,0])
 order=[2,0,1,3];torch.testing.assert_close(m.denoise(y[:,order],step,h[:,order],presence[:,order]),a[:,order],atol=1e-6,rtol=1e-5)
 padded=y.clone();padded[:,3]+=100;torch.testing.assert_close(m.denoise(padded,step,h,presence)[:,:3],a[:,:3])

def test_ground_vehicle_clears_previous_air_exit_inference():
 r=fixture();r['tables']['kills']=[];r['tables']['parachute_landing']=[dict(actor=0,t=0)];positions=r['tables']['positions']
 for row in positions:row.update(vehicle_type=None,vehicle_class=None)
 positions[2].update(vehicle_type='FlyingVehicle',vehicle_class='BP_Motorglider_C')
 positions[4].update(vehicle_type='WheeledVehicle',vehicle_class='Dacia_A_01_v2_C')
 states,_,_=actor_states(r['tables'],0)
 assert states[2]['mode']==8 and states[3]['mode']==7 and states[4]['mode']==2 and states[5]['mode']==1
 assert states[5]['landed']
