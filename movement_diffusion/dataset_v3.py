"""V3 paired views with causal episode/mobility and direct circle geometry."""
from pathlib import Path
import numpy as np
from .dataset import TrajectoryDataset as Previous,GraphBuilder as PreviousBuilder,collate as previous_collate,NODE_TYPES
DEFAULT=Path(__file__).resolve().parents[1]/'data/processed/heterograph_diffusion_v3'

def extra(b,k,p):
 return np.array([b['mobility'][k,p]/9,b['reentry_confidence'][k,p]/2,min(b['episode'][k,p],4)/4,min(max(b['times'][k]-b['episode_start'][k,p],0),600)/600,b['z'][k,p]/1000,b['vz'][k,p]/50,float(b['mobility_known'][k,p])],np.float32)
class GraphBuilder(PreviousBuilder):
 def graph(self,b,k,ego,origin,mid,train,view,target_actors):
  g=super().graph(b,k,ego,origin,mid,train,view,target_actors);actors=list(range(len(b['team']))) if view=='server' else np.flatnonzero(b['team']==b['team'][ego]).tolist()
  for nid,p in zip(np.flatnonzero(g['node_type']==NODE_TYPES.index('player')),actors):
   allowed=view=='server' or p==ego
   g['x'][nid,25:32]=extra(b,k,p) if allowed else 0;g['feature_mask'][nid,25:32]=bool(allowed and b['valid'][k,p])
  return g
class TrajectoryDataset(Previous):
 def __init__(self,root=DEFAULT,split='train',view='observer'):
  if split=='test':raise PermissionError('V3 final test is locked; no loading, fitting or evaluation permitted in this experiment')
  super().__init__(root,split,view);self.builder=GraphBuilder(self.root)
 def __getitem__(self,i):
  s=super().__getitem__(i);mi,k,ego,*_=self.index[i];mid=self.rows[int(mi)]['match_id'];b=self.bank(mid);members=np.flatnonzero((b['team']==b['team'][ego])&b['valid'][k]);members=[int(ego)]+[int(p) for p in members if p!=ego];t=s['targets'];c=s['conditions'];zones=np.zeros((4,14),np.float32);state=np.zeros((4,14),np.float32);history=np.zeros((4,4,6),np.float32)
  for slot,p in enumerate(members):
   for f in range(6):
    j=k+f+1
    if j>=len(b['times']):continue
    t['trajectory_valid'][slot,f]&=b['target_episode'][j,p]==b['episode'][k,p];t['mode'][slot,f]=b['target_mode'][j,p];t['mode_valid'][slot,f]=t['trajectory_valid'][slot,f] and t['mode'][slot,f]>0
   allowed=self.view=='server' or p==ego;state[slot,:7]=extra(b,k,p) if allowed else 0;state[slot,7:]=allowed
   for h,j in enumerate(range(k-3,k+1)):
    if b['valid'][j,p] and b['episode'][j,p]==b['episode'][k,p]:history[slot,h]=[*(b['x'][j,p,:2]-s['meta']['origin_m'])/1000,*np.where(b['mask'][j,p,2:4],b['x'][j,p,2:4],0)/30,b['x'][j,p,5]/15,1]
   for z,start in enumerate([3,6]):
    if b['game_mask'][k,start:start+3].all() and b['game'][k,start+2]>0:
     delta=(b['game'][k,start:start+2]-b['x'][k,p,:2])/1000;radius=b['game'][k,start+2]/1000;dist=np.linalg.norm(delta);speed=np.dot(b['x'][k,p,2:4],delta/max(dist,1e-6))/30
     zones[slot,z*7:(z+1)*7]=[*delta,radius,dist-radius,speed,float(dist<=radius),1]
  t['trajectory'][~t['trajectory_valid']]=0;t['team_valid']=t['trajectory_valid'][c['current_presence']].all(0);t['team_center']=t['trajectory'][c['current_presence']].mean(0);t['team_center'][~t['team_valid']]=0
  c.update(zone_geometry=zones,mobility_state=state,motion_history=history)
  s['meta'].update(episode=int(b['episode'][k,ego]),mobility=int(b['mobility'][k,ego]),reentry_confidence=int(b['reentry_confidence'][k,ego]),current_members=len(members))
  return s

def collate(samples):
 import torch
 b=previous_collate(samples)
 for k in ['zone_geometry','mobility_state','motion_history']:b['conditions'][k]=torch.from_numpy(np.stack([s['conditions'][k] for s in samples]))
 return b
