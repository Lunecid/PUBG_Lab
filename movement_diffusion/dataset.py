"""Paired views share sample IDs and future labels; graph construction is as-of only."""
import csv,json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from .bank import FEATURES,PRIVATE
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/processed/heterograph_diffusion_v1'
NODE_TYPES=['cell','region','player','team','vehicle_evidence','contact_evidence','zone','game','equipment']
BASE_REL=['member_of','located_in','occupies','within_region','region_adjacent_uncertain','observed_nonvehicle','observed_vehicle','inferred_candidate','vehicle_for','contact_for','zone_conditions','game_conditions','owns_equipment','recent_damage_from']
RELATIONS=[s for r in BASE_REL for s in (r,'reverse_'+r)]
WIDTH=32;EDGE_WIDTH=5
SCALE=np.array([1000,1000,30,30,100,15,1,300,100,100,100,100,100,100,1,300,300,4,20,20,20,1],np.float32)

class SpatialPrior:
 def __init__(self,root):
  d=np.load(root/'spatial_prior.npz');self.terrain=d['terrain'];self.out=defaultdict(list)
  for a,b,mode,count,support in d['edges']:self.out[int(a)].append((int(b),int(mode),int(count),int(support)))
  self.root=root;self.region_cells=defaultdict(list)
  for cell in range(6724):self.region_cells[(cell//82//4)*21+(cell%82//4)].append(cell)
 @lru_cache(maxsize=2)
 def local(self,mid,train):
  if not train:return {}
  d=np.load(self.root/'spatial_match'/f'{mid}.npz');return {(int(a),int(b),int(m)):int(c) for a,b,m,c in d['edges']}
 def edges(self,cell,subtract):
  return [(b,m,c-subtract.get((cell,b,m),0),s-int((cell,b,m) in subtract)) for b,m,c,s in self.out[cell] if c-subtract.get((cell,b,m),0)>0]

class GraphBuilder:
 def __init__(self,root):
  self.root=Path(root);self.vocab=json.loads((self.root/'vocabulary.json').read_text());self.spatial=SpatialPrior(self.root);self.item_count=len(self.vocab['items'])+2
 def graph(self,b,k,ego,origin,mid,train,view,target_actors):
  server=view=='server';ownteam=b['team'][ego];actors=list(range(len(b['team']))) if server else np.flatnonzero(b['team']==ownteam).tolist();nodes=[];nt=[];fm=[];cat=[];ids={};es=[];er=[];ea=[];em=[];t=float(b['times'][k]);subtract=self.spatial.local(mid,train)
  def node(key,typ,values,mask=None,category=0):
   if key in ids:return ids[key]
   values=np.asarray(values,np.float32);ok=np.ones(len(values),bool) if mask is None else np.asarray(mask,bool);v=np.zeros(WIDTH,np.float32);m=np.zeros(WIDTH,bool);v[:len(values)]=np.where(ok,values,0);m[:len(values)]=ok
   assert np.isfinite(v).all();i=len(nodes);ids[key]=i;nodes.append(v);fm.append(m);nt.append(NODE_TYPES.index(typ));cat.append(category);return i
  def edge(a,c,relation,attrs=(0,1,0,0,0),attr_mask=(True,True,True,True,True)):
   rid=BASE_REL.index(relation)*2;es.extend([(a,c),(c,a)]);er.extend([rid,rid+1]);ea.extend([attrs,attrs]);em.extend([attr_mask,attr_mask])
  gamevals=[t/1800,b['game'][k,0]/10,b['game'][k,1]/100,b['game'][k,2]/40,max(0,t-b['game_time'][k])/15];gamemask=[True,*b['game_mask'][k,:3],b['game_time'][k]>=0]
  if not server:gamemask[3]=False
  game=node(('game',0),'game',gamevals,gamemask)
  for reg,cells in self.spatial.region_cells.items():
   a=self.spatial.terrain[cells];xy=a[:,:2].mean(0);node(('region',reg),'region',[*(xy-origin)/1000,a[:,2].mean()/100,a[:,3].mean(),a[:,4].mean(),1])
  for reg in self.spatial.region_cells:
   for other in (reg+1,reg+21):
    if other in self.spatial.region_cells and (other-reg==21 or other//21==reg//21):edge(ids['region',reg],ids['region',other],'region_adjacent_uncertain')
  fine=set()
  # Exactly the same crop in paired views: own current team positions, never future route.
  for p in np.flatnonzero((b['team']==ownteam)&b['valid'][k]):
   cx,cy=(b['x'][k,p,:2]//100).astype(int)
   for dy in range(-2,3):
    for dx in range(-2,3):
     if 0<=cx+dx<82 and 0<=cy+dy<82:fine.add((cy+dy)*82+cx+dx)
  for c in sorted(fine):
   a=self.spatial.terrain[c];out=self.spatial.edges(c,subtract);support=sum(e[3] for e in out);i=node(('cell',c),'cell',[*(a[:2]-origin)/1000,a[2]/100,a[3],a[4],np.log1p(support)/10,float(not out),1]);reg=c//82//4*21+c%82//4;edge(i,ids['region',reg],'within_region')
  for c in sorted(fine):
   observed=self.spatial.edges(c,subtract);obs={(z[0],z[1]) for z in observed}
   for other,mode,count,support in observed:
    if other in fine:edge(ids['cell',c],ids['cell',other],'observed_vehicle' if mode else 'observed_nonvehicle',(np.log1p(support)/10,1,0,0,0))
   for other in (c-82,c+82,c-1,c+1):
    if other not in fine or abs(c%82-other%82)+abs(c//82-other//82)!=1 or (other,0) in obs:continue
    a,bb=self.spatial.terrain[c],self.spatial.terrain[other]
    if min(a[3],bb[3])>=.6 and abs(a[2]-bb[2])/100<=.3:edge(ids['cell',c],ids['cell',other],'inferred_candidate')
  teams=sorted({int(b['team'][p]) for p in actors})
  for team in teams:
   members=np.flatnonzero(b['team']==team);v=members[b['valid'][k,members]];vals=np.zeros(12);m=np.zeros(12,bool);vals[8:]=[len(v)/4,len(members)/4,np.sum((b['x'][k,members,6]==0)&b['mask'][k,members,6])/4,1];m[8:]=True
   if len(v):
    xy=b['x'][k,v,:2];ages=b['x'][k,v,5];center=np.average(xy,axis=0,weights=np.exp(-ages/10));delta=xy-center;dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);med=xy[np.argmin(dist.sum(1))];vals[:8]=[*(center-origin)/1000,*(med-origin)/1000,np.linalg.norm(delta,axis=1).max()/1000,dist.max()/1000,float(np.mean(ages))/15,float(np.mean(dist))/1000];m[:8]=True
   teamnode=node(('team',team),'team',vals,m);edge(game,teamnode,'game_conditions')
   for c in sorted({int(b['x'][k,p,1]//100)*82+int(b['x'][k,p,0]//100) for p in v}):
    if c in fine:edge(teamnode,ids['cell',c],'occupies')
  for p in actors:
   xx=b['x'][k,p].copy();mm=b['mask'][k,p].copy();xx[:2]-=origin;xx/=SCALE;own=p==ego
   if not server and not own:mm[PRIVATE]=False
   if not server:mm[13]=False
   vals=[*xx,float(own),float(b['team'][p]==ownteam),float(b['item_seen'][k,p])];m=[*mm,True,True,bool(server or own)]
   pn=node(('player',p),'player',vals,m);edge(pn,ids['team',int(b['team'][p])],'member_of')
   if b['valid'][k,p]:
    cell=int(b['x'][k,p,1]//100)*82+int(b['x'][k,p,0]//100);reg=cell//82//4*21+cell%82//4;edge(pn,ids['region',reg],'located_in')
    if cell in fine:edge(pn,ids['cell',cell],'located_in')
   if server or own:
    if b['mask'][k,p,14] and b['x'][k,p,14]>0:
     vn=node(('vehicle',p),'vehicle_evidence',[b['x'][k,p,5]/15,0,0],[True,False,False],self.item_count+int(b['vehicle'][k,p]));edge(vn,pn,'vehicle_for')
    if b['x'][k,p,6]>0:
     for slot,code in enumerate(b['items'][k,p]):
      if code:
       en=node(('item',p,slot),'equipment',[b['x'][k,p,15]/300,float(b['x'][k,p,17]>0)],[True,True],int(code));edge(en,pn,'owns_equipment')
    if np.sum(b['x'][k,p,[8,9,10,11,12,20]])>0:
     cn=node(('contact',p),'contact_evidence',[*b['x'][k,p,8:13]/100,b['x'][k,p,16]/300,b['x'][k,p,20]/20,0,0],[True]*7+[False,False]);edge(cn,pn,'contact_for')
  if server:
   for r in b['damage']:
    et,v,a,d,x,y=r
    if t-30<et<=t and a>=0 and d>0:
     edge(ids['player',int(a)],ids['player',int(v)],'recent_damage_from',(np.log1p(d)/10,0,(t-et)/30,(x-origin[0])/1000 if np.isfinite(x) else 0,(y-origin[1])/1000 if np.isfinite(y) else 0),(True,True,True,bool(np.isfinite(x)),bool(np.isfinite(y))))
  for z,(start,name) in enumerate([(3,'safe'),(6,'poison')]):
   v=b['game'][k,start:start+3];ok=b['game_mask'][k,start:start+3]
   if ok.all() and v[2]>0:
    zn=node(('zone',z),'zone',[*(v[:2]-origin)/1000,v[2]/1000,(t-b['game_time'][k])/15,float(z)]);edge(zn,ids['team',int(ownteam)],'zone_conditions');edge(zn,ids['player',ego],'zone_conditions')
  return {'x':np.stack(nodes),'feature_mask':np.stack(fm),'node_type':np.array(nt,np.int64),'category':np.array(cat,np.int64),'edge_index':np.array(es,np.int64).T.reshape(2,-1),'edge_type':np.array(er,np.int64),'edge_attr':np.array(ea,np.float32).reshape(-1,EDGE_WIDTH),'edge_mask':np.array(em,bool).reshape(-1,EDGE_WIDTH),'target_nodes':np.array([ids['player',int(p)] if p>=0 else -1 for p in target_actors],np.int64),'team_node':ids['team',int(ownteam)],'time_s':t}

class TrajectoryDataset:
 def __init__(self,root=DEFAULT,split='train',view='observer'):
  self.root=Path(root);assert view in ('observer','server');assert split in ('train','validation');self.split=split;self.view=view;self.index=np.load(self.root/f'{split}_index.npy',mmap_mode='r');self.rows={int(r['match_index']):r for r in csv.DictReader((self.root/'matches.csv').open())};self.builder=GraphBuilder(self.root)
 def __len__(self):return len(self.index)
 @lru_cache(maxsize=2)
 def bank(self,mid):
  with np.load(self.root/'banks'/f'{mid}.npz') as d:return {k:d[k] for k in d.files}
 def __getitem__(self,i):
  mi,k,ego,kind,has,complete,n=self.index[i].tolist();row=self.rows[mi];mid=row['match_id'];b=self.bank(mid);origin=b['x'][k,ego,:2].copy();members=np.flatnonzero((b['team']==b['team'][ego])&b['valid'][k]);members=[ego]+[int(p) for p in members if p!=ego];slots=np.array(members+[-1]*(4-len(members)),np.int64);presence=slots>=0;graphs=[self.builder.graph(b,j,ego,origin,mid,self.split=='train',self.view,slots) for j in range(k-3,k+1)]
  y=np.zeros((4,6,2),np.float32);valid=np.zeros((4,6),bool);dt=np.zeros((4,6),np.float32);mode=np.zeros((4,6),np.int64);mode_mask=valid.copy()
  for p,a in enumerate(slots):
   if a<0:continue
   for f in range(6):
    j=k+f+1
    if j>=len(b['times']):continue
    ok=bool(b['target_valid'][j,a] and b['target_time'][j,a]>b['times'][k]);valid[p,f]=ok
    if ok:y[p,f]=(b['target'][j,a]-origin)/1000;dt[p,f]=b['target_time'][j,a]-b['times'][j]
    mode[p,f]=int(b['target_vehicle'][j,a]);mode_mask[p,f]=ok
  joint=valid[presence].all(0);center=y[presence].mean(0);center[~joint]=0;initial=np.zeros((4,2),np.float32);velocity=initial.copy();age=np.zeros(4,np.float32)
  for p,a in enumerate(slots):
   if a>=0:initial[p]=(b['x'][k,a,:2]-origin)/1000;velocity[p]=np.where(b['mask'][k,a,2:4],b['x'][k,a,2:4],0)/1000;age[p]=b['x'][k,a,5]
  return {'conditions':{'graphs':graphs,'current_presence':presence,'initial_xy':initial,'velocity':velocity,'position_age_s':age},'targets':{'trajectory':y,'trajectory_valid':valid,'team_center':center,'team_valid':joint,'mode':mode,'mode_valid':mode_mask,'time_error_s':dt},'meta':{'sample_id':f'{mid}:{k*10}:{ego}','match_id':mid,'anchor_s':k*10,'actor_kind':kind,'view':self.view,'origin_m':origin,'supervised':has,'complete_joint_60s':complete}}

def collate(samples):
 import torch
 graphs=[g for s in samples for g in s['conditions']['graphs']];batch={};offset=0;edges=[];tn=[];teams=[];node_graph=[]
 for j,g in enumerate(graphs):
  edges.append(g['edge_index']+offset);tn.append(np.where(g['target_nodes']>=0,g['target_nodes']+offset,-1));teams.append(g['team_node']+offset);node_graph.append(np.full(len(g['x']),j));offset+=len(g['x'])
 for key in ('x','feature_mask','node_type','category','edge_type','edge_attr','edge_mask'):batch[key]=torch.from_numpy(np.concatenate([g[key] for g in graphs]))
 batch.update(edge_index=torch.from_numpy(np.concatenate(edges,axis=1)),node_graph=torch.from_numpy(np.concatenate(node_graph)),target_nodes=torch.from_numpy(np.stack(tn).reshape(len(samples),4,4)),team_nodes=torch.tensor(teams).reshape(len(samples),4),batch_size=len(samples))
 for key in ('current_presence','initial_xy','velocity','position_age_s'):batch[key]=torch.from_numpy(np.stack([s['conditions'][key] for s in samples]))
 targets={key:torch.from_numpy(np.stack([s['targets'][key] for s in samples])) for key in samples[0]['targets']}
 return {'conditions':batch,'targets':targets,'meta':[s['meta'] for s in samples]}
