"""Match-clustered paired summaries, reproducible scientific plots, and viewer data."""
import csv,json,sys,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset import TrajectoryDataset
OUT=ROOT/'reports/heterograph_diffusion_experiment_v2'
SEEDS=[41,42,43];MODELS=['constant_velocity','stationary','observer','server'];NAMES={'constant_velocity':'Constant velocity','stationary':'Last position','observer':'Observer diffusion','server':'Server diffusion'}
COLORS={'constant_velocity':'#5b6575','stationary':'#a5aab4','observer':'#147eaa','server':'#bf6821'}
PRIMARY=['ego_meanpathADE_m','team_meanpathADE_m','joint_energy_score_m']
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def match_values(records,key,subset=None):
 grouped=defaultdict(list)
 for i,r in enumerate(records):
  if (subset is None or subset[i]) and key in r['metrics']:grouped[r['match_id']].append(r['metrics'][key])
 return {m:float(np.mean(v)) for m,v in grouped.items()}
def summarize(records,subset=None):
 keys=sorted({k for rs in records.values() for r in rs[0] for k in r['metrics']});out={};vectors={}
 for model,replicas in records.items():
  out[model]={};vectors[model]={}
  for key in keys:
   values=[match_values(r,key,subset) for r in replicas];mids=sorted(set.intersection(*[set(v) for v in values]))
   if not mids:continue
   a=np.array([[v[m] for m in mids] for v in values]);mean=a.mean(0);rng=np.random.default_rng(710)
   boot=mean[rng.integers(len(mean),size=(3000,len(mean)))].mean(1)
   out[model][key]={'mean':float(mean.mean()),'ci95_match_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'matches':len(mean),'seed_means':a.mean(1).tolist(),'seed_std':float(a.mean(1).std(ddof=1)) if len(a)>1 else 0.}
   vectors[model][key]=dict(zip(mids,mean.tolist()))
 pairs={}
 for a,b in [('observer','constant_velocity'),('server','constant_velocity'),('server','observer')]:
  name=f'{a}_minus_{b}';pairs[name]={}
  for key in keys:
   if key not in vectors[a] or key not in vectors[b]:continue
   mids=sorted(vectors[a][key].keys()&vectors[b][key].keys());delta=np.array([vectors[a][key][m]-vectors[b][key][m] for m in mids]);rng=np.random.default_rng(711);boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1)
   pairs[name][key]={'difference_m':float(delta.mean()),'ci95_paired_match_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'matches':len(mids)}
   if key in PRIMARY:pairs[name][key]['ci_familywise95_bonferroni_9']=np.quantile(boot,[.05/18,1-.05/18]).tolist()
 return {'models':out,'paired_differences':pairs}

def plots(summary):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
 keys=PRIMARY+['ego_expectedADE_m','team_expectedADE_m','pair_distance_MAE_m'];labels=['Individual mean-path ADE','Team-centroid mean-path ADE','Joint energy score','Individual expected ADE','Team-centroid expected ADE','Pair-distance MAE']
 fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for ax,key,label in zip(axes.ravel(),keys,labels):
  for i,model in enumerate(MODELS):
   row=summary['models'][model][key];lo,hi=row['ci95_match_bootstrap'];ax.errorbar(row['mean'],i,xerr=[[row['mean']-lo],[hi-row['mean']]],color=COLORS[model],fmt='o',capsize=4,ms=6);ax.text(hi+3,i,f"{row['mean']:.1f}",va='center',fontsize=9)
  ax.set_yticks(range(4),[NAMES[m] for m in MODELS]);ax.invert_yaxis();ax.set_xlim(0,max(summary['models'][m][key]['ci95_match_bootstrap'][1] for m in MODELS)*1.25+5);ax.set_xlabel('meters; lower is better');ax.set_title(label,loc='left');ax.grid(axis='x',alpha=.2)
 fig.suptitle('Held-out Erangel matches | 188 matches, 1,504 windows, 3 training seeds',fontsize=17)
 fig.supxlabel('Points: match-macro means; bars: 95% match bootstrap intervals conditional on three trained seeds.\nDiffusion: 20 joint scenarios/window. Mean path need not be a traversable or sampled path.',fontsize=10)
 for ext in ['png','pdf']:fig.savefig(OUT/f'metrics_comparison.{ext}',dpi=160)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True,layout='constrained')
 for view,ax in zip(['observer','server'],axes):
  for seed,style in zip(SEEDS,['-','--',':']):
   rows=[json.loads(s) for s in (OUT/f'{view}_seed{seed}.jsonl').read_text().splitlines()];ax.plot([r['step'] for r in rows],[r['loss_mean_last50'] for r in rows],style,label=f'seed {seed}',lw=1.5)
  ax.set_title(NAMES[view],loc='left');ax.set_xlabel('optimizer updates');ax.set_ylabel('training loss, last 50 batches');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Training diagnostics | fixed budget, no validation-based checkpoint selection')
 fig.savefig(OUT/'training_curves.png',dpi=160);fig.savefig(OUT/'training_curves.pdf');plt.close(fig)

def case_plots(meta,target,pred,records):
 # All three case rules fixed here before inspecting any validation results.
 # Median-error examples are illustrative, not randomly sampled evidence.
 err=np.array([np.mean([r[i]['metrics']['joint_meanpathADE_m'] for r in records['observer']]) for i in range(len(meta))])
 movement=np.linalg.norm(target['trajectory'][:,0,-1]-target['initial_xy'][:,0],axis=-1)*1000
 common=(target['team_valid'].all(1))&(target['presence'].sum(1)>=2)&(movement>=50)
 candidates=[('Human without vehicle evidence',common&np.array([m['actor_kind']==1 and not m['vehicle'] for m in meta])),('Vehicle at anchor',common&np.array([m['vehicle'] for m in meta])),('Bot observer',common&np.array([m['actor_kind']==2 for m in meta]))]
 selected=[]
 for label,mask in candidates:
  ids=np.flatnonzero(mask)
  if len(ids):selected.append((label,int(ids[np.argsort(err[ids])[len(ids)//2]])))
 # Also show the worst observed model error (explicit failure case).
 ids=np.flatnonzero(target['team_valid'].any(1))
 selected.append(('Largest observer mean-path error',int(ids[np.argmax(err[ids])])) )
 image=np.asarray(Image.open(ROOT/'reports/heterograph_case_40_2_2026-10-15/map.jpg'));pc=['#007ca8','#bd5e24','#754baa','#63852b'];manifest=[]
 for case,(label,i) in enumerate(selected):
  p=target['presence'][i];origin=np.array(meta[i]['origin_m']);y=target['trajectory'][i]*1000+origin;initial=target['initial_xy'][i]*1000+origin;valid=target['trajectory_valid'][i]
  cols=['constant_velocity','observer_42','server_42'];allpoints=[initial[p],y[valid]]
  for model in cols:allpoints.append((pred[model][i].mean(0)*1000+origin)[p].reshape(-1,2))
  pts=np.concatenate(allpoints);lo=pts.min(0);hi=pts.max(0);span=max((hi-lo).max()*1.16,500);center=(hi+lo)/2
  fig,axes=plt.subplots(2,3,figsize=(15,9),gridspec_kw={'height_ratios':[2.5,1]},layout='constrained')
  for col,model in enumerate(cols):
   ax=axes[0,col];ax.imshow(image,extent=(0,8160,8160,0),alpha=.38);samples=pred[model][i]*1000+origin;mean=samples.mean(0)
   for member in np.flatnonzero(p):
    truth=np.vstack([initial[member],np.where(valid[member,:,None],y[member],np.nan)]);generated=np.vstack([initial[member],mean[member]])
    ax.plot(*truth.T,color=pc[member],lw=2.5,label=f'P{member+1} true');ax.plot(*generated.T,color=pc[member],ls='--',lw=1.8);ax.scatter(*initial[member],c=pc[member],marker='o',s=25,zorder=5)
   if len(samples)>1:
    for path in samples[:,0]:ax.plot(*np.vstack([initial[0],path]).T,color=pc[0],lw=.6,alpha=.14)
   ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]+span/2,center[1]-span/2);ax.set_aspect('equal');ax.set_xlabel('Erangel X (m)');ax.set_ylabel('Erangel Y (m)');ax.set_title({'constant_velocity':'Constant velocity','observer_42':'Observer diffusion','server_42':'Server diffusion'}[model],loc='left');ax.grid(alpha=.15)
   if col==0:ax.legend(fontsize=8,loc='best',ncols=2)
   tc=target['team_center'][i]*1000+origin;centerpred=samples[:,p].mean(1);tm=target['team_valid'][i];ax=axes[1,col];h=np.arange(10,61,10);truth_disp=np.linalg.norm(tc-initial[p].mean(0),axis=-1);d=np.linalg.norm(centerpred-initial[p].mean(0),axis=-1);ax.plot(h,np.where(tm,truth_disp,np.nan),'-o',color='#222',label='True center');ax.plot(h,d.mean(0),'--o',color='#147eaa',label='Mean scenario distance');
   if len(samples)>1:ax.fill_between(h,*np.quantile(d,[.1,.9],axis=0),color='#147eaa',alpha=.15,label='10–90% scenarios')
   ax.set_xlabel('Forecast horizon (s)');ax.set_ylabel('Center displacement (m)');ax.grid(alpha=.2)
   if col==0:ax.legend(fontsize=8)
  upper=max(ax.get_ylim()[1] for ax in axes[1])
  for ax in axes[1]:ax.set_ylim(0,upper)
  fig.suptitle(f"{label} | {meta[i]['match_id'][:8]} | t={meta[i]['anchor_s']} s | {p.sum()} fixed members",fontsize=16)
  fig.supxlabel('Solid: telemetry target; dashed: generated mean path; faint lines: 20 ego scenarios. Seed 42 only.\nMap crop excludes some scenarios; same scales within a case. Cases use separate local map scales.\nIllustrative selection by stated error rule. Historical basemap is not verified 40.2 collision geometry.',fontsize=9)
  for ext in ['png','pdf']:fig.savefig(OUT/f'validation_case_{case+1}.{ext}',dpi=150)
  plt.close(fig);manifest.append({'label':label,'evaluation_position':i,'sample_id':meta[i]['sample_id'],'selection':'largest error' if case==len(selected)-1 else 'median observer joint mean-path ADE among complete moving multi-member subgroup','figure':f'validation_case_{case+1}.png'})
 save(OUT/'visualization_cases.json',manifest)
 return manifest

def viewer_data(meta,target,pred,records,cases):
 # Every evaluated sample available in a small local file, loaded on demand.
 folder=OUT/'viewer_samples';folder.mkdir(exist_ok=True);ds=TrajectoryDataset(split='validation')
 for i,m in enumerate(meta):
  mi,k,ego,*_=ds.index[m['index']];b=ds.bank(ds.rows[int(mi)]['match_id']);zones=[]
  for z,start in enumerate([3,6]):
   if b['game_mask'][k,start:start+3].all() and b['game'][k,start+2]>0:zones.append({'kind':['safe','poison'][z],'xy':b['game'][k,start:start+2].tolist(),'radius':float(b['game'][k,start+2])})
  obj={'meta':m,'zones_asof':zones,'presence':target['presence'][i].tolist(),'initial_xy':target['initial_xy'][i].tolist(),'truth':target['trajectory'][i].tolist(),'truth_valid':target['trajectory_valid'][i].tolist(),'team_valid':target['team_valid'][i].tolist(),'time_error_s':target['time_error_s'][i].tolist(),'predictions':{key:np.round(value[i],6).tolist() for key,value in pred.items()},'metrics':{key:records[key][0][i]['metrics'] for key in ('constant_velocity','stationary')},'diffusion_metrics':{view:{str(seed):records[view][j][i]['metrics'] for j,seed in enumerate(SEEDS)} for view in ('observer','server')}}
  (folder/f'{i:04d}.json').write_text(json.dumps(obj,separators=(',',':')))
 save(OUT/'viewer_index.json',{'samples':meta,'cases':cases});shutil.copyfile(ROOT/'reports/heterograph_case_40_2_2026-10-15/map.jpg',OUT/'map.jpg')
 shutil.copyfile(ROOT/'tools/movement_diffusion/viewer_v2.html',OUT/'viewer.html')

def main():
 baseline=read(OUT/'evaluation_baselines.json')['records'];records={'constant_velocity':[baseline],'stationary':[[{**r,'metrics':r['stationary_metrics']} for r in baseline]]}
 for view in ('observer','server'):records[view]=[read(OUT/f'evaluation_{view}_{seed}.json')['records'] for seed in SEEDS]
 ids=[r['sample_id'] for r in baseline];assert len(ids)==1504 and len(set(ids))==1504
 for replicas in records.values():
  for r in replicas:assert [x['sample_id'] for x in r]==ids
 diagnostics=read(OUT/'evaluation_data_diagnostics.json')['samples']
 assert [r['sample_id'] for r in diagnostics]==ids
 overall=summarize(records);groups={}
 for label,mask in [('before_post_death_positive_evidence',[not r['post_death_positive_evidence_seen_in_match'] for r in diagnostics]),('after_post_death_positive_evidence',[r['post_death_positive_evidence_seen_in_match'] for r in diagnostics]),('full_history',[r['full_ego_history'] for r in diagnostics]),('human',[r['actor_kind']==1 for r in baseline]),('bot',[r['actor_kind']==2 for r in baseline]),('vehicle',[r['vehicle'] for r in baseline]),('on_foot',[not r['vehicle'] for r in baseline]),('combat',[r['combat'] for r in baseline]),('no_recent_combat',[not r['combat'] for r in baseline]),('multi_member',[r['current_members']>1 for r in baseline]),('complete60s',[bool(r['complete_joint_60s']) for r in baseline])]:groups[label]={'windows':sum(mask),**summarize(records,mask)}
 summary={'aggregation':'Average windows inside match, average three training-seed scores, then equal-weight matches. Paired bootstrap resamples matches jointly; seed variability reported separately. No p-values or causal claim. Primary 3 metrics x3 paired comparisons also have Bonferroni intervals.','overall':overall,'subgroups':groups}
 save(OUT/'comparison_summary.json',summary)
 with (OUT/'metrics_table.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['group','model','metric','mean_m','ci95_low','ci95_high','matches','seed_std'])
  for group,stats in [('all',overall)]+list(groups.items()):
   for model,values in stats['models'].items():
    for metric,r in values.items():w.writerow([group,model,metric,r['mean'],*r['ci95_match_bootstrap'],r['matches'],r['seed_std']])
 plots(overall);meta=read(OUT/'evaluation_samples.json');target=dict(np.load(OUT/'evaluation_targets.npz'));pred=dict(np.load(OUT/'predictions.npz'));cases=case_plots(meta,target,pred,records);viewer_data(meta,target,pred,records,cases)
 print(json.dumps({'primary':{model:{k:overall['models'][model][k] for k in PRIMARY} for model in MODELS},'cases':cases},indent=2))
if __name__=='__main__':main()
