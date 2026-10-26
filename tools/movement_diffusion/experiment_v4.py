"""Frozen V4 protocol: fit-only balanced sampling, paired regression and diffusion."""
import argparse,copy,hashlib,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset_v3 import TrajectoryDataset,collate,DEFAULT
from movement_diffusion.model_v4 import FormationRegressor,FormationDiffusion,center_relative
from movement_diffusion.evaluation import metrics,match_macro
from tools.movement_diffusion.experiment_v2 import worker_init
OUT=ROOT/'reports/heterograph_diffusion_experiment_v4';OLD=ROOT/'reports/heterograph_diffusion_experiment_v3'
SEEDS=[41,42,43];REG_STEPS=2000;DIFF_STEPS=4000
SOURCES=['movement_diffusion/bank_v3.py','movement_diffusion/dataset_v3.py','movement_diffusion/dataset.py','movement_diffusion/model.py','movement_diffusion/model_v3.py','movement_diffusion/model_v4.py','movement_diffusion/evaluation.py','tools/movement_diffusion/experiment_v4.py','tools/movement_diffusion/prepare_v4.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d):(OUT/name).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))
def prepare():
 assert not list(OUT.glob('*.pt'))
 spec={'status':'Development iteration informed by V3; not preregistered or independent confirmation','regression_steps':REG_STEPS,'regression_batch':8,'residual_steps':DIFF_STEPS,'residual_batch':32,'seeds':SEEDS,'models':'6 formation regressors, 6 hierarchical residual diffusions; 3 observer independent residual controls; 1 observer seed41 uniform-sampling regressor control','sampling':'6 original match-uniform draws + 1 pickup/redeploy alternating + 1 post-reentry draw. Rare draws uniform match then window, fit only. Same plan in both views. No importance weighting; changed fitting distribution is explicit.','architecture':'Regression: shared center head plus zero-sum member displacements; actual future pair-distance Huber. Diffusion: frozen exact regression/graph/zone backbone, center and zero-sum relative residuals, train-only RMS normalization, x0 denoising plus pair-distance loss. Independent residual x0 control shares same frozen regression, plan, optimizer budget, complete-label masks and inputs.','loss_weights':{'team':.1,'mode':.05,'zone_radial':.05,'formation':.2},'optimizer':'AdamW lr5e-4 to5e-5,100stepwarmup,weight_decay1e-4,gradclip2,EMA.99,final only','evaluation':'Byte-identical V3 primary1504 windows in188 matches; secondary371 direct-transport windows; K20 independent samples,32DDIM steps, same sampling seeds, unchanged metric functions and target masks. No paired-noise symmetry or recentering of generated samples.','point_reporting':'Primary diffusion mean-path remains sample mean; frozen regression mean reported separately. Diffusion cannot claim the frozen backbone as independent point-prediction learning.','selection':'No V4 validation-based checkpoint selection, tuning or rerun. All outcomes including failures reported.','test':'94 locked retrospective candidates excluded; historical V2 training exposure remains, so not pristine test.','uncertainty':'Match bootstrap on fixed3trainingseeds is descriptive; validation used repeatedly for development.','sources':{s:sha(ROOT/s) for s in SOURCES},'plans':{str(s):sha(OUT/f'training_indices_seed{s}.npy') for s in SEEDS},'evaluation_sha256':sha(OUT/'evaluation_indices.npy'),'transport_plan_sha256':sha(OUT/'transport_audit_plan.json'),'split_lock_sha256':sha(DEFAULT/'split_lock.json')}
 write('experiment_spec.json',spec)
 for p in SOURCES:
  dest=OUT/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
 print('V4_PROTOCOL_FROZEN',flush=True)
def verify():
 spec=json.loads((OUT/'experiment_spec.json').read_text());assert all(sha(ROOT/p)==h for p,h in spec['sources'].items());assert sha(DEFAULT/'split_lock.json')==spec['split_lock_sha256'];return spec

def reg_train(view,seed,workers=2,uniform=False):
 spec=verify();name=f'{view}_regression_{seed}'+('_uniform' if uniform else '');path=OUT/f'{name}.pt'
 if path.exists():return
 torch.manual_seed(seed);np.random.seed(seed);ds=TrajectoryDataset(view=view);plan=np.load((OLD if uniform else OUT)/f'training_indices_seed{seed}.npy');m=FormationRegressor(len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4);ema=copy.deepcopy(m);opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-4)
 loader=DataLoader(ds,batch_sampler=plan.tolist(),collate_fn=collate,num_workers=workers,worker_init_fn=worker_init,prefetch_factor=2 if workers else None);start=time.monotonic();logs=[];recent=[]
 with (OUT/f'{name}.jsonl').open('w') as f:
  for step,b in enumerate(loader,1):
   lr=min(1,step/100)*(5e-5+4.5e-4*.5*(1+math.cos(math.pi*max(0,step-100)/(REG_STEPS-100))))
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True);loss,parts=m.loss(b['conditions'],b['targets']);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),2.,error_if_nonfinite=True);opt.step()
   with torch.no_grad():
    for a,z in zip(ema.parameters(),m.parameters()):a.lerp_(z,.01)
   recent.append(float(loss.detach()))
   if step==1 or step%100==0:
    row=dict(step=step,loss_mean=float(np.mean(recent)),lr=lr,grad_norm=float(norm),elapsed_s=time.monotonic()-start,**parts);recent=[];logs.append(row);f.write(json.dumps(row)+'\n');f.flush();print(name,step,round(row['loss_mean'],4),round(row['elapsed_s'],1),flush=True)
 torch.save(dict(model=ema.state_dict(),steps=REG_STEPS,view=view,seed=seed,kind='regression',uniform=uniform,source_sha256=spec['sources'],plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),log=logs,elapsed_s=time.monotonic()-start),path);print('REG_COMPLETE',name,flush=True)

def get_cache(view,seed,workers):
 path=OUT/f'train_features_{view}_{seed}.pt';ckpath=OUT/f'{view}_regression_{seed}.pt';ds=TrajectoryDataset(view=view);reg=FormationRegressor(len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4);reg.load_state_dict(torch.load(ckpath,weights_only=False)['model']);reg.eval();reg.requires_grad_(False)
 if path.exists():
  cache=torch.load(path,weights_only=False);assert cache['regressor_sha256']==sha(ckpath);return reg,cache
 m=FormationDiffusion(reg);plan=np.load(OUT/f'training_indices_seed{seed}.npy');loader=DataLoader(ds,batch_sampler=plan.tolist(),collate_fn=collate,num_workers=workers,worker_init_fn=worker_init,prefetch_factor=2 if workers else None);contexts=[];targets=[]
 for n,b in enumerate(loader):
  contexts.append(m.context(b['conditions']));targets.append({k:b['targets'][k] for k in ['trajectory','trajectory_valid']})
  if (n+1)%500==0:print('CACHE',view,seed,n+1,flush=True)
 cache={'context':{k:torch.cat([r[k] for r in contexts]) for k in contexts[0]},'target':{k:torch.cat([r[k] for r in targets]) for k in targets[0]},'regressor_sha256':sha(ckpath),'plan_sha256':sha(OUT/f'training_indices_seed{seed}.npy')};torch.save(cache,path);return reg,cache

def residual_train(view,seed,workers=2):
 spec=verify();reg,cache=get_cache(view,seed,workers);ctx,t=cache['context'],cache['target'];p=ctx['presence'];valid=t['trajectory_valid'][:,:,[1,3,5]];complete=(valid|~p[:,:,None]).all(1);mask=complete[:,None]&p[:,:,None];residual=t['trajectory'][:,:,[1,3,5]]-ctx['mean'][:,:,[1,3,5]];center,relative=center_relative(residual,p)
 def rms(x,mask):return float(((x.square()*mask[...,None]).sum()/(mask.sum()*2).clamp_min(1)).sqrt().clamp_min(.01))
 scales=dict(center_scale=rms(center,complete[:,None]),relative_scale=rms(relative,mask),independent_scale=rms(residual,mask));write(f'train_residual_scales_{view}_{seed}.json',{**scales,'complete_knots':int(complete.sum()),'complete_fraction':float(complete.float().mean()),'fit_only':True})
 for structured in ([True,False] if view=='observer' else [True]):
  kind='diffusion' if structured else 'independent';name=f'{view}_{kind}_{seed}';path=OUT/f'{name}.pt'
  if path.exists():continue
  torch.manual_seed(seed+10000);m=FormationDiffusion(copy.deepcopy(reg),structured);m.eval()
  for k,v in scales.items():getattr(m,k).fill_(v)
  params=[p for p in m.parameters() if p.requires_grad];ema=copy.deepcopy(m);opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-4);rng=np.random.default_rng(seed+20000);logs=[];start=time.monotonic();recent=[]
  with (OUT/f'{name}.jsonl').open('w') as f:
   for step in range(1,DIFF_STEPS+1):
    ids=rng.integers(len(p),size=32);cc={k:v[ids] for k,v in ctx.items()};tt={k:v[ids] for k,v in t.items()};lr=min(1,step/100)*(5e-5+4.5e-4*.5*(1+math.cos(math.pi*max(0,step-100)/(DIFF_STEPS-100))))
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);loss,parts=m.loss_cached(cc,tt);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
     for a,z in zip(ema.parameters(),m.parameters()):
      if z.requires_grad:a.lerp_(z,.01)
    recent.append(float(loss.detach()))
    if step==1 or step%200==0:
     row=dict(step=step,loss_mean=float(np.mean(recent)),lr=lr,elapsed_s=time.monotonic()-start,**parts);recent=[];logs.append(row);f.write(json.dumps(row)+'\n');f.flush();print(name,step,round(row['loss_mean'],4),flush=True)
  for k,v in reg.state_dict().items():torch.testing.assert_close(ema.regressor.state_dict()[k],v,rtol=0,atol=0)
  torch.save(dict(model=ema.state_dict(),kind=kind,seed=seed,view=view,steps=DIFF_STEPS,source_sha256=spec['sources'],regressor_sha256=cache['regressor_sha256'],plan_sha256=cache['plan_sha256'],scales=scales,log=logs,elapsed_s=time.monotonic()-start),path);print('RESIDUAL_COMPLETE',name,flush=True)

def load_model(ds,view,kind,seed,uniform=False):
 suffix='_uniform' if uniform else '';ck=torch.load(OUT/f'{view}_{kind}_{seed}{suffix}.pt',weights_only=False);reg=FormationRegressor(len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4);m=reg if kind=='regression' else FormationDiffusion(reg,kind=='diffusion');m.load_state_dict(ck['model']);m.eval();return m

def evaluate():
 verify();meta=json.loads((OUT/'evaluation_samples.json').read_text());tp=json.loads((OUT/'transport_audit_plan.json').read_text());records={};transport={};predictions={};transport_preds={}
 for view in ['observer','server']:
  ds=TrajectoryDataset(split='validation',view=view);models={}
  for kind in (['regression','diffusion','independent'] if view=='observer' else ['regression','diffusion']):
   for seed in SEEDS:
    name=f'{view}_{kind}_{seed}';models[name]=(seed,load_model(ds,view,kind,seed));records[name]=[];transport[name]=[];predictions[name]=[];transport_preds[name]=[]
  if view=='observer':
   name='observer_regression_41_uniform';models[name]=(41,load_model(ds,view,'regression',41,True));records[name]=[];transport[name]=[];predictions[name]=[];transport_preds[name]=[]
  for group,plan in [('primary',meta),('transport',tp['plan'])]:
   for number,md in enumerate(plan):
    s=ds[int(md['index'])];assert s['meta']['match_id']==md['match_id'];c=collate([s])['conditions'];t=s['targets'];presence=s['conditions']['current_presence']
    if group=='primary':assert s['meta']['sample_id']==md['sample_id']
    for name,(seed,m) in models.items():
     torch.manual_seed(seed*100000+(50000 if group=='transport' else 0)+number);pr=m.sample(c,20)[0].numpy();assert np.isfinite(pr).all();r={**md,'sample_id':s['meta']['sample_id'],'metrics':metrics(pr,t,presence)};(records if group=='primary' else transport)[name].append(r);(predictions if group=='primary' else transport_preds)[name].append(pr)
    if (number+1)%200==0:print('EVAL',view,group,number+1,flush=True)
 for name,rows in records.items():write(f'evaluation_{name}.json',dict(records=rows,match_macro=match_macro(rows)))
 write('transport_audit_results.json',dict(records=transport,counts=tp['counts']));np.savez_compressed(OUT/'predictions.npz',**{k:np.stack(v) for k,v in predictions.items()});np.savez_compressed(OUT/'transport_predictions.npz',**{k:np.stack(v) for k,v in transport_preds.items()});print('EVALUATION_COMPLETE',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train-seed','train-uniform','evaluate']);ap.add_argument('--seed',type=int,default=41);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 if a.stage=='prepare':prepare()
 elif a.stage=='evaluate':evaluate()
 elif a.stage=='train-uniform':reg_train('observer',41,a.workers,True)
 else:
  for view in ['observer','server']:reg_train(view,a.seed,a.workers);residual_train(view,a.seed,a.workers)
if __name__=='__main__':main()
