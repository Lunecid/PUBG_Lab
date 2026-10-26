"""Train paired information views using identical seeds/anchors; optional bounded runs."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset import TrajectoryDataset,collate
from movement_diffusion.model import JointGraphDiffusion,constant_velocity
from movement_diffusion.evaluation import metrics,match_macro

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--view',choices=['observer','server','both'],default='both');ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--eval-matches',type=int,default=20);ap.add_argument('--scenarios',type=int,default=10);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--actor-kind',choices=['all','human','bot'],default='all');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);results={}
 for view in (['observer','server'] if args.view=='both' else [args.view]):
  train=TrajectoryDataset(split='train',view=view);val=TrajectoryDataset(split='validation',view=view);kind_code={'all':0,'human':1,'bot':2}[args.actor_kind];eligible=np.flatnonzero((train.index[:,4]>0)&((train.index[:,3]==kind_code) if kind_code else True));rng=np.random.default_rng(args.seed);torch.manual_seed(args.seed);model=JointGraphDiffusion(len(train.builder.vocab['items'])+len(train.builder.vocab['vehicles'])+4);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3);logs=[];started=time.time()
  # Draw a match, then a window: long surviving matches do not receive extra match weight.
  bymatch={m:eligible[train.index[eligible,0]==m] for m in np.unique(train.index[eligible,0])};matchids=sorted(bymatch)
  for step in range(args.steps):
   ids=[int(rng.choice(bymatch[int(rng.choice(matchids))])) for _ in range(args.batch_size)];batch=collate([train[i] for i in ids]);optimizer.zero_grad();loss,parts=model.loss(batch['conditions'],batch['targets']);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
   if step%max(1,args.steps//10)==0 or step==args.steps-1:logs.append({'step':step,'loss':float(loss.detach()),**parts});print(view,logs[-1],flush=True)
  checkpoint={'model':model.state_dict(),'config':vars(args)|{'output':str(args.output)},'view':view,'vocabulary':train.builder.vocab};torch.save(checkpoint,args.output/f'{view}.pt');records=[];model.eval();vrng=np.random.default_rng(args.seed+1);valid_eval=(val.index[:,4]>0)&((val.index[:,3]==kind_code) if kind_code else True);mids=np.unique(val.index[valid_eval,0]);chosen=vrng.choice(mids,min(args.eval_matches,len(mids)),replace=False)
  for mid in sorted(chosen):
   candidates=np.flatnonzero((val.index[:,0]==mid)&valid_eval);i=int(vrng.choice(candidates));sample=val[i];batch=collate([sample]);torch.manual_seed(args.seed+int(mid));generated=model.sample(batch['conditions'],args.scenarios)[0].numpy();result=metrics(generated,sample['targets'],sample['conditions']['current_presence']);cv=constant_velocity(batch['conditions'])[0].numpy();records.append({'sample_id':sample['meta']['sample_id'],'match_id':sample['meta']['match_id'],'actor_kind':sample['meta']['actor_kind'],'metrics':result,'constant_velocity_metrics':metrics(cv[None],sample['targets'],sample['conditions']['current_presence'])})
  results[view]={'training_steps':args.steps,'training_log':logs,'eval_records':records,'match_macro':match_macro(records),'constant_velocity_match_macro':match_macro([{**r,'metrics':r['constant_velocity_metrics']} for r in records]),'elapsed_s':time.time()-started}
 if args.view=='both':assert [r['sample_id'] for r in results['observer']['eval_records']]==[r['sample_id'] for r in results['server']['eval_records']]
 (args.output/'results.json').write_text(json.dumps({'interpretation':'Exploratory matched-view run; no independent test split, no significance or convergence claim. Increase evaluation coverage and use multiple seeds before scientific conclusions.','config':vars(args)|{'output':str(args.output)},'results':results},indent=2));print('Saved',args.output)
if __name__=='__main__':main()
