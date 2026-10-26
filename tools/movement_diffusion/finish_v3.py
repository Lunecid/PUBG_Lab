"""Wait for authorized fixed runs, then evaluate development data only."""
import os,sys,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/heterograph_diffusion_experiment_v3'
expected=[f'{v}_{k}_{s}.pt' for v in ['observer','server'] for k in ['regression','diffusion'] for s in [41,42,43]]+[f'observer_{k}_41_nozone.pt' for k in ['regression','diffusion']]
start=time.time();last=-1
while not all((OUT/x).exists() for x in expected):
 n=sum((OUT/x).exists() for x in expected)
 if n!=last:print('CHECKPOINTS',n,len(expected),flush=True);last=n
 for p in [OUT/'train41.log',OUT/'train42.log',OUT/'train43.log',OUT/'train_ablation.log']:
  if p.exists() and 'Traceback (most recent call last)' in p.read_text():raise RuntimeError('Training failure; inspect '+str(p))
 if time.time()-start>3600:raise TimeoutError('Waiting for checkpoints exceeded one hour; did not terminate training')
 time.sleep(5)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLCONFIGDIR='/private/tmp/pubg-mpl',XDG_CACHE_HOME='/private/tmp/pubg-cache')
for name,args in [('evaluation',['experiment_v3.py','evaluate']),('analysis',['analyze_v3.py']),('diagnostics',['diagnose_v3.py']),('report',['report_v3.py'])]:
 print('START',name,flush=True)
 with (OUT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'tools/movement_diffusion'/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FINISHED',name,flush=True)
(OUT/'pipeline_complete.json').write_text(json.dumps({'completed':True,'test_inference_performed':False,'elapsed_pipeline_s':time.time()-start},indent=2));print('PIPELINE_COMPLETE',flush=True)
