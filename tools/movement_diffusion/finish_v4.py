"""Wait for fixed jobs then run frozen evaluations and presentation checks once."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/heterograph_diffusion_experiment_v4'
expected=[f'{v}_{k}_{s}' for v in ['observer','server'] for k in ['regression','diffusion'] for s in [41,42,43]]+[f'observer_independent_{s}' for s in [41,42,43]]+['observer_regression_41_uniform']
start=time.monotonic()
while not all((OUT/(name+'.pt')).exists() for name in expected):
 for p in OUT.glob('run_*.log'):
  if 'Traceback (most recent call last)' in p.read_text():raise RuntimeError(f'Training failed: {p.name}')
 if time.monotonic()-start>7200:raise TimeoutError('Training not complete')
 time.sleep(20)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLCONFIGDIR='/private/tmp/pubg-mpl',XDG_CACHE_HOME='/private/tmp/pubg-cache')
for file,args in [('experiment_v4.py',['evaluate']),('analyze_v4.py',[]),('visualize_v4.py',[]),('report_v4.py',[]),('finalize_v4.py',[])]:
 print('START',file,flush=True);subprocess.run([sys.executable,str(ROOT/'tools/movement_diffusion'/file),*args],cwd=ROOT,env=env,check=True);print('DONE',file,flush=True)
print('V4_PIPELINE_COMPLETE',flush=True)
