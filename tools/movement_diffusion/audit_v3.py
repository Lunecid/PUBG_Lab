"""Read-only rebuilt-bank audit; never loads held-out test raw or banks."""
import csv,json,sys,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.bank import read_raw
from movement_diffusion.bank_v3 import build_bank
from movement_diffusion.dataset_v3 import TrajectoryDataset,DEFAULT
OUT=ROOT/'reports/heterograph_diffusion_experiment_v3';OLD=ROOT/'data/processed/heterograph_diffusion_v1'
def main():
 rows=list(csv.DictReader((DEFAULT/'matches.csv').open()));groups={s:{r['match_id'] for r in rows if r['split']==s} for s in ['train','validation','test']};assert not any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]);v=json.loads((DEFAULT/'vocabulary.json').read_text());aggregate=defaultdict(lambda:[0,0]);asof=0;rawhash={};bankhash={};resurrect=0
 for r in rows:
  mid=r['match_id'];bank=DEFAULT/'banks'/f'{mid}.npz';spatial=DEFAULT/'spatial_match'/f'{mid}.npz'
  if r['split']=='test':assert not bank.exists() and not spatial.exists();continue
  assert (r['split']=='train')==spatial.exists();rawhash[mid]=hashlib.sha256((OLD/'raw'/f'{mid}.json.gz').read_bytes()).hexdigest();bankhash[mid]=hashlib.sha256(bank.read_bytes()).hexdigest()
  with np.load(bank) as b:
   valid=b['valid'];assert (b['source_time'][valid]<=np.broadcast_to(b['times'][:,None],valid.shape)[valid]).all();assert (b['source_time'][valid]>=b['episode_start'][valid]).all();asof+=int(valid.sum());resurrect+=int((valid&(b['episode']>0)).sum())
  if spatial.exists():
   for a,b,m,c in np.load(spatial)['edges']:aggregate[int(a),int(b),int(m)][0]+=int(c);aggregate[int(a),int(b),int(m)][1]+=1
 actual=np.load(DEFAULT/'spatial_prior.npz')['edges'];expected=np.array([[*k,*value] for k,value in sorted(aggregate.items())]);np.testing.assert_array_equal(actual,expected)
 prefix=[]
 for mid in sorted(groups['train'])[:5]:
  raw=read_raw(OLD/'raw'/f'{mid}.json.gz');full=dict(np.load(DEFAULT/'banks'/f'{mid}.npz'))
  for cut in [300,600,900]:
   partial={**raw,'tables':{k:[r for r in rs if r['t']<=cut] for k,rs in raw['tables'].items()}};bank=build_bank(partial,v)[0]
   for key in ['x','mask','valid','source_time','items','item_seen','vehicle','episode','episode_start','mobility','reentry_confidence','z','vz','game','game_mask']:
    np.testing.assert_array_equal(bank[key],full[key][:len(bank['times'])],err_msg=mid+':'+str(cut)+':'+key)
   prefix.append([mid,cut])
 ds=TrajectoryDataset();dsval=TrajectoryDataset(split='validation');assert set(ds.index[:,0]).isdisjoint(set(dsval.index[:,0]));assert len({rawhash[m] for m in rawhash})==len(rawhash)
 report={'split_counts':{s:len(x) for s,x in groups.items()},'test_loading':'No test bank, index or spatial file; loader raises PermissionError','test_historical_limitation':'94 test candidates were trained by v2; retrospective holdout is not a pristine external test','source_asof_valid_positions':asof,'restored_valid_grid_positions':resurrect,'spatial_edges_exact_train_only':len(actual),'prefix_rebuild_checks':prefix,'unique_raw_matches':len(rawhash),'bank_sha256':bankhash,'raw_sha256':rawhash,'notes':['No normalization learned from dev/test; fixed physical scales','Target interpolation is label-only; no velocity or spatial connection across episode or mobility boundaries','Target labels cannot bridge a future death/reentry for a fixed current-life actor','Observed vehicle class is evidence; airborne after exit and post-death fallback are inferred, no claim of exact engine state']}
 (OUT/'leakage_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['raw_sha256','bank_sha256']},indent=2))
if __name__=='__main__':main()
