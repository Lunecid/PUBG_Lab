"""Dataset-only validation diagnostics; no model scores or checkpoint selection."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset import TrajectoryDataset
from movement_diffusion.bank import read_raw
OUT=ROOT/'reports/heterograph_diffusion_experiment_v2'
def main():
 ds=TrajectoryDataset(split='validation');ids=np.load(OUT/'evaluation_indices.npy');by=defaultdict(list)
 for pos,i in enumerate(ids):by[int(ds.index[i,0])].append((pos,int(i)))
 extras=[None]*len(ids);validpoint=0;teamsteps=0;targeterrors=[];spreads=[];fullhist=0;vehicles=0
 for mi,entries in by.items():
  mid=ds.rows[mi]['match_id'];raw=read_raw(ds.root/'raw'/f'{mid}.json.gz');death={}
  for r in raw['tables']['kills']:death[r['actor']]=min(death.get(r['actor'],float('inf')),r['t'])
  post=[r['t'] for r in raw['tables']['positions'] if r['t']>death.get(r['actor'],float('inf')) and (r['hp'] or 0)>0];first=min(post) if post else None;b=ds.bank(mid)
  for pos,i in entries:
   _,k,ego,*_=ds.index[i];sample=ds[i];p=sample['conditions']['current_presence'];target=sample['targets'];ages=sample['conditions']['position_age_s'];extras[pos]={'sample_id':sample['meta']['sample_id'],'post_death_positive_evidence_seen_in_match':bool(first is not None and first<=k*10),'first_post_death_positive_evidence_s':first,'ego_position_age_s':float(ages[0]),'mean_current_member_age_s':float(ages[p].mean()),'full_ego_history':bool(b['valid'][k-3:k+1,ego].all()),'current_member_span_m':float(np.linalg.norm(sample['conditions']['initial_xy'][p][:,None]-sample['conditions']['initial_xy'][p][None],axis=-1).max()*1000)}
   validpoint+=int(target['trajectory_valid'].sum());teamsteps+=int(target['team_valid'].sum());targeterrors.extend(abs(target['time_error_s'][target['trajectory_valid']]).tolist());spreads.append(extras[pos]['current_member_span_m']);fullhist+=extras[pos]['full_ego_history'];vehicles+=bool(b['x'][k,ego,14])
 result={'samples':extras,'summary':{'windows':len(ids),'valid_individual_future_points':validpoint,'valid_team_future_steps':teamsteps,'mean_abs_target_clock_error_s':float(np.mean(targeterrors)),'current_vehicle_windows':int(vehicles),'all4_ego_history_positions_known_windows':int(fullhist),'post_death_positive_evidence_seen_windows':sum(s['post_death_positive_evidence_seen_in_match'] for s in extras),'current_member_span_m_quantiles':np.quantile(spreads,[0,.25,.5,.75,.95,1]).tolist()},'meaning':'Post-death positive HP evidence is a recall/stale-data warning proxy, not confirmed recall. Subgroup is diagnostic only; no future outcomes used to define it.'}
 (OUT/'evaluation_data_diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
