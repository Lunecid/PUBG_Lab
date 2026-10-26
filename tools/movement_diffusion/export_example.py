"""Write two inspectable paired batches, with condition files separate from labels."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from movement_diffusion.dataset import TrajectoryDataset,collate,DEFAULT
out=DEFAULT/'examples';out.mkdir(exist_ok=True);labels=None;metadata={}
for view in ('observer','server'):
 ds=TrajectoryDataset(view=view);samples=[ds[len(ds)//3],ds[len(ds)*2//3]];batch=collate(samples);c={k:v.numpy() if hasattr(v,'numpy') else np.array(v) for k,v in batch['conditions'].items()};np.savez_compressed(out/f'conditions_{view}.npz',**c);y={k:v.numpy() for k,v in batch['targets'].items()}
 if labels is not None:
  for key in labels:np.testing.assert_array_equal(labels[key],y[key])
 else:labels=y
 metadata[view]=[{**m,'origin_m':m['origin_m'].tolist()} for m in batch['meta']]
np.savez_compressed(out/'shared_targets.npz',**labels);(out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(out)
