"""Count actual fixed-plan current-mode exposure without accessing test banks."""
import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processed/heterograph_diffusion_v3'
OUT = ROOT / 'reports/heterograph_diffusion_experiment_v3'


def main():
    rows = {int(r['match_index']): r for r in csv.DictReader((DATA / 'matches.csv').open())}
    modes = json.loads((DATA / 'summary.json').read_text())['modes']
    result = {}
    for split in ['train', 'validation']:
        index = np.load(DATA / f'{split}_index.npy', mmap_mode='r')
        plans = ({str(seed): np.load(OUT / f'training_indices_seed{seed}.npy').flatten()
                  for seed in [41, 42, 43]} if split == 'train'
                 else {'evaluation': np.load(OUT / 'evaluation_indices.npy')})
        selected = {key: index[ids] for key, ids in plans.items()}
        counts = {key: collections.Counter() for key in plans}
        reentry = {key: 0 for key in plans}
        for mi in sorted({int(m) for a in selected.values() for m in a[:, 0]}):
            row = rows[mi]
            assert row['split'] == split
            with np.load(DATA / 'banks' / (row['match_id'] + '.npz')) as bank:
                mobility, episode = bank['mobility'], bank['episode']
                for key, a in selected.items():
                    chosen = a[a[:, 0] == mi]
                    k, ego = chosen[:, 1], chosen[:, 2]
                    counts[key].update(modes[int(x)] for x in mobility[k, ego])
                    reentry[key] += int((episode[k, ego] > 0).sum())
        result[split] = {key: {'draws': len(plans[key]), 'current_mode_counts': dict(counts[key]),
                              'reentry_episode_draws': reentry[key]} for key in plans}
    path = OUT / 'sampling_coverage.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(json.dumps(result, indent=2))
    print('FIXED_PLAN_COVERAGE_VERIFIED')


if __name__ == '__main__':
    main()
