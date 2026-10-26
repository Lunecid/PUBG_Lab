#!/usr/bin/env python3
"""One graph architecture, frozen equal-budget tuning and an edge-shuffle control."""
import argparse
import fcntl
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import sac2027_fair_tune as F
from model_stgcn.ptr.gatv2 import SnapshotGATv2
from model_stgcn.ptr.util import set_seed


def build(cfg, architecture, seed):
    assert architecture == 'gatv2'
    set_seed(seed)
    return SnapshotGATv2(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True, choices=['erangel_tpp_squad', 'taego_tpp_squad'])
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    F.torch.set_num_threads(2)
    spec = F.freeze()
    root = F.OUT/a.cohort/'gatv2'; root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.run.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    F.build_model = build
    prep = F.T.prepare(F.T.parse_args(['--dump-dir', str(F.DUMP), '--cohort', a.cohort,
        '--arm', 'M11R-norank', '--out-dir', str(root)]))
    device = F.T._device(a.device)
    trials = []
    for i, setting in enumerate(spec['neural_trials']):
        dest = root/f'trial{i:02d}'
        if (dest/'trial.json').exists(): record = json.loads((dest/'trial.json').read_text())
        else:
            model, record = F.fit(prep, 'gatv2', setting, 42, device, dest)
            del model
        assert record['setting'] == setting
        trials.append(dict(trial=i, **record))
    chosen = min(trials, key=lambda r: (r['val_nll'], r['trial']))
    F.save(root/'selection.json', dict(chosen_trial=chosen['trial'], setting=chosen['setting'],
        val_nll=chosen['val_nll'], n_trials=16, device=str(device)))
    for condition in ('full', 'shuffled'):
        # Same permutation for all relation channels preserves the graph and its
        # weight/degree distributions, while breaking alignment to team tokens.
        for ds in prep.datasets.values():
            ds.shuffle_seed = None if condition == 'full' else 20261008
        for seed in spec['seeds']:
            dest = (root/f'selected_seed{seed}' if condition == 'full' else
                F.BASE/'graph_controls'/a.cohort/'shuffled'/f'seed{seed}')
            if (dest/'metrics.json').exists(): continue
            if condition == 'full' and seed == 42:
                ckpt = F.torch.load(root/f"trial{chosen['trial']:02d}"/'best.pt', map_location='cpu', weights_only=False)
                model = build(F.T.PTRConfig(**ckpt['config']), 'gatv2', seed).to(device)
                model.load_state_dict(ckpt['state_dict'])
                dest.mkdir(parents=True, exist_ok=True); F.torch.save(ckpt, dest/'best.pt')
            else: model, _ = F.fit(prep, 'gatv2', chosen['setting'], seed, device, dest)
            results = {}
            for split in ('val', 'test'):
                frame = F.T.predict(model, prep.datasets[split], device, 8, rank_from_hazard=True)
                F.M.write_predictions(frame, dest/f'pred_{split}.csv')
                results[split] = F.M.hazard_metrics(frame.y, frame.p_hazard)
            F.save(dest/'metrics.json', dict(seed=seed, selection=chosen, condition=condition,
                metrics=results, device=str(device), snapshot_only=True,
                source='PyG 2.7.0 GATv2Conv; continuous 4-channel edges; no new threshold',
                exploratory='Architecture chosen after earlier model results were available; graph test data did not select it.'))
            del model
    F.save(root/'complete.json', dict(cohort=a.cohort, trials=16, seeds=spec['seeds'], shuffle_control=True))
    print('COMPLETE gatv2', a.cohort, flush=True)


if __name__ == '__main__': main()
