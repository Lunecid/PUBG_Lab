"""Run one independent preplanned seed; no changes to the frozen experiment."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment_v2 import train

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args();args.output=ROOT/'reports/heterograph_diffusion_experiment_v2';args.steps=2500;args.batch_size=8;args.workers=2;torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 plan=np.load(args.output/f'training_indices_seed{args.seed}.npy')
 for view in ['observer','server']:train(args,view,args.seed,plan)
 print('REPLICA_COMPLETE',args.seed,flush=True)
if __name__=='__main__':main()
