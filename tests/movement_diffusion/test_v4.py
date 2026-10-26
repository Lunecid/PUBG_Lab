import numpy as np
import torch
from pubg_lab.movement.models.model_v4 import center_relative, pair_loss, FormationRegressor, FormationDiffusion
from pubg_lab.movement.data.dataset_v3 import TrajectoryDataset, collate

def test_center_relative_round_trip_and_padding():
 p=torch.tensor([[True,True,True,False]]);x=torch.randn(1,4,3,2);c,r=center_relative(x,p)
 torch.testing.assert_close((c+r)[:,:3],x[:,:3]);torch.testing.assert_close(r.sum(1),torch.zeros_like(c[:,0]),atol=1e-6,rtol=1e-6)
 x[:,3]+=100;c2,r2=center_relative(x,p);torch.testing.assert_close(c2,c);torch.testing.assert_close(r2,r)

def test_pair_loss_translation_missingness_and_actual_separation():
 y=torch.randn(2,4,6,2);valid=torch.ones(2,4,6,dtype=torch.bool);valid[:,3]=False
 assert pair_loss(y+5,y,valid)<1e-10
 x=y.clone();x[:,3]+=100;assert pair_loss(x,y,valid)==0
 x[:,1]+=2;assert pair_loss(x,y,valid)>0
 assert pair_loss(x,y,torch.zeros_like(valid))==0

def test_hierarchical_denoiser_equivariance_and_single_member():
 torch.manual_seed(3);m=FormationDiffusion(FormationRegressor(150));p=torch.tensor([[True,True,True,False]]);ctx={'presence':p,'h':torch.randn(1,4,116)};y=m.noise(ctx);step=torch.tensor([12]);a=m.denoise(y,step,ctx);order=[2,0,1,3];b=m.denoise(torch.cat([y[:,:1],y[:,1:][:,order]],1),step,{'presence':p[:,order],'h':ctx['h'][:,order]})
 torch.testing.assert_close(a[:,:1],b[:,:1]);torch.testing.assert_close(a[:,1:][:,order],b[:,1:],atol=1e-6,rtol=1e-5)
 ctx['presence']=torch.tensor([[True,False,False,False]]);assert not m.noise(ctx)[:,1:].any()

def test_real_losses_and_frozen_backbone():
 torch.set_num_threads(2);ds=TrajectoryDataset();b=collate([ds[200],ds[210]]);c,t=b['conditions'],b['targets'];reg=FormationRegressor(len(ds.builder.vocab['items'])+len(ds.builder.vocab['vehicles'])+4);loss,_=reg.loss(c,t);loss.backward();assert torch.isfinite(loss) and reg.center_head[-1].weight.grad is not None
 for structured in [True,False]:
  reg.zero_grad(set_to_none=True);m=FormationDiffusion(reg,structured);ctx=m.context(c);loss,_=m.loss_cached(ctx,t);loss.backward();assert torch.isfinite(loss);assert all(p.grad is None for p in reg.parameters());pred=m.sample(c,3);assert pred.shape==(2,3,4,6,2) and torch.isfinite(pred).all()

def test_incomplete_future_and_absent_member_targets_do_not_affect_loss():
 torch.manual_seed(9);m=FormationDiffusion(FormationRegressor(150));p=torch.tensor([[True,True,False,False]]);ctx={'presence':p,'h':torch.randn(1,4,116),'mean':torch.randn(1,4,6,2)};truth=torch.randn(1,4,6,2);valid=torch.ones(1,4,6,dtype=torch.bool);valid[:,2:]=False;valid[:,1,3]=False
 t={'trajectory':truth,'trajectory_valid':valid};torch.manual_seed(2);a=m.loss_cached(ctx,t)[0];changed=truth.clone();changed[:,2:]+=999;changed[:,:,3]+=999;torch.manual_seed(2);b=m.loss_cached(ctx,{'trajectory':changed,'trajectory_valid':valid})[0];torch.testing.assert_close(a,b)

def test_shared_center_noise_preserves_pair_distances():
 p=torch.tensor([[True,True,True,True]]);m=FormationDiffusion(FormationRegressor(150));latent=torch.zeros(1,5,3,2);latent[:,:1]=torch.randn(1,1,3,2);res=m.decode_residual(latent,p);y=torch.randn(1,4,3,2);assert pair_loss(y+res,y,torch.ones(1,4,3,dtype=torch.bool))<1e-10
