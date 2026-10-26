import math
import torch
from tools.sac2027_objective_controls import survival_loss, labels
from tools.sac2027_fair_tune import KRAFTONMatched
from model_stgcn.ptr.model import PTRConfig
from src.training.losses import WeightedCoxHazardLoss

def test_standard_cox_breslow_ties_and_censoring():
    z=torch.zeros(1,3,requires_grad=True)
    t=torch.tensor([[1.,1.,2.]])
    event=torch.tensor([[True,True,False]])
    valid=torch.ones_like(event)
    loss=survival_loss(z,t,event,valid)
    assert abs(float(loss.detach())-math.log(3))<1e-6
    loss.backward();assert torch.isfinite(z.grad).all()

def test_weighted_labels_preserve_upstream_censoring_and_target_scale():
    from types import SimpleNamespace
    ds=SimpleNamespace(matches=[dict(team_ids=[1,2,3],team_death_s=torch.tensor([500.,1500.,float("inf")]),match_end_s=1800.)])
    batch={"target_mask":torch.ones(1,1,3,dtype=torch.bool)}
    t,event,end=labels(ds,[0],batch,torch.device("cpu"),upstream_weighted=True)
    assert torch.equal(t,torch.tensor([[.25,.75,1.]]))
    assert torch.equal(event,torch.tensor([[True,True,False]]))
    assert torch.equal(end,torch.tensor([1.]))
    z=torch.tensor([[2.,1.,0.]])
    assert torch.allclose(survival_loss(z,t,event,batch["target_mask"].flatten(0,1),weighted=True,end=end),
                          WeightedCoxHazardLoss()(z,t,torch.ones_like(event)))

def test_weighted_matches_upstream_and_inactive_rows_have_finite_gradients():
    z=torch.tensor([[2.,1.,0.],[1.,2.,0.]],requires_grad=True)
    t=torch.tensor([[1.,2.,3.],[1.,2.,3.]])
    valid=torch.ones_like(z,dtype=torch.bool)
    event=t<3
    ours=survival_loss(z,t,event,valid,weighted=True,end=torch.tensor([3.,3.]))
    theirs=WeightedCoxHazardLoss()(z,t,valid)
    assert torch.allclose(ours,theirs,atol=1e-6)
    padded=torch.cat([z,torch.full((2,1),10000.)],1).detach().requires_grad_()
    mask=torch.cat([valid,torch.zeros(2,1,dtype=torch.bool)],1)
    times=torch.cat([t,torch.zeros(2,1)],1)
    events=torch.cat([event,torch.zeros(2,1,dtype=torch.bool)],1)
    loss=survival_loss(padded,times,events,mask,weighted=True,end=torch.tensor([3.,3.]))
    assert torch.allclose(loss,ours,atol=1e-6)
    loss.backward();assert torch.isfinite(padded.grad).all()
    assert torch.equal(padded.grad[:,-1],torch.zeros(2))

def test_truncated_cox_does_not_use_the_order_of_later_events():
    z=torch.tensor([[2.,1.,0.]])
    early=torch.tensor([[True,False,False]])
    valid=torch.ones_like(early)
    a=survival_loss(z,torch.tensor([[1.,3.,4.]]),early,valid)
    b=survival_loss(z,torch.tensor([[1.,4.,3.]]),early,valid)
    assert torch.equal(a,b)
    zero=survival_loss(z,torch.tensor([[1.,3.,4.]]),torch.zeros_like(early),valid)
    assert float(zero)==0.

def test_cox_gives_no_gradient_to_a_common_offset_and_the_bce_term_does():
    # Mechanism behind cox_truncated_bce: a per-match offset changes neither risk-set ratio, so only the BCE term
    # receives its gradient, the expected-minus-observed eliminations of that match divided by the target rows.
    from model_stgcn.ptr.loss import hazard_bce
    torch.manual_seed(0)
    z=torch.randn(2,1,4)
    y=torch.tensor([[[1.,0.,0.,1.]],[[0.,1.,0.,0.]]])
    mask=torch.ones(2,1,4,dtype=torch.bool)
    t=torch.tensor([[1.,5.,5.,2.],[5.,1.,5.,5.]])
    offset=torch.zeros(2,1,1,requires_grad=True)
    s=z+offset
    cox=survival_loss(s.flatten(0,1),t,y.flatten(0,1).bool(),mask.flatten(0,1))
    g_cox,=torch.autograd.grad(cox,offset,retain_graph=True)
    assert torch.allclose(g_cox,torch.zeros_like(g_cox),atol=1e-6)
    g_bce,=torch.autograd.grad(hazard_bce(s,y,mask),offset)
    assert torch.allclose(g_bce,(torch.sigmoid(z)-y).sum(-1,keepdim=True)/mask.sum(),atol=1e-6)

def test_matched_krafton_padding_and_future_phase_isolation():
    torch.manual_seed(42)
    model=KRAFTONMatched(PTRConfig(in_features=153,d_model=64,n_heads=4,n_layers=1,dropout=0)).eval()
    x=torch.randn(2,3,4,153)
    keys=torch.tensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[1,1,1,0],[1,1,0,0],[0,0,0,0]]],dtype=torch.bool)
    batch=dict(x=x,node_mask=keys,query_mask=keys.clone())
    a=model(batch)["hazard_logit"]
    assert torch.isfinite(a).all()
    noisy=x.clone();noisy[~keys]=1e4
    b=model({**batch,"x":noisy})["hazard_logit"]
    assert torch.allclose(a[keys],b[keys],atol=1e-6)
    noisy=x.clone();noisy[:,2]*=100
    c=model({**batch,"x":noisy})["hazard_logit"]
    assert torch.allclose(a[:,:2],c[:,:2],atol=1e-6)
    for bi in range(2):
        for pi in range(2):
            expected=model.head(model.backbone(x[bi:bi+1,pi],valid_mask=keys[bi:bi+1,pi]))[0]
            assert torch.allclose(a[bi,pi,keys[bi,pi]],expected[keys[bi,pi]],atol=1e-6)
