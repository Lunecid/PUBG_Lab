import copy
import numpy as np
import torch
from pubg_lab.movement.data.dataset import TrajectoryDataset, collate
from pubg_lab.movement.models.model import JointGraphDiffusion, constant_velocity

def fixture():
 torch.set_num_threads(2);torch.manual_seed(7)
 d=TrajectoryDataset();s=d[len(d)//3];c=collate([s])['conditions'];m=JointGraphDiffusion(len(d.builder.vocab['items'])+len(d.builder.vocab['vehicles'])+4);m.eval();return m,c,s

def test_noise_terminal_and_roundtrip():
 m,_,_=fixture();assert m.alpha_bar[-1]<1e-5
 x=torch.randn(2,4,6,2);e=torch.randn_like(x)
 for a in m.alpha_bar:
  z=a.sqrt()*x+(1-a).sqrt()*e;v=a.sqrt()*e-(1-a).sqrt()*x
  torch.testing.assert_close(a.sqrt()*z-(1-a).sqrt()*v,x)

def test_member_permutation_equivariance():
 m,c,_=fixture();p=torch.tensor([0,3,1,2]);pc=copy.deepcopy(c);pc['target_nodes']=c['target_nodes'][:,:,p]
 for key in ('initial_xy','velocity','position_age_s','current_presence'):pc[key]=c[key][:,p]
 with torch.no_grad():
  h=m.encode(c);hp=m.encode(pc);torch.testing.assert_close(hp,h[:,p],atol=1e-5,rtol=1e-5)
  z=torch.randn(1,4,6,2);t=torch.tensor([8]);v=m.predict_v(z,t,h,c['current_presence']);vp=m.predict_v(z[:,p],t,hp,pc['current_presence']);torch.testing.assert_close(vp,v[:,p],atol=1e-5,rtol=1e-5)

def test_constant_velocity_accounts_for_age():
 c={'initial_xy':torch.zeros(1,4,2),'velocity':torch.ones(1,4,2)*.01,'position_age_s':torch.ones(1,4)*5}
 torch.testing.assert_close(constant_velocity(c)[0,0,:,0],torch.arange(15,66,10)*.01)

def test_sampling_uses_only_conditions_and_is_reproducible():
 m,c,s=fixture();torch.manual_seed(21);a=m.sample(c,3);s['targets']['trajectory']+=1000;torch.manual_seed(21);b=m.sample(c,3);torch.testing.assert_close(a,b,rtol=0,atol=0);assert a.shape==(1,3,4,6,2) and torch.isfinite(a).all()

def test_unbiased_energy_score_for_two_point_distribution():
 from pubg_lab.movement.evaluation.evaluation import metrics
 y=np.zeros((4,6,2));valid=np.zeros((4,6),bool);valid[0]=True;p=np.array([True,False,False,False]);pred=np.zeros((2,4,6,2));pred[0,0,:,0]=-.001;pred[1,0,:,0]=.001
 r=metrics(pred,{'trajectory':y,'trajectory_valid':valid,'team_center':np.zeros((6,2)),'team_valid':np.ones(6,bool)},p)
 assert abs(r['joint_energy_score_m'])<1e-8 and r['ego_expectedADE_m']==1 and r['ego_meanpathADE_m']==0
