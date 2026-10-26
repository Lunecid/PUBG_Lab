"""Same-input joint regression and smooth-knot diffusion with direct circle conditioning.
Both independently train the same graph/history/zone encoder and mean/mode heads.
Diffusion additionally models residual uncertainty around its learned mean.
"""
import math,torch
from torch import nn
from .model import JointGraphDiffusion as OldEncoder,constant_velocity

class JointMotion(nn.Module):
 def __init__(self,categories,kind='regression',hidden=48):
  super().__init__();self.kind=kind;self.steps=32;self.scale=.2
  self.encoder=OldEncoder(categories,hidden=hidden)
  del self.encoder.noisy_project,self.encoder.joint_attention,self.encoder.denoiser
  self.zone=nn.Sequential(nn.Linear(14,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
  self.fuse=nn.Sequential(nn.Linear(hidden*2+14+24,hidden),nn.SiLU(),nn.LayerNorm(hidden))
  self.attention=nn.MultiheadAttention(hidden,4,batch_first=True)
  self.mean_head=nn.Sequential(nn.Linear(hidden*2+14,hidden),nn.SiLU(),nn.Linear(hidden,6))
  self.mode_head=nn.Linear(hidden*2+14,6*10)
  self.noise_head=nn.Sequential(nn.Linear(12+hidden*2+14+16,hidden*2),nn.SiLU(),nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,6)) if kind=='diffusion' else None
  basis=torch.tensor([[.5,0,0],[1,0,0],[.5,.5,0],[0,1,0],[0,.5,.5],[0,0,1.]])
  self.register_buffer('basis',basis)
  nn.init.zeros_(self.mean_head[-1].weight);nn.init.zeros_(self.mean_head[-1].bias)
 def encode(self,c):
  h=self.encoder.encode(c);z=self.zone(c['zone_geometry']);h=self.fuse(torch.cat([h,z,c['mobility_state'],c['motion_history'].flatten(2)],-1));a=self.attention(h,h,h,key_padding_mask=~c['current_presence'],need_weights=False)[0]
  return torch.cat([h,a,c['zone_geometry']],-1)*c['current_presence'][...,None]
 def curve(self,knots,c):
  anchor=c['initial_xy']+c['velocity']*c['position_age_s'][:,:,None]
  return anchor[:,:,None]+torch.einsum('tk,bpkd->bptd',self.basis,knots)
 def mean(self,h,c):return self.curve(self.mean_head(h).reshape(-1,4,3,2)*self.scale,c)
 def denoise(self,y,step,h,presence):
  f=torch.exp(torch.arange(8,device=y.device)*(-math.log(10000)/7));e=step[:,None]*f[None];e=torch.cat([e.sin(),e.cos()],-1)[:,None].expand(-1,4,-1)
  pooled=(y*presence[:,:,None,None]).sum(1)/presence.sum(1)[:,None,None].clamp_min(1)
  joint=pooled.flatten(1)[:,None].expand(-1,4,-1)
  return self.noise_head(torch.cat([y.flatten(2),joint,h,e],-1)).reshape(-1,4,3,2)
 def loss(self,c,t):
  h=self.encode(c);mean=self.mean(h,c);mask=t['trajectory_valid']&c['current_presence'][:,:,None];err=(mean-t['trajectory'])/self.scale
  # Huber keeps rare aircraft transitions from dominating routine movement.
  point=(torch.nn.functional.smooth_l1_loss(mean/self.scale,t['trajectory']/self.scale,reduction='none')*mask[...,None]).sum()/(mask.sum()*2).clamp_min(1)
  w=c['current_presence'].float();center=(mean*w[:,:,None,None]).sum(1)/w.sum(1)[:,None,None].clamp_min(1);tm=t['team_valid'];team=(torch.nn.functional.smooth_l1_loss(center/self.scale,t['team_center']/self.scale,reduction='none')*tm[...,None]).sum()/(tm.sum()*2).clamp_min(1)
  logits=self.mode_head(h).reshape(-1,4,6,10);mm=t['mode_valid']&mask;ce=torch.nn.functional.cross_entropy(logits.flatten(0,2),t['mode'].flatten(),reduction='none').reshape_as(mask);mode=(ce*mm).sum()/mm.sum().clamp_min(1)
  # Current announced circles only: signed-distance error of the actual path.
  radial=point*0
  for z in range(2):
   geo=c['zone_geometry'][:,:,z*7:(z+1)*7];center_z=c['initial_xy']+geo[:,:,:2];pr=torch.linalg.vector_norm(mean-center_z[:,:,None],dim=-1)-geo[:,:,2,None];tr=torch.linalg.vector_norm(t['trajectory']-center_z[:,:,None],dim=-1)-geo[:,:,2,None];zm=mask&geo[:,:,6,None].bool();radial+=(torch.nn.functional.smooth_l1_loss(pr/self.scale,tr/self.scale,reduction='none')*zm).sum()/zm.sum().clamp_min(1)
  loss=point+.1*team+.05*mode+.05*radial;parts={'point':float(point.detach()),'team':float(team.detach()),'mode':float(mode.detach()),'radial':float(radial.detach())}
  if self.kind=='diffusion':
   km=mask[:,:,[1,3,5]];clean=torch.where(km[...,None],(t['trajectory'][:,:,[1,3,5]]-mean.detach()[:,:,[1,3,5]])/self.scale,0.);step=torch.randint(self.steps,(len(h),));ab=self.encoder.alpha_bar[step][:,None,None,None];noise=torch.randn_like(clean);y=ab.sqrt()*clean+(1-ab).sqrt()*noise;v=ab.sqrt()*noise-(1-ab).sqrt()*clean;out=self.denoise(y,step,h,c['current_presence']);dl=((out-v).square()*km[...,None]).sum()/(km.sum()*2).clamp_min(1);loss=loss+dl;parts['diffusion']=float(dl.detach())
  return loss,parts
 @torch.no_grad()
 def sample(self,c,scenarios=20):
  h=self.encode(c);mean=self.mean(h,c)
  if self.kind=='regression':return mean[:,None]
  h=h.repeat_interleave(scenarios,0);p=c['current_presence'].repeat_interleave(scenarios,0);y=torch.randn(len(h),4,3,2)*p[:,:,None,None]
  for step in reversed(range(self.steps)):
   a=self.encoder.alpha_bar[step];prev=self.encoder.alpha_bar[step-1] if step else torch.tensor(1.);v=self.denoise(y,torch.full((len(h),),step),h,p);clean=a.sqrt()*y-(1-a).sqrt()*v;eps=(1-a).sqrt()*y+a.sqrt()*v;y=(prev.sqrt()*clean+(1-prev).sqrt()*eps)*p[:,:,None,None]
  residual=torch.einsum('tk,bpkd->bptd',self.basis,y).reshape(c['batch_size'],scenarios,4,6,2)*self.scale
  return mean[:,None]+residual
