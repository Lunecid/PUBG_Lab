import torch
from tools.sac2027_history_controls import PastSummary, PrefixOrder


class Capture(torch.nn.Module):
    def forward(self,batch):
        return batch["x"]


def test_history_summary_ignores_retrospective_query_eligibility():
    x=torch.tensor([[[[1.,2.]],[[100.,200.]],[[3.,4.]]]])
    node=torch.tensor([[[True],[False],[True]]])
    query=torch.ones_like(node)
    model=PastSummary(Capture())
    baseline=model(dict(x=x,node_mask=node,query_mask=query))
    # The last token has one causally observed predecessor, not two.
    assert torch.equal(baseline[0,2,0,2:4],torch.tensor([1.,2.]))
    assert torch.allclose(baseline[0,2,0,4],torch.tensor(1/7))
    changed=x.clone();changed[:,1]=1e6
    after=model(dict(x=changed,node_mask=node,query_mask=node))
    assert torch.equal(baseline[:,2],after[:,2])
    # A future token cannot alter the preceding summaries either.
    changed=x.clone();changed[:,2]=1e6
    after=model(dict(x=changed,node_mask=node,query_mask=query))
    assert torch.equal(baseline[:,:2],after[:,:2])


def test_prefix_shuffle_never_moves_future_information_into_a_forecast():
    class SumObserved(torch.nn.Module):
        def __init__(self):
            super().__init__();self.seen=[]
        def forward(self,batch):
            self.seen.append({k:v.clone() for k,v in batch.items() if isinstance(v,torch.Tensor)})
            return {'hazard_logit':(batch['x'][...,0]*batch['node_mask']).cumsum(1)}
    x=torch.arange(1.,17.).reshape(2,4,2,1)
    mask=torch.ones(2,4,2,dtype=torch.bool);mask[:,1,0]=False
    batch=dict(x=x,node_mask=mask,query_mask=torch.ones_like(mask),
               pos=x.expand(2,4,2,2).clone(),match_ids=['match-a','match-b'])
    for shuffle in (False,True):
        base=SumObserved();model=PrefixOrder(base,shuffle=shuffle)
        result=model(batch)['hazard_logit']
        expected=(x[...,0]*mask).cumsum(1)
        assert torch.equal(result,expected)
        for cut,prefix in enumerate(base.seen):
            assert torch.equal(prefix['x'][:,-1],x[:,cut])
            assert torch.equal(prefix['pos'][:,-1],batch['pos'][:,cut])
            assert torch.equal(prefix['x'][...,0].sort(dim=1).values,x[:,:cut+1,...,0].sort(dim=1).values)
        future=x.clone();future[:,3]=1e6
        changed=model({**batch,'x':future})['hazard_logit']
        assert torch.equal(result[:,:3],changed[:,:3])
