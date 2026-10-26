import copy
import torch
from pubg_lab.survival.ptr.models.gatv2 import SnapshotGATv2, graph_edges
from pubg_lab.survival.ptr.models.model import PTRConfig


def fixture():
    torch.manual_seed(31)
    batch = dict(x=torch.randn(2, 3, 4, 7),
        node_mask=torch.ones(2, 3, 4, dtype=torch.bool),
        query_mask=torch.ones(2, 3, 4, dtype=torch.bool),
        rel=torch.rand(2, 3, 4, 4, 4))
    batch['node_mask'][..., -1] = False
    batch['query_mask'][..., -1] = False
    model = SnapshotGATv2(PTRConfig(in_features=7, n_phases=3,
        d_model=16, n_heads=4, n_layers=2, dropout=0)).eval()
    return batch, model


def test_edge_orientation_and_zero_self_features():
    b, _ = fixture()
    b['rel'].zero_(); b['rel'][0, 0, 1, 0, 1] = 2
    index, attr = graph_edges(b)
    off = index[0] != index[1]
    assert index[:, off].tolist() == [[1], [0]]
    assert attr[off].tolist() == [[0, 2, 0, 0]]
    assert torch.count_nonzero(attr[~off]) == 0
    assert torch.equal(index[0]//4, index[1]//4)


def test_padding_nan_and_gradient_isolation():
    b, model = fixture(); baseline = model(b)['hazard_logit']
    b['x'][..., -1, :] = float('nan')
    b['rel'][..., -1, :] = float('nan'); b['rel'][..., -1] = float('nan')
    b['x'].requires_grad_()
    out = model(b)['hazard_logit']
    torch.testing.assert_close(out, baseline)
    out.sum().backward()
    assert torch.isfinite(b['x'].grad).all()
    assert torch.count_nonzero(b['x'].grad[..., -1, :]) == 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_query_only_row_never_sends_messages():
    b, model = fixture()
    b['query_mask'][..., -1] = True
    baseline = model(b)['hazard_logit'][..., :3]
    b['x'][..., -1, :] = 10000
    torch.testing.assert_close(model(b)['hazard_logit'][..., :3], baseline)
    index, _ = graph_edges(b)
    assert not (((index[0] % 4 == 3) & (index[0] != index[1])).any())


def test_snapshot_and_match_independence():
    b, model = fixture(); baseline = model(b)['hazard_logit'][0, 0]
    b['x'][1] = 999; b['x'][0, 1:] = -999
    b['rel'][1] = 99; b['rel'][0, 1:] = 77
    torch.testing.assert_close(model(b)['hazard_logit'][0, 0], baseline)


def test_team_permutation_equivariance():
    b, model = fixture(); order = torch.tensor([2, 0, 3, 1])
    baseline = model(b)['hazard_logit']
    swapped = {k: v[..., order, :] if k == 'x' else
        v[..., order, :][..., order] if k == 'rel' else v[..., order]
        for k, v in b.items()}
    torch.testing.assert_close(model(swapped)['hazard_logit'], baseline[..., order], atol=2e-6, rtol=2e-6)


def test_edge_attributes_have_effect_and_gradient():
    b, model = fixture(); b['rel'].requires_grad_()
    result = model(b)['hazard_logit']; result.square().sum().backward()
    assert b['rel'].grad.abs().sum() > 0
    changed = {**b, 'rel': b['rel'].detach()*5}
    assert not torch.allclose(model(changed)['hazard_logit'], result)
