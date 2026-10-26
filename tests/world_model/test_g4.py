import csv
import json
import sqlite3
from dataclasses import fields,replace
import numpy as np
import pytest
import torch
import test_dataset
from pubg_lab.world_model.features.temporal_prior import TemporalPriorDataset, PastWorldEncoder
from pubg_lab.world_model.features.inputs import ConditionVocabulary
from pubg_lab.world_model.training.pretraining import PretrainHeads, pretrain_losses


@pytest.fixture
def setup():
    h=test_dataset.DatasetTests();h.setUp()
    rows=list(csv.DictReader((h.root/"matches.csv").open()))
    for r in rows:r["start_time"]="1970-01-01T00:00:50+00:00"
    with (h.root/"matches.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    plan=dict(cohort="primary_br",train_match_ids=["fit","excluded"],validation_match_ids=["dev"])
    (h.root/"ablations/full.json").write_text(json.dumps(plan))
    (h.root/"cohort.json").write_text(json.dumps(plan))
    p=h.root/"past.sqlite"
    with sqlite3.connect(p) as db:
        db.executescript("""CREATE TABLE donors(match_id,role,cohort,map,patch,start_ts,end_ts);
        CREATE TABLE contributions(match_id,map,sx,sy,dx,dy,mode,n);""")
        for mid,end,n in [("fit",10,7),("excluded",100,99)]:
            db.execute("INSERT INTO donors VALUES(?,?,?,?,?,?,?)",(mid,"train","normal_br","A","40.2",0,end))
            db.execute("INSERT INTO contributions VALUES(?,?,?,?,?,?,?,?)",(mid,"A",1,1,2,1,1,n))
    yield h,p
    h.tearDown()


def dataset(setup,role="validation",policy="H1"):
    h,p=setup
    ds=TemporalPriorDataset(h.root,h.root/"cohort.json",p,role=role,policy=policy)
    return ds,ConditionVocabulary.from_dataset(ds)


def equal_inputs(a,b):
    for f in fields(a):
        x,y=getattr(a,f.name),getattr(b,f.name)
        assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y, f.name


def test_time_limit_no_double_count_and_legacy_catalog_never_read(setup):
    ds,v=dataset(setup)
    a=ds.load("dev",13,0,v)
    assert len(a.inputs.past_edges)==1
    assert a.inputs.past_edges[0,3:].tolist()==[7,1]
    assert not len(a.inputs.observed) and not a.inputs.cells[:,2:6].any()
    edges={tuple(r[:2]) for r in a.inputs.past_edges.tolist()}
    assert not edges & {tuple(r) for r in a.inputs.inferred.tolist()}
    with sqlite3.connect(setup[0].root/"catalog.sqlite") as db:
        db.execute("DROP TABLE spatial_support")
    b=ds.bank("dev")
    b["target"][14:]+=999999
    b["x"][14:]+=99999
    b["spatial_edges"][:]=99999
    with sqlite3.connect(setup[1]) as db:
        db.execute("UPDATE contributions SET n=99999999 WHERE match_id='excluded'")
    equal_inputs(a.inputs,ds.load("dev",13,0,v).inputs)
    ds.prior.close()


def test_exact_patch_filter_and_self_exclusion(setup):
    ds,v=dataset(setup,policy="H0")
    assert not len(ds.load("dev",13,0,v).inputs.past_edges)
    ds.prior.close()
    ds,v=dataset(setup,role="train")
    assert not len(ds.load("fit",13,0,v).inputs.past_edges)
    with pytest.raises(PermissionError): ds.load("dev",13,0,v)
    with pytest.raises(PermissionError): ds.load("protected",13,0,v)
    ds.prior.close()


def test_past_gradient_and_duplicate_guard(setup):
    torch.set_num_threads(1);torch.manual_seed(20261020)
    torch.backends.mha.set_fastpath_enabled(False)
    ds,v=dataset(setup)
    sample=ds.load("dev",13,0,v)
    sample.inputs=replace(sample.inputs,past_numeric=sample.inputs.past_numeric.clone().requires_grad_())
    enc=PastWorldEncoder(len(ds.item_vocabulary)+2,v.sizes)
    heads=PretrainHeads(d=64)
    total,_,_=pretrain_losses(enc,heads,sample,13)
    total.backward()
    for name in ("message.weight","gate.weight","support.weight","mode.weight","patch.weight"):
        for block in enc.spatial:
            grad=dict(block.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    assert sample.inputs.past_numeric.grad.abs().sum()>0
    g=replace(sample.inputs,observed=sample.inputs.past_edges)
    with pytest.raises(ValueError,match="duplicate"):enc(g)
    cells=sample.inputs.cells.clone();cells[:,2]=1
    with pytest.raises(ValueError,match="duplicate"):enc(replace(sample.inputs,cells=cells))
    ds.prior.close()


def test_locked_mixture_schedule():
    from types import SimpleNamespace
    from collections import Counter
    from projects.world_model.training.train_world_model_g4 import schedule, STRATA
    samples=[SimpleNamespace(provenance=dict(stratum=st,match_id="m",team_id=1))
             for st in ("natural",)+STRATA]
    plan=schedule(samples,192,20261020,True)
    c=Counter(samples[i].provenance["stratum"] for batch in plan for i in batch)
    assert c["natural"]==576
    assert all(c[st]==48 for st in STRATA)
    assert plan==schedule(samples,192,20261020,True)
    assert all(i==0 for b in schedule(samples,192,20261020,False) for i in b)


def test_unlearned_change_flags_have_no_random_effect(setup):
    ds,v=dataset(setup);s=ds.load("dev",13,0,v)
    enc=PastWorldEncoder(len(ds.item_vocabulary)+2,v.sizes)
    for block in enc.spatial:
        assert not block.support.weight[:,3:].any()
        assert not block.gate.weight[:,3:].any()
    flags=s.inputs.past_numeric.clone();flags[:,1:]=1
    a=enc(s.inputs)["player"];b=enc(replace(s.inputs,past_numeric=flags))["player"]
    assert torch.equal(a,b)
    ds.prior.close()


def test_metric_gives_each_team_then_match_equal_weight():
    from projects.world_model.reporting.report_world_model_g4 import match_means
    rows=[dict(match_id="a",team_id=1,ADE30=x) for x in (1.,3.)]
    rows += [dict(match_id="a",team_id=2,ADE30=10.),
             dict(match_id="b",team_id=1,ADE30=20.),
             dict(match_id="b",team_id=2,ADE30=None)]
    result=match_means(rows,"ADE30")
    assert result=={"a":6.,"b":20.}
    assert np.mean(list(result.values()))==13.
