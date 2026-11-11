"""Training-only acceptance: same state, loss, gradients, data and optimizer semantics."""
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from openstroke.obb.engine import ctx_batch
from openstroke.obb.models.openobb25 import OpenOBB25
from openstroke.obb.models.stroke_graph import graph_loss,soft_f2_loss
from openstroke.obb.models.vector import VectorBranch
from openstroke.obb.runtime import ByteLRU,token_groups
from openstroke.obb.training_perf import StepTimer
sys.path.insert(0,str(Path(__file__).parent))
from test_openobb25 import IMG,NC,G,_batch
from legacy_member_loss import legacy_member_loss

torch.set_num_threads(1)


def test_bucketed_attention_keeps_every_valid_token_and_all_parameter_gradients():
    torch.manual_seed(6)
    old=VectorBranch([8,8,8],d=16,layers=1).train()
    for c in old.out: torch.nn.init.normal_(c.weight,std=0.02)
    new=copy.deepcopy(old)
    lengths=[65,17,3,0];v=torch.rand(4,65,22);v[:,:,0]=0;v[:,:,21]=0
    mask=torch.arange(65)[None,:] < torch.tensor(lengths)[:,None]
    features=[torch.randn(4,8,s,s) for s in (16,8,4)]
    a,ta=old([x.clone() for x in features],v,mask,return_tokens=True)
    b,tb=new([x.clone() for x in features],v,mask,return_tokens=True,groups=token_groups(lengths))
    assert old.state_dict().keys() == new.state_dict().keys()
    torch.testing.assert_close(ta[mask],tb[mask],rtol=1e-5,atol=2e-6)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6)
    (sum(x.square().mean() for x in a)+ta[mask].square().mean()).backward()
    (sum(x.square().mean() for x in b)+tb[mask].square().mean()).backward()
    for (name,x),(_,y) in zip(old.named_parameters(),new.named_parameters()):
        assert (x.grad is None)==(y.grad is None),name
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=2e-6,msg=name)
    assert sum(len(ids)*(width+1)**2 for ids,width in token_groups(lengths)) < len(lengths)*(65+1)**2


@pytest.mark.parametrize("empty",[False,True])
def test_member_loss_and_unused_parameter_status_match_frozen_reference(empty):
    torch.manual_seed(9)
    m=OpenOBB25("s",num_classes=NC,img_size=IMG,max_det=50,groups=G,vec_dim=32,vec_layers=1,endpoint_members=True).train()
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"))
    def picks(out,tg,size):
        result=[]
        for t in tg:
            boxes=t["boxes"]*torch.tensor([size,size,size,size,1.])
            if empty: boxes=boxes+torch.tensor([10000.,10000.,0.,0.,0.])
            result.append((boxes,torch.arange(len(boxes))))
        return result
    m.member_targets=picks
    tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    out=m._forward_train(torch.rand(2,3,IMG,IMG),tg,ctx)
    before=legacy_member_loss(m,out,tg,IMG);after=m.member_loss(out,tg,IMG)
    params=[p for name,p in m.named_parameters() if name.startswith(("member.","vec."))]
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=1e-6,atol=1e-6)
    a=torch.autograd.grad(sum(before.values()),params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(sum(after.values()),params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None),"None versus zero gradient changes AdamW weight decay"
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6)


@pytest.mark.parametrize("target",[[0.,0.,0.],[1.,0.,1.],[1.,1.,1.]])
def test_balanced_edge_loss_and_soft_f2_gradients_match_original(target):
    y=torch.tensor(target);a=torch.tensor([0.2,-0.7,1.1],requires_grad=True);b=a.detach().clone().requires_grad_()
    lb=F.binary_cross_entropy_with_logits(a,y,reduction="none");pos=y>0.5
    old=(lb[pos].mean() if pos.any() else lb.sum()*0)+(lb[~pos].mean() if (~pos).any() else lb.sum()*0)
    _,new=graph_loss(torch.zeros(2,2),b,torch.tensor([0,1]),y)
    torch.testing.assert_close(old,new)
    torch.testing.assert_close(torch.autograd.grad(old,a)[0],torch.autograd.grad(new,b)[0])
    a=a.detach().requires_grad_();b=b.detach().requires_grad_();w=torch.tensor([1.,2.,3.]);w=w/w.sum();p=a.sigmoid()
    tp=(w*p*y).sum();fn=(w*(1-p)*y).sum();fp=(w*p*(1-y)).sum()
    old=1-5*tp/(5*tp+4*fn+fp).clamp_min(1e-8) if (w*y).sum()>0 else fp
    new=soft_f2_loss(b,y.bool(),torch.ones(3,dtype=torch.bool),torch.tensor([1.,2.,3.]))
    torch.testing.assert_close(old,new)
    torch.testing.assert_close(torch.autograd.grad(old,a)[0],torch.autograd.grad(new,b)[0])


def test_vector_cache_returns_private_augmentation_copies(tmp_path):
    from openstroke.obb.data.dota import DotaPatches
    d=DotaPatches.__new__(DotaPatches);d.root=tmp_path;d.split="train";d.size=128;d.members=True
    d.vector_cache=ByteLRU(1024*1024)
    path=tmp_path/"vectors/train";path.mkdir(parents=True)
    t=_batch()[0]["vec"].numpy();np.savez_compressed(path/"p.npz",tokens=t[:,:22],inst=t[:,22],pid=t[:,23],view=128)
    a=d._load_vec({"name":"p"});expected=copy.deepcopy(a)
    a["pts"][:]=999;a["attr"][:]=-123
    b=d._load_vec({"name":"p"})
    assert all(np.array_equal(expected[k],b[k]) for k in expected)
    assert d.vector_cache.hits==1


def test_progress_does_not_format_cuda_values_when_frame_is_throttled():
    from openstroke.obb.console import EpochBar
    calls=[];bar=EpochBar(0,1,10,128,stream=io.StringIO(),every=999)
    means=lambda:calls.append(True) or {}
    bar.update(1,means,0,None)
    assert not calls
    bar.update(10,means,0,None,force=True)
    assert calls==[True]


def test_phase_timer_adds_no_work_or_rng_changes():
    timer=StepTimer("cpu",samples=1,skip=0);state=torch.get_rng_state().clone()
    timer.begin(0)
    for name in ("transfer","forward_vector","criterion","backward"):timer.mark(name)
    timer.finish();values,count=timer.summary()
    assert count==1 and all(v>=0 for v in values.values())
    assert torch.equal(state,torch.get_rng_state())


@pytest.mark.parametrize("unknown", [False, True])
def test_optional_member_stats_keeps_original_metadata_fallback(unknown):
    from openstroke.obb.models.openobb25 import member_stats
    targets = _batch()
    if unknown:
        for target in targets:
            target["vec"][:, 22] = -2
    out = {"pred_logits": torch.full((2, 1, NC), -20.),
           "pred_boxes": torch.zeros(2, 1, 5),
           "member_logits": torch.zeros(2, 1, 1)}
    acc = {}
    member_stats(out, targets, acc, IMG)
    assert acc.get("objects", 0) == (0 if unknown else sum(len(t["boxes"]) for t in targets))
