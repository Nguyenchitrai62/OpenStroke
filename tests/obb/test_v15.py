"""Acceptance of v15 instance training/deployment without training a real model on CPU."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openstroke.obb.engine import ctx_batch
from openstroke.obb.models.openobb25 import OpenOBB25
from openstroke.obb.models.stroke_instances import LineContext,instance_losses
from openstroke.obb.instance_decode import tile_instances,merge_instances
sys.path.insert(0,str(Path(__file__).parent))
from test_openobb25 import IMG,NC,G,_batch
from test_vector_validation import make_page,predictions


torch.set_num_threads(1)
OPTS=dict(num_classes=NC,img_size=IMG,max_det=50,groups=G,vec_dim=32,vec_layers=1)


def model(**kw):return OpenOBB25("s",**OPTS,instance_queries=8,instance_free=4,instance_line_context=True,**kw)


def test_line_context_preserves_valid_old_features_at_zero_initialization():
    t=_batch();side=ctx_batch(t,torch.device("cpu"));tok=torch.randn(*side["vec_mask"].shape,32)
    layer=LineContext(32,window=4).eval()
    with torch.no_grad():out=layer(tok,side["vec"],side["vec_mask"],side["vec_lengths"])
    torch.testing.assert_close(tok[side["vec_mask"]],out[side["vec_mask"]],rtol=0,atol=0)


def test_instances_have_finite_losses_gradients_and_all_represented_gt_without_box_iou_gate():
    torch.manual_seed(10);m=model(instance_f2=True).train()
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"))
    tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    out=m(torch.rand(2,3,IMG,IMG),tg,ctx)
    losses={k:v for k,v in out.items() if k.startswith("instance_")}
    assert set(losses)=={"instance_cls","instance_bce","instance_dice","instance_center","instance_f2"}
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    assert m.instances.classify.weight.grad is not None and torch.isfinite(m.instances.classify.weight.grad).all()
    assert m.instances.mask_embed.weight.grad is not None and m.instances.mask_embed.weight.grad.abs().sum()>0
    # A deliberately useless box proposal cannot remove the two labelled instances from mask matching.
    assert ctx["vec_instance_ids"]==((0,1),(0,1))
    classes=torch.zeros(1,2,NC+1,requires_grad=True)
    masks=torch.randn(1,2,ctx["vec"].shape[1],requires_grad=True)
    stage=dict(classes=classes,masks=masks,centers=torch.full((1,2,2),99.))
    side={k:(v[:1] if torch.is_tensor(v) else v[:1]) for k,v in ctx.items() if k!="vec_groups"}
    side["mask"]=side.pop("vec_mask")
    value=instance_losses([stage],side,tg[:1])
    assert value["instance_bce"]>0 and value["instance_dice"]>0
    value["instance_bce"].backward()
    assert masks.grad[0,0].abs().sum()>0 and masks.grad[0,1].abs().sum()>0


def test_inference_ignores_layer_instance_and_primitive_id_columns_in_features(tmp_path):
    torch.manual_seed(8);m=model().eval();targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"))
    changed=copy.deepcopy(ctx);changed["vec"][...,21:]=torch.randn_like(changed["vec"][...,21:])*100
    x=torch.rand(2,3,IMG,IMG)
    with torch.no_grad():a=m(x,ctx=ctx)["instances"];b=m(x,ctx=changed)["instances"]
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    args=dict(arch="v25",size="s",img=IMG,num_top=50,groups=G,vec_dim=32,vec_layers=1,
              instance_queries=8,instance_free=4,instance_line_context=True,vectors=True)
    ckpt=tmp_path/"v15.pt";torch.save(dict(model=m.state_dict(),args=args,classes=["Duct","Elbow","Tee"],
        object_score={"instances":dict(best_conf=0.3,F2=0.6,P=0.7,R=0.58,protocol="dataset_vectors.sampled8.v1")}),ckpt)
    from openstroke.obb.strip import strip
    from openstroke.obb.predict import load_model,conf_thresholds
    deploy=strip(str(ckpt));restored,names,saved=load_model(deploy,torch.device("cpu"))
    assert restored.instances.queries==8 and saved["instance_line_context"]
    assert conf_thresholds(names,saved,None)==[0.3]*3
    with torch.no_grad():result=restored(x,ctx=ctx)["instances"]
    for key in a:torch.testing.assert_close(a[key],result[key],rtol=2e-4,atol=2e-4)


def test_mask_decode_keeps_adjacent_same_class_objects_and_stitches_shared_ids():
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"))
    inst=ctx["vec"][:1,:,22];known=ctx["vec_mask"][:1]
    masks=torch.stack([torch.where(inst[0]==j,10.,-10.) for j in [0,1]])[None]
    out=dict(instances=dict(classes=torch.tensor([[[10.,-10.,-10.,-10.],[10.,-10.,-10.,-10.]]]),masks=masks),vec=ctx["vec"][:1],vec_mask=known)
    dets=tile_instances(out,0,IMG,1.,0,0)
    assert len(dets)==2 and not(set(dets[0][4])&set(dets[1][4]))
    page={"tokens":ctx["vec"][0].numpy().copy()};page["tokens"][:,1:17]*=IMG
    joined=merge_instances(dets+dets,["Duct","Elbow","Tee"],page)
    assert len(joined)==2 and all(row["source"]=="instances" for row in joined)


def test_native_scoring_reads_only_instance_predictions(tmp_path):
    from openstroke.obb.vector_validation import DatasetVectorGT
    images,_,stem=make_page(tmp_path)
    gt=DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    pred=predictions(stem)
    for row in pred[stem]:row["source"]="instances"
    pred[stem].append(dict(class_id=0,score=0.99,members=[99],source="members"))
    assert gt.evaluate(pred,"instances",["Duct","Elbow"])["F2"]==1.


def test_empty_primitive_image_has_finite_background_training():
    m=model().train();targets=[dict(labels=torch.zeros(0,dtype=torch.long),boxes=torch.zeros(0,5),vec=torch.zeros(0,24)) for _ in range(2)]
    ctx=ctx_batch(targets,torch.device("cpu"));tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    out=m(torch.rand(2,3,IMG,IMG),tg,ctx)
    assert torch.isfinite(out["instance_cls"])
    assert out["instance_bce"]==0


def test_gt_subset_excludes_pipeline_samples_and_validates_label_provenance(tmp_path,monkeypatch):
    from openstroke.obb import gt_dataset as data
    source=tmp_path/"source";rows=[]
    for split,stem in [("train","gttrain"),("val","gtval")]:
        for kind,suffix in [("images",".png"),("labels",".txt"),("vectors",".npz")]:
            path=source/kind/split/(stem+suffix);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"0 0 0 1 0 1 1 0 1")
        tok=np.zeros((1,24),np.float32);tok[:,22]=-1
        np.savez(source/"vectors"/split/(stem+".npz"),tokens=tok)
        rows.append(dict(split=split,stem=stem,label_sha256=data.label_hash(source/"labels"/split/(stem+".txt"))))
    (source/"data.yaml").write_text("names: {0: Duct}\n")
    (source/"images/train/pseudo.png").write_bytes(b"excluded")
    monkeypatch.setattr(data,"GT_SPEC",rows)
    out=tmp_path/"gt"
    path=data.build(source,out)
    assert path.is_file() and not(out/"images/train/pseudo.png").exists()
    assert data.build(source,out)==path
    (out/"images/train/pseudo.png").write_bytes(b"bad")
    with pytest.raises(ValueError,match="Unexpected or missing"):data.validate(out)


def test_cropped_gt_strokes_become_unknown_targets_instead_of_false_background(tmp_path):
    from openstroke.obb.data.vectors import tile_npz
    page={"tokens":_batch()[0]["vec"].numpy().copy()}
    page["tokens"][:,1:17]*=IMG
    tile=tmp_path/"tile.npz"
    tile_npz(page,0,0,IMG,tile,inst_map={0:0})
    with np.load(tile) as z:inst=z["inst"]
    assert (inst[:4]==0).all() and (inst[4:8]==-2).all() and (inst[8:]==-1).all()


def test_cli_flags_are_real_and_mixed_dataset_is_rejected_before_prepare(tmp_path,monkeypatch):
    from openstroke.obb import cli
    from openstroke.obb.train import build_parser
    parser=build_parser()
    args=parser.parse_args(["--data","unused","--out",str(tmp_path/"run"),"--instance-queries","256","--instance-free","128","--instance-line-context","--instance-f2"])
    assert args.instance_queries==256 and args.instance_line_context and args.instance_f2
    monkeypatch.setattr(cli,"page_stats",lambda *a:(0,0,0))
    monkeypatch.setattr(cli,"prepare_data",lambda *a,**k:pytest.fail("mixed GT/pseudo dataset must be rejected before tiling"))
    with pytest.raises(ValueError,match="GT-only dataset"):
        cli.train(data=str(tmp_path),model="openobb25s",tile=True,tile_scale=1,instance_queries=8,instance_free=4)


def test_public_api_and_epoch_evaluator_use_instance_output_not_box_merging(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from openstroke.obb.model import OpenOBB
    from openstroke.obb import predict as pp,object_validation,vector_validation,cli
    api=OpenOBB.__new__(OpenOBB)
    api.model=SimpleNamespace(graph=None,instances=object(),member=object(),vec=object())
    api.device=torch.device("cpu");api.compile=False;api._names={0:"Duct"}
    api.args=dict(img=128,scale=1.,fit=False,instance_queries=8,object_conf=0.2)
    monkeypatch.setattr(pp,"page_vectors",lambda *a:dict(tokens=np.zeros((1,24))))
    monkeypatch.setattr(pp,"predict_image",lambda *a,**k:[])
    monkeypatch.setattr(pp,"merge_dets",lambda *a,**k:pytest.fail("v15 must not use box NMS"))
    from openstroke.obb import instance_decode
    obj=dict(class_id=0,score=0.8,poly=[0.,0.,10.,0.,10.,10.,0.,10.],members=[0],source="instances")
    monkeypatch.setattr(instance_decode,"merge_instances",lambda *a:[obj])
    result=api.predict(np.zeros((128,128,3),np.uint8),verbose=False)[0]
    assert result.members==[[0]] and len(result.obb)==1
    images,_,stem=make_page(tmp_path)
    gt=vector_validation.DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    monkeypatch.setattr(vector_validation,"DatasetVectorGT",lambda *a,**k:gt)
    pred=predictions(stem)
    for row in pred[stem]:row["source"]="instances"
    monkeypatch.setattr(cli,"predict",lambda *a,**k:pred)
    evaluator=object_validation.VectorEpochEvaluator(None,images,["Duct","Elbow"],dict(batch=1,instance_queries=8))
    try:
        assert evaluator.how=="instances" and evaluator.evaluate_checkpoint("unused")["F2"]==1.
    finally:evaluator.close()


def test_small_raster_with_few_dense_anchors_still_has_requested_query_count():
    from openstroke.obb.models.stroke_instances import StrokeInstances
    head=StrokeInstances(16,8,8,2,queries=16,free=8,line_context=False).eval()
    vec=torch.zeros(1,2,24);vec[:,:,1:17]=torch.rand(1,2,16)
    side=dict(tok=torch.rand(1,2,16),vec=vec,mask=torch.ones(1,2,dtype=torch.bool))
    with torch.no_grad():out=head(side,torch.rand(1,8,4,4),[torch.rand(1,8,1,1)],torch.zeros(1,1,2),torch.ones(1,2),16)
    assert out[-1]["classes"].shape==(1,16,3)


def test_instance_decoder_float_and_bfloat16_training_are_finite():
    from openstroke.obb.models.stroke_instances import StrokeInstances
    head=StrokeInstances(16,8,8,2,queries=8,free=4,line_context=True).train()
    vec=torch.zeros(1,4,24);vec[:,:,1:17]=torch.rand(1,4,16)
    side=dict(tok=torch.rand(1,4,16),vec=vec,mask=torch.ones(1,4,dtype=torch.bool),vec_lengths=(4,))
    with torch.autocast("cpu",dtype=torch.bfloat16):
        result=head(side,torch.rand(1,8,4,4),[torch.rand(1,8,4,4)],torch.zeros(1,16,2),torch.rand(16,2)*16,16)
    assert all(torch.isfinite(row["classes"]).all() and torch.isfinite(row["masks"]).all() for row in result)


def test_eval_sdpa_matches_native_encoder_with_padding_and_empty_primitive_rows():
    from openstroke.obb.models.vector import encoder_sdpa
    torch.manual_seed(14)
    layer=torch.nn.TransformerEncoderLayer(32,4,64,dropout=0.,activation="gelu",batch_first=True,norm_first=True)
    encoder=torch.nn.TransformerEncoder(layer,2,enable_nested_tensor=False).eval()
    x=torch.randn(3,65,32);keep=torch.arange(65)[None]<torch.tensor([65,12,1])[:,None]
    with torch.no_grad():
        expected=encoder(x,src_key_padding_mask=~keep)
        actual=encoder_sdpa(encoder,x,keep)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)


def test_validation_model_loading_cannot_change_training_rng(tmp_path,monkeypatch):
    from openstroke.obb import object_validation,vector_validation,cli
    images,_,stem=make_page(tmp_path)
    gt=vector_validation.DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    monkeypatch.setattr(vector_validation,"DatasetVectorGT",lambda *a,**k:gt)
    pred=predictions(stem)
    for row in pred[stem]:row["source"]="instances"
    def predict(*a,**kw):
        torch.rand(100)
        return pred
    monkeypatch.setattr(cli,"predict",predict)
    evaluator=object_validation.VectorEpochEvaluator(None,images,["Duct","Elbow"],dict(batch=1,instance_queries=8))
    before=torch.get_rng_state().clone()
    try:assert evaluator.evaluate_checkpoint("unused")["F2"]==1.
    finally:evaluator.close()
    assert torch.equal(before,torch.get_rng_state())


def test_vector_defaults_make_the_short_command_complete():
    from openstroke.obb.cli import vector_defaults
    extra={"instances":True,"gt_only":False,"nbs":64}
    data,batch,tile,scale,warm=vector_defaults("d",extra,{},None,False,None,3)
    assert (data,batch,tile,scale,warm)==("d",32,True,1,0.5)
    assert extra.pop("class_set")=="hvac6"
    assert extra=={"nbs":32,"lr":1e-4,"max_tokens":8192,"stroke_graph":False,"instance_queries":256,"instance_free":128,"instance_line_context":True,"new_lr_mult":10.0}
    extra={"instances":False,"gt_only":False,"lr":3e-4,"stroke_graph":True}
    _,batch,_,_,warm=vector_defaults("d",extra,{"max_tokens":4096},16,True,1,1)
    assert (batch,warm)==(16,1) and extra=={"lr":3e-4,"stroke_graph":True,"instance_queries":0,"class_set":"hvac6"}


def test_gt_ensure_leaves_other_datasets_untouched(tmp_path):
    from openstroke.obb.gt_dataset import ensure
    (tmp_path/"data.yaml").write_text("path: .\n")
    assert ensure(tmp_path/"data.yaml")==str(tmp_path/"data.yaml")


def test_new_decoder_gets_its_own_learning_rate_and_the_rest_is_unchanged():
    from openstroke.obb.engine import param_groups
    m=model()
    plain=param_groups(m,1e-4,0.5,1e-4)
    fast=param_groups(m,1e-4,0.5,1e-4,new_mult=10.0)
    lr={id(p):g["lr"] for g in fast for p in g["params"]}
    old={id(p):g["lr"] for g in plain for p in g["params"]}
    for n,p in m.named_parameters():
        assert lr[id(p)]==(1e-3 if n.startswith("instances.") else old[id(p)]),n
    assert fast[-1]["lr"]==plain[-1]["lr"]


def model16(**kw):return OpenOBB25("s",**OPTS,instance_queries=8,instance_free=4,instance_line_context=True,
                                   instance_seed="strokes",instance_roi=True,instance_masked=True,**kw)


def test_v16_trains_with_finite_losses_and_gradients_into_every_new_part():
    torch.manual_seed(16);m=model16().train()
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"))
    tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    out=m(torch.rand(2,3,IMG,IMG),tg,ctx)
    losses={k:v for k,v in out.items() if k.startswith("instance_")}
    assert "instance_seed" in losses and all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    for part in (m.instances.objectness,m.instances.seed_stroke,m.instances.roi[-1],m.instances.mask_embed):
        assert part.weight.grad is not None and torch.isfinite(part.weight.grad).all()
    assert m.instances.objectness.weight.grad.abs().sum()>0


def test_v16_inference_and_bfloat16_are_finite():
    torch.manual_seed(3);m=model16().eval();targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"))
    with torch.no_grad():out=m(torch.rand(2,3,IMG,IMG),None,ctx)
    assert torch.isfinite(out["instances"]["masks"]).all() and torch.isfinite(out["instances"]["classes"]).all()
    m.train()
    with torch.autocast("cpu",dtype=torch.bfloat16):
        o=m(torch.rand(2,3,IMG,IMG),[{k:v for k,v in t.items() if k!="vec"} for t in targets],ctx)
    assert all(torch.isfinite(v.float()) for k,v in o.items() if k.startswith("instance_"))


def test_v16_region_of_a_straight_run_of_strokes_is_long_and_thin():
    from openstroke.obb.models.stroke_instances import StrokeInstances
    center=torch.stack([torch.linspace(0.2,0.8,20),torch.full((20,),0.5)],-1)[None]
    logits=torch.full((1,1,20),10.)
    mean,inv,(axes,cos,sin),ok=StrokeInstances.region(logits,center,torch.ones(1,20,dtype=torch.bool))
    assert ok.all() and axes[0,0,0]>10*axes[0,0,1] and abs(float(cos[0,0]))>0.99
    torch.testing.assert_close(mean[0,0],torch.tensor([0.5,0.5]))


def test_v16_stroke_seeds_keep_one_stroke_per_cell():
    from openstroke.obb.models.stroke_instances import StrokeInstances
    head=StrokeInstances(8,4,4,3,queries=4,free=0,line_context=False,seed="strokes")
    memory=torch.randn(1,5,8);center=torch.tensor([[[0.1,0.1],[0.101,0.101],[0.9,0.9],[0.5,0.5],[0.3,0.7]]])
    score=torch.tensor([[5.,4.,3.,2.,-1.]]);mask=torch.ones(1,5,dtype=torch.bool)
    seed,refs=head.stroke_seeds(memory,center,mask,score,4)
    picked={tuple(round(v,3) for v in r.tolist()) for r in refs[0]}
    assert (0.1,0.1) in picked and (0.101,0.101) not in picked and len(picked)==4


def test_cli_v16_preset():
    from openstroke.obb.cli import vector_defaults
    extra={"instances":"v16","gt_only":False}
    vector_defaults("d",extra,{},None,False,None,3)
    assert extra["instance_seed"]=="strokes" and extra["instance_roi"] and extra["instance_masked"] and extra["instance_queries"]==256


def test_v17_member_refiner_starts_exactly_as_the_model_it_refines():
    torch.manual_seed(17)
    plain=OpenOBB25("s",**OPTS).eval()
    refined=OpenOBB25("s",**OPTS,member_refine=True).eval()
    missing=refined.load_state_dict(plain.state_dict(),strict=False)
    assert all(k.startswith("refine.") for k in missing.missing_keys) and not missing.unexpected_keys
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"));x=torch.rand(2,3,IMG,IMG)
    with torch.no_grad():
        a=plain(x,None,ctx);b=refined(x,None,ctx)
    torch.testing.assert_close(a["member_logits"],b["member_logits"],rtol=0,atol=0)
    plain.train();refined.train()
    tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    torch.manual_seed(1);la=plain(x,tg,ctx)
    torch.manual_seed(1);lb=refined(x,tg,ctx)
    torch.testing.assert_close(la["member_bce"],lb["member_bce"]);torch.testing.assert_close(la["member_dice"],lb["member_dice"])
    (lb["member_bce"]+lb["member_dice"]).backward()
    assert refined.refine.embed.weight.grad is not None and refined.refine.embed.weight.grad.abs().sum()>0


def test_cli_member_refine_defaults():
    from openstroke.obb.cli import vector_defaults
    extra={"member_refine":True,"gt_only":False}
    vector_defaults("d",extra,{},None,False,None,3)
    assert extra["new_lr_mult"]==10.0 and extra["stroke_graph"] is False and "instance_queries" not in extra


def test_v19_panoptic_keeps_e69_outputs_and_trains_every_stroke():
    torch.manual_seed(19)
    plain=OpenOBB25("s",**OPTS).eval()
    pan=OpenOBB25("s",**OPTS,stroke_panoptic=True).eval()
    res=pan.load_state_dict(plain.state_dict(),strict=False)
    assert all(k.startswith("panoptic.") for k in res.missing_keys) and not res.unexpected_keys
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"));x=torch.rand(2,3,IMG,IMG)
    with torch.no_grad():a=plain(x,None,ctx);b=pan(x,None,ctx)
    torch.testing.assert_close(a["member_logits"],b["member_logits"],rtol=0,atol=0)
    assert b["stroke_sem"].shape[:2]==ctx["vec"].shape[:2] and b["stroke_centre"].shape[-1]==2
    pan.train();tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    out=pan(x,tg,ctx)
    assert torch.isfinite(out["panoptic_cls"]) and torch.isfinite(out["panoptic_off"])
    (out["panoptic_cls"]+out["panoptic_off"]).backward()
    assert pan.panoptic.cls.weight.grad.abs().sum()>0 and pan.panoptic.off.weight.grad.abs().sum()>0


def test_v19_panoptic_decode_groups_same_class_strokes_by_centre():
    import numpy as np
    from openstroke.obb.models.stroke_panoptic import PanopticPage,panoptic_objects
    page=PanopticPage(2)
    prob=np.array([[0.9,0.05,0.05],[0.8,0.1,0.1],[0.1,0.85,0.05],[0.05,0.05,0.9]])
    centre=np.array([[10.,10.],[14.,12.],[11.,10.],[500.,500.]])
    page.add(np.array([1,2,3,4]),prob,centre)
    page.add(np.array([1]),prob[:1],centre[:1])             # the same stroke seen by a second tile
    objs=panoptic_objects(page,["a","b"],radius=24.)
    got=sorted((o["class"],tuple(o["members"])) for o in objs)
    assert got==[("a",(1,2)),("b",(3,))] and all(o["source"]=="panoptic" for o in objs)


def test_v19_real_tile_prediction_pools_stroke_classes_by_primitive_id():
    import numpy as np
    from openstroke.obb.predict import predict_image
    from openstroke.obb.models.stroke_panoptic import PanopticPage
    torch.manual_seed(5)
    m=OpenOBB25("s",**OPTS,stroke_panoptic=True).eval()
    tok=np.zeros((6,24),np.float32)
    for i in range(6):
        tok[i,1:17]=np.linspace([10+20*i,20],[30+20*i,40],8).ravel()
    tok[:,17:20]=0.;tok[:,20]=1.;tok[:,21]=-1;tok[:,22]=-2;tok[:,23]=np.arange(100,106)
    page=PanopticPage(NC)
    dets=predict_image(m,np.full((IMG,IMG,3),255,np.uint8),IMG,32,2,torch.device("cpu"),50,vec=dict(tokens=tok),panoptic=page)
    assert set(page.prob)=={100,101,102,103,104,105} and isinstance(dets,list)
    c=page.centre[100]/page.count[100]
    assert 0<=c[0]<=IMG and 0<=c[1]<=IMG


def test_v19_loss_balances_rare_object_strokes_against_background():
    from openstroke.obb.models.stroke_panoptic import panoptic_loss
    vec=torch.zeros(1,100,24);vec[0,:,1:17]=torch.rand(100,16);vec[0,:,22]=-1;vec[0,0,22]=0
    mask=torch.ones(1,100,dtype=torch.bool)
    logits=torch.zeros(1,100,NC+1);logits[0,:,-1]=5.               # confident background everywhere
    out=panoptic_loss(logits,torch.zeros(1,100,2),vec,mask,[{"labels":torch.tensor([1])}])
    fg=torch.nn.functional.cross_entropy(logits[0,:1],torch.tensor([1]))
    bg=torch.nn.functional.cross_entropy(logits[0,1:],torch.full((99,),NC))
    torch.testing.assert_close(out["panoptic_cls"],fg+bg)          # the single object stroke weighs as much as 99 background ones


def test_v19_validator_keeps_the_best_foreground_threshold(monkeypatch,tmp_path):
    from openstroke.obb import object_validation,vector_validation,cli
    images,_,stem=make_page(tmp_path)
    gt=vector_validation.DatasetVectorGT(images,["Duct","Elbow"],comparison_pages=("doc000022",))
    monkeypatch.setattr(vector_validation,"DatasetVectorGT",lambda *a,**k:gt)
    good=predictions(stem)[stem]
    rows=[dict(r,source="panoptic@0.85") for r in good]+[dict(class_id=1,score=0.9,members=[10,11,30],source="panoptic@0.5")]
    rows+=[dict(r,source="members") for r in good[:1]]
    monkeypatch.setattr(cli,"predict",lambda *a,**k:{stem:rows})
    ev=object_validation.VectorEpochEvaluator(None,images,["Duct","Elbow"],dict(batch=1,stroke_panoptic=True))
    try:
        r=ev.evaluate_checkpoint("unused")
        assert r["how"]=="panoptic@0.85" and r["F2"]==1. and set(r["paths"])>={"panoptic@0.5","members"}
    finally:ev.close()


def test_train_page_f2_diagnostic_scores_distinct_train_documents(monkeypatch,tmp_path):
    import shutil
    from openstroke.obb import object_validation,cli
    images,_,stem=make_page(tmp_path)
    for kind,suffix in (("images",".png"),("vectors",".npz"),("labels",".txt")):
        (tmp_path/kind/"train").mkdir(parents=True)
        shutil.copy(tmp_path/kind/"val"/(stem+suffix),tmp_path/kind/"train"/(stem+suffix))
    calls=[]
    def fake(model,source,**k):
        calls.append(Path(source).name)
        return {stem:[dict(r,source="members") for r in predictions(stem)[stem]]}
    monkeypatch.setattr(cli,"predict",fake)
    from openstroke.obb import vector_validation
    real=vector_validation.DatasetVectorGT
    monkeypatch.setattr(vector_validation,"DatasetVectorGT",lambda imgs,names,comparison_pages=("doc000022",),**k:real(imgs,names,comparison_pages=("doc000022",),**k))
    ev=object_validation.VectorEpochEvaluator(None,images,["Duct","Elbow"],dict(batch=1,train_f2_pages=3))
    try:
        r=ev.evaluate_checkpoint("unused")
        assert calls==["val","train"] and r["train"]["F2"]==1.
    finally:ev.close()


def test_v20_deep_stroke_context_starts_exactly_as_the_checkpoint_and_learns():
    torch.manual_seed(20)
    plain=OpenOBB25("s",**OPTS).eval()
    deep=OpenOBB25("s",**OPTS,stroke_context=2).eval()
    deep.vec.context.window=4                                   # several windows on the small test tiles
    res=deep.load_state_dict(plain.state_dict(),strict=False)
    assert all(k.startswith("vec.context.") for k in res.missing_keys) and res.missing_keys and not res.unexpected_keys
    targets=_batch();ctx=ctx_batch(targets,torch.device("cpu"));x=torch.rand(2,3,IMG,IMG)
    with torch.no_grad():a=plain(x,None,ctx);b=deep(x,None,ctx)
    for k in ("pred_logits","pred_boxes","member_logits"):
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    deep.train();tg=[{k:v for k,v in t.items() if k!="vec"} for t in targets]
    out=deep(x,tg,ctx)
    (out["member_bce"]+out["member_dice"]).backward()
    g=deep.vec.context.out.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0


def test_v20_window_order_round_trips_every_stroke():
    from openstroke.obb.models.stroke_context import DeepStrokeContext
    torch.manual_seed(1)
    m=DeepStrokeContext(8,4,width=16,blocks=4,window=4,heads=2)
    torch.nn.init.normal_(m.out.weight)
    tok=torch.randn(1,11,8);vec=torch.rand(1,11,24);mask=torch.ones(1,11,dtype=torch.bool);mask[0,9:]=False
    y=m(tok,vec,mask,torch.randn(1,4,8,8))
    assert torch.isfinite(y).all() and torch.equal(y[0,9:],tok[0,9:])
    perm=torch.randperm(9)                                      # same strokes in another order: same per-stroke output
    v2=vec.clone();t2=tok.clone();v2[0,:9]=vec[0,perm];t2[0,:9]=tok[0,perm]
    y2=m(t2,v2,mask,torch.randn(1,4,8,8)*0+m.img.weight.new_zeros(1,4,8,8))
    y1=m(tok,vec,mask,m.img.weight.new_zeros(1,4,8,8))
    torch.testing.assert_close(y2[0,:9],y1[0,perm],rtol=1e-5,atol=1e-5)


def test_v21_gray_augment_keeps_image_and_stroke_colours_consistent():
    import numpy as np
    from openstroke.obb.data.dota import to_gray
    img=np.zeros((8,8,3),np.uint8);img[...,2]=200                 # red in BGR
    vec={"pts":np.zeros((2,16),np.float32),"attr":np.array([[0,1,0,0,1,0,0,5],[0,0,0,1,1,0,-1,6]],np.float32)}
    g,v=to_gray(img,vec)
    assert (g[...,0]==g[...,1]).all() and (g[...,1]==g[...,2]).all()
    np.testing.assert_allclose(v["attr"][0,1:4],[0.299]*3,rtol=1e-6)
    np.testing.assert_allclose(v["attr"][1,1:4],[0.114]*3,rtol=1e-6)
    assert v["attr"][0,6]==0 and v["attr"][1,7]==6                 # inst / pid untouched


def test_v22_copy_paste_moves_image_strokes_and_label_together():
    import random
    import numpy as np
    from openstroke.obb.data.dota import DotaPatches, OID_STEP
    random.seed(0);np.random.seed(0)
    ds=DotaPatches.__new__(DotaPatches)
    ds.size=128;ds.items=[{"name":"src","objs":[[3,0,0]]}];ds._cp_accept=None
    src=np.full((128,128,3),255,np.uint8);src[20:40,30]=0          # a vertical line drawn at x=30, y 20..40
    pts=np.array([np.linspace([30,20],[30,40],8).ravel()],np.float32)
    attr=np.array([[0,0,0,0,1,0,0,77]],np.float32)                 # type, rgb, width, layer, inst 0, pid 77
    poly=np.array([[28,18,32,18,32,42,28,42]],np.float32)
    ds._load=lambda m:(src,poly.copy(),np.array([[3,0]]))
    ds._load_vec=lambda m:{"pts":pts.copy(),"attr":attr.copy()}
    tgt=np.full((128,128,3),255,np.uint8)
    vec={"pts":np.zeros((0,16),np.float32),"attr":np.zeros((0,8),np.float32)}
    img,polys,labels,v=ds._paste_objects(tgt,np.zeros((0,8),np.float32),np.zeros((0,2),np.int64),vec,1)
    assert labels.tolist()==[[3,8*OID_STEP]] and v["attr"][0,6]==8*OID_STEP and v["attr"][0,7]==-1
    shift=v["pts"][0,:2]-pts[0,:2]
    np.testing.assert_allclose(polys[0].reshape(4,2)-poly[0].reshape(4,2),np.tile(shift,(4,1)))
    x,y=(v["pts"][0,0::2].mean()).round().astype(int),(v["pts"][0,1::2].mean()).round().astype(int)
    assert img[y,x].max()==0 and (img<255).sum()<200 and (tgt==255).all()   # dark pixels at the moved stroke, source tile untouched


def test_cli_passes_openstroke_augment_flags_to_the_trainer():
    from openstroke.obb.cli import IGNORED, check_keys
    for k in ("copy_paste","gray_p"):
        assert k not in IGNORED
    check_keys({"copy_paste":4,"gray_p":0.5})


def test_hvac6_build_merges_square_round_variants_and_drops_tee(tmp_path):
    import shutil,yaml
    import numpy as np
    from openstroke.obb import gt_dataset
    names=["Duct","Elbow","Elbow vuông","Concentric Reducer","Eccentric Reducer","Reducer vuông tròn","Chân rẽ","Chân rẽ vuông tròn","Tee"]
    src=tmp_path/"gt";spec=[{"stem":"docA_p0001","split":"train","label_sha256":"x"}]
    for k in ("images","labels","vectors"):(src/k/"train").mkdir(parents=True)
    (src/"images/train/docA_p0001.png").write_bytes(b"")
    (src/"labels/train/docA_p0001.txt").write_text("8 0 0 1 0 1 1 0 1\n7 0 0 1 0 1 1 0 1\n0 0 0 1 0 1 1 0 1\n")
    tok=np.zeros((4,24),np.float32);tok[:,22]=[0,1,2,-1];tok[:,23]=[10,11,12,13]
    np.savez(src/"vectors/train/docA_p0001.npz",tokens=tok)
    (src/"data.yaml").write_text(yaml.safe_dump({"path":".","train":"images/train","val":"images/val","names":dict(enumerate(names))},allow_unicode=True),encoding="utf-8")
    old=gt_dataset.GT_SPEC;gt_dataset.GT_SPEC=spec
    try:gt_dataset.build6(src,tmp_path/"gt6")
    finally:gt_dataset.GT_SPEC=old
    lines=(tmp_path/"gt6/labels/train/docA_p0001.txt").read_text().split()
    assert [lines[0],lines[9]]==["6","0"]                              # Chân rẽ vuông tròn -> Chân rẽ; Duct kept; Tee gone
    with np.load(tmp_path/"gt6/vectors/train/docA_p0001.npz") as z:
        assert z["tokens"][:,22].tolist()==[-1,0,1,-1] and z["tokens"][:,23].tolist()==[10,11,12,13]


def test_hvac6_native_scoring_merges_gt_and_predictions(tmp_path):
    from openstroke.obb.vector_validation import DatasetVectorGT
    images,_,stem=make_page(tmp_path)
    (tmp_path/"labels/val"/(stem+".txt")).write_text("7 0 0 1 0 1 1 0 1\n8 0 0 1 0 1 1 0 1\n")   # Chân rẽ vuông tròn, Tee
    names=["Duct","Elbow","Elbow vuông","Concentric Reducer","Eccentric Reducer","Reducer vuông tròn","Chân rẽ","Chân rẽ vuông tròn","Tee"]
    gt=DatasetVectorGT(images,names,comparison_pages=("doc000022",),merge=True)
    pred={stem:[dict(class_id=6,score=0.9,members=[10,11],source="members")]}           # predicted as plain Chân rẽ
    r=gt.evaluate(pred,"members",names)
    assert r["F2"]==1.0                                                                    # one GT object left (Tee dropped), matched


def test_hvac6_square_round_reducer_accepts_either_base(tmp_path):
    from openstroke.obb.vector_validation import DatasetVectorGT
    images,_,stem=make_page(tmp_path)
    (tmp_path/"labels/val"/(stem+".txt")).write_text("5 0 0 1 0 1 1 0 1\n0 0 0 1 0 1 1 0 1\n")   # Reducer vuông tròn, Duct
    names=["Duct","Elbow","Elbow vuông","Concentric Reducer","Eccentric Reducer","Reducer vuông tròn","Chân rẽ","Chân rẽ vuông tròn","Tee"]
    gt=DatasetVectorGT(images,names,comparison_pages=("doc000022",),merge=True)
    for base in (3,4):                                                      # concentric or eccentric: both correct
        pred={stem:[dict(class_id=base,score=0.9,members=[10,11],source="members"),dict(class_id=0,score=0.9,members=[30],source="members")]}
        assert gt.evaluate(pred,"members",names)["F2"]==1.0
    pred={stem:[dict(class_id=1,score=0.9,members=[10,11],source="members"),dict(class_id=0,score=0.9,members=[30],source="members")]}
    assert gt.evaluate(pred,"members",names)["F2"]<1.0                     # an elbow is still wrong
