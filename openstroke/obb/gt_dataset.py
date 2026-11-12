"""Materialize the verified GT-only subset of the existing HVAC7 upload, without new data.

    python -m openstroke.obb.gt_dataset --source /content/.../obb_hvac7 --out /content/openstroke15_gt

The source ZIP has no provenance manifest. The frozen stem/label digest inventory below
comes from its original pages.json (18 GT train, 8 GT val); pipeline pseudo pages are excluded.
Images/vectors/labels are byte-identical hardlinks/copies, never relabelled.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml
import numpy as np

GT_SPEC_ALL = [{'stem': 'doc000022_p0001_gt11', 'split': 'val', 'label_sha256': '3cca3129a758560e5e565346dc4b0665a9c46d18a8643f1694bfe4278283a111'}, {'stem': 'doc000030_p0001_gt15', 'split': 'val', 'label_sha256': 'e6e2ec2f2f53348bd8f7c6b587530a8e65b236296510cf0e92b2a1cc608598e8'}, {'stem': 'doc000180_p0001_gt114', 'split': 'val', 'label_sha256': 'dbb8c5cb5fdd11cbade33f3f59e1bee1eeabf68317ca3bc32e1fb7a0b77f05bd'}, {'stem': 'doc000181_p0001_gt115', 'split': 'train', 'label_sha256': 'c90487bef6d9585a8e07b2e5b61cb9ff506de5d423708508a536573322d3c393'}, {'stem': 'doc000187_p0001_gt117', 'split': 'val', 'label_sha256': '012f7cfb949998ca87e49cceb7d5e80ce25e416756e010f86cef8a5c695ba16b'}, {'stem': 'doc000189_p0001_gt118', 'split': 'val', 'label_sha256': '9c548b66766ee5eab29bc6d9a25374681349a349dc1ceb8d68cfbdb18e9f9d24'}, {'stem': 'doc000190_p0001_gt119', 'split': 'val', 'label_sha256': '5096d94c7646cf66132e19b3213703f575623a4f6122c673f164409c00e40925'}, {'stem': 'doc000191_p0001_gt120', 'split': 'train', 'label_sha256': 'bdcea19f7c51d9339edf2dd41d83886b42f9305f251d99f5d2d9c706487e42cc'}, {'stem': 'doc000194_p0001_gt121', 'split': 'train', 'label_sha256': 'a7255c1b31cde7ccf92a2ed4d006f1e41863acfaa4c117ce4ff0a91fb29c247d'}, {'stem': 'doc000195_p0001_gt122', 'split': 'train', 'label_sha256': 'f69004a34ecf693e4e08d1907f8305b665895809cd392cbd96ddaf55228b5a2b'}, {'stem': 'doc000196_p0001_gt123', 'split': 'train', 'label_sha256': 'd281079b81aa87a44c3715e4ae2548604be5a3925e9cae1bacc8cdd65b5d309a'}, {'stem': 'doc000198_p0001_gt124', 'split': 'train', 'label_sha256': '253373982f50fe2157fa428fa7e95b3724f902699376b0e02be683d6e1b7b124'}, {'stem': 'doc000199_p0001_gt125', 'split': 'train', 'label_sha256': '7f0941214365c790b2cd9e294bb70e4d2285fd3c3e0eeaee8b7707d62b8a0a26'}, {'stem': 'doc000200_p0001_gt126', 'split': 'val', 'label_sha256': '169ff089ae320502a1daf33779446fc50de500a878ea261af98879d356d8f300'}, {'stem': 'doc000201_p0001_gt127', 'split': 'val', 'label_sha256': '093320e0c5770d48a26208bd993ade902348bf831be5eabbe6f2959b2b8aec6f'}, {'stem': 'doc000211_p0001_gt129', 'split': 'train', 'label_sha256': 'e3d9910977ffbc2d9c4b92390d0482103218d486d0078b250e01c6eeef08a38f'}, {'stem': 'doc000214_p0001_gt130', 'split': 'train', 'label_sha256': '0da2133ca1dec4c574463d1a613c5081af1477f3f7155798b130f1a47bc3832c'}, {'stem': 'doc000214_p0002_gt130', 'split': 'train', 'label_sha256': 'db0590a5a4e6544c44cea89d3eaad2a7ef49cc485156a9c2286db9fa8333967e'}, {'stem': 'doc000214_p0003_gt130', 'split': 'train', 'label_sha256': 'b8cbefb4fc7eae0cd5e53c6ce025ae1d87c752404aa4b88652bfdec084d94dd2'}, {'stem': 'doc000214_p0004_gt130', 'split': 'train', 'label_sha256': '34428409eff0d257a6e6f46fbc4fc0e7ee5ae4a839e8c53a23cc1d0f4d50976c'}, {'stem': 'doc000214_p0005_gt130', 'split': 'train', 'label_sha256': 'bdcc014d740ba88e29654eed8b4bc87761ad1a7aeb8e34e272854f2c19d42abc'}, {'stem': 'doc000214_p0006_gt130', 'split': 'train', 'label_sha256': '15ab96aa19cbec8cd25e809e3c86f331bc27d096f2ae55257ea73e6847f56a72'}, {'stem': 'doc000214_p0007_gt130', 'split': 'train', 'label_sha256': '38c5a449f5bb421042df3483bae595a27b58049177fd27881497efb99adbaeef'}, {'stem': 'doc000214_p0008_gt130', 'split': 'train', 'label_sha256': 'eb4f7733ff7283aa54a95976911e8841d6afe6175257c5b2c7ff8f5504507c1d'}, {'stem': 'doc000214_p0009_gt130', 'split': 'train', 'label_sha256': '39b9a00693db63d6c5967e1b2c441b1a8b38170c61d01d94400bcbef3471d338'}, {'stem': 'doc000214_p0010_gt130', 'split': 'train', 'label_sha256': 'd31359ef6cdabede72ac246683b2b53be37a7334d8509751efe12ec8d44cafed'}]
# F129: only documents mirrored from the remote HVAC deploy (official GT); the two local-only val documents are left out
GT_SPEC = [r for r in GT_SPEC_ALL if r["stem"][:9] not in ("doc000022", "doc000030")]


def label_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(data):
    data=Path(data);folder=data.parent if data.suffix in (".yaml",".yml") else data
    path=folder/"gt_provenance.json"
    if not path.is_file():
        raise ValueError("OpenStroke15 needs the GT-only dataset. Run python -m openstroke.obb.gt_dataset --source <obb_hvac7> --out <gt folder> first.")
    proof=json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema")=="openstroke.hvac7.gt_only.hvac6.v1":   # derived from a validated GT folder (labels rewritten)
        if proof.get("records")!=GT_SPEC:
            raise ValueError("GT provenance differs from the frozen HVAC7 GT inventory")
        return folder/"data.yaml"
    if proof.get("schema")!="openstroke.hvac7.gt_only.v1" or proof.get("records")!=GT_SPEC:
        raise ValueError("GT provenance differs from the frozen HVAC7 GT inventory")
    cfg=yaml.safe_load((folder/"data.yaml").read_text(encoding="utf-8"))
    if cfg.get("path")!="." or cfg.get("train")!="images/train" or cfg.get("val")!="images/val":
        raise ValueError("GT dataset YAML paths differ from the verified layout")
    for split in ("train","val"):
        expected={r["stem"] for r in GT_SPEC if r["split"]==split}
        for kind,suffix in [("images",".png"),("labels",".txt"),("vectors",".npz")]:
            found={p.stem for p in (folder/kind/split).iterdir() if p.is_file()}
            if found!=expected:raise ValueError(f"Unexpected or missing {kind}/{split} files; do not mix pseudo samples")
    for row in GT_SPEC:
        if label_hash(folder/"labels"/row["split"]/(row["stem"]+".txt"))!=row["label_sha256"]:
            raise ValueError(f"GT labels changed: {row['stem']}")
        with np.load(folder/"vectors"/row["split"]/(row["stem"]+".npz")) as z:
            tokens=np.asarray(z["tokens"])
            labels=(folder/"labels"/row["split"]/(row["stem"]+".txt")).read_text(encoding="utf-8").splitlines()
            if tokens.ndim!=2 or tokens.shape[1]<24 or not np.isfinite(tokens[:,:24]).all():
                raise ValueError(f"GT vector tokens need finite inst/pid columns: {row['stem']}")
            inst,pid=tokens[:,22],tokens[:,23]
            if (np.any(inst!=np.rint(inst)) or np.any(inst<-1) or np.any(inst>=len(labels))
                or np.any(pid!=np.rint(pid)) or np.any(pid<0) or len(np.unique(pid))!=len(pid)):
                raise ValueError(f"Invalid GT instance/primitive IDs: {row['stem']}")
    return folder/"data.yaml"


def build6(gt_folder, out):
    """GT folder (9 classes) -> the same pages with the 6 product classes: square-round variants take their base class, Tee
    objects are removed (their strokes become background, inst -1, the other objects are re-indexed). data.yaml keeps the 9
    names so a 9-class checkpoint loads unchanged; merged classes simply have no objects."""
    from openstroke.objects import HVAC6_MERGE
    gt_folder,out=Path(gt_folder),Path(out)
    if (out/"gt_provenance.json").is_file():
        return validate(out)
    cfg=yaml.safe_load((gt_folder/"data.yaml").read_text(encoding="utf-8"))
    names=cfg["names"];names=[names[k] for k in sorted(names,key=int)] if isinstance(names,dict) else list(names)
    canon=[n.replace("_"," ") for n in names]
    def target(n):
        m=HVAC6_MERGE.get(n,n)
        return None if m is None else canon.index(m)
    to=[target(n) for n in canon]
    stats={"renamed":0,"dropped":0}
    for row in GT_SPEC:
        sp=row["split"]
        for kind in ("images","labels","vectors"):
            (out/kind/sp).mkdir(parents=True,exist_ok=True)
        img=gt_folder/"images"/sp/(row["stem"]+".png");dst=out/"images"/sp/img.name
        if not dst.exists():
            try:os.link(img,dst)
            except OSError:shutil.copy2(img,dst)
        lines=(gt_folder/"labels"/sp/(row["stem"]+".txt")).read_text(encoding="utf-8").splitlines()
        new,remap=[],{}
        for i,line in enumerate(l for l in lines if l.strip()):
            parts=line.split();c=int(parts[0]);m=to[c]
            if m is None:
                stats["dropped"]+=1;continue
            stats["renamed"]+=int(m!=c);remap[i]=len(new);new.append(" ".join([str(m)]+parts[1:]))
        (out/"labels"/sp/(row["stem"]+".txt")).write_text("\n".join(new)+("\n" if new else ""),encoding="utf-8")
        with np.load(gt_folder/"vectors"/sp/(row["stem"]+".npz")) as z:
            d={k:np.array(z[k]) for k in z.files}
        inst=np.rint(d["tokens"][:,22]).astype(np.int64)
        d["tokens"][:,22]=np.array([remap.get(int(v),-1) if v>=0 else v for v in inst],d["tokens"].dtype)
        if "inst" in d:
            iv=d["inst"].astype(np.int64);d["inst"]=np.array([remap.get(int(v),-1) if v>=0 else v for v in iv],d["inst"].dtype)
        np.savez_compressed(out/"vectors"/sp/(row["stem"]+".npz"),**d)
    (out/"data.yaml").write_text(yaml.safe_dump(dict(cfg,path=".",train="images/train",val="images/val"),allow_unicode=True,sort_keys=False),encoding="utf-8")
    (out/"gt_provenance.json").write_text(json.dumps(dict(schema="openstroke.hvac7.gt_only.hvac6.v1",records=GT_SPEC,merge=HVAC6_MERGE,
                                                        derived_from=str(gt_folder),stats=stats),ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"[openstroke] 6 product classes: {stats['renamed']} objects renamed, {stats['dropped']} Tee objects removed -> {out}")
    return validate(out)


def ensure(data, class_set="hvac9"):
    """data= of a vector run -> the GT-only dataset: itself if already GT-only, the GT subset (built once next to it, in
    <folder>_gt) if it is the HVAC7 upload with pipeline pages, otherwise unchanged (another dataset)."""
    data=Path(data);folder=data.parent if data.suffix in (".yaml",".yml") else data
    if (folder/"gt_provenance.json").is_file():
        proof=json.loads((folder/"gt_provenance.json").read_text(encoding="utf-8"))
        if class_set=="hvac6" and proof.get("schema")!="openstroke.hvac7.gt_only.hvac6.v1":
            return str(build6(folder,folder.parent/(folder.name+"6")))
        return str(validate(folder))
    if not all((folder/"labels"/r["split"]/(r["stem"]+".txt")).is_file() for r in GT_SPEC):
        return str(data)
    out=folder.parent/(folder.name+"_gtr")          # F129: remote-only GT set (new folder: older _gt builds hold 26 pages)
    built=build(folder,out)
    if class_set=="hvac6":
        built=build6(out,folder.parent/(folder.name+"_gtr6"))
    print(f"[openstroke] GT-only data: {built} ({sum(r['split']=='train' for r in GT_SPEC)} train / "
          f"{sum(r['split']=='val' for r in GT_SPEC)} val pages; pipeline pages left out)")
    return str(built)


def build(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents:
        raise ValueError("Choose an output directory separate from the source dataset")
    if out.exists() and any(out.iterdir()):
        return validate(out)
    for row in GT_SPEC:
        for kind,suffix in [("images",".png"),("labels",".txt"),("vectors",".npz")]:
            f=source/kind/row["split"]/(row["stem"]+suffix)
            if not f.is_file():raise FileNotFoundError(f"Existing upload lacks GT source file: {f}")
        if label_hash(source/"labels"/row["split"]/(row["stem"]+".txt"))!=row["label_sha256"]:
            raise ValueError(f"GT source label digest differs: {row['stem']}")
    cfg=yaml.safe_load((source/"data.yaml").read_text(encoding="utf-8"))
    for row in GT_SPEC:
        for kind,suffix in [("images",".png"),("labels",".txt"),("vectors",".npz")]:
            src=source/kind/row["split"]/(row["stem"]+suffix);dst=out/kind/row["split"]/(row["stem"]+suffix)
            dst.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
    cfg.update(path=".",train="images/train",val="images/val")
    (out/"data.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    (out/"gt_provenance.json").write_text(json.dumps(dict(schema="openstroke.hvac7.gt_only.v1",records=GT_SPEC,excluded_pipeline_pages=25),indent=1),encoding="utf-8")
    return validate(out)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--source",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args(argv);path=build(a.source,a.out)
    print(f"GT-only dataset: {path} | train {sum(r['split']=='train' for r in GT_SPEC)} pages, val {sum(r['split']=='val' for r in GT_SPEC)} pages | pseudo and local-only pages excluded")
    return 0


if __name__=="__main__":raise SystemExit(main())
