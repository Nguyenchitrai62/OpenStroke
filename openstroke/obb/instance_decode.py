"""One shape -> one predicted instance; tile stitching uses shared primitive length, not box NMS."""
import cv2
import numpy as np
import torch


def tile_instances(out,b,size,scale,x0,y0,threshold=0.5,conf=0.001):
    row=out["instances"]
    probability=row["classes"][b].float().softmax(-1)
    score,label=probability[:,:-1].max(-1)
    active=(score>=conf)&(score>probability[:,-1])
    mask=row["masks"][b].float().sigmoid()
    eligible=(mask>=threshold)&out["vec_mask"][b][None]&active[:,None]
    quality=(mask*eligible).sum(-1)/eligible.sum(-1).clamp_min(1)
    score=score*quality
    eligible=eligible&(score>=conf)[:,None]
    weighted=(mask*score[:,None]).masked_fill(~eligible,-1)
    winner=weighted.argmax(0) if mask.shape[1] else torch.zeros(0,dtype=torch.long,device=mask.device)
    live=eligible.any(0)
    vec=out["vec"][b].detach().float().cpu().numpy()
    pid=np.rint(vec[:,23]).astype(np.int64)
    pts=vec[:,1:17].reshape(-1,8,2)*size
    winner,live=winner.cpu().numpy(),live.cpu().numpy()
    score,label=score.detach().cpu().numpy(),label.cpu().numpy()
    result=[]
    for q in np.unique(winner[live]):
        selected=np.flatnonzero(live&(winner==q))
        selected=selected[pid[selected]>=0]
        if not len(selected):continue
        poly=cv2.boxPoints(cv2.minAreaRect(pts[selected].reshape(-1,2).astype(np.float32))).reshape(-1)
        result.append((f"img__{scale}__{x0}___{y0}",int(label[q]),float(score[q]),poly,sorted(set(pid[selected].tolist()))))
    return result


def merge_instances(detections,names,page):
    from openstroke.obb.data.vectors import page_columns
    tokens,_,_,pid=page_columns(page)
    points=tokens[:,1:17].reshape(-1,8,2)
    lengths=np.linalg.norm(np.diff(points.astype(np.float64),axis=1),axis=2).sum(1)
    at={int(p):i for i,p in enumerate(pid)}
    weight={int(p):float(lengths[i]) for i,p in enumerate(pid)}
    mass=lambda members:sum(weight.get(p,0.) for p in members)
    objects=[];by_pid={}
    for _,cls,score,_,members in sorted(detections,key=lambda r:-r[2]):
        members=set(int(p) for p in members)
        if not members:continue
        if members-at.keys():raise ValueError("Predicted instance contains unknown primitive IDs")
        candidates=sorted({i for p in members for i in by_pid.get(p,())})
        for i in candidates:
            obj=objects[i]
            if obj[0]==cls and mass(obj[2]&members)/max(min(mass(obj[2]),mass(members)),1e-9)>=0.7:
                obj[2].update(members)
                break
        else:
            i=len(objects);objects.append([int(cls),float(score),members])
        for p in members:by_pid.setdefault(p,set()).add(i)
    claimed=set();result=[]
    for cls,score,members in objects:
        members=members-claimed
        if not members:continue
        claimed.update(members)
        poly=cv2.boxPoints(cv2.minAreaRect(points[[at[p] for p in sorted(members)]].reshape(-1,2).astype(np.float32)))
        result.append(dict(class_id=cls,**{"class":names[cls]},score=float(score),poly=poly.reshape(-1).astype(float).tolist(),
                           members=sorted(members),source="instances"))
    return result
