"""OpenStroke15: layer-free line context and primitive-instance queries.

Original implementation of mechanisms motivated by VecFormer (2505.23395),
CADSpotting (2412.07377), SymPoint (2401.10556) and PTv3 (2312.10035).
This is not a reproduction: eight existing arc samples, 2-D serialized windows,
no LFE/layers, no custom CUDA, and object membership supervision in stroke space.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def serialized_order(center, transpose=False):
    """2-D Morton order; every primitive is retained (not a geometric edge classifier)."""
    xy=(center.detach().clamp(0,1)*1023).long()
    x,y=xy.unbind(-1)
    if transpose:x,y=y,x
    code=torch.zeros_like(x)
    for bit in range(10):
        code=code|(((x>>bit)&1)<<(2*bit))|(((y>>bit)&1)<<(2*bit+1))
    return torch.argsort(code,stable=True)


class LineContext(nn.Module):
    def __init__(self,d,window=128):
        super().__init__();self.window=window
        self.segment=nn.Sequential(nn.Linear(7,d),nn.GELU(),nn.Linear(d,d))
        self.pool=nn.Linear(d,d)
        nn.init.zeros_(self.pool.weight);nn.init.zeros_(self.pool.bias)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,4,2*d,dropout=0.,activation="gelu",batch_first=True,norm_first=True) for _ in range(2)])
        for block in self.blocks:
            nn.init.zeros_(block.self_attn.out_proj.weight);nn.init.zeros_(block.self_attn.out_proj.bias)
            nn.init.zeros_(block.linear2.weight);nn.init.zeros_(block.linear2.bias)

    def forward(self,tok,vec,mask,lengths=None):
        output=tok.new_zeros(tok.shape)
        for b in range(len(tok)):
            rows=torch.arange(lengths[b],device=tok.device) if lengths is not None else mask[b].nonzero(as_tuple=True)[0]
            if len(rows)==0:continue
            points=vec[b,rows,1:17].float().reshape(-1,8,2)
            delta=points[:,1:]-points[:,:-1]
            length=delta.norm(dim=-1,keepdim=True)
            mid=(points[:,1:]+points[:,:-1])*0.5
            center=mid.mean(1)
            geometry=torch.cat([length,delta/length.clamp_min(1e-8),mid,center[:,None].expand(-1,7,-1)],-1)
            line=self.segment(geometry)
            h=tok[b,rows]+self.pool(line.amax(1)+line.mean(1))
            for transpose,block in enumerate(self.blocks):
                order=serialized_order(center,bool(transpose))
                n=len(order);pad=(-n)%self.window
                packed=F.pad(h[order],(0,0,0,pad)).reshape(-1,self.window,h.shape[-1])
                valid=torch.arange(n+pad,device=h.device).reshape(-1,self.window)<n
                packed=block(packed,src_key_padding_mask=~valid).reshape(-1,h.shape[-1])[:n]
                inverse=torch.argsort(order)
                h=packed[inverse]
            output[b,rows]=h
        return output


class StrokeInstances(nn.Module):
    def __init__(self,vec_dim,image_channels,neck_channels,nc,queries=256,free=128,line_context=True):
        super().__init__()
        if queries<1 or not 0<=free<=queries:raise ValueError("instance_free must be between 0 and instance_queries")
        if vec_dim%4:raise ValueError("instance decoder dimension must be divisible by four")
        self.nc,self.queries,self.free=nc,int(queries),int(free)
        self.lines=LineContext(vec_dim) if line_context else None
        self.image=nn.Conv2d(image_channels,vec_dim,1)
        nn.init.zeros_(self.image.weight);nn.init.zeros_(self.image.bias)
        self.seed=nn.Linear(neck_channels,vec_dim)
        self.position=nn.Sequential(nn.Linear(2,vec_dim),nn.GELU(),nn.Linear(vec_dim,vec_dim))
        self.content=nn.Parameter(torch.zeros(1,queries,vec_dim))
        self.drawing=nn.Parameter(torch.zeros(1,1,vec_dim))
        self.layers=nn.ModuleList([nn.TransformerDecoderLayer(vec_dim,4,2*vec_dim,dropout=0.,activation="gelu",batch_first=True,norm_first=True) for _ in range(3)])
        self.norm=nn.LayerNorm(vec_dim)
        self.classify=nn.Linear(vec_dim,nc+1)
        self.mask_embed=nn.Linear(vec_dim,vec_dim)
        self.center=nn.Linear(vec_dim,2)
        self.radius=nn.Linear(vec_dim,1)
        self.spatial=nn.Parameter(torch.tensor(1.))
        for layer in (self.center,self.radius,self.mask_embed):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.classify.bias)
        with torch.no_grad():self.classify.bias[-1]=2.

    def forward(self,side,f4,feats,dense,anchors,size):
        tok,vec,mask=side["tok"],side["vec"],side["mask"]
        lengths=side.get("vec_lengths")
        memory=self.lines(tok,vec,mask,lengths) if self.lines is not None else tok
        batch,m,d=memory.shape
        points=vec[...,1:17].float().reshape(batch,m,8,2)
        if m:
            image=self.image(f4)
            extent=points.new_tensor([f4.shape[-1]*4,f4.shape[-2]*4])
            grid=points*float(size)/extent*2-1
            sample=F.grid_sample(image,grid.reshape(batch,m*8,1,2),align_corners=False).squeeze(-1).transpose(1,2).reshape(batch,m,8,d)
            memory=memory+sample.mean(2)+sample.amax(2)
        center=points.mean(2)
        context=torch.cat([memory,self.drawing.expand(batch,-1,-1)],1)
        valid=torch.cat([mask,torch.ones(batch,1,dtype=torch.bool,device=mask.device)],1)
        nseed=self.queries-self.free
        flat=torch.cat([x.flatten(2).transpose(1,2) for x in feats],1)
        available=min(nseed,len(anchors))
        index=dense.detach().amax(-1).topk(available,dim=1).indices
        if available<nseed:
            if not available:raise ValueError("Instance queries require at least one dense anchor")
            index=index.repeat(1,math.ceil(nseed/available))[:,:nseed]
        seed=self.seed(flat.gather(1,index[...,None].expand(-1,-1,flat.shape[-1])))
        refs=(anchors[index]/float(size)).clamp(0,1)
        prior=F.pad(dense.detach().gather(1,index[...,None].expand(-1,-1,self.nc)),(0,1))
        if self.free:
            side_n=math.ceil(math.sqrt(self.free))
            gy,gx=torch.meshgrid(torch.arange(side_n,device=tok.device),torch.arange(side_n,device=tok.device),indexing="ij")
            grid=torch.stack([gx,gy],-1).reshape(-1,2)
            choose=torch.linspace(0,len(grid)-1,self.free,device=tok.device).round().long()
            free_pos=(grid[choose].float()+0.5)/side_n
            refs=torch.cat([refs,free_pos[None].expand(batch,-1,-1)],1)
            seed=torch.cat([seed,tok.new_zeros(batch,self.free,d)],1)
            prior=torch.cat([prior,tok.new_zeros(batch,self.free,self.nc+1)],1)
        q=seed+self.content+self.position(refs)
        stages=[]
        for layer in self.layers:
            q=layer(q,context,memory_key_padding_mask=~valid)
            h=self.norm(q)
            ct=refs+0.25*self.center(h).tanh()
            radius=0.05+0.2*F.softplus(self.radius(h).float())
            distances=(center[:,None]-ct[:,:,None]).square().sum(-1)
            logits=torch.einsum("bqd,bmd->bqm",self.mask_embed(h),memory)/math.sqrt(d)
            logits=logits.float()+2-self.spatial.square()*distances/radius.square()
            stages.append(dict(classes=self.classify(h).float()+prior.float(),masks=logits.masked_fill(~mask[:,None],-30.),centers=ct))
        return stages


def instance_losses(stages,side,targets,soft_f2=False):
    """One matching from final masks; auxiliary stages reuse it. No box-IoU gate.

    All train primitive IDs/instance IDs are targets only, never decoder inputs.
    One combined GPU->CPU cost transfer per batch for SciPy matching.
    """
    from scipy.optimize import linear_sum_assignment
    vec,mask=side["vec"],side["mask"]
    final=stages[-1];device=vec.device
    specifications=[];costs=[]
    ids_cpu=side.get("vec_instance_ids")
    for b,t in enumerate(targets):
        if ids_cpu is None:
            ids=vec[b,:,22][mask[b]].round().long().unique()
            ids=ids[ids>=0].detach().cpu().tolist()
        else:ids=list(ids_cpu[b])
        ids=[i for i in ids if 0<=i<len(t["labels"])]  # an id without a target row (crop/augment) is not supervised
        if len(ids)>final["classes"].shape[1]:           # crowded tile: keep the instances with the most stroke length
            pts_all=vec[b,:,1:17].float().reshape(-1,8,2);seg=(pts_all[:,1:]-pts_all[:,:-1]).norm(dim=-1).sum(-1)
            col=vec[b,:,22].round().long()
            ids=sorted(ids,key=lambda i:-float(seg[(col==i)&mask[b]].sum()))[:final["classes"].shape[1]]
        supervised = not (side.get("vec_has_inst") is not None and
                          side.get("vec_lengths",[1]*len(targets))[b]>0 and not side["vec_has_inst"][b])
        ids=torch.as_tensor(ids,dtype=torch.long,device=device)
        known=mask[b] & (vec[b,:,22]!=-2)
        gt=vec[b,:,22].round().long()[None]==ids[:,None]
        pts=vec[b,:,1:17].float().reshape(-1,8,2)
        weight=(pts[:,1:]-pts[:,:-1]).norm(dim=-1).sum(-1)*known
        weight=weight/weight.sum().clamp_min(1e-8)
        gt=gt.float();labels=t["labels"][ids]
        center=t["boxes"][ids,:2]
        specifications.append((gt,weight,labels,center,supervised))
        pred=final["masks"][b].detach().float().sigmoid()
        inter=(pred*weight)@gt.T
        dice=1-(2*inter+1e-6)/((pred*weight).sum(-1)[:,None]+(gt*weight).sum(-1)[None]+1e-6)
        cls=-final["classes"][b].detach().softmax(-1)[:,labels]
        raw=final["masks"][b].detach().float()
        positive=gt*weight;negative=(1-gt)*weight
        positive=positive/positive.sum(-1,keepdim=True).clamp_min(1e-8)
        negative=negative/negative.sum(-1,keepdim=True).clamp_min(1e-8)
        bce=-(F.logsigmoid(raw)@positive.T+F.logsigmoid(-raw)@negative.T)
        cost=2*cls+2*dice+bce+torch.cdist(final["centers"][b].detach().float(),center.float(),p=1)
        costs.append(cost)
    flat=torch.cat([c.flatten() for c in costs]).cpu().numpy()
    assignments=[];offset=0
    for cost in costs:
        size=cost.numel();array=flat[offset:offset+size].reshape(cost.shape);offset+=size
        row,col=linear_sum_assignment(array)
        assignments.append((torch.as_tensor(row,device=device),torch.as_tensor(col,device=device)))
    sums={k:final["classes"].sum()*0 for k in ("instance_cls","instance_bce","instance_dice","instance_center")}
    if soft_f2:sums["instance_f2"]=final["classes"].sum()*0
    count=max(sum(bool(spec[4]) for spec in specifications)*len(stages),1)
    if not any(spec[4] for spec in specifications):
        return {key:side["tok"].sum()*0 for key in sums}
    for stage in stages:
        for b,((gt,w,labels,ct,supervised),(rows,cols)) in enumerate(zip(specifications,assignments)):
            if not supervised:continue
            cls=stage["classes"][b];target=torch.full((len(cls),),cls.shape[-1]-1,dtype=torch.long,device=device)
            target[rows]=labels[cols]
            class_weight=cls.new_ones(cls.shape[-1]);class_weight[-1]=0.1
            sums["instance_cls"]+=2*F.cross_entropy(cls,target,weight=class_weight)/count
            if not len(rows):continue
            logit=stage["masks"][b,rows].float();truth=gt[cols];pred=logit.sigmoid()
            positive=truth*w;negative=(1-truth)*w
            bce=F.binary_cross_entropy_with_logits(logit,truth,reduction="none")
            sums["instance_bce"]+=2*((bce*positive).sum(-1)/positive.sum(-1).clamp_min(1e-8)+(bce*negative).sum(-1)/negative.sum(-1).clamp_min(1e-8)).mean()/count
            tp=(pred*truth*w).sum(-1);fn=((1-pred)*truth*w).sum(-1);fp=(pred*(1-truth)*w).sum(-1)
            sums["instance_dice"]+=2*(1-(2*tp+1e-6)/(2*tp+fn+fp+1e-6)).mean()/count
            sums["instance_center"]+=F.l1_loss(stage["centers"][b,rows].float(),ct[cols].float())/count
            if soft_f2:sums["instance_f2"]+=(1-(5*tp+1e-6)/(5*tp+4*fn+fp+1e-6)).mean()/count
    return sums
