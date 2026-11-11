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
    def __init__(self,vec_dim,image_channels,neck_channels,nc,queries=256,free=128,line_context=True,
                 seed="boxes",roi=False,masked=False):
        super().__init__()
        if seed not in ("boxes","strokes"):raise ValueError("instance_seed must be 'boxes' or 'strokes'")
        self.seed_from,self.masked=seed,bool(masked)
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
        # OpenStroke16 (docs/OPENSTROKE16.md): (A) queries seeded by strokes, (B) image check of each query's rotated region,
        # (C) masked attention on the strokes of that region and an elliptic region prior. New outputs start at zero.
        self.objectness=nn.Linear(vec_dim,1) if seed=="strokes" else None
        self.seed_stroke=nn.Linear(vec_dim,vec_dim) if seed=="strokes" else None
        if self.objectness is not None:
            nn.init.zeros_(self.objectness.weight);nn.init.zeros_(self.objectness.bias)
        self.roi=nn.Sequential(nn.Linear(2*vec_dim,vec_dim),nn.GELU(),nn.Linear(vec_dim,vec_dim)) if roi else None
        if self.roi is not None:
            nn.init.zeros_(self.roi[-1].weight);nn.init.zeros_(self.roi[-1].bias)

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
        objectness=None
        if self.objectness is not None:
            objectness=self.objectness(memory).squeeze(-1).float()
            seed,refs=self.stroke_seeds(memory,center,mask,objectness,nseed)
            prior=memory.new_zeros(batch,nseed,self.nc+1)
        else:
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
        if not (self.masked or self.roi is not None):   # OpenStroke15, unchanged
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
            if objectness is not None:
                stages[-1]["objectness"]=objectness
            return stages
        image_d=self.image(f4) if self.roi is not None else None
        stages=[]
        prev,region=self.predict(q,refs,memory,center,mask,None)   # Mask2Former: first regions from the initial queries
        for layer in self.layers:
            attn=None
            if self.masked:
                allow=(prev.detach()>0)&mask[:,None]
                allow=allow|(~allow.any(-1,keepdim=True)&mask[:,None])        # an empty region sees every stroke
                allow=torch.cat([allow,allow.new_ones(*allow.shape[:2],1)],-1)  # the drawing token is always visible
                attn=(~allow).repeat_interleave(layer.multihead_attn.num_heads,0)
            q=layer(q,context,memory_mask=attn,memory_key_padding_mask=~valid)
            logits,region_next=self.predict(q,refs,memory,center,mask,region)
            h=self.norm(q)
            if self.roi is not None:
                h=h+self.roi_feature(image_d,region_next,size,f4)
            ct=refs+0.25*self.center(self.norm(q)).tanh()
            stages.append(dict(classes=self.classify(h).float()+prior.float(),masks=logits.masked_fill(~mask[:,None],-30.),centers=ct))
            prev,region=logits,region_next
        if objectness is not None:
            stages[-1]["objectness"]=objectness
        return stages

    def predict(self,q,refs,memory,center,mask,region):
        """Mask logits of every query over the strokes (OpenStroke16). Prior: the circle around the query (as v15) until a
        region is known, then the ellipse of that region when masked=True. Returns (logits, region of these logits)."""
        d=memory.shape[-1]
        h=self.norm(q)
        logits=torch.einsum("bqd,bmd->bqm",self.mask_embed(h),memory).float()/math.sqrt(d)
        ct=refs+0.25*self.center(h).tanh()
        radius=0.05+0.2*F.softplus(self.radius(h).float())
        distances=(center[:,None]-ct[:,:,None]).square().sum(-1)/radius.square()
        if region is not None and self.masked:
            mean,inv,_,ok=region
            diff=center[:,None].float()-mean[:,:,None]
            maha=torch.einsum("bqmi,bqij,bqmj->bqm",diff,inv,diff)
            distances=torch.where(ok[...,None],maha,distances)
        logits=logits+2-self.spatial.square()*distances
        return logits,self.region(logits.masked_fill(~mask[:,None],-30.),center,mask)

    @staticmethod
    def region(logits,center,mask):
        """Rotated region of each query from its soft stroke set (detached): mean, inverse of the (2 sigma)^2 covariance,
        (half axes, cos, sin), valid flag."""
        w=logits.detach().float().sigmoid()*mask[:,None]
        total=w.sum(-1,keepdim=True)
        c=center.float()[:,None]
        mean=(w[...,None]*c).sum(2)/total.clamp_min(1e-6)
        diff=c-mean[:,:,None]
        cov=torch.einsum("bqm,bqmi,bqmj->bqij",w,diff,diff)/total[...,None].clamp_min(1e-6)
        cov=4*cov+torch.eye(2,device=cov.device)*1e-4                      # (2 sigma)^2, floor 1 % of the tile
        a,b,cc=cov[...,0,0],cov[...,0,1],cov[...,1,1]
        tr,det=a+cc,a*cc-b*b
        root=((tr/2).square()-det).clamp_min(0).sqrt()
        l1,l2=tr/2+root,(tr/2-root).clamp_min(1e-4)
        angle=0.5*torch.atan2(2*b,a-cc)
        inv=torch.stack([torch.stack([cc,-b],-1),torch.stack([-b,a],-1)],-2)/det.clamp_min(1e-10)[...,None,None]
        axes=torch.stack([l1.sqrt(),l2.sqrt()],-1)
        return mean,inv,(axes,angle.cos(),angle.sin()),total.squeeze(-1)>0.5

    def roi_feature(self,image,region,size,f4,k=5):
        """(B) image features on a k x k grid inside each query's rotated region (mean + max), zero-initialised MLP."""
        mean,_,(axes,cos,sin),ok=region
        u=torch.linspace(-1,1,k,device=image.device)
        gu,gv=torch.meshgrid(u,u,indexing="ij")
        lu,lv=gu.reshape(-1)[None,None],gv.reshape(-1)[None,None]
        lx,ly=lu*axes[...,:1],lv*axes[...,1:]
        x=mean[...,:1]+lx*cos[...,None]-ly*sin[...,None]
        y=mean[...,1:]+lx*sin[...,None]+ly*cos[...,None]
        pts=torch.stack([x,y],-1)                                            # (B, Q, k*k, 2) in tile units
        extent=pts.new_tensor([f4.shape[-1]*4,f4.shape[-2]*4])
        grid=(pts*float(size)/extent*2-1).to(image.dtype)
        sample=F.grid_sample(image,grid,align_corners=False)                # (B, d, Q, k*k)
        feat=torch.cat([sample.mean(-1),sample.amax(-1)],1).transpose(1,2)   # (B, Q, 2d)
        return self.roi(feat)*ok[...,None]

    def stroke_seeds(self,memory,center,mask,objectness,nseed,cells=32):
        """(A) the nseed strokes most likely to belong to an object, at most one per cell of a cells x cells grid."""
        batch,m,d=memory.shape
        score=objectness.detach().masked_fill(~mask,-1e9)
        cell=(center.detach().float().clamp(0,1-1e-6)*cells).long()
        cell=cell[...,1]*cells+cell[...,0]
        best=torch.full((batch,cells*cells),-1e9,device=score.device).scatter_reduce(1,cell,score,"amax",include_self=True)
        score=score.masked_fill(score<best.gather(1,cell),-1e9)
        if m<nseed:
            score=F.pad(score,(0,nseed-m),value=-1e9)
        top,index=score.topk(nseed,dim=1)
        live=top>-1e8
        index=index.clamp_max(max(m-1,0))
        if m:
            picked=memory.gather(1,index[...,None].expand(-1,-1,d))
            refs=center.gather(1,index[...,None].expand(-1,-1,2)).float().clamp(0,1)
        else:
            picked=memory.new_zeros(batch,nseed,d);refs=memory.new_zeros(batch,nseed,2).float()
        seed=self.seed_stroke(picked)*live[...,None]
        refs=torch.where(live[...,None],refs,torch.full_like(refs,0.5))
        return seed,refs


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
        zero=side["tok"].sum()*0+(final["objectness"].sum()*0 if "objectness" in final else 0)
        return {key:zero for key in list(sums)+(["instance_seed"] if "objectness" in final else [])}
    if "objectness" in final:                     # (A): every stroke of an object is a seed candidate, recall-weighted
        sums["instance_seed"]=final["classes"].sum()*0
        n_sup=max(sum(bool(x[4]) for x in specifications),1)
        for b,spec in enumerate(specifications):
            if not spec[4]:continue
            known=(mask[b]&(vec[b,:,22]!=-2)).float()
            target=(vec[b,:,22]>=0).float()
            loss=F.binary_cross_entropy_with_logits(final["objectness"][b],target,reduction="none",
                                                    pos_weight=final["objectness"].new_tensor(3.))
            sums["instance_seed"]+=(loss*known).sum()/known.sum().clamp_min(1)/n_sup
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
