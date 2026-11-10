"""OpenStroke9: endpoint image samples and box-relative geometry for vector membership."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb8 import _sample


@torch.no_grad()
def endpoint_geometry(boxes, pts, box_ids, point_ids):
    """Six features per gated pair, invariant to stroke reversal and box axis sign."""
    box = boxes[box_ids].float()
    points = pts.float()
    ends = points[:, (0, -1)]
    tangents = points[:, (1, -1)] - points[:, (0, -2)]
    tangents = F.normalize(tangents, dim=-1)[point_ids]

    # Use the long axis even when the box representation stores it as height.
    wide = box[:, 2] >= box[:, 3]
    c, s = box[:, 4].cos(), box[:, 4].sin()
    axis = torch.stack([torch.where(wide, c, -s), torch.where(wide, s, c)], dim=-1)
    normal = torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)
    half_length = box[:, 2:4].amax(dim=1).clamp_min(1) / 2
    half_width = box[:, 2:4].amin(dim=1).clamp_min(1) / 2
    offset = ends[point_ids] - box[:, None, :2]
    axial = (offset * axis[:, None]).sum(-1).abs() / half_length[:, None] - 1
    lateral = (offset * normal[:, None]).sum(-1).abs() / half_width[:, None] - 1
    parallel = (tangents * axis[:, None]).sum(-1).abs()
    return torch.stack([axial.amin(1), axial.amax(1), lateral.amin(1), lateral.amax(1),
                        parallel.amin(1), parallel.amax(1)], dim=-1).clamp(-4, 4)


class EndpointMembers(nn.Module):
    def __init__(self, image_channels, query_dim):
        super().__init__()
        self.image = nn.Linear(image_channels, query_dim, bias=False)
        self.geometry = nn.Linear(query_dim, 6)
        nn.init.zeros_(self.image.weight)
        nn.init.zeros_(self.geometry.weight)
        nn.init.zeros_(self.geometry.bias)

    def forward(self, logit, query, boxes, pts, gate, f4):
        # P2 comes from the existing detector. Only the projection learns from this route.
        with torch.no_grad():
            ends = pts[:, (0, -1)].float()
            height, width = f4.shape[-2:]
            visible = ((ends >= 0) & (ends < ends.new_tensor([4 * width, 4 * height]))).all(-1)
            samples = _sample(f4[None], ends.flatten(0, 1)[None], stride=4)[0]
            samples = samples.reshape(len(ends), 2, f4.shape[0])
            pooled = (samples * visible[..., None]).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        logit = logit + (query @ self.image(pooled).t()) * (query.shape[-1] ** -0.5)

        # Materialize the geometry only for the pairs already admitted by the v25 gate.
        box_ids, point_ids = gate.nonzero(as_tuple=True)
        geometry = endpoint_geometry(boxes, pts, box_ids, point_ids)
        scores = (self.geometry(query)[box_ids] * geometry).sum(-1)
        return logit.index_put((box_ids, point_ids), scores.to(logit.dtype), accumulate=True)
