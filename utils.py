import torch

def xy_to_cxcy(xy):
    return torch.cat([
        (xy[:, 2:] + xy[:, :2]) / 2, # find center
        xy[:, 2:] - xy[:, :2],       # find width and height
    ], dim=1)


def cxcy_to_xy(cxcy):
    return torch.cat([
        cxcy[:, :2] - cxcy[:, 2:] / 2,
        cxcy[:, :2] + cxcy[:, 2:] / 2,
    ], dim=1)


def cxcy_to_gcxgcy(cxcy, priors):
    # Empirically chosen variances from the original paper for scaling the gradient
    var_1 = 10
    var_2 = 5 

    return torch.cat([
        (cxcy[:, :2] - priors[:, :2]) / (priors[:, 2:] / var_1),
        torch.log(cxcy[:, 2:] / priors[:, 2:]) * var_2
    ], dim=1)


def gcxgcy_to_cxcy(gcxgcy, priors):
    var_1 = 10
    var_2 = 5

    return torch.cat([
        gcxgcy[:, :2] * priors[:, 2:] / var_1 + priors[:, :2],
        torch.exp(gcxgcy[:, 2:] / var_2) * priors[:, 2:]
    ], dim=1)


def find_intersection(a, b):
    lower_bounds = torch.max(a[:, :2].unsqueeze(1), b[:, :2].unsqueeze(0))
    upper_bounds = torch.min(a[:, 2:].unsqueeze(1), b[:, 2:].unsqueeze(0))
    dims = torch.clamp(upper_bounds - lower_bounds, min=0)

    return dims[:, :, 0] * dims[:, :, 1]


def compute_area(a):
    xmin = a[:, 0]
    xmax = a[:, 2]
    ymin = a[:, 1]
    ymax = a[:, 3]

    return (xmax - xmin) * (ymax - ymin)


def find_iou(a, b):
    intersection = find_intersection(a, b)

    # Calculate the area of each box in both sets
    a_areas = compute_area(a)
    b_areas = compute_area(b)

    # Simply the sum of the two areas minus the intersection (to not double count it)
    union = a_areas.unsqueeze(1) + b_areas.unsqueeze(0) - intersection

    return intersection / union # hence the name "intersection over union


def decimate(v, m):
    assert v.dim() == len(m)
    
    for d in range(v.dim()):
        if m[d] is not None:
            v = v.index_select(
                dim=d,
                index=torch.arange(start=0, end=v.size(d), step=m[d]).long(),
            )
    
    return v


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

