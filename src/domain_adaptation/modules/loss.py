import torch
import math
def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    # source covariance
    tmp_s = torch.ones((1, ns)).type_as(source) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).type_as(target) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)
    
    if math.isnan(loss):
        loss = torch.tensor(0.)

    return loss