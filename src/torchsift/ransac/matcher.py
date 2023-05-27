import torch
from torch import Tensor
from .ransac import ransac
import ot  # type: ignore


def optimal_transport(xd: Tensor, yd: Tensor) -> Tensor:
    costs = torch.cdist(xd, yd, p=2)
    a = torch.ones(xd.shape[1], dtype=torch.float32, device=xd.device) / xd.shape[0]
    b = torch.ones(yd.shape[1], dtype=torch.float32, device=xd.device) / yd.shape[0]
    gs = []
    for i in range(xd.shape[0]):
        c = costs[i]
        g = ot.emd(a, b, c, numItermax=5000, numThreads="max")
        gs.append(g)
    gs = torch.stack(gs, dim=0)  # type: ignore
    gs = gs / gs.amax(dim=(-1, -2), keepdim=True)  # type: ignore
    return gs  # type: ignore


def match(xk: Tensor, xd: Tensor, yk: Tensor, yd: Tensor, **kwargs):
    w = optimal_transport(xd, yd)
    best_model, inliers, best_errors = ransac(xk, yk, w, **kwargs)
    return best_model, inliers, best_errors
