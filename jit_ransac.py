from typing import Literal, Optional, Tuple, overload

import ot  # type: ignore
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sampson(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
    # x (B, N, 3)
    # y (B, N, 3)
    # F (B, 3, 3)
    Fx = torch.matmul(F, x.transpose(-1, -2)).transpose(-1, -2)  # (B, N, 3)
    Fty = torch.matmul(F.transpose(-1, -2), y.transpose(-1, -2)).transpose(
        -1, -2
    )  # (B, N, 3)
    d: Tensor = (y * Fx).sum(dim=-1) ** 2 / (Fx[..., :2] ** 2 + Fty[..., :2] ** 2).sum(
        dim=-1
    )  # (B, N)
    return d


@torch.jit.script
def find_fundamental3d(p1: Tensor, p2: Tensor, w: Tensor) -> Tensor:
    x1, y1, z1 = torch.chunk(p1, dim=-1, chunks=3)  # Bx1xN
    x2, y2, z2 = torch.chunk(p2, dim=-1, chunks=3)  # Bx1xN

    X = torch.cat(
        [
            x1 * x2,
            x1 * y2,
            z1 * z2,
            y1 * x2,
            y1 * y2,
            y1 * z2,
            z1 * x2,
            z1 * y2,
            z1 * z2,
        ],
        dim=-1,
    )
    X = X.transpose(-2, -1) @ torch.diag_embed(w).transpose(-1, -2) @ X
    _, _, V = torch.linalg.svd(X)
    f: Tensor = V[..., -1].reshape(-1, 3, 3)
    return f


@torch.jit.script
def _one_step_ransac_solve_fundamental_for_epipolar_sampson_errors(
    src_points: Tensor, tgt_points: Tensor, weights: Tensor, sample_raito: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """Run one iteration of ransac
    src_points (B, N, D)
    tgt_points (B, N, D)
    weights (B, N)
    """
    sample_size = int(src_points.shape[-2] * sample_raito)
    index = torch.randperm(src_points.shape[-2])[:sample_size]
    src_samples = src_points.index_select(-2, index)
    tgt_samples = tgt_points.index_select(-2, index)
    sample_weights = weights.index_select(-1, index)
    f = find_fundamental3d(src_samples, tgt_samples, sample_weights)
    sample_errs = sampson(src_samples, tgt_samples, f)
    sample_err_vars = torch.var(sample_errs, dim=-1).clamp(min=1e-8)
    errs = sampson(src_points, tgt_points, f)
    inliers = errs < sample_err_vars.unsqueeze(-1)
    return errs, inliers


@torch.jit.script
def ransac(
    src_points: Tensor,
    tgt_points: Tensor,
    weights: Optional[Tensor] = None,
    sample_ratio: float = 0.5,
    iters: int = 32,
):
    if weights is None:
        weights = torch.ones_like(src_points[..., 0], dtype=torch.float32)
    B, N, D = src_points.shape
    B, M, D = tgt_points.shape
    min_avg_error = torch.full(
        (B,),
        dtype=torch.float32,
        fill_value=float("inf"),
        device=src_points.device,
    )
    errs = torch.full(
        (B, N),
        dtype=torch.float32,
        fill_value=float("inf"),
        device=src_points.device,
    )
    inliers = torch.empty(
        (B, N),
        device=errs.device,
        dtype=torch.bool,
    )
    for _ in range(iters):
        (
            curr_error,
            curr_inliners,
        ) = _one_step_ransac_solve_fundamental_for_epipolar_sampson_errors(
            src_points, tgt_points, weights, sample_ratio
        )
        curr_avg_error = torch.nanmean(curr_error, dim=-1)
        update = curr_avg_error < min_avg_error
        errs[update] = curr_error[update]
        inliers[update] = curr_inliners[update]

    return errs, inliers

@torch.jit.script
def ratio_test_threshold_match(
    xd: Tensor, yd: Tensor, r: float = 0.5
) -> Tuple[Tensor, Tensor]:
    sims = torch.matmul(xd, yd.transpose(-1, -2))
    top2_d, top2_ind = torch.topk(sims, 2, dim=-1, largest=True)
    valid = top2_d[..., 0] * r > top2_d[..., 1]
    tgt_ind = top2_ind[..., 0]
    return tgt_ind, valid
