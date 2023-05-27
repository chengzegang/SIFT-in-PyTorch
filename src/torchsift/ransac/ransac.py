import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable, Tuple
from .solver import find_fundamental_equirectangular
from .metric import sampson


def _batch_randperm(b: int, size: int, device: str | torch.device = "cpu") -> Tensor:
    r"""Generate a batch of random permutations.

    Args:
        b (int): Batch size.
        size (int): Size of the permutations.
        device (str | torch.device, optional): Device to use. Defaults to 'cpu'.

    Returns:
        Tensor: A batch of random permutations.

    """

    perms = torch.argsort(torch.rand((b, size), device=device), dim=-1)
    return perms


def ransac(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    solver: Callable | Module = find_fundamental_equirectangular,
    evaluator: Callable | Module = sampson,
    ransac_ratio: float = 0.6,
    ransac_it: int = 8,
    ransac_thr: float = 0.8,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""RANSAC algorithm to find the best model.

    Args:
        x (Tensor): The first set of features with shape :math:`(B, N, D)`.
        y (Tensor): The second set of features with shape :math:`(B, M, D)`.
        mask (Tensor): The mask with shape :math:`(B, N, M)`.
        solver (Callable): The solver function to find the model.
        evaluator (Callable): The evaluator function to evaluate the model.
        ransac_ratio (float, optional): The ratio of inliers to consider the model as the best. Defaults to 0.6.
        ransac_it (int, optional): The number of iterations. Defaults to 16.
        ransac_thr (float, optional): The threshold to consider a point as an inlier. Defaults to 0.75.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The matching mask with shape :math:`(B, N, M)`.
    """

    B, N, D = x.shape
    B, M, D = y.shape

    r_n = int(ransac_ratio * N)
    r_m = int(ransac_ratio * M)
    perm1 = _batch_randperm(ransac_it, r_n, device=x.device).view(-1)
    perm2 = _batch_randperm(ransac_it, r_m, device=x.device).view(-1)

    s_x = x[:, perm1].view(B * ransac_it, r_n, D)
    s_y = y[:, perm2].view(B * ransac_it, r_m, D)
    s_m = mask[:, perm1].view(B, ransac_it, r_n, M)
    s_m = s_m.gather(-1, perm2.view(1, ransac_it, 1, r_m).repeat(B, 1, r_n, 1)).view(
        B * ransac_it, r_n, r_m
    )  # (B * ransac_it, N, M)

    models = solver(s_x, s_y, s_m)  # (B * ransac_it, D, D)
    x = x.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    y = y.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    mask = mask.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    errors = evaluator(models, x, y, mask)  # (B * ransac_it, N, M)
    errors = errors.view(B, ransac_it, N, M)
    models = models.view(B, ransac_it, models.shape[-2], models.shape[-1])
    avg_errors = torch.nanmean(errors, dim=(-1, -2))
    best_model_idx = torch.argmin(avg_errors, dim=-1)

    best_model = torch.gather(
        models,
        dim=1,
        index=best_model_idx.view(-1, 1, 1, 1).repeat(
            1, 1, models.shape[-2], models.shape[-1]
        ),
    ).squeeze(1)

    best_errors = torch.gather(
        errors,
        dim=1,
        index=best_model_idx.view(-1, 1, 1, 1).repeat(
            1, 1, errors.shape[-2], errors.shape[-1]
        ),
    ).squeeze(1)
    thrs = torch.nanquantile(
        best_errors.flatten(-2), ransac_thr, dim=-1, keepdim=True
    ).unsqueeze(-1)
    inliers = best_errors <= thrs
    best_errors[~inliers] = torch.nan
    return best_model, inliers, best_errors
