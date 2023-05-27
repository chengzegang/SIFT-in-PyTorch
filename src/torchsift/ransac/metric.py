
import torch
from torch import Tensor


from .convert import adj_to_list, to_homogeneous


def sampson(
    Fm: Tensor, pts1: Tensor, pts2: Tensor, mask: Tensor, eps: float = 1e-8
) -> Tensor:
    """Return Sampson distance for correspondences given the fundamental matrix.

    Args:
        pts1: correspondences from the left images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        pts2: correspondences from the right images with shape :math:`(*, N, (2|3))`. If they are not homogeneous,
              converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to avoid ambiguity with torch.nn.functional.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.
    """
    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = to_homogeneous(pts1)

    if pts2.shape[-1] == 2:
        pts2 = to_homogeneous(pts2)

    pts1, pts2, pair_mask = adj_to_list(pts1, pts2, mask)
    F_t: Tensor = Fm.transpose(-1, -2)
    line1_in_2: Tensor = pts1 @ F_t  # (B, N, D) @ (B, D, D) -> (B, N, D)
    line2_in_1: Tensor = pts2 @ Fm  # (B, N, D) @ (B, D, D) -> (B, N, D)

    # numerator = (x'^T F x) ** 2
    numerator: Tensor = (pts2 * line1_in_2).sum(dim=-1).pow(2)

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator: Tensor = line1_in_2[..., :2].norm(2, dim=-1).pow(2) + line2_in_1[
        ..., :2
    ].norm(2, dim=-1).pow(2)
    out: Tensor = numerator / denominator
    out_mat = torch.full_like(mask, torch.nan).type_as(out)
    out_mat[mask > 0] = out[pair_mask > 0]
    return out_mat
