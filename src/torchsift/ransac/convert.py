from typing import Callable, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F


def adj_to_list(
    x: Tensor, y: Tensor, mask: Tensor, padding: float = 0
) -> Tuple[Tensor, Tensor, Tensor]:
    x_pair = []
    y_pair = []
    for x1, x2, m in zip(x, y, mask):
        # print(x1.shape, x2.shape, m.shape)
        indices = torch.nonzero(m, as_tuple=True)
        i = indices[-2]
        j = indices[-1]
        if torch.numel(i) == 0 or torch.numel(j) == 0:
            x_pair.append(torch.full(x[0].size(), torch.nan, device=x.device))
            y_pair.append(torch.full(y[0].size(), torch.nan, device=y.device))
            continue
        x_pair.append(x1[..., i, :])
        y_pair.append(x2[..., j, :])
    x_pair = torch.nested.as_nested_tensor(x_pair, device=x.device).to_padded_tensor(  # type: ignore
        torch.nan
    )
    y_pair = torch.nested.as_nested_tensor(y_pair, device=y.device).to_padded_tensor(  # type: ignore
        torch.nan
    )
    mask = torch.isfinite(x_pair).all(dim=-1) & torch.isfinite(y_pair).all(dim=-1)  # type: ignore
    x_pair = x_pair.nan_to_num(padding)  # type: ignore
    y_pair = y_pair.nan_to_num(padding)  # type: ignore
    return x_pair, y_pair, mask  # type: ignore


def to_homogeneous(points: Tensor) -> Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    h_points = F.pad(points, [0, 1], "constant", 1.0)
    return h_points
