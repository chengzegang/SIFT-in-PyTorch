import torch
from torch import Tensor


def proj(x: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """
    x: (*, N, 3)
    R: (*, 3, 3)
    t: (*, 3)
    """
    x = (x @ R.transpose(-1, -2)) + t.unsqueeze(-2)
    return x


def reproj(x: Tensor, R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor) -> Tensor:
    x = proj(x, R1.transpose(-1, -2), -t1)
    x = proj(x, R2, t2)
    return x


def creproj(x: Tensor, R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor) -> Tensor:
    """
    x: (B1, N, 3)
    R1: (B1, 3, 3)
    t1: (B1, 3)
    R2: (B2, 3, 3)
    t2: (B2, 3)
    """
    x = proj(x, R1.transpose(-1, -2), -t1)  # (B, N, 3)
    x = x.unsqueeze(1)
    R2 = R2.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    x = proj(x, R2, t2)  # (B, B, N, 3)
    return x


def sreproj(x: Tensor, R: Tensor, t: Tensor, loc: Tensor | None = None) -> Tensor:
    """
    x: (B, N, 3)
    R: (B, 3, 3)
    t: (B, 3)
    """
    B, N, _ = x.shape
    x = proj(x, R.transpose(-1, -2), -t)  # (B, N, 3)
    if loc is not None:
        x = x + loc
    x = x.unsqueeze(1)
    R = R.unsqueeze(0)
    t = t.unsqueeze(0)
    x = proj(x, R, t)  # (B, B, N, 3)
    return x


def _spherical(XYZ: Tensor) -> Tensor:
    """Converts 3D cartesian coordinates to spherical coordinates.

    Args:
        XYZ (Tensor): Tensor of 3D cartesian coordinates with shape :math:`(B, N, 3)`.
    Returns:
        Tensor: Tensor of 2D spherical coordinates with shape :math:`(B, N, 2)`.
    """
    XYZ = XYZ.view(XYZ.shape[0], -1, XYZ.shape[-1])

    lat = torch.asin(XYZ[..., 2])
    lon = torch.atan2(XYZ[..., 1], XYZ[..., 0])

    x = lon / torch.pi
    y = lat / torch.pi * 2

    xy = torch.stack((x, y), dim=-1)
    return xy


def _spherical_inv(xy: Tensor) -> Tensor:
    """Converts 2D spherical coordinates to 3D cartesian coordinates.

    Args:
        xy (Tensor): Tensor of 2D spherical coordinates with shape :math:`(B, N, 2)`.

    Returns:
        Tensor: Tensor of 3D cartesian coordinates with shape :math:`(B, N, 3)`.
    """
    xy = xy.view(xy.shape[0], -1, xy.shape[-1])

    lon = torch.pi * xy[..., 0]
    lat = torch.pi * xy[..., 1] / 2

    XYZ = torch.stack(
        (
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ),
        dim=-1,
    )

    return XYZ


def spherical(x: Tensor, inverse: bool = False) -> Tensor:
    if inverse:
        return _spherical_inv(x)
    else:
        return _spherical(x)
