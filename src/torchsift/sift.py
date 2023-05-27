from __future__ import annotations

from typing import Tuple, List

import torch
from kornia.color import rgb_to_grayscale
from kornia.feature import (
    BlobDoG,
    LAFOrienter,
    ScaleSpaceDetector,
    SIFTDescriptor,
    extract_patches_from_pyramid,
    get_laf_center,
)
import torchvision.transforms.functional as TF  # type: ignore
from PIL import Image
from kornia.geometry import ConvQuadInterp3d, ScalePyramid
from torch.nn import Module
import torch.nn.functional as F


class SIFT(Module):
    def __init__(
        self,
        num_features: int = 512,
        patch_size: int = 41,
        angle_bins: int = 8,
        spatial_bins: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.descriptor = SIFTDescriptor(
            patch_size, angle_bins, spatial_bins, rootsift=True
        )
        self.detector = ScaleSpaceDetector(
            num_features,
            resp_module=BlobDoG(),
            scale_space_response=True,  # We need that, because DoG operates on scale-space
            nms_module=ConvQuadInterp3d(10),
            scale_pyr_module=ScalePyramid(3, 1.6, patch_size, double_image=True),
            ori_module=LAFOrienter(19),
            mr_size=6.0,
            minima_are_also_good=True,
        )

    def detect(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.detector.to(x.device)
        with torch.no_grad():
            lafs, resps = self.detector(x.contiguous())
            return lafs, resps

    def describe(self, x: torch.Tensor, lafs: torch.Tensor) -> torch.Tensor:
        self.descriptor.to(x.device)
        with torch.no_grad():
            patches = extract_patches_from_pyramid(x, lafs, self.patch_size)
            B, N, CH, H, W = patches.size()
            descs = self.descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
            descs = F.normalize(descs, dim=-1, p=2)
            return descs  # type: ignore

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x = rgb_to_grayscale(x).float()
        lafs, resps = self.detect(x)
        descs = self.describe(x, lafs)
        kpts = get_laf_center(lafs)
        x = kpts[..., 0]
        y = kpts[..., 1]
        x = x / W * 2 - 1
        y = y / H * 2 - 1
        kpts = torch.stack([x, y], dim=-1)
        return kpts, descs


def detect(
    paths: List[str], num_features: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    descs = []
    kpts = []
    sift = SIFT(num_features=num_features).cuda()
    for path in paths:
        img = Image.open(path)
        img = TF.pil_to_tensor(img).float().cuda().unsqueeze(0)
        k, d = sift(img)
        descs.append(d.cpu())
        kpts.append(k.cpu())
    descs_t = torch.cat(descs, dim=0)
    kpts_t = torch.cat(kpts, dim=0)
    return kpts_t, descs_t
