from torchsift import sift
from torchsift.ransac.matcher import match
import torch

x = torch.randn(2, 3, 128, 256).float()
y = torch.randn(2, 3, 128, 256).float()


sift = sift.SIFT(512)
xk, xd = sift(x)
yk, yd = sift(y)
fs, inliners, errs = match(xk, xd, yk, yd)
