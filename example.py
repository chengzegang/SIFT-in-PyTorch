from src.torchsift import sift
from src.torchsift.ransac.matcher import match
import torch

x = torch.randn(2, 3, 128, 256).float()
y = torch.randn(2, 3, 128, 256).float()


model = sift.SIFT(512)
xk, xd = model(x)
yk, yd = model(y)
fs, inliners, errs = match(xk, xd, yk, yd)
