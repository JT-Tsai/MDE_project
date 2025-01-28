import torch
import torch.nn as nn
import math

def psnr(sr, hr, PIXEL_MAX = 1.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return math.inf
    else:
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
class PSNR(nn.Module):
    def __init__(self, PIXEL_MAX = 1.0):
        super(PSNR, self).__init__()
        self.PIXEL_MAX = PIXEL_MAX

    def forward(self, sr, hr):
        mse = torch.mean((sr - hr) ** 2)
        return 20 * torch.log10(self.PIXEL_MAX / torch.sqrt(mse))