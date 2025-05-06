
import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as _ssim


def rgb_to_y_tensor(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def rgb_to_y_np(rgb_np):
    r, g, b = rgb_np[..., 0], rgb_np[..., 1], rgb_np[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def gaussian_kernel(window_size, sigma):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g[:, None] * g[None, :]


def PSNR(x, y, max_val=1.0):
    x = rgb_to_y_np(x)
    y = rgb_to_y_np(y)
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr 


def PSNR_Tensor(x, y, max_val=1.0):
    x = rgb_to_y_tensor(x)
    y = rgb_to_y_tensor(y)
    mse = F.mse_loss(x, y, reduction='none').mean(dim=[1,2,3])
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.mean().item()


def SSIM(x, y):
    return _ssim(x, y, channel_axis=2)


