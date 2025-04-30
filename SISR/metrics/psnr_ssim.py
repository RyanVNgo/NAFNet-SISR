
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr

def PSNR(x, y):
    return _psnr(y, x)

def SSIM(x, y):
    return _ssim(x, y, channel_axis=2)

