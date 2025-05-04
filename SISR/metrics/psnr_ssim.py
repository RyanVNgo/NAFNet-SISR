
import numpy as np
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr
from torcheval.metrics import PeakSignalNoiseRatio

def PSNR(x, y):
    return _psnr(y, x)


def PSNR_Tensor(x, y, device='cpu'):
    metric = PeakSignalNoiseRatio(data_range=1.0, device=device)
    metric.update(x, y)
    return metric.compute()


def SSIM(x, y):
    return _ssim(x, y, channel_axis=2)


