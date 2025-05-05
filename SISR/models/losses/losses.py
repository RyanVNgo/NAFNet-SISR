
import torch
import torch.nn as nn
import numpy as np
from torcheval.metrics import PeakSignalNoiseRatio


class L1Loss(nn.Module):
    def __init__(self, weight=1.0):
        super(L1Loss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        return self.weight * nn.functional.l1_loss(pred, target)


class MSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super(MSELoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        return self.weight * nn.functional.mse_loss(pred, target)


class PSNRLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(PSNRLoss, self).__init__()
        self.weight = weight
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2) + 1e-8
        return self.weight * -20 * torch.log10(1.0 / torch.sqrt(mse))


class HuberLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(HuberLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        return self.weight * nn.functional.huber_loss(pred, target)

