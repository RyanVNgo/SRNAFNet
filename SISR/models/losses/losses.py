
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.amp import autocast

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


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, conv_index, rgb_range=1, weight=1.0, device='cpu'):
        super(VGGLoss, self).__init__()
        vgg_features = models.vgg19(weights='DEFAULT').features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        self.device = device
        self.vgg.to(device)
        self.sub_mean.to(device)
        self.weight = weight

    def forward(self, sr, hr):
        with autocast(self.device, enabled=False):
            sr = sr.to(self.device).float()
            hr = hr.to(self.device).float()
            def _forward(x):
                x = self.sub_mean(x)
                x = self.vgg(x)
                return x
                
            vgg_sr = _forward(sr)
            with torch.no_grad():
                vgg_hr = _forward(hr.detach())

            loss = F.mse_loss(vgg_sr, vgg_hr) 

        return loss * self.weight
