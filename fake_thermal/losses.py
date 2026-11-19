import torch
from torch.nn import functional as F


def ssim(pred, target):
    mu_x = F.avg_pool2d(pred, 7, 1)
    mu_y = F.avg_pool2d(target, 7, 1)
    var_x = F.avg_pool2d(pred.square(), 7, 1) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), 7, 1) - mu_y.square()
    cov = F.avg_pool2d(pred * target, 7, 1) - mu_x * mu_y
    return (((2 * mu_x * mu_y + .01 ** 2) * (2 * cov + .03 ** 2)) /
            ((mu_x.square() + mu_y.square() + .01 ** 2) *
             (var_x.clamp_min(0) + var_y.clamp_min(0) + .03 ** 2))).mean().clamp(-1, 1)


def gradient_loss(pred, target):
    dx = F.l1_loss(pred[..., :, 1:] - pred[..., :, :-1],
                   target[..., :, 1:] - target[..., :, :-1])
    dy = F.l1_loss(pred[..., 1:, :] - pred[..., :-1, :],
                   target[..., 1:, :] - target[..., :-1, :])
    return .5 * (dx + dy)


def thermal_loss(pred, target):
    return F.l1_loss(pred, target) + .5 * (1 - ssim(pred, target)) + .1 * gradient_loss(pred, target)
