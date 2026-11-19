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


def focal_frequency_loss(pred, target, alpha=1.):
    """Focal Frequency Loss (Jiang et al., ICCV 2021): adaptively weight hard
    frequencies. Weights are detached spectral magnitudes raised to alpha and
    normalized per sample, so the term stays bounded by the squared error."""
    x = torch.fft.rfft2(pred.float(), dim=(-2, -1), norm='ortho')
    y = torch.fft.rfft2(target.float(), dim=(-2, -1), norm='ortho')
    distance = (x - y).abs().square()
    weight = distance.detach().pow(alpha / 2)
    peak = weight.flatten(1).max(dim=1).values.clamp_min(1e-8)
    return (weight / peak.view(-1, 1, 1, 1) * distance).mean()


def thermal_loss(pred, target, frequency_alpha=0.):
    loss = F.l1_loss(pred, target) + .5 * (1 - ssim(pred, target)) + .1 * gradient_loss(pred, target)
    if frequency_alpha:
        loss = loss + focal_frequency_loss(pred, target, frequency_alpha)
    return loss
