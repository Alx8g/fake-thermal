"""Conditional multi-scale PatchGAN discriminator (pix2pixHD pattern).

Regression losses (L1, SSIM, LPIPS) are minimized by the conditional mean,
which for an ambiguous RGB-to-thermal mapping is a blurred average. An
adversarial critic penalizes outputs that do not look like real thermal
footage, and feature matching on the critic's activations keeps the generator
anchored to the paired target. Wang et al., "High-Resolution Image Synthesis
and Semantic Manipulation with Conditional GANs", CVPR 2018.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


class PatchDiscriminator(nn.Module):
    """70x70-receptive-field PatchGAN with spectral norm; returns every
    intermediate activation so the generator can match features."""

    def __init__(self, in_channels=4, width=64):
        super().__init__()
        widths = (width, width * 2, width * 4, width * 8)
        layers, previous = [], in_channels
        for i, channels in enumerate(widths):
            stride = 1 if i == len(widths) - 1 else 2
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(previous, channels, 4, stride, 1)),
                nn.LeakyReLU(.2, inplace=True)))
            previous = channels
        layers.append(spectral_norm(nn.Conv2d(previous, 1, 4, 1, 1)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):
    """Critics at full and half resolution, conditioned on the RGB input."""

    def __init__(self, scales=2, width=64):
        super().__init__()
        self.critics = nn.ModuleList(PatchDiscriminator(4, width) for _ in range(scales))

    def forward(self, rgb, thermal):
        x = torch.cat([rgb * 2 - 1, thermal * 2 - 1], dim=1)
        outputs = []
        for i, critic in enumerate(self.critics):
            if i:
                x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
            outputs.append(critic(x))
        return outputs


def discriminator_loss(real, fake):
    """Hinge loss over every scale's final logits."""
    return sum(F.relu(1 - r[-1]).mean() + F.relu(1 + f[-1]).mean()
               for r, f in zip(real, fake)) / len(real)


def generator_adversarial_loss(fake):
    return sum(-f[-1].mean() for f in fake) / len(fake)


def feature_matching_loss(real, fake):
    """L1 between critic activations on real and generated thermal."""
    total, count = 0., 0
    for r, f in zip(real, fake):
        for real_feature, fake_feature in zip(r[:-1], f[:-1]):
            total = total + F.l1_loss(fake_feature, real_feature.detach())
            count += 1
    return total / count
