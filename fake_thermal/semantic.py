"""Appearance-based heat priors and luminance-guided upsampling, not thermometry."""
import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from fake_thermal.device import select_device


CLASS_HEAT = {
    'person': .9,
    'cat': .85, 'dog': .85, 'horse': .85, 'sheep': .85, 'cow': .85, 'bird': .85,
    'car': .6, 'bus': .6, 'motorbike': .6, 'train': .6,
}
VEHICLES = ('car', 'bus', 'motorbike', 'train')


def probability_prior(probabilities, categories, vehicles=True):
    confidence = np.zeros(probabilities.shape[1:], dtype=np.float32)
    weighted = np.zeros_like(confidence)
    for name, heat in CLASS_HEAT.items():
        if not vehicles and name in VEHICLES:
            continue
        probability = probabilities[categories.index(name)]
        confidence += probability
        weighted += heat * probability
    prior = np.divide(weighted, confidence, out=np.zeros_like(weighted), where=confidence > 0)
    return prior, np.clip(confidence, 0, 1)


def emissive_prior(rgb):
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    orange = (np.clip((55 - hue) / 5, 0, 1) *
              np.clip((saturation - .4) / .3, 0, 1) * np.clip((value - .75) / .2, 0, 1))
    white = np.clip((.15 - saturation) / .1, 0, 1) * np.clip((value - .94) / .05, 0, 1)
    confidence = cv2.GaussianBlur(np.maximum(orange, white), (5, 5), 1)
    weighted = cv2.GaussianBlur(orange + .8 * white, (5, 5), 1)
    total = cv2.GaussianBlur(orange + white, (5, 5), 1)
    prior = np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)
    return prior, confidence


def fuse_heat(model, prior, confidence, strength=.7):
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Semantic strength must be between 0 and 1')
    return model + strength * np.clip(confidence, 0, 1) * np.maximum(prior - model, 0)


class SemanticPrior:
    def __init__(self, size=320, device='auto', emissive=True, vehicles=True):
        """``emissive=False`` drops the colour heuristic for flames and lights;
        ``vehicles=False`` drops vehicle classes (parked cars read cold in real
        thermal), leaving people and animals."""
        self.size, self.emissive, self.vehicles = size, emissive, vehicles
        self.device = select_device(device)
        self.weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = None

    @torch.inference_mode()
    def __call__(self, rgb, output_shape):
        if self.model is None:
            self.model = deeplabv3_mobilenet_v3_large(weights=self.weights).eval().to(
                device=self.device, dtype=torch.float32)
        h, w = rgb.shape[:2]
        size = (round(w * self.size / min(h, w)), round(h * self.size / min(h, w)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        x = x.to(self.device)
        x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor(
            [.229, .224, .225])[None, :, None, None]
        probabilities = self.model(x)['out'][0].softmax(0).cpu().numpy()
        prior, confidence = probability_prior(probabilities, self.weights.meta['categories'],
                                              self.vehicles)
        emissive, emissive_confidence = emissive_prior(small)
        if not self.emissive:
            emissive_confidence = np.zeros_like(emissive_confidence)
        total = confidence + emissive_confidence
        prior = np.divide(prior * confidence + emissive * emissive_confidence, total,
                          out=np.zeros_like(prior), where=total > 0)
        confidence = np.maximum(confidence, emissive_confidence)
        target = (output_shape[1], output_shape[0])
        return (cv2.resize(prior, target, interpolation=cv2.INTER_LINEAR),
                cv2.resize(confidence, target, interpolation=cv2.INTER_LINEAR))


def guided_upsample(heat, rgb, radius=4, epsilon=.001):
    """Fast guided filter: fit at heat resolution, reconstruct with full-size luminance."""
    if (heat.ndim != 2 or not heat.size or rgb.ndim != 3 or rgb.shape[2] != 3 or
            not rgb.size or radius < 1 or epsilon <= 0):
        raise ValueError('Expected nonempty heat HxW, RGB HxWx3, positive radius and epsilon')
    guide = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(guide, (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_AREA)
    heat = heat.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)

    def mean(x):
        return cv2.boxFilter(x, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mean_guide, mean_heat = mean(small), mean(heat)
    covariance = mean(small * heat) - mean_guide * mean_heat
    variance = np.maximum(mean(small * small) - mean_guide * mean_guide, 0)
    a = covariance / (variance + epsilon)
    b = mean_heat - a * mean_guide
    size = (rgb.shape[1], rgb.shape[0])
    a = cv2.resize(mean(a), size, interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(mean(b), size, interpolation=cv2.INTER_LINEAR)
    return np.clip(a * guide + b, 0, 1)
