"""Appearance-based heat priors and luminance-guided upsampling, not thermometry."""
from pathlib import Path
import urllib.request
import warnings

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

# Typical relative heat of body parts in a thermal frame: exposed skin is the
# hottest thing on a person, clothing reads warm but cooler, hair and
# accessories (masks, caps, glasses) cooler still. Order follows the MediaPipe
# multiclass selfie segmenter: background, hair, body skin, face skin,
# clothes, others.
PART_HEAT = np.float32([0., .55, .92, .95, .62, .5])
PARTS_MODEL = ('https://storage.googleapis.com/mediapipe-models/image_segmenter/'
               'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite')


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


def part_prior(masks):
    """HxWx6 part confidences to (heat target, person confidence)."""
    parts = masks[..., 1:]
    total = parts.sum(-1)
    weighted = (parts * PART_HEAT[1:]).sum(-1)
    target = np.divide(weighted, total, out=np.full_like(weighted, CLASS_HEAT['person']), where=total > 1e-3)
    return target.astype(np.float32), np.clip(1 - masks[..., 0], 0, 1).astype(np.float32)


class PartSegmenter:
    """Face, skin, hair and clothing masks for people large enough to resolve.

    Runs Google's MediaPipe multiclass selfie segmenter (Apache 2.0 model card)
    on a padded crop around each sizeable person found by the class prior.
    Needs the optional ``mediapipe`` package; the model (~16 MB) is downloaded
    to the Torch hub cache on first use."""

    def __init__(self, min_height=.12, pad=.15, min_agreement=.4, min_overlap=.3, band=.06):
        self.min_height, self.pad, self.min_agreement = min_height, pad, min_agreement
        self.min_overlap, self.band = min_overlap, band
        self.segmenter = None

    def _load(self):
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision
        path = Path(torch.hub.get_dir()) / 'fake-thermal' / Path(PARTS_MODEL).name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            urllib.request.urlretrieve(PARTS_MODEL, temporary)
            temporary.replace(path)
        options = vision.ImageSegmenterOptions(base_options=tasks.BaseOptions(model_asset_path=str(path)),
                                               output_confidence_masks=True, output_category_mask=False)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment(self, rgb):
        import mediapipe as mp
        result = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=np.ascontiguousarray(rgb)))
        return np.stack([np.squeeze(m.numpy_view()) for m in result.confidence_masks], -1)

    def __call__(self, rgb, person, shape):
        """Part heat target, person confidence and coverage weight at ``shape``
        (h, w), or None where no person was resolved. ``person`` is the class
        prior's person probability at ``shape``."""
        if self.segmenter is None:
            self._load()
        h, w = shape
        scale_y, scale_x = rgb.shape[0] / h, rgb.shape[1] / w
        count, labels, stats, _ = cv2.connectedComponentsWithStats((person > .5).astype(np.uint8))
        target = np.zeros(shape, np.float32)
        confidence = np.zeros(shape, np.float32)
        cover = np.zeros(shape, np.float32)
        for index in range(1, count):
            x, y, bw, bh = stats[index, :4]
            if bh < self.min_height * h:
                continue
            px, py = round(bw * self.pad), round(bh * self.pad)
            x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(w, x + bw + px), min(h, y + bh + py)
            crop = rgb[round(y0 * scale_y):round(y1 * scale_y), round(x0 * scale_x):round(x1 * scale_x)]
            if min(crop.shape[:2]) < 16:
                continue
            masks = cv2.resize(self.segment(crop), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            part, seen = part_prior(masks)
            inside = labels[y0:y1, x0:x1] == index
            found = seen > .5
            overlap = (found & inside).sum() / max(1, (found | inside).sum())
            if float(seen[inside].mean()) < self.min_agreement or overlap < self.min_overlap:
                # The part model missed this person, or painted the crop around them
                # (it is a selfie model; small night-time figures confuse it): keep the
                # class prior there.
                continue
            # Only a band around the person, fading to zero at its edge: confidence the
            # part model leaks over the crop background would otherwise show the crop
            # as a warm rectangle.
            r = max(2, round(self.band * bh))
            size = (2 * r + 1, 2 * r + 1)
            near = cv2.dilate(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size))
            fade = np.clip(2 * cv2.GaussianBlur(near.astype(np.float32), size, 0) - 1, 0, 1)
            use = (fade > 0) & (seen >= confidence[y0:y1, x0:x1])
            target[y0:y1, x0:x1] = np.where(use, part, target[y0:y1, x0:x1])
            confidence[y0:y1, x0:x1] = np.where(use, seen, confidence[y0:y1, x0:x1])
            cover[y0:y1, x0:x1] = np.maximum(cover[y0:y1, x0:x1], fade)
        if not cover.any():
            return None
        return target, confidence, cover


def fuse_heat(model, prior, confidence, strength=.7):
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Semantic strength must be between 0 and 1')
    return model + strength * np.clip(confidence, 0, 1) * np.maximum(prior - model, 0)


class SemanticPrior:
    def __init__(self, size=320, device='auto', emissive=True, vehicles=True, sharp=False, parts=False):
        """``emissive=False`` drops the colour heuristic for flames and lights;
        ``vehicles=False`` drops vehicle classes (parked cars read cold in real
        thermal), leaving people and animals. ``parts=True`` replaces the
        uniform person heat with skin, clothing and hair levels wherever a
        person is large enough to segment (needs ``mediapipe``); distant
        people keep the uniform level."""
        self.size, self.emissive, self.vehicles, self.sharp = size, emissive, vehicles, sharp
        self.parts = None
        if parts:
            try:
                import mediapipe  # noqa: F401
                self.parts = PartSegmenter()
            except ImportError:
                warnings.warn('Part-aware people heat needs: pip install mediapipe; using uniform people heat')
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
        if self.sharp:
            # The class map comes from a 320px segmentation; stretched bilinearly it
            # gives people a soft, flame-like glow. Re-fit its edges to the photo.
            guide = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
            prior = cv2.resize(prior, target, interpolation=cv2.INTER_LINEAR)
            confidence = guided_upsample(confidence, guide, radius=2, epsilon=1e-3)
        else:
            prior = cv2.resize(prior, target, interpolation=cv2.INTER_LINEAR)
            confidence = cv2.resize(confidence, target, interpolation=cv2.INTER_LINEAR)
        if self.parts is not None:
            person = cv2.resize(probabilities[self.weights.meta['categories'].index('person')], target,
                                interpolation=cv2.INTER_LINEAR)
            resolved = self.parts(rgb, person, output_shape)
            if resolved is not None:
                part, seen, cover = resolved
                prior = cover * part + (1 - cover) * prior
                confidence = cover * seen + (1 - cover) * confidence
        return prior, confidence


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
