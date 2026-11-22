"""Visible-only counterfactuals for paired turbo training and evaluation."""
import random

import cv2
import numpy as np

from fake_thermal.augment import add_cast, add_lights, add_print


SYNTHETIC_EDITS = ('glare', 'print')
MASKED_EDITS = ('glass', 'clothing')


def edit_appearance(rgb, kind, mask=None, *, rng=random, np_rng=np.random):
    """Edit float RGB in [0, 1] and return RGB plus a soft influence map.

    Glass and clothing require an aligned material mask in [0, 1]. They change the
    visibility of detail or the fabric's apparent brightness, not the object.
    """
    if kind == 'glare':
        edited = rng.choice((add_cast, add_lights))(rgb, rng=rng, np_rng=np_rng)
    elif kind == 'print':
        edited = add_print(rgb, rng=rng, np_rng=np_rng)
    elif kind in MASKED_EDITS:
        if mask is None or mask.shape != rgb.shape[:2]:
            raise ValueError(f'{kind} needs an aligned appearance mask')
        if kind == 'glass':
            # Normalize the blur inside the mask so pillars and trim do not
            # bleed into the windshield.
            sigma = max(1., min(rgb.shape[:2]) / 96)
            blurred = cv2.GaussianBlur(rgb * mask[..., None], (0, 0), sigma)
            coverage = cv2.GaussianBlur(mask, (0, 0), sigma)
            local = (rgb * mask[..., None]).sum(axis=(0, 1)) / max(float(mask.sum()), 1e-6)
            altered = .5 * (local + blurred / coverage[..., None].clip(min=1e-6))
        else:
            altered = .45 * rgb if rng.random() < .5 else .35 + .65 * rgb
        edited = rgb + mask[..., None] * (altered - rgb)
    else:
        raise ValueError(f'Unknown appearance edit: {kind}')

    edited = np.clip(edited, 0, 1).astype(np.float32)
    change = np.abs(edited - rgb).max(axis=2)
    # Ignore sub-byte rounding; include a narrow halo for a shifted thermal edge.
    influence = np.clip((change - 1 / 255) / .12, 0, 1)
    radius = max(1, round(min(rgb.shape[:2]) / 128))
    influence = cv2.dilate(influence, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    return edited, influence
