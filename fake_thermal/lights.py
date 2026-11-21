"""Light sources in the photo are not heat sources in the thermal frame.

A street lamp, headlight, traffic signal or ceiling LED is the brightest thing
in its part of the photo, and the translator has two bad habits with it: it
copies the glow as heat, or it overcorrects and paints a cold disc with a warm
ring where the lamp was. A thermal camera sees neither: a lamp head is a small
object at about the temperature of its housing. Both habits are fixed the same
way: find small regions much brighter than their surroundings, and inside the
glow each one casts in the photo, pull the heat back into the range of the
heat around it. Ordinary grain passes through; only the disc or the glow,
which stand far outside that range, are changed.

Only light that stands out from a darker surround is touched, so sunlit white
shirts, skies and chrome in daylight are left alone; pixels on people can be
excluded as well.
"""
import cv2
import numpy as np


def find_lights(rgb, shape, level=.9, ratio=2., lift=.45, max_area=.0015, glow=.25, spread=12.):
    """Binary uint8 mask (``shape``) over each light source in ``rgb`` and its glow, or None.

    A light is a connected group of pixels whose brightest channel is at least
    ``level``, no larger than ``max_area`` of the frame, and at least ``ratio``
    times and ``lift`` above the brighter quarter of a ring around it (the ring
    starts at three times the light's radius, beyond most bloom), so it must be
    dark nearly all the way round: a sunlit patch beside a shadow is not a
    light. Its mask is the glow around it (the connected pixels still ``glow``
    of the way from the surround up to the light) and a disc of three times
    its radius, plus a two-pixel margin. A
    glow more than ``spread`` times the light's own area is a lit surface, not a
    lamp, and is skipped.
    """
    h, w = shape
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    value = small.max(axis=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((value >= level).astype(np.uint8), connectivity=8)
    mask = np.zeros((h, w), np.uint8)
    limit = max_area * h * w
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if area > limit:
            continue
        radius = max(1., np.sqrt(area / np.pi))
        outer = int(round(6 * radius + 6))
        x0, y0 = max(0, x - outer), max(0, y - outer)
        x1, y1 = min(w, x + bw + outer), min(h, y + bh + outer)
        piece = (labels[y0:y1, x0:x1] == index).astype(np.uint8)
        distance = cv2.distanceTransform(1 - piece, cv2.DIST_L2, 5)
        ring = (distance > 3 * radius + 2) & (distance <= 5 * radius + 4)
        if ring.sum() < 8:
            continue
        local = value[y0:y1, x0:x1]
        surround = float(np.percentile(local[ring], 75))
        bright = float(local[piece > 0].mean())
        if bright < ratio * max(surround, .02) or bright - surround < lift:
            continue
        halo = (local >= surround + glow * (bright - surround)) & (distance <= 5 * radius + 4)
        _, halo_labels = cv2.connectedComponents(halo.astype(np.uint8), connectivity=8)
        keep = np.unique(halo_labels[piece > 0])
        region = np.isin(halo_labels, keep[keep > 0]) | (piece > 0)
        if region.sum() > spread * max(area, 6):  # tiny lamps blur over their neighbours
            continue
        # the translator's cold disc or hot bloom spreads past the glow in the
        # photo, to about three times the light's radius
        region |= distance <= 3 * radius + 2
        region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8))
        mask[y0:y1, x0:x1] |= region
    return mask if mask.any() else None


def neutralize_lights(heat, rgb, exclude=None, tolerance=1., **options):
    """Pull the heat over light sources back into the range of the heat around them.

    ``heat`` is the translator's HxW output in [0, 1]; ``rgb`` the photo at any
    size; ``exclude`` an optional HxW weight (for example the person confidence
    of the semantic prior) above .5 where nothing is changed. Inside each glow
    the heat keeps its grain but may not stray more than ``tolerance`` standard
    deviations of the surrounding grain from the smoothly interpolated
    surroundings.
    """
    mask = find_lights(rgb, heat.shape, **options)
    if mask is None:
        return heat
    if exclude is not None:
        mask = mask & (exclude < .5).astype(np.uint8)
        if not mask.any():
            return heat
    heat = heat.astype(np.float32)
    keep = 1 - mask.astype(np.float32)
    # smooth interpolation of the surroundings into the holes
    base = heat.copy()
    todo = mask > 0
    for sigma in (3, 8, 24, 64):
        weight = cv2.GaussianBlur(keep, (0, 0), sigma)
        blurred = cv2.GaussianBlur(heat * keep, (0, 0), sigma)
        ready = todo & (weight > .05)
        base[ready] = blurred[ready] / weight[ready]
        todo &= ~ready
        if not todo.any():
            break
    base[todo] = float(heat[keep > 0].mean()) if keep.any() else float(heat.mean())
    # how much the heat normally varies around the lights
    grain = heat - cv2.GaussianBlur(heat, (0, 0), 2)
    near = (cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0) & (mask == 0)
    sigma = float(np.std(grain[near])) if near.sum() > 16 else float(np.std(grain))
    limit = tolerance * max(sigma, .01)
    fixed = base + np.clip(heat - base, -limit, limit)
    soft = np.maximum(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.), mask)
    return np.clip(heat * (1 - soft) + fixed * soft, 0, 1).astype(np.float32)
