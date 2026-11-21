"""Photo noise is not heat texture.

Night and low-light photos carry sensor noise, and the turbo translator turns
it into a sandy speckle all over the thermal frame, the commonest giveaway of
its night output. A real thermal frame of the same scene has smooth sky and
road. Denoising the photo first, by an amount set from its measured noise,
removes the speckle. Clean photos are left as they are; the threshold sits
above what fine daylight texture such as sand or gravel reads as (about 1.4).
"""
import cv2
import numpy as np


def _picture(gray):
    """Pixels inside the picture, away from black letterbox borders."""
    return cv2.erode((gray > 8).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def noise_sigma(rgb):
    """Noise level of an HxWx3 uint8 photo in 8-bit units (Immerkaer's estimator
    on the grey image), ignoring black letterbox borders."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    response = np.abs(cv2.filter2D(gray, -1, kernel))
    inside = _picture(gray)
    if inside.sum() < 64:
        return 0.
    return float(np.sqrt(np.pi / 2) * response[inside].mean() / 6)


def denoise_photo(rgb, gain=2.5, least=4., most=15.):
    """Non-local-means denoise with filter strength ``gain`` x the measured noise,
    capped at ``most``; photos whose strength would fall below ``least`` (noise
    under 1.6) are returned unchanged."""
    strength = gain * noise_sigma(rgb)
    if strength < least:
        return rgb
    strength = min(strength, most)
    return cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(rgb), None, strength, strength, 7, 21)
