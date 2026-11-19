"""Microbolometer sensor simulation: the fingerprint real thermal carries.

Real LWIR cameras add persistent column banding (readout integrated circuit),
fixed-pattern response variation and per-frame read noise to an optically
soft image. Regression models trained on such footage predict the scene
content but average the noise away, which reads as synthetic smoothness.
This module re-adds the measured sensor signature after prediction.
"""
import cv2
import numpy as np


class ThermalSensor:
    """Persistent per-camera sensor noise plus per-frame read noise.

    The fixed-pattern components (columns, pixels) are generated once per
    instance so every frame of a video carries the same camera fingerprint;
    only the read noise flickers. Parameter defaults were calibrated so the
    simulated high-frequency variance matches pooled real footage
    (scripts/evaluate_realism.py).
    """

    def __init__(self, noise=.015, columns=.01, pattern=.016, blur=.6,
                 vignette=.09, seed=42):
        if min(noise, columns, pattern) < 0 or blur < 0:
            raise ValueError('Sensor parameters must be nonnegative')
        if not 0 <= vignette <= 1:
            raise ValueError('Vignette must be in [0, 1]')
        self.noise, self.columns, self.pattern = noise, columns, pattern
        self.blur, self.vignette = blur, vignette
        self.rng = np.random.default_rng(seed)
        self.fixed = None

    def update(self, heat):
        if heat.ndim != 2 or not heat.size:
            raise ValueError('Sensor input must be a single-channel heat map')
        h, w = heat.shape
        if self.fixed is None or self.fixed.shape != heat.shape:
            # column banding that drifts with height: straight uniform stripes
            # read as an overlay, so build the pattern from a coarse grid
            coarse = self.rng.normal(0, self.columns, (16, w)).astype(np.float32)
            column = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            pixel = self.rng.normal(0, self.pattern, (h, w)).astype(np.float32)
            self.fixed = column + pixel
        result = cv2.GaussianBlur(heat, (0, 0), self.blur) if self.blur > 0 else heat.astype(np.float32)
        result = result + self.fixed + self.rng.normal(
            0, self.noise, heat.shape).astype(np.float32)
        if self.vignette:
            y, x = np.mgrid[0:h, 0:w]
            radius = np.hypot((x - (w - 1) / 2) / ((w - 1) / 2),
                              (y - (h - 1) / 2) / ((h - 1) / 2))
            result = result * (1 - self.vignette * radius ** 2)
        return np.clip(result, 0, 1).astype(np.float32)
