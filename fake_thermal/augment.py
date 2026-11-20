"""Photo-only augmentations that teach a translator that visible light is not heat.

Each function changes the RGB side of a training pair and leaves the thermal
target alone. The commonest giveaways of generated thermal are visible-light
detail copied into the heat map: street lamps, headlights, lit windows and
light pools on the road glowing as if hot, and printed paint or logos showing
through. Pasting synthetic lights and print onto the photo while the target
stays as it was breaks that correlation. Night darkening covers footage where
the photo carries little information.

All functions take and return HxWx3 float32 arrays in [0, 1].
"""
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LIGHT_COLOURS = [(1., .94, .82), (1., .67, .27), (.86, .92, 1.), (1., .16, .12),
                 (.24, 1., .47), (1., .75, 0.), (1., 1., 1.)]
WINDOW_COLOURS = [(1., .84, .55), (1., .9, .7), (.82, .88, 1.), (1., .78, .45)]


def _screen(rgb, light):
    """Screen blend: light adds without ever passing white."""
    return 1 - (1 - rgb) * (1 - np.clip(light, 0, 1))


def _blob(shape, cy, cx, sy, sx, angle=0.):
    """Anisotropic Gaussian rendered only inside its +-4 sigma window."""
    h, w = shape
    reach = 4 * max(sy, sx)
    y0, y1 = max(0, int(cy - reach)), min(h, int(cy + reach) + 1)
    x0, x1 = max(0, int(cx - reach)), min(w, int(cx + reach) + 1)
    out = np.zeros(shape, np.float32)
    if y0 >= y1 or x0 >= x1:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    c, s = np.cos(angle), np.sin(angle)
    u, v = c * dx + s * dy, -s * dx + c * dy
    out[y0:y1, x0:x1] = np.exp(-.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return out


def _jitter(colour, amount=.08):
    return np.clip(np.array(colour, np.float32) + np.random.uniform(-amount, amount, 3), 0, 1)


def _point_light(shape, cy, cx, radius, colour):
    """Over-exposed core, a coloured halo and sometimes a lens streak."""
    core = _blob(shape, cy, cx, radius / 1.5, radius / 1.5)[..., None]
    halo_r = radius * random.uniform(3, 10)
    halo = _blob(shape, cy, cx, halo_r, halo_r)[..., None]
    white = .4 + .6 * np.array(colour, np.float32)
    light = core * random.uniform(1., 1.6) * white + halo * random.uniform(.12, .6) * np.array(colour, np.float32)
    if random.random() < .25:
        length = radius * random.uniform(8, 30)
        streak = _blob(shape, cy, cx, radius * .45, length, random.choice([0., np.pi / 2, random.uniform(0, np.pi)]))
        light = light + streak[..., None] * random.uniform(.15, .45) * white
    return light


def add_lights(rgb):
    """1-6 lamps, headlight pairs, rows of street lights, lit windows or light pools."""
    h, w = rgb.shape[:2]
    light = np.zeros_like(rgb)
    for _ in range(random.randint(1, 6)):
        kind = random.choices(['point', 'pair', 'row', 'windows', 'pool'], [4, 2, 1, 2, 1.5])[0]
        colour = _jitter(random.choice(LIGHT_COLOURS))
        cy, cx = random.uniform(0, h), random.uniform(0, w)
        radius = float(np.exp(random.uniform(np.log(1.5), np.log(10))))
        if kind == 'point':
            light += _point_light((h, w), cy, cx, radius, colour)
        elif kind == 'pair':  # headlights or tail lights
            gap = radius * random.uniform(4, 12)
            colour = _jitter(random.choice([LIGHT_COLOURS[0], LIGHT_COLOURS[2], LIGHT_COLOURS[3]]))
            for side in (-.5, .5):
                light += _point_light((h, w), cy, cx + side * gap, radius, colour)
        elif kind == 'row':  # street lights receding along a line
            dy, dx = random.uniform(-.3, .3), random.choice([-1, 1])
            step = random.uniform(30, 90)
            for i in range(random.randint(3, 6)):
                scale = .75 ** i
                light += _point_light((h, w), cy + i * step * dy * scale, cx + i * step * dx * scale,
                                      max(1.2, radius * scale), colour)
        elif kind == 'windows':
            light += _windows((h, w))
        else:  # light pool on the ground, mostly in the lower part of the frame
            cy = random.uniform(.45 * h, h)
            sx = random.uniform(30, 160)
            pool = _blob((h, w), cy, cx, sx * random.uniform(.2, .5), sx)
            light += pool[..., None] * random.uniform(.1, .35) * colour
    return _screen(rgb, light)


def _windows(shape):
    """A grid of lit windows on a facade, some dark, soft edged with a faint halo."""
    h, w = shape
    rows, cols = random.randint(1, 4), random.randint(1, 5)
    ww, wh = random.uniform(6, 40), random.uniform(8, 55)
    gx, gy = ww * random.uniform(1.3, 2.2), wh * random.uniform(1.3, 2.)
    top, left = random.uniform(-wh, h - wh), random.uniform(-ww, w - ww)
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for r in range(rows):
        for c in range(cols):
            if random.random() < .65:
                x, y = left + c * gx, top + r * gy
                draw.rectangle([x, y, x + ww, y + wh], fill=int(255 * random.uniform(.5, 1.)))
    m = np.asarray(mask, np.float32) / 255
    colour = _jitter(random.choice(WINDOW_COLOURS))
    if random.random() < .5:
        m = np.maximum(m, np.asarray(mask.filter(ImageFilter.GaussianBlur(ww * .6)), np.float32) / 255 * .35)
    return m[..., None] * random.uniform(.55, .95) * colour


def add_print(rgb):
    """Printed text, logos and road paint stripes, which a thermal camera does not see."""
    h, w = rgb.shape[:2]
    canvas = Image.fromarray((rgb * 255).astype(np.uint8))
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1, 4)):
        colour = tuple(int(c) for c in random.choice(
            [(255, 255, 255), (250, 220, 40), (20, 20, 20), (200, 30, 30), (30, 80, 200),
             tuple(np.random.randint(0, 256, 3))]))
        alpha = int(255 * random.uniform(.55, 1.))
        if random.random() < .55:
            text = ''.join(random.choices(string.ascii_uppercase + string.digits + '  ', k=random.randint(2, 10)))
            font = ImageFont.load_default(size=random.randint(12, 64))
            draw.text((random.uniform(-20, w - 20), random.uniform(-10, h - 10)), text,
                      fill=colour + (alpha,), font=font)
        else:
            x, y = random.uniform(0, w), random.uniform(h * .3, h)
            length, angle = random.uniform(40, 320), random.uniform(0, np.pi)
            dashes = random.random() < .5
            for i in range(6 if dashes else 1):
                start = i * length * 1.6 if dashes else 0
                x0, y0 = x + np.cos(angle) * start, y + np.sin(angle) * start
                draw.line([x0, y0, x0 + np.cos(angle) * length, y0 + np.sin(angle) * length],
                          fill=colour + (alpha,), width=random.randint(3, 18))
    canvas.paste(overlay, (0, 0), overlay)
    return np.asarray(canvas, np.float32) / 255


def darken(rgb):
    """Night exposure: crushed shadows, lower gain and sensor noise."""
    out = rgb ** random.uniform(1.3, 2.6) * random.uniform(.25, .7)
    out = out + np.random.normal(0, random.uniform(.003, .02), out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


def photo_only(rgb, p_light=0., p_print=0., p_dark=0.):
    """Apply the augmentations to a float RGB crop; lights are likelier after darkening."""
    dark = random.random() < p_dark
    if dark:
        rgb = darken(rgb)
    if random.random() < (min(1., p_light * 1.8) if dark else p_light):
        rgb = add_lights(rgb)
    if random.random() < p_print:
        rgb = add_print(rgb)
    return rgb.astype(np.float32)
