import ctypes
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fake_thermal.infer import PALETTES, apply_palette


DCTL = Path(__file__).resolve().parents[1] / 'resolve' / 'Fake Thermal.dctl'

SHIM = r"""
#include <math.h>
#define __DEVICE__
#define __CONSTANT__ const
#define DEFINE_UI_PARAMS(name, label, kind, default, ...) float name = default;
#define _fminf fminf
#define _fmaxf fmaxf
typedef struct { float x, y, z; } float3;
float3 make_float3(float x, float y, float z) { return (float3){x, y, z}; }
#include "Fake Thermal.dctl"
void run(int p, float c, float l, float a, const float *rgb, float *out) {
    palette = p; contrast = c; level = l; amount = a;
    float3 result = transform(1, 1, 0, 0, rgb[0], rgb[1], rgb[2]);
    out[0] = result.x; out[1] = result.y; out[2] = result.z;
}
"""


@unittest.skipUnless(shutil.which('cc'), 'C compiler required to test DCTL math')
class TestResolveDctl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build = tempfile.TemporaryDirectory()
        cls.addClassCleanup(build.cleanup)
        library = Path(build.name) / 'thermal.so'
        source = re.sub(r'DEFINE_UI_PARAMS\((\w+), \w+, DCTLUI_COMBO_BOX, (\d+), ({[^}]+}), {[^}]+}\)',
                        r'int \1 = \2; enum \3;', DCTL.read_text(encoding='utf-8'))
        (Path(build.name) / DCTL.name).write_text(source, encoding='utf-8')
        subprocess.run(['cc', '-shared', '-fPIC', '-I', build.name, '-o', str(library),
                        '-x', 'c', '-', '-lm'], input=SHIM, text=True, check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.library.run.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        cls.library.run.restype = None

    def transform(self, rgb, palette=0, contrast=1, level=0, amount=1):
        pixel = ctypes.c_float * 3
        out = pixel()
        self.library.run(palette, contrast, level, amount, pixel(*rgb), out)
        return list(out)

    def test_palettes_match_cli_within_half_an_eight_bit_code(self):
        heat = np.linspace(0, 1, 1001, dtype=np.float32)
        names = ('ironbow', 'white-hot', 'black-hot', 'rainbow', 'arctic', 'lava', 'glowbow')
        self.assertEqual(set(names), set(PALETTES))
        for choice, name in enumerate(names):
            with self.subTest(palette=name):
                dctl = np.array([self.transform((h, h, h), choice) for h in heat])
                np.testing.assert_allclose(dctl * 255, apply_palette(heat, name),
                                           atol=0.5001, rtol=0)

    def test_luma_controls_and_blend(self):
        rgb = (0.2, 0.4, 0.8)
        heat = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        expected = (heat - 0.5) * 2 + 0.5 + 0.1
        np.testing.assert_allclose(self.transform(rgb, 1, contrast=2, level=0.1), [expected] * 3)
        np.testing.assert_allclose(self.transform(rgb, amount=0), rgb)
        np.testing.assert_allclose(self.transform(rgb, 1, level=0.5, amount=0.25),
                                   np.array(rgb) * 0.75 + (heat + 0.5) * 0.25)
        self.assertEqual(self.transform((0, 0, 0), 1, contrast=4), [0] * 3)
        self.assertEqual(self.transform((1, 1, 1), 1, contrast=4), [1] * 3)
