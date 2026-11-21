import unittest

import numpy as np

from fake_thermal.sensor import ThermalSensor


class SensorTests(unittest.TestCase):
    def test_fixed_pattern_persists_across_frames(self):
        sensor = ThermalSensor(noise=0, seed=3)
        first = sensor.update(np.zeros((64, 64), np.float32))
        second = sensor.update(np.zeros((64, 64), np.float32))
        np.testing.assert_array_equal(first, second)
        self.assertGreater(first.std(), 0)

    def test_read_noise_flickers_between_frames(self):
        sensor = ThermalSensor(columns=0, pattern=0, blur=0, vignette=0, seed=3)
        first = sensor.update(np.zeros((64, 64), np.float32))
        second = sensor.update(np.zeros((64, 64), np.float32))
        self.assertFalse(np.array_equal(first, second))
        self.assertLess(first.std(), .2)

    def test_output_stays_in_range_and_keeps_shape(self):
        sensor = ThermalSensor(seed=3)
        heat = np.full((37, 51), .5, np.float32)
        result = sensor.update(heat)
        self.assertEqual(result.shape, heat.shape)
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_identity_with_all_parameters_off(self):
        sensor = ThermalSensor(noise=0, columns=0, pattern=0, blur=0, vignette=0)
        heat = np.random.rand(64, 64).astype(np.float32)
        np.testing.assert_allclose(sensor.update(heat), heat, atol=1e-6)

    def test_rejects_bad_configuration(self):
        for kwargs in ({'vignette': 2}, {'noise': -1}, {'blur': -.5}):
            with self.assertRaises(ValueError):
                ThermalSensor(**kwargs)
        with self.assertRaises(ValueError):
            ThermalSensor().update(np.zeros((4, 4, 3), np.float32))


if __name__ == '__main__':
    unittest.main()
