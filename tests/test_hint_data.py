import unittest

import numpy as np

from s2hand.hint_data import udp_gaussian_targets


class HIntDataTest(unittest.TestCase):
    def test_udp_targets_keep_layout_and_visibility(self):
        points = np.zeros((42, 2), np.float32)
        visibility = np.zeros(42, np.float32)
        points[3] = (384, 512)
        visibility[3] = 1
        heatmaps, weights = udp_gaussian_targets(points, visibility)
        self.assertEqual(heatmaps.shape, (42, 256, 192))
        self.assertEqual(weights.shape, (42,))
        self.assertEqual(float(heatmaps[np.arange(42) != 3].sum()), 0.0)
        self.assertGreater(float(heatmaps[3].max()), 0.99)
        self.assertEqual(float(weights.sum()), 1.0)

    def test_udp_targets_drop_fully_outside_point(self):
        points = np.zeros((42, 2), np.float32)
        visibility = np.zeros(42, np.float32)
        points[7] = (100000, 100000)
        visibility[7] = 1
        heatmaps, weights = udp_gaussian_targets(points, visibility)
        self.assertEqual(float(heatmaps.sum()), 0.0)
        self.assertEqual(float(weights.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
