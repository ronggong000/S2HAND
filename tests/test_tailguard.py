import json
import unittest
from pathlib import Path

import numpy as np

from s2hand.tailguard import (
    crop_to_original,
    detector_association_features,
    expanded_square_box,
    geometry_features,
    load_frozen_config,
    original_to_crop,
    should_trigger,
)


class TailGuardTests(unittest.TestCase):
    def test_expanded_square_box(self):
        box = expanded_square_box([10, 20, 30, 30], 2.0)
        np.testing.assert_allclose(box, [0, 5, 40, 45])

    def setUp(self):
        self.points = np.asarray(
            [[0, 0]]
            + [[finger * 10 + joint, joint * 8] for finger in range(5) for joint in range(1, 5)],
            np.float32,
        )

    def test_release_config(self):
        path = Path(__file__).parents[1] / "configs" / "tailguard_v1.json"
        config = load_frozen_config(path)
        self.assertEqual(config["rules"][0]["threshold"], 0.5107561469078065)
        self.assertTrue(should_trigger({"native_conf_mean": 0.4}, True, config))
        self.assertFalse(should_trigger({"native_conf_mean": 0.7}, True, config))
        self.assertFalse(should_trigger({"native_conf_mean": 0.4}, False, config))

    def test_geometry_translation_scale_invariance(self):
        before = geometry_features(self.points)
        after = geometry_features(self.points * 3.7 + np.asarray([123, -57]))
        for key in before:
            self.assertAlmostEqual(before[key], after[key], places=5)

    def test_detector_association(self):
        features = detector_association_features(self.points, [-10, -10, 60, 60])
        self.assertGreater(features["native_joint_in_detector_fraction"], 0.5)

    def test_coordinate_roundtrip(self):
        input_size = np.asarray([192, 256], np.float32)
        center = np.asarray([400, 300], np.float32)
        scale = np.asarray([240, 320], np.float32)
        encoded = original_to_crop(self.points, input_size, center, scale)
        decoded = crop_to_original(encoded, input_size, center, scale)
        np.testing.assert_allclose(decoded, self.points, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
