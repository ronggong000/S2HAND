import unittest

import cv2
import numpy as np

from s2hand.inference import (
    HEATMAP_SIZE,
    INPUT_SIZE,
    bbox_center_scale,
    decode_udp,
    prepare_person,
    udp_warp_matrix,
)


class InferenceProtocolTest(unittest.TestCase):
    def test_bbox_padding_and_aspect(self):
        center, scale = bbox_center_scale([10, 20, 110, 220])
        np.testing.assert_allclose(center, [60, 120])
        np.testing.assert_allclose(scale, [187.5, 250.0])
        self.assertAlmostEqual(scale[0] / scale[1], INPUT_SIZE[0] / INPUT_SIZE[1])

    def test_udp_crop_maps_center(self):
        center, scale = bbox_center_scale([20, 40, 180, 280])
        matrix = udp_warp_matrix(center, scale)
        mapped = cv2.transform(center.reshape(1, 1, 2), matrix)[0, 0]
        np.testing.assert_allclose(mapped, [(INPUT_SIZE[0] - 1) / 2,
                                            (INPUT_SIZE[1] - 1) / 2], atol=1e-4)

    def test_prepare_shape_and_normalization(self):
        image = np.zeros((100, 80, 3), np.uint8)
        tensor, metadata = prepare_person(image, [0, 0, 79, 99])
        self.assertEqual(tuple(tensor.shape), (3, 1024, 768))
        self.assertEqual(metadata["bbox_center"].shape, (2,))
        self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_decode_center_peak_roundtrip(self):
        heatmaps = np.zeros((308, HEATMAP_SIZE[1], HEATMAP_SIZE[0]), np.float32)
        heatmaps[:, HEATMAP_SIZE[1] // 2, HEATMAP_SIZE[0] // 2] = 1.0
        center, scale = bbox_center_scale([40, 20, 240, 420])
        points, scores = decode_udp(
            heatmaps, {"bbox_center": center, "bbox_scale": scale}
        )
        self.assertEqual(points.shape, (308, 2))
        self.assertEqual(scores.shape, (308,))
        np.testing.assert_allclose(scores, 1.0)
        np.testing.assert_allclose(points.mean(axis=0), center, atol=2.0)


if __name__ == "__main__":
    unittest.main()
