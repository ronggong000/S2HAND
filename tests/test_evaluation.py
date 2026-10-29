import unittest

import numpy as np

from s2hand.evaluation import hand2d_metrics


class Mapping(dict):
    @property
    def files(self):
        return list(self)


def pair():
    target = np.zeros((2, 21, 2), np.float32)
    prediction = target.copy()
    ground_truth = Mapping(
        schema_version=np.asarray("hand2d.v1"),
        sample_id=np.asarray(["a", "b"]),
        keypoints_xy=target,
        visibility=np.ones((2, 21), np.float32),
        hand_bbox_xyxy=np.asarray([[0, 0, 100, 80], [0, 0, 100, 80]], np.float32),
    )
    predicted = Mapping(
        schema_version=np.asarray("hand2d.v1"),
        sample_id=np.asarray(["a", "b"]),
        coordinate_system=np.asarray("original_image_pixels"),
        keypoints_xy=prediction,
        detected=np.asarray([True, False]),
    )
    return ground_truth, predicted


class EvaluationTest(unittest.TestCase):
    def test_missed_hand_is_primary_metric_failure(self):
        ground_truth, predicted = pair()
        row = hand2d_metrics(ground_truth, predicted)
        self.assertEqual(row["hands"], 2)
        self.assertEqual(row["evaluated_joints"], 21)
        self.assertAlmostEqual(row["detection_recall"], 0.5)
        self.assertAlmostEqual(row["pck_0.05"], 0.5)
        self.assertAlmostEqual(row["pck_0.10"], 0.5)
        self.assertAlmostEqual(row["auc_0_20"], 0.5, places=6)
        self.assertAlmostEqual(row["tail_hand_nme_gt_0.20"], 0.5)

    def test_rejects_order_mismatch(self):
        ground_truth, predicted = pair()
        predicted["sample_id"] = predicted["sample_id"][::-1]
        with self.assertRaises(ValueError):
            hand2d_metrics(ground_truth, predicted)


if __name__ == "__main__":
    unittest.main()
