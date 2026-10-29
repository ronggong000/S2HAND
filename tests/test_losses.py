import unittest

import torch

from s2hand.losses import (
    HAND_CHANNELS, HAND_EDGES, hand_geometry_losses,
    hand_heatmaps_from_output, s2hand_objective,
)


class S2HandLossTest(unittest.TestCase):
    def test_hand_graph_has_twenty_edges(self):
        self.assertEqual(len(HAND_EDGES), 20)
        self.assertEqual(len(set(HAND_EDGES)), 20)

    def test_sapiens_channels_are_reordered_to_anatomical_order(self):
        output = torch.arange(308.0).view(1, 308, 1, 1)
        selected = hand_heatmaps_from_output(output).flatten().tolist()
        self.assertEqual(selected, [float(index) for index in HAND_CHANNELS])
        self.assertNotEqual(tuple(HAND_CHANNELS), tuple(range(21, 63)))

    def test_identical_softargmax_target_is_zero_geometry(self):
        torch.manual_seed(3)
        heatmaps = torch.randn(1, 42, 8, 6, requires_grad=True)
        from s2hand.losses import softargmax_hand_uv
        target = softargmax_hand_uv(heatmaps.detach())
        visible = torch.ones(1, 2, 21)
        coordinate, topology, _ = hand_geometry_losses(heatmaps, target, visible)
        self.assertLess(float(coordinate), 1e-10)
        self.assertLess(float(topology), 1e-10)

    def test_objective_is_finite_and_differentiable(self):
        torch.manual_seed(9)
        prediction = torch.randn(2, 42, 8, 6, requires_grad=True)
        target_heatmaps = torch.randn_like(prediction)
        target_weights = torch.ones(2, 42)
        gt_uv = torch.rand(2, 2, 21, 2) * torch.tensor([768.0, 1024.0])
        visibility = torch.ones(2, 2, 21)
        visibility[0, 0, 5:9] = 0
        loss, terms = s2hand_objective(
            prediction, target_heatmaps, target_weights, gt_uv, visibility
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(terms["loss_heatmap"]), 0.0)


if __name__ == "__main__":
    unittest.main()
