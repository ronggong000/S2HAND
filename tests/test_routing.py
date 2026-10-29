import json
import unittest
from pathlib import Path

import torch

from s2hand.loader import RoutedPoseHead, TRAINING_WARM_START


ROOT = Path(__file__).resolve().parents[1]


class RoutingTest(unittest.TestCase):
    def test_mt_protects_exactly_268_channels(self):
        torch.manual_seed(7)
        head = RoutedPoseHead(
            channels=4, deconv_channels=(4, 4), decoder_channels=(4, 4, 4)
        ).eval()
        head.freeze_public_route()
        features = torch.randn(1, 4, 2, 2)
        with torch.no_grad():
            baseline = head(features)
            for parameter in head.conv_layers.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
            for parameter in head.conv_pose.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
            adapted = head(features)
        protected = head.protected_mask
        self.assertEqual(int(protected.sum()), 268)
        self.assertTrue(bool(protected[41]))
        self.assertTrue(bool(protected[62]))
        self.assertTrue(torch.equal(baseline[:, protected], adapted[:, protected]))
        self.assertFalse(torch.equal(baseline[:, ~protected], adapted[:, ~protected]))

    def test_protected_route_does_not_backpropagate_to_shared_features(self):
        torch.manual_seed(11)
        head = RoutedPoseHead(
            channels=4, deconv_channels=(4, 4), decoder_channels=(4, 4, 4)
        ).eval()
        head.freeze_public_route()
        features = torch.randn(1, 4, 2, 2, requires_grad=True)
        output = head(features)
        output[:, head.protected_mask].sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features.grad)))

    def test_release_block_selections_match_manifests(self):
        release = json.loads((ROOT / "release_manifest.json").read_text())
        expected = {
            "S2HAND-MT-5B": [],
            "S2HAND-Pose-0.4B": [16, 17, 18],
            "S2HAND-Pose-0.8B": [17, 18, 19],
            "S2HAND-Pose-1B": [28, 29, 30],
            "S2HAND-Pose-5B": [38, 39, 40],
        }
        actual = {
            row["name"]: row["transformer_blocks_zero_based"]
            for row in release["models"]
        }
        self.assertEqual(actual, expected)
        for name, blocks in expected.items():
            manifest = json.loads((ROOT / "manifests" / f"{name}.json").read_text())
            tensor_names = manifest["tensor_names"]
            observed = sorted({
                int(name.split(".")[2])
                for name in tensor_names if name.startswith("backbone.blocks.")
            })
            self.assertEqual(observed, blocks)

    def test_training_warm_start_is_not_a_deployment_model(self):
        manifest = json.loads(
            (ROOT / "manifests" / f"{TRAINING_WARM_START}.json").read_text()
        )
        self.assertEqual(manifest["name"], TRAINING_WARM_START)
        self.assertIs(manifest["deployment_model"], False)
        self.assertEqual(
            manifest["role"], "training_and_fisher_initialization_only"
        )


if __name__ == "__main__":
    unittest.main()
