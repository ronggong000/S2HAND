import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from s2hand.training import (
    ValidationSelector, checkpoint_tensors, configure_trainable,
    hint_metrics, optimizer_for, save_checkpoint,
)


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(2, 2, 1))
        self.conv_pose = nn.Conv2d(2, 308, 1)
        self.deconv_layers = nn.Conv2d(2, 2, 1)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(5)])
        self.rope_embed = nn.Module()
        self.rope_embed.register_buffer("periods", torch.ones(2, dtype=torch.bfloat16))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.decode_head = _Head()


class TrainingTest(unittest.TestCase):
    def test_tail_gate_and_auc_selection(self):
        selector = ValidationSelector(0.60, 0.10)
        self.assertFalse(selector.consider(
            {"selection_AUC": 0.70, "catastrophic_rate": 0.106}, 3000))
        self.assertFalse(selector.consider(
            {"selection_AUC": 0.59, "catastrophic_rate": 0.09}, 3000))
        self.assertTrue(selector.consider(
            {"selection_AUC": 0.61, "catastrophic_rate": 0.105}, 6000))
        self.assertEqual(selector.best_step, 6000)
        self.assertAlmostEqual(selector.best_auc, 0.61)

    def test_trainable_scope_and_optimizer_groups(self):
        mt = _Model()
        names = configure_trainable(mt, "mt")
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("decode_head.conv_") for name in names))
        optimizer = optimizer_for(mt, "mt", fused=False)
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["decoder"])

        pose = _Model()
        names = configure_trainable(pose, "pose", (1, 3))
        self.assertTrue(any(name.startswith("backbone.blocks.1.") for name in names))
        self.assertTrue(any(name.startswith("backbone.blocks.3.") for name in names))
        self.assertFalse(any(name.startswith("backbone.blocks.2.") for name in names))
        optimizer = optimizer_for(pose, "pose", fused=False)
        self.assertEqual([group["name"] for group in optimizer.param_groups],
                         ["decoder", "blocks"])

    def test_checkpoint_contains_only_trainables_and_rope(self):
        model = _Model()
        configure_trainable(model, "pose", (2,))
        tensors = checkpoint_tensors(model)
        self.assertIn("backbone.rope_embed.periods", tensors)
        self.assertIn("backbone.blocks.2.weight", tensors)
        self.assertNotIn("backbone.blocks.1.weight", tensors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.safetensors"
            manifest = save_checkpoint(model, path, {"step": 12})
            self.assertTrue(path.is_file())
            self.assertEqual(manifest["step"], 12)
            self.assertEqual(set(manifest["tensor_names"]), set(tensors))

    def test_hint_metrics_macro_average_and_tail(self):
        rows = []
        for source, error in (("NewDays", 0.0), ("EPICK", 0.10)):
            rows.append({
                "points": np.full((21, 2), error, np.float32),
                "gt": np.zeros((21, 2), np.float32),
                "existence": np.ones(21, np.float32),
                "occlusion": np.zeros(21, np.float32),
                "norm": 1.0,
                "source": source,
            })
        metrics = hint_metrics(rows)
        self.assertAlmostEqual(
            metrics["selection_AUC"],
            (metrics["NewDays_all_AUC"] + metrics["EPICK_all_AUC"]) / 2,
        )
        self.assertEqual(metrics["catastrophic_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
