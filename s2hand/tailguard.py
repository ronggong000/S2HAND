#!/usr/bin/env python3
"""Optional confidence-triggered conditional re-crop router for S2HAND."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HAND_EDGES = tuple(
    edge
    for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8),
                   (0, 9, 10, 11, 12), (0, 13, 14, 15, 16),
                   (0, 17, 18, 19, 20))
    for edge in zip(finger[:-1], finger[1:])
)


def load_frozen_config(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "tailguard.router.v1":
        raise ValueError("Expected tailguard.router.v1")
    if not str(payload.get("decision", "")).startswith("ACCEPT_"):
        raise ValueError("Frozen config is not accepted")
    config = payload["selected"]["config"]
    if config.get("kind") != "rules" or not config.get("rules"):
        raise ValueError("Frozen config has no routing rules")
    return config


def geometry_features(points):
    """Translation/scale-invariant geometry features for one 21-joint hand."""
    points = np.asarray(points, np.float32)
    if points.shape != (21, 2):
        raise ValueError(f"Expected (21,2), got {points.shape}")
    lengths = np.asarray([
        np.linalg.norm(points[child] - points[parent]) for parent, child in HAND_EDGES
    ], np.float32)
    positive = lengths[lengths > 1e-4]
    median = float(np.median(positive)) if len(positive) else 1.0
    normalized = lengths / max(median, 1e-4)
    mad = float(np.median(np.abs(normalized - np.median(normalized))))
    lo, hi = points.min(axis=0), points.max(axis=0)
    return {
        "native_bone_mad_norm": mad,
        "native_max_isolated_joint_distance": float(normalized.max()),
        "native_hand_spread_norm": float(np.linalg.norm(hi - lo) / max(median, 1e-4)),
    }


def detector_association_features(points, detector_box):
    points = np.asarray(points, np.float32)
    box = np.asarray(detector_box, np.float32)
    if points.shape != (21, 2) or box.shape != (4,):
        raise ValueError("Expected points (21,2) and box (4,)")
    inside = ((points[:, 0] >= box[0]) & (points[:, 0] <= box[2])
              & (points[:, 1] >= box[1]) & (points[:, 1] <= box[3]))
    center = 0.5 * (box[:2] + box[2:])
    scale = max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
    return {
        "native_joint_in_detector_fraction": float(inside.mean()),
        "native_center_to_detector_norm": float(
            np.linalg.norm(np.median(points, axis=0) - center) / scale
        ),
        "native_wrist_to_detector_norm": float(np.linalg.norm(points[0] - center) / scale),
    }


def should_trigger(features, detector_present, config):
    """Deterministically evaluate an OR-of-rules frozen TailGuard config."""
    if not detector_present:
        return False
    decisions = []
    for rule in config["rules"]:
        value = float(features[rule["feature"]])
        threshold = float(rule["threshold"])
        decisions.append(value <= threshold if rule["direction"] == "low"
                         else value >= threshold)
    return bool(any(decisions))


def crop_to_original(points, input_size, bbox_center, bbox_scale):
    """Map codec-space coordinates back to original-image pixels."""
    points = np.asarray(points, np.float32)
    input_size = np.asarray(input_size, np.float32)
    center = np.asarray(bbox_center, np.float32)
    scale = np.asarray(bbox_scale, np.float32)
    return points / input_size * scale + center - 0.5 * scale


def original_to_crop(points, input_size, bbox_center, bbox_scale):
    points = np.asarray(points, np.float32)
    input_size = np.asarray(input_size, np.float32)
    center = np.asarray(bbox_center, np.float32)
    scale = np.asarray(bbox_scale, np.float32)
    return (points - center + 0.5 * scale) / scale * input_size
