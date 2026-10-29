#!/usr/bin/env python3
"""Run native S2HAND, then conditionally re-run low-confidence hands.

The two inputs use the compact ``detection.v1`` NPZ schema documented below.
Person detections provide the normal whole-person first pass. Hand detections
provide left/right boxes only for routing and the conditional crop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s2hand.inference import infer_people
from s2hand.loader import NAMES, load_model
from s2hand.losses import HAND_CHANNELS
from s2hand.tailguard import expanded_square_box, load_frozen_config, should_trigger


SIDE_ID = {"left": 0, "right": 1}
SIDE_CHANNELS = {
    "left": np.asarray(HAND_CHANNELS[:21], np.int64),
    "right": np.asarray(HAND_CHANNELS[21:], np.int64),
}


class DetectionCache:
    """Validated read-only accessor for flattened ``detection.v1`` NPZ files."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        if str(data["schema_version"]) != "detection.v1":
            raise ValueError(f"{path} is not detection.v1")
        self.image_path = np.asarray(data["image_path"])
        self.offsets = np.asarray(data["offsets"], np.int64)
        self.boxes = np.asarray(data["boxes_xyxy"], np.float32)
        self.scores = np.asarray(data["scores"], np.float32)
        self.hand_side = (np.asarray(data["hand_side"], np.int8)
                          if "hand_side" in data else None)
        if len(self.offsets) != len(self.image_path) + 1:
            raise ValueError("Invalid detection offsets")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.boxes):
            raise ValueError("Detection offsets do not span boxes")
        if len(self.scores) != len(self.boxes):
            raise ValueError("Detection scores and boxes differ in length")
        if self.hand_side is not None and len(self.hand_side) != len(self.boxes):
            raise ValueError("Detection hand-side and boxes differ in length")
        self.index = {str(value): index for index, value in enumerate(self.image_path)}
        if len(self.index) != len(self.image_path):
            raise ValueError("Detection cache contains duplicate image paths")

    def rows(self, image_path: str):
        index = self.index[image_path]
        start, end = self.offsets[index:index + 2]
        return np.arange(start, end, dtype=np.int64)

    def best_hand(self, image_path: str, side: str):
        if self.hand_side is None:
            raise ValueError("Hand detection cache has no hand_side field")
        rows = self.rows(image_path)
        rows = rows[self.hand_side[rows] == SIDE_ID[side]]
        if not len(rows):
            return None
        selected = rows[np.argmax(self.scores[rows])]
        return self.boxes[selected].copy(), float(self.scores[selected])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=NAMES, default="S2HAND-MT-5B")
    parser.add_argument("--person-detections", type=Path, required=True)
    parser.add_argument("--hand-detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/tailguard_v1.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--base-path")
    parser.add_argument("--delta-path")
    parser.add_argument("--manifest-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error("batch size and max images must be positive")

    people, hands = DetectionCache(args.person_detections), DetectionCache(args.hand_detections)
    paths = [str(path) for path in people.image_path if str(path) in hands.index]
    if args.max_images is not None:
        paths = paths[:args.max_images]
    if not paths:
        raise RuntimeError("Detection caches have no common image paths")
    frozen_payload = json.loads(args.config.read_text())
    frozen = load_frozen_config(args.config)
    crop_context = float(frozen_payload["selected"]["crop_context"])
    model = load_model(
        args.model, device=args.device, cache_dir=args.cache_dir,
        base_path=args.base_path, delta_path=args.delta_path,
        manifest_path=args.manifest_path, local_files_only=args.offline,
    )

    output_rows = []
    native_forwards = recrop_forwards = 0
    for image_path in paths:
        person_rows = people.rows(image_path)
        if not len(person_rows):
            continue
        native = infer_people(
            model,
            [(image_path, people.boxes[row]) for row in person_rows],
            device=args.device, batch_size=args.batch_size, bf16=not args.fp32,
        )
        native_forwards += len(person_rows)
        recrop_jobs = []
        pending = []
        for side in ("left", "right"):
            channels = SIDE_CHANNELS[side]
            candidate = int(np.argmax(native["keypoint_scores"][:, channels].mean(1)))
            native_points = native["keypoints_xy"][candidate, channels].copy()
            native_scores = native["keypoint_scores"][candidate, channels].copy()
            detection = hands.best_hand(image_path, side)
            present = detection is not None
            trigger = should_trigger(
                {"native_conf_mean": float(native_scores.mean())}, present, frozen
            )
            row = {
                "image_path": image_path, "side": side,
                "native_points": native_points, "native_scores": native_scores,
                "points": native_points.copy(), "scores": native_scores.copy(),
                "detector_present": present, "triggered": trigger,
                "detector_box": (np.full(4, np.nan, np.float32)
                                 if not present else detection[0].astype(np.float32)),
                "recrop_box": np.full(4, np.nan, np.float32),
            }
            output_rows.append(row)
            if trigger:
                box = expanded_square_box(detection[0], crop_context)
                row["recrop_box"] = box
                pending.append((row, side))
                recrop_jobs.append((image_path, box))
        if recrop_jobs:
            recropped = infer_people(
                model, recrop_jobs, device=args.device,
                batch_size=args.batch_size, bf16=not args.fp32,
            )
            recrop_forwards += len(recrop_jobs)
            for index, (row, side) in enumerate(pending):
                channels = SIDE_CHANNELS[side]
                # Replace all 21 joints coherently, including the wrist.
                row["points"] = recropped["keypoints_xy"][index, channels].copy()
                row["scores"] = recropped["keypoint_scores"][index, channels].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("tailguard.prediction.v1"),
        model_name=np.asarray(args.model),
        config=np.asarray(str(args.config)),
        image_path=np.asarray([row["image_path"] for row in output_rows]),
        hand_side=np.asarray([row["side"] for row in output_rows]),
        native_keypoints_xy=np.stack([row["native_points"] for row in output_rows]),
        native_scores=np.stack([row["native_scores"] for row in output_rows]),
        keypoints_xy=np.stack([row["points"] for row in output_rows]),
        scores=np.stack([row["scores"] for row in output_rows]),
        detector_present=np.asarray([row["detector_present"] for row in output_rows], bool),
        triggered=np.asarray([row["triggered"] for row in output_rows], bool),
        detector_box_xyxy=np.stack([row["detector_box"] for row in output_rows]),
        recrop_box_xyxy=np.stack([row["recrop_box"] for row in output_rows]),
        native_pose_forwards=np.asarray(native_forwards, np.int64),
        recrop_pose_forwards=np.asarray(recrop_forwards, np.int64),
    )
    print(f"Saved {len(output_rows)} hand rows; triggers={recrop_forwards}; "
          f"native_forwards={native_forwards}: {args.output}")


if __name__ == "__main__":
    main()
