"""Standalone HInt TRAIN/VAL adapter used by the S2HAND paper protocol."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .inference import INPUT_SIZE, HEATMAP_SIZE, MEAN, STD, udp_warp_matrix

SOURCES = {"newdays": "NewDays", "epick": "EPICK"}


def hand_side(annotation: dict, stem: str) -> str:
    raw = annotation.get("side", annotation.get("hand_type"))
    if raw is not None:
        raw = str(raw).lower()
        if raw in {"l", "left", "0"}:
            return "left"
        if raw in {"r", "right", "1"}:
            return "right"
    if annotation.get("right") is not None:
        return "right" if bool(annotation["right"]) else "left"
    match = re.search(r"_([lr])$", stem.lower())
    if match:
        return "left" if match.group(1) == "l" else "right"
    raise ValueError(f"Cannot determine hand side from {stem}")


def annotation_bbox(annotation: dict) -> np.ndarray:
    box = np.asarray(annotation["bbox"], np.float32).reshape(-1, 4)[0]
    if box[2] > box[0] and box[3] > box[1]:
        return box
    points = np.asarray(annotation["keypoints"], np.float32).reshape(21, -1)[:, :2]
    exists = np.asarray(annotation.get("existence", np.ones(21))).reshape(21) > 0
    valid = exists & np.isfinite(points).all(1)
    if valid.sum() < 2:
        raise ValueError("Invalid HInt bbox and insufficient visible keypoints")
    minimum, maximum = points[valid].min(0), points[valid].max(0)
    if np.any(maximum <= minimum):
        raise ValueError("Degenerate recovered HInt bbox")
    return np.concatenate((minimum, maximum)).astype(np.float32)


def _aspect_scale(box: np.ndarray, context: float) -> tuple[np.ndarray, np.ndarray]:
    center = (box[:2] + box[2:]) * 0.5
    scale = (box[2:] - box[:2]) * float(context)
    aspect = INPUT_SIZE[0] / INPUT_SIZE[1]
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    return center.astype(np.float32), scale.astype(np.float32)


def udp_gaussian_targets(
    keypoints: np.ndarray, visibility: np.ndarray, sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode 42 input-space points using the released UDP target rule."""
    keypoints = np.asarray(keypoints, np.float32)
    visibility = np.asarray(visibility, np.float32)
    if keypoints.shape != (42, 2) or visibility.shape != (42,):
        raise ValueError("Expected keypoints [42,2] and visibility [42]")
    width, height = HEATMAP_SIZE
    factor = (np.asarray(INPUT_SIZE) - 1) / (np.asarray(HEATMAP_SIZE) - 1)
    points = keypoints / factor
    heatmaps = np.zeros((42, height, width), np.float32)
    weights = visibility.copy()
    radius = int(sigma * 3)
    size = 2 * radius + 1
    grid_x = np.arange(size, dtype=np.float32)
    grid_y = grid_x[:, None]
    for joint in range(42):
        if weights[joint] < 0.5:
            continue
        point = points[joint]
        center = (point + 0.5).astype(np.int64)
        left, top = center - radius
        right, bottom = center + radius + 1
        if left >= width or top >= height or right < 0 or bottom < 0:
            weights[joint] = 0
            continue
        x0 = size // 2 + point[0] - center[0]
        y0 = size // 2 + point[1] - center[1]
        gaussian = np.exp(
            -((grid_x - x0) ** 2 + (grid_y - y0) ** 2) / (2 * sigma**2)
        )
        gx1, gx2 = max(0, -left), min(width, right) - left
        gy1, gy2 = max(0, -top), min(height, bottom) - top
        hx1, hx2 = max(0, left), min(width, right)
        hy1, hy2 = max(0, top), min(height, bottom)
        np.maximum(
            heatmaps[joint, hy1:hy2, hx1:hx2], gaussian[gy1:gy2, gx1:gx2],
            out=heatmaps[joint, hy1:hy2, hx1:hx2],
        )
    return heatmaps, weights


def _photo_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = image.astype(np.float32)
    if rng.random() < 0.75:
        output = output * rng.uniform(0.78, 1.22) + rng.uniform(-18, 18)
        output = 255 * np.power(np.clip(output, 0, 255) / 255, rng.uniform(.82, 1.18))
    output = np.clip(output, 0, 255).astype(np.uint8)
    if rng.random() < 0.30:
        factor = rng.uniform(0.35, 0.80)
        small = cv2.resize(output, (max(2, round(768 * factor)),
                                    max(2, round(1024 * factor))), cv2.INTER_AREA)
        output = cv2.resize(small, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.20:
        kernel = int(rng.integers(3, 12)) | 1
        output = cv2.GaussianBlur(output, (kernel, kernel), rng.uniform(.3, 1.5))
    if rng.random() < 0.18:
        ok, encoded = cv2.imencode(
            ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 91))]
        )
        if ok:
            output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return output


class HIntSupervisedDataset(Dataset):
    """One annotated HInt hand per record, packed into the 42-hand layout."""

    def __init__(self, root: str | Path, split: str, subset: str,
                 augment: bool = False, seed: int = 42):
        split = split.upper()
        if split not in {"TRAIN", "VAL"} or subset not in SOURCES:
            raise ValueError("Use TRAIN/VAL and newdays/epick")
        self.root, self.split, self.subset = Path(root), split, subset
        self.augment, self.seed = bool(augment), int(seed)
        directory = self.root / f"{split}_{subset}_img"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        self.records = sorted(directory.glob("*.json"))
        if not self.records:
            raise RuntimeError(f"No annotations in {directory}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.records[index]
        image = cv2.imread(str(path.with_suffix(".jpg")), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path.with_suffix(".jpg"))
        annotations = json.loads(path.read_text())
        annotations = [annotations] if isinstance(annotations, dict) else annotations
        if len(annotations) != 1:
            raise ValueError(f"Expected one hand in {path}")
        annotation = annotations[0]
        points = np.asarray(annotation["keypoints"], np.float32).reshape(21, -1)[:, :2]
        exists = np.asarray(
            annotation.get("existence", np.ones(21)), np.float32
        ).reshape(21)
        occlusion = np.asarray(
            annotation.get("occlusion", np.zeros(21)), np.float32
        ).reshape(21)
        side = hand_side(annotation, path.stem)
        original_side = side
        original_points = points.copy()
        hand_box = annotation_bbox(annotation)
        rng = np.random.default_rng(np.random.randint(2**32 - 1))
        context = rng.uniform(1.875, 3.125) if self.augment else 2.5
        center, scale = _aspect_scale(hand_box, context)
        rotation = 0.0
        if self.augment:
            center += rng.uniform(-.08, .08, 2).astype(np.float32) * scale
            if rng.random() < .60:
                rotation = float(rng.uniform(-30, 30))
        affine = udp_warp_matrix(center, scale, INPUT_SIZE, rotation)
        sx, sy = np.linalg.norm(affine[0, :2]), np.linalg.norm(affine[1, :2])
        interpolation = cv2.INTER_AREA if min(sx, sy) < 1 else cv2.INTER_CUBIC
        image = cv2.warpAffine(image, affine, INPUT_SIZE, flags=interpolation,
                               borderMode=cv2.BORDER_CONSTANT)
        points = cv2.transform(points[None], affine)[0]
        visible = (exists > 0) & (points[:, 0] >= 0) & (points[:, 0] < 768)
        visible &= (points[:, 1] >= 0) & (points[:, 1] < 1024)
        if self.augment and rng.random() < .5:
            image, points[:, 0] = cv2.flip(image, 1), 767 - points[:, 0]
            side = "right" if side == "left" else "left"
        if self.augment:
            image = _photo_augment(image, rng)
            if rng.random() < .20 and visible.any():
                target = points[int(rng.choice(np.flatnonzero(visible)))]
                size = int(rng.uniform(35, 130))
                x0, y0 = np.rint(target - size / 2).astype(int)
                color = tuple(int(value) for value in rng.integers(0, 256, 3))
                cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), color, -1)
        mapped, weights = np.zeros((42, 2), np.float32), np.zeros(42, np.float32)
        offset = 0 if side == "left" else 21
        mapped[offset:offset + 21], weights[offset:offset + 21] = points, visible
        heatmaps, weights = udp_gaussian_targets(mapped, weights)
        rgb = image[..., ::-1].astype(np.float32)
        normalized = (rgb - MEAN) / STD
        return {
            "images": torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))),
            "target_heatmaps": torch.from_numpy(heatmaps),
            "target_weights": torch.from_numpy(weights),
            "gt_uv": torch.from_numpy(mapped.reshape(2, 21, 2)),
            "visibility": torch.from_numpy(weights.reshape(2, 21)),
            "source": SOURCES[self.subset],
            # Canonical validation metadata. These fields are intentionally
            # returned for TRAIN too, keeping one stable collate contract.
            "sample_id": f"{self.split.lower()}_{self.subset}_{index:08d}",
            "image_path": str(path.with_suffix(".jpg")),
            "hand_side": original_side,
            "gt_original": torch.from_numpy(original_points),
            "existence_original": torch.from_numpy(exists),
            "occlusion_original": torch.from_numpy(occlusion),
            "hand_bbox_original": torch.from_numpy(hand_box),
            "metric_norm": torch.tensor(
                max(float(hand_box[2] - hand_box[0]),
                    float(hand_box[3] - hand_box[1]), 1.0),
                dtype=torch.float32,
            ),
            "crop_center": torch.from_numpy(center.copy()),
            "crop_scale": torch.from_numpy(scale.copy()),
        }


class SourceBalancedHInt(Dataset):
    """Deterministic 1:1 NewDays:EPICK exposure used by released training."""

    def __init__(self, root: str | Path, split="TRAIN", augment=True,
                 length=24_000, seed=42):
        if length <= 0 or length % 2:
            raise ValueError("length must be a positive even number")
        self.datasets = (
            HIntSupervisedDataset(root, split, "newdays", augment, seed),
            HIntSupervisedDataset(root, split, "epick", augment, seed),
        )
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        source = index % 2
        return self.datasets[source][(index // 2) % len(self.datasets[source])]
