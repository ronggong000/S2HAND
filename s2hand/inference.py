"""Image preprocessing and UDP/DARK decoding for S2HAND.

This module mirrors the public Sapiens2 top-down pose test pipeline without
requiring an MMEngine/MMCV research checkout. Bounding boxes use ``xyxy``
coordinates in the original image. Images passed to :func:`prepare_person` are
OpenCV BGR arrays.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

INPUT_SIZE = (768, 1024)  # width, height
HEATMAP_SIZE = (192, 256)  # width, height
BBOX_PADDING = 1.25
BLUR_KERNEL_SIZE = 11
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def bbox_center_scale(
    bbox_xyxy: Sequence[float],
    *,
    padding: float = BBOX_PADDING,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one ``xyxy`` box to Sapiens2's padded, aspect-fixed crop."""
    box = np.asarray(bbox_xyxy, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError(f"Expected one xyxy box with shape (4,), got {box.shape}")
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid xyxy box: {box.tolist()}")

    center = (box[:2] + box[2:]) * 0.5
    scale = (box[2:] - box[:2]) * float(padding)
    aspect = float(input_size[0]) / float(input_size[1])
    width, height = map(float, scale)
    if width > height * aspect:
        height = width / aspect
    else:
        width = height * aspect
    return center.astype(np.float32), np.asarray([width, height], np.float32)


def udp_warp_matrix(
    center: Sequence[float],
    scale: Sequence[float],
    output_size: tuple[int, int] = INPUT_SIZE,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    """Return the exact unbiased-data-processing affine matrix."""
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if center.shape != (2,) or scale.shape != (2,):
        raise ValueError("center and scale must each contain two values")
    angle = math.radians(float(rotation_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    sx = (output_size[0] - 1.0) / scale[0]
    sy = (output_size[1] - 1.0) / scale[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = cosine * sx
    matrix[0, 1] = -sine * sx
    matrix[0, 2] = sx * (-center[0] * cosine + center[1] * sine + 0.5 * scale[0])
    matrix[1, 0] = sine * sy
    matrix[1, 1] = cosine * sy
    matrix[1, 2] = sy * (-center[0] * sine - center[1] * cosine + 0.5 * scale[1])
    return matrix


def prepare_person(
    image_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Crop and normalize one detected person for S2HAND.

    Returns a ``3x1024x768`` float tensor and metadata needed to restore the
    decoded keypoints to original-image coordinates.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got {image_bgr.shape}")
    center, scale = bbox_center_scale(bbox_xyxy)
    matrix = udp_warp_matrix(center, scale)
    sx = np.linalg.norm(matrix[0, :2])
    sy = np.linalg.norm(matrix[1, :2])
    interpolation = cv2.INTER_AREA if min(sx, sy) < 1.0 else cv2.INTER_CUBIC
    crop_bgr = cv2.warpAffine(image_bgr, matrix, INPUT_SIZE, flags=interpolation)
    crop_rgb = crop_bgr[..., ::-1].astype(np.float32)
    normalized = (crop_rgb - MEAN) / STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
    return tensor, {"bbox_center": center, "bbox_scale": scale}


def _heatmap_maximum(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = heatmaps.shape
    flat = heatmaps.reshape(channels, -1)
    indices = np.argmax(flat, axis=1)
    y, x = np.unravel_index(indices, (height, width))
    points = np.stack((x, y), axis=-1).astype(np.float32)
    scores = np.amax(flat, axis=1)
    points[scores <= 0.0] = -1
    return points[None], scores[None]


def _gaussian_blur(heatmaps: np.ndarray, kernel: int) -> np.ndarray:
    border = (kernel - 1) // 2
    channels, height, width = heatmaps.shape
    for channel in range(channels):
        original_max = np.max(heatmaps[channel])
        padded = np.zeros((height + 2 * border, width + 2 * border), np.float32)
        padded[border:-border, border:-border] = heatmaps[channel]
        padded = cv2.GaussianBlur(padded, (kernel, kernel), 0)
        blurred = padded[border:-border, border:-border]
        blurred_max = np.max(blurred)
        if blurred_max > 0:
            blurred *= original_max / blurred_max
        heatmaps[channel] = blurred
    return heatmaps


def _dark_udp(points: np.ndarray, heatmaps: np.ndarray, kernel: int) -> np.ndarray:
    """Distribution-aware UDP refinement used by the released pose codec."""
    _, channels = points.shape[:2]
    height, width = heatmaps.shape[1:]
    heatmaps = _gaussian_blur(heatmaps, kernel)
    np.clip(heatmaps, 1e-3, 50.0, out=heatmaps)
    np.log(heatmaps, out=heatmaps)
    padded = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    index = points[0, :, 0] + 1 + (points[0, :, 1] + 1) * (width + 2)
    index += (width + 2) * (height + 2) * np.arange(channels)
    index = index.astype(int).reshape(-1, 1)
    center = padded[index]
    xp = padded[index + 1]
    yp = padded[index + width + 2]
    xpyp = padded[index + width + 3]
    xmym = padded[index - width - 3]
    xm = padded[index - 1]
    ym = padded[index - width - 2]
    derivative = np.concatenate((0.5 * (xp - xm), 0.5 * (yp - ym)), axis=1)
    derivative = derivative.reshape(channels, 2, 1)
    dxx = xp - 2 * center + xm
    dyy = yp - 2 * center + ym
    dxy = 0.5 * (xpyp - xp - yp + 2 * center - xm - ym + xmym)
    hessian = np.concatenate((dxx, dxy, dxy, dyy), axis=1).reshape(channels, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    points[0] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze(-1)
    return points


def decode_udp(
    heatmaps: np.ndarray | torch.Tensor,
    metadata: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one ``308x256x192`` heatmap tensor to original-image xy."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().float().cpu().numpy()
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.shape != (308, HEATMAP_SIZE[1], HEATMAP_SIZE[0]):
        raise ValueError(
            f"Expected heatmaps (308,{HEATMAP_SIZE[1]},{HEATMAP_SIZE[0]}), "
            f"got {heatmaps.shape}"
        )
    points, scores = _heatmap_maximum(heatmaps)
    points = _dark_udp(points, heatmaps.copy(), BLUR_KERNEL_SIZE)
    points = points / np.asarray([HEATMAP_SIZE[0] - 1, HEATMAP_SIZE[1] - 1])
    points *= np.asarray(INPUT_SIZE, dtype=np.float32)
    center = np.asarray(metadata["bbox_center"], dtype=np.float32)
    scale = np.asarray(metadata["bbox_scale"], dtype=np.float32)
    points = points / np.asarray(INPUT_SIZE, np.float32) * scale + center - 0.5 * scale
    return points[0].astype(np.float32), scores[0].astype(np.float32)


def read_boxes(path: str | Path) -> list[tuple[str, list[float]]]:
    """Read ``image_path x1 y1 x2 y2`` rows from a whitespace TSV file."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected image path and four box values")
        rows.append((fields[0], [float(value) for value in fields[1:]]))
    if not rows:
        raise ValueError(f"No input rows in {path}")
    return rows


def infer_people(
    model: torch.nn.Module,
    rows: Iterable[tuple[str, Sequence[float]]],
    *,
    device: str | torch.device,
    batch_size: int = 1,
    bf16: bool = True,
) -> dict[str, np.ndarray]:
    """Run S2HAND for an iterable of image/box pairs."""
    rows = list(rows)
    image_paths, boxes, points, scores = [], [], [], []
    device = torch.device(device)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prepared = []
        for image_path, box in chunk:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            tensor, metadata = prepare_person(image, box)
            prepared.append((tensor, metadata))
            image_paths.append(str(image_path))
            boxes.append(np.asarray(box, np.float32))
        images = torch.stack([item[0] for item in prepared]).to(device)
        with torch.inference_mode(), torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"
        ):
            heatmaps = model(images)
        for heatmap, (_, metadata) in zip(heatmaps, prepared):
            point, score = decode_udp(heatmap, metadata)
            points.append(point)
            scores.append(score)
    return {
        "image_paths": np.asarray(image_paths),
        "boxes_xyxy": np.asarray(boxes, np.float32),
        "keypoints_xy": np.asarray(points, np.float32),
        "keypoint_scores": np.asarray(scores, np.float32),
    }
