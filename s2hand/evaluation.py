"""Protocol-locked 2D hand metrics used by the S2HAND paper.

Missed hands and non-finite joints are failures at every PCK threshold.  NME
and EPE are descriptive conditional metrics over finite detected predictions;
the primary PCK/AUC metrics retain every annotated joint in the denominator.
"""
from __future__ import annotations

import numpy as np

THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def validate_pair(ground_truth, prediction) -> None:
    """Reject schema, ordering, coordinate, or shape mismatches."""
    gt_schema = str(ground_truth["schema_version"])
    pred_schema = str(prediction["schema_version"])
    if gt_schema not in {"hand2d.v1", "hand2d3d.v1"}:
        raise ValueError(f"Unsupported ground-truth schema: {gt_schema!r}")
    if pred_schema != "hand2d.v1":
        raise ValueError(f"Unsupported prediction schema: {pred_schema!r}")
    if not np.array_equal(ground_truth["sample_id"], prediction["sample_id"]):
        raise ValueError("Prediction sample ids/order do not match ground truth")
    if str(prediction["coordinate_system"]) != "original_image_pixels":
        raise ValueError("Predictions must use original-image pixel coordinates")
    target = np.asarray(ground_truth["keypoints_xy"])
    points = np.asarray(prediction["keypoints_xy"])
    if target.shape != points.shape or target.ndim != 3 or target.shape[1:] != (21, 2):
        raise ValueError(f"Expected matched [N,21,2] coordinates, got {target.shape} and {points.shape}")
    if np.asarray(prediction["detected"]).shape != (len(target),):
        raise ValueError("detected must have shape [N]")


def hand2d_metrics(ground_truth, prediction) -> dict[str, float | int | None]:
    """Compute the exact publication metrics for one aligned hand set."""
    validate_pair(ground_truth, prediction)
    target = np.asarray(ground_truth["keypoints_xy"], np.float32)
    points = np.asarray(prediction["keypoints_xy"], np.float32)
    visible = np.asarray(ground_truth["visibility"], np.float32) > 0
    detected = np.asarray(prediction["detected"], bool)
    boxes = np.asarray(ground_truth["hand_bbox_xyxy"], np.float32)
    if visible.shape != target.shape[:2] or boxes.shape != (len(target), 4):
        raise ValueError("visibility or hand_bbox_xyxy has an invalid shape")

    scale = np.maximum(
        np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), 1.0
    )
    finite = np.isfinite(points).all(axis=2)
    valid_prediction = visible & finite & detected[:, None]
    error_px = np.linalg.norm(points - target, axis=2)
    error_norm = error_px / scale[:, None]
    denominator = max(int(visible.sum()), 1)

    row: dict[str, float | int | None] = {
        "hands": int(len(target)),
        "annotated_joints": int(visible.sum()),
        "evaluated_joints": int(valid_prediction.sum()),
        "detection_recall": float(detected.mean()) if len(detected) else 0.0,
    }
    for threshold in THRESHOLDS:
        row[f"pck_{threshold:.2f}"] = float(
            ((error_norm <= threshold) & valid_prediction).sum() / denominator
        )
    grid = np.linspace(0.0, 0.20, 201)
    curve = np.asarray([
        ((error_norm <= threshold) & valid_prediction).sum() / denominator
        for threshold in grid
    ])
    # np.trapezoid was introduced in NumPy 2.0; np.trapz keeps the declared
    # NumPy >=1.26 runtime compatible and is numerically identical here.
    row["auc_0_20"] = float(np.trapz(curve, grid) / 0.20)

    finite_joint_error = error_norm[valid_prediction]
    finite_joint_error_px = error_px[valid_prediction]
    row["nme_detected_finite"] = (
        float(finite_joint_error.mean()) if finite_joint_error.size else None
    )
    row["epe_px_detected_finite"] = (
        float(finite_joint_error_px.mean()) if finite_joint_error_px.size else None
    )
    per_hand = np.full(len(target), np.inf, np.float32)
    for index in range(len(target)):
        mask = visible[index] & finite[index]
        if detected[index] and mask.any():
            per_hand[index] = float(error_norm[index, mask].mean())
    row["tail_hand_nme_gt_0.20"] = float((per_hand > 0.20).mean()) if len(per_hand) else 0.0
    finite_hands = per_hand[np.isfinite(per_hand)]
    row["finite_hand_nme"] = float(finite_hands.mean()) if finite_hands.size else None
    return row


def side_swap_metrics(ground_truth, prediction) -> dict[str, float | int | None]:
    """Measure assignment swaps in images containing both labeled hand sides."""
    validate_pair(ground_truth, prediction)
    required = {"image_path", "hand_side"}
    missing = sorted(required - set(ground_truth.files))
    if missing:
        raise ValueError(f"Side-swap evaluation requires ground-truth fields: {missing}")
    image_paths = np.asarray(ground_truth["image_path"]).astype(str)
    sides = np.asarray(ground_truth["hand_side"]).astype(str)
    points = np.asarray(prediction["keypoints_xy"], np.float32)
    target = np.asarray(ground_truth["keypoints_xy"], np.float32)
    visible = np.asarray(ground_truth["visibility"], np.float32) > 0
    detected = np.asarray(prediction["detected"], bool)
    groups: dict[str, list[int]] = {}
    for index, path in enumerate(image_paths):
        groups.setdefault(path, []).append(index)
    eligible = swaps = 0
    for indices in groups.values():
        for index in indices:
            opponents = [other for other in indices if sides[other] != sides[index]]
            if not opponents or not detected[index] or not np.isfinite(points[index]).all():
                continue
            own = visible[index]
            if not own.any():
                continue
            own_error = np.linalg.norm(points[index, own] - target[index, own], axis=1).mean()
            opposite_errors = []
            for other in opponents:
                mask = visible[other]
                if mask.any():
                    opposite_errors.append(
                        np.linalg.norm(points[index, mask] - target[other, mask], axis=1).mean()
                    )
            if opposite_errors:
                eligible += 1
                swaps += int(min(opposite_errors) < own_error)
    return {
        "side_swap_rate": None if eligible == 0 else float(swaps / eligible),
        "side_swap_eligible": eligible,
        "side_swaps": swaps,
    }
