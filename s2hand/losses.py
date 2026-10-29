"""Training losses used by S2HAND-MT and S2HAND-Pose.

The deployed model is still decoded with UDP/DARK. During training, the two
geometry terms use a differentiable soft-argmax over the 42 hand heatmaps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

HAND_EDGES = tuple(
    edge
    for start in (1, 5, 9, 13, 17)
    for edge in ((0, start), (start, start + 1),
                 (start + 1, start + 2), (start + 2, start + 3))
)
COORDINATE_WEIGHT = 0.0030615822269171302
TOPOLOGY_WEIGHT = 0.0016362079765548576
# COCO WholeBody anatomical order: left wrist + five finger chains, then
# right wrist + five finger chains. Sapiens2 stores each finger tip-to-root,
# so this is deliberately not ``range(21, 63)``.
HAND_CHANNELS = (
    62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50,
    57, 56, 55, 54, 61, 60, 59, 58,
    41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29,
    36, 35, 34, 33, 40, 39, 38, 37,
)


def hand_heatmaps_from_output(output: torch.Tensor) -> torch.Tensor:
    """Select Sapiens2 hand heatmaps in COCO/HInt anatomical order."""
    if output.ndim != 4 or output.shape[1] != 308:
        raise ValueError(f"Expected [B,308,H,W], got {tuple(output.shape)}")
    channels = torch.as_tensor(HAND_CHANNELS, device=output.device)
    return output.index_select(1, channels)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def softargmax_hand_uv(
    heatmaps: torch.Tensor,
    beta: float = 64.0,
    input_size: tuple[int, int] = (768, 1024),
) -> torch.Tensor:
    """Decode ``[B,42,H,W]`` to ``[B,2,21,2]`` in input pixels."""
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 42:
        raise ValueError(f"Expected [B,42,H,W], got {tuple(heatmaps.shape)}")
    batch, joints, height, width = heatmaps.shape
    probability = torch.softmax(
        heatmaps.float().flatten(2) * float(beta), dim=-1
    ).view(batch, joints, height, width)
    x_grid = torch.arange(width, device=heatmaps.device, dtype=torch.float32)
    y_grid = torch.arange(height, device=heatmaps.device, dtype=torch.float32)
    x = (probability.sum(dim=2) * x_grid).sum(-1)
    y = (probability.sum(dim=3) * y_grid).sum(-1)
    x *= float(input_size[0]) / float(width - 1)
    y *= float(input_size[1]) / float(height - 1)
    return torch.stack((x, y), dim=-1).reshape(batch, 2, 21, 2)


def visible_hand_scale(gt_uv: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    """Visible-joint bounding span per hand in network-input pixels."""
    valid = (visibility > 0) & torch.isfinite(gt_uv).all(-1)
    minimum = torch.where(valid[..., None], gt_uv,
                          torch.full_like(gt_uv, torch.inf)).amin(dim=-2)
    maximum = torch.where(valid[..., None], gt_uv,
                          torch.full_like(gt_uv, -torch.inf)).amax(dim=-2)
    span = (maximum - minimum).amax(dim=-1)
    enough = valid.sum(dim=-1) >= 2
    return torch.where(
        enough & torch.isfinite(span), span.clamp_min(16.0), torch.ones_like(span)
    )


def hand_geometry_losses(
    heatmaps: torch.Tensor,
    gt_uv: torch.Tensor,
    visibility: torch.Tensor,
    beta: float = 64.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return scale-normalized coordinate and 2D bone-vector losses."""
    predicted_uv = softargmax_hand_uv(heatmaps, beta=beta)
    gt_uv, visibility = gt_uv.float(), visibility.float()
    if gt_uv.shape != predicted_uv.shape or visibility.shape != gt_uv.shape[:-1]:
        raise ValueError(
            f"Expected gt_uv {tuple(predicted_uv.shape)} and visibility "
            f"{tuple(predicted_uv.shape[:-1])}, got {tuple(gt_uv.shape)} and "
            f"{tuple(visibility.shape)}"
        )
    valid_joint = (visibility > 0) & torch.isfinite(gt_uv).all(-1)
    scale = visible_hand_scale(gt_uv, visibility)

    normalized_delta = (
        predicted_uv - torch.nan_to_num(gt_uv)
    ) / scale[..., None, None]
    coordinate = _masked_mean(
        F.smooth_l1_loss(
            normalized_delta, torch.zeros_like(normalized_delta),
            reduction="none", beta=0.05,
        ),
        valid_joint,
    )
    predicted_bones = torch.stack(
        [predicted_uv[:, :, child] - predicted_uv[:, :, parent]
         for parent, child in HAND_EDGES], dim=2,
    )
    target_bones = torch.stack(
        [gt_uv[:, :, child] - gt_uv[:, :, parent]
         for parent, child in HAND_EDGES], dim=2,
    )
    edge_valid = torch.stack(
        [valid_joint[:, :, parent] & valid_joint[:, :, child]
         for parent, child in HAND_EDGES], dim=2,
    )
    topology = _masked_mean(
        F.smooth_l1_loss(
            predicted_bones / scale[..., None, None],
            torch.nan_to_num(target_bones) / scale[..., None, None],
            reduction="none", beta=0.025,
        ),
        edge_valid,
    )
    return coordinate, topology, {
        "predicted_uv": predicted_uv.detach(),
        "valid_joints": valid_joint.float().sum().detach(),
        "valid_edges": edge_valid.float().sum().detach(),
        "mean_hand_scale": _masked_mean(scale, valid_joint.any(-1)).detach(),
    }


def hand_heatmap_mse(
    output: torch.Tensor,
    target: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    """Sapiens2 target-weighted heatmap MSE for 42 hand channels."""
    if output.shape != target.shape:
        raise ValueError(f"Output/target mismatch: {output.shape} vs {target.shape}")
    if target_weights.shape != output.shape[:2]:
        raise ValueError(
            f"Expected target weights {output.shape[:2]}, got {target_weights.shape}"
        )
    mask = target_weights.to(output.dtype)[..., None, None]
    return (F.mse_loss(output, target, reduction="none") * mask).mean()


def s2hand_objective(
    hand_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_weights: torch.Tensor,
    gt_uv: torch.Tensor,
    visibility: torch.Tensor,
    *,
    coordinate_weight: float = COORDINATE_WEIGHT,
    topology_weight: float = TOPOLOGY_WEIGHT,
    beta: float = 64.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the exact three-term S2HAND hand-training objective."""
    heatmap = hand_heatmap_mse(hand_heatmaps, target_heatmaps, target_weights)
    coordinate, topology, diagnostics = hand_geometry_losses(
        hand_heatmaps, gt_uv, visibility, beta=beta
    )
    total = heatmap + coordinate * float(coordinate_weight) + topology * float(topology_weight)
    terms = {
        "loss": total,
        "loss_heatmap": heatmap,
        "loss_coordinate": coordinate * float(coordinate_weight),
        "loss_topology": topology * float(topology_weight),
        **diagnostics,
    }
    return total, terms
