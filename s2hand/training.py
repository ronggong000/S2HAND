"""Standalone optimizer, validation, and checkpoint utilities for S2HAND."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from safetensors.torch import save_file

from .inference import HEATMAP_SIZE, INPUT_SIZE, _dark_udp, _heatmap_maximum
from .losses import HAND_CHANNELS, hand_heatmaps_from_output, s2hand_objective


@dataclass
class ValidationSelector:
    """Best-AUC selector with the preregistered HInt catastrophic-tail gate."""

    baseline_auc: float
    baseline_tail: float
    max_tail_increase: float = 0.005
    best_auc: float | None = None
    best_step: int = 0

    def __post_init__(self) -> None:
        if self.best_auc is None:
            self.best_auc = float(self.baseline_auc)

    def consider(self, metrics: dict[str, float], step: int) -> bool:
        eligible = (
            float(metrics["catastrophic_rate"])
            <= float(self.baseline_tail) + float(self.max_tail_increase)
        )
        improved = eligible and float(metrics["selection_AUC"]) > float(self.best_auc)
        if improved:
            self.best_auc = float(metrics["selection_AUC"])
            self.best_step = int(step)
        return improved


def configure_trainable(model: torch.nn.Module, variant: str,
                        blocks: tuple[int, ...] = (38, 39, 40)) -> list[str]:
    """Freeze everything except the adapted decoder and optional pose blocks."""
    if variant not in {"mt", "pose"}:
        raise ValueError("variant must be 'mt' or 'pose'")
    model.requires_grad_(False)
    model.decode_head.conv_layers.requires_grad_(True)
    model.decode_head.conv_pose.requires_grad_(True)
    if variant == "pose":
        count = len(model.backbone.blocks)
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("Pose blocks must be unique and non-empty")
        if min(blocks) < 0 or max(blocks) >= count:
            raise ValueError(f"Pose blocks must be in [0,{count - 1}]")
        for index in blocks:
            model.backbone.blocks[index].requires_grad_(True)
    return [name for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def optimizer_for(model: torch.nn.Module, variant: str, *,
                  decoder_lr: float = 2e-5, block_lr: float = 2e-6,
                  fused: bool = True) -> torch.optim.Optimizer:
    """Construct the paper AdamW groups without layer decay or weight decay."""
    decoder, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("backbone.blocks.") else decoder).append(parameter)
    if not decoder:
        raise RuntimeError("No adapted-decoder parameters are trainable")
    groups = [{"params": decoder, "lr": float(decoder_lr), "name": "decoder"}]
    if variant == "pose":
        if not backbone:
            raise RuntimeError("Pose variant has no trainable Transformer parameters")
        groups.append({"params": backbone, "lr": float(block_lr), "name": "blocks"})
    elif backbone:
        raise RuntimeError("MT must not update Transformer parameters")
    kwargs = dict(betas=(0.9, 0.999), weight_decay=0.0)
    if fused and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except TypeError:  # Older PyTorch without fused AdamW.
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def enable_activation_checkpointing(model: torch.nn.Module,
                                    start_block: int) -> None:
    """Checkpoint backbone blocks from ``start_block`` without changing keys."""
    from torch.utils.checkpoint import checkpoint

    for index in range(int(start_block), len(model.backbone.blocks)):
        block = model.backbone.blocks[index]
        if getattr(block, "_s2hand_checkpointed", False):
            continue
        original = block.forward

        def checkpointed(self, x, rope=None, _original=original):
            if not self.training or not torch.is_grad_enabled():
                return _original(x, rope=rope)
            return checkpoint(
                lambda value: _original(value, rope=rope), x,
                use_reentrant=False,
            )

        block.forward = MethodType(checkpointed, block)
        block._s2hand_checkpointed = True


def training_step(model: torch.nn.Module, batch: dict, optimizer,
                  device: torch.device, *, bf16: bool = True,
                  grad_clip: float = 1.0) -> dict[str, float]:
    """Run one batch with the exact released three-term hand objective."""
    optimizer.zero_grad(set_to_none=True)
    images = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
    targets = batch["target_heatmaps"].to(device, non_blocking=True)
    weights = batch["target_weights"].to(device, non_blocking=True)
    gt_uv = batch["gt_uv"].to(device, non_blocking=True)
    visibility = batch["visibility"].to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.bfloat16,
                        enabled=bf16 and device.type == "cuda"):
        output = model(images)
        loss, terms = s2hand_objective(
            hand_heatmaps_from_output(output), targets, weights, gt_uv, visibility
        )
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(grad_clip),
    )
    optimizer.step()
    result = {name: float(value.detach().float().cpu())
              for name, value in terms.items() if value.numel() == 1}
    result["grad_norm"] = float(torch.as_tensor(norm).detach().float().cpu())
    return result


def _decode_one(heatmaps: np.ndarray, side: str, center: np.ndarray,
                scale: np.ndarray, hand_box: np.ndarray,
                region_scale: float = 1.30) -> np.ndarray:
    offset = 0 if side == "left" else 21
    selected = heatmaps[np.asarray(HAND_CHANNELS[offset:offset + 21])].copy()
    box_center = (hand_box[:2] + hand_box[2:]) * 0.5
    half = (hand_box[2:] - hand_box[:2]) * (float(region_scale) / 2.0)
    region_min, region_max = box_center - half, box_center + half
    height, width = selected.shape[-2:]
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, INPUT_SIZE[0], width, dtype=np.float32),
        np.linspace(0.0, INPUT_SIZE[1], height, dtype=np.float32),
    )
    grid = np.stack((grid_x, grid_y), axis=-1)
    grid = grid / np.asarray(INPUT_SIZE, np.float32) * scale + center - 0.5 * scale
    inside = ((grid[..., 0] >= region_min[0]) & (grid[..., 0] <= region_max[0]) &
              (grid[..., 1] >= region_min[1]) & (grid[..., 1] <= region_max[1]))
    if not inside.any():
        raise RuntimeError(f"Target region contains no heatmap pixels: {hand_box}")
    floor = (float(np.nanmin(selected))
             - max(float(np.nanmax(selected) - np.nanmin(selected)), 1.0) - 100.0)
    selected[:, ~inside] = floor
    points, _ = _heatmap_maximum(selected)
    points = _dark_udp(points, selected.copy(), 11)[0]
    points /= np.asarray([HEATMAP_SIZE[0] - 1, HEATMAP_SIZE[1] - 1], np.float32)
    points *= np.asarray(INPUT_SIZE, np.float32)
    points = points / np.asarray(INPUT_SIZE, np.float32) * scale + center - 0.5 * scale
    return np.clip(points, region_min, region_max).astype(np.float32)


def hint_metrics(rows: list[dict]) -> dict[str, float]:
    """Exact HInt VAL metrics used for paper checkpoint selection."""
    if not rows:
        raise ValueError("No HInt validation rows")
    errors = np.stack([
        np.linalg.norm(row["points"] - row["gt"], axis=-1)
        / max(float(row["norm"]), 1.0)
        for row in rows
    ])
    existence = np.stack([row["existence"] for row in rows]) > 0
    occlusion = np.stack([row["occlusion"] for row in rows]) > 0
    sources = np.asarray([row["source"] for row in rows])
    metrics: dict[str, float] = {}
    scopes = {"combined": np.ones(len(rows), bool),
              "NewDays": sources == "NewDays", "EPICK": sources == "EPICK"}
    for scope, hand_mask in scopes.items():
        if not hand_mask.any():
            continue
        for group, joint_mask in {
            "all": existence,
            "visible": existence & ~occlusion,
            "occluded": existence & occlusion,
        }.items():
            selector = joint_mask & hand_mask[:, None]
            if not selector.any():
                continue
            values = errors[selector]
            grid = np.linspace(0.0, 0.20, 201)
            curve = np.asarray([np.mean(values <= threshold) for threshold in grid])
            prefix = f"{scope}_{group}_"
            metrics[prefix + "NME"] = float(values.mean())
            for threshold in (0.05, 0.10, 0.15):
                metrics[prefix + f"PCK@{threshold:.2f}"] = float(
                    np.mean(values <= threshold)
                )
            # Keep compatibility with the declared NumPy >=1.26 requirement.
            metrics[prefix + "AUC"] = float(np.trapz(curve, grid) / 0.20)
    primary = [metrics[f"{source}_all_AUC"] for source in ("NewDays", "EPICK")
               if f"{source}_all_AUC" in metrics]
    if len(primary) != 2:
        raise RuntimeError("Checkpoint selection requires both NewDays and EPICK VAL")
    metrics["selection_AUC"] = float(np.mean(primary))
    counts = existence.sum(1)
    if np.any(counts == 0):
        raise RuntimeError("HInt validation hand has no annotated joints")
    per_hand_nme = (errors * existence).sum(1) / counts
    metrics["catastrophic_rate"] = float(np.mean(per_hand_nme > 0.20))
    metrics["hands"] = float(len(rows))
    return metrics


@torch.inference_mode()
def validate_hint(model: torch.nn.Module, loader, device: torch.device, *,
                  bf16: bool = True, region_scale: float = 1.30) -> dict[str, float]:
    """Run canonical target-region HInt validation."""
    was_training = model.training
    model.eval()
    rows = []
    for batch in loader:
        images = batch["images"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda"):
            outputs = model(images).float().cpu().numpy()
        size = outputs.shape[0]
        for index in range(size):
            side = batch["hand_side"][index]
            center = batch["crop_center"][index].numpy().astype(np.float32)
            scale = batch["crop_scale"][index].numpy().astype(np.float32)
            hand_box = batch["hand_bbox_original"][index].numpy().astype(np.float32)
            rows.append({
                "points": _decode_one(outputs[index], side, center, scale, hand_box,
                                      region_scale),
                "gt": batch["gt_original"][index].numpy().astype(np.float32),
                "existence": batch["existence_original"][index].numpy(),
                "occlusion": batch["occlusion_original"][index].numpy(),
                "norm": float(batch["metric_norm"][index]),
                "source": batch["source"][index],
            })
    if was_training:
        model.train()
    return hint_metrics(rows)


def checkpoint_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return trainable tensors plus the BF16 RoPE buffer needed for replay."""
    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    tensors["backbone.rope_embed.periods"] = (
        model.backbone.rope_embed.periods.detach().cpu().contiguous()
    )
    return tensors


def save_checkpoint(model: torch.nn.Module, path: str | Path,
                    metadata: dict | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = checkpoint_tensors(model)
    save_file(tensors, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "s2hand_training_checkpoint_v1",
        "file": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "tensor_names": sorted(tensors),
        **(metadata or {}),
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
