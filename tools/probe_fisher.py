#!/usr/bin/env python3
"""Rank S2HAND Transformer blocks by diagonal-Fisher hand sensitivity.

No optimizer is constructed and no model tensor is modified or saved. The
probe consumes cached, identically ordered supervision batches so every block
is evaluated on exactly the same evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from s2hand.loader import (
    NAMES, TRAINING_WARM_START, load_model, load_training_warm_start,
)
from s2hand.losses import hand_heatmaps_from_output, s2hand_objective

REQUIRED_BATCH_KEYS = (
    "images", "target_heatmaps", "target_weights", "gt_uv", "visibility"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_blocks(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    selected = [int(item) for item in re.split(r"[,;:]", value) if item.strip()]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"Invalid block list: {value!r}")
    if min(selected) < 0 or max(selected) >= count:
        raise ValueError(f"Blocks must be in [0,{count - 1}], got {selected}")
    return selected


def load_batches(paths: list[Path], max_batches: int | None) -> list[dict[str, torch.Tensor]]:
    batches = []
    for path in paths:
        tensors = load_file(str(path), device="cpu")
        missing = sorted(set(REQUIRED_BATCH_KEYS) - set(tensors))
        if missing:
            raise ValueError(f"{path} is missing tensors: {missing}")
        size = int(tensors["images"].shape[0])
        if any(int(tensors[key].shape[0]) != size for key in REQUIRED_BATCH_KEYS):
            raise ValueError(f"Batch dimension mismatch in {path}")
        for index in range(size):
            batches.append({key: tensors[key][index:index + 1] for key in REQUIRED_BATCH_KEYS})
            if max_batches is not None and len(batches) >= max_batches:
                return batches
    if not batches:
        raise ValueError("No Fisher batches were loaded")
    return batches


def measure_block(
    model: torch.nn.Module,
    block_index: int,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    bf16: bool,
) -> dict:
    block = model.backbone.blocks[block_index]
    block.requires_grad_(True)
    parameters = tuple(block.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    traces, relatives, losses = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for cached in batches:
        batch = {key: value.to(device) for key, value in cached.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda"):
            output = model(batch["images"].float())
            loss, _ = s2hand_objective(
                hand_heatmaps_from_output(output),
                batch["target_heatmaps"],
                batch["target_weights"],
                batch["gt_uv"],
                batch["visibility"],
            )
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=False, create_graph=False,
            allow_unused=False,
        )
        trace = torch.zeros((), device=device, dtype=torch.float64)
        relative = torch.zeros_like(trace)
        for parameter, gradient in zip(parameters, gradients):
            gradient = gradient.detach().float()
            trace += gradient.square().sum().double()
            relative += (gradient * parameter.detach().float()).square().sum().double()
        traces.append(float(trace.item()))
        relatives.append(float(relative.item()) / parameter_count)
        losses.append(float(loss.detach().item()))
        del output, loss, gradients, trace, relative, batch

    block.requires_grad_(False)
    for parameter in parameters:
        parameter.grad = None
    result = {
        "block": block_index,
        "parameters": parameter_count,
        "batches": len(batches),
        "fisher_trace_mean": float(np.mean(traces)),
        "fisher_trace_std": float(np.std(traces, ddof=1)) if len(traces) > 1 else 0.0,
        "fisher_per_parameter_mean": float(np.mean(traces) / parameter_count),
        "relative_fisher_per_parameter_mean": float(np.mean(relatives)),
        "loss_mean": float(np.mean(losses)),
    }
    if device.type == "cuda":
        result["peak_allocated_gib"] = float(
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        torch.cuda.empty_cache()
    return result


def write_results(output: Path, payload: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ranked = sorted(payload["records"],
                    key=lambda row: row["fisher_trace_mean"], reverse=True)
    payload["ranking"] = [row["block"] for row in ranked]
    (output / "fisher_scores.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "rank", "block", "parameters", "batches", "fisher_trace_mean",
        "fisher_trace_std", "fisher_per_parameter_mean",
        "relative_fisher_per_parameter_mean", "loss_mean", "peak_allocated_gib",
    )
    with (output / "fisher_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({"rank": rank, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=(TRAINING_WARM_START, *NAMES),
        default=TRAINING_WARM_START,
        help="Initialization to probe; the default exactly matches the paper probe",
    )
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--base-path")
    parser.add_argument("--delta-path")
    parser.add_argument("--manifest-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    device = torch.device(args.device)
    loader = load_training_warm_start if args.model == TRAINING_WARM_START else load_model
    loader_kwargs = dict(
        device=device, cache_dir=args.cache_dir, base_path=args.base_path,
        delta_path=args.delta_path, manifest_path=args.manifest_path,
        local_files_only=args.offline,
    )
    if loader is load_model:
        loader_kwargs["model_name"] = args.model
    model = loader(**loader_kwargs)
    model.requires_grad_(False)
    batches = load_batches(args.batch, args.max_batches)
    selected = parse_blocks(args.blocks, len(model.backbone.blocks))
    records = []
    for position, block in enumerate(selected, 1):
        print(f"[{position}/{len(selected)}] block {block}", flush=True)
        records.append(measure_block(
            model, block, batches, device, bf16=not args.fp32
        ))
        write_results(args.output, {
            "schema": "s2hand_diagonal_fisher_v1",
            "model": args.model,
            "no_optimizer_steps": True,
            "identical_cached_batches_per_block": True,
            "batch_files": [
                {"name": path.name, "sha256": sha256(path)}
                for path in args.batch
            ],
            "batches_per_block": len(batches),
            "records": records,
        })
    print(f"Saved Fisher ranking to {args.output}")


if __name__ == "__main__":
    main()
