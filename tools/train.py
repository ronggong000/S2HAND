#!/usr/bin/env python3
"""Retrain S2HAND-MT-5B or S2HAND-Pose-5B from the released warm-start."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from s2hand.hint_data import HIntSupervisedDataset, SourceBalancedHInt
from s2hand.loader import load_training_warm_start
from s2hand.training import (
    ValidationSelector, configure_trainable, enable_activation_checkpointing,
    optimizer_for, save_checkpoint, training_step, validate_hint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_train_loader(root: Path, *, seed: int, workers: int,
                      length: int = 24_000) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = SourceBalancedHInt(
        root, split="TRAIN", augment=True, length=length, seed=seed
    )
    return DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
        worker_init_fn=seed_worker, generator=generator,
    )


def make_val_loader(root: Path, *, workers: int,
                    max_hands_per_source: int | None = None) -> DataLoader:
    sources = [
        HIntSupervisedDataset(root, "VAL", "newdays", augment=False),
        HIntSupervisedDataset(root, "VAL", "epick", augment=False),
    ]
    if max_hands_per_source is not None:
        if max_hands_per_source < 1:
            raise ValueError("max_hands_per_source must be positive")
        sources = [Subset(dataset, range(min(max_hands_per_source, len(dataset))))
                   for dataset in sources]
    dataset = ConcatDataset(sources)
    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
    )


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def repeat_loader(loader):
    """Repeat epochs without retaining batches as ``itertools.cycle`` does."""
    while True:
        yield from loader


def run_stage(model, loader, optimizer, *, steps: int, start_step: int,
              device: torch.device, bf16: bool, output: Path,
              log_interval: int, val_loader=None, val_interval: int = 3000,
              selector: ValidationSelector | None = None, variant: str,
              checkpoint_metadata: dict) -> int:
    model.train()
    iterator = repeat_loader(loader)
    last = start_step
    for local_step in range(1, steps + 1):
        last = start_step + local_step
        terms = training_step(
            model, next(iterator), optimizer, device, bf16=bf16, grad_clip=1.0
        )
        if local_step == 1 or last % log_interval == 0:
            row = {"kind": "train", "step": last, **terms}
            append_jsonl(output / "history.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if val_loader is not None and last % val_interval == 0:
            metrics = validate_hint(model, val_loader, device, bf16=bf16)
            row = {"kind": "validation", "step": last, **metrics}
            append_jsonl(output / "history.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if selector is None:
                raise RuntimeError("Validation requires a checkpoint selector")
            if selector.consider(metrics, last):
                save_checkpoint(model, output / "best_delta.safetensors", {
                    "variant": variant, "step": last, "selection": metrics,
                    **checkpoint_metadata,
                })
                (output / "best_metrics.json").write_text(
                    json.dumps(row, indent=2) + "\n", encoding="utf-8"
                )
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("mt", "pose"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=6000,
                        help="MT only; set smaller solely for smoke tests")
    parser.add_argument("--adaptation-steps", type=int, default=12000,
                        help="MT adaptation steps")
    parser.add_argument("--pose-steps", type=int, default=18000)
    parser.add_argument("--val-interval", type=int, default=3000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-val-hands-per-source", type=int,
                        help="Diagnostic smoke only; omit for paper selection")
    parser.add_argument("--blocks", default="38,39,40")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--base-path")
    parser.add_argument("--delta-path")
    parser.add_argument("--manifest-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.val_interval, args.log_interval) < 0:
        parser.error("workers and intervals must be non-negative")
    if args.val_interval < 1 or args.log_interval < 1:
        parser.error("validation and log intervals must be positive")
    blocks = tuple(int(value) for value in args.blocks.split(",") if value.strip())
    if args.variant == "mt" and (args.warmup_steps < 0 or args.adaptation_steps < 1):
        parser.error("MT requires non-negative warmup and positive adaptation steps")
    if args.variant == "pose" and args.pose_steps < 1:
        parser.error("Pose steps must be positive")

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, warm_manifest = load_training_warm_start(
        device=device, cache_dir=args.cache_dir, base_path=args.base_path,
        delta_path=args.delta_path, manifest_path=args.manifest_path,
        local_files_only=args.offline, return_manifest=True,
    )
    warm_reference = {
        "name": warm_manifest["name"],
        "sha256": warm_manifest["sha256"],
        "pinned_release_revision": "e37456b131139e186d9cb5552beb3e99820004d4",
    }
    checkpoint_metadata = {
        "name": f"S2HAND-{'MT' if args.variant == 'mt' else 'Pose'}-5B-custom",
        "scale": "5b",
        "official_base": warm_manifest["official_base"],
        "training_warm_start": warm_reference,
        "validation_scope": (
            "full_hint_val" if args.max_val_hands_per_source is None
            else f"diagnostic_first_{args.max_val_hands_per_source}_per_source"
        ),
    }
    configure_trainable(model, args.variant, blocks)
    if args.variant == "pose" and not args.no_checkpointing:
        enable_activation_checkpointing(model, min(blocks))
    train_loader = make_train_loader(
        args.data_root, seed=args.seed, workers=args.workers
    )
    val_loader = make_val_loader(
        args.data_root, workers=args.workers,
        max_hands_per_source=args.max_val_hands_per_source,
    )
    bf16 = not args.fp32

    if args.variant == "mt":
        optimizer = optimizer_for(model, "mt")
        step = 0
        if args.warmup_steps:
            step = run_stage(
                model, train_loader, optimizer, steps=args.warmup_steps,
                start_step=0, device=device, bf16=bf16, output=args.output,
                log_interval=args.log_interval, variant="mt",
                checkpoint_metadata=checkpoint_metadata,
            )
            save_checkpoint(model, args.output / "warmup_delta.safetensors", {
                "variant": "mt", "step": step, "stage": "warmup",
                **checkpoint_metadata,
            })
        # Deliberate optimizer reset between the 6k and 12k stages.
        optimizer = optimizer_for(model, "mt")
    else:
        optimizer = optimizer_for(model, "pose")
        step = 0

    baseline = validate_hint(model, val_loader, device, bf16=bf16)
    selector = ValidationSelector(
        baseline["selection_AUC"], baseline["catastrophic_rate"], best_step=step
    )
    baseline_row = {"kind": "validation", "stage": "baseline",
                    "step": step, **baseline}
    append_jsonl(args.output / "history.jsonl", baseline_row)
    (args.output / "baseline_metrics.json").write_text(
        json.dumps(baseline_row, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(model, args.output / "best_delta.safetensors", {
        "variant": args.variant, "step": step, "selection": baseline,
        "baseline_selected": True, **checkpoint_metadata,
    })
    (args.output / "best_metrics.json").write_text(
        json.dumps(baseline_row, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(baseline_row, sort_keys=True), flush=True)

    remaining = args.adaptation_steps if args.variant == "mt" else args.pose_steps
    step = run_stage(
        model, train_loader, optimizer, steps=remaining, start_step=step,
        device=device, bf16=bf16, output=args.output,
        log_interval=args.log_interval, val_loader=val_loader,
        val_interval=args.val_interval, selector=selector,
        variant=args.variant, checkpoint_metadata=checkpoint_metadata,
    )
    save_checkpoint(model, args.output / "final_delta.safetensors", {
        "variant": args.variant, "step": step, "stage": "final",
        "selected_best_step": selector.best_step,
        "selected_best_auc": selector.best_auc,
        **checkpoint_metadata,
    })
    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "steps": step,
        "selected_best_step": selector.best_step,
        "selected_best_auc": selector.best_auc,
        "baseline_auc": selector.baseline_auc,
        "baseline_catastrophic_rate": selector.baseline_tail,
        "tail_gate_max_increase": selector.max_tail_increase,
        "blocks": list(blocks) if args.variant == "pose" else [],
        "optimizer_reset_after_warmup": args.variant == "mt",
        "max_val_hands_per_source": args.max_val_hands_per_source,
        "paper_selection_protocol": args.max_val_hands_per_source is None,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
