#!/usr/bin/env python3
"""Run S2HAND on images with person bounding boxes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from s2hand.inference import infer_people, read_boxes
from s2hand.loader import NAMES, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=NAMES, default="S2HAND-Pose-0.4B")
    parser.add_argument("--inputs", type=Path, required=True,
                        help="TSV rows: image_path x1 y1 x2 y2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA BF16 autocast")
    parser.add_argument("--cache-dir")
    parser.add_argument("--base-path")
    parser.add_argument("--delta-path")
    parser.add_argument("--manifest-path")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    model = load_model(
        args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        base_path=args.base_path,
        delta_path=args.delta_path,
        manifest_path=args.manifest_path,
        local_files_only=args.offline,
    )
    output = infer_people(
        model,
        read_boxes(args.inputs),
        device=args.device,
        batch_size=args.batch_size,
        bf16=not args.fp32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, model=np.asarray(args.model), **output)
    print(f"Saved {len(output['image_paths'])} predictions to {args.output}")


if __name__ == "__main__":
    main()
