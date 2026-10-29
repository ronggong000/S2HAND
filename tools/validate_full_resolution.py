#!/usr/bin/env python3
"""Run a deterministic full-resolution S2HAND reconstruction smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from s2hand.loader import NAMES, load_model


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--delta-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(2027)
    model = load_model(
        args.model,
        device=args.device,
        base_path=args.base_path,
        delta_path=args.delta_path,
        manifest_path=args.manifest_path,
        local_files_only=True,
    )
    # A deterministic spatial ramp exercises every patch without private data.
    y = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32).view(1, 1, 1024, 1)
    x = torch.linspace(-1.0, 1.0, 768, dtype=torch.float32).view(1, 1, 1, 768)
    images = torch.cat((x.expand(1, 1, 1024, 768),
                        y.expand(1, 1, 1024, 768),
                        (x + y) * 0.5), dim=1).to(args.device)
    with torch.inference_mode(), torch.autocast(
        torch.device(args.device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(args.device).type == "cuda",
    ):
        first = model(images).float()
        second = model(images).float()
    expected_shape = (1, 308, 256, 192)
    passed = (
        tuple(first.shape) == expected_shape
        and bool(torch.isfinite(first).all())
        and bool(torch.equal(first, second))
    )
    payload = {
        "model": args.model,
        "device": str(args.device),
        "input": "deterministic_spatial_ramp_v1",
        "autocast": "bfloat16" if torch.device(args.device).type == "cuda" else "disabled",
        "output_shape": list(first.shape),
        "expected_shape": list(expected_shape),
        "finite": bool(torch.isfinite(first).all()),
        "repeat_bitwise_equal": bool(torch.equal(first, second)),
        "output_float32_sha256": tensor_sha256(first),
        "output_min": float(first.min()),
        "output_max": float(first.max()),
        "output_mean": float(first.mean()),
        "output_std": float(first.std()),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
