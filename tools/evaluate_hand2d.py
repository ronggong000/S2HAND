#!/usr/bin/env python3
"""Evaluate aligned hand2d.v1 NPZ files with the S2HAND paper protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s2hand.evaluation import hand2d_metrics, side_swap_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side-swap", action="store_true")
    args = parser.parse_args()
    with np.load(args.ground_truth, allow_pickle=False) as ground_truth, \
         np.load(args.prediction, allow_pickle=False) as prediction:
        metrics = hand2d_metrics(ground_truth, prediction)
        if args.side_swap:
            metrics.update(side_swap_metrics(ground_truth, prediction))
        payload = {
            "schema": "s2hand_hand2d_metrics_v1",
            "dataset": str(ground_truth["dataset"]),
            "split": str(ground_truth["split"]),
            "model": str(prediction["model_name"]),
            "input_protocol": str(prediction["input_protocol"]),
            "normalization": "maximum side of the tight GT hand-keypoint box",
            "misses": "failures at every PCK/AUC threshold and in tail rate",
            "metrics": metrics,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
