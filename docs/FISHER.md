# Block-wise Fisher probe

S2HAND-Pose selects three Transformer blocks using diagonal-Fisher sensitivity
to the frozen S2HAND hand objective. The probe does not construct an optimizer,
take an update, or modify a checkpoint. Each block receives the same cached
supervision batches.

## Cached batch format

Each input is a safetensors file containing:

| Tensor | Shape | Meaning |
|---|---:|---|
| `images` | `N x 3 x 1024 x 768` | RGB, ImageNet-normalized person crops. |
| `target_heatmaps` | `N x 42 x 256 x 192` | UDP Gaussian hand targets. |
| `target_weights` | `N x 42` | Per-joint heatmap weights. |
| `gt_uv` | `N x 2 x 21 x 2` | Input-crop hand coordinates. |
| `visibility` | `N x 2 x 21` | Positive for annotated joints. |

The batch files themselves are dataset-derived and are not redistributed.
They can be generated after obtaining HInt under its own terms.

## Run

```bash
python tools/probe_fisher.py \
  --model S2HAND-MT-5B \
  --batch fisher_batch_00.safetensors \
  --batch fisher_batch_01.safetensors \
  --max-batches 16 \
  --blocks all \
  --output fisher_5b \
  --device cuda
```

The primary ranking is the mean trace of the empirical diagonal Fisher,
implemented as the sum of squared gradients of the complete hand loss. The
CSV also reports Fisher per parameter and relative Fisher per parameter. The
paper's released 5B selection is zero-based blocks 38, 39, and 40.
