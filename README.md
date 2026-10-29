# S2HAND

**Accuracy-Preservation Trade-Offs for Hand Specialization in Whole-Body Pose
Models**

Singyu Muk · Xuanren Song · Kalin Stefanov

S2HAND specializes the hand predictions of Sapiens2 while retaining its native
308-keypoint whole-body pose interface. It avoids combining an independent hand
model with a body model, which otherwise introduces another detector, coordinate
alignment, and wrist-stitching decisions.

> This repository is being prepared privately while the accompanying manuscript
> is under review. Interfaces and documentation may change before public release.

## Variants

- **S2HAND-MT-5B** adds a lightweight copied pose decoder. The adapted route
  supplies the 40 non-wrist finger heatmaps; the public route supplies the other
  268 body, foot, face, and wrist heatmaps. The shared Transformer and auxiliary
  task heads remain unchanged.
- **S2HAND-Pose** also updates three Fisher-selected Transformer blocks and the
  hand decoder, trading exact preservation for stronger hand recovery. We provide
  0.4B, 0.8B, 1B, and 5B variants.
- **TailGuard** is an optional confidence-triggered re-cropping policy. It is not
  another pose model and does not change the checkpoint.

## Checkpoints

S2HAND checkpoints are distributed as trainable deltas in the private
[`mage0001/S2HAND`](https://huggingface.co/mage0001/S2HAND) repository. The
loader retrieves the exact official Sapiens2 base, verifies both SHA-256 hashes,
applies the delta, and reconstructs the protected decoder route.

| Model | Updated Transformer blocks (zero-based) | Delta size |
|---|---:|---:|
| S2HAND-MT-5B | none | 3.31 MiB |
| S2HAND-Pose-0.4B | 16--18 | 195.50 MiB |
| S2HAND-Pose-0.8B | 17--19 | 284.78 MiB |
| S2HAND-Pose-1B | 28--30 | 410.58 MiB |
| S2HAND-Pose-5B | 38--40 | 1019.04 MiB |

Official base checkpoints are not duplicated in this repository.
The pinned base revisions, delta hashes, Hugging Face revisions, and TailGuard
configuration are collected in [release_manifest.json](release_manifest.json).
The hand objective, data roles, routing, and Fisher-selected block policy are
documented in [docs/TRAINING.md](docs/TRAINING.md).
The two final 5B variants branch independently from a released compact common
training warm-start; that artifact is clearly marked non-deployment.
The no-update block-selection probe and its cached-batch schema are documented
in [docs/FISHER.md](docs/FISHER.md).
The aligned 2D hand schemas and strict paper metrics are documented in
[docs/EVALUATION.md](docs/EVALUATION.md).
The standalone HInt directory contract and source-balanced training adapter
are documented in [docs/DATA.md](docs/DATA.md).

## Installation

Python 3.12 and PyTorch 2.7 or newer are recommended.

```bash
git clone https://github.com/ronggong000/S2HAND.git
cd S2HAND
pip install -e .
hf auth login  # required while the model repository is private
```

## Load a model

```python
import torch
from s2hand import load_model

model = load_model("S2HAND-Pose-0.4B", device="cuda")

# Person crops are RGB, Bx3x1024x768, and ImageNet-normalized.
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    heatmaps = model(normalized_person_crops)  # Bx308x256x192
```

## Run on images

S2HAND is a top-down estimator and therefore takes a person box. Create a TSV
with one row per person:

```text
images/example.jpg  x1  y1  x2  y2
```

Then run the same 1.25x padded 3:4 crop, UDP affine transform, ImageNet
normalization, and UDP/DARK decoder used in our evaluation:

```bash
python tools/infer.py \
  --model S2HAND-Pose-0.4B \
  --inputs people.tsv \
  --output predictions.npz \
  --device cuda \
  --batch-size 4
```

The output contains all 308 keypoints in original-image coordinates and their
heatmap scores. Person detection is intentionally external so detector and crop
protocols remain explicit. This command performs D0 inference without horizontal
flip testing, matching the paper's default evaluation.

## Reproducibility status

- All five deltas have manifests with their exact official base, tensor list,
  byte size, and SHA-256.
- Both 5B deltas were rebased and replayed against the public Sapiens2-5B base.
- Routed-head behavior was checked against the research implementation.
- The 0.4B full backbone and head passed a bitwise CPU smoke test on a reduced
  input.
- The 0.4B, 0.8B, and 1B variants passed fresh full-resolution A100/BF16
  deterministic replay; output hashes are recorded in their manifests. See
  [VALIDATION.md](VALIDATION.md).

## License

S2HAND is derived from Sapiens2. The code and model artifacts in this repository
are distributed under the included [Sapiens2 License](LICENSE.md), not MIT.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external components.
