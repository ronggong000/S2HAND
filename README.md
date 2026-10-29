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

## Installation

Python 3.12 and PyTorch 2.7 or newer are recommended.

```bash
git clone https://github.com/ronggong000/S2HAND.git
cd S2HAND
pip install -r requirements.txt
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

This initial loader exposes the verified heatmap forward pass. A full image
pipeline additionally requires person detection, the Sapiens2 affine crop, UDP
coordinate decoding, and optional flip testing. Those entry points are being
cleaned for release rather than copied directly from the cluster experiment
tree.

## Reproducibility status

- All five deltas have manifests with their exact official base, tensor list,
  byte size, and SHA-256.
- Both 5B deltas were rebased and replayed against the public Sapiens2-5B base.
- Routed-head behavior was checked against the research implementation.
- The 0.4B full backbone and head passed a bitwise CPU smoke test on a reduced
  input.
- Fresh full-resolution replay for all smaller variants remains on the release
  checklist; see [VALIDATION.md](VALIDATION.md).

## License

S2HAND is derived from Sapiens2. The code and model artifacts in this repository
are distributed under the included [Sapiens2 License](LICENSE.md), not MIT.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external components.
