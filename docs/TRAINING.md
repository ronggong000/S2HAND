# Training protocol

S2HAND uses the public 308-channel Sapiens2 pose interface. The hand channels
are indices 21--62: two 21-joint hands. Wrist channels 41 and 62 stay on the
public route in S2HAND-MT, leaving 40 adapted finger heatmaps.

## Objective

The training objective is implemented in `s2hand.losses.s2hand_objective`:

1. target-weighted heatmap MSE over the 42 hand channels;
2. Smooth-L1 error on differentiable soft-argmax hand coordinates, normalized
   by each hand's visible-joint span; and
3. Smooth-L1 error on the 20 directed hand bone vectors, with the same scale
   normalization.

The coordinate and topology weights are respectively
`0.0030615822269171302` and `0.0016362079765548576`. They were calibrated on 16
training batches so each auxiliary term contributed 0.1 times the heatmap-loss
gradient norm. Deployment still uses UDP/DARK, not soft-argmax.

## Data and selection

- Training data: HInt TRAIN, sampled 1:1 from NewDays and EPICK.
- H2SMV-FPV is not used for training, validation, Fisher selection, checkpoint
  selection, or TailGuard tuning.
- Checkpoint selection: HInt validation NewDays/EPICK macro AUC@0.20, with the
  preregistered catastrophic-tail gate.
- Random seed: 42 for the released models.

## Common warm-start

Both final 5B variants start from the same source-balanced HInt warm-start,
which is reconstructed by applying the non-deployment
`S2HAND-Training-WarmStart-5B` artifact to the pinned official 5B base. The
warm-start was trained for 12,000 steps with 1:1 NewDays:EPICK sampling. It
updates the 42 hand projection rows and the complete three-layer 1x1 decoder;
a frozen public checkpoint supplies the protected route. Its objective is
target-weighted hand heatmap MSE plus frozen-public teacher distillation.

The released MT and Pose models then branch independently from this common
initialization. The final Pose model is not a continuation of the final MT
checkpoint.

## S2HAND-MT

Start from the common warm-start. Copy the complete three-layer 1x1 pose
decoder into an adapted route and retain the official decoder as a frozen
public route. Run a 6,000-step decoder warmup followed by 12,000 steps with the
three-term objective above, using decoder learning rate `2e-5`. At inference,
take only the 40
non-wrist finger heatmaps from the adapted route and take all other 268
heatmaps from the frozen public route. The shared Transformer, deconvolution
stack, auxiliary task heads, wrists, body, feet, and face routes remain public.

## S2HAND-Pose

Start independently from the same common warm-start. Train the complete
adapted hand decoder for 18,000 steps at learning rate `2e-5` and exactly three
Transformer blocks at learning rate `2e-6`. Released block
indices are listed in `release_manifest.json`. The 5B model uses zero-based
blocks 38--40. Blocks were chosen by diagonal-Fisher hand-loss sensitivity;
the probe takes no optimizer step.

The non-deployment warm-start is released so the lineage and Fisher probe do
not depend on an unpublished intermediate checkpoint. The clean repository
also exposes the exact data adapter, objective, optimizer groups, protected
routing, canonical HInt validation, tail-gated checkpoint selector, and
end-to-end launcher without an MMEngine or Slurm dependency.

## Run training

After arranging HInt as described in [DATA.md](DATA.md), retrain either 5B
variant with one CUDA GPU with at least 48 GiB memory. The mechanical training
smoke was validated on an L40S; an A100 is faster for the full run:

```bash
python tools/train.py \
  --variant mt \
  --data-root /path/to/HInt_annotation_partial \
  --output runs/s2hand_mt_5b \
  --device cuda

python tools/train.py \
  --variant pose \
  --data-root /path/to/HInt_annotation_partial \
  --output runs/s2hand_pose_5b \
  --device cuda \
  --blocks 38,39,40
```

Both commands load the pinned `S2HAND-Training-WarmStart-5B` artifact. MT runs
6,000 steps, saves the stage boundary, deliberately recreates AdamW, and then
runs the 12,000-step adaptation stage. Pose runs an independent 18,000-step
optimization. Pose activation checkpointing begins at the earliest selected
block; `--no-checkpointing` disables it for debugging.

Validation first establishes a pre-stage baseline. It then runs every 3,000
steps using the fixed 2.5x HInt crop, a 1.30x target-region heatmap mask, and
UDP/DARK decoding. `best_delta.safetensors` changes only if NewDays/EPICK macro
AUC@0.20 improves while the per-hand NME>0.20 rate stays within 0.005 of the
baseline. The baseline remains the selected checkpoint if no trained state
passes both conditions. `history.jsonl`, baseline/best metrics, the 6k MT
boundary, final delta, and SHA-256 manifests are written to the output folder.
Load a selected retrained checkpoint directly with:

```python
from s2hand import load_training_checkpoint
model = load_training_checkpoint("runs/s2hand_pose_5b/best_delta.safetensors",
                                 device="cuda")
```

The hand channels are gathered in COCO/HInt anatomical order before applying
the loss. Although the Sapiens2 hand channels occupy indices 21--62, their
native order is not anatomical and must not be replaced by a contiguous slice.
