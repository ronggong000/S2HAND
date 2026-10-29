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

## S2HAND-MT

Start from the official Sapiens2 pose checkpoint. Copy the complete three-layer
1x1 pose decoder into an adapted route and a frozen public route. Train the
adapted decoder with the objective above. At inference, take only the 40
non-wrist finger heatmaps from the adapted route and take all other 268
heatmaps from the frozen public route. The shared Transformer, deconvolution
stack, auxiliary task heads, wrists, body, feet, and face routes remain public.

## S2HAND-Pose

Continue from S2HAND-MT. Train the same adapted hand decoder and exactly three
Transformer blocks at 0.1 times the decoder learning rate. Released block
indices are listed in `release_manifest.json`. The 5B model uses zero-based
blocks 38--40. Blocks were chosen by diagonal-Fisher hand-loss sensitivity;
the probe takes no optimizer step.

The clean repository currently exposes the exact objective, model
reconstruction, protected routing, inference, and validation utilities. The
dataset adapter and end-to-end optimizer launcher are being separated from the
original Sapiens2 research checkout; until that lands, do not treat this page
as a one-command claim of retraining the paper checkpoints from scratch.
