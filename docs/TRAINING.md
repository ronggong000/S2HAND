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
currently exposes the exact objective, model reconstruction, protected
routing, inference, and validation utilities. The dataset adapter and
end-to-end optimizer launcher are being separated from the original Sapiens2
research checkout; until that lands, do not treat this page as a one-command
claim of retraining the paper checkpoints from scratch.
