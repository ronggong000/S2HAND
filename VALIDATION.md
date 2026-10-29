# Loader validation

Before upload, all five models passed:

- Exact base/delta SHA256 verification against their manifests.
- Strict loading of every official base tensor with no missing/meta parameters.
- Exact replacement tensor values in the original runtime dtypes.
- Bitwise-equal routed-head outputs against the original research head implementation
  on deterministic feature tensors, including protection of both wrist channels.
- Correct BF16 RoPE runtime buffer semantics, including the rebased 5B override.

Additionally, the 0.4B model's full backbone-and-head forward pass matched the
original research implementation bitwise on a reduced-resolution 64x48 CPU input.

On 2026-10-29, S2HAND-Pose-0.4B, 0.8B, and 1B each passed a fresh
full-resolution A100/BF16 reconstruction test. Every run verified the official
base and delta hashes, produced finite `1x308x256x192` heatmaps, and was
bitwise-equal across two forwards of the same deterministic spatial-ramp input.
The output hashes are recorded in the per-model manifests.

The standalone image pipeline is also covered by unit tests for:

- 1.25x bbox padding and 3:4 aspect-ratio correction;
- UDP affine mapping at the crop center;
- RGB/ImageNet input tensor shape and finite normalization; and
- 308-channel UDP/DARK decoding back to original-image coordinates.

The protocol-locked evaluator was also compared with the NewDays paper closure
aggregate for S2HAND-Pose. Hand count, evaluated-joint count, recall,
PCK@.05/.10/.15/.20, AUC@.20, severe-tail rate, and finite-hand NME all matched
exactly.

The standalone training adapter's UDP targets were compared element-by-element
with the original research encoder on randomized in-frame and outside-frame
points; target weights and all heatmap values were exactly equal. Unit tests
also cover the 42-channel layout, visibility handling, and protected-route
gradient stop.
