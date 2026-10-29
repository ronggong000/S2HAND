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

On 2026-10-30, the standalone Pose-5B trainer passed a real one-step L40S/BF16
mechanics test from the pinned training warm-start. It completed canonical
NewDays/EPICK validation, a forward/backward update of blocks 38--40 and the
adapted decoder, gradient clipping, best/final checkpoint serialization, and
manifest hashing. The 1,068,545,694-byte final delta contained exactly the 60
expected tensors. A separate process then loaded that generated delta over the
pinned official base with `load_training_checkpoint` and reproduced finite
canonical validation metrics. The two-hand diagnostic slice is a mechanics
test only and is not reported as a paper benchmark result.

The released S2HAND-MT-5B route was also checked with its verified final delta
and a real HInt validation image. All 268 protected heatmaps were bitwise equal
to the frozen public decoder route (maximum absolute difference `0.0`), while
the 40 adapted finger heatmaps differed from that route. The ordinary model
forward was bitwise equal to an independently assembled explicit 40/268 route.

The clean checkout also replayed the complete 1,175-hand HInt NewDays/EPICK
validation split for the released S2HAND-Pose-5B delta. The catastrophic-tail
rate matched the original research record exactly (`0.0144680851`). Selection
AUC was `0.7656657143`, versus `0.7656003214` in the original record (absolute
difference `0.0000653929`), and combined NME differed by `0.0000097603`.
This is a near-exact protocol replay rather than a bitwise claim: the largest
single metric difference was `0.0019462` for a discontinuous occluded-joint PCK
threshold count.

Finally, after an editable clean-checkout installation, the documented
inference, hand-evaluation, Fisher-probe, and training CLIs all parsed their
published command-line interfaces successfully. All relative Markdown links in
the README, validation record, and `docs/` resolve to existing files.
