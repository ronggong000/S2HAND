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

These tests verify loading, architecture and routing. Full-resolution CUDA
inference and a fresh full-dataset paper-metric replay were not run on this CPU
host. The script returns heatmaps; the caller supplies the corresponding person
crop preprocessing and UDP coordinate decoder for paper-protocol predictions.
