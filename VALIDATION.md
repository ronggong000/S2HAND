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

The standalone image pipeline is also covered by unit tests for:

- 1.25x bbox padding and 3:4 aspect-ratio correction;
- UDP affine mapping at the crop center;
- RGB/ImageNet input tensor shape and finite normalization; and
- 308-channel UDP/DARK decoding back to original-image coordinates.

These tests verify loading, architecture, routing, preprocessing, and decoding.
Full-resolution CUDA inference and a fresh full-dataset paper-metric replay are
still tracked separately and are not implied by the CPU unit tests.
