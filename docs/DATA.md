# HInt training data

S2HAND uses HInt TRAIN only, with equal exposure from the NewDays and EPICK
subsets. Obtain HInt under its own terms and arrange the released annotations
as follows:

```text
HInt_annotation_partial/
  TRAIN_newdays_img/*.jpg
  TRAIN_newdays_img/*.json
  TRAIN_epick_img/*.jpg
  TRAIN_epick_img/*.json
  VAL_newdays_img/*.jpg
  VAL_newdays_img/*.json
  VAL_epick_img/*.jpg
  VAL_epick_img/*.json
```

`s2hand.hint_data.SourceBalancedHInt` implements the paper's deterministic
1:1 source exposure. Shuffling changes order but not the equal source counts.
Each annotation file must contain one HInt hand with 21
keypoints, existence flags, a side label (or side-suffixed filename), and an
`xyxy` hand box. The adapter preserves boxes that extend outside the image and
uses affine padding, matching the locked protocol.

Training uses 768x1024 hand-context crops, UDP 192x256 targets with sigma 6,
random context in [1.875, 3.125], center jitter 0.08, rotations up to 30
degrees, horizontal flip, photometric degradation, and local occlusion. The
standalone target encoder was checked element-by-element against the original
Sapiens2 research implementation: weights and all heatmap values were exactly
equal on an outside-frame randomized test.

H2SMV-FPV is an independent test set. It is not used for model training,
Fisher selection, checkpoint selection, or TailGuard tuning.
