# TailGuard inference

TailGuard is a frozen routing policy, not another learned pose model. It first
runs the selected S2HAND model on ordinary person detections. For each hand, it
computes the mean of the 21 native heatmap scores. If that value is at most
`0.5107561469078065` and a same-side hand detection exists, the same model is
run once more on a square box expanded by `2.0`. Otherwise the native result is
returned. Missing hand detections never trigger a crop.

On a trigger, the conditional crop supplies all 21 hand joints, including its
wrist. Combining a native wrist with recropped fingers would mix coordinate
frames and can create an invalid wrist-to-finger bone. All outputs are restored
to original-image pixels before replacement.

## Detection cache

Both inputs to `tools/tailguard_infer.py` are flattened `detection.v1` NPZ
files. Required fields are:

| Field | Shape | Meaning |
|---|---:|---|
| `schema_version` | scalar string | Must be `detection.v1`. |
| `image_path` | `N` | Unique image paths. |
| `offsets` | `N+1` | Start/end offsets into flattened detections. |
| `boxes_xyxy` | `M x 4` | Detection boxes in original-image pixels. |
| `scores` | `M` | Detector scores. |

The hand cache additionally requires `hand_side` with `0=left` and `1=right`.
The highest-scoring same-side box is used. The person cache may include any
number of people; for each side, TailGuard selects the person prediction with
the highest mean score for that hand. Images absent from either cache are not
processed.

## Output

The `tailguard.prediction.v1` archive contains paired native and routed
21-joint predictions, confidence scores, detector and re-crop boxes, trigger
flags, and native/re-crop forward counts. This pairing makes latency and
accuracy auditing possible without a second inference run.

The released threshold and context are fixed in
[`configs/tailguard_v1.json`](../configs/tailguard_v1.json). Do not retune them
on a test split.
