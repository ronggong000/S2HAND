# 2D hand evaluation

The paper uses aligned `hand2d.v1` NPZ files. Coordinates are always in
original-image pixels and each row represents one labeled hand. Evaluation is
strict: a missed hand or non-finite predicted joint is incorrect at every PCK
threshold and lowers AUC. The severe-tail rate is the fraction of hands whose
mean normalized joint error exceeds 0.20; missed hands are included as severe
failures.

Errors are normalized by the maximum side of the tight ground-truth hand
keypoint box. AUC@0.20 integrates PCK over 201 uniformly spaced thresholds
from 0 to 0.20 and divides by 0.20.

## Ground-truth fields

| Field | Shape | Meaning |
|---|---:|---|
| `schema_version` | scalar string | `hand2d.v1` or `hand2d3d.v1`. |
| `sample_id` | `N` | Stable, ordered sample identifiers. |
| `keypoints_xy` | `N x 21 x 2` | Ground-truth image coordinates. |
| `visibility` | `N x 21` | Positive for evaluated joints. |
| `hand_bbox_xyxy` | `N x 4` | Tight GT hand-keypoint box. |
| `dataset`, `split` | scalar string | Dataset identity and official split. |

`image_path` and `hand_side` are additionally required for the optional
side-swap metric.

## Prediction fields

| Field | Shape | Meaning |
|---|---:|---|
| `schema_version` | scalar string | `hand2d.v1`. |
| `sample_id` | `N` | Must exactly match GT order. |
| `coordinate_system` | scalar string | `original_image_pixels`. |
| `keypoints_xy` | `N x 21 x 2` | Predicted coordinates. |
| `detected` | `N` | Whether the native/shared pipeline detected the hand. |
| `model_name`, `input_protocol` | scalar string | Provenance recorded in output. |

## Run

```bash
python tools/evaluate_hand2d.py \
  --ground-truth canonical_test.npz \
  --prediction predictions_test.npz \
  --output metrics_test.json \
  --side-swap
```

Native Pipeline and controlled-crop predictions must not be combined in one
ranking. `input_protocol` is therefore copied into every result file.
