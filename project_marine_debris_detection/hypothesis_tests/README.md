# Marine Debris Detection - Hypothesis Testing

This directory contains the statistical evaluation framework for Chapter 4.

## Models

1. YOLOv8
2. RT-DETR
3. RTMDet

## Dataset

Final 13-class dataset:

E:/rename/yolo_work_out/resrcher/YOLO_DATASETS/trashcan_trash_icra19_AUGMENTED_MINOR_ONLY_V6_2_REPAIRED_VALIDATED

Evaluation split:

TEST

Image size:

640 x 640

Number of classes:

13

## Hypotheses

### H1
There is a statistically significant difference between the baseline
performances of YOLOv8, RT-DETR and RTMDet.

### H2
The degradation caused by controlled image corruptions differs
significantly among YOLOv8, RT-DETR and RTMDet.

### H3
The degradation caused by source-to-target domain shift differs
significantly among YOLOv8, RT-DETR and RTMDet.

### H4
Limited target-domain adaptation significantly improves the performance
of the selected model compared with the non-adapted model.

## Statistical principle

Because the same test images are evaluated by all models, comparisons
should be treated as paired comparisons.

For three models:

Friedman test

For pairwise post-hoc comparisons:

Wilcoxon signed-rank test

Multiple comparisons:

Holm correction

The primary inferential unit is the matched test image.

Aggregate mAP50 and mAP50-95 are retained for reporting and descriptive
comparison. Inferential testing should preferably use per-image matched
metrics rather than treating one aggregate mAP value per model as a
sample.

## Important

This project generator does NOT execute experiments.

The generated scripts must be run later, after checkpoint paths and
target-domain paths have been verified.