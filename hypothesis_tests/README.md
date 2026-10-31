# Marine Debris Detection - Hypothesis Testing
This directory contains the statistical evaluation framework for Chapter 4.

## Models
1. YOLOv8n  2. RT-DETR-L  3. RTMDet-Tiny

## Dataset
Final 13-class dataset.
Evaluation split: TEST
Image size: 640 × 640
Number of classes: 13

## Hypotheses
### H1
There is a statistically significant difference between the baseline performances of YOLOv8n, RT-DETR-L and RTMDet-Tiny.

### H2
The absolute degradation (D = clean_F1 − corrupted_F1) caused by controlled image corruptions differs significantly among the three models.

### H3
The degradation caused by **controlled synthetic domain shift** differs significantly among the three models.  
**Note:** The target domain is a synthetic transformation of the same test images. No real UAV or coastal dataset was used.

### H4
Descriptive comparison of performance before and after limited target-domain adaptation on the selected model.  
**Note:** No statistical hypothesis test (Wilcoxon or CI) is performed on target data because paired target labels are unavailable.

## Statistical principle
Because the same test images are evaluated by all models, comparisons are treated as paired comparisons.
- Three models → Friedman test
- Pairwise post-hoc → Wilcoxon signed-rank test + Holm correction
- Primary unit: per-image matched metrics

## Important
This project does NOT execute training from scratch.  
The scripts are for statistical analysis only.
