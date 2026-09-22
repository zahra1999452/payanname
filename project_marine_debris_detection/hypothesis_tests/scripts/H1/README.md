# H1 - Baseline Model Comparison

Research question:

Do YOLOv8, RT-DETR and RTMDet exhibit statistically significant
differences in baseline detection performance?

Evaluation data:

Final 13-class dataset TEST split.

Primary descriptive metrics:

- Precision
- Recall
- mAP50
- mAP50-95
- FPS
- Latency

Inferential analysis:

1. Match observations by test image.
2. Friedman test across the three models.
3. Wilcoxon signed-rank pairwise tests.
4. Holm correction for multiple comparisons.

Do not use independent-samples tests because the same test images
are evaluated by all three models.