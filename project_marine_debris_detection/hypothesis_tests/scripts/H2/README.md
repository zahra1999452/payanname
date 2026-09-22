# H2 - Controlled Image Corruption Robustness

The same TEST images are transformed with controlled image corruptions.

Suggested conditions:

- brightness
- contrast
- Gaussian noise
- Gaussian blur
- motion blur
- JPEG compression

For each model and corruption:

Baseline performance is compared with corrupted performance.

Define relative degradation:

D = (M_clean - M_corrupted) / M_clean

where M can be F1, recall, mAP50, or another predefined metric.

The inferential comparison should focus on matched per-image degradation.

Friedman test:
compare degradation across the three models.

Post-hoc:
Wilcoxon signed-rank + Holm correction.