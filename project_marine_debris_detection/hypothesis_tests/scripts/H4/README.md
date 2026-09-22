# H4 - Limited Target-Domain Adaptation

H4 tests whether limited target-domain adaptation improves the
performance of the selected model.

Comparison:

Non-adapted model
versus
Adapted model

The target test set must remain identical between the two conditions.

The target test set must NOT be used during adaptation/training.

Recommended split:

target_train
target_val
target_test

The target_test set is used only for final evaluation.

Inferential test:

Wilcoxon signed-rank test on matched per-image metrics.

If only one aggregate mAP value is available for each condition,
an inferential paired test cannot be justified.