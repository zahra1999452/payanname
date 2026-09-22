# H3 - Source-to-Target Domain Shift

H3 examines robustness when the model is evaluated on a target domain
that differs from the source training distribution.

The source model checkpoints remain unchanged.

No adaptation is performed for H3.

The target dataset must contain:

images/

and, for quantitative evaluation:

labels/

The same evaluation protocol and image size should be used.

Primary quantity:

Domain-shift degradation

D_shift = (M_source - M_target) / M_source

where M is a predefined metric such as F1 or recall.

Model comparison:

Friedman test on matched target observations when the data structure
allows it.

Pairwise post-hoc tests:

Wilcoxon signed-rank + Holm correction.