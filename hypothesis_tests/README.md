# Scope of thesis evidence — 1 November 2026

The root README is authoritative. H1–H3 stored CSVs and historical evaluation
source are supplied in analysis_source_and_evidence.zip. The root H2 script,
per_image_degradation.csv, h2_inferential_results.csv and run_manifest.json allow
the six final H2 comparisons to be rechecked without a GPU or model weights.

H3: controlled synthetic transformation of the same 2462 test images. Historical
uav_coastal_domain_shift names do not identify an independent real dataset.
H3 inference uses D_mix; it must not be confused with absolute F1 drop D_abs.

H4: descriptive/unverified only. Source F1: 0.757359 -> 0.723990,
delta = -0.033369. No verified paired H4 p-value or confidence interval is supplied.
Improvement on an independent target domain remains untested. Full H4 raw pairs,
final weights and complete H3 target images/labels are unavailable.

Training from scratch was not rerun in this revision. Most supplementary source
retains historical paths and requires configuration plus missing data/weights.
See the root README for executable commands, evidence inventory and limitations.
