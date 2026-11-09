# Pre-registration stage 1, amendment 1
Date: 2026-11-10. Made after stage 1 was frozen and before any model was trained or any outcome was computed.
Reason: the H5 prediction is one-sided ("metaphor-based searchers do not outperform classical ones"), but section 8 specified two-sided equivalence (TOST). A clearly worse result for the metaphor-based searchers would fail TOST while matching the prediction.
Change: the confirmatory test for H5 is the upper arm of the TOST only: a one-sided Wilcoxon signed-rank test that the per-dataset difference (mean outer AUC of GWO and WOA minus mean outer AUC of GA, binary PSO and DE) is below +0.01. H5 is supported if this test rejects. The full two-sided TOST (margin 0.01) and a two-sided Wilcoxon test are reported as secondary results. Wording rule: "not better than classical searchers by more than 0.01 AUC" if supported; "equivalent" only if the full TOST also succeeds.
Nothing else in stage 1 changes.
