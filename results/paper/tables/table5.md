**Table 5. Configurations at 2,000 evaluations.**

| Configuration | Mean outer AUC | Mean optimism gap | Mean subset size | Mean stability | CPU s per run, median [IQR] |
|---|---|---|---|---|---|
| Random search, C0 (no control) | 0.6627 | 0.1229 | 34.1 | 0.081 | 48.8 [41.3, 63.6] |
| Hill climbing, C0 (no control) | 0.6639 | 0.1805 | 71.7 | 0.189 | 42.4 [37.6, 50.0] |
| (1+1)-EA, C0 (no control) | 0.6646 | 0.1776 | 73.5 | 0.192 | 29.8 [25.3, 37.4] |
| GA, C0 (no control) | 0.6655 | 0.1805 | 75.6 | 0.203 | 44.9 [37.8, 58.7] |
| BPSO-S, C0 (no control) | 0.6645 | 0.1262 | 88.6 | 0.211 | 50.0 [42.4, 66.6] |
| BPSO-V, C0 (no control) | 0.6662 | 0.1403 | 86.7 | 0.213 | 50.0 [42.3, 66.5] |
| DE-S, C0 (no control) | 0.6648 | 0.1388 | 87.4 | 0.215 | 49.9 [42.3, 65.3] |
| DE-V, C0 (no control) | 0.6645 | 0.0903 | 92.3 | 0.212 | 50.8 [43.1, 66.9] |
| GWO-S, C0 (no control) | 0.6645 | 0.1168 | 90.2 | 0.212 | 49.9 [42.6, 66.6] |
| GWO-V, C0 (no control) | 0.6653 | 0.0951 | 92.4 | 0.214 | 50.7 [43.0, 67.2] |
| WOA-S, C0 (no control) | 0.6694 | 0.1302 | 30.7 | 0.078 | 34.8 [27.6, 45.5] |
| WOA-V, C0 (no control) | 0.6645 | 0.1032 | 91.6 | 0.213 | 47.1 [39.8, 60.6] |
| GA, C1 early stopping | 0.6613 | 0.0624 | 88.8 | 0.207 | 17.4 [13.3, 24.9] |
| BPSO-S, C1 early stopping | 0.6619 | 0.0582 | 90.9 | 0.210 | 17.9 [14.3, 23.2] |
| DE-S, C1 early stopping | 0.6606 | 0.0566 | 92.0 | 0.213 | 19.3 [14.7, 26.4] |
| GWO-S, C1 early stopping | 0.6597 | 0.0561 | 92.1 | 0.211 | 18.0 [13.8, 24.6] |
| GA, C2 variance penalty | 0.6623 | 0.1146 | 69.8 | 0.213 | 131.0 [110.2, 166.5] |
| BPSO-S, C2 variance penalty | 0.6609 | 0.0797 | 87.2 | 0.217 | 149.5 [126.1, 196.2] |
| DE-S, C2 variance penalty | 0.6614 | 0.0856 | 85.9 | 0.218 | 148.4 [125.6, 193.4] |
| GWO-S, C2 variance penalty | 0.6617 | 0.0736 | 89.5 | 0.215 | 149.7 [126.9, 196.4] |
| GA, C3 flatness | 0.6663 | 0.1152 | 90.6 | 0.217 | 49.5 [41.8, 64.8] |
| BPSO-S, C3 flatness | 0.6644 | 0.1064 | 91.0 | 0.213 | 50.0 [42.4, 65.3] |
| DE-S, C3 flatness | 0.6642 | 0.0944 | 92.5 | 0.214 | 50.6 [42.6, 65.6] |
| GWO-S, C3 flatness | 0.6648 | 0.0937 | 92.9 | 0.216 | 50.1 [42.5, 65.8] |
| Random search, C4 knee | 0.6601 | 0.0984 | 7.2 | 0.030 | 48.8 [41.3, 63.6] a |
| Hill climbing, C4 knee | 0.6642 | 0.1769 | 67.5 | 0.181 | 42.4 [37.6, 50.0] a |
| (1+1)-EA, C4 knee | 0.6644 | 0.1739 | 69.4 | 0.185 | 29.8 [25.3, 37.4] a |
| GA, C4 knee | 0.6658 | 0.1748 | 71.4 | 0.194 | 44.9 [37.8, 58.7] a |
| BPSO-S, C4 knee | 0.6645 | 0.1080 | 76.7 | 0.183 | 50.0 [42.4, 66.6] a |
| BPSO-V, C4 knee | 0.6662 | 0.1225 | 77.5 | 0.188 | 50.0 [42.3, 66.5] a |
| DE-S, C4 knee | 0.6623 | 0.1262 | 78.7 | 0.189 | 49.9 [42.3, 65.3] a |
| DE-V, C4 knee | 0.6607 | 0.0681 | 79.0 | 0.174 | 50.8 [43.1, 66.9] a |
| GWO-S, C4 knee | 0.6652 | 0.0981 | 80.6 | 0.187 | 49.9 [42.6, 66.6] a |
| GWO-V, C4 knee | 0.6620 | 0.0699 | 79.7 | 0.178 | 50.7 [43.0, 67.2] a |
| WOA-S, C4 knee | 0.6520 | 0.1001 | 5.5 | 0.040 | 34.8 [27.6, 45.5] a |
| WOA-V, C4 knee | 0.6632 | 0.0769 | 79.7 | 0.180 | 47.1 [39.8, 60.6] a |
| All K features | 0.6652 | 0.0189 | 190.5 | 0.583 (5 undef.) | 0.0 [0.0, 0.0] |
| L1 logistic regression | 0.6724 | 0.1030 | 25.8 | 0.294 | 0.5 [0.3, 1.7] |
| Elastic net | 0.6737 | 0.0844 | 46.4 | 0.301 | 17.4 [8.1, 36.8] |
| F-test top-k | 0.6750 | 0.0466 | 94.3 | 0.348 | 0.1 [0.1, 0.2] |
| mRMR top-k | 0.6708 | 0.0628 | 66.6 | 0.262 | 0.1 [0.1, 0.2] |
| Random-forest importance top-k | 0.6713 | 0.0598 | 67.3 | 0.256 | 2.9 [2.6, 3.5] |

At 2,000 evaluations; means over the 31 confirmation datasets of per-dataset means (outer folds x seeds). Stability: Nogueira et al. (2018), undefined values left out. CPU seconds: the method phase of each run (search for searchers, selection for baselines), single-threaded; median and interquartile range over all runs of the configuration, pooled over datasets, folds and seeds. a: shared with C0 - C4 adds no evaluations and is computed inside the C0 run, so its row shows the C0 run time. Sources: results/analysis/confirmation/tables (anytime.csv; per_dataset_auc_2000.csv and per_dataset_gap_2000.csv for the baselines; size_stability_summary.csv) and the timings of the confirmation run files.
