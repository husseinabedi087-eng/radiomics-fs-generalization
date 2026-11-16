**Table S8. Controls averaged over GA, BPSO-S, DE-S and GWO-S at 2,000 evaluations.**

| Control | Datasets | Mean optimism gap | Mean outer AUC | Gap minus C0 | AUC minus C0 |
|---|---|---|---|---|---|
| C0 (no control) | 31 | 0.1406 | 0.6648 | +0.0000 | +0.0000 |
| C1 early stopping (P = 5) | 31 | 0.0583 | 0.6609 | -0.0823 | -0.0040 |
| C2 variance penalty (lambda = 1) | 31 | 0.0884 | 0.6616 | -0.0522 | -0.0033 |
| C3 flatness (r = 8) | 31 | 0.1024 | 0.6649 | -0.0381 | +0.0001 |
| C4 knee | 31 | 0.1268 | 0.6644 | -0.0138 | -0.0004 |

For each dataset, the mean over the four searchers of the per-dataset mean (outer folds x seeds); then the mean over the 31 confirmation datasets. Differences are means over datasets of the per-dataset difference from C0. Computed from the full-precision per_dataset_{gap,auc}_2000.csv tables of the frozen analysis. Descriptive; no test.
