**Table 3. Confirmatory tests (31 confirmation datasets).**

| Hypothesis | Test | d | Datasets | Statistic | p raw | p Holm | Mean d [95% CI] | Rank-biserial | Wording |
|---|---|---|---|---|---|---|---|---|---|
| H1 | Wilcoxon signed-rank, one-sided (d > 0) | optimism gap, mean of 12 C0 | 31 | W = 496 | 4.7e-10 | 2.3e-09 | 0.1335 [0.1033, 0.1665] | 1.00 | optimism gap greater than zero |
| H2 | Wilcoxon signed-rank, one-sided (d > 0) | searcher + C4 minus C0 (4 searchers) | 31 | W = 236 | 0.60 | 0.69 | -0.0004 [-0.0021, 0.0013] | -0.05 | no detectable difference |
| H3 | TOST, Wilcoxon form, margin 0.02 | GA + C4 minus L1 | 31 | W_low = 359, W_up = 60 | 0.014 | 0.043 | -0.0067 [-0.0170, 0.0043] | -0.34 | equivalent (margin 0.02) |
| H4 | Likelihood-ratio test of searcher x control (mixed model) | - | 31 | LR = 13.33, df = 12 | 0.35 | 0.69 | - | - | no detectable interaction |
| H5 | Wilcoxon signed-rank, one-sided (d - 0.01 < 0) | GWO/WOA minus GA/BPSO/DE (C0) | 31 | W = 52 | 1.8e-05 | 7.1e-05 | 0.0008 [-0.0028, 0.0045] | 0.12 | equivalent (margin 0.01) |

Holm correction across H1-H5 at alpha = 0.05 (H3 enters with p_TOST, H4 with the likelihood-ratio p). 95% CI: percentile bootstrap of the mean over 31 confirmation datasets (10,000 resamples). Exact Wilcoxon distribution in every test. Source: results/analysis/confirmation/results.json.
