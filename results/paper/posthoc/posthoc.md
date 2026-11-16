# POST HOC (not pre-registered; confirmation data)

## A. Search-attributable optimism

| Quantity | n | Mean | Median | 95% CI of mean | Datasets > 0 |
|---|---|---|---|---|---|
| Mean gap, 12 C0 | 31 | 0.1335 | 0.1019 | [0.1029, 0.1661] | - |
| Gap of all_k | 31 | 0.0189 | 0.0042 | [0.0002, 0.0401] | - |
| d_A = C0 mean gap - all_k gap | 31 | 0.1146 | 0.1046 | [0.0955, 0.1341] | 31 |

## B. Spearman correlation with log(p/n)

| Quantity | rho | 95% bootstrap CI | n |
|---|---|---|---|
| Mean gap, 12 C0 | 0.404 | [0.043, 0.685] | 31 |
| Gap of all_k | 0.596 | [0.311, 0.785] | 31 |
| d_A | 0.132 | [-0.252, 0.483] | 31 |

## C. Dataset difficulty (best mean outer AUC among the six baselines)

Min 0.441, Q1 0.582, median 0.698, Q3 0.783, max 0.968, mean 0.689. Datasets with best AUC < 0.60: 9; >= 0.75: 10. Best baseline counts: all_k 7, l1 5, enet 6, ftest 6, mrmr 4, rf 3.

| Stratum | n | H2 d mean [95% CI] | H3 d mean [95% CI] | Guided C0 - random C0 mean [95% CI] |
|---|---|---|---|---|
| best < 0.75 | 21 | -0.0010 [-0.0033, 0.0013] | -0.0028 [-0.0173, 0.0123] | 0.0058 [-0.0099, 0.0222] |
| best >= 0.75 | 10 | 0.0008 [-0.0009, 0.0029] | -0.0148 [-0.0235, -0.0050] | -0.0042 [-0.0110, 0.0031] |

## D. Equivalence details

| d | Hodges-Lehmann | HL 90% CI (exact signed-rank) | Mean | 90% bootstrap CI of mean | Margin | Datasets with abs(d) > margin |
|---|---|---|---|---|---|---|
| H3 | -0.0092 | [-0.0172, 0.0002] | -0.0067 | [-0.0155, 0.0024] | 0.02 | 17 |
| H5 | 0.0007 | [-0.0017, 0.0033] | 0.0008 | [-0.0023, 0.0039] | 0.01 | 9 |

## E. H5 by transfer function (post hoc test: one-sided Wilcoxon of d - 0.01 < 0)

| Transfer | n | Mean | 95% CI | Hodges-Lehmann | W | p (post hoc) |
|---|---|---|---|---|---|---|
| S-shaped | 31 | 0.0020 | [-0.0045, 0.0084] | 0.0018 | 125 | 0.0074 |
| V-shaped | 31 | -0.0005 | [-0.0024, 0.0014] | -0.0001 | 2 | 1.4e-09 |

## F. Brier score, H5 without WOA-S (metaphor = GWO-S, GWO-V, WOA-V; descriptive)

| n | Mean | 95% CI | Median | W | p (two-sided) |
|---|---|---|---|---|---|
| 31 | -0.0005 | [-0.0013, 0.0003] | -0.0003 | 188 | 0.25 |

## G. Mixed model on per-dataset means (outer_auc): 620 rows, 31 datasets

Dataset variance 0.0197; residual variance 6.03e-05; interaction LRT 3.88, df 12, p 0.99; log-likelihood full 1996.32, reduced 1994.38. Fits: lbfgs converged=True; lbfgs converged=True. Warnings: none.

| Term | Estimate | SE | 95% CI |
|---|---|---|---|
| Intercept | +0.70687 | 0.04497 | [+0.61874, +0.79501] |
| C(searcher)[T.de_S] | +0.00038 | 0.00197 | [-0.00349, +0.00424] |
| C(searcher)[T.ga] | +0.00108 | 0.00197 | [-0.00278, +0.00495] |
| C(searcher)[T.gwo_S] | +0.00006 | 0.00197 | [-0.00380, +0.00393] |
| C(control)[T.C1] | -0.00253 | 0.00197 | [-0.00640, +0.00133] |
| C(control)[T.C2] | -0.00354 | 0.00197 | [-0.00741, +0.00032] |
| C(control)[T.C3] | -0.00009 | 0.00197 | [-0.00395, +0.00377] |
| C(control)[T.C4] | +0.00000 | 0.00197 | [-0.00386, +0.00387] |
| C(searcher)[T.de_S]:C(control)[T.C1] | -0.00169 | 0.00279 | [-0.00715, +0.00378] |
| C(searcher)[T.ga]:C(control)[T.C1] | -0.00171 | 0.00279 | [-0.00718, +0.00375] |
| C(searcher)[T.gwo_S]:C(control)[T.C1] | -0.00231 | 0.00279 | [-0.00777, +0.00316] |
| C(searcher)[T.de_S]:C(control)[T.C2] | +0.00009 | 0.00279 | [-0.00538, +0.00555] |
| C(searcher)[T.ga]:C(control)[T.C2] | +0.00028 | 0.00279 | [-0.00518, +0.00575] |
| C(searcher)[T.gwo_S]:C(control)[T.C2] | +0.00067 | 0.00279 | [-0.00479, +0.00614] |
| C(searcher)[T.de_S]:C(control)[T.C3] | -0.00056 | 0.00279 | [-0.00603, +0.00490] |
| C(searcher)[T.ga]:C(control)[T.C3] | +0.00080 | 0.00279 | [-0.00467, +0.00626] |
| C(searcher)[T.gwo_S]:C(control)[T.C3] | +0.00035 | 0.00279 | [-0.00512, +0.00581] |
| C(searcher)[T.de_S]:C(control)[T.C4] | -0.00254 | 0.00279 | [-0.00801, +0.00292] |
| C(searcher)[T.ga]:C(control)[T.C4] | +0.00022 | 0.00279 | [-0.00524, +0.00569] |
| C(searcher)[T.gwo_S]:C(control)[T.C4] | +0.00065 | 0.00279 | [-0.00482, +0.00611] |
| log_pn | -0.02853 | 0.02502 | [-0.07757, +0.02051] |

## G. Mixed model on per-dataset means (optimism_gap): 620 rows, 31 datasets

Dataset variance 0.00479; residual variance 0.000356; interaction LRT 165.49, df 12, p 4.0e-29; log-likelihood full 1495.43, reduced 1412.69. Fits: lbfgs converged=True; lbfgs converged=True. Warnings: ConvergenceWarning: The MLE may be on the boundary of the parameter space..

| Term | Estimate | SE | 95% CI |
|---|---|---|---|
| Intercept | +0.07200 | 0.02243 | [+0.02803, +0.11596] |
| C(searcher)[T.de_S] | +0.01259 | 0.00479 | [+0.00321, +0.02198] |
| C(searcher)[T.ga] | +0.05430 | 0.00479 | [+0.04491, +0.06369] |
| C(searcher)[T.gwo_S] | -0.00940 | 0.00479 | [-0.01878, -0.00001] |
| C(control)[T.C1] | -0.06804 | 0.00479 | [-0.07742, -0.05865] |
| C(control)[T.C2] | -0.04650 | 0.00479 | [-0.05588, -0.03711] |
| C(control)[T.C3] | -0.01980 | 0.00479 | [-0.02919, -0.01042] |
| C(control)[T.C4] | -0.01814 | 0.00479 | [-0.02753, -0.00876] |
| C(searcher)[T.de_S]:C(control)[T.C1] | -0.01415 | 0.00677 | [-0.02742, -0.00087] |
| C(searcher)[T.ga]:C(control)[T.C1] | -0.05006 | 0.00677 | [-0.06333, -0.03678] |
| C(searcher)[T.gwo_S]:C(control)[T.C1] | +0.00735 | 0.00677 | [-0.00593, +0.02062] |
| C(searcher)[T.de_S]:C(control)[T.C2] | -0.00667 | 0.00677 | [-0.01994, +0.00661] |
| C(searcher)[T.ga]:C(control)[T.C2] | -0.01940 | 0.00677 | [-0.03268, -0.00613] |
| C(searcher)[T.gwo_S]:C(control)[T.C2] | +0.00328 | 0.00677 | [-0.01000, +0.01655] |
| C(searcher)[T.de_S]:C(control)[T.C3] | -0.02456 | 0.00677 | [-0.03783, -0.01128] |
| C(searcher)[T.ga]:C(control)[T.C3] | -0.04547 | 0.00677 | [-0.05874, -0.03219] |
| C(searcher)[T.gwo_S]:C(control)[T.C3] | -0.00329 | 0.00677 | [-0.01657, +0.00998] |
| C(searcher)[T.de_S]:C(control)[T.C4] | +0.00553 | 0.00677 | [-0.00774, +0.01881] |
| C(searcher)[T.ga]:C(control)[T.C4] | +0.01241 | 0.00677 | [-0.00086, +0.02569] |
| C(searcher)[T.gwo_S]:C(control)[T.C4] | -0.00058 | 0.00677 | [-0.01385, +0.01270] |
| log_pn | +0.03646 | 0.01235 | [+0.01226, +0.06067] |

## H. Runs per searcher in the pure-noise structural-bias test (docs/VERIFICATION.md)

30 runs per encoding x 2 encodings = 60 runs per searcher for the location tests; the dispersion diagnostic uses the 30 original-encoding runs; 12 searchers.
