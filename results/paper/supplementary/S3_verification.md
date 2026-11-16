**S3a. 1. Continuous cores vs MEALPY 3.0.3**

| Algorithm | Function | MEALPY model | Median ours | Median MEALPY | p | Result |
|---|---|---|---|---|---|---|
| PSO core (BPSO) | sphere | LDW_PSO | 1.281e+04 | 1.297e+04 | 0.9031 | ok |
| PSO core (BPSO) | rastrigin | LDW_PSO | 303.3 | 309 | 0.3942 | ok |
| DE | sphere | OriginalDE | 2666 | 2053 | 0.2085 | ok |
| DE | rastrigin | OriginalDE | 217.4 | 225.5 | 0.9676 | ok |
| GWO | sphere | OriginalGWO | 0.03978 | 0.04666 | 0.2733 | ok |
| GWO | rastrigin | OriginalGWO | 58.49 | 59.23 | 0.7150 | ok |
| WOA | sphere | OriginalWOA | 7.459e-12 | 1.263e-11 | 0.7764 | ok |
| WOA | rastrigin | OriginalWOA | 5.027e-06 | 131.5 | 0.2085 | ok |

**S3b. 2. OneMax (K = 200, budget 2,000, seeds 0-4)**

| Searcher | Per seed (250 -> 2,000) | Mean 250 | Mean 2,000 | All optimal at 250 | (a) | (b) |
|---|---|---|---|---|---|---|
| random | 1.000 -> 1.000; 0.995 -> 1.000; 1.000 -> 1.000; 1.000 -> 1.000; 1.000 -> 1.000 | 0.999 | 1.000 | no | pass | pass |
| hill_climb | 0.810 -> 1.000; 0.820 -> 1.000; 0.840 -> 1.000; 0.855 -> 1.000; 0.805 -> 1.000 | 0.826 | 1.000 | no | pass | pass |
| one_plus_one | 0.730 -> 1.000; 0.810 -> 1.000; 0.740 -> 0.985; 0.755 -> 0.995; 0.690 -> 0.980 | 0.745 | 0.992 | no | pass | pass |
| ga | 0.700 -> 0.985; 0.700 -> 0.970; 0.695 -> 0.975; 0.680 -> 0.965; 0.710 -> 0.965 | 0.697 | 0.972 | no | pass | pass |
| bpso_S | 0.690 -> 0.690; 0.640 -> 0.700; 0.655 -> 0.710; 0.680 -> 0.685; 0.655 -> 0.710 | 0.664 | 0.699 | no | pass | pass |
| bpso_V | 0.660 -> 0.780; 0.665 -> 0.750; 0.680 -> 0.795; 0.690 -> 0.795; 0.655 -> 0.760 | 0.670 | 0.776 | no | pass | pass |
| de_S | 0.625 -> 0.755; 0.625 -> 0.745; 0.645 -> 0.765; 0.635 -> 0.705; 0.635 -> 0.780 | 0.633 | 0.750 | no | pass | pass |
| de_V | 0.585 -> 0.610; 0.565 -> 0.595; 0.600 -> 0.635; 0.585 -> 0.600; 0.585 -> 0.600 | 0.584 | 0.608 | no | pass | pass |
| gwo_S | 0.590 -> 0.615; 0.630 -> 0.695; 0.640 -> 0.670; 0.585 -> 0.605; 0.670 -> 0.750 | 0.623 | 0.667 | no | pass | pass |
| gwo_V | 0.575 -> 0.595; 0.580 -> 0.670; 0.575 -> 0.595; 0.585 -> 0.615; 0.570 -> 0.635 | 0.577 | 0.622 | no | pass | pass |
| woa_S | 1.000 -> 1.000; 1.000 -> 1.000; 1.000 -> 1.000; 1.000 -> 1.000; 1.000 -> 1.000 | 1.000 | 1.000 | yes | pass | pass |
| woa_V | 0.570 -> 0.630; 0.620 -> 0.650; 0.585 -> 0.610; 0.590 -> 0.645; 0.610 -> 0.650 | 0.595 | 0.637 | no | pass | pass |

**S3c. 3. Structural bias (Kononova et al., 2015, adapted to binary masks)**

| Searcher | Size original | Size inverted | p (MW) | p (Wilcoxon vs 100) | Encoding test | Wilcoxon |
|---|---|---|---|---|---|---|
| random | 102.2 (53.1) | 98.8 (56.5) | 0.9176 | 0.7734 | pass | - |
| hill_climb | 98.3 (6.5) | 101.4 (6.8) | 0.0768 | 0.2089 | pass | - |
| one_plus_one | 98.9 (7.1) | 102.4 (7.5) | 0.0562 | 0.3478 | pass | - |
| ga | 101.5 (5.6) | 101.4 (7.7) | 0.8823 | 0.1377 | pass | - |
| bpso_S | 100.1 (5.8) | 99.4 (7.4) | 0.4772 | 0.8911 | pass | - |
| bpso_V | 98.7 (6.4) | 101.7 (7.8) | 0.1527 | 0.2193 | pass | - |
| de_S | 97.8 (5.9) | 100.1 (7.1) | 0.1826 | 0.0408 | pass | - |
| de_V | 100.7 (7.1) | 100.6 (6.8) | 0.7502 | 0.6650 | pass | - |
| gwo_S | 100.3 (6.7) | 99.8 (7.4) | 0.7898 | 0.7888 | pass | - |
| gwo_V | 100.9 (7.2) | 99.3 (8.6) | 0.4155 | 0.6107 | pass | - |
| woa_S | 98.7 (71.3) | 107.8 (68.0) | 0.4964 | 0.8290 | pass | - |
| woa_V | 101.2 (7.9) | 99.8 (6.6) | 0.4544 | 0.4203 | pass | - |

**S3d. 3b. Structural bias: dispersion (diagnostic, not a gate)**

| Searcher | SD | Min | Median | Max | Size <= 20 | Size >= 180 | p (Brown-Forsythe vs random) |
|---|---|---|---|---|---|---|---|
| random | 53.1 | 6 | 108.5 | 197 | 1 | 3 | reference |
| hill_climb | 6.5 | 83 | 99.0 | 111 | 0 | 0 | <0.0001 |
| one_plus_one | 7.1 | 85 | 99.0 | 112 | 0 | 0 | <0.0001 |
| ga | 5.6 | 90 | 102.0 | 116 | 0 | 0 | <0.0001 |
| bpso_S | 5.8 | 87 | 100.5 | 113 | 0 | 0 | <0.0001 |
| bpso_V | 6.4 | 85 | 100.0 | 111 | 0 | 0 | <0.0001 |
| de_S | 5.9 | 85 | 97.5 | 113 | 0 | 0 | <0.0001 |
| de_V | 7.1 | 83 | 101.0 | 116 | 0 | 0 | <0.0001 |
| gwo_S | 6.7 | 86 | 101.5 | 113 | 0 | 0 | <0.0001 |
| gwo_V | 7.2 | 86 | 100.5 | 113 | 0 | 0 | <0.0001 |
| woa_S | 71.3 | 1 | 104.0 | 195 | 7 | 6 | 0.0511 |
| woa_V | 7.9 | 85 | 101.5 | 116 | 0 | 0 | <0.0001 |

**S3e. 3b. Structural bias: dispersion (diagnostic, not a gate) - per-iteration mechanism (woa_S, woa_V)**

| Searcher | Iteration | At bounds | Same-bound share | Sign coherence | Zero move |
|---|---|---|---|---|---|
| woa_S | 1 | 0.000 | n/a | 0.686 | 0.033 |
| woa_S | 10 | 0.274 | 0.989 | 0.888 | 0.129 |
| woa_S | 25 | 0.318 | 0.985 | 0.896 | 0.156 |
| woa_S | 50 | 0.321 | 0.983 | 0.925 | 0.163 |
| woa_S | 75 | 0.314 | 0.989 | 0.941 | 0.162 |
| woa_S | 99 | 0.310 | 0.988 | 0.969 | 0.227 |
| woa_V | 1 | 0.000 | n/a | 0.686 | 0.033 |
| woa_V | 10 | 0.282 | 0.992 | 0.873 | 0.127 |
| woa_V | 25 | 0.304 | 0.992 | 0.903 | 0.141 |
| woa_V | 50 | 0.307 | 0.995 | 0.931 | 0.144 |
| woa_V | 75 | 0.286 | 0.996 | 0.939 | 0.132 |
| woa_V | 99 | 0.275 | 0.997 | 0.950 | 0.133 |

**S3f. 4. Determinism**

| Searcher | Same seed identical | Different seeds differ |
|---|---|---|
| random | yes | yes |
| hill_climb | yes | yes |
| one_plus_one | yes | yes |
| ga | yes | yes |
| bpso_S | yes | yes |
| bpso_V | yes | yes |
| de_S | yes | yes |
| de_V | yes | yes |
| gwo_S | yes | yes |
| gwo_V | yes | yes |
| woa_S | yes | yes |
| woa_V | yes | yes |

**S3g. 5. Planted subset (reported, not a gate)**

| Searcher | Informative selected (per seed) | Mean recovery | Subset size | Inner AUC | Seconds per run |
|---|---|---|---|---|---|
| random | 6/10, 3/10, 9/10 | 0.60 | 70, 19, 92 | 0.765, 0.760, 0.767 | 120.3 |
| hill_climb | 8/10, 9/10, 9/10 | 0.87 | 73, 63, 66 | 0.993, 0.976, 0.995 | 88.1 |
| one_plus_one | 6/10, 9/10, 9/10 | 0.80 | 66, 75, 78 | 0.954, 0.982, 0.981 | 71.3 |
| ga | 10/10, 10/10, 10/10 | 1.00 | 80, 78, 74 | 0.978, 0.941, 0.987 | 138.7 |
| bpso_S | 8/10, 8/10, 9/10 | 0.83 | 91, 92, 96 | 0.790, 0.804, 0.840 | 84.9 |
| bpso_V | 8/10, 10/10, 10/10 | 0.93 | 90, 93, 92 | 0.864, 0.881, 0.869 | 157.1 |
| de_S | 10/10, 10/10, 10/10 | 1.00 | 87, 99, 88 | 0.859, 0.846, 0.888 | 52.6 |
| de_V | 7/10, 7/10, 7/10 | 0.70 | 100, 94, 102 | 0.729, 0.735, 0.725 | 121.9 |
| gwo_S | 9/10, 6/10, 6/10 | 0.70 | 92, 89, 89 | 0.801, 0.747, 0.783 | 73.4 |
| gwo_V | 8/10, 8/10, 4/10 | 0.67 | 111, 107, 84 | 0.767, 0.748, 0.682 | 32.8 |
| woa_S | 3/10, 3/10, 5/10 | 0.37 | 28, 17, 9 | 0.760, 0.740, 0.779 | 22.4 |
| woa_V | 8/10, 7/10, 7/10 | 0.73 | 98, 96, 96 | 0.760, 0.797, 0.758 | 35.7 |

**S3h. Budget accounting (stage 1, section 9)**

Checked by the tests in tests/test_budget.py (every scored subset, including control evaluations, increments the counter). These are pass/fail tests without a result file; they are part of the fast suite, which passed at commit 0b3da97 (155 passed; docs/PREREG_STAGE2.md, section 7.1).

**Sources**

Tables S3a-S3g are copied from docs/VERIFICATION.md (SHA-256 7e5b8ed6147828c7b0b83a4771983399ed7a8c3a61d22de044a6a828b0ce7ef6); the 59 raw result files in results/verification/ are listed with their SHA-256 and code commit in S3h_raw_files.csv.
