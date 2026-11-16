**Table S6a. Random search against each guided C0 searcher (two-sided Wilcoxon, Holm within the family).**

| Guided C0 searcher | Datasets | Mean d | Median d | p | p Holm | Rank-biserial | Method |
|---|---|---|---|---|---|---|---|
| hill_climb__C0 | 31 | 0.0012 | -0.0005 | 0.92 | 1.0 | -0.02 | exact |
| one_plus_one__C0 | 31 | 0.0020 | 0.0017 | 0.74 | 1.0 | 0.07 | exact |
| ga__C0 | 31 | 0.0029 | 0.0008 | 0.72 | 1.0 | 0.08 | exact |
| bpso_S__C0 | 31 | 0.0018 | 0.0007 | 0.65 | 1.0 | 0.10 | exact |
| bpso_V__C0 | 31 | 0.0035 | 0.0010 | 0.79 | 1.0 | 0.06 | exact |
| de_S__C0 | 31 | 0.0022 | 0.0016 | 0.78 | 1.0 | 0.06 | exact |
| de_V__C0 | 31 | 0.0019 | -0.0012 | 0.88 | 1.0 | 0.03 | exact |
| gwo_S__C0 | 31 | 0.0018 | 0.0027 | 0.82 | 1.0 | 0.05 | exact |
| gwo_V__C0 | 31 | 0.0026 | 0.0016 | 0.76 | 1.0 | 0.06 | exact |
| woa_S__C0 | 31 | 0.0067 | 0.0034 | 0.16 | 1.0 | 0.29 | exact |
| woa_V__C0 | 31 | 0.0019 | 0.0005 | 0.84 | 1.0 | 0.04 | exact |

**Table S6b. Brier-score differences (lower is better; two-sided, descriptive).**

| Comparison | Datasets | Statistic | p (two-sided) | Mean d | Median d | CI low | CI high | Rank-biserial |
|---|---|---|---|---|---|---|---|---|
| H2: searcher + C4 minus C0 | 31 | 107 | 0.0048 | -0.0012 | -0.0011 | -0.0019 | -0.0004 | -0.57 |
| H3: GA + C4 minus L1 | 31 | 91 | 0.0015 | 0.0101 | 0.0080 | 0.0041 | 0.0159 | 0.63 |
| H5: metaphor minus classical | 31 | 66 | 0.00016 | -0.0052 | -0.0026 | -0.0083 | -0.0026 | -0.73 |

**Table S6c. Friedman test across all selectors at 2,000 evaluations (context only).**

| Datasets | Selectors | Friedman statistic | p | Pairwise comparisons | Rejected after Holm |
|---|---|---|---|---|---|
| 31 | 42 | 74.09 | 0.0012 | 861 | 0 |

The 861 pairwise Wilcoxon comparisons are in S6d_friedman_pairwise.csv (CSV only).
