**EXPLORATORY (development data). Table S9a. Stage-2 hyperparameter candidates (12 development datasets).**

| Control | Value | Score | Chosen |
|---|---|---|---|
| C1 | 3 | 0.6582 |  |
| C1 | 5 | 0.6600 | yes |
| C1 | 10 | 0.6596 |  |
| C2 | 0.5 | 0.6663 |  |
| C2 | 1.0 | 0.6664 | yes |
| C2 | 2.0 | 0.6587 |  |
| C3 | 2 | 0.6667 |  |
| C3 | 4 | 0.6641 |  |
| C3 | 8 | 0.6668 | yes |

**EXPLORATORY (development data). Table S9b. Primary-control candidates.**

| Control | Value | Score | Chosen |
|---|---|---|---|
| C1 | 5 | 0.6600 |  |
| C2 | 1.0 | 0.6664 |  |
| C3 | 8 | 0.6668 |  |
| C4 | - | 0.6723 | yes |

**EXPLORATORY (development data). Table S9c. Primary-wrapper candidates (ranked).**

| Searcher | Control | Value | Score | Chosen |
|---|---|---|---|---|
| ga | C4 | - | 0.6760 | yes |
| ga | C2 | 1.0 | 0.6745 |  |
| bpso_S | C4 | - | 0.6744 |  |
| de_S | C4 | - | 0.6717 |  |
| gwo_S | C1 | 5 | 0.6700 |  |
| gwo_S | C2 | 1.0 | 0.6695 |  |
| gwo_S | C3 | 8 | 0.6691 |  |
| bpso_S | C3 | 8 | 0.6686 |  |
| de_S | C3 | 8 | 0.6684 |  |
| bpso_S | C2 | 1.0 | 0.6683 |  |
| gwo_S | C4 | - | 0.6672 |  |
| bpso_S | C1 | 5 | 0.6629 |  |
| ga | C3 | 8 | 0.6610 |  |
| de_S | C1 | 5 | 0.6537 |  |
| ga | C1 | 5 | 0.6536 |  |
| de_S | C2 | 1.0 | 0.6534 |  |

Source: results/stage2/stage2_choices.json (development data; per-dataset scores are in that file).

**EXPLORATORY (development data). Table S9d. H1-H5 in the development dry run (configurations missing by design).**

| Hypothesis | Datasets | p raw | p Holm | Mean d | CI low | CI high | Rank-biserial | Wording |
|---|---|---|---|---|---|---|---|---|
| H1 | 12 | 0.00024 | 0.0012 | 0.1494 | 0.1026 | 0.2026 | 1.00 | optimism gap greater than zero |
| H2 | 12 | 0.21 | 0.21 | 0.0029 | -0.0028 | 0.0087 | 0.28 | no detectable difference |
| H3 | 12 | 0.088 | 0.18 | 0.0008 | -0.0264 | 0.0274 | 0.00 | no detectable difference |
| H4 | 12 | 0.023 | 0.069 | - | - | - | - | no detectable interaction |
| H5 | 12 | 0.013 | 0.054 | -0.0071 | -0.0197 | 0.0055 | -0.38 | no detectable difference |

Source: results/analysis/development_dryrun/results.json. The development plan ran 4 of the 12 C0 searchers, so H1 and H5 use fewer configurations than the confirmatory analysis.
