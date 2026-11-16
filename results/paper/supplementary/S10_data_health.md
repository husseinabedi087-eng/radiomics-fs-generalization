**Table S10a. Plans checked.**

| Report | Plan | Units | Missing | Status | Commits | Before stage 2 | Threads |
|---|---|---|---|---|---|---|---|
| development | development | 2880 | 0 | {"ok": 2880} | 3b69ca5 | 0 | 2880 runs [['openblas', 1], ['openmp', 1]] |
| development | dev_enet_fix | 60 | 0 | {"ok": 60} | a1535d6 | 0 | 60 runs [['openblas', 1], ['openmp', 1]] |
| confirmation | confirmation | 24800 | 0 | {"ok": 24800} | 862a7ea | 0 | 24800 runs [['openblas', 1], ['openmp', 1]] |
| confirmation | confirmation_s2 | 680 | 0 | {"ok": 680} | 862a7ea | 0 | 680 runs [['openblas', 1], ['openmp', 1]] |

**Table S10b. Convergence warnings (configurations with any warning; all configurations in the CSV).**

| Report | Plan | Configuration | Warnings | Runs with warnings | Runs |
|---|---|---|---|---|---|
| development | development | baseline__enet | 1228 | 60 | 60 |
| confirmation | confirmation | baseline__l1 | 101 | 29 | 310 |

77 of 79 plan x configuration cells had no warning.

**Table S10c. Baseline runs that returned an empty subset.**

| Report | Plan | Configuration | Runs | Datasets | Per dataset |
|---|---|---|---|---|---|
| development | development | baseline__enet | 2 | 1 | Ahn2021 (2) |
| development | development | baseline__l1 | 1 | 1 | Ahn2021 (1) |
| development | dev_enet_fix | baseline__enet | 2 | 1 | Ahn2021 (2) |
| confirmation | confirmation | baseline__enet | 5 | 4 | Hosny2018C (2), NSCLC-Radiogenomics (1), Petrillo2023 (1), Veeraraghavan2020 (1) |
| confirmation | confirmation | baseline__l1 | 5 | 3 | NSCLC-Radiogenomics (2), Petrillo2023 (2), Veeraraghavan2020 (1) |
| confirmation | confirmation_s2 | baseline__l1 | 15 | 6 | C4KC-KiTS (1), Hosny2018C (5), NSCLC-Radiogenomics (2), Petrillo2023 (4), Sasaki2019 (2), Veeraraghavan2020 (1) |

**Table S10d. C1 early stops.**

| Report | Plan | C1 runs | Stopped early | Ended before 250 evaluations |
|---|---|---|---|---|
| development | development | 720 | 692 | 0 |
| confirmation | confirmation | 3720 | 3706 | 0 |

Source: results/health/{development,confirmation}.json, written by src/save_health.py from the unchanged src/run_health.py (no AUC, Brier score or subset read). The development report covers the development plan (elastic net at max_iter 2,000) and its elastic-net rerun dev_enet_fix (max_iter 10,000); the analysis uses the rerun.
