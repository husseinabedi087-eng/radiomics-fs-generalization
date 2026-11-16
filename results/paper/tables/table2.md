**Table 2. Generalization controls and stage-2 choices.**

| Control | Description | Cost in evaluations | Parameter | Grid | Chosen | Runner-up | Development margin (mean outer AUC) |
|---|---|---|---|---|---|---|---|
| C1 | Early stopping on a sealed 20% selection fold | 1 selection-fold scoring after every 100 search evaluations (101 per round) | P | {3, 5, 10} | 5 | 10 | 0.0004 |
| C2 | Variance-penalized fitness (mean - lambda x SD over a 3 x 3 split) | 1 per scored subset (9 model fits) | lambda | {0.5, 1, 2} | 1.0 | 0.5 | 0.0001 |
| C3 | Flatness fitness (mean over the subset and r one-bit neighbours) | 1 + r per scored subset | r | {2, 4, 8} | 8 | 2 | 0.0001 |
| C4 | Knee of the archive's (inner fitness, size) front, from the C0 run | 0 (computed from the C0 archive) | - | - | primary control (H2) | C3 | 0.0056 |

Development margin: score of the chosen value minus that of the runner-up (results/stage2/stage2_choices.json; 12 development datasets, 5 runs per cell). For C4 the margin is over the best other control at its chosen value (primary control for H2).
