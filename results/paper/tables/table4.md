**Table 4. Mixed-model control effects relative to C0 and interaction tests.**

| Term | Outer AUC | Optimism gap (descriptive) |
|---|---|---|
| C1 early stopping vs C0 (main effect) | -0.0036 | -0.0624 |
| C2 variance penalty vs C0 (main effect) | -0.0038 | -0.0430 |
| C3 flatness vs C0 (main effect) | -0.0006 | -0.0334 |
| C4 knee vs C0 (main effect) | -0.0016 | -0.0164 |
| Interaction DE-S x C1 | +0.0014 | -0.0039 |
| Interaction DE-S x C2 | +0.0015 | -0.0016 |
| Interaction DE-S x C3 | +0.0004 | -0.0064 |
| Interaction DE-S x C4 | +0.0007 | -0.0008 |
| Interaction GA x C1 | -0.0004 | -0.0299 |
| Interaction GA x C2 | +0.0015 | -0.0142 |
| Interaction GA x C3 | -0.0003 | -0.0297 |
| Interaction GA x C4 | +0.0021 | +0.0027 |
| Interaction GWO-S x C1 | -0.0010 | +0.0084 |
| Interaction GWO-S x C2 | +0.0012 | +0.0039 |
| Interaction GWO-S x C3 | +0.0010 | +0.0028 |
| Interaction GWO-S x C4 | +0.0009 | -0.0020 |
| log(evaluations) | +0.0003 | +0.0132 |
| log(p/n) | -0.0286 | +0.0361 |
| Interaction LRT: statistic | 13.33 | 1304.78 |
| Interaction LRT: df | 12 | 12 |
| Interaction LRT: p | 0.35 | 4.7e-272 |
| Rows (runs x checkpoints) | 74,400 | 74,400 |

Linear mixed model fitted by maximum likelihood on run-level values at 250, 500, 1,000 and 2,000 evaluations for GA, BPSO-S, DE-S and GWO-S under C0-C4; random intercept for dataset and a variance component for outer partition within dataset. Treatment coding: control main effects are for the reference searcher BPSO-S; for another searcher the control effect is the main effect plus its interaction term. Every fit converged with lbfgs; each logged 'The MLE may be on the boundary of the parameter space.' H4 is the outer-AUC test; the optimism-gap model is descriptive. Source: results/analysis/confirmation/results.json.
