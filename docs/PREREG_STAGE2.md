# Pre-registration, stage 2

Study: Search less, generalize more: overfitting-aware evolutionary feature selection for radiomics
Author: Hussein Fadhil Abedi (sole author)
Date written: 2026-11-13
Status at writing: development run complete (2,880 runs, 0 errors; run commit 3b69ca5; results committed in 7424ecd), elastic net development rerun complete (60 runs; plan dev_enet_fix; committed in c8574f1). Release validation passed for the pilot, development and dev_enet_fix plans (3,420 run files). Analysis code complete and tested; development dry run committed. No confirmation dataset has been loaded. The confirmation set stays sealed by src/data_access.py until docs/PREREG_STAGE2.sha256 is committed.

This document completes docs/PREREG_STAGE1.md (SHA-256 in docs/PREREG_STAGE1.sha256) and docs/PREREG_STAGE1_AMENDMENT1.md. It fixes the items listed in section 10 of stage 1, records the operational definitions that stage 1 left open, and lists every code change made after stage 1 was frozen (commit 8fa82c7, 2026-11-10). Sections 1, 4, 8 and 11 of stage 1 are unchanged.

## 1. How the choices were made

- Script: src/stage2_select.py, committed in 0e21454 before the development run started (git blob 827d000b1888ea6cf46975bd71fd52a2de66f441, identical at the run commit and at the time of selection). The script was run unchanged.
- Output: results/stage2/stage2_choices.json, SHA-256 d38695a8ecca2587d0acf7ebd0570cae20f8925115d8e95ca32c4e8c4b075f14.
- Note on that file: its field script_commit holds the repository HEAD at the time the script ran (7424ecd), not the commit of the script (0e21454). The file is kept unedited so that its hash stays valid (docs/DECISIONS.md, 2026-11-12).
- Score: for each candidate, the mean over the 12 development datasets of the per-dataset mean outer AUC at the 2,000-evaluation checkpoint (5 runs per dataset and configuration).
- Rule: the candidate with the highest score, as stated in stage 1, section 10. No tie-break was needed.

## 2. Control hyperparameters

Averaged over GA, binary PSO (S), DE (S) and GWO (S).

| Control | Chosen | Runner-up | Margin (mean outer AUC) | Datasets where the runner-up scored higher |
| --- | --- | --- | --- | --- |
| C1 patience P | 5 | 10 | 0.0004 | 3 of 12 |
| C2 lambda | 1.0 | 0.5 | 0.0001 | 7 of 12 |
| C3 neighbours r | 8 | 2 | 0.0001 | 4 of 12 |

All three margins are smaller than the run-to-run variation expected from 5 runs per cell. The values are fixed by the pre-registered rule without adjustment. The paper reports these margins and states that the C1, C2 and C3 results are conditional on these values. None of these values affects H2 or H3, because the primary control (section 3) has no hyperparameter.

## 3. Primary control for H2

Chosen: C4 (knee selection). Runner-up: C3 with r = 8, margin 0.0056. C4 scored higher than C3 on 7 of 12 development datasets, higher than C2 on 10 of 12, and higher than C1 on 8 of 12.

H2 as tested: for each confirmation dataset, the outer AUC of searcher + C4 minus the outer AUC of the same searcher under C0, averaged over GA, binary PSO (S), DE (S) and GWO (S). One-sided Wilcoxon signed-rank test, prediction greater than zero. C4 is computed from the C0 runs, so both arms share search seeds, partitions and evaluated subsets.

## 4. Primary controlled wrapper for H3

Chosen: GA + C4. Runner-up: GA + C2 (lambda = 1), margin 0.0015, with binary PSO (S) + C4 third (margin 0.0016).

H3 as tested: per-dataset outer AUC of GA + C4 minus that of L1-regularized logistic regression; TOST in the Wilcoxon signed-rank form with margin 0.02; a two-sided Wilcoxon test is also reported. Unchanged from stage 1.

## 5. Operational definitions of the methods

Stage 1 leaves these details open. They are stated as implemented at run commit 3b69ca5 and are not changed for confirmation.

C4, knee selection (src/fsbench/archive.py; src/fsbench/experiment.py):

- Computed during the run at each checkpoint c (250, 500, 1,000, 2,000) from the first c evaluations of the run's archive. The outer fold is scored after the knee is chosen.
- Front: non-dominated set on (inner fitness, maximized; subset size, minimized), sorted by increasing size. Among exact duplicates the earliest evaluation is kept; a larger subset enters the front only with strictly higher fitness.
- Both objectives are min-max normalized over the front (not the whole archive). The knee is the front point with the largest perpendicular distance to the line joining the two extremes.
- A front with one or two points returns its smallest subset. Distances within 1e-12 count as tied; the smallest subset among tied points is returned.

C1, early stopping:

- The selection fold is 20% of the outer-training rows, drawn by a stratified shuffle split with seed derived from SHA-256 of "dataset|repeat|fold|c1_selection". It is the same for every search seed and searcher within an outer fold.
- After every 100 search evaluations the best-so-far subset is scored once on the selection fold, so each round uses 101 evaluations; selection scorings count against the 2,000-evaluation budget.
- After an early stop, every later checkpoint records the returned subset (the one with the best selection-fold AUC), flagged filled_after_stop.

C2: standard deviation with ddof = 0 over the 9 AUCs of the 3 x 3 repeated stratified split.

C3: the r neighbours flip r distinct bits drawn without replacement, from a random stream separate from the searcher's own stream; each neighbour counts as one evaluation.

Searchers:

- Binary PSO follows the discrete binary PSO of Kennedy and Eberhart (1997).
- DE, GWO and WOA follow the MEALPY 3.0.3 "Original" variants with greedy replacement.
- The S-shaped transfer is the sigmoid; the V-shaped transfer is |tanh| (Mirjalili and Lewis, 2013), applied to the step rather than the position. This was settled during verification, before any data run.

Baselines: the elastic net uses max_iter = 10,000 (section 9). An empty subset from a baseline gets outer AUC 0.5 and a Brier score from predicting the outer-training prevalence for every test case, and is flagged empty_subset. In the development run this occurred 3 times, all on Ahn2021 (elastic net, 2 folds; L1, 1 fold).

## 6. Compute tier and S2

Decision: core tier. No configuration is dropped.

Projection (src/project_compute.py, commit c38cfb3; output results/compute/projection.json, commit 93b3593, SHA-256 04963b97038548a39849194561536f69b351a78f4b8c23370e10a8c1205955b2): 359 to 389 CPU hours, about 60 to 65 hours on 6 workers, against the 10-day limit of stage 1, section 10. A timing benchmark on synthetic data (src/bench_scaling.py, commit 468d682; output in results/compute/bench_scaling_output.txt) showed search time flat in training-set size from 166 to 775 samples. No confirmation dataset was loaded for the projection; only sizes from data/meta/audit.csv were used.

Core tier (configs/confirmation.yaml, commit 5873f6f; 24,800 run units): 31 datasets, 10 outer folds (5 x 2), 3 search seeds per outer fold (30 runs per dataset and configuration):

- C0: random subset search, hill climbing, (1+1)-EA, GA, and binary PSO, DE, GWO and WOA with S and V transfer (12 configurations), each with C4 recorded in-run at no extra evaluations;
- C1 (P = 5), C2 (lambda = 1.0) and C3 (r = 8) on GA, binary PSO (S), DE (S) and GWO (S) (12 configurations);
- all non-search selectors of stage 1, section 6; random-forest importance with 3 model seeds.

S2 is included (configs/confirmation_s2.yaml, commit 5873f6f; 680 run units). The H3 comparison (GA + C4 against L1-regularized logistic regression) is repeated without the univariate pre-filter, on the 17 confirmation datasets with at most 1,000 features after the data-handling rules. Without the pre-filter, K is the number of features that are non-constant in the outer-training part; partitions, inner split, search seeds, budget, checkpoints and C4 logging are unchanged. The option was added in commit f40eb50; with the pre-filter on, a development run reproduces its committed run payload exactly. A smoke run on one development dataset (p = 984, one outer fold) confirmed K = 984 and complete C4 records; it is not part of any result. Projected extra cost: 7 to 19 CPU hours. S2 is a sensitivity analysis and is not part of the Holm family.

## 7. Analysis

### 7.1 Code

All confirmatory and secondary numbers are produced by

    python -m analysis.run --config configs/analysis_confirmation.yaml

with src/analysis/ at commit 0b3da97a0a47162350fdfa46b5195887abb7639e and configs/analysis_confirmation.yaml at commit 44d6c32. The code reads only run files listed in results/MANIFEST.csv, validates the plans before analysing, and refuses to run on confirmation runs until docs/PREREG_STAGE2.sha256 is committed. Package versions are pinned in environment/pip_freeze.txt and recorded in every results.json.

Testing before this document was frozen:

- The fast test suite passes at 0b3da97 (155 passed, 2 opt-in development-data tests skipped, 12 slow tests deselected), including a synthetic fixture with the complete confirmation design (all 42 configurations, an S2 plan, at least 31 datasets with a sensitivity subset) on which no analysis is skipped, and tests that rerun the pipeline in separate processes with different hash seeds and require byte-identical outputs.
- The development dry run (configs/analysis_development.yaml) was run twice into separate folders with identical output manifests. Its outputs are committed in 9445500 and labelled exploratory; results.json SHA-256 58f04afa17981a52fbe12cb56a5a22a16c0f03906fc999ba0ed6cfc27efc363d.
- The first dry run used the analysis code at 12d9d4f. Two analysis commits followed it. 44d6c32 added the H4 acceptance rule (section 7.2), verbatim recording of fit warnings, an --output-dir option and the full-design synthetic test. 0b3da97 made the stability sum independent of the process hash seed (summing over a set had made the last bits vary between processes) and added the cross-process determinism tests. These changes were made after the exploratory development results had been seen. They cover edge-case handling and reproducibility only; no hypothesis test, margin, statistic or definition used for H1 to H5 was changed, and the dry run repeated at 0b3da97 reproduced every p-value of the first dry run.

A change to the analysis code after the confirmation set is unsealed is a deviation and is recorded in docs/DECISIONS.md.

### 7.2 Definitions used by the code

- Run-level outer AUC at a checkpoint: the outer AUC of the subset the run returns at that checkpoint; for C4, the knee record at that checkpoint.
- Optimism gap of a run: the standard inner AUC of the returned subset (unpenalized mean over the fixed 3-fold inner split) minus its outer AUC.
- Per-dataset score: the mean over all runs of that dataset and configuration (outer folds x search seeds; for random-forest importance, outer folds x model seeds). Primary checkpoint: 2,000 evaluations.
- Wilcoxon signed-rank tests: scipy.stats.wilcoxon with zero_method = "wilcox" and method = "auto"; the method used and the number of non-zero differences are recorded.
- H1: per-dataset mean optimism gap over the 12 C0 configurations; alternative greater than 0.
- H2: as in section 3; alternative greater than 0.
- H3: d = GA + C4 minus L1. p_low from d + 0.02 (alternative greater), p_up from d - 0.02 (alternative less), p_TOST = max(p_low, p_up). Two-sided test on d also reported.
- H4: statsmodels MixedLM on run-level outer AUC; rows are runs x checkpoints for GA, binary PSO (S), DE (S) and GWO (S) under C0 to C4. Fixed effects C(searcher) * C(control) + log(budget) + log(p/n); random intercept for dataset and a variance component for outer partition (repeat, fold) within dataset; maximum likelihood (reml = False). Likelihood-ratio test against the model without the interaction; degrees of freedom = difference in fixed-effect count. A fit is accepted when statsmodels reports converged = True; warnings are recorded verbatim and do not reject a fit. lbfgs is tried first and powell only if lbfgs does not converge; if neither converges, H4 is reported as not estimable and enters the Holm family with p = 1. In the development dry run every fit converged with lbfgs and logged "The MLE may be on the boundary of the parameter space", consistent with a near-zero partition variance. The same model on the optimism gap is descriptive only.
- H5 (amendment 1): d = mean over GWO (S, V) and WOA (S, V) C0 minus mean over GA, binary PSO (S, V) and DE (S, V) C0; p from d - 0.01 with alternative less. The full TOST is reported as context.
- Holm correction across H1 (p), H2 (p), H3 (p_TOST), H4 (LRT p) and H5 (p) at alpha = 0.05 (statsmodels multipletests); raw and adjusted p-values are reported for all five.
- Effect sizes: mean and median of d, 95% percentile bootstrap CI of the mean over datasets (10,000 resamples, numpy default_rng(20261113)), and the matched-pairs rank-biserial correlation.
- Stability: Nogueira et al. (2018) across the 10 outer folds per dataset, configuration and seed index, then averaged over seeds; features identified by their original column; d = number of features after the data-handling rules. The index is undefined when every feature or no feature is selected on average, or with fewer than 2 folds. Undefined seed values are left out of the average over seeds and are not reported separately; a dataset is undefined for a configuration only when all its seeds are undefined. Undefined datasets are left out of the mean over datasets, and their number is reported per configuration.
- Secondary families (not in the Holm family of section 8 of stage 1): random search against each of the 11 guided C0 searchers (two-sided, Holm within the family); Brier score for the H1 to H5 comparisons (descriptive); Friedman test across all selectors with Holm-corrected pairwise Wilcoxon tests (context only); S1 and S2 as in stage 1, section 11 and section 6 above.

## 8. Changes after stage 1 was frozen

Code commits touching src/ or configs/ after 8fa82c7:

| Commit | Date | Change | Effect on results |
| --- | --- | --- | --- |
| a71df32 | 2026-11-10 | Searchers, evaluator, transfer functions, archive, inner fitness | None; before any run |
| f44ad90 | 2026-11-10 | Verification report builder | Report only |
| c6bbdfd | 2026-11-11 | Dispersion section in the verification report | Report only |
| 4868766 | 2026-11-11 | Pipeline, partitions, baselines, controls, experiment; evaluator changes | None; before any study run. The fast test suite rerun after a1535d6 passed with structural-bias results identical to docs/VERIFICATION.md |
| 4f10b80 | 2026-11-11 | Runner, validation, compute estimator, pilot plan | None; before any run |
| 468d682 | 2026-11-12 | bench_scaling.py (synthetic data) | None |
| 0e21454 | 2026-11-12 | Development plan; C4 logged in-run; plan name in run paths; stage2_select.py | C0 runs also log C4 records; search, controls and baselines unchanged. Pilot runs (087d30d) lack C4 |
| 3b69ca5 | 2026-11-12 | Per-plan release validation | Validation only |
| c38cfb3 | 2026-11-12 | project_compute.py | None |
| a1535d6 | 2026-11-12 | Elastic net max_iter 2,000 to 10,000; diagnostic script; plan dev_enet_fix | Elastic net only (section 9) |
| f40eb50 | 2026-11-13 | Pipeline option prefilter (default true), for S2 | None on the default path |
| 5873f6f | 2026-11-13 | Confirmation plans (core and S2), generated from metadata; estimator handles prefilter | None |
| 12d9d4f | 2026-11-13 | Analysis package; development dry-run configuration | None on runs |
| 44d6c32 | 2026-11-13 | Confirmation analysis configuration; mixed-model acceptance rule; verbatim recording of fit warnings; --output-dir option; full-design synthetic test | None on runs |
| 0b3da97 | 2026-11-13 | Stability sum in sorted order; cross-process determinism tests | Last-bit change in stability values only |

Verification (stage 1, section 9), recorded in docs/VERIFICATION.md and docs/DECISIONS.md:

- The mirrored-encoding test had no power to detect bias. It was replaced by the pure-noise structural-bias test (Kononova et al., 2015) with a dispersion diagnostic.
- The OneMax criterion "best fitness increases with budget" was made operational per searcher.
- woa_S passed the encoding location test (Mann-Whitney p = 0.50) but showed corner mass under pure noise (7 runs at size 20 or below and 6 at 180 or above; Brown-Forsythe p = 0.051 against random search). On OneMax its starting best fitness is already 1.0, which reflects the same pull toward the all-ones corner rather than search ability. It is kept unchanged, because it follows the canonical WOA update with the standard S-shaped transfer; the bias is reported as a finding (docs/DECISIONS.md, 2026-11-11).

Hypotheses: H5 is tested by the upper arm of TOST only (docs/PREREG_STAGE1_AMENDMENT1.md).

Runs: the first pilot attempt was interrupted by system hibernation and discarded in full. No other run was discarded.

## 9. Data health in the development run

- No convergence warning in any searcher configuration; every run used one BLAS and one OpenMP thread.
- The elastic net baseline produced 1,228 convergence warnings in all 60 of its runs (1,221 in inner tuning fits, 7 in final fits, all at C of 0.4281 or above). Its max_iter was raised from 2,000 to 10,000 (commit a1535d6). The rule was fixed before the rerun and adopted regardless of outcome, and no AUC was read for the decision. The 60 development elastic net runs were repeated as plan dev_enet_fix with zero warnings; the original runs are kept unchanged and are not used by the analysis (docs/DECISIONS.md, 2026-11-13).
- The 3 empty subsets on Ahn2021 (section 5) come from strong regularization on that dataset, not from a fault.
- No C1 run stopped before the first checkpoint; all 720 C1 runs reached at least 404 evaluations.

## 10. Unsealing

This document and docs/PREREG_STAGE2.sha256 are committed together. The confirmation set is loaded for the first time after that commit. The confirmation run uses the same machine and environment as the development run (environment/), at the commit recorded in its run files.
