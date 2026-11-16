# Pre-registration, stage 1

Study: Search less, generalize more: overfitting-aware evolutionary feature selection for radiomics
Author: Hussein Fadhil Abedi (sole author)
Target journal: Swarm and Evolutionary Computation (Elsevier)
Date written: 2026-11-10
Status at writing: data downloaded and audited; 43 datasets included; dataset-level split sealed (12 development, 31 confirmation); no model trained; no feature-target statistic computed.

This document is frozen by committing it together with its SHA-256 in docs/PREREG_STAGE1.sha256. Stage 2 (docs/PREREG_STAGE2.md) is written after the development runs and before any confirmation dataset is loaded. Stage 2 may only fix the items listed in section 10. It may not change the questions, hypotheses, outcomes, margins or tests in this document.

## 1. Research questions

- RQ1. How large is the optimism of uncontrolled wrapper feature selection on radiomics data, and how does it change with the search budget and with log(p/n)?
- RQ2. Do generalization controls reduce that optimism and raise out-of-sample AUC at the same search budget?
- RQ3. Does the best controlled wrapper match or beat L1-regularized logistic regression?
- RQ4. Does the effect of a control depend on the search algorithm?
- RQ5. Do metaphor-based searchers (GWO, WOA) perform differently from classical ones (GA, binary PSO, DE) at a matched budget?

## 2. Data

- Collection: radMLBench 1.2 (Demircioglu, 2024). 50 datasets downloaded; 7 excluded under the inclusion criteria and the license review (docs/DECISIONS.md, data/meta/data_handling_rules.json); 43 included.
- Units of inference: datasets. The BraTS-2021 / UCSF-PDGM / UPENN-GBM cluster was assigned to one group as a unit.
- Development set (12 datasets): used for the pilot, for tuning control hyperparameters and for choosing the primary controlled wrapper. Results on development datasets are exploratory.
- Confirmation set (31 datasets): sealed by src/data_access.py until docs/PREREG_STAGE2.sha256 is committed. All confirmatory tests use confirmation datasets only.
- All data handling follows data/meta/data_handling_rules.json, applied by src/data_access.py. No other transformation is applied outside the cross-validation folds.

## 3. Pipeline inside each outer training fold

All steps below are fitted on the outer-training part only and applied to the outer-test part.

1. Remove features that are constant in the outer-training part.
2. Median imputation (the audit found no missing values; the step is kept as a safeguard).
3. Standardization to zero mean and unit variance.
4. Univariate pre-filter: keep the top K features ranked by the ANOVA F statistic, with K = min(200, number of remaining features). Every selector, search-based or not, starts from the same K-feature pool.

## 4. Evaluation protocol

- Outer loop: stratified 5-fold cross-validation, repeated 2 times (10 outer folds per dataset). Partition seeds are derived from the SHA-256 of the dataset name and the repeat index; partition indices are saved and shared by every method.
- Inner fitness (all wrappers): mean AUC over a fixed stratified 3-fold split of the outer-training part, using L2-regularized logistic regression (C = 1.0, lbfgs, max_iter = 2000, no class weighting) fitted on the candidate subset. The inner split is fixed per outer fold and shared by every method. An empty subset has fitness 0.5.
- Outer evaluation (all selectors): L2-regularized logistic regression with the same settings, fitted on the full outer-training part using the selected subset, scored on the outer-test part.
- Logistic regression with lbfgs is deterministic, so the inner and outer models have no model seed. The only model with a random state is the random forest used by one baseline (section 6); it has its own model seed, independent of search seeds.
- Search seeds: 3 per outer fold for confirmation runs (30 runs per configuration and dataset), derived deterministically from dataset, repeat, fold and seed index, and logged for every run.
- Budget: 2,000 evaluations per run. One evaluation is the scoring of one candidate subset, including subsets a control scores for its own purposes. Each run logs its best-so-far subset at 250, 500, 1,000 and 2,000 evaluations.
- Logged per run and checkpoint: outer AUC (primary), best inner fitness, Brier score, subset size, the subset itself, evaluations used, wall-clock time, and the inner fitness and size of every evaluated subset (needed for control C4).

## 5. Searchers

Every searcher works on a binary mask over the K-feature pool. Initial masks are drawn bit by bit with probability 0.5.

| Searcher | Settings |
| --- | --- |
| Random subset search | 2,000 masks; subset size drawn uniformly from 1 to K, then features drawn uniformly |
| Bit-flip hill climbing | One random bit flipped per step; move accepted if fitness is greater than or equal |
| (1+1)-EA | Standard bit mutation with rate 1/K; offspring accepted if fitness is greater than or equal |
| Genetic algorithm | Population 20, 100 generations; binary tournament; uniform crossover with probability 0.9; bit-flip mutation 1/K; elitism of 1 |
| Binary PSO | 20 particles, 100 iterations; inertia 0.9 to 0.4 linearly; c1 = c2 = 2; velocity clamp 6 |
| Differential evolution | DE/rand/1/bin; population 20, 100 generations; F = 0.5; CR = 0.9; continuous positions in [-6, 6] |
| Grey wolf optimizer | 20 wolves, 100 iterations; a decreases linearly from 2 to 0 |
| Whale optimization algorithm | 20 whales, 100 iterations; a decreases linearly from 2 to 0; spiral constant b = 1 |

Binary PSO, DE, GWO and WOA are run with two transfer functions: S-shaped (sigmoid) and V-shaped (|tanh|). This gives 12 uncontrolled configurations (control C0). Implementations are checked against MEALPY and pass the sanity tests in section 9 before any data run.

## 6. Non-search selectors

Hyperparameters of these selectors are tuned by inner 3-fold cross-validated AUC on the same inner split. Their cost is not counted against the 2,000-evaluation budget, which applies to searchers only. After selection, the outer model is the logistic regression of section 4, so the selector is the only thing that differs between methods.

| Selector | Settings |
| --- | --- |
| All K features | No selection |
| L1-regularized logistic regression | C over 20 log-spaced values from 0.001 to 100; selected features are those with non-zero coefficients |
| Elastic net logistic regression | l1_ratio = 0.5; same C grid; saga solver |
| F-test top-k | k in {5, 10, 20, 50, 100, 200}, capped at K |
| mRMR top-k | FCQ variant; same k grid |
| Random-forest importance top-k | 500 trees, impurity importance; same k grid; own model seed |

## 7. Generalization controls

Controls C1 to C3 are applied to GA, binary PSO (S), DE (S) and GWO (S). C4 is applied to every searcher.

- C1, early stopping on a sealed selection fold: 20% of the outer-training part (stratified) is held aside; the inner fitness uses the other 80%. Every 100 evaluations the best-so-far subset is scored on the selection fold (one evaluation each). The run returns the subset with the best selection-fold AUC and stops after P checkpoints without improvement. P is chosen on development data from {3, 5, 10}.
- C2, variance-penalized fitness: fitness = mean minus lambda times standard deviation of AUC over a 3 x 3 repeated stratified inner split. lambda is chosen on development data from {0.5, 1, 2}.
- C3, flatness fitness: the fitness of a subset is the mean of its own inner fitness and that of r random one-bit neighbours; each neighbour counts as one evaluation. r is chosen on development data from {2, 4, 8}.
- C4, knee selection: from the archive of all subsets a run evaluated, take the non-dominated front on (inner fitness, subset size) and return the knee point, defined as the point with the largest perpendicular distance to the line joining the two extremes of the front. C4 uses the C0 runs and costs no extra evaluations.

Each control hyperparameter is chosen as a single value for all searchers (section 10).

## 8. Hypotheses and confirmatory tests

All confirmatory tests use the 31 confirmation datasets. The per-dataset score is the mean over outer folds and search seeds at the 2,000-evaluation checkpoint, unless stated otherwise. One Holm correction is applied across the five primary hypotheses at alpha = 0.05.

| ID | Prediction | Test |
| --- | --- | --- |
| H1 | Uncontrolled wrappers are optimistic: best inner fitness exceeds outer AUC | One-sided Wilcoxon signed-rank on the per-dataset mean optimism gap over the 12 C0 configurations |
| H2 | The primary control (chosen in stage 2) raises outer AUC compared with the same searchers without control | One-sided Wilcoxon signed-rank on the per-dataset difference, averaged over GA, binary PSO (S), DE (S) and GWO (S) |
| H3 | The primary controlled wrapper (chosen in stage 2) is equivalent to L1-regularized logistic regression | Two one-sided tests (TOST) on the per-dataset outer AUC difference, margin 0.02, using the Wilcoxon signed-rank form; a two-sided Wilcoxon test is also reported |
| H4 | The effect of controls does not depend on the searcher | Likelihood-ratio test of the searcher-by-control interaction in the mixed model below; a non-significant result is reported as "no detectable interaction", not as absence of interaction |
| H5 | Metaphor-based searchers do not outperform classical ones | TOST on the per-dataset difference between the mean outer AUC of GWO and WOA and that of GA, binary PSO and DE (C0, both transfer functions where relevant), margin 0.01 |

Mixed model (H4 and the RQ1 description): outer AUC (and, separately, optimism gap) as the response; fixed effects for searcher, control, log budget, log(p/n) and searcher x control; random intercepts for dataset and for outer partition within dataset.

Wording rule: "equivalent" is used only when TOST succeeds. Otherwise results are described as "no detectable difference" or by their direction.

Effect sizes with 95% bootstrap confidence intervals (10,000 resamples over datasets) are reported for every primary comparison.

## 9. Code verification before any data run

- Each searcher reproduces MEALPY behaviour on continuous benchmark functions where an equivalent exists, within tolerance.
- OneMax (K = 200): every searcher's best fitness increases with budget.
- Planted subset: synthetic data with 10 informative features among 200; recovery rate is reported, not used as a gate.
- Mirrored encoding: rerunning a searcher with the meaning of 0 and 1 swapped must give the same distribution of subset sizes, within sampling error; a systematic difference indicates structural bias and is fixed before the main runs.
- Budget accounting: a test confirms that every scored subset, including control evaluations, increments the counter.

## 10. What stage 2 will fix, from development data only

- Control hyperparameters P (C1), lambda (C2) and r (C3): the value with the highest mean outer AUC over development datasets and the four controlled searchers.
- The primary control for H2: the control with the highest mean outer AUC over development datasets.
- The primary controlled wrapper for H3: the searcher-control pair with the highest mean outer AUC over development datasets.
- The compute tier: core (all C0 configurations, all non-search selectors, the four controls with their chosen values) or reduced (only if pilot timings show the core tier cannot finish in 10 days on the study machine; then the V-shaped variants are dropped first).
- Bug fixes found during development runs. Any such fix is listed with its commit.

Stage 2 may not change sections 1, 4 (except bug fixes), 8 or 11.

## 11. Secondary and sensitivity analyses

- Anytime curves: outer AUC and optimism gap at 250, 500, 1,000 and 2,000 evaluations.
- Random subset search compared with every guided searcher at matched budget.
- Subset size and stability (Nogueira et al., 2018) across the 10 outer folds.
- Brier score for every comparison in section 8.
- S1: all primary tests repeated without the sensitivity-exclusion set (Arita2018, Hosny2018A, Hosny2018B, Hosny2018C, Keek2020, LNDb, UCSF-PDGM, Wang2024), because their released features were probably imputed before release.
- S2 (if compute allows, decided in stage 2): the H3 comparison repeated without the pre-filter on confirmation datasets with at most 1,000 features.
- Friedman test with Holm-corrected pairwise comparisons across all selectors, as context only.

## 12. Deviations

Any change after this document is frozen is recorded in docs/DECISIONS.md with the date, the reason and the commit. Any analysis not listed here is labelled post hoc in the paper.
