# Decision log

- 2026-11-06: target journal Swarm and Evolutionary Computation (Elsevier); sole author Hussein Fadhil Abedi, all CRediT roles; BFOA excluded.
- 2026-11-08: study designed as overfitting-aware evolutionary feature selection on radMLBench.
- 2026-11-09: affiliation not yet decided.
- 2026-11-09: raw data (data/raw/) is not committed or redistributed, because the source licenses are unknown; readers re-download the datasets with src/download_data.py and verify them against the SHA-256 values in data/meta/radmlbench_index.csv.

## 2026-11-09: post-audit, pre-model data-handling decisions

Made after inspecting data properties in the audit (data/meta/audit.csv) and before any model was trained or any outcome association was computed. Machine-readable form: data/meta/data_handling_rules.json, applied automatically by src/data_access.py.

- Excluded (criterion 2, minority class below 20): Colorectal-Liver-Metastases, Granata2024, HCC-TACE-Seg, Li2020, QIN-HEADNECK.
- Excluded (criterion 4, lesion-level rows without patient ID): Zhang2023.
- Arita2018: drop the 10 IDHmt_* and IDHwt_* columns (location maps built from outcome labels; target leakage by construction).
- Granata2021: drop LESION_TYPE (clinical variable, not a radiomic feature).
- Song2020: drop the exact duplicate row (including target), keeping the first occurrence.
- Hosny2018B, Dong2022: strip leading and trailing whitespace from IDs.
- Fusco2022, Wang2024: convert the boolean target to integers (True -> 1, False -> 0).
- Criterion 4 resolved as pass: Wang2024 (first biopsy only, one row per patient); Hosny2018B (IDs are patient IDs of the form LUNG1-xxx, one row per patient, the one missing ID is kept).
- Sensitivity exclusion set: Arita2018, Hosny2018A, Hosny2018B, Hosny2018C, Keek2020, LNDb, UCSF-PDGM, Wang2024 (metadata reports missing values but the data has none, likely imputed on the full dataset before release); a pre-registered sensitivity analysis repeats the main comparisons without these datasets.
- License review: pending.
- No model has been trained and no feature-target statistic has been computed.

## 2026-11-10: split clusters and known limitations (pre-model)

- Split cluster BraTS-2021, UPENN-GBM, UCSF-PDGM: possible patient overlap (BraTS 2021 includes UPenn and UCSF cases); the cluster is assigned to development or confirmation as one unit.
- Split cluster Arita2018, Sasaki2019: possible patient overlap (both from the KNBTG consortium); the cluster is assigned to development or confirmation as one unit. The ID-only check (src/id_checks.py) found 0 shared normalised KNBTG IDs; the cluster is kept as a precaution.
- Known limitation, Petrillo2023: the radMLBench curator chose the endpoint with the highest AUC in the source study, so absolute AUC may be optimistic; this affects all compared methods alike.
- No model has been trained and no feature-target statistic has been computed.

## 2026-11-10: license review and final inclusion (43 datasets)

- License review complete; per-dataset licence, category, evidence and decision in data/meta/license_decisions.csv (evidence files in data/meta/license_evidence/).
- Excluded, criterion 3: Deng2023 (no license file in the GitHub repository; the article, not open access, states data are provided upon journal request only).
- Split cluster Arita2018/Sasaki2019 removed after the ID-only check: 0 overlapping normalised IDs; grade II/III glioma vs newly diagnosed glioblastoma. The BraTS-2021/UPENN-GBM/UCSF-PDGM cluster is kept.
- Criterion 4 resolved as pass: Hunter2023 (520 distinct patient prefixes, one nodule per patient); UPENN-GBM (187 distinct patients, all preoperative _11 scans).
- Known limitation, Veeraraghavan2020: radMLBench metadata lists MRI/breast cancer, but the source study is contrast-enhanced CT in endometrial cancer; descriptive only.
- Included by licence category: open 22, open repository licence 2, non-commercial 4, custom research terms 7, TCIA restricted images with derived features 4, no license file under the PLOS data availability policy 4.
- Final inclusion: 43 of 50 datasets (excluded: Colorectal-Liver-Metastases, Granata2024, HCC-TACE-Seg, Li2020, QIN-HEADNECK, Zhang2023, Deng2023).
- All included datasets are used for non-commercial research only; raw data are not redistributed; every source is cited.
- No model has been trained and no feature-target statistic has been computed.

## 2026-11-10: git history rewrite (privacy)

- Before anything was pushed, data/meta/audit.csv and data/meta/audit_notes.md were removed from every commit before "audit: mask IDs", because those earlier versions contained real dataset IDs (including initials). The audit outputs now first appear, with masked IDs, in "audit: mask IDs".
- Every other file, every commit message, and every author and committer name, email and date are unchanged; commit hashes from "radMLBench audit (no models, no outcome statistics)" onward changed. The final tree is identical to the tree before the rewrite.
- The backup refs were deleted, the reflog expired and git gc --prune=now run, so the unmasked versions are no longer stored in the repository.
- Verified: no commit contains "KNBTG" or "LUNG1-" in data/meta/audit.csv or data/meta/audit_notes.md; docs/INCLUSION_CRITERIA.md first appears in "inclusion criteria and decision log (before data download)", dated 2026-11-09T20:56:53+03:00, before the first dataset download at 2026-11-09T18:06:23+00:00.

## 2026-11-10: full ID scan and second history rewrite (privacy)

- src/scan_ids.py searched every tracked file, every blob in the git history and every commit message for the real IDs of all 50 datasets (IDs of at least 5 characters containing a letter, underscore or hyphen; whole-token matches; only masked forms are printed).
- First scan: one real ID found, used as a docstring example in mask_id() in src/data_audit.py (from "audit: mask IDs" onward). It was replaced by the made-up example 'EXAMPLE-123' -> 'AAAAAAA-999'.
- Before the rewrite, the tests that loaded Arita2018 and Song2020 were replaced by tests of the pure function apply_rules() on synthetic DataFrames, so test outcomes do not reveal the group of any dataset; the only real dataset loaded by the tests is the first development dataset.
- History was rewritten once more (nothing had been pushed): only src/data_audit.py changed, in the commits that contained the example; every other file, every commit message, and every author and committer name, email and date are unchanged. The backup refs were deleted, the reflog expired and git gc --prune=now run.
- Second scan (22 commits, 129 blobs, 22 commit messages): no matches.
- No model has been trained and no feature-target statistic has been computed.

## 2026-11-10: sealed dataset-level split (development/confirmation)

- Created by src/make_split.py (commit 29f38ac) from data/meta/audit.csv and data/meta/data_handling_rules.json only; no parquet file was opened. Full record in data/split/split_record.json; list hashes in data/split/split.sha256.
- Units: the 43 included datasets form 41 units; the BraTS-2021/UCSF-PDGM/UPENN-GBM cluster is one unit (log_p_over_n = mean of its members).
- Stratification: tertiles of post-rule log_p_over_n (numpy.quantile, default method; q1 = 1.230473, q2 = 1.691978); 14, 13 and 14 units per tertile. Within each tertile, units sorted by name and shuffled with numpy.random.default_rng(20261108); the first max(1, floor(k/3)) units go to development.
- Result: development 12 datasets (4 units per tertile), confirmation 31 datasets.
- Seal: src/data_access.py refuses to load any confirmation dataset until docs/PREREG_STAGE2.sha256 exists and is tracked by git. tests/test_seal.py passed (10 tests); the only real dataset loaded by the tests is the first development dataset.
- No model has been trained and no feature-target statistic has been computed.

## 2026-11-10: stage-1 pre-registration frozen (Gate 1)

- docs/PREREG_STAGE1.md frozen, committed together with docs/PREREG_STAGE1.sha256.
- SHA-256 (LF bytes, computed in Python): d5c7039a69b84d8e8eac8f2ec35b297de944eebe5b08a2073217cd47faf6302b
- Frozen after the sealed dataset-level split and before any model was trained or any confirmation dataset was loaded; no feature-target statistic has been computed.
- Any later change to the stage-1 content is a deviation and is recorded here with date, reason and commit.

## 2026-11-10: stage-1 pre-registration, amendment 1 (H5 one-sided test, before any model)

- docs/PREREG_STAGE1_AMENDMENT1.md, SHA-256 (LF bytes, computed in Python): 1599d12f5a5304668e51f6dbeabd933e010e89ac88af13de12a21b9ad7d65722
- Pre-data clarification, not results-driven: made after stage 1 was frozen and before any model was trained or any outcome was computed.
- Reason: the H5 prediction is one-sided, but section 8 specified two-sided equivalence (TOST); a clearly worse result for the metaphor-based searchers would fail TOST while matching the prediction.
- Change: the confirmatory H5 test is the upper arm of the TOST only (one-sided Wilcoxon signed-rank, per-dataset difference GWO/WOA minus GA/binary PSO/DE below +0.01); the full two-sided TOST (margin 0.01) and a two-sided Wilcoxon test are secondary. Nothing else in stage 1 changes.
- docs/PREREG_STAGE1.md is not modified (SHA-256 still d5c7039a69b84d8e8eac8f2ec35b297de944eebe5b08a2073217cd47faf6302b).

## 2026-11-10: part 5 implementation decisions (synthetic problems only, before any real-data run)

All decisions below were made on synthetic problems (OneMax, NK landscapes, pure-noise fitness, Sphere, Rastrigin, simulated logistic data) before any model was trained on real data. No real dataset was loaded.

- MEALPY-aligned cores: the continuous cores of PSO (for BPSO), DE, GWO and WOA follow the MEALPY 3.0.3 "Original" variants (LDW_PSO for the PSO core, OriginalDE strategy 0, OriginalGWO, OriginalWOA), including per-agent greedy replacement; the differences from the original papers are listed in each searcher's docstring (src/fsbench/searchers/). The binary BPSO itself follows Kennedy & Eberhart (1997) without greedy replacement.
- MEALPY 3.0.3 declares numpy<=1.26.0; installing it downgraded numpy, pandas and scipy, which were restored to the versions used for the download and audit (numpy 2.5.4, pandas 3.1.1, scipy 1.18.1). MEALPY runs correctly under these versions; the remaining declared conflict is recorded in environment/pip_check.txt. MEALPY is used only in the verification test tests/test_mealpy.py, never in study runs.
- V-shaped transfer (|tanh|, Mirjalili & Lewis 2013), input per searcher: BPSO, the velocity; GWO and WOA, the step x_new - x_old; DE, the scaled difference F(x_r2 - x_r3) on the crossed coordinates, flipping the base vector's bits (other coordinates keep the target's bits). Applying |tanh| to the position itself (bounds [-6, 6]) flipped almost every bit and stopped DE, GWO and WOA from improving on OneMax. S-shaped (sigmoid): BPSO, the velocity; DE, GWO and WOA, the position.
- OneMax (section 9, "every searcher's best fitness increases with budget"), operationalized per searcher over 5 seeds: (a) for every seed, best at 2,000 evaluations >= best at 250 (a decrease is a bug); and (b) either the mean best at 2,000 > the mean best at 250, or every seed reached the optimum by 250. All per-seed values are reported, including bpso_S seed 0 stagnating at 0.690, the documented behaviour of the original binary PSO (a bit that agrees with pbest and gbest has its velocity decay to 0 and is re-randomized by sigmoid(0) = 0.5).
- Mirrored-encoding check (section 9): the earlier symmetric-landscape test had no power (every searcher returned exactly K/2) and was replaced by the structural-bias test of Kononova et al. (2015) adapted to binary masks: pure-noise fitness, deterministic per mask (uniform in [0, 1] from a BLAKE2b hash of the mask bytes and the seed); K = 200; budget 2,000; 30 seeds; original vs inverted encoding compared with a two-sided Mann-Whitney U (must be p > 0.01); mean returned size tested against K/2 with a one-sample Wilcoxon (p reported). It runs in about 70 s and is not marked slow.
- Knee rule for control C4: both objectives (inner fitness, subset size) are min-max normalized over the non-dominated front before the perpendicular distance is computed; ties go to the smaller subset.
- Verification results are saved as raw JSON files in results/verification/ by the tests; docs/VERIFICATION.md (supplement) is generated from them by src/make_verification_report.py.
- The commit "searchers, evaluator and verification tests (not yet run)" (a71df32) was preceded by a local run of the fast synthetic tests by Claude Code during development; the message refers to the author's run.

## 2026-11-11: woa_S corner bias under pure noise (finding; no searcher changed)

- Finding (synthetic pure-noise fitness, K = 200, budget 2,000, 30 seeds; tests/test_dispersion.py, raw files results/verification/dispersion_*.json, docs/VERIFICATION.md section 3b): woa_S returns corner masks. SD of the returned subset size 71.3 (min 1, max 195; 7 seeds <= 20 features, 6 seeds >= 180), not distinguishable from random_search, whose sizes are uniform on 1..200 by construction (Brown-Forsythe p = 0.051). The other ten searchers, woa_V included, have SD 5.6 to 7.9, near the Binomial(200, 0.5) SD of 7.1 (Brown-Forsythe vs random_search p <= 1.1e-09). The location tests of section 3 (Mann-Whitney between encodings, Wilcoxon against K/2) pass for woa_S because its mean stays near K/2; they do not detect dispersion.
- Mechanism as measured: from iteration 10 on, 0.27 to 0.33 of woa_S position coordinates sit at the bounds (+-6), and within each whale at least 0.98 of its saturated coordinates sit on the same bound; the sign coherence of the proposed moves rises from 0.69 to 0.97 over the run. This matches the author's hypothesis: the spiral move X* + c |X* - X| uses one scalar c for all coordinates with a non-negative distance term, and the encircling and exploration moves share one scalar A, so coordinates move in the same direction and saturate together. woa_V shows the same position saturation (0.27 to 0.32 at the bounds; same-bound share >= 0.99; coherence 0.69 to 0.95) but no size bias, because the V-shaped rule uses the step and is symmetric in its sign, whereas the S-shaped rule maps saturated positions directly to fixed bits.
- Decision: woa_S is kept unchanged. It follows the canonical WOA update (verified against MEALPY 3.0.3 OriginalWOA) with the standard S-shaped transfer; changing it would make it a different algorithm from the one the study evaluates. The bias is reported as a finding. No searcher was changed.
- Slow verification run (planted subset): pytest reported a wall time of 23 h 21 min, which includes machine sleep; the measured compute of the 36 runs was 50 min (2998 s, summed from the raw files). Standby on AC power was then disabled by the author (powercfg /change standby-timeout-ac 0) for later runs.
- No model has been trained on real data and no feature-target statistic has been computed on real data.

## 2026-11-11: part 6a implementation decisions (pre-data)

Approved by the author; made on synthetic data only, before any model was trained on real data (commit "study pipeline components (not yet run on real data)").

- Baseline tuning (section 6): every hyperparameter value is scored by the inner 3-fold cross-validated AUC on the fixed inner split. Inside each inner fold the selector is refitted on the inner-training part, the section 4 L2 logistic regression is fitted on the selected features, and the AUC is computed on the inner-test part (an empty selection scores 0.5 in that fold); this applies to L1 and elastic net as well as to the top-k selectors. Ties go to the sparser setting (smaller k, smaller C). The selector is then refitted on the whole outer-training part with the chosen value.
- Fixed solver random_state = 0 for liblinear (L1) and saga (elastic net). This is a fixed constant, not a study seed; the only model with a model seed is the random forest.
- C1: the selection fold is scored after every 100 search evaluations; selection scorings do not count toward this 100-evaluation cadence but each counts as one evaluation against the total budget of 2,000 (section 7). The selection-fold model is the L2 logistic regression fitted on the whole 80% C1 fit part. Improvement means a strictly higher selection-fold AUC (the first such subset is kept on ties). After an early stop, the later checkpoints report the returned subset and are flagged filled_after_stop.
- C2: the standard deviation of the 9 AUC values uses ddof = 0.
- C3: the r neighbours flip r distinct bits chosen uniformly, drawn from a random stream that is derived from the search seed but independent of the searcher's own stream (the searcher's random numbers are the same as in C0); the best-so-far is tracked on the smoothed value; the archive stores the raw inner fitness of every scored subset.
- Evaluator split: fsbench.Evaluator has score() (one counted, archived evaluation) and returned() (the subset the run returns: the best inner fitness for C0, the control's choice for C1 and C3). C0 behaviour is unchanged (all earlier tests pass; determinism masks identical before and after).
- FSBENCH_RESULTS_DIR: an environment variable that redirects the raw verification results written by tests/conftest.py, so that development runs by Claude Code never overwrite the committed results in results/verification/.
- Restored metadata: a development run before that override overwrote the 12 committed results/verification/determinism_*.json files; only their metadata (git commit, dirty flag, UTC time) differed, the results were identical; the committed files were restored from git unchanged.
- Standard inner AUC: computed, outside the budget, for the returned subset and for every checkpoint subset.

## 2026-11-12: pilot attempt 1 discarded

- Pilot attempt 1 (configs/pilot.yaml, 480 planned runs, 6 workers, commit 4f10b80) started on 2026-11-11 at 17:48:29 UTC (earliest run start). 390 runs were finished and recorded in results/MANIFEST.csv (last at 18:32:51 UTC); 431 run files had been written in total (41 finished by workers but never recorded by the main process); the last run started at 21:31:23 UTC. No temporary files were left.
- Reason: the author put the machine into hibernate during the pilot and it did not resume correctly, so the run times are contaminated; the pilot exists to measure run times (section 10).
- No performance result (AUC, Brier score, subset) was inspected by the author or by Claude Code; only run counts, start and finish times, plan names, status values and commits were read to write this entry.
- Deleted: results/runs/development/ (431 files), results/MANIFEST.csv and results/progress.csv (pilot rows only), and results/partitions/ (3 files; deleted rather than verified, because verifying them would have required loading the datasets' labels; the runner recreates them deterministically from the dataset names). The pilot is rerun from scratch with the same plan and commit-checked code.

## 2026-11-12: development protocol, C4 logging and stage-2 selection rules (before any development run)

Pilot attempt 2 (configs/pilot.yaml, commit 087d30d) finished 480 runs with 0 errors in 0.95 h at 6 workers; committed in 5c25f5c. Health check (no AUC, Brier score or subset was read): convergence warnings only in the elastic-net baseline (289 in total, in all 15 runs; no change made); no baseline returned an empty subset; all 60 C1 runs (P = 5) stopped early, none before 250 evaluations (earliest possible stop: 606), so the 1000 and 2000 checkpoints of C1 often hold the stopped subset; every run recorded 1 BLAS and 1 OpenMP thread.

- Development protocol (decided by the author on 2026-11-12; replaces the full-protocol plan mentioned the same day, before any development run). Reason: compute; the development runs serve only the stage-2 choices of section 10, and the confirmatory tests on the 31 confirmation datasets are unaffected. PREREG_STAGE1 does not fix the development protocol. configs/development.yaml: the 12 development datasets; repeat 0 only (5 outer folds); search seed index 0 only; C0 for ga, bpso_S, de_S and gwo_S (their C0 runs also give C4); C1, C2 and C3 on the same four searchers at all three grid values (P in 3, 5, 10; lambda in 0.5, 1, 2; r in 2, 4, 8); the six baselines (random forest with 3 model seeds) for context. 2880 units. Projected from the pilot fit (src/estimate_compute.py --plan configs/development.yaml): 43.4 CPU hours, about 7.2 h at 6 workers; every development dataset has n_train within the pilot range (up to 166), so no extrapolation in n; C1 at P = 3 and 10 is approximated by the P = 5 fit.
- C4 logging (pre-data implementation correction; no development run and no C4 result existed). PREREG_STAGE1 section 7 says C4 uses the C0 runs. The C0 run files stored only the inner fitness and size of each archived subset, not the subsets, so the C4 subset could not be scored on the outer fold. Correction (src/fsbench/experiment.py): the C0 runs now also store what C4 needs. For each checkpoint c in 250, 500, 1,000 and 2,000, every C0 run logs under "c4" the knee subset of the archive of its first c evaluations (fsbench.archive.knee_point, the min-max normalized rule recorded on 2026-11-10), with its size, subset, archive index, standard inner AUC, outer AUC and Brier score, so the anytime analysis (section 11) covers C4. The knee is chosen from the archive's inner fitness and size only; the outer fold is scored after the choice, as for every other method. No extra evaluations; wall time in timings["c4_s"]. Tests: tests/test_experiment.py (test_c4_knee_records: the logged knee equals knee_point of the archive prefix; test_c4_knee_choice_ignores_outer_test_labels: flipping the outer-test labels changes only the outer AUC and Brier score of the C4 records).
- Reuse of pilot run files: the author allowed reuse only if src/fsbench/ is byte-identical to commit 087d30d. The C4 fix changes src/fsbench/, so no pilot file is reused; every development unit runs fresh.
- Run paths now include the plan name: results/runs/<group>/<plan>/<dataset>/<config>/r<repeat>_f<fold>_s<seed index>.json.gz (src/fsbench/run.py), so that runs of different plans cannot collide or be resumed from each other. The 480 pilot files were moved (git mv, bytes unchanged) to results/runs/development/pilot/, and the path column of results/MANIFEST.csv and results/progress.csv was rewritten accordingly (no other field changed); validate_release.py still passes for the pilot. validate_release.py now checks each plan separately (expected units present; every file under results/runs/<group>/<plan>/; payload plan, group, dataset, configuration, repeat, fold and seed index match the unit; files re-hashed and commits listed per plan) and fails if two plans expect the same path; tested in tests/test_runner.py (test_two_plans_with_identical_units_never_collide).
- Stage-2 selection rules (src/stage2_select.py, written and committed before any development run; not run). Score: outer AUC at the 2,000-evaluation checkpoint (C4: the knee record at 2,000), averaged per development dataset over outer folds, search seeds and the searchers concerned, then averaged over datasets (the section 8 convention). (1) P, lambda, r: the grid value with the highest score over the four controlled searchers; ties to the smaller value. (2) Primary control for H2: the highest score over the four searchers among C1, C2, C3 at their chosen values and C4; ties to the cheaper control, C4 < C1 < C3 < C2 (pilot wall time: C4 free, C1 about 17 s, C3 about 47 s, C2 about 135 s per run). (3) Primary controlled wrapper for H3: the highest-scoring of the 16 pairs (four searchers x C1, C2, C3, C4); ties to the cheaper control, then the cheaper searcher by pilot C0 mean wall time, ga (44.2 s) < de_S (47.9 s) < gwo_S (48.4 s) < bpso_S (48.6 s). A tie is a difference of at most 1e-12. The script refuses to run on missing, non-ok, tampered or mixed-commit runs and writes its choices with every candidate's scores to results/stage2/stage2_choices.json. Tested on fabricated run files (tests/test_stage2_select.py).

## 2026-11-12: stage2_select.py run unchanged (script commit 0e21454)

- 2026-11-12: stage2_select.py run unchanged (script commit 0e21454). The field script_commit in stage2_choices.json records HEAD at run time (7424ecd), not the script commit; the file is kept unedited to preserve its hash.
- results/stage2/stage2_choices.json: SHA-256 d38695a8ecca2587d0acf7ebd0570cae20f8925115d8e95ca32c4e8c4b075f14; runs from commit 3b69ca5 (development run committed in 7424ecd). Choices: P = 5, lambda = 1.0, r = 8, primary control C4, primary wrapper ga + C4; no tie-break was needed. Stage 2 is not yet written.

## 2026-11-12: bench_scaling.py result and confirmation compute projection

- 2026-11-12: bench_scaling.py (synthetic data, K = 200, one thread) shows search time flat in n_train from 166 to 775 (slope b = 0.01 for ga C0, -0.03 for ga C2). The confirmation core tier projects at 359-389 CPU h, 60-65 h at 6 workers; planning figure 65 h (development fit). The linear-in-n bound is ruled out. S2 projects at 7-19 CPU h; the benchmark does not cover K > 200. No confirmation dataset was loaded.
- Records: bench_scaling.py at commit 468d682; its console output, as pasted by the author, in results/compute/bench_scaling_output.txt; projection script src/project_compute.py (commit c38cfb3), which writes results/compute/projection.json.

## 2026-11-12: elastic net max_iter fix (bug fix, section 10)

- Problem: the elastic-net baseline (saga, l1_ratio = 0.5, max_iter = 2000, tol = scikit-learn default 1e-4) produced 1,228 ConvergenceWarnings in the development run, in all 60 enet runs; every other configuration had none. The run files store one warning total per run, so the C values and fits (inner tuning or final) involved cannot be read from them; src/diag_enet_warnings.py refits the saga models on the development folds (warning counts only, no performance value) to attribute them.
- Fix rule, set before any rerun: max_iter = 10000 for the elastic net only, tol unchanged; adopted whatever the outcome; if warnings remain, max_iter stays 10000, the residual counts are reported, and there is no further iteration. Implemented in src/fsbench/baselines.py (ENET_MAX_ITER). No other selector, searcher or model changes.
- Rerun: configs/dev_enet_fix.yaml reruns only baseline__enet on the 12 development datasets (repeat 0, folds 0-4) under its own plan path, results/runs/development/dev_enet_fix/; the original enet runs (plan development) are kept unchanged. The stage-2 choices (results/stage2/stage2_choices.json) use only search runs and are not affected.

## 2026-11-13: elastic net max_iter fix adopted (development rerun)

- 2026-11-13: elastic net baseline max_iter raised from 2000 to 10000 (commit a1535d6), fixed before rerun and adopted regardless of outcome. At 2000, 1,221 inner and 438 final fits warned, all at C >= 0.43; at 10000, none. Development enet rerun as plan dev_enet_fix; original runs kept. Fast test suite at HEAD: 131 passed, structural-bias results identical to VERIFICATION.md. No AUC was read for this decision.
- Records: rerun committed in c8574f1 (60 runs, 0 errors, 0 warnings; validate_release.py passes for pilot, development and dev_enet_fix). Attribution by src/diag_enet_warnings.py, output results/diagnostics/enet_warnings.json (SHA-256 45fad676441f25cff41ebc69bc2ece600021cb3efe56cce1e26249d73db07eb9). The 438 final-fit warnings count the final fit at every C of the grid; the run files count only the final fit at the chosen C, so the original 1,228 warnings per the run files reconcile exactly as 1,221 inner + 7 final at the chosen C, with none from the L2 scoring models. The fast test suite was run with FSBENCH_RESULTS_DIR redirected, so the committed results/verification/ files are unchanged.

## 2026-11-13: H4 mixed-model acceptance rule (analysis code, before any confirmatory run)

- A mixed-model fit is accepted when statsmodels reports converged = True. Optimizer order: lbfgs first; powell only if lbfgs does not converge (or raises). Every warning raised during a fit is recorded verbatim (category and message) in results.json and does not reject the fit. If neither optimizer converges for the full or the reduced model, H4 is reported as "not estimable" and enters the Holm family with p = 1. Implemented in src/analysis/stats.py (ACCEPTANCE_RULE) and tested in tests/test_analysis_stats.py.
- Fixed before any confirmatory run, after the exploratory development dry run, in which every fit converged with lbfgs and logged one ConvergenceWarning whose text had not been recorded.

## 2026-11-13: stage 2 pre-registration frozen

- 2026-11-13: stage 2 pre-registration frozen. Confirmation set unsealed by this commit. No confirmation dataset was loaded before it.
- docs/PREREG_STAGE2.md, SHA-256 32fc0d8b2cd5260ff53b4343707e61504ab3a6994a76c5436d3ba709bf10f85c (docs/PREREG_STAGE2.sha256), committed together with its hash file; the analysis code it cites is src/analysis/ at 0b3da97 and configs/analysis_confirmation.yaml at 44d6c32.

## Pre-registration status

Pre-registration status: stage 1 frozen (8fa82c7, amendment 1 in 485ee76); stage 2 frozen (this commit: "pre-registration stage 2 frozen (unseals confirmation)"); confirmation set unsealed; no confirmation run yet.
