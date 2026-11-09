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

## Pre-registration status

Pre-registration status: stage 1 not frozen, stage 2 not frozen; data downloaded (2026-11-09); dataset-level split sealed (2026-11-10); no model trained.
