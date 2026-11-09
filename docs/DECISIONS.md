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

## Pre-registration status

Pre-registration status: stage 1 not frozen, stage 2 not frozen; data downloaded (2026-11-09); no model trained.
