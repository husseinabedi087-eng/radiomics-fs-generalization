# Inclusion criteria

Written on 2026-11-09, before any data was downloaded.

A radMLBench dataset is included if it meets all of the following criteria:

1. Binary outcome.
2. At least 20 samples in the minority class.
3. The source dataset's license or terms permit reuse in research and publication of derived results. If the license cannot be determined from radMLBench metadata or the source publication, mark it "unknown" and flag it for review; do not exclude it automatically.
4. Patient structure: if a dataset has more than one row per patient (several lesions, scans or timepoints) and provides a patient ID, it is included and will be split by patient later. If it has more than one row per patient and no patient ID, it is excluded. If the structure cannot be determined, flag it for review.
5. At least 1 feature after removing ID and target columns.
