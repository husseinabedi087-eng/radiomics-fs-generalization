# Overfitting-Aware Evolutionary Feature Selection for Radiomics

This project studies overfitting-aware evolutionary feature selection for radiomics.
Using the radMLBench collection of 50 public radiomics datasets with binary outcomes,
it examines how evolutionary feature-selection methods can be designed and evaluated so
that the selected feature subsets generalize beyond the data used during the search,
rather than overfitting the selection procedure itself. The work targets
*Swarm and Evolutionary Computation* (Elsevier); sole author: Hussein Fadhil Abedi.

Pre-registration status: stage 1 and stage 2 frozen.

- Stage 1: docs/PREREG_STAGE1.md, frozen in commit 8fa82c7 (SHA-256 d5c7039a69b84d8e8eac8f2ec35b297de944eebe5b08a2073217cd47faf6302b);
  amendment 1 (H5 one-sided): docs/PREREG_STAGE1_AMENDMENT1.md, commit 485ee76 (SHA-256 1599d12f5a5304668e51f6dbeabd933e010e89ac88af13de12a21b9ad7d65722).
- Stage 2: docs/PREREG_STAGE2.md (SHA-256 32fc0d8b2cd5260ff53b4343707e61504ab3a6994a76c5436d3ba709bf10f85c), frozen in the commit
  "pre-registration stage 2 frozen (unseals confirmation)", which also unseals the confirmation set
  (find its hash with `git log --format=%h -1 -- docs/PREREG_STAGE2.sha256`).
