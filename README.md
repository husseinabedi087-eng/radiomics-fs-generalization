# More search, same model: a pre-registered study of optimism in evolutionary and swarm feature selection for radiomics

Code, pre-registration, run files and results of a pre-registered benchmark of wrapper feature
selection for radiomics. Author: Hussein Fadhil Abedi.

## Summary

Wrapper feature selectors tune a feature subset to an inner cross-validation score, and the subset can
overfit that score. This study measures that optimism and tests four generalization controls on 43
public radiomics datasets with binary outcomes from radMLBench 1.2, split at the dataset level into 12
development and 31 confirmation datasets before any model was trained. Twelve searchers (random search,
hill climbing, (1+1)-EA, a genetic algorithm, and binary PSO, DE, GWO and WOA with S- and V-shaped
transfer functions) ran with a budget of 2,000 evaluations under 5 x 2 outer cross-validation with three
search seeds per fold, together with six non-search selectors. All choices left open by stage 1 were
fixed on development data and frozen in stage 2 before the confirmation set was unsealed.

Confirmatory results on the 31 confirmation datasets (Holm correction across five hypotheses;
`results/analysis/confirmation/results.json`):

- **H1, optimism:** the inner AUC of the returned subset exceeded its outer AUC. Mean optimism gap over the
  12 uncontrolled searchers 0.134 AUC (95% bootstrap CI 0.103 to 0.167), Holm-adjusted p = 2.3e-09.
- **H2, primary control:** knee selection on the archive's fitness-size front (C4, chosen in stage 2) did
  not raise outer AUC. Mean difference -0.0004 (CI -0.0021 to 0.0013), adjusted p = 0.69, "no detectable
  difference".
- **H3, against L1 logistic regression:** GA with knee selection was equivalent to L1-regularized logistic
  regression within a margin of 0.02 AUC. Mean difference -0.0067 (CI -0.0170 to 0.0043), adjusted
  p_TOST = 0.043. Without the univariate pre-filter (sensitivity analysis S2, 17 datasets), equivalence
  was not shown (p_TOST = 0.13).
- **H4, searcher x control interaction:** no detectable interaction (likelihood-ratio test, 12 df,
  adjusted p = 0.69).
- **H5, metaphor-based searchers:** GWO and WOA did not outperform GA, binary PSO and DE; the difference
  was equivalent to zero within 0.01 AUC (mean 0.0008, CI -0.0028 to 0.0045, adjusted p = 7.1e-05).

## Pre-registration timeline

| Step | Commit | Document or output |
| --- | --- | --- |
| Stage 1 frozen | `8fa82c7` | `docs/PREREG_STAGE1.md` (SHA-256 in `docs/PREREG_STAGE1.sha256`) |
| Amendment 1 (H5 one-sided) | `485ee76` | `docs/PREREG_STAGE1_AMENDMENT1.md` |
| Development run and stage-2 selection | `7424ecd`, `c8574f1`, `8980619` | `results/runs/development/`, `results/stage2/stage2_choices.json` |
| Stage 2 frozen, confirmation set unsealed | `862a7ea` | `docs/PREREG_STAGE2.md` (SHA-256 in `docs/PREREG_STAGE2.sha256`) |
| Confirmation run (24,800 runs) and S2 (680 runs) | `33ceef6` | `results/runs/confirmation/`, `results/MANIFEST.csv` |
| Confirmatory analysis (frozen analysis code `0b3da97`) | `00f67d4` | `results/analysis/confirmation/` |
| Paper figures, tables and supplementary material | `5a6cd75`, `60fa5f1` | `results/paper/` |

Every decision and deviation is logged with its date and commit in `docs/DECISIONS.md`. Every run file
records the commit it was produced at; `src/validate_release.py` checks them.

## Repository layout

```
configs/        run plans (pilot, development, dev_enet_fix, confirmation, confirmation_s2) and analysis configs
data/meta/      dataset index with SHA-256 hashes, audit, licence decisions and evidence, data-handling rules
data/split/     sealed dataset-level split (development / confirmation) and its hash
docs/           pre-registration (stage 1, amendment 1, stage 2), decision log, verification report
environment/    Python version, pinned packages (pip_freeze.txt), platform and BLAS configuration
results/        run files (runs/), MANIFEST.csv, partitions, stage-2 choices, verification, health reports,
                analysis outputs (analysis/), paper figures, tables and supplementary material (paper/)
src/fsbench/    searchers, evaluator, controls, pipeline, baselines and the runner
src/analysis/   frozen analysis code (H1-H5, Holm, sensitivity and secondary analyses)
src/paper/      figure, table and supplementary-material scripts
src/            download, audit, split, validation, health and compute-estimation scripts
tests/          unit, verification and end-to-end tests
```

## How to reproduce

The study ran on Windows 11 with Python 3.12.10, single-threaded BLAS and 6 worker processes. The
confirmation run takes about three days on 6 workers.

### 1. Environment

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -r environment/pip_freeze.txt
$env:PYTHONPATH = "src"
```

bash:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -r environment/pip_freeze.txt
export PYTHONPATH=src
```

`--no-deps` is required: `pip_freeze.txt` lists the complete environment, and mealpy 3.0.3 (used only by
the verification tests) declares numpy <= 1.26 although the study uses numpy 2.5.4
(`environment/pip_check.txt`).

### 2. Data

```
python src/download_data.py
git status data/meta
```

The script saves every radMLBench 1.2 dataset unchanged to `data/raw/` (not tracked) and rewrites the
metadata in `data/meta/`. `git status` should show no change: `data/meta/radmlbench_index.csv` holds the
SHA-256 of every parquet file, and `src/data_access.py` refuses to load a file whose hash differs.
Confirmation datasets load only because `docs/PREREG_STAGE2.sha256` is committed.

### 3. Tests

```
python -m pytest -m "not slow" -q
python -m pytest -m slow -q
```

Set `FSBENCH_RESULTS_DIR` to a temporary folder first if the committed files in `results/verification/`
must not be overwritten.

### 4. Runs

Runs already listed in `results/MANIFEST.csv` with a matching file hash are skipped, so the commands below
only check the committed runs. To recompute from scratch, use a separate clone and remove
`results/MANIFEST.csv`, `results/progress.csv` and `results/runs/` first.

```
python -m fsbench.run --plan configs/development.yaml --n-jobs 6
python -m fsbench.run --plan configs/dev_enet_fix.yaml --n-jobs 6
python src/stage2_select.py
python -m fsbench.run --plan configs/confirmation.yaml --n-jobs 6
python -m fsbench.run --plan configs/confirmation_s2.yaml --n-jobs 6
python src/validate_release.py --plan configs/pilot.yaml --plan configs/development.yaml --plan configs/dev_enet_fix.yaml --plan configs/confirmation.yaml --plan configs/confirmation_s2.yaml
```

### 5. Analysis, figures and supplementary material

```
python -m analysis.run --config configs/analysis_confirmation.yaml
python src/paper/make_figures.py
python src/save_health.py --name development --plan configs/development.yaml --plan configs/dev_enet_fix.yaml
python src/save_health.py --name confirmation --plan configs/confirmation.yaml --plan configs/confirmation_s2.yaml
python src/paper/make_supplementary.py
```

The outputs are deterministic: rerunning a step reproduces the files listed in its manifest
(`results/analysis/confirmation/analysis_manifest.csv`, `results/paper/paper_manifest.csv`,
`results/paper/supplementary/supplementary_manifest.csv`) byte for byte on the same environment.

The commands are the same in PowerShell and bash; on Windows, backslashes in paths also work.

## Post hoc analyses

`results/paper/posthoc/` holds post hoc analyses on the confirmation data that were not pre-registered (commits `d5d4a68` and `f28c3ef`). Every file in that folder is listed with its SHA-256 in `results/paper/posthoc/posthoc_manifest.csv`. To reproduce them, run `src/paper/posthoc.py` before `src/paper/posthoc_runlevel.py`, because `posthoc.py` clears the folder:

```
python src/paper/posthoc.py
python src/paper/posthoc_runlevel.py
```

## Data

The raw radMLBench data are not redistributed in this repository (`data/raw/` is not tracked); they are
downloaded from radMLBench by `src/download_data.py`. Each dataset keeps its own licence and terms of use,
which apply to any use of the data; see `data/meta/license_decisions.csv` for the licence category and
evidence of every dataset and the reasons for the seven excluded datasets. The repository contains no
patient or sample identifiers.

## Licences

- Code: MIT (`LICENSE`).
- Run files, analysis outputs, figures and tables: CC BY 4.0 (`LICENSE-results.md`). The underlying
  datasets keep their own licences.

## Citation

See `CITATION.cff`. Paper: [citation to be added after publication].

## Contact

Hussein Fadhil Abedi, [contact to be added].
