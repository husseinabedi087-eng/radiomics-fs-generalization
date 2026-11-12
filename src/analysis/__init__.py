"""Analysis of the study runs (PREREG_STAGE1 sections 8 and 11; amendment 1).

Entry point: python -m analysis.run --config configs/analysis_<label>.yaml [--results-dir results]
Reads only run files listed in results/MANIFEST.csv for the plans named in the config, after the same
checks as src/validate_release.py; never reads a rounded table.
"""
