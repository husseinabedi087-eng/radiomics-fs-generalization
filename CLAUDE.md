# Standing rules for this repository

- Never train a model or compute a performance metric unless the current task says so explicitly.
- Load data only through src/data_access.py (once it exists). Never read a confirmation dataset directly.
- Every run logs a search seed and a model seed separately.
- Every statistic is computed from raw run files, never from rounded tables.
- Text files are written with LF line endings; hashes are computed in Python.
- Do not add co-author or attribution lines to commits.
- Claude Code writes code; the author runs every data, audit and experiment script himself in a separate PowerShell window. Give him the exact commands and wait for his output.
- Stop and ask when anything is ambiguous.
