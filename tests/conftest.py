"""Shared test helpers.

- `report`: lines printed in a "fsbench report" section at the end of the pytest run (visible
  without -s).
- `save_result`: writes one raw result file (JSON, LF) to results/verification/<name>.json with the
  git commit, UTC time and package versions. docs/VERIFICATION.md is built from these raw files by
  src/make_verification_report.py, never from printed or rounded values.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "verification"
_REPORT = []


@pytest.fixture
def report():
    return _REPORT.append


def _meta():
    def git(*a):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip()
    import scipy
    import sklearn
    try:
        import mealpy
        mealpy_version = mealpy.__version__
    except ImportError:
        mealpy_version = None
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "src", "tests")),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "mealpy": mealpy_version}


@pytest.fixture
def save_result():
    def save(name, data):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"name": name, "meta": _meta(), "data": data}
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n"
        with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return save


def pytest_terminal_summary(terminalreporter):
    if _REPORT:
        terminalreporter.section("fsbench report")
        for line in _REPORT:
            terminalreporter.write_line(line)
