"""The default path (prefilter: true) reproduces a committed development run payload exactly.

Opt-in, because it loads one development dataset through src/data_access.py: run it with
    $env:FSBENCH_DEVDATA_TESTS = "1"; python -m pytest tests/test_default_payload_devdata.py -q
Cell: Brancato2023 (the smallest development dataset by n_train), repeat 0, fold 0, ga__C0, seed index 0,
budget 2000, as in configs/development.yaml. Reference: the committed run file, produced at commit
3b69ca5. Between 3b69ca5 and a1fa306 src/fsbench/ changed only in baselines.py (elastic net max_iter),
which a ga__C0 run does not use, so the reference is also the payload at a1fa306 (checked below).
Compared: every field except the run-environment ones (commit, versions, timestamps, pid, thread
info, wall time, timings). Nothing is printed; the test only asserts equality.
"""

import gzip
import json
import os
import subprocess
from pathlib import Path

import pytest

from fsbench import run

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "results/runs/development/development/Brancato2023/ga__C0/r0_f0_s0.json.gz"
VOLATILE = {"git_commit", "versions", "utc_start", "utc_finished", "pid", "blas_threads", "wall_s", "timings"}

pytestmark = pytest.mark.skipif(os.environ.get("FSBENCH_DEVDATA_TESTS") != "1",
                                reason="loads development data; set FSBENCH_DEVDATA_TESTS=1")


def test_reference_commit_equivalence():
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", "3b69ca5", "a1fa306", "--", "src/fsbench"],
                          capture_output=True, text=True, check=True).stdout.split()
    assert diff == ["src/fsbench/baselines.py"]


def test_default_prefilter_payload_identical(tmp_path):
    unit = next(u for u in run.expand(run.load_plan(REPO / "configs/development.yaml"))
                if (u["dataset"], u["repeat"], u["fold"], u["configuration"], u["seed_index"])
                == ("Brancato2023", 0, 0, "ga__C0", 0))
    assert unit["prefilter"] is True
    run._init_worker()
    row = run.execute(unit, str(tmp_path), "test", str(REPO / "results/partitions"))
    assert row["status"] == "ok"
    new = json.loads(gzip.decompress((tmp_path / unit["path"]).read_bytes()))
    ref = json.loads(gzip.decompress(REF.read_bytes()))
    strip = lambda d: {k: v for k, v in d.items() if k not in VOLATILE}
    assert "prefilter" not in new
    assert strip(new) == strip(ref)
