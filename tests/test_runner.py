"""Runner, manifest, resume and release validation on the test-only synthetic group (no real data)."""

import gzip
import json

import pytest

from fsbench import run
import validate_release

PLAN = """name: runner_test
group: synthetic
datasets: [SYNTH_80_30_7]
repeats: [0]
folds: [0]
search_seed_indices: [0, 1]
budget: 60
c0_searchers: [ga, woa_S]
control_searchers: [ga]
controls:
  C1: 2
  C3: 2
baselines: [ftest, rf]
rf_model_seed_indices: [0, 1]
"""


@pytest.fixture
def plan_file(tmp_path):
    p = tmp_path / "plan.yaml"
    p.write_text(PLAN, encoding="utf-8", newline="\n")
    return p


def test_expand_counts_and_seeds(plan_file):
    units = run.expand(run.load_plan(plan_file))
    # per seed index: 2 C0 + 2 controls; baselines: ftest once, rf twice
    assert len(units) == 2 * 4 + 1 + 2
    s0 = {u["search_seed"] for u in units if u["kind"] == "search" and u["seed_index"] == 0}
    assert len(s0) == 1  # common random numbers: one search seed shared by every configuration
    assert {u["path"] for u in units} == {u["path"] for u in run.expand(run.load_plan(plan_file))}
    rf = [u["model_seed"] for u in units if u.get("selector") == "rf"]
    assert len(set(rf)) == 2


def test_synthetic_group_only_accepts_synthetic_datasets(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(PLAN.replace("SYNTH_80_30_7", "HNSCC"), encoding="utf-8")
    with pytest.raises(ValueError):
        run.load_plan(p)


def test_run_resume_and_validate(plan_file, tmp_path):
    results = tmp_path / "results"
    assert run.main(["--plan", str(plan_file), "--n-jobs", "1", "--results-dir", str(results)]) == 0
    rows = run.read_manifest(results / "MANIFEST.csv")
    assert len(rows) == 11 and all(r["status"] == "ok" for r in rows)
    payload = json.loads(gzip.decompress((results / rows[0]["path"]).read_bytes()))
    assert payload["git_commit"] == rows[0]["commit"] and set(payload["timings"]) >= {
        "fold_preparation_s", "standard_inner_auc_s", "outer_scoring_s"}
    # resume: nothing to do
    assert run.main(["--plan", str(plan_file), "--n-jobs", "1", "--results-dir", str(results)]) == 0
    assert len(run.read_manifest(results / "MANIFEST.csv")) == 11
    assert validate_release.main(["--plan", str(plan_file), "--results-dir", str(results)]) == 0
    # corrupt one file: validation fails, resume reruns exactly that unit, validation passes again
    f = results / rows[0]["path"]
    data = bytearray(f.read_bytes())
    data[30] ^= 0xFF
    f.write_bytes(bytes(data))
    assert validate_release.main(["--plan", str(plan_file), "--results-dir", str(results)]) == 1
    assert run.main(["--plan", str(plan_file), "--n-jobs", "1", "--results-dir", str(results)]) == 0
    assert len(run.read_manifest(results / "MANIFEST.csv")) == 12
    assert validate_release.main(["--plan", str(plan_file), "--results-dir", str(results)]) == 0
