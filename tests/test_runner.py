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


def test_two_plans_with_identical_units_never_collide(plan_file, tmp_path):
    """Same units under two plan names: separate paths, both validate, files of one plan untouched by the other."""
    other = tmp_path / "other.yaml"
    other.write_text(PLAN.replace("name: runner_test", "name: runner_other"), encoding="utf-8", newline="\n")
    a, b = run.expand(run.load_plan(plan_file)), run.expand(run.load_plan(other))
    assert len(a) == len(b) and not {u["path"] for u in a} & {u["path"] for u in b}
    assert all(u["path"].startswith("runs/synthetic/runner_test/") for u in a)
    results = tmp_path / "results"
    assert run.main(["--plan", str(plan_file), "--n-jobs", "1", "--results-dir", str(results)]) == 0
    before = {p: (results / p).read_bytes() for p in (u["path"] for u in a)}
    assert run.main(["--plan", str(other), "--n-jobs", "1", "--results-dir", str(results)]) == 0
    assert all((results / p).read_bytes() == data for p, data in before.items())
    assert len(run.read_manifest(results / "MANIFEST.csv")) == 2 * len(a)
    assert validate_release.main(["--plan", str(plan_file), "--plan", str(other), "--results-dir", str(results)]) == 0
    assert validate_release.main(["--plan", str(plan_file), "--plan", str(plan_file), "--results-dir", str(results)]) == 1


def test_prefilter_false_keeps_all_features(tmp_path):
    """S2 option: K = every non-constant feature; same partitions, seeds, budget, checkpoints and C4 logging."""
    base = PLAN.replace("SYNTH_80_30_7", "SYNTH_80_260_7").replace("c0_searchers: [ga, woa_S]", "c0_searchers: [ga]") \
               .replace("control_searchers: [ga]", "control_searchers: []").replace("controls:\n  C1: 2\n  C3: 2\n", "controls: {}\n") \
               .replace("baselines: [ftest, rf]", "baselines: [l1]").replace("search_seed_indices: [0, 1]", "search_seed_indices: [0]") \
               .replace("budget: 60", "budget: 260")  # past the first C4 checkpoint (250)
    on, off = tmp_path / "on.yaml", tmp_path / "off.yaml"
    on.write_text(base, encoding="utf-8", newline="\n")
    off.write_text(base.replace("name: runner_test", "name: runner_nopf") + "prefilter: false\n", encoding="utf-8", newline="\n")
    u_on, u_off = run.expand(run.load_plan(on)), run.expand(run.load_plan(off))
    key = lambda u: (u["dataset"], u["repeat"], u["fold"], u["configuration"], u.get("search_seed"), u["budget"])
    assert sorted(map(key, u_on)) == sorted(map(key, u_off))  # same units, seeds and budget
    results = tmp_path / "results"
    for p in (on, off):
        assert run.main(["--plan", str(p), "--n-jobs", "1", "--results-dir", str(results)]) == 0
    load = lambda u: json.loads(gzip.decompress((results / u["path"]).read_bytes()))
    for a, b in zip(sorted(u_on, key=key), sorted(u_off, key=key)):
        ra, rb = load(a), load(b)
        assert "prefilter" not in ra and rb["prefilter"] is False
        assert ra["K"] == 200 and rb["K"] == 260 and ra["n_features_raw"] == rb["n_features_raw"] == 260
        assert (ra["n_train"], ra["n_test"], ra.get("search_seed"), ra.get("budget")) == \
               (rb["n_train"], rb["n_test"], rb.get("search_seed"), rb.get("budget"))
        if a["kind"] == "search":
            assert sorted(rb["checkpoints"], key=int) == ["250", "500", "1000", "2000"][:len(rb["checkpoints"])]
            assert set(rb["c4"]) == set(ra["c4"]) == {"250"} and len(rb["archive"]["fitness"]) == rb["n_evals"]
    assert validate_release.main(["--plan", str(on), "--plan", str(off), "--results-dir", str(results)]) == 0


def test_prefilter_must_be_boolean(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(PLAN + "prefilter: no_thanks\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        run.load_plan(p)
