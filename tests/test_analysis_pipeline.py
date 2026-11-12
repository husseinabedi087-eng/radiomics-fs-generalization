"""End-to-end analysis on synthetic run files written by a fixture (synthetic group; no real data).

The fixture writes plans, run files with the runner's payload layout and a MANIFEST.csv into a
temporary results directory. Two studies:
  - a small one (some configurations missing): validation, byte-identical outputs on a second run,
    refusal when a planned run file is missing or validation fails;
  - the full confirmation design (all 12 C0 searchers with C4 records, C1/C2/C3 on the four controlled
    searchers, all six baselines with rf at 3 model seeds, an S2 plan, 32 datasets, a sensitivity subset):
    no analysis is skipped and no configuration is missing.
"""

import csv
import gzip
import hashlib
import json
import subprocess

import numpy as np
import pytest

from analysis import analyses as an
from analysis import run as arun
from fsbench import run

HEAD = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
DATASETS = [f"SYNTH_100_60_{i}" for i in range(12)]
PLAN = f"""name: analysis_test
group: synthetic
datasets: [{", ".join(DATASETS)}]
repeats: [0]
folds: [0, 1, 2]
search_seed_indices: [0]
budget: 2000
c0_searchers: [random, ga, bpso_S, de_S, gwo_S, gwo_V, woa_S]
control_searchers: [ga, bpso_S, de_S, gwo_S]
controls:
  C1: 5
  C2: 1.0
  C3: 8
baselines: [l1, rf]
rf_model_seed_indices: [0, 1]
"""

FULL_DATASETS = [f"SYNTH_100_{60 + 10 * (i % 5)}_{i}" for i in range(32)]
FULL_PLAN = f"""name: full_design
group: synthetic
datasets: [{", ".join(FULL_DATASETS)}]
repeats: [0]
folds: [0, 1]
search_seed_indices: [0]
budget: 2000
c0_searchers: [random, hill_climb, one_plus_one, ga, bpso_S, bpso_V, de_S, de_V, gwo_S, gwo_V, woa_S, woa_V]
control_searchers: [ga, bpso_S, de_S, gwo_S]
controls:
  C1: 5
  C2: 1.0
  C3: 8
baselines: [all_k, l1, enet, ftest, mrmr, rf]
rf_model_seed_indices: [0, 1, 2]
prefilter: true
"""
FULL_S2_PLAN = f"""name: full_design_s2
group: synthetic
datasets: [{", ".join(FULL_DATASETS[:17])}]
repeats: [0]
folds: [0, 1]
search_seed_indices: [0]
budget: 2000
c0_searchers: [ga]
control_searchers: []
controls: {{}}
baselines: [l1]
prefilter: false
"""


def _rec(rng, auc, p, size=6):
    names = [f"f{j:04d}" for j in sorted(rng.choice(p, size=size, replace=False))]
    return {"outer_auc": auc, "standard_inner_auc": auc + 0.04 + rng.normal(0, 0.005),
            "outer_brier": 0.25 - 0.2 * (auc - 0.5), "size": size, "subset": names,
            "pool_indices": list(range(size)), "best_inner_fitness": auc + 0.04}


def _payload(u, datasets, prefilter):
    rng = np.random.default_rng(int(hashlib.sha256(u["path"].encode()).hexdigest()[:8], 16))
    base = 0.65 + 0.005 * datasets.index(u["dataset"])
    p = int(u["dataset"].split("_")[2])
    r = {"plan": u["plan"], "group": u["group"], "configuration": u["configuration"],
         "seed_index": u["seed_index"], "git_commit": HEAD, "status": "ok", "dataset": u["dataset"],
         "repeat": u["repeat"], "fold": u["fold"], "K": p, "n_features_raw": p, "n_train": 80, "n_test": 20,
         "search_seed": u.get("search_seed"), "model_seed": u.get("model_seed"), "timings": {}}
    if not prefilter:
        r["prefilter"] = False
    if u["kind"] == "search":
        cps = {str(c): _rec(rng, base + 0.002 * np.log(c) + rng.normal(0, 0.01), p) for c in (250, 500, 1000, 2000)}
        r.update({"kind": "search", "searcher": u["searcher"], "control": u["control"],
                  "control_param": None if u["control"] == "C0" else {run.CONTROL_KEYS[u["control"]]: u["param"]},
                  "budget": 2000, "n_evals": 2000, "checkpoints": cps})
        if u["control"] == "C0":
            r["c4"] = {k: {**_rec(rng, v["outer_auc"] + 0.02, p, size=3), "archive_index": 1} for k, v in cps.items()}
    else:
        r.update({"kind": "baseline", "selector": u["selector"], "flags": [],
                  "returned": _rec(rng, base + rng.normal(0, 0.01), p)})
    return r


def write_study(root, plan_text=PLAN, datasets=DATASETS, s2_text=None, s1=None, label="synthetic_test"):
    """Write plan(s), run files, MANIFEST.csv, the sensitivity list and an analysis config; return
    (config path, results dir)."""
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    rows = []
    plan_files = []
    for name, text in [("plan.yaml", plan_text)] + ([("plan_s2.yaml", s2_text)] if s2_text else []):
        pf = root / name
        pf.write_text(text, encoding="utf-8", newline="\n")
        plan_files.append(pf)
        plan = run.load_plan(pf)
        for u in run.expand(plan):
            data = gzip.compress(json.dumps(_payload(u, datasets, plan.get("prefilter", True))).encode(), mtime=0)
            f = results / u["path"]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
            rows.append({"path": u["path"], "sha256": hashlib.sha256(data).hexdigest(), "dataset": u["dataset"],
                         "group": u["group"], "configuration": u["configuration"], "repeat": u["repeat"],
                         "fold": u["fold"], "seed_index": u["seed_index"], "search_seed": u.get("search_seed"),
                         "model_seed": u.get("model_seed"), "commit": HEAD, "status": "ok", "utc_finished": ""})
    with open(results / "MANIFEST.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=run.MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    s1_file = root / "s1.txt"
    s1_file.write_text("# test sensitivity set\n" + "\n".join(s1 or datasets[:2]) + "\n", encoding="utf-8",
                       newline="\n")
    s2_line = f"s2_plan: {plan_files[1].as_posix()}" if s2_text else "s2_plan: null"
    cfg = root / "analysis.yaml"
    cfg.write_text(f"label: {label}\nplans:\n  - file: {plan_files[0].as_posix()}\n{s2_line}\n"
                   f"control_values: {{C1: 5, C2: 1.0, C3: 8}}\nsensitivity_list: {s1_file.as_posix()}\n",
                   encoding="utf-8", newline="\n")
    return cfg, results


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return write_study(tmp_path_factory.mktemp("study"))


def _snapshot(out):
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_full_pipeline_is_byte_identical_and_complete(study, tmp_path):
    cfg, results = study
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 0
    out = results / "analysis" / "synthetic_test"
    first = _snapshot(out)
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 0
    assert _snapshot(out) == first
    # a separate output folder gives the same bytes, manifest included
    other = tmp_path / "elsewhere"
    assert arun.main(["--config", str(cfg), "--results-dir", str(results), "--output-dir", str(other)]) == 0
    assert _snapshot(other) == first

    res = json.loads(first["results.json"])
    assert res["header"] == "SYNTHETIC TEST DATA" and res["exploratory"] is False
    conf = res["confirmatory"]
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert not conf[h].get("skipped") and conf[h]["p_holm"] is not None and conf[h]["wording"]
    assert conf["H2"]["p_holm"] < 0.05 and conf["H2"]["effect"]["mean"] == pytest.approx(0.02, abs=0.005)
    assert conf["H1"]["configurations_missing"] == ["hill_climb__C0", "one_plus_one__C0", "bpso_V__C0", "de_V__C0",
                                                     "woa_V__C0"]
    assert conf["H5"]["metaphor_used"] == ["gwo_S__C0", "gwo_V__C0", "woa_S__C0"]
    assert "woa_V__C0" in res["configurations"]["missing"] and "hill_climb__C0" in res["configurations"]["missing"]
    assert res["sensitivity"]["S1"]["excluded_datasets"] == DATASETS[:2]
    assert len(res["sensitivity"]["S1"]["datasets_remaining"]) == 10
    assert res["sensitivity"]["S2"]["skipped"]
    assert res["secondary"]["random_vs_guided"]["comparisons"]
    assert res["secondary"]["friedman"]["p"] is not None
    assert set(res["versions"]) >= {"numpy", "scipy", "statsmodels", "matplotlib", "pandas"}
    for fit in conf["H4"]["outer_auc"]["fit_full"]:
        assert isinstance(fit["warnings"], list)

    for name, data in first.items():
        if name.endswith(".csv"):
            assert data.startswith(b"# SYNTHETIC TEST DATA\n"), name
    with open(out / "analysis_manifest.csv", encoding="utf-8") as fh:
        next(fh)
        listed = {r["path"]: r["sha256"] for r in csv.DictReader(fh)}
    assert set(listed) == set(first) - {"analysis_manifest.csv"}
    assert all(hashlib.sha256(first[p]).hexdigest() == h for p, h in listed.items())
    figs = {n for n in first if n.startswith("figures/")}
    assert {"figures/anytime.pdf", "figures/anytime.png", "figures/h2_differences.pdf",
            "figures/h3_differences.png", "figures/optimism_gap_vs_log_pn.pdf"} <= figs


def test_full_confirmation_design_nothing_skipped_or_missing(tmp_path):
    s1 = FULL_DATASETS[:8]
    cfg, results = write_study(tmp_path, FULL_PLAN, FULL_DATASETS, FULL_S2_PLAN, s1=s1, label="full_design")
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 0
    res = json.loads((results / "analysis" / "full_design" / "results.json").read_bytes())
    conf = res["confirmatory"]
    assert res["configurations"]["missing"] == [] and res["configurations"]["excluded_control_values"] == []
    assert sorted(res["configurations"]["present"]) == sorted(an.EXPECTED)
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert not conf[h].get("skipped"), h
        assert conf[h]["p_raw"] is not None and conf[h]["p_holm"] is not None
    assert conf["H1"]["configurations_used"] == [f"{s}__C0" for s in an.C0_SEARCHERS]
    assert conf["H1"]["configurations_missing"] == [] and conf["H1"]["effect"]["n_datasets"] == 32
    assert conf["H2"]["searchers_used"] == an.CTRL_SEARCHERS
    assert len(conf["H5"]["metaphor_used"]) == 4 and len(conf["H5"]["classical_used"]) == 5
    assert conf["H5"]["configurations_missing"] == []
    assert conf["H4"]["outer_auc"]["estimable"] and conf["H4"]["outer_auc"]["df"] == 12
    assert conf["H4"]["controls"] == ["C0", "C1", "C2", "C3", "C4"]
    rvg = res["secondary"]["random_vs_guided"]
    assert not rvg.get("skipped") and rvg["configurations_missing"] == []
    assert len(rvg["comparisons"]) == 11 and all(c["p_holm"] is not None for c in rvg["comparisons"])
    s1r = res["sensitivity"]["S1"]
    assert not s1r["results"].get("skipped") and len(s1r["datasets_remaining"]) == 24
    assert all(not s1r["results"][h].get("skipped") for h in ("H1", "H2", "H3", "H4", "H5"))
    s2 = res["sensitivity"]["S2"]
    assert not s2.get("skipped") and s2["effect"]["n_datasets"] == 17 and s2["tost"]["p_tost"] is not None
    stab = {r["config"]: r for r in res["secondary"]["size_stability"]["summary"]}
    assert set(stab) == set(an.EXPECTED)
    assert all(r["n_datasets"] == 32 and r["mean_stability"] is not None for r in stab.values())
    fr = res["secondary"]["friedman"]
    assert not fr.get("skipped") and len(fr["selectors"]) == len(an.EXPECTED) == 42
    assert fr["n_pairs"] == 42 * 41 // 2
    brier = res["secondary"]["brier"]
    assert all(k in brier for k in ("H2", "H3", "H4", "H5"))
    figs = {p.name for p in (results / "analysis" / "full_design" / "figures").iterdir()}
    assert len(figs) == 8


def test_refuses_when_a_planned_run_file_is_missing(tmp_path):
    cfg, results = write_study(tmp_path)
    victim = next((results / "runs").rglob("*.json.gz"))
    victim.unlink()
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 2
    assert not (results / "analysis" / "synthetic_test").exists()


def test_refuses_when_validation_fails(tmp_path):
    cfg, results = write_study(tmp_path)
    (tmp_path / "plan.yaml").write_text(PLAN.replace("C1: 5", "C1: [3, 5]"), encoding="utf-8", newline="\n")
    # the plan now expects C1_P3 runs that do not exist: validation refuses before any analysis
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 2


def test_refuses_to_clear_a_foreign_output_folder(study, tmp_path):
    cfg, results = study
    foreign = tmp_path / "not_an_analysis"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x", encoding="utf-8")
    assert arun.main(["--config", str(cfg), "--results-dir", str(results), "--output-dir", str(foreign)]) == 2
    assert (foreign / "keep.txt").exists()


def test_outputs_identical_across_processes_with_different_hash_seeds(study, tmp_path):
    """Separate processes (different PYTHONHASHSEED, different output folders) give byte-identical outputs."""
    import os
    import sys
    from pathlib import Path
    cfg, results = study
    repo = Path(__file__).resolve().parent.parent
    snaps = []
    for seed in ("1", "2"):
        out = tmp_path / f"out_{seed}"
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(repo / "src")}
        proc = subprocess.run([sys.executable, "-m", "analysis.run", "--config", str(cfg), "--results-dir",
                               str(results), "--output-dir", str(out)], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        snaps.append(_snapshot(out))
    assert snaps[0] == snaps[1]


def test_stability_independent_of_hash_seed():
    import os
    import sys
    from pathlib import Path
    code = ("from analysis.stats import nogueira_stability as n; import numpy as np; "
            "r = np.random.default_rng(0); "
            "s = [[f'feat_{j}' for j in r.choice(300, 40, replace=False)] for _ in range(10)]; print(repr(n(s, 300)))")
    repo = Path(__file__).resolve().parent.parent
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(repo / "src")}).stdout
            for seed in ("1", "2", "3", "4")}
    assert len(outs) == 1
