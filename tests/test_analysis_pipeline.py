"""End-to-end analysis on synthetic run files written by a fixture (synthetic group; no real data).

The fixture writes a plan, run files with the runner's payload layout and a MANIFEST.csv into a
temporary results directory; the analysis must validate them, analyse them, write byte-identical
outputs on a second run, and refuse when a planned run file is missing.
"""

import csv
import gzip
import hashlib
import json
import subprocess

import numpy as np
import pytest

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


def _rec(rng, auc, p, size=6):
    names = [f"f{j:04d}" for j in sorted(rng.choice(p, size=size, replace=False))]
    return {"outer_auc": auc, "standard_inner_auc": auc + 0.04 + rng.normal(0, 0.005),
            "outer_brier": 0.25 - 0.2 * (auc - 0.5), "size": size, "subset": names,
            "pool_indices": list(range(size)), "best_inner_fitness": auc + 0.04}


def write_study(root):
    plan_file = root / "plan.yaml"
    plan_file.write_text(PLAN, encoding="utf-8", newline="\n")
    results = root / "results"
    rows = []
    for u in run.expand(run.load_plan(plan_file)):
        rng = np.random.default_rng(int(hashlib.sha256(u["path"].encode()).hexdigest()[:8], 16))
        base = 0.65 + 0.01 * DATASETS.index(u["dataset"])
        _, n, p, _ = u["dataset"].split("_")
        p = int(p)
        r = {"plan": u["plan"], "group": u["group"], "configuration": u["configuration"],
             "seed_index": u["seed_index"], "git_commit": HEAD, "status": "ok", "dataset": u["dataset"],
             "repeat": u["repeat"], "fold": u["fold"], "K": p, "n_features_raw": p, "n_train": 80, "n_test": 20,
             "search_seed": u.get("search_seed"), "model_seed": u.get("model_seed"), "timings": {}}
        if u["kind"] == "search":
            cps = {str(c): _rec(rng, base + 0.002 * np.log(c) + rng.normal(0, 0.01), p) for c in (250, 500, 1000, 2000)}
            r.update({"kind": "search", "searcher": u["searcher"], "control": u["control"],
                      "control_param": None if u["control"] == "C0" else {run.CONTROL_KEYS[u["control"]]: u["param"]},
                      "budget": 2000, "n_evals": 2000, "checkpoints": cps})
            if u["control"] == "C0":
                r["c4"] = {k: {**_rec(rng, v["outer_auc"] + 0.02, p, size=3), "archive_index": 1}
                           for k, v in cps.items()}
        else:
            r.update({"kind": "baseline", "selector": u["selector"], "flags": [],
                      "returned": _rec(rng, base + rng.normal(0, 0.01), p)})
        data = gzip.compress(json.dumps(r).encode(), mtime=0)
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
    s1 = root / "s1.txt"
    s1.write_text("# test sensitivity set\n" + "\n".join(DATASETS[:2]) + "\n", encoding="utf-8", newline="\n")
    cfg = root / "analysis.yaml"
    cfg.write_text(f"label: synthetic_test\nplans:\n  - file: {plan_file.as_posix()}\ns2_plan: null\n"
                   f"control_values: {{C1: 5, C2: 1.0, C3: 8}}\nsensitivity_list: {s1.as_posix()}\n",
                   encoding="utf-8", newline="\n")
    return cfg, results


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return write_study(tmp_path_factory.mktemp("study"))


def _snapshot(out):
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_full_pipeline_is_byte_identical_and_complete(study):
    cfg, results = study
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 0
    out = results / "analysis" / "synthetic_test"
    first = _snapshot(out)
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 0
    assert _snapshot(out) == first

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

    # every output carries the header; the manifest hashes every output
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


def test_refuses_when_a_planned_run_file_is_missing(tmp_path):
    cfg, results = write_study(tmp_path)
    victim = next((results / "runs").rglob("*.json.gz"))
    victim.unlink()
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 2
    assert not (results / "analysis" / "synthetic_test").exists()


def test_refuses_when_validation_fails(tmp_path):
    cfg, results = write_study(tmp_path)
    plan = (tmp_path / "plan.yaml")
    plan.write_text(PLAN.replace("C1: 5", "C1: [3, 5]"), encoding="utf-8", newline="\n")
    # the plan now expects C1_P3 runs that do not exist: validation refuses before any analysis
    assert arun.main(["--config", str(cfg), "--results-dir", str(results)]) == 2
