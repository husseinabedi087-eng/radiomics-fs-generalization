"""stage2_select.py on fabricated run files with planted outer AUCs (no data, no model, no real run)."""

import csv
import gzip
import hashlib
import json

import pytest

import stage2_select
from fsbench import run

PLAN = """name: development
group: development
datasets: [Ahn2021, Arita2018]
repeats: [0]
folds: [0, 1]
search_seed_indices: [0]
budget: 2000
c0_searchers: [ga, bpso_S, de_S, gwo_S]
control_searchers: [ga, bpso_S, de_S, gwo_S]
controls:
  C1: [3, 5, 10]
  C2: [0.5, 1.0, 2.0]
  C3: [2, 4, 8]
baselines: []
"""


def build(tmp_path, auc):
    """auc(searcher, control, value) -> outer AUC planted in every run of that cell."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan_file = tmp_path / "development.yaml"
    plan_file.write_text(PLAN, encoding="utf-8", newline="\n")
    results = tmp_path / "results"
    rows = []
    for u in run.expand(run.load_plan(plan_file)):
        ctrl = u["control"]
        rec = lambda a: {"outer_auc": a}
        r = {"plan": "development", "group": "development", "status": "ok", "git_commit": "abc",
             "searcher": u["searcher"], "control": ctrl, "search_seed": u["search_seed"],
             "checkpoints": {"2000": rec(auc(u["searcher"], ctrl, None if ctrl == "C0" else float(u["param"])))}}
        if ctrl == "C0":
            r["c4"] = {"2000": rec(auc(u["searcher"], "C4", None))}
        data = gzip.compress(json.dumps(r).encode())
        f = results / u["path"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        rows.append({"path": u["path"], "sha256": hashlib.sha256(data).hexdigest(), "status": "ok"})
    with open(results / "MANIFEST.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "sha256", "status"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return plan_file, results


def choices(tmp_path, auc):
    plan_file, results = build(tmp_path, auc)
    assert stage2_select.main(["--plan", str(plan_file), "--results-dir", str(results)]) == 0
    return json.loads((results / "stage2/stage2_choices.json").read_text(encoding="utf-8"))


def test_highest_score_wins(tmp_path):
    planted = {("C1", 10.0): 0.70, ("C2", 0.5): 0.72, ("C3", 4.0): 0.69}

    def auc(s, c, v):
        base = {"C0": 0.6, "C4": 0.65}.get(c, planted.get((c, v), 0.6))
        return base + (0.05 if (s, c) == ("de_S", "C1") else 0.0)
    out = choices(tmp_path, auc)
    assert out["choices"]["P"] == 10 and out["choices"]["lambda"] == 0.5 and out["choices"]["r"] == 4
    # C2 mean 0.72 beats C1 mean 0.70 + 0.05 / 4
    assert out["choices"]["primary_control"] == "C2"
    assert out["choices"]["primary_wrapper"] == {"searcher": "de_S", "control": "C1"}  # 0.75


def test_ties_go_to_smaller_value_and_cheaper_control_and_searcher(tmp_path):
    out = choices(tmp_path, lambda s, c, v: 0.6 if c == "C0" else 0.7)
    assert (out["choices"]["P"], out["choices"]["lambda"], out["choices"]["r"]) == (3, 0.5, 2)
    assert out["choices"]["primary_control"] == "C4"
    assert out["choices"]["primary_wrapper"] == {"searcher": "ga", "control": "C4"}
    assert out["rule2_primary_control"]["tie_broken"] is True


def test_cost_order_between_c1_c3_c2(tmp_path):
    out = choices(tmp_path, lambda s, c, v: {"C0": 0.6, "C4": 0.5}.get(c, 0.7))
    assert out["choices"]["primary_control"] == "C1"
    out = choices(tmp_path / "b", lambda s, c, v: {"C0": 0.6, "C4": 0.5, "C1": 0.5}.get(c, 0.7))
    assert out["choices"]["primary_control"] == "C3"
    out = choices(tmp_path / "c", lambda s, c, v: 0.71 if (c == "C2" and s in ("gwo_S", "bpso_S")) else 0.6)
    assert out["choices"]["primary_wrapper"] == {"searcher": "gwo_S", "control": "C2"}  # gwo_S cheaper than bpso_S


def test_refuses_incomplete_or_tampered_runs(tmp_path):
    plan_file, results = build(tmp_path, lambda s, c, v: 0.7)
    victim = next((results / "runs").rglob("*.json.gz"))
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF  # flip the last byte
    victim.write_bytes(bytes(data))
    assert stage2_select.main(["--plan", str(plan_file), "--results-dir", str(results)]) == 1
    victim.unlink()
    assert stage2_select.main(["--plan", str(plan_file), "--results-dir", str(results)]) == 1
    assert not (results / "stage2").exists()
