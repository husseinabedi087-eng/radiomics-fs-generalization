"""Apply the stage-2 selection rules of PREREG_STAGE1 section 10 to the development runs.

    python src/stage2_select.py [--plan configs/development.yaml] [--out results/stage2/stage2_choices.json]

Written before any development run (docs/DECISIONS.md, 2026-11-12: development protocol and stage-2
selection rules). Reads only development-group run files of the given plan; refuses to run if any
expected unit is missing, not "ok", fails its manifest SHA-256, belongs to another plan or group, or if
the runs come from more than one commit.

Score (same convention as section 8): outer AUC at the 2,000-evaluation checkpoint
(checkpoints["2000"]["outer_auc"]; for C4 the knee record c4["2000"]["outer_auc"] of the C0 run). For a
set of (searcher, control, value) cells, the per-dataset score is the mean over every run of those cells
on that dataset (outer folds x search seeds x searchers); the overall score is the mean of the
per-dataset scores over the development datasets.

Rules:
  1. Control hyperparameters. For each of C1 (P), C2 (lambda), C3 (r): the grid value with the highest
     score over the four controlled searchers. Ties: the smaller value.
  2. Primary control for H2: among C1, C2, C3 at their chosen values and C4 (knee of the C0 runs of the
     four controlled searchers), the control with the highest score over the four searchers. Ties: the
     cheaper control, C4 < C1 < C3 < C2 (pilot wall time).
  3. Primary controlled wrapper for H3: among the 16 pairs (four searchers x C1, C2, C3 at their chosen
     values and C4), the pair with the highest score. Ties: the cheaper control (as in 2), then the
     cheaper searcher by pilot C0 mean wall time, ga < de_S < gwo_S < bpso_S.
  Two scores are tied when they differ by at most 1e-12 (floating-point noise only).

The compute tier (section 10) is decided from pilot timings, not here. Output: a JSON file with the
choices, every candidate's overall and per-dataset scores, the run count per cell, the commit of the
runs and the commit of this script.
"""

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from fsbench.run import expand, load_plan  # noqa: E402

SEARCHERS = ["ga", "bpso_S", "de_S", "gwo_S"]
GRID = {"C1": [3, 5, 10], "C2": [0.5, 1.0, 2.0], "C3": [2, 4, 8]}
CONTROL_COST_ORDER = ["C4", "C1", "C3", "C2"]
SEARCHER_COST_ORDER = ["ga", "de_S", "gwo_S", "bpso_S"]
CHECKPOINT = "2000"
TIE_TOL = 1e-12


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def load_runs(plan, results_root):
    """{(dataset, searcher, control, value): [outer AUC, ...]} for the plan's search units."""
    with open(results_root / "MANIFEST.csv", encoding="utf-8", newline="") as f:
        latest = {r["path"]: r for r in csv.DictReader(f)}
    cells = defaultdict(list)
    commits = set()
    problems = []
    for u in expand(plan):
        if u["kind"] != "search" or u["searcher"] not in SEARCHERS:
            continue
        row = latest.get(u["path"])
        if row is None or row["status"] != "ok" or not (results_root / u["path"]).is_file():
            problems.append(f"missing or not ok: {u['path']}")
            continue
        data = (results_root / u["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            problems.append(f"SHA-256 mismatch: {u['path']}")
            continue
        r = json.loads(gzip.decompress(data))
        if r.get("plan") != plan["name"] or r.get("group") != "development" or r.get("status") != "ok":
            problems.append(f"wrong plan, group or status: {u['path']}")
            continue
        if (r["searcher"], r["control"], r["search_seed"]) != (u["searcher"], u["control"], u["search_seed"]):
            problems.append(f"unit mismatch: {u['path']}")
            continue
        commits.add(r["git_commit"])
        ds = u["dataset"]
        if u["control"] == "C0":
            if "c4" not in r or CHECKPOINT not in r["c4"]:
                problems.append(f"no C4 record at {CHECKPOINT}: {u['path']}")
                continue
            cells[(ds, u["searcher"], "C4", None)].append(r["c4"][CHECKPOINT]["outer_auc"])
        else:
            cells[(ds, u["searcher"], u["control"], float(u["param"]))].append(
                r["checkpoints"][CHECKPOINT]["outer_auc"])
    if len(commits) > 1:
        problems.append(f"runs come from {len(commits)} commits: {sorted(commits)}")
    return cells, commits, problems


def score(cells, datasets, keys):
    """Mean over datasets of the per-dataset mean over every run in the (searcher, control, value) keys."""
    per_ds = {}
    for d in datasets:
        vals = [v for k in keys for v in cells[(d, *k)]]
        per_ds[d] = sum(vals) / len(vals)
    return sum(per_ds.values()) / len(per_ds), per_ds


def pick(candidates, tiebreak):
    """candidates: [(label, score)]; highest score, ties (within TIE_TOL) by tiebreak(label) ascending."""
    best = max(s for _, s in candidates)
    tied = [c for c in candidates if best - c[1] <= TIE_TOL]
    return min(tied, key=lambda c: tiebreak(c[0]))[0], len(tied) > 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(REPO_ROOT / "configs/development.yaml"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=None, help="default: <results-dir>/stage2/stage2_choices.json")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    if plan["group"] != "development":
        sys.exit("stage2_select uses development runs only")
    for ctrl, grid in GRID.items():
        values = plan.get("controls", {}).get(ctrl)
        values = values if isinstance(values, list) else [values]
        if sorted(float(v) for v in values) != [float(v) for v in grid]:
            sys.exit(f"plan does not contain the full {ctrl} grid {grid}")
    if not set(SEARCHERS) <= set(plan.get("c0_searchers", [])) or set(plan.get("control_searchers", [])) != set(SEARCHERS):
        sys.exit("plan must run C0 and the controls on " + ", ".join(SEARCHERS))
    results_root = Path(args.results_dir).resolve()
    cells, commits, problems = load_runs(plan, results_root)
    if problems:
        print("refusing to select: " + str(len(problems)) + " problems")
        for p in problems[:20]:
            print("  " + p)
        return 1
    datasets = list(plan["datasets"])
    n_expected = len(plan["repeats"]) * len(plan["folds"]) * len(plan["search_seed_indices"])
    counts = {f"{d}|{s}|{c}|{v}": len(x) for (d, s, c, v), x in cells.items()}
    assert all(n == n_expected for n in counts.values()), "unequal run counts per cell"

    out = {"plan": plan["name"], "checkpoint": int(CHECKPOINT), "datasets": datasets, "searchers": SEARCHERS,
           "runs_per_cell": n_expected, "runs_commit": sorted(commits)[0],
           "script_commit": git("rev-parse", "HEAD"), "tie_tolerance": TIE_TOL,
           "control_cost_order": CONTROL_COST_ORDER, "searcher_cost_order": SEARCHER_COST_ORDER}

    # 1. control hyperparameters
    chosen, rule1 = {}, {}
    for ctrl, grid in GRID.items():
        cands = []
        for v in grid:
            s, per_ds = score(cells, datasets, [(sr, ctrl, float(v)) for sr in SEARCHERS])
            cands.append((v, s))
            rule1.setdefault(ctrl, []).append({"value": v, "score": s, "per_dataset": per_ds})
        chosen[ctrl], tie = pick(cands, tiebreak=lambda v: v)
        rule1[ctrl] = {"candidates": rule1[ctrl], "chosen": chosen[ctrl], "tie_broken": tie}
    out["rule1_hyperparameters"] = rule1

    def key(ctrl, searcher):
        return (searcher, ctrl, None if ctrl == "C4" else float(chosen[ctrl]))

    # 2. primary control
    cands, rule2 = [], []
    for ctrl in ["C1", "C2", "C3", "C4"]:
        s, per_ds = score(cells, datasets, [key(ctrl, sr) for sr in SEARCHERS])
        cands.append((ctrl, s))
        rule2.append({"control": ctrl, "value": chosen.get(ctrl), "score": s, "per_dataset": per_ds})
    primary_control, tie2 = pick(cands, tiebreak=CONTROL_COST_ORDER.index)
    out["rule2_primary_control"] = {"candidates": rule2, "chosen": primary_control, "tie_broken": tie2}

    # 3. primary controlled wrapper
    cands, rule3 = [], []
    for ctrl in ["C1", "C2", "C3", "C4"]:
        for sr in SEARCHERS:
            s, per_ds = score(cells, datasets, [key(ctrl, sr)])
            cands.append(((sr, ctrl), s))
            rule3.append({"searcher": sr, "control": ctrl, "value": chosen.get(ctrl), "score": s,
                          "per_dataset": per_ds})
    (w_searcher, w_control), tie3 = pick(
        cands, tiebreak=lambda sc: (CONTROL_COST_ORDER.index(sc[1]), SEARCHER_COST_ORDER.index(sc[0])))
    out["rule3_primary_wrapper"] = {"candidates": rule3, "chosen": {"searcher": w_searcher, "control": w_control,
                                                                    "value": chosen.get(w_control)},
                                    "tie_broken": tie3}
    out["choices"] = {"P": chosen["C1"], "lambda": chosen["C2"], "r": chosen["C3"],
                      "primary_control": primary_control,
                      "primary_wrapper": {"searcher": w_searcher, "control": w_control}}

    path = Path(args.out) if args.out else results_root / "stage2" / "stage2_choices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["choices"], indent=2))
    print(f"written: {path} (sha256 {hashlib.sha256(path.read_bytes()).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
