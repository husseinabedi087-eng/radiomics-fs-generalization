"""Data-health report for the runs of one or more plans (no performance value is read).

    python src/run_health.py --plan configs/confirmation.yaml --plan configs/confirmation_s2.yaml

Reads, from the run files listed in results/MANIFEST.csv for the given plans, ONLY these fields:
status, git_commit, utc_start, blas_threads, configuration, kind, control, convergence_warnings,
flags (baselines), n_evals, c1.stopped_early, K, n_features_raw, prefilter and timings. It never reads or
prints an outer or inner AUC, a Brier score or a selected subset.

Reports per plan: status and commits; convergence warnings per configuration; baselines flagged
empty_subset; C1 runs that ended before the first checkpoint (n_evals < 250); BLAS/OpenMP thread
settings; runs that started before the commit that added docs/PREREG_STAGE2.sha256 (confirmation group);
K against the number of features (prefilter); wall time of the main phase per configuration.
"""

import argparse
import csv
import gzip
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from fsbench.run import expand, load_plan  # noqa: E402

STAGE2 = "docs/PREREG_STAGE2.sha256"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="append", required=True)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args(argv)
    root = Path(args.results_dir)
    with open(root / "MANIFEST.csv", encoding="utf-8", newline="") as f:
        latest = {r["path"]: r for r in csv.DictReader(f)}
    added = subprocess.run(["git", "-C", str(REPO_ROOT), "log", "--diff-filter=A", "--format=%cI", "--", STAGE2],
                           capture_output=True, text=True).stdout.strip().splitlines()
    frozen = datetime.fromisoformat(added[-1]) if added else None
    print(f"stage 2 committed at: {frozen.isoformat() if frozen else 'not committed'}")

    for plan_path in args.plan:
        plan = load_plan(plan_path)
        units = expand(plan)
        status, commits, threads = Counter(), Counter(), Counter()
        warn = defaultdict(lambda: [0, 0, 0])
        empty = []
        c1_early, c1_n, c1_stopped = [], 0, 0
        before_freeze = 0
        k_below_p, k_eq_p = 0, 0
        wall = defaultdict(list)
        missing = 0
        for u in units:
            row = latest.get(u["path"])
            f = root / u["path"]
            if row is None or not f.is_file():
                missing += 1
                continue
            r = json.loads(gzip.decompress(f.read_bytes()))
            cfg = r["configuration"]
            status[r["status"]] += 1
            commits[r["git_commit"][:7]] += 1
            threads[json.dumps(r.get("blas_threads"))] += 1
            if frozen and plan["group"] == "confirmation" and datetime.fromisoformat(r["utc_start"]) < frozen:
                before_freeze += 1
            if r["status"] != "ok":
                continue
            w = warn[cfg]
            w[0] += r["convergence_warnings"]
            w[1] += r["convergence_warnings"] > 0
            w[2] += 1
            if r["kind"] == "baseline" and "empty_subset" in r.get("flags", []):
                empty.append((cfg, r["dataset"], r["repeat"], r["fold"], r.get("model_seed")))
            if r["kind"] == "search" and r["control"] == "C1":
                c1_n += 1
                c1_stopped += bool(r["c1"]["stopped_early"])
                if r["n_evals"] < 250:
                    c1_early.append((cfg, r["dataset"], r["repeat"], r["fold"], r["seed_index"], r["n_evals"]))
            if r["K"] < r["n_features_raw"]:
                k_below_p += 1
            else:
                k_eq_p += 1
            t = r["timings"]
            wall[cfg].append(t.get("search_s", t.get("selection_s")))

        print(f"\n=== plan {plan['name']} ({plan['group']}, prefilter {plan.get('prefilter', True)}): "
              f"{len(units)} units, {missing} missing")
        print(f"status: {dict(status)}; commits: {dict(commits)}")
        print(f"runs started before the stage-2 commit: {before_freeze}")
        print("thread settings (runs):")
        for k, v in threads.items():
            print(f"  {v:6d}  {k}")
        print("convergence warnings per configuration (total; runs with > 0 / runs):")
        for cfg in sorted(warn):
            t, k, n = warn[cfg]
            print(f"  {cfg:<24} {t:7d}   {k:4d}/{n}")
        print(f"configurations with any warning: {sum(1 for v in warn.values() if v[0])} of {len(warn)}")
        print(f"baseline runs flagged empty_subset: {len(empty)}")
        for cfg, n in sorted(Counter(e[0] for e in empty).items()):
            ds = Counter(e[1] for e in empty if e[0] == cfg)
            print(f"  {cfg}: {n} runs on {len(ds)} datasets: " + ", ".join(f"{d} ({c})" for d, c in sorted(ds.items())))
        if c1_n:
            print(f"C1 runs: {c1_n}; stopped early: {c1_stopped}; ended before the first checkpoint (n_evals < 250): "
                  f"{len(c1_early)}")
            for e in c1_early:
                print("  ", e)
        print(f"K below the number of features: {k_below_p} runs; K equal: {k_eq_p} runs")
        print("main-phase wall time per configuration (s): min / median / max")
        for cfg in sorted(wall):
            ts = wall[cfg]
            print(f"  {cfg:<24} {min(ts):8.1f} {statistics.median(ts):8.1f} {max(ts):8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
