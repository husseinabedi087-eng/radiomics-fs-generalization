"""Project the compute of the study plans from the pilot run files (part 6b, item 10).

    python src/estimate_compute.py [--plan-name pilot] [--workers 6] [--plan configs/development.yaml]

Inputs: the pilot run files (results/runs/development/**/*.json.gz with plan == pilot, status ok) and,
for the projection, n and p of every development and confirmation dataset from data/meta/audit.csv and
data/meta/data_handling_rules.json (metadata only; no confirmation data is loaded).

Model, per configuration (searcher x control, or baseline selector), fitted by least squares on the
pilot runs:   log(t) = a + b log(n_train) + c log(K)
where t is the wall time of the search or selection phase, n_train the outer-training size and K the
pool size. If K takes fewer than two distinct values in the pilot, c is fixed to 1 (time proportional
to K) and only a and b are fitted. The standard inner AUC and outer scoring phases are modelled the same
way; fold preparation is modelled per fold. Projections are printed per plan and phase, as CPU hours and
as wall hours at --workers workers with perfect packing (a lower bound on wall time).

Plans projected (section 10):
  development: 12 development datasets, 2 repeats x 5 folds, 3 search seeds per fold; C0 x 12;
               C1, C2, C3 on 4 searchers at all 3 grid values (36 controlled configurations); baselines
               (random forest with 3 model seeds).
  confirmation, core tier: 31 confirmation datasets, same folds and seeds; C0 x 12; C1, C2, C3 on 4
               searchers at one chosen value (12 controlled configurations); baselines. C4 costs nothing.
  confirmation, reduced tier: as core without the V-shaped variants (bpso_V, de_V, gwo_V, woa_V).

--plan <yaml>: additionally project exactly the units of that plan (fsbench.run.expand). A controlled
configuration whose grid value was not in the pilot uses the pilot fit of the same searcher and control
(the pilot ran P = 5, lambda = 1, r = 4 only), so C1 at P = 3 or 10 is only approximated.
"""

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
C0 = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V", "gwo_S", "gwo_V",
      "woa_S", "woa_V"]
CTRL_SEARCHERS = ["ga", "bpso_S", "de_S", "gwo_S"]
BASELINES = ["all_k", "l1", "enet", "ftest", "mrmr", "rf"]
PHASES = ["main", "standard_inner_auc_s", "outer_scoring_s"]


def family(run):
    if run["kind"] == "search":
        return f"{run['searcher']}|{run['control']}"
    return f"baseline|{run['selector']}"


def load_runs(plan_name, results_dir):
    runs = []
    for f in (Path(results_dir) / "runs").rglob("*.json.gz"):
        r = json.loads(gzip.decompress(f.read_bytes()))
        if r.get("plan") == plan_name and r.get("status") == "ok":
            runs.append(r)
    return runs


def fit(points):
    """points: list of (n_train, K, t). Returns (a, b, c, k_fixed, residual_sd)."""
    n = np.log([p[0] for p in points])
    K = np.log([p[1] for p in points])
    t = np.log([max(p[2], 1e-6) for p in points])
    if len(set(np.round(K, 9))) >= 2:
        A = np.column_stack([np.ones_like(n), n, K])
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        a, b, c, fixed = coef[0], coef[1], coef[2], False
    else:
        A = np.column_stack([np.ones_like(n), n])
        coef, *_ = np.linalg.lstsq(A, t - K, rcond=None)
        a, b, c, fixed = coef[0], coef[1], 1.0, True
    resid = t - (a + b * n + c * K)
    return a, b, c, fixed, float(np.std(resid, ddof=0))


def predict(model, n_train, K):
    a, b, c, _, _ = model
    return math.exp(a + b * math.log(n_train) + c * math.log(K))


def dataset_sizes():
    rules = json.loads((REPO_ROOT / "data/meta/data_handling_rules.json").read_text(encoding="utf-8"))
    with open(REPO_ROOT / "data/meta/audit.csv", encoding="utf-8", newline="") as f:
        audit = {r["dataset"]: r for r in csv.DictReader(f)}
    drop_cols = {d: len(v["columns"]) for d, v in rules.get("drop_columns", {}).items()}
    dup = set(rules.get("drop_duplicate_rows", {}))
    out = {}
    for group in ("development", "confirmation"):
        names = [l.strip() for l in (REPO_ROOT / f"data/split/{group}.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        for d in names:
            n = int(audit[d]["n_samples"]) - (int(audit[d]["n_exact_duplicate_rows_with_target"]) if d in dup else 0)
            p = int(audit[d]["n_features"]) - drop_cols.get(d, 0)
            out[d] = (group, n, p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-name", default="pilot")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3, help="search seeds per outer fold")
    ap.add_argument("--folds", type=int, default=10, help="outer folds per dataset (repeats x folds)")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--plan", action="append", default=[], help="plan file(s) to project unit by unit")
    args = ap.parse_args(argv)

    runs = load_runs(args.plan_name, args.results_dir)
    if not runs:
        print(f"no ok run files for plan '{args.plan_name}'")
        return 1
    pts = defaultdict(lambda: defaultdict(list))
    prep = []
    for r in runs:
        tm = r["timings"]
        main_t = tm.get("search_s", tm.get("selection_s"))
        fam = family(r)
        pts[fam]["main"].append((r["n_train"], r["K"], main_t))
        pts[fam]["standard_inner_auc_s"].append((r["n_train"], r["K"], tm["standard_inner_auc_s"]))
        pts[fam]["outer_scoring_s"].append((r["n_train"], r["K"], tm["outer_scoring_s"]))
        if not tm.get("fold_preparation_cached", False):
            prep.append((r["n_train"], r["n_features_raw"], tm["fold_preparation_s"]))
    models = {fam: {ph: fit(p) for ph, p in d.items()} for fam, d in pts.items()}
    prep_model = fit(prep) if prep else None

    pilot_n = sorted({r["n_train"] for r in runs})
    pilot_K = sorted({r["K"] for r in runs})
    print(f"Pilot: {len(runs)} ok runs; n_train {pilot_n[0]}-{pilot_n[-1]}; K values {pilot_K}")
    print("\nPer-configuration fit of the main phase: t = exp(a) n_train^b K^c (seconds)")
    for fam in sorted(models):
        a, b, c, fixed, sd = models[fam]["main"]
        mean_t = np.mean([p[2] for p in pts[fam]["main"]])
        print(f"  {fam:<22} b = {b:5.2f}  c = {c:5.2f}{' (fixed)' if fixed else '        '}  "
              f"log-residual SD {sd:4.2f}  pilot mean {mean_t:7.1f} s  (n = {len(pts[fam]['main'])})")

    sizes = dataset_sizes()
    max_n = max(n for _, n, _ in sizes.values())
    print(f"\nProjection range: n_train up to {0.8 * max_n:.0f} (pilot max {pilot_n[-1]}): "
          f"{'EXTRAPOLATION in n' if 0.8 * max_n > pilot_n[-1] else 'within pilot range'}")

    def plan_cost(group, ctrl_values_per_control, include_v=True):
        searchers = [s for s in C0 if include_v or not s.endswith("_V")]
        fams = [(f"{s}|C0", args.seeds) for s in searchers]
        fams += [(f"{s}|{c}", args.seeds * ctrl_values_per_control) for c in ("C1", "C2", "C3") for s in CTRL_SEARCHERS]
        fams += [(f"baseline|{b}", 3 if b == "rf" else 1) for b in BASELINES]
        per_phase = defaultdict(float)
        for d, (g, n, p) in sizes.items():
            if g != group:
                continue
            n_train, K = 0.8 * n, min(200, p)
            for fam, reps in fams:
                if fam not in models:
                    print(f"  WARNING: no pilot runs for {fam}; skipped")
                    continue
                for ph in PHASES:
                    per_phase[ph] += predict(models[fam][ph], n_train, K) * reps * args.folds
            if prep_model:
                per_phase["fold_preparation"] += predict(prep_model, n_train, p) * args.folds
        return per_phase

    print(f"\nProjected compute ({args.folds} outer folds per dataset, {args.seeds} search seeds per fold; "
          f"wall time at {args.workers} workers assumes perfect packing):")
    for label, group, nvals, include_v in [("development plan (all 3 grid values per control)", "development", 3, True),
                                           ("confirmation, core tier", "confirmation", 1, True),
                                           ("confirmation, reduced tier (no V-shaped)", "confirmation", 1, False)]:
        cost = plan_cost(group, nvals, include_v)
        total = sum(cost.values())
        parts = ", ".join(f"{k.removesuffix('_s')} {v / 3600:.0f} h" for k, v in cost.items())
        print(f"  {label}: {total / 3600:.0f} CPU hours ({parts}) -> {total / 3600 / args.workers:.1f} h "
              f"({total / 3600 / args.workers / 24:.1f} days) at {args.workers} workers")

    for plan_path in args.plan:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from fsbench.run import expand, load_plan
        plan = load_plan(plan_path)
        units = expand(plan)
        per_phase, per_ds, missing = defaultdict(float), defaultdict(float), set()
        folds = set()
        for u in units:
            g, n, p = sizes[u["dataset"]]
            n_train, K = 0.8 * n, min(200, p)
            fam = f"{u['searcher']}|{u['control']}" if u["kind"] == "search" else f"baseline|{u['selector']}"
            if fam not in models:
                missing.add(fam)
                continue
            for ph in PHASES:
                t = predict(models[fam][ph], n_train, K)
                per_phase[ph] += t
                per_ds[u["dataset"]] += t
            key = (u["dataset"], u["repeat"], u["fold"])
            if prep_model and key not in folds:
                folds.add(key)
                per_phase["fold_preparation"] += predict(prep_model, n_train, p)
        total = sum(per_phase.values())
        parts = ", ".join(f"{k.removesuffix('_s')} {v / 3600:.1f} h" for k, v in per_phase.items())
        print(f"\nPlan {plan['name']} ({plan_path}): {len(units)} units")
        if missing:
            print(f"  WARNING: no pilot runs for {sorted(missing)}; those units are not counted")
        print(f"  {total / 3600:.1f} CPU hours ({parts}) -> {total / 3600 / args.workers:.1f} h "
              f"({total / 3600 / args.workers / 24:.2f} days) at {args.workers} workers")
        print("  by dataset (CPU hours; n_train):")
        for d in sorted(per_ds, key=per_ds.get, reverse=True):
            print(f"    {d:<24} {per_ds[d] / 3600:6.1f}   n_train ~{0.8 * sizes[d][1]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
