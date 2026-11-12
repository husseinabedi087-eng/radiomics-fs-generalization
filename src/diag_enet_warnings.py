"""Attribute elastic-net convergence warnings to C values and to fits (development data only).

    python src/diag_enet_warnings.py [--n-jobs 6]

Why: the run files store one convergence-warning total per run (fsbench.fitness.WarningCounter), which
cannot say which C of the grid or which fit (inner tuning or final) produced the warnings.

What it does, for every development dataset and outer fold of configs/development.yaml (repeat 0, folds
0-4), with the same data access, partitions and section 3 pipeline as the runner
(fsbench.run.load_data, fsbench.partitions.get, fsbench.experiment.prepare_fold): for max_iter in
{2000, 10000} and every C of the grid, it refits the elastic net exactly as fsbench.baselines does
(saga, l1_ratio = 0.5, random_state = 0, tol default) on each inner-training part (the tuning fits) and
on the whole outer-training part (the final fit at that C), and counts ConvergenceWarnings per fit.
It never computes an AUC, Brier score or any other performance value and never scores a subset.

It also reads, from the enet run files, only the "hyperparameter" (the chosen C) and the
"convergence_warnings" total, for the original runs (plan development, max_iter 2000) and, if present,
the rerun (plan dev_enet_fix, max_iter 10000). Reconciliation per run: tuning-fit warnings over the
grid + final-fit warnings at the chosen C, compared with the run file total; any difference comes from
the L2 logistic regressions that score the selected subsets (not refitted here).

Output: printed tables and results/diagnostics/enet_warnings.json.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT = REPO_ROOT / "results" / "diagnostics" / "enet_warnings.json"
MAX_ITERS = (2000, 10000)
PLANS = {2000: "development", 10000: "dev_enet_fix"}


def _fit_warnings(X, y, C, max_iter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, l1_ratio=0.5, solver="saga", max_iter=max_iter, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X, y)
    return sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def dataset_counts(ds, folds):
    from threadpoolctl import threadpool_limits
    from fsbench import experiment, partitions, run
    from fsbench.baselines import C_GRID
    with threadpool_limits(1):
        X, y = run.load_data(ds, "development")
        parts = partitions.get(ds, y)
        out = []
        for f in folds:
            fd = experiment.prepare_fold(ds, X, y, partitions.fold(parts, 0, f))
            for mi in MAX_ITERS:
                for C in C_GRID:
                    inner = sum(_fit_warnings(fd.X_train[tr], fd.y_train[tr], float(C), mi) for tr, _ in fd.inner)
                    final = _fit_warnings(fd.X_train, fd.y_train, float(C), mi)
                    out.append({"dataset": ds, "fold": f, "max_iter": mi, "C": float(C), "inner": inner, "final": final})
        return out


def run_file(plan, ds, f):
    p = REPO_ROOT / "results" / "runs" / "development" / plan / ds / "baseline__enet" / f"r0_f{f}_s0.json.gz"
    if not p.exists():
        return None
    r = json.loads(gzip.decompress(p.read_bytes()))
    return {"C": r["hyperparameter"]["C"], "warnings": r["convergence_warnings"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args(argv)
    from fsbench.run import load_plan
    plan = load_plan(REPO_ROOT / "configs" / "development.yaml")
    datasets, folds = list(plan["datasets"]), list(plan["folds"])
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=mp.get_context("spawn")) as pool:
        rows = [r for part in pool.map(dataset_counts, datasets, [folds] * len(datasets)) for r in part]

    idx = {(r["dataset"], r["fold"], r["max_iter"], r["C"]): r for r in rows}
    by_c = {mi: defaultdict(lambda: [0, 0]) for mi in MAX_ITERS}
    for r in rows:
        by_c[r["max_iter"]][r["C"]][0] += r["inner"]
        by_c[r["max_iter"]][r["C"]][1] += r["final"]
    print("ConvergenceWarnings by C (summed over 12 datasets x 5 folds): inner tuning fits (3 per fold) / final fit")
    print(f"  {'C':>10}  " + "  ".join(f"max_iter {mi:>5}: inner  final" for mi in MAX_ITERS))
    for C in sorted(by_c[MAX_ITERS[0]]):
        print(f"  {C:10.4g}  " + "  ".join(f"{'':>15}{by_c[mi][C][0]:5d}  {by_c[mi][C][1]:5d}" for mi in MAX_ITERS))
    for mi in MAX_ITERS:
        tot_i = sum(v[0] for v in by_c[mi].values())
        tot_f = sum(v[1] for v in by_c[mi].values())
        print(f"  total max_iter {mi}: inner {tot_i}, final (all C) {tot_f}")

    print("\nPer run: run-file total vs refit (inner over the grid + final at the chosen C)")
    recon = []
    for ds in datasets:
        for f in folds:
            rec = {"dataset": ds, "fold": f}
            for mi in MAX_ITERS:
                rf = run_file(PLANS[mi], ds, f)
                inner = sum(idx[(ds, f, mi, float(C))]["inner"] for C in by_c[mi])
                if rf is None:
                    rec[str(mi)] = {"run_file": None, "refit_inner": inner}
                    continue
                final = idx[(ds, f, mi, float(rf["C"]))]["final"]
                rec[str(mi)] = {"run_file": rf["warnings"], "chosen_C": rf["C"], "refit_inner": inner,
                                "refit_final_at_chosen_C": final, "unexplained": rf["warnings"] - inner - final}
            recon.append(rec)
            cells = []
            for mi in MAX_ITERS:
                x = rec[str(mi)]
                cells.append(f"max_iter {mi}: " + ("no run file" if x["run_file"] is None else
                             f"run {x['run_file']:4d} = inner {x['refit_inner']:4d} + final {x['refit_final_at_chosen_C']:2d}"
                             f" + other {x['unexplained']:3d}"))
            print(f"  {ds:<22} f{f}  " + " | ".join(cells))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"max_iters": list(MAX_ITERS), "rows": rows, "reconciliation": recon,
               "by_C": {str(mi): {str(C): {"inner": v[0], "final": v[1]} for C, v in sorted(by_c[mi].items())}
                        for mi in MAX_ITERS}}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    import hashlib
    print(f"\nwritten: {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
