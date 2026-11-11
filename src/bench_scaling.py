"""Time the search phase against n_train on synthetic data (part 6b follow-up to the pilot).

    python src/bench_scaling.py [--repeats 2] [--budget 2000]

Why: the pilot covered n_train 70-166 only, and estimate_compute.py extrapolates to n_train ~775. This
script measures how the search phase scales with n_train beyond the pilot range. It uses no real data.

What it times: fsbench.experiment.run_search (the same inner-fitness, evaluator and searcher code
as the study) on one ga C0 run and one ga C2 run (lambda = 1), with budget 2000 and K = 200. The
synthetic data come from fsbench.run.synthetic_dataset (logistic, 10 informative features, p = 200, so
the section 3 pool has K = 200). The total n is chosen so that outer fold 0 (repeat 0) has
n_train close to 166, 400 and 775. Partitions are derived in memory (fsbench.partitions.derive) and
never written. Each (configuration, n_train) cell runs `--repeats` times with search seeds 0, 1, ...

Only the logged search-phase wall time ("timings"/"search_s") is read and printed; the run's
performance fields are discarded unread. BLAS/OpenMP threads are fixed to 1, as in the runner. Runs are
sequential in one process; absolute times are therefore not comparable with the pilot (6 concurrent
workers), but the ratios are.

Output: per cell the times, their mean, the ratio of the mean to the n_train = 166 mean, the implied
exponent b = log(ratio) / log(n_train / n_train_166) per n_train, and a least-squares b over all
runs (log t = a + b log n_train).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import math
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from fsbench import experiment, partitions
from fsbench.run import synthetic_dataset

TARGET_N_TRAIN = [166, 400, 775]
P = 200
DATA_SEED = 11
CONFIGS = [("ga", "C0", None), ("ga", "C2", 1.0)]


def n_total_for(n_train):
    # 5 outer folds: outer training is 4/5 of n
    return int(round(n_train * 5 / 4))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--budget", type=int, default=2000)
    args = ap.parse_args(argv)

    times = {}  # (searcher, control) -> {n_train: [t, ...]}
    with threadpool_limits(1):
        for target in TARGET_N_TRAIN:
            name = f"SYNTH_{n_total_for(target)}_{P}_{DATA_SEED}"
            X, y = synthetic_dataset(name)
            fd = experiment.prepare_fold(name, X, y, partitions.fold(partitions.derive(name, y), 0, 0))
            n_train = len(fd.y_train)
            if fd.K != 200:
                raise RuntimeError(f"{name}: pool K = {fd.K}, expected 200")
            for searcher, control, param in CONFIGS:
                for k in range(args.repeats):
                    t0 = time.perf_counter()
                    out = experiment.run_search(fd, searcher, control, param, search_seed=k, budget=args.budget)
                    wall = time.perf_counter() - t0
                    t = out["timings"]["search_s"]
                    del out
                    times.setdefault((searcher, control), {}).setdefault(n_train, []).append(t)
                    print(f"  {searcher} {control:<3} n_train {n_train:4d}  seed {k}: search {t:7.1f} s "
                          f"(whole unit {wall:7.1f} s)", flush=True)

    print(f"\nSearch-phase time vs n_train (K = 200, budget {args.budget}, {args.repeats} repeats, 1 thread):")
    for (searcher, control), by_n in times.items():
        ns = sorted(by_n)
        base_n, base_t = ns[0], float(np.mean(by_n[ns[0]]))
        label = f"{searcher} {control}" + (" (lambda = 1)" if control == "C2" else "")
        print(f"\n  {label}")
        for n in ns:
            m = float(np.mean(by_n[n]))
            ratio = m / base_t
            b = "" if n == base_n else f"  implied b = {math.log(ratio) / math.log(n / base_n):5.2f}"
            print(f"    n_train {n:4d}: times {', '.join(f'{t:.1f}' for t in by_n[n])} s; mean {m:7.1f} s; "
                  f"ratio {ratio:5.2f}{b}")
        ln = np.log([n for n in ns for _ in by_n[n]])
        lt = np.log([t for n in ns for t in by_n[n]])
        b_ls = np.polyfit(ln, lt, 1)[0]
        print(f"    least-squares b over all runs: {b_ls:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
