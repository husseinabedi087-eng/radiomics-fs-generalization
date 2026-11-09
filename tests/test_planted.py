"""Planted subset (PREREG_STAGE1 section 9): synthetic logistic data, n = 150, 200 features of which
10 are informative. Fitness is the inner fitness of section 4 (mean AUC over a fixed stratified
3-fold split, L2 logistic regression C = 1, lbfgs, max_iter = 2000). For every searcher, the
recovery rate (informative features in the returned subset / 10) and the subset size at 2,000
evaluations are reported. This is a report, not a pass/fail gate.

Run with: python -m pytest -m slow -v -s
"""

import time

import numpy as np
import pytest

from fsbench import Evaluator
from fsbench.fitness import fixed_inner_split, make_inner_cv_auc
from fsbench.searchers import CONFIGS, run_config

N, P, N_INFORMATIVE = 150, 200, 10
DATA_SEED = 2026
SEARCH_SEEDS = range(3)


def planted_data(seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, P))
    informative = np.sort(rng.choice(P, size=N_INFORMATIVE, replace=False))
    beta = np.zeros(P)
    beta[informative] = rng.choice([-1.0, 1.0], size=N_INFORMATIVE)
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-X @ beta))).astype(int)
    return X, y, informative


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CONFIGS))
def test_planted_recovery_report(name, report, save_result):
    X, y, informative = planted_data()
    fitness = make_inner_cv_auc(X, y, fixed_inner_split(y, seed=DATA_SEED))
    rows, per_seed = [], []
    for seed in SEARCH_SEEDS:
        t0 = time.perf_counter()
        ev = Evaluator(fitness, P, budget=2000)
        res = run_config(name, ev, P, np.random.default_rng(seed))
        hits = int(res.mask[informative].sum())
        rows.append((hits / N_INFORMATIVE, int(res.mask.sum()), res.fitness, time.perf_counter() - t0))
        per_seed.append({"seed": seed, "informative_selected": hits, "recovery": hits / N_INFORMATIVE,
                         "subset_size": int(res.mask.sum()), "inner_auc": res.fitness,
                         "seconds": rows[-1][3], "n_fitness_calls": ev.n_fitness_calls})
    save_result(f"planted_{name}", {"searcher": name, "n": N, "p": P, "n_informative": N_INFORMATIVE,
                                    "data_seed": DATA_SEED, "budget": 2000, "per_seed": per_seed})
    rec = np.array([r[0] for r in rows])
    line = (f"planted {name:<13} recovery mean {rec.mean():.2f} (per seed {', '.join(f'{r:.1f}' for r in rec)})  "
            f"size {', '.join(str(r[1]) for r in rows)}  inner AUC {', '.join(f'{r[2]:.3f}' for r in rows)}  "
            f"time/run {np.mean([r[3] for r in rows]):.1f}s")
    print(line)
    report(line)
    assert len(rows) == len(SEARCH_SEEDS)  # report only; no recovery threshold
