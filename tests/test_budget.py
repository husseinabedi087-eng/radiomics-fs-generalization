"""Budget accounting (PREREG_STAGE1 sections 4 and 9)."""

import numpy as np
import pytest

from fsbench import CHECKPOINTS, BudgetExhausted, Evaluator
from fsbench.searchers import CONFIGS, run_config

K = 40


def weighted_fitness(K, seed=0):
    w = np.random.default_rng(seed).normal(size=K)
    return lambda m: float(w[m].sum())


@pytest.mark.parametrize("name", list(CONFIGS))
def test_searcher_stops_at_exactly_2000(name):
    ev = Evaluator(weighted_fitness(K), K, budget=2000)
    res = run_config(name, ev, K, np.random.default_rng(1))
    assert ev.n_evals == 2000 and res.n_evals == 2000
    assert len(ev.archive_fitness()) == 2000
    assert sorted(res.checkpoints) == list(CHECKPOINTS)
    best = [res.checkpoints[c].best_fitness for c in CHECKPOINTS]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
    assert res.checkpoints[2000].best_fitness == res.fitness
    assert np.array_equal(res.checkpoints[2000].best_mask, res.mask)


def test_empty_mask_counts_and_skips_fitness():
    calls = []
    ev = Evaluator(lambda m: calls.append(1) or 1.0, K, budget=5)
    assert ev(np.zeros(K, dtype=bool)) == 0.5
    assert ev.n_evals == 1 and not calls
    assert ev.archive_sizes().tolist() == [0]


def test_repeated_mask_counts_even_when_cached():
    calls = []
    ev = Evaluator(lambda m: calls.append(1) or 0.7, K, budget=5, cache=True)
    m = np.zeros(K, dtype=bool)
    m[:3] = True
    for _ in range(3):
        ev(m)
    assert ev.n_evals == 3 and len(calls) == 1


def test_budget_exhausted_raises_at_limit():
    ev = Evaluator(lambda m: 0.6, K, budget=3)
    m = np.ones(K, dtype=bool)
    for _ in range(3):
        ev(m)
    with pytest.raises(BudgetExhausted):
        ev(m)
    assert ev.n_evals == 3


def test_checkpoints_record_best_so_far():
    ev = Evaluator(lambda m: float(m.sum()), K, budget=600)
    rng = np.random.default_rng(0)
    for _ in range(600):
        ev(rng.random(K) < 0.5)
    assert sorted(ev.checkpoints) == [250, 500]
    fits = ev.archive_fitness()
    assert ev.checkpoints[250].best_fitness == fits[:250].max()
    assert ev.checkpoints[500].best_fitness == fits[:500].max()
