"""Determinism: the same search seed gives the identical returned mask; different seeds give
different masks. Deterministic (noise-free) synthetic fitness; budget 2,000."""

import numpy as np
import pytest

from fsbench import Evaluator
from fsbench.searchers import CONFIGS, run_config

K = 60


def rugged_fitness(k_epi=4):
    # NK landscape (Kauffman): bit i contributes table_i[pattern of bit i and k_epi other bits].
    # Many local optima, so the returned mask depends on the search path rather than every seed
    # reaching one dominant optimum. Deterministic: fixed tables from a fixed seed.
    rng = np.random.default_rng(123)
    neighbours = np.array([np.r_[i, rng.choice(np.delete(np.arange(K), i), k_epi, replace=False)] for i in range(K)])
    tables = rng.random((K, 2 ** (k_epi + 1)))
    powers = 2 ** np.arange(k_epi + 1)

    def f(m):
        idx = (m[neighbours].astype(int) * powers).sum(axis=1)
        return float(tables[np.arange(K), idx].mean())

    return f


def returned_mask(name, seed):
    ev = Evaluator(rugged_fitness(), K, budget=2000)
    return run_config(name, ev, K, np.random.default_rng(seed)).mask


@pytest.mark.parametrize("name", list(CONFIGS))
def test_same_seed_same_mask(name):
    assert np.array_equal(returned_mask(name, 7), returned_mask(name, 7))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_different_seeds_different_masks(name):
    assert not np.array_equal(returned_mask(name, 1), returned_mask(name, 2)), f"{name}: seeds 1 and 2 gave the same mask"
