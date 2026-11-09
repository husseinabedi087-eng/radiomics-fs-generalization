"""Determinism: the same search seed gives the identical returned mask; different seeds give
different masks. Deterministic (noise-free) synthetic fitness; budget 2,000."""

import hashlib

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


def digest(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_determinism(name, save_result):
    a, b = returned_mask(name, 7), returned_mask(name, 7)
    m1, m2 = returned_mask(name, 1), returned_mask(name, 2)
    same_seed_identical = bool(np.array_equal(a, b))
    different_seeds_differ = not np.array_equal(m1, m2)
    save_result(f"determinism_{name}", {
        "searcher": name, "K": K, "budget": 2000, "landscape": "NK, k = 4, tables from seed 123",
        "seed7_run1": digest(a), "seed7_run2": digest(b), "seed1": digest(m1), "seed2": digest(m2),
        "same_seed_identical": same_seed_identical, "different_seeds_differ": different_seeds_differ})
    assert same_seed_identical, f"{name}: seed 7 gave different masks in two runs"
    assert different_seeds_differ, f"{name}: seeds 1 and 2 gave the same mask"
