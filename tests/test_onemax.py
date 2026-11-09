"""OneMax, K = 200 (PREREG_STAGE1 section 9): best fitness at 2,000 evaluations is higher than at
250 for every searcher and every seed."""

import numpy as np
import pytest

from fsbench import Evaluator
from fsbench.searchers import CONFIGS, run_config

K = 200
SEEDS = range(5)


def onemax(mask):
    return float(mask.sum()) / K


@pytest.mark.parametrize("name", list(CONFIGS))
def test_onemax_improves_with_budget(name, report):
    gains = []
    for seed in SEEDS:
        ev = Evaluator(onemax, K, budget=2000)
        res = run_config(name, ev, K, np.random.default_rng(seed))
        f250, f2000 = res.checkpoints[250].best_fitness, res.checkpoints[2000].best_fitness
        gains.append((seed, f250, f2000))
    report(f"onemax {name:<13} " + "  ".join(f"s{s}: {a:.3f}->{b:.3f}" for s, a, b in gains))
    failed = [(s, a, b) for s, a, b in gains if not b > a]
    assert not failed, f"{name}: no improvement from 250 to 2000 for seeds {failed}"
