"""Random subset search (section 5): each mask has a size drawn uniformly from 1 to K, then that
many features drawn uniformly without replacement. Runs until the budget is used (2,000 masks)."""

import numpy as np

from ._base import budgeted


@budgeted
def run(evaluator, K, rng, trace):
    while True:
        size = int(rng.integers(1, K + 1))
        mask = np.zeros(K, dtype=bool)
        mask[rng.choice(K, size=size, replace=False)] = True
        evaluator(mask)
