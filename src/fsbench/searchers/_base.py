"""Shared pieces for all searchers."""

import functools
from dataclasses import dataclass, field

import numpy as np

from ..evaluator import BudgetExhausted


@dataclass
class SearchResult:
    mask: np.ndarray          # best-so-far mask (by inner fitness; first found among ties)
    fitness: float
    n_evals: int
    checkpoints: dict         # {n_evals: Checkpoint}
    trace: dict = field(default_factory=dict)  # algorithm-specific (e.g. schedule values used)


def budgeted(search_fn):
    """Run search_fn(evaluator, K, rng, trace=..., **kw) until it returns or the budget runs out."""

    @functools.wraps(search_fn)
    def run(evaluator, K, rng, **kwargs):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator created from the search seed")
        trace = {}
        try:
            search_fn(evaluator, K, rng, trace=trace, **kwargs)
        except BudgetExhausted:
            pass
        mask, value = evaluator.returned()
        return SearchResult(mask=mask.copy(), fitness=float(value),
                            n_evals=evaluator.n_evals, checkpoints=dict(evaluator.checkpoints), trace=trace)

    return run


def random_mask(K, rng):
    """Initial mask, drawn bit by bit with probability 0.5 (section 5)."""
    return rng.random(K) < 0.5


def linear_schedule(start, end, n):
    """n values from start to end inclusive, linearly spaced (n >= 1)."""
    if n == 1:
        return np.array([float(start)])
    return start + (end - start) * np.arange(n) / (n - 1)


def make_binary_objective(evaluator, K, transfer, rng):
    """Objective for the continuous cores: (position, previous payload) -> (fitness, payload),
    payload = (mask, position) of the agent.

    The initial call (prev is None) draws the mask bit by bit with probability 0.5.
    S-shaped: the transfer is applied to the new position. V-shaped: the transfer is applied to the
    step x_new - x_old (the analogue of the PSO velocity, as for the step vector of the binary
    dragonfly algorithm, Mirjalili 2016, Neural Comput. Appl. 27, 1053-1073) and flips bits of the
    agent's current mask."""
    from ..transfer import binarize

    def objective(position, prev):
        if prev is None:
            mask = random_mask(K, rng)
        else:
            prev_mask, prev_pos = prev
            x = position - prev_pos if transfer == "V" else position
            mask = binarize(x, prev_mask, transfer, rng)
        return evaluator(mask), (mask, np.array(position, copy=True))

    return objective


def make_continuous_objective(f):
    """Objective for minimizing f on a continuous benchmark: maximize -f; no payload."""

    def objective(position, prev_payload):
        return -float(f(position)), None

    return objective
