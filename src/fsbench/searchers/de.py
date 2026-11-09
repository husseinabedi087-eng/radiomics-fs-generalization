"""Differential evolution, DE/rand/1/bin (section 5); Storn & Price (1997), J. Global Optim. 11, 341-359.

Core (shared by the binary searcher and the continuous check against MEALPY 3.0.3 OriginalDE,
strategy 0): population 20, 100 generations including the initial one; F = 0.5, CR = 0.9.
For each target i in turn: three distinct indices r1, r2, r3 != i drawn uniformly;
mutant = x_r1 + F (x_r2 - x_r3); binomial crossover with rate CR plus one forced coordinate
j_rand (Storn & Price; MEALPY omits j_rand, which changes nothing measurable at CR = 0.9);
trial clipped to the bounds; the trial replaces the target immediately if its fitness is greater
than or equal (asynchronous update, as in MEALPY's default mode).

Binary searcher: continuous positions in [-6, 6], initial positions uniform; initial masks drawn
bit by bit with probability 0.5. Trial mask:
  S-shaped: bit = 1 if u < sigmoid(trial position) on every coordinate (Kennedy & Eberhart 1997 rule
            applied to the position).
  V-shaped: on the crossed coordinates, the base vector's (x_r1) bit, flipped if
            u < |tanh(F (x_r2 - x_r3))| (Mirjalili & Lewis 2013 rule applied to the scaled difference
            vector, DE's analogue of the PSO velocity); on the other coordinates, the target's bit.
            This mirrors the continuous operator (mutant = base + scaled difference, then crossover).
Position and mask are replaced together.
"""

import numpy as np

from ._base import budgeted, make_continuous_objective, random_mask
from ..transfer import binarize

POP_SIZE = 20
N_ITER = 100
F = 0.5
CR = 0.9
BOUND = 6.0


def core(objective, dim, lb, ub, rng, pop_size=POP_SIZE, n_iter=N_ITER, trace=None):
    """objective(trial, target_payload, ctx) -> (fitness, payload); ctx holds the DE operator parts
    (base payload, scaled difference, crossover mask) for binary encodings."""
    x = rng.uniform(lb, ub, (pop_size, dim))
    fit, payload = np.empty(pop_size), [None] * pop_size
    for i in range(pop_size):
        fit[i], payload[i] = objective(x[i], None, None)
    for _ in range(n_iter - 1):
        for i in range(pop_size):
            r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
            diff = F * (x[r2] - x[r3])
            mutant = x[r1] + diff
            cross = rng.random(dim) < CR
            cross[rng.integers(dim)] = True
            trial = np.clip(np.where(cross, mutant, x[i]), lb, ub)
            ctx = {"base_payload": payload[r1], "diff": diff, "cross": cross}
            f_new, p_new = objective(trial, payload[i], ctx)
            if f_new >= fit[i]:
                x[i], fit[i], payload[i] = trial, f_new, p_new
    return x, fit, payload


def _binary_objective(evaluator, K, transfer, rng):
    def objective(trial, target_mask, ctx):
        if target_mask is None:
            mask = random_mask(K, rng)
        elif transfer == "S":
            mask = binarize(trial, None, "S", rng)
        else:
            mutant_mask = binarize(ctx["diff"], ctx["base_payload"], "V", rng)
            mask = np.where(ctx["cross"], mutant_mask, target_mask)
        return evaluator(mask), mask
    return objective


@budgeted
def run(evaluator, K, rng, trace, transfer="S"):
    core(_binary_objective(evaluator, K, transfer, rng), K, -BOUND, BOUND, rng, trace=trace)


def minimize(f, dim, lb, ub, rng, pop_size=POP_SIZE, n_iter=N_ITER):
    cont = make_continuous_objective(f)
    _, fit, _ = core(lambda trial, prev, ctx: cont(trial, prev), dim, lb, ub, rng, pop_size, n_iter)
    return float(-fit.max())
