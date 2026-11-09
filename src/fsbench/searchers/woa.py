"""Whale optimization algorithm (section 5); Mirjalili & Lewis (2016), Adv. Eng. Softw. 95, 51-67.

Core (shared by the binary searcher and the continuous check against MEALPY 3.0.3 OriginalWOA,
whose update it follows): 20 whales, 100 iterations including the initial one; spiral constant
b = 1. The best whale X* is fixed at the start of each update iteration. For each whale, with
r ~ U(0,1) (one draw for both coefficients, as in MEALPY), A = 2 a r - a, C = 2 r, l ~ U(-1, 1):
    with probability 0.5:  if |A| < 1:  x_new = X* - A |C X* - x|
                           else:        x_new = X_r - A |C X_r - x|,  X_r uniform in the bounds
                                        (MEALPY draws a fresh random position; the paper uses a
                                        random member of the population)
    otherwise:             x_new = X* + exp(b l) cos(2 pi l) |X* - x|
x_new is clipped to the bounds and replaces the whale only if its fitness is greater than or equal
(greedy selection, as in MEALPY). a decreases linearly from 2 (first update) to 0 (last update):
a_schedule().

Binary searcher: positions in [-6, 6], initial positions uniform; initial masks drawn bit by bit
with probability 0.5. New mask from x_new with the S-shaped rule (bit = 1 if u < sigmoid(x)) or the
V-shaped rule applied to the whale's current mask (bit flipped if u < |tanh(x)|); see fsbench.transfer.
"""

import numpy as np

from ._base import budgeted, linear_schedule, make_binary_objective, make_continuous_objective

POP_SIZE = 20
N_ITER = 100
BOUND = 6.0
B_SPIRAL = 1.0


def a_schedule(n_updates):
    """Values of a for the n_updates update iterations: linear from 2.0 to 0.0 inclusive."""
    return linear_schedule(2.0, 0.0, n_updates)


def core(objective, dim, lb, ub, rng, pop_size=POP_SIZE, n_iter=N_ITER, trace=None):
    x = rng.uniform(lb, ub, (pop_size, dim))
    fit, payload = np.empty(pop_size), [None] * pop_size
    for i in range(pop_size):
        fit[i], payload[i] = objective(x[i], None)
    if trace is not None:
        trace["a"] = []
    for a in a_schedule(n_iter - 1):
        if trace is not None:
            trace["a"].append(float(a))
        best = x[int(np.argmax(fit))].copy()
        for i in range(pop_size):
            r = rng.random()
            A, C = 2 * a * r - a, 2 * r
            l = rng.uniform(-1, 1)
            if rng.random() < 0.5:
                if abs(A) < 1:
                    new = best - A * np.abs(C * best - x[i])
                else:
                    xr = rng.uniform(lb, ub, dim)
                    new = xr - A * np.abs(C * xr - x[i])
            else:
                new = best + np.exp(B_SPIRAL * l) * np.cos(2 * np.pi * l) * np.abs(best - x[i])
            new = np.clip(new, lb, ub)
            f_new, p_new = objective(new, payload[i])
            if f_new >= fit[i]:
                x[i], fit[i], payload[i] = new, f_new, p_new
    return x, fit, payload


@budgeted
def run(evaluator, K, rng, trace, transfer="S"):
    core(make_binary_objective(evaluator, K, transfer, rng), K, -BOUND, BOUND, rng, trace=trace)


def minimize(f, dim, lb, ub, rng, pop_size=POP_SIZE, n_iter=N_ITER):
    _, fit, _ = core(make_continuous_objective(f), dim, lb, ub, rng, pop_size, n_iter)
    return float(-fit.max())
