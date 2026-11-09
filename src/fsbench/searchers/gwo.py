"""Grey wolf optimizer (section 5); Mirjalili, Mirjalili & Lewis (2014), Adv. Eng. Softw. 69, 46-61.

Core (shared by the binary searcher and the continuous check against MEALPY 3.0.3 OriginalGWO):
20 wolves, 100 iterations including the initial one. At the start of each update iteration the
three best wolves of the current population are the leaders (alpha, beta, delta). For each wolf:
    A_k = a (2 r - 1),  C_k = 2 r'   (r, r' ~ U(0,1) per coordinate, k = 1, 2, 3)
    X_k = leader_k - A_k |C_k leader_k - x|,   x_new = (X_1 + X_2 + X_3) / 3,  clipped to the bounds.
As in MEALPY's OriginalGWO, the new position replaces the wolf only if its fitness is greater than
or equal (greedy selection; the 2014 paper moves every wolf unconditionally).
The coefficient a decreases linearly from 2 (first update) to 0 (last update): a_schedule().

Binary searcher: positions in [-6, 6], initial positions uniform; initial masks drawn bit by bit
with probability 0.5. New mask from x_new with the S-shaped rule (bit = 1 if u < sigmoid(x)) or the
V-shaped rule applied to the wolf's current mask (bit flipped if u < |tanh(x)|); see fsbench.transfer.
"""

import numpy as np

from ._base import budgeted, linear_schedule, make_binary_objective, make_continuous_objective

POP_SIZE = 20
N_ITER = 100
BOUND = 6.0


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
        leaders = x[np.argsort(-fit, kind="stable")[:3]].copy()
        for i in range(pop_size):
            parts = []
            for lead in leaders:
                A = a * (2 * rng.random(dim) - 1)
                C = 2 * rng.random(dim)
                parts.append(lead - A * np.abs(C * lead - x[i]))
            new = np.clip(sum(parts) / 3.0, lb, ub)
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
