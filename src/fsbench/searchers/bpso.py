"""Binary particle swarm optimization (section 5).

Binary version (used for feature selection), Kennedy & Eberhart (1997):
    20 particles; 100 iterations including the initial one (20 evaluations each, 2,000 in total).
    Initial masks drawn bit by bit with probability 0.5; initial velocities 0.
    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x),  r1, r2 ~ U(0,1) per bit,  c1 = c2 = 2,
    v clipped to [-6, 6]; w decreases linearly from 0.9 (first update) to 0.4 (last update).
    New mask from v with the S- or V-shaped rule in fsbench.transfer (S: bit = 1 if u < sigmoid(v);
    V: bit flipped if u < |tanh(v)|, Mirjalili & Lewis 2013). Every particle moves every iteration.
    pbest is replaced if the new fitness is greater than or equal; gbest is the best pbest,
    updated after each iteration.

Continuous core (only for the check against MEALPY 3.0.3 LDW_PSO, section 9), following that
implementation: positions and velocities uniform in the bounds and in [-v_max, v_max] with
v_max = 0.5 (ub - lb); the same velocity update with the same linear inertia schedule and velocity
clipping to v_max; x_new = x + v, coordinates outside the bounds replaced by uniform random values;
the particle position and pbest are replaced only if the new value is better or equal (greedy, as
in MEALPY); gbest updated after each iteration.
"""

import numpy as np

from ._base import budgeted, linear_schedule, random_mask
from ..transfer import binarize

POP_SIZE = 20
N_ITER = 100
C1 = C2 = 2.0
W_START, W_END = 0.9, 0.4
V_CLAMP = 6.0


def inertia_schedule(n_updates):
    return linear_schedule(W_START, W_END, n_updates)


@budgeted
def run(evaluator, K, rng, trace, transfer="S"):
    x = np.array([random_mask(K, rng) for _ in range(POP_SIZE)])
    v = np.zeros((POP_SIZE, K))
    fit = np.array([evaluator(m) for m in x])
    pbest, pfit = x.copy(), fit.copy()
    g = int(np.argmax(pfit))
    ws = inertia_schedule(N_ITER - 1)
    trace["w"] = []
    t = 0
    while True:
        w = ws[min(t, len(ws) - 1)]
        trace["w"].append(float(w))
        for i in range(POP_SIZE):
            xi = x[i].astype(float)
            r1, r2 = rng.random(K), rng.random(K)
            v[i] = np.clip(w * v[i] + C1 * r1 * (pbest[i] - xi) + C2 * r2 * (pbest[g] - xi), -V_CLAMP, V_CLAMP)
            x[i] = binarize(v[i], x[i], transfer, rng)
            f_new = evaluator(x[i])
            if f_new >= pfit[i]:
                pbest[i], pfit[i] = x[i].copy(), f_new
        g = int(np.argmax(pfit))
        t += 1


def minimize(f, dim, lb, ub, rng, pop_size=POP_SIZE, n_iter=N_ITER):
    """Continuous core; returns the best (lowest) value of f found."""
    lb, ub = float(lb), float(ub)
    v_max = 0.5 * (ub - lb)
    x = rng.uniform(lb, ub, (pop_size, dim))
    v = rng.uniform(-v_max, v_max, (pop_size, dim))
    fit = np.array([-f(xi) for xi in x])
    pbest, pfit = x.copy(), fit.copy()
    g = int(np.argmax(pfit))
    for w in inertia_schedule(n_iter - 1):
        for i in range(pop_size):
            r1, r2 = rng.random(dim), rng.random(dim)
            v[i] = np.clip(w * v[i] + C1 * r1 * (pbest[i] - x[i]) + C2 * r2 * (pbest[g] - x[i]), -v_max, v_max)
            new = x[i] + v[i]
            out = (new < lb) | (new > ub)
            new = np.where(out, rng.uniform(lb, ub, dim), new)
            f_new = -f(new)
            if f_new >= fit[i]:
                x[i], fit[i] = new, f_new
            if f_new >= pfit[i]:
                pbest[i], pfit[i] = new.copy(), f_new
        g = int(np.argmax(pfit))
    return float(-pfit.max())
