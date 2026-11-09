"""Genetic algorithm (section 5).

Population 20, 100 generations including the initial one (20 evaluations per generation, 2,000 in
total). Initial masks drawn bit by bit with probability 0.5. Each generation creates 20 offspring:
parents by binary tournament (two individuals drawn uniformly with replacement; the fitter wins,
the first drawn on ties); uniform crossover with probability 0.9 (each bit from either parent with
probability 0.5; two complementary children per pair), otherwise the children copy the parents;
bit-flip mutation with rate 1/K per bit. Elitism of 1: the best individual of the previous
generation replaces the worst offspring. The run continues until the budget is used.
"""

import numpy as np

from ._base import budgeted, random_mask


@budgeted
def run(evaluator, K, rng, trace, pop_size=20, p_crossover=0.9):
    # 100 generations x 20 evaluations = the 2,000-evaluation budget; the budget ends the run.
    pop = np.array([random_mask(K, rng) for _ in range(pop_size)])
    fit = np.array([evaluator(m) for m in pop])
    gen = 1
    while True:
        def tournament():
            a, b = rng.integers(pop_size, size=2)
            return a if fit[a] >= fit[b] else b

        children = []
        while len(children) < pop_size:
            p1, p2 = pop[tournament()], pop[tournament()]
            if rng.random() < p_crossover:
                take = rng.random(K) < 0.5
                c1, c2 = np.where(take, p1, p2), np.where(take, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children += [c1 ^ (rng.random(K) < 1.0 / K), c2 ^ (rng.random(K) < 1.0 / K)]
        children = np.array(children[:pop_size])
        child_fit = np.empty(pop_size)
        for i in range(pop_size):
            child_fit[i] = evaluator(children[i])
        elite = int(np.argmax(fit))
        worst = int(np.argmin(child_fit))
        children[worst], child_fit[worst] = pop[elite], fit[elite]
        pop, fit = children, child_fit
        gen += 1
        trace["generations"] = gen
