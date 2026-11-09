"""Non-dominated front and knee rule for control C4 (PREREG_STAGE1 section 7)."""

import numpy as np

from fsbench import Evaluator
from fsbench.archive import knee_point, pareto_front


def test_front_maximizes_fitness_minimizes_size():
    fitness = [0.60, 0.70, 0.65, 0.80, 0.80, 0.75, 0.90]
    size = [1, 3, 3, 6, 8, 5, 12]
    # (3, 0.65) dominated by (3, 0.70); (8, 0.80) dominated by (6, 0.80)
    assert pareto_front(fitness, size).tolist() == [0, 1, 5, 3, 6]


def test_front_keeps_first_of_exact_duplicates():
    assert pareto_front([0.7, 0.7], [4, 4]).tolist() == [0]


def test_knee_is_largest_distance_to_extremes_line():
    # front points (size, fitness): (1, 0.5), (2, 0.9), (10, 0.95); the corner at size 2 is the knee
    fitness, size = [0.5, 0.9, 0.95], [1, 2, 10]
    assert knee_point(fitness, size) == 1
    assert knee_point(fitness, size, normalize=False) == 1


def test_knee_ties_broken_by_smaller_size():
    # normalized front (x = size, y = fitness): (0, 0), (0.25, 0.75), (0.40, 0.90), (1, 1);
    # both interior points lie at distance 0.5 / sqrt(2) from the line y = x -> smaller size wins
    fitness = [0.0, 0.75, 0.90, 1.0]
    size = [0, 25, 40, 100]
    assert knee_point(fitness, size) == 1


def test_knee_with_two_points_returns_smaller_size():
    assert knee_point([0.6, 0.8], [2, 9]) == 0


def test_evaluator_archive_feeds_knee():
    K = 10
    by_size = {1: 0.50, 2: 0.90, 3: 0.92, 4: 0.94, 5: 0.95, 6: 0.94}  # clear corner at size 2
    ev = Evaluator(lambda m: by_size[int(m.sum())], K, budget=20)
    for k in range(1, 7):
        m = np.zeros(K, dtype=bool)
        m[:k] = True
        ev(m)
    i = knee_point(ev.archive_fitness(), ev.archive_sizes())
    assert ev.archive_mask(i).sum() == 2
