"""Non-dominated front and knee point for control C4 (PREREG_STAGE1 section 7).

Objectives: inner fitness (maximize) and subset size (minimize).

Knee rule: on the front, take the two extremes (smallest size and largest fitness) and return
the point with the largest perpendicular distance to the line joining them; ties are broken by
smaller size, then by earlier evaluation.

Scaling: by default both objectives are min-max normalized over the front before distances are
computed, because the perpendicular distance otherwise depends on the units (fitness ~ 0.5-1,
size ~ 1-200) and would be dominated by size. normalize=False uses raw units.
"""

import numpy as np


def pareto_front(fitness, size):
    """Indices of non-dominated points (first evaluation kept among exact duplicates),
    sorted by increasing size."""
    f = np.asarray(fitness, dtype=float)
    s = np.asarray(size, dtype=float)
    # sort by size ascending, fitness descending, then index (earliest first)
    order = np.lexsort((np.arange(len(f)), -f, s))
    front, best_f = [], -np.inf
    for i in order:
        if f[i] > best_f:  # strictly better fitness than every smaller-or-equal-size point
            front.append(int(i))
            best_f = f[i]
    return np.array(front, dtype=int)


def knee_point(fitness, size, normalize=True):
    """Index (into the input arrays) of the knee of the non-dominated front."""
    f = np.asarray(fitness, dtype=float)
    s = np.asarray(size, dtype=float)
    front = pareto_front(f, s)
    if len(front) <= 2:
        return int(front[0])  # no interior point; the smaller size wins the tie
    x, y = s[front], f[front]
    if normalize:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / np.hypot(y1 - y0, x1 - x0)
    best = np.flatnonzero(np.isclose(dist, dist.max(), rtol=0, atol=1e-12))
    return int(front[best[0]])  # front is sorted by size, so the first is the smallest
