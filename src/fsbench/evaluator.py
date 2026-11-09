"""Budget-aware evaluator (PREREG_STAGE1 section 4).

Every call is one evaluation, including repeated masks: an optional cache avoids recomputing
the fitness of a mask seen before, but the call still counts against the budget. A call made
when the budget is already used raises BudgetExhausted. An empty mask returns 0.5 without
calling the fitness function, and still counts.

Best-so-far is updated only on strict improvement, so among equal fitness values the first
evaluated mask is kept. At each checkpoint (250, 500, 1000, 2000 evaluations by default) the
best-so-far mask and fitness are recorded. Every call appends (fitness, subset size) and the
mask itself to the archive (used by control C4, section 7).
"""

from dataclasses import dataclass

import numpy as np

CHECKPOINTS = (250, 500, 1000, 2000)
EMPTY_MASK_FITNESS = 0.5


class BudgetExhausted(Exception):
    """Raised when an evaluation is requested after the budget has been used."""


@dataclass(frozen=True)
class Checkpoint:
    n_evals: int
    best_fitness: float
    best_size: int
    best_mask: np.ndarray


class Evaluator:
    def __init__(self, fitness, K, budget=2000, checkpoints=CHECKPOINTS, cache=True):
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.fitness = fitness
        self.K = int(K)
        self.budget = int(budget)
        self.checkpoint_at = tuple(sorted(int(c) for c in checkpoints))
        self.use_cache = cache
        self._cache = {}
        self.n_evals = 0
        self.n_fitness_calls = 0  # calls that actually reached the fitness function
        self.best_fitness = -np.inf
        self.best_mask = None
        self.checkpoints = {}
        self._arch_fitness = []
        self._arch_size = []
        self._arch_masks = []  # np.packbits of each evaluated mask

    @property
    def remaining(self):
        return self.budget - self.n_evals

    def __call__(self, mask):
        if self.n_evals >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} evaluations used")
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.K:
            raise ValueError(f"mask has length {mask.shape[0]}, expected {self.K}")
        self.n_evals += 1
        size = int(mask.sum())
        packed = np.packbits(mask)
        if size == 0:
            fit = EMPTY_MASK_FITNESS
        else:
            key = packed.tobytes()
            if self.use_cache and key in self._cache:
                fit = self._cache[key]
            else:
                fit = float(self.fitness(mask.copy()))
                self.n_fitness_calls += 1
                if self.use_cache:
                    self._cache[key] = fit
        if fit > self.best_fitness:
            self.best_fitness = fit
            self.best_mask = mask.copy()
        self._arch_fitness.append(fit)
        self._arch_size.append(size)
        self._arch_masks.append(packed)
        if self.n_evals in self.checkpoint_at:
            self.checkpoints[self.n_evals] = Checkpoint(
                n_evals=self.n_evals, best_fitness=self.best_fitness,
                best_size=int(self.best_mask.sum()), best_mask=self.best_mask.copy())
        return fit

    # ---- archive access ----
    def archive_fitness(self):
        return np.asarray(self._arch_fitness, dtype=float)

    def archive_sizes(self):
        return np.asarray(self._arch_size, dtype=int)

    def archive_mask(self, i):
        return np.unpackbits(self._arch_masks[i], count=self.K).astype(bool)
