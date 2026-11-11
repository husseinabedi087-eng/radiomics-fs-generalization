"""Generalization controls C1-C3 (PREREG_STAGE1 section 7). C4 is computed later from C0 archives.

Every subset a control scores counts against the budget (section 4). Each control provides an
evaluator with the same interface as fsbench.Evaluator; `returned()` gives the subset the run returns
(used for the checkpoints and the result).

C1, early stopping on a sealed selection fold: the inner fitness is the mean AUC over a fixed
    stratified 3-fold split of the other 80% (C1 fit part). After every 100 search evaluations the
    best-so-far subset (by inner fitness) is scored on the selection fold: L2 logistic regression
    fitted on the whole C1 fit part, AUC on the selection fold; this costs one evaluation. The run
    returns the subset with the best selection-fold AUC (strict improvement; the first such subset
    is kept on ties) and stops after P selection checkpoints without improvement.
C2, variance-penalized fitness: mean - lambda * SD of AUC over the 3 x 3 repeated stratified split
    of the outer-training part; one evaluation per subset.
C3, flatness fitness: mean of the subset's own inner fitness and that of r one-bit neighbours (each
    flips one bit chosen uniformly without replacement); 1 + r evaluations per subset. Neighbour
    bits come from a generator separate from the searcher's (derived from the search seed), so the
    searcher's own random stream is the same as in C0.
"""

import numpy as np

from .evaluator import BudgetExhausted, Evaluator
from .fitness import holdout_auc

SELECTION_EVERY = 100


class EarlyStopped(BudgetExhausted):
    """C1 patience exhausted; treated like the end of the budget by the searchers."""


class C1Evaluator(Evaluator):
    def __init__(self, inner_fitness, K, X_fit, y_fit, X_sel, y_sel, patience, counter, budget=2000, **kw):
        super().__init__(inner_fitness, K, budget=budget, **kw)
        self.X_fit, self.y_fit, self.X_sel, self.y_sel = X_fit, y_fit, X_sel, y_sel
        self.patience = int(patience)
        self.counter = counter
        self.n_search = 0
        self.sel_best_auc = -np.inf
        self.sel_best_mask = None
        self.sel_best_inner = None
        self.no_improve = 0
        self.selection_log = []  # (n_evals after scoring, selection AUC, size)
        self.stopped_early = False

    def returned(self):
        if self.sel_best_mask is None:
            return super().returned()
        return self.sel_best_mask, self.sel_best_inner

    def __call__(self, mask):
        fit = super().__call__(mask)
        self.n_search += 1
        if self.n_search % SELECTION_EVERY == 0:
            self._selection_checkpoint()
        return fit

    def _selection_checkpoint(self):
        cand = self.best_mask.copy()
        self._count_evaluation()  # the selection-fold scoring is one evaluation
        auc = holdout_auc(self.X_fit, self.y_fit, self.X_sel, self.y_sel, cand, self.counter)
        self.selection_log.append((self.n_evals, auc, int(cand.sum())))
        if auc > self.sel_best_auc:
            self.sel_best_auc, self.sel_best_mask, self.sel_best_inner = auc, cand, self.best_fitness
            self.no_improve = 0
        else:
            self.no_improve += 1
        self._record_checkpoint_if_due()
        if self.no_improve >= self.patience:
            self.stopped_early = True
            raise EarlyStopped(f"C1 stopped after {self.patience} selection checkpoints without improvement")


class C3Evaluator(Evaluator):
    def __init__(self, inner_fitness, K, r, rng, budget=2000, **kw):
        super().__init__(inner_fitness, K, budget=budget, **kw)
        self.r = int(r)
        self.rng = rng
        self.smooth_best = -np.inf
        self.smooth_best_mask = None

    def returned(self):
        if self.smooth_best_mask is None:
            return super().returned()
        return self.smooth_best_mask, self.smooth_best

    def __call__(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        values = [self.score(mask)]
        for j in self.rng.choice(self.K, size=self.r, replace=False):
            nb = mask.copy()
            nb[j] = not nb[j]
            values.append(self.score(nb))
        val = float(np.mean(values))
        if val > self.smooth_best:
            self.smooth_best, self.smooth_best_mask = val, mask.copy()
        return val
