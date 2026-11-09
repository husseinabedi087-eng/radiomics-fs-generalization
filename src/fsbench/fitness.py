"""Inner fitness of section 4: mean AUC over a fixed stratified 3-fold split, using L2-regularized
logistic regression (C = 1.0, lbfgs, max_iter = 2000, no class weighting) fitted on the candidate
subset. The split is fixed by the caller and shared by every method. The empty subset is handled
by the Evaluator (fitness 0.5) and never reaches this function.

Used here only on synthetic data (tests/test_planted.py).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def fixed_inner_split(y, seed, n_splits=3):
    """List of (train_idx, test_idx) from a stratified 3-fold split with a fixed integer seed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(skf.split(np.zeros(len(y)), y))


def make_inner_cv_auc(X, y, splits):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    def fitness(mask):
        cols = np.flatnonzero(mask)
        aucs = []
        for tr, te in splits:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            model.fit(X[np.ix_(tr, cols)], y[tr])
            aucs.append(roc_auc_score(y[te], model.decision_function(X[np.ix_(te, cols)])))
        return float(np.mean(aucs))

    return fitness
