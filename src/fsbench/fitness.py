"""Model fitting and scoring (PREREG_STAGE1 section 4).

Every model in the study's scoring is L2-regularized logistic regression: C = 1.0, lbfgs,
max_iter = 2000, no class weighting. lbfgs is deterministic, so these models have no model seed.

- inner fitness: mean AUC over a fixed split (list of (train, test) positions) of the given data,
  fitted on the candidate subset;
- outer evaluation: fitted on the whole outer-training part, scored on the outer-test part (AUC and
  Brier score). An empty subset predicts the outer-training prevalence (AUC 0.5).
Convergence warnings are counted in a shared WarningCounter, never silenced without counting.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

LR_PARAMS = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}


class WarningCounter:
    def __init__(self):
        self.convergence = 0

    def fit(self, model, X, y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        self.convergence += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        return model


def l2_model():
    return LogisticRegression(**LR_PARAMS)


def split_auc(X, y, cols, splits, counter):
    """Mean AUC over (train, test) position pairs for the columns `cols`; 0.5 if cols is empty."""
    if len(cols) == 0:
        return 0.5
    aucs = []
    for tr, te in splits:
        m = counter.fit(l2_model(), X[np.ix_(tr, cols)], y[tr])
        aucs.append(roc_auc_score(y[te], m.decision_function(X[np.ix_(te, cols)])))
    return float(np.mean(aucs))


def make_split_fitness(X, y, splits, counter):
    """Fitness f(mask) = mean AUC over the fixed split (section 4 inner fitness)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)

    def fitness(mask):
        return split_auc(X, y, np.flatnonzero(mask), splits, counter)
    return fitness


def make_c2_fitness(X, y, splits, lam, counter):
    """C2: mean - lambda * SD of AUC over the 3 x 3 repeated split (ddof = 0)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)

    def fitness(mask):
        cols = np.flatnonzero(mask)
        aucs = []
        for tr, te in splits:
            m = counter.fit(l2_model(), X[np.ix_(tr, cols)], y[tr])
            aucs.append(roc_auc_score(y[te], m.decision_function(X[np.ix_(te, cols)])))
        return float(np.mean(aucs) - lam * np.std(aucs))
    return fitness


def holdout_auc(X_fit, y_fit, X_eval, y_eval, mask, counter):
    """AUC of a model fitted on (X_fit, y_fit) and scored on (X_eval, y_eval); 0.5 if mask is empty."""
    cols = np.flatnonzero(mask)
    if len(cols) == 0:
        return 0.5
    m = counter.fit(l2_model(), X_fit[:, cols], y_fit)
    return float(roc_auc_score(y_eval, m.decision_function(X_eval[:, cols])))


def outer_evaluation(X_train, y_train, X_test, y_test, mask, counter):
    """(outer AUC, outer Brier). Used only for logging; never fed back into any search or selection."""
    cols = np.flatnonzero(mask)
    if len(cols) == 0:
        p = np.full(len(y_test), float(np.mean(y_train)))
        return 0.5, float(brier_score_loss(y_test, p))
    m = counter.fit(l2_model(), X_train[:, cols], y_train)
    p = m.predict_proba(X_test[:, cols])[:, 1]
    return float(roc_auc_score(y_test, m.decision_function(X_test[:, cols]))), float(brier_score_loss(y_test, p))


# ---- helpers kept for the synthetic planted-subset test ----

def fixed_inner_split(y, seed, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(skf.split(np.zeros(len(y)), y))


def make_inner_cv_auc(X, y, splits):
    return make_split_fitness(X, y, splits, WarningCounter())
