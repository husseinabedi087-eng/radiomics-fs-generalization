"""Non-search selectors (PREREG_STAGE1 section 6).

All work on the K-feature pool of the outer-training part (after the section 3 pipeline).

Tuning: each hyperparameter value is scored by the inner 3-fold cross-validated AUC on the same
fixed inner split as the wrappers. Inside each inner fold the selector is refitted on the inner-
training part only, the L2 logistic regression of section 4 is fitted on the selected features of
the inner-training part, and the AUC is computed on the inner-test part (an empty selection scores
0.5 in that fold). The value with the highest mean AUC is chosen; ties go to the sparser setting
(smaller k, or smaller C). The selector is then refitted on the whole outer-training part with that
value. Tuning cost is not counted against the search budget (section 6).

Selectors
  all_k     all K pool features, no tuning.
  l1        L1 logistic regression (liblinear, l1_ratio = 1), C over 20 log-spaced values from 0.001
            to 100; selected = non-zero coefficients.
  enet      elastic net logistic regression (saga, l1_ratio = 0.5, max_iter = 2000), same C grid.
            liblinear and saga use random_state = 0 (fixed; not a study seed).
            An empty final selection (no non-zero coefficient) is returned as an empty subset,
            scored 0.5 downstream, and flagged.
  ftest     top-k by the ANOVA F statistic, k in {5, 10, 20, 50, 100, 200} capped at K.
  mrmr      top-k by mRMR, FCQ variant (Ding & Peng 2005): relevance = ANOVA F statistic,
            redundancy = mean absolute Pearson correlation with the already selected features,
            greedy on relevance / redundancy; the first feature has the largest F. Same k grid.
  rf        top-k by random-forest impurity importance (500 trees, own model seed). Same k grid.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

from .fitness import holdout_auc

C_GRID = np.logspace(-3, 2, 20)
K_GRID = (5, 10, 20, 50, 100, 200)
SELECTORS = ("all_k", "l1", "enet", "ftest", "mrmr", "rf")
NEEDS_MODEL_SEED = ("rf",)


def k_grid(K):
    return sorted({min(k, K) for k in K_GRID})


def _stable_rank(scores):
    s = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(-s, kind="stable")


def f_ranking(X, y):
    F, _ = f_classif(X, y)
    return _stable_rank(F)


def mrmr_ranking(X, y, k_max):
    F, _ = f_classif(X, y)
    F = np.where(np.isfinite(F), F, -np.inf)
    Xc = X - X.mean(axis=0)
    sd = Xc.std(axis=0)
    sd[sd == 0] = 1.0
    Z = Xc / sd
    absr = np.abs(Z.T @ Z) / len(X)
    p = X.shape[1]
    selected = [int(np.argmax(F))]  # argmax returns the first index among ties
    red_sum = absr[selected[0]].copy()
    remaining = np.ones(p, dtype=bool)
    remaining[selected[0]] = False
    while len(selected) < min(k_max, p):
        mean_red = red_sum / len(selected)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(mean_red > 0, F / mean_red, np.where(F > 0, np.inf, F))
        score = np.where(remaining, score, -np.inf)
        j = int(np.argmax(score))
        selected.append(j)
        remaining[j] = False
        red_sum += absr[j]
    return np.array(selected)


def rf_ranking(X, y, model_seed):
    rf = RandomForestClassifier(n_estimators=500, random_state=int(model_seed), n_jobs=1)
    rf.fit(X, y)
    return _stable_rank(rf.feature_importances_)


def _penalized(C, kind):
    if kind == "l1":
        return LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", random_state=0)
    return LogisticRegression(C=C, l1_ratio=0.5, solver="saga", max_iter=2000, random_state=0)


def _penalized_selection(X, y, C, kind, counter):
    m = counter.fit(_penalized(C, kind), X, y)
    return np.flatnonzero(m.coef_.ravel() != 0)


def _mask(cols, K):
    m = np.zeros(K, dtype=bool)
    m[np.asarray(cols, dtype=int)] = True
    return m


def run(name, X, y, inner_splits, counter, model_seed=None):
    """Select a subset of the K pool. Returns dict(mask, hyperparameter, grid, inner_scores, flags)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    K = X.shape[1]
    if name == "all_k":
        return {"mask": np.ones(K, dtype=bool), "hyperparameter": None, "grid": [], "inner_scores": [],
                "flags": []}
    if name in NEEDS_MODEL_SEED and model_seed is None:
        raise ValueError(f"{name} needs a model seed")

    if name in ("l1", "enet"):
        grid = [float(c) for c in C_GRID]
        scores = []
        for C in grid:
            fold_aucs = []
            for tr, te in inner_splits:
                cols = _penalized_selection(X[tr], y[tr], C, name, counter)
                fold_aucs.append(holdout_auc(X[tr], y[tr], X[te], y[te], _mask(cols, K), counter))
            scores.append(float(np.mean(fold_aucs)))
        best = int(np.argmax(scores))  # first maximum = smallest C
        cols = _penalized_selection(X, y, grid[best], name, counter)
        flags = [] if len(cols) else ["empty_subset"]
        return {"mask": _mask(cols, K), "hyperparameter": {"C": grid[best]}, "grid": grid, "inner_scores": scores,
                "flags": flags}

    grid = k_grid(K)
    rank_fn = {"ftest": lambda A, b: f_ranking(A, b),
               "mrmr": lambda A, b: mrmr_ranking(A, b, max(grid)),
               "rf": lambda A, b: rf_ranking(A, b, model_seed)}[name]
    fold_ranks = [rank_fn(X[tr], y[tr]) for tr, _ in inner_splits]
    scores = []
    for k in grid:
        fold_aucs = [holdout_auc(X[tr], y[tr], X[te], y[te], _mask(rank[:k], K), counter)
                     for (tr, te), rank in zip(inner_splits, fold_ranks)]
        scores.append(float(np.mean(fold_aucs)))
    best = int(np.argmax(scores))  # first maximum = smallest k
    final_rank = rank_fn(X, y)
    return {"mask": _mask(final_rank[:grid[best]], K), "hyperparameter": {"k": grid[best]}, "grid": grid,
            "inner_scores": scores, "flags": []}
