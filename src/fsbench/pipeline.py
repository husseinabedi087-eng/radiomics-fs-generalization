"""Pre-processing inside each outer training fold (PREREG_STAGE1 section 3).

Fitted on the outer-training part only, then applied to the outer-test part:
  1. drop features that are constant in the outer-training part (one distinct non-missing value,
     or all missing);
  2. median imputation (medians of the outer-training part);
  3. standardization to zero mean and unit variance (outer-training mean and SD);
  4. univariate pre-filter: the top K features by the ANOVA F statistic computed on the
     outer-training part, K = min(200, number of remaining features); ties keep the original
     column order (stable sort); a non-finite F counts as the lowest.
Returns the transformed train and test arrays (columns = the K-feature pool, in decreasing F) and
the pool-index -> original feature name mapping.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

K_MAX = 200


@dataclass
class PoolTransform:
    kept_columns: list      # after dropping constants (original names)
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    pool_order: np.ndarray  # positions (within kept_columns) of the K pool features, by decreasing F
    f_scores: np.ndarray    # F of the pool features, same order
    pool_names: list        # pool index -> original feature name

    def transform(self, X):
        A = np.asarray(pd.DataFrame(X)[self.kept_columns], dtype=float)
        A = np.where(np.isnan(A), self.medians, A)
        A = (A - self.means) / self.scales
        return A[:, self.pool_order]


def fit(X_train, y_train, k_max=K_MAX):
    X_train = pd.DataFrame(X_train)
    n_unique = X_train.nunique(dropna=True)
    kept = [c for c in X_train.columns if n_unique[c] > 1]
    A = np.asarray(X_train[kept], dtype=float)
    medians = np.nanmedian(A, axis=0)
    A = np.where(np.isnan(A), medians, A)
    means = A.mean(axis=0)
    scales = A.std(axis=0)
    scales[scales == 0] = 1.0  # cannot occur after dropping constants and median imputation of a non-constant column
    A = (A - means) / scales
    F, _ = f_classif(A, np.asarray(y_train))
    F = np.where(np.isfinite(F), F, -np.inf)
    K = min(k_max, len(kept))
    order = np.argsort(-F, kind="stable")[:K]
    return PoolTransform(kept_columns=kept, medians=medians, means=means, scales=scales, pool_order=order,
                         f_scores=F[order], pool_names=[str(kept[i]) for i in order])


def fit_transform(X_train, y_train, X_test, k_max=K_MAX):
    """Fit on the outer-training part only; return (Xtr_pool, Xte_pool, pool_names, transform)."""
    t = fit(X_train, y_train, k_max)
    return t.transform(X_train), t.transform(X_test), t.pool_names, t
