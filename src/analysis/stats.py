"""Statistical procedures (PREREG_STAGE1 section 8; amendment 1).

All tests act on per-dataset values. Wilcoxon signed-rank: scipy.stats.wilcoxon with
zero_method="wilcox", method="auto"; the method scipy actually used is recorded by replicating its
"auto" rule (scipy 1.18): n > 50 -> asymptotic; otherwise exact if there are no zeros and no ties among
the |d|, else a permutation test if n <= 13, else asymptotic (n counted before zero removal).
"""

import warnings

import numpy as np
from scipy import stats

ALPHA = 0.05
BOOT_SEED = 20261113
N_BOOT = 10_000


def _f(x):
    """float, or None for nan/inf (JSON)."""
    x = float(x)
    return x if np.isfinite(x) else None


def wilcoxon_method_used(d):
    d = np.asarray(d, dtype=float)
    n = d.size
    if n > 50:
        return "asymptotic"
    nz = np.abs(d[d != 0])
    has_zeros = nz.size < n
    has_ties = nz.size != np.unique(nz).size
    if not (has_ties or has_zeros):
        return "exact"
    return "permutation" if n <= 13 else "asymptotic"


def wilcoxon(d, alternative):
    d = np.asarray(d, dtype=float)
    n_nonzero = int(np.count_nonzero(d))
    out = {"alternative": alternative, "n": int(d.size), "n_after_zero_removal": n_nonzero,
           "method": wilcoxon_method_used(d) if d.size else None}
    if n_nonzero == 0:
        return {**out, "statistic": None, "p": None}
    res = stats.wilcoxon(d, zero_method="wilcox", method="auto", alternative=alternative)
    return {**out, "statistic": _f(res.statistic), "p": _f(res.pvalue)}


def tost(d, margin):
    """Two one-sided Wilcoxon tests: lower arm d + m > 0, upper arm d - m < 0; p_TOST = max."""
    d = np.asarray(d, dtype=float)
    low = wilcoxon(d + margin, "greater")
    up = wilcoxon(d - margin, "less")
    p = None if low["p"] is None or up["p"] is None else max(low["p"], up["p"])
    return {"margin": margin, "lower": low, "upper": up, "p_tost": p}


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-), zeros dropped, average ranks."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return None
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return _f((wp - wm) / (wp + wm))


def bootstrap_ci(d, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Percentile bootstrap CI of the mean over datasets; a fresh generator per call (reproducible)."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    return _f(lo), _f(hi)


def effect(d):
    d = np.asarray(d, dtype=float)
    lo, hi = bootstrap_ci(d)
    return {"n_datasets": int(d.size), "mean": _f(d.mean()) if d.size else None,
            "median": _f(np.median(d)) if d.size else None, "ci95_mean": [lo, hi],
            "rank_biserial": rank_biserial(d), "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}}


def holm(pvalues, alpha=ALPHA):
    """Holm over the non-missing p-values (statsmodels multipletests). Returns (adjusted, reject) aligned
    with the input; None where the input is None."""
    from statsmodels.stats.multitest import multipletests
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    adj, rej = [None] * len(pvalues), [None] * len(pvalues)
    if idx:
        r, pa, _, _ = multipletests([pvalues[i] for i in idx], alpha=alpha, method="holm")
        for k, i in enumerate(idx):
            adj[i], rej[i] = _f(pa[k]), bool(r[k])
    return adj, rej


def _fit_mixed(formula, data, response):
    """ML fit with optimizer order lbfgs, then powell if lbfgs does not converge."""
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    tried = []
    res = None
    for method in ("lbfgs", "powell"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(f"{response} ~ {formula}", data, groups=data["dataset"], re_formula="1",
                                vc_formula={"partition": "0 + C(partition)"})
            res = model.fit(reml=False, method=method)
        n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        tried.append({"method": method, "converged": bool(res.converged), "convergence_warnings": n_conv,
                      "llf": _f(res.llf)})
        if res.converged:
            break
    return res, tried


def mixed_lrt(data, response):
    """H4: C(searcher) * C(control) + log_budget + log_pn vs the model without the interaction; LRT with
    df = difference in fixed-effect count. Random intercept for dataset, variance component for outer
    partition within dataset. A covariate that is constant in the data is dropped (recorded); a fit that
    fails is recorded as fit_error with p = None."""
    covariates = [c for c in ("log_budget", "log_pn") if data[c].nunique() > 1]
    dropped = [c for c in ("log_budget", "log_pn") if c not in covariates]  # constant: collinear with intercept
    tail = "".join(f" + {c}" for c in covariates)
    full_f = "C(searcher) * C(control)" + tail
    red_f = "C(searcher) + C(control)" + tail
    data = data.sort_values(["dataset", "partition", "searcher", "control", "checkpoint", "seed_index"],
                            kind="stable").reset_index(drop=True)
    base = {"response": response, "formula_full": full_f, "formula_reduced": red_f,
            "covariates_dropped_as_constant": dropped, "n_rows": int(len(data))}
    try:
        full, full_tried = _fit_mixed(full_f, data, response)
        red, red_tried = _fit_mixed(red_f, data, response)
    except (np.linalg.LinAlgError, ValueError) as e:
        return {**base, "fit_error": f"{type(e).__name__}: {e}", "p": None}
    df = int(len(full.fe_params) - len(red.fe_params))
    lr = 2.0 * (full.llf - red.llf)
    p = stats.chi2.sf(lr, df) if df > 0 else np.nan
    inter = {k: _f(v) for k, v in full.fe_params.items() if ":" in k}
    return {**base,
            "n_datasets": int(data["dataset"].nunique()), "n_partitions": int(data[["dataset", "partition"]]
                                                                            .drop_duplicates().shape[0]),
            "fit_full": full_tried, "fit_reduced": red_tried, "llf_full": _f(full.llf), "llf_reduced": _f(red.llf),
            "lr_statistic": _f(lr), "df": df, "p": _f(p), "interaction_coefficients": inter,
            "fe_params_full": {k: _f(v) for k, v in full.fe_params.items()}}


def nogueira_stability(subsets, d):
    """Nogueira, Sechidis & Brown (2018) stability of M subsets of a d-feature set:
    1 - mean_f(s_f^2) / ((k/d)(1 - k/d)), s_f^2 = M/(M-1) p_f (1 - p_f), k = mean subset size.
    None when undefined (M < 2, or k = 0 or k = d)."""
    M = len(subsets)
    if M < 2 or d <= 0:
        return None
    counts = {}
    for s in subsets:
        for f in set(s):
            counts[f] = counts.get(f, 0) + 1
    kbar = sum(len(set(s)) for s in subsets) / M
    if kbar <= 0 or kbar >= d:
        return None
    pf = np.array(list(counts.values()), dtype=float) / M
    s2_sum = (M / (M - 1) * pf * (1 - pf)).sum()  # features never selected contribute 0
    return _f(1 - (s2_sum / d) / ((kbar / d) * (1 - kbar / d)))
