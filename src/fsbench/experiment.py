"""One unit of work on one outer fold (PREREG_STAGE1 sections 3, 4, 6, 7).

prepare_fold() applies the section 3 pipeline (fitted on the outer-training part only) and converts
the stored partitions to positions. run_search() and run_baseline() make every search and selection
decision from the outer-training part only; the outer-test labels are read in exactly one place,
outer_evaluation(), which is called after the search or selection has finished and whose result is
only logged. tests/test_experiment.py checks that permuting the outer-test labels changes nothing
except the logged outer AUC and Brier score.

Standard inner AUC: for every reported subset (each checkpoint and the returned subset), the mean AUC
over the fixed 3-fold inner split of the whole outer-training part (section 4 inner fitness),
computed outside the budget, so that the optimism gap (standard inner AUC - outer AUC) is comparable
across controls.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import baselines, controls, pipeline
from .evaluator import CHECKPOINTS, Evaluator
from .fitness import WarningCounter, make_c2_fitness, make_split_fitness, outer_evaluation
from .partitions import to_positions
from .searchers import run_config

CONTROLS = ("C0", "C1", "C2", "C3")
CONTROL_PARAM = {"C1": "P", "C2": "lambda", "C3": "r"}


@dataclass
class FoldData:
    dataset: str
    repeat: int
    fold: int
    X_train: np.ndarray       # outer-training part, K pool (section 3)
    y_train: np.ndarray
    X_test: np.ndarray        # outer-test part, K pool
    y_test: np.ndarray        # read only by outer_evaluation()
    pool_names: list
    inner: list               # [(train_pos, test_pos)] within the outer-training part
    c1_fit: np.ndarray        # positions within the outer-training part
    c1_selection: np.ndarray
    c1_inner: list            # [(train_pos, test_pos)] within the C1 fit part
    c2: list                  # [(train_pos, test_pos)] within the outer-training part
    n_features_raw: int
    pipeline_seconds: float

    @property
    def K(self):
        return self.X_train.shape[1]


def prepare_fold(dataset, X, y, fold):
    """X: features (DataFrame), y: 0/1 array, fold: one fold dict from fsbench.partitions."""
    t0 = time.perf_counter()
    y = np.asarray(y)
    tr, te = np.asarray(fold["outer_train"]), np.asarray(fold["outer_test"])
    Xtr, Xte, names, _ = pipeline.fit_transform(X.iloc[tr], y[tr], X.iloc[te])
    pos = lambda idx: to_positions(idx, tr)
    c1_fit = pos(fold["c1"]["fit"])
    return FoldData(
        dataset=dataset, repeat=fold["repeat"], fold=fold["fold"], X_train=Xtr, y_train=y[tr], X_test=Xte,
        y_test=y[te], pool_names=names,
        inner=[(pos(s["train"]), pos(s["test"])) for s in fold["inner"]],
        c1_fit=c1_fit, c1_selection=pos(fold["c1"]["selection"]),
        c1_inner=[(to_positions(pos(s["train"]), c1_fit), to_positions(pos(s["test"]), c1_fit))
                  for s in fold["c1"]["inner"]],
        c2=[(pos(s["train"]), pos(s["test"])) for s in fold["c2"]],
        n_features_raw=X.shape[1], pipeline_seconds=time.perf_counter() - t0)


def _control_rng(search_seed):
    return np.random.default_rng(np.random.SeedSequence([int(search_seed), 3]))


def _subset_record(fd, mask, standard, counter, timer, value=None):
    """Everything logged for one reported subset. The outer evaluation happens here, last.
    timer accumulates the wall time of the standard inner AUC and of the outer scoring separately."""
    mask = np.asarray(mask, dtype=bool)
    t0 = time.perf_counter()
    std_auc = standard(mask)
    t1 = time.perf_counter()
    auc, brier = outer_evaluation(fd.X_train, fd.y_train, fd.X_test, fd.y_test, mask, counter)
    t2 = time.perf_counter()
    timer["standard_inner_auc_s"] += t1 - t0
    timer["outer_scoring_s"] += t2 - t1
    rec = {"size": int(mask.sum()), "subset": [fd.pool_names[i] for i in np.flatnonzero(mask)],
           "pool_indices": [int(i) for i in np.flatnonzero(mask)],
           "standard_inner_auc": std_auc, "outer_auc": auc, "outer_brier": brier}
    if value is not None:
        rec["best_inner_fitness"] = float(value)
    return rec


def run_search(fd, searcher, control="C0", param=None, search_seed=0, budget=2000):
    if control not in CONTROLS:
        raise ValueError(f"control must be one of {CONTROLS}")
    counter = WarningCounter()
    K = fd.K
    std_fitness = make_split_fitness(fd.X_train, fd.y_train, fd.inner, counter)
    std_cache = {}

    def standard(mask):
        key = np.packbits(mask).tobytes()
        if key not in std_cache:
            std_cache[key] = 0.5 if not mask.any() else std_fitness(mask)
        return std_cache[key]

    if control == "C0":
        ev = Evaluator(std_fitness, K, budget=budget)
    elif control == "C1":
        Xf, yf = fd.X_train[fd.c1_fit], fd.y_train[fd.c1_fit]
        ev = controls.C1Evaluator(make_split_fitness(Xf, yf, fd.c1_inner, counter), K, Xf, yf,
                                  fd.X_train[fd.c1_selection], fd.y_train[fd.c1_selection],
                                  patience=param, counter=counter, budget=budget)
    elif control == "C2":
        ev = Evaluator(make_c2_fitness(fd.X_train, fd.y_train, fd.c2, float(param), counter), K, budget=budget)
    else:
        ev = controls.C3Evaluator(std_fitness, K, r=param, rng=_control_rng(search_seed), budget=budget)

    t0 = time.perf_counter()
    res = run_config(searcher, ev, K, np.random.default_rng(int(search_seed)))
    search_seconds = time.perf_counter() - t0

    # ---- search finished; from here on nothing feeds back into the search ----
    timer = {"standard_inner_auc_s": 0.0, "outer_scoring_s": 0.0}
    checkpoints = {}
    for c in CHECKPOINTS:
        cp = res.checkpoints.get(c)
        if cp is not None:
            rec = _subset_record(fd, cp.best_mask, standard, counter, timer, cp.best_fitness)
            rec["filled_after_stop"] = False
        else:  # run ended before c (C1 early stop): the returned subset stays the answer
            rec = _subset_record(fd, res.mask, standard, counter, timer, res.fitness)
            rec["filled_after_stop"] = True
        checkpoints[str(c)] = rec
    returned = _subset_record(fd, res.mask, standard, counter, timer, res.fitness)

    out = {"kind": "search", "dataset": fd.dataset, "repeat": fd.repeat, "fold": fd.fold,
           "searcher": searcher, "control": control,
           "control_param": None if control == "C0" else {CONTROL_PARAM[control]: param},
           "search_seed": int(search_seed), "model_seed": None, "budget": budget, "K": K,
           "n_features_raw": fd.n_features_raw, "n_train": int(len(fd.y_train)), "n_test": int(len(fd.y_test)),
           "n_evals": int(ev.n_evals), "checkpoints": checkpoints, "returned": returned,
           "archive": {"fitness": ev.archive_fitness().tolist(), "size": ev.archive_sizes().tolist()},
           "convergence_warnings": counter.convergence,
           "timings": {"fold_preparation_s": fd.pipeline_seconds, "search_s": search_seconds, **timer},
           "status": "ok"}
    if control == "C1":
        out["c1"] = {"selection_log": [list(x) for x in ev.selection_log], "stopped_early": ev.stopped_early,
                     "best_selection_auc": None if ev.sel_best_mask is None else float(ev.sel_best_auc)}
    return out


def run_baseline(fd, name, model_seed=None):
    counter = WarningCounter()
    std_fitness = make_split_fitness(fd.X_train, fd.y_train, fd.inner, counter)
    t0 = time.perf_counter()
    sel = baselines.run(name, fd.X_train, fd.y_train, fd.inner, counter, model_seed=model_seed)
    select_seconds = time.perf_counter() - t0
    # ---- selection finished ----
    timer = {"standard_inner_auc_s": 0.0, "outer_scoring_s": 0.0}
    mask = sel["mask"]
    returned = _subset_record(fd, mask, lambda m: 0.5 if not m.any() else std_fitness(m), counter, timer)
    return {"kind": "baseline", "dataset": fd.dataset, "repeat": fd.repeat, "fold": fd.fold, "selector": name,
            "search_seed": None, "model_seed": None if model_seed is None else int(model_seed), "K": fd.K,
            "n_features_raw": fd.n_features_raw, "n_train": int(len(fd.y_train)), "n_test": int(len(fd.y_test)),
            "hyperparameter": sel["hyperparameter"], "grid": sel["grid"], "grid_inner_scores": sel["inner_scores"],
            "flags": sel["flags"], "returned": returned, "convergence_warnings": counter.convergence,
            "timings": {"fold_preparation_s": fd.pipeline_seconds, "selection_s": select_seconds, **timer},
            "status": "ok"}
