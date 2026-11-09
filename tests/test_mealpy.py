"""Continuous cores vs MEALPY 3.0.3 (PREREG_STAGE1 section 9).

For BPSO (continuous core vs LDW_PSO), DE (vs OriginalDE, strategy 0 = rand/1/bin), GWO (vs
OriginalGWO) and WOA (vs OriginalWOA): Sphere and Rastrigin, d = 30, population 20, 100 iterations
including the initial population (MEALPY: epoch = 99 after its initial population, i.e. 2,000
evaluations each), 20 seeds. Medians of the final best values and a two-sided Mann-Whitney U
p-value are reported; a difference with p < 0.01 is flagged and fails the test.
"""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from fsbench.searchers import CONTINUOUS_CORES

mealpy = pytest.importorskip("mealpy")
from mealpy import DE, GWO, PSO, WOA, FloatVar  # noqa: E402

DIM, POP, EPOCH = 30, 20, 99
SEEDS = range(20)
ALPHA = 0.01


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


FUNCTIONS = {"sphere": (sphere, -100.0, 100.0), "rastrigin": (rastrigin, -5.12, 5.12)}

MEALPY_MODELS = {
    "bpso": lambda: PSO.LDW_PSO(epoch=EPOCH, pop_size=POP, c1=2.0, c2=2.0, w_min=0.4, w_max=0.9),
    "de": lambda: DE.OriginalDE(epoch=EPOCH, pop_size=POP, wf=0.5, cr=0.9, strategy=0),
    "gwo": lambda: GWO.OriginalGWO(epoch=EPOCH, pop_size=POP),
    "woa": lambda: WOA.OriginalWOA(epoch=EPOCH, pop_size=POP),
}


def run_mealpy(algo, f, lb, ub, seed):
    problem = {"obj_func": f, "bounds": FloatVar(lb=(lb,) * DIM, ub=(ub,) * DIM), "minmax": "min", "log_to": None}
    return float(MEALPY_MODELS[algo]().solve(problem, seed=seed).target.fitness)


def run_ours(algo, f, lb, ub, seed):
    return CONTINUOUS_CORES[algo](f, DIM, lb, ub, np.random.default_rng(seed), pop_size=POP, n_iter=EPOCH + 1)


@pytest.mark.parametrize("fname", list(FUNCTIONS))
@pytest.mark.parametrize("algo", list(MEALPY_MODELS))
def test_core_matches_mealpy(algo, fname, report, save_result):
    f, lb, ub = FUNCTIONS[fname]
    ours = [run_ours(algo, f, lb, ub, s) for s in SEEDS]
    ref = [run_mealpy(algo, f, lb, ub, s) for s in SEEDS]
    p = float(mannwhitneyu(ours, ref, alternative="two-sided").pvalue)
    save_result(f"mealpy_{algo}_{fname}", {
        "algorithm": algo, "function": fname, "dim": DIM, "pop_size": POP,
        "iterations_including_initial": EPOCH + 1, "mealpy_model": type(MEALPY_MODELS[algo]()).__name__,
        "bounds": [lb, ub], "seeds": list(SEEDS), "final_best_ours": ours, "final_best_mealpy": ref,
        "p_mannwhitney": p, "flag": p < ALPHA})
    report(f"mealpy {algo:<5} {fname:<10} median ours {np.median(ours):.4g}  mealpy {np.median(ref):.4g}  "
           f"p = {p:.4f}{'  FLAG' if p < ALPHA else ''}")
    assert p >= ALPHA, f"{algo}/{fname}: our core differs from MEALPY (p = {p:.4g})"
