"""OneMax, K = 200 (PREREG_STAGE1 section 9: "every searcher's best fitness increases with budget").

Operationalized per searcher over 5 seeds (DECISIONS.md, 2026-11-10, part 5 decisions):
  (a) for every seed, best fitness at 2,000 evaluations >= best at 250 (a decrease is a bug);
  (b) either the mean best at 2,000 > the mean best at 250, or every seed reached the optimum
      (fitness 1.0) by 250 evaluations.
All per-seed values are reported and saved as raw results.
"""

import numpy as np
import pytest

from fsbench import Evaluator
from fsbench.searchers import CONFIGS, run_config

K = 200
SEEDS = list(range(5))
OPTIMUM = 1.0


def onemax(mask):
    return float(mask.sum()) / K


@pytest.mark.parametrize("name", list(CONFIGS))
def test_onemax_improves_with_budget(name, report, save_result):
    per_seed = []
    for seed in SEEDS:
        ev = Evaluator(onemax, K, budget=2000)
        res = run_config(name, ev, K, np.random.default_rng(seed))
        per_seed.append({"seed": seed, "best_250": res.checkpoints[250].best_fitness,
                         "best_2000": res.checkpoints[2000].best_fitness})
    b250 = np.array([r["best_250"] for r in per_seed])
    b2000 = np.array([r["best_2000"] for r in per_seed])
    crit_a = bool(np.all(b2000 >= b250))
    all_optimal_at_250 = bool(np.all(b250 >= OPTIMUM))
    crit_b = bool(b2000.mean() > b250.mean() or all_optimal_at_250)
    save_result(f"onemax_{name}", {"searcher": name, "K": K, "per_seed": per_seed,
                                   "mean_250": float(b250.mean()), "mean_2000": float(b2000.mean()),
                                   "all_optimal_at_250": all_optimal_at_250,
                                   "criterion_a": crit_a, "criterion_b": crit_b, "passed": crit_a and crit_b})
    report(f"onemax {name:<13} " + "  ".join(f"s{r['seed']}: {r['best_250']:.3f}->{r['best_2000']:.3f}" for r in per_seed)
           + f"  | mean {b250.mean():.3f}->{b2000.mean():.3f}  a={'ok' if crit_a else 'FAIL'} b={'ok' if crit_b else 'FAIL'}")
    assert crit_a, f"{name}: best fitness decreased between 250 and 2000 evaluations (bug)"
    assert crit_b, f"{name}: mean best fitness did not increase and the optimum was not reached by 250"
