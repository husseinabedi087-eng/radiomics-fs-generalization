"""Structural bias: dispersion of returned subset sizes (diagnostic, reported, not a gate).

Same pure-noise setup as tests/test_bias.py (fitness uniform in [0, 1] from a BLAKE2b hash of the mask
bytes and the seed; K = 200; budget 2,000; 30 seeds; original encoding).

(a) For every searcher: SD, min, median and max of the returned subset size, and a Brown-Forsythe
    test (scipy.stats.levene, center="median") of equal dispersion against random_search, whose
    sizes are uniform on 1..200 by construction (SD about 57.7).
(b) WOA only (woa_S, woa_V), per update iteration, averaged over seeds:
      - fraction of position coordinates at the bounds (+-6), for the current population (before
        the move) and for the proposed positions;
      - same-bound share: per whale, the fraction of its saturated coordinates that sit on the
        majority bound (0.5 = mixed, 1 = all at +6 or all at -6);
      - sign coherence of the proposed move x_new - x_current: the fraction of coordinates with a
        non-zero move that move in the majority direction (0.5 = no coherence, 1 = all one way);
      - fraction of coordinates with a zero move (typically clipped at a bound).
    The moves are recorded by wrapping the objective that woa.run passes to woa.core; the searcher
    code is not modified. The test checks that the wrapped run returns the same mask as woa.run.

Raw results: results/verification/dispersion_sizes.json, dispersion_woa_S.json, dispersion_woa_V.json.
"""

import warnings

import numpy as np
from scipy.stats import levene

from fsbench import BudgetExhausted, Evaluator
from fsbench.searchers import CONFIGS, run_config, woa
from fsbench.searchers._base import make_binary_objective
from test_bias import BUDGET, K, SEEDS, noise_fitness

REFERENCE = "random"


def returned_sizes(name):
    out = []
    for s in SEEDS:
        ev = Evaluator(noise_fitness(s, False), K, budget=BUDGET)
        out.append(int(run_config(name, ev, K, np.random.default_rng(s)).mask.sum()))
    return out


def test_dispersion_of_returned_sizes(report, save_result):
    sizes = {name: returned_sizes(name) for name in CONFIGS}
    ref = np.array(sizes[REFERENCE])
    rows = {}
    for name, s in sizes.items():
        s = np.array(s)
        p = None if name == REFERENCE else float(levene(s, ref, center="median").pvalue)
        rows[name] = {"sizes": s.tolist(), "sd": float(s.std(ddof=1)), "min": int(s.min()),
                      "median": float(np.median(s)), "max": int(s.max()),
                      "n_le_20": int((s <= 20).sum()), "n_ge_180": int((s >= K - 20).sum()),
                      "p_brown_forsythe_vs_random": p}
        report(f"dispersion {name:<13} SD {rows[name]['sd']:5.1f}  min {s.min():3d}  median {np.median(s):6.1f}  "
               f"max {s.max():3d}  <=20: {rows[name]['n_le_20']:2d}  >=180: {rows[name]['n_ge_180']:2d}  "
               f"Brown-Forsythe vs random p = {'-' if p is None else f'{p:.4g}'}")
    save_result("dispersion_sizes", {"K": K, "budget": BUDGET, "seeds": SEEDS, "reference": REFERENCE,
                                     "searchers": rows})
    assert len(rows) == len(CONFIGS)  # diagnostic only


def woa_trace(transfer, seed):
    """Run woa.core exactly as woa.run does, recording every proposed move."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(noise_fitness(seed, False), K, budget=BUDGET)
    inner = make_binary_objective(ev, K, transfer, rng)
    moves = []  # (current position, proposed position) for every update move

    def objective(position, prev):
        if prev is not None:
            moves.append((prev[1].copy(), np.array(position, copy=True)))
        return inner(position, prev)

    try:
        woa.core(objective, K, -woa.BOUND, woa.BOUND, rng, trace={})
    except BudgetExhausted:
        pass
    reference = woa.run(Evaluator(noise_fitness(seed, False), K, budget=BUDGET), K,
                        np.random.default_rng(seed), transfer=transfer).mask
    assert np.array_equal(ev.best_mask, reference), "instrumented run differs from woa.run"

    pop = woa.POP_SIZE
    n_iter = len(moves) // pop
    at_bound_cur, at_bound_new, coherence, zero, bound_side = [], [], [], [], []
    for t in range(n_iter):
        cur = np.array([m[0] for m in moves[t * pop:(t + 1) * pop]])
        new = np.array([m[1] for m in moves[t * pop:(t + 1) * pop]])
        at_bound_cur.append(float(np.mean(np.abs(cur) >= woa.BOUND)))
        sides = []  # per whale: share of its saturated coordinates that sit on the majority bound
        for row in cur:
            sat = row[np.abs(row) >= woa.BOUND]
            if sat.size:
                sides.append(max(np.mean(sat > 0), np.mean(sat < 0)))
        bound_side.append(float(np.mean(sides)) if sides else float("nan"))
        at_bound_new.append(float(np.mean(np.abs(new) >= woa.BOUND)))
        d = new - cur
        coh = []
        for row in d:
            nz = row[row != 0]
            if nz.size:
                coh.append(max(np.mean(nz > 0), np.mean(nz < 0)))
        coherence.append(float(np.mean(coh)) if coh else float("nan"))
        zero.append(float(np.mean(d == 0)))
    return {"at_bound_current": at_bound_cur, "at_bound_proposed": at_bound_new,
            "bound_side_coherence": bound_side, "sign_coherence": coherence, "zero_move": zero,
            "returned_size": int(ev.best_mask.sum())}


def test_woa_bound_saturation_and_sign_coherence(report, save_result):
    for transfer in ("S", "V"):
        per_seed = [woa_trace(transfer, s) for s in SEEDS]
        keys = ["at_bound_current", "at_bound_proposed", "bound_side_coherence", "sign_coherence", "zero_move"]
        with warnings.catch_warnings():  # iteration 1 has no saturated coordinates: all-NaN column
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_curves = {k: np.nanmean(np.array([r[k] for r in per_seed]), axis=0).tolist() for k in keys}
        save_result(f"dispersion_woa_{transfer}", {
            "searcher": f"woa_{transfer}", "K": K, "budget": BUDGET, "seeds": SEEDS, "bound": woa.BOUND,
            "mean_per_iteration": mean_curves,
            "per_seed": [{"seed": s, "returned_size": r["returned_size"], **{k: r[k] for k in keys}}
                         for s, r in zip(SEEDS, per_seed)]})
        c = mean_curves
        idx = [0, len(c["sign_coherence"]) // 4, len(c["sign_coherence"]) // 2, len(c["sign_coherence"]) - 1]
        report(f"woa_{transfer} per iteration (iterations {', '.join(str(i + 1) for i in idx)}): "
               f"at bounds (current) {', '.join(f'{c['at_bound_current'][i]:.2f}' for i in idx)}; "
               f"same-bound share {', '.join(f'{c['bound_side_coherence'][i]:.2f}' for i in idx)}; "
               f"sign coherence {', '.join(f'{c['sign_coherence'][i]:.2f}' for i in idx)}; "
               f"zero move {', '.join(f'{c['zero_move'][i]:.2f}' for i in idx)}")
