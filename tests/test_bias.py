"""Structural bias (PREREG_STAGE1 section 9, mirrored encoding), following the structural-bias test of
Kononova et al. (2015), "Structural bias in population-based algorithms", Information Sciences 298,
468-490, adapted to binary masks.

The fitness is pure noise: deterministic per mask, uniform in [0, 1] from a BLAKE2b hash of the mask
bytes and the seed. With no signal, an unbiased searcher has no reason to prefer any subset size.
K = 200, budget 2,000, 30 seeds. Each searcher runs on the original encoding and on the inverted
encoding (its bit b means "feature not selected": the mask scored is ~b and the returned mask is
decoded as ~b), with the same search seed and the same noise landscape in decoded space.

  - Original vs inverted returned subset sizes: two-sided Mann-Whitney U; must have p > 0.01.
  - Original returned sizes vs K/2: one-sample two-sided Wilcoxon signed-rank; p reported and
    flagged if <= 0.01 (reported, not asserted).
A failure is reported; the searcher and the threshold are not changed without the author's decision.
"""

import hashlib

import numpy as np
import pytest
from scipy.stats import mannwhitneyu, wilcoxon

from fsbench import Evaluator
from fsbench.searchers import CONFIGS, run_config

K = 200
BUDGET = 2000
SEEDS = list(range(30))
ALPHA = 0.01


def noise_fitness(seed, invert):
    salt = int(seed).to_bytes(8, "little")

    def f(mask):
        m = ~mask if invert else mask
        h = hashlib.blake2b(np.packbits(m).tobytes() + salt, digest_size=8).digest()
        return int.from_bytes(h, "little") / 2.0 ** 64
    return f


def returned_size(name, seed, invert):
    ev = Evaluator(noise_fitness(seed, invert), K, budget=BUDGET)
    res = run_config(name, ev, K, np.random.default_rng(seed))
    mask = ~res.mask if invert else res.mask
    return int(mask.sum())


@pytest.mark.parametrize("name", list(CONFIGS))  # about 70 s in total: not marked slow (limit 2 min)
def test_structural_bias(name, report, save_result):
    orig = [returned_size(name, s, False) for s in SEEDS]
    inv = [returned_size(name, s, True) for s in SEEDS]
    p_mw = 1.0 if len(set(orig) | set(inv)) == 1 else float(mannwhitneyu(orig, inv, alternative="two-sided").pvalue)
    diffs = np.array(orig) - K / 2
    p_wx = 1.0 if np.all(diffs == 0) else float(wilcoxon(diffs, alternative="two-sided").pvalue)
    save_result(f"bias_{name}", {"searcher": name, "K": K, "budget": BUDGET, "seeds": SEEDS,
                                 "sizes_original": orig, "sizes_inverted": inv,
                                 "p_mannwhitney": p_mw, "p_wilcoxon_vs_half": p_wx,
                                 "flag_mannwhitney": p_mw <= ALPHA, "flag_wilcoxon": p_wx <= ALPHA})
    line = (f"bias {name:<13} mean size orig {np.mean(orig):6.1f}  inv {np.mean(inv):6.1f}  "
            f"MW p = {p_mw:.4f}{' FLAG' if p_mw <= ALPHA else ''}  "
            f"Wilcoxon vs K/2 p = {p_wx:.4f}{' FLAG' if p_wx <= ALPHA else ''}")
    print(line)
    report(line)
    assert p_mw > ALPHA, f"{name}: returned sizes differ between encodings (p = {p_mw:.4g})"
