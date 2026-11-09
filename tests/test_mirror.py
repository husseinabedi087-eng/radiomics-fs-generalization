"""Mirrored encoding (PREREG_STAGE1 section 9): structural bias check.

Fitness symmetric under bit inversion: f(mask) = 1 - |sum(mask) - K/2| / K + noise, with noise
from a seeded generator. Each searcher runs twice per seed with the same search seed: once on the
original encoding, once on the inverted encoding (the searcher's bit b means "feature NOT selected":
it is scored as f(~b) and its returned mask is decoded as ~b). Over 20 seeds, the returned subset
sizes of the two encodings must not differ (two-sided Mann-Whitney U, p > 0.01). The threshold is
fixed; failures are reported, not loosened.
"""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from fsbench import Evaluator
from fsbench.searchers import CONFIGS, run_config

K = 50
SEEDS = range(20)
NOISE_SD = 0.01
ALPHA = 0.01


def symmetric_fitness(noise_rng, invert):
    def f(mask):
        m = ~mask if invert else mask
        return 1.0 - abs(int(m.sum()) - K / 2) / K + noise_rng.normal(0.0, NOISE_SD)
    return f


def returned_size(name, seed, invert):
    ev = Evaluator(symmetric_fitness(np.random.default_rng(10_000 + seed), invert), K, budget=2000, cache=False)
    res = run_config(name, ev, K, np.random.default_rng(seed))
    mask = ~res.mask if invert else res.mask
    return int(mask.sum())


@pytest.mark.parametrize("name", list(CONFIGS))
def test_mirror_symmetry(name, report):
    orig = [returned_size(name, s, False) for s in SEEDS]
    inv = [returned_size(name, s, True) for s in SEEDS]
    if len(set(orig) | set(inv)) == 1:
        p = 1.0  # identical constant samples: no difference (the U test is undefined here)
    else:
        p = float(mannwhitneyu(orig, inv, alternative="two-sided").pvalue)
    report(f"mirror {name:<13} median size original {np.median(orig):5.1f}  inverted {np.median(inv):5.1f}  "
           f"p = {p:.4f}{'  FLAG' if p <= ALPHA else ''}")
    assert p > ALPHA, f"{name}: returned sizes differ between encodings (p = {p:.4g})"
