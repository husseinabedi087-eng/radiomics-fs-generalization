"""GWO and WOA: the coefficient a decreases linearly from 2.0 to 0.0 (PREREG_STAGE1 section 5)."""

import numpy as np
import pytest

from fsbench import Evaluator
from fsbench.searchers import gwo, woa

N_UPDATES = 99  # 100 iterations including the initial one


@pytest.mark.parametrize("module", [gwo, woa], ids=["gwo", "woa"])
def test_a_schedule_linear(module):
    a = module.a_schedule(N_UPDATES)
    assert len(a) == N_UPDATES
    assert a[0] == pytest.approx(2.0, abs=1e-12)
    assert a[-1] == pytest.approx(0.0, abs=1e-12)
    assert a[(N_UPDATES - 1) // 2] == pytest.approx(1.0, abs=1e-12)  # midpoint (index 49 of 0..98)
    assert np.allclose(np.diff(a), -2.0 / (N_UPDATES - 1))


@pytest.mark.parametrize("module", [gwo, woa], ids=["gwo", "woa"])
@pytest.mark.parametrize("transfer", ["S", "V"])
def test_run_uses_the_schedule(module, transfer):
    K = 30
    ev = Evaluator(lambda m: float(m.sum()), K, budget=2000)
    res = module.run(ev, K, np.random.default_rng(0), transfer=transfer)
    used = np.array(res.trace["a"])
    assert np.allclose(used, module.a_schedule(N_UPDATES))
