"""Analysis statistics on synthetic per-dataset values and synthetic records (no real data)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from analysis import analyses as an
from analysis import stats as st

SEED = 20261113


def records(n_datasets, c4_effect, seed, folds=(0, 1), searchers=an.CTRL_SEARCHERS):
    """Synthetic analysis records: C0 and C4 for the controlled searchers, outer AUC = dataset level +
    noise (+ c4_effect for C4)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_datasets):
        base = 0.6 + 0.2 * rng.random()
        for f in folds:
            for s in searchers:
                for ctrl in ("C0", "C4"):
                    cps = {}
                    for c in (250, 500, 1000, 2000):
                        auc = base + (c4_effect if ctrl == "C4" else 0.0) + rng.normal(0, 0.02)
                        cps[c] = {"auc": auc, "std_auc": auc + 0.05, "brier": 0.2, "size": 5, "subset": ["a"]}
                    out.append({"plan": "t", "dataset": f"D{i:02d}", "config": f"{s}__{ctrl}", "family": s,
                                "control": ctrl, "control_value": None, "kind": "search", "repeat": 0, "fold": f,
                                "seed_index": 0, "K": 50, "n_train": 80, "n_test": 20, "p": 50, "checkpoints": cps})
    return out


def test_h2_planted_effect_detected_and_null_not():
    pos = an.confirmatory(an.run_table(records(15, 0.03, SEED)))
    null = an.confirmatory(an.run_table(records(15, 0.0, SEED)))
    assert pos["H2"]["test"]["p"] < 0.01 and pos["H2"]["effect"]["mean"] > 0.02
    assert null["H2"]["test"]["p"] > 0.05
    assert pos["H2"]["searchers_used"] == an.CTRL_SEARCHERS
    # the other hypotheses run on what is present and list what is missing
    assert pos["H3"]["skipped"] and "baseline__l1" in pos["H3"]["configurations_missing"]
    assert "random__C0" in pos["H1"]["configurations_missing"]


def test_tost_equivalence():
    rng = np.random.default_rng(SEED)
    near0 = rng.normal(0, 0.005, 20)
    far = 0.05 + rng.normal(0, 0.005, 20)
    assert an.h3_test(near0)["p_raw"] < 0.05
    assert an.h3_test(far)["p_raw"] > 0.05
    t = st.tost(near0, 0.02)
    assert t["p_tost"] == max(t["lower"]["p"], t["upper"]["p"])
    assert t["lower"]["p"] == sps.wilcoxon(near0 + 0.02, alternative="greater").pvalue


def test_holm_matches_hand_calculation():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    # sorted 0.005, 0.01, 0.03, 0.04, 0.2 -> x5, x4, x3, x2, x1 = 0.025, 0.04, 0.09, 0.08, 0.2; running max
    adj, rej = st.holm(p)
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.025, 0.2])
    assert rej == [True, False, False, True, False]
    adj2, rej2 = st.holm([0.01, None, 0.04])
    assert adj2[1] is None and np.allclose([adj2[0], adj2[2]], [0.02, 0.04])


def test_h5_direction_and_shift():
    rng = np.random.default_rng(SEED)
    worse = -0.02 + rng.normal(0, 0.005, 20)       # metaphor clearly worse: supported
    slightly_better = 0.005 + rng.normal(0, 0.002, 20)  # better, but by less than 0.01: supported (shift)
    much_better = 0.03 + rng.normal(0, 0.005, 20)   # better by more than 0.01: not supported
    assert an.h5_test(worse)["p_raw"] < 0.05
    sb = an.h5_test(slightly_better)
    assert sb["p_raw"] < 0.05 and sb["effect"]["mean"] > 0
    assert sb["test"]["p"] == sps.wilcoxon(slightly_better - 0.01, alternative="less").pvalue
    assert an.h5_test(much_better)["p_raw"] > 0.05
    assert an._wording("H5", {**an.h5_test(worse), "reject_holm": True}).startswith("not better")


def _h4_frame(interaction, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(10):
        ds_eff = rng.normal(0, 0.05)
        for part in range(3):
            p_eff = rng.normal(0, 0.01)
            for s in an.CTRL_SEARCHERS:
                for c in an.CONTROLS:
                    for cp in (250, 500, 1000, 2000):
                        y = 0.7 + ds_eff + p_eff + 0.005 * np.log(cp) + rng.normal(0, 0.01)
                        if interaction and s == "ga" and c == "C2":
                            y += 0.05
                        rows.append({"dataset": f"D{i}", "partition": f"0_{part}", "searcher": s, "control": c,
                                     "checkpoint": cp, "seed_index": 0, "log_budget": np.log(cp),
                                     "log_pn": np.log(0.5 + i / 10), "auc": y})
    return pd.DataFrame(rows)


def test_mixed_model_detects_planted_interaction_only():
    hit = st.mixed_lrt(_h4_frame(True, SEED), "auc")
    miss = st.mixed_lrt(_h4_frame(False, SEED), "auc")
    assert hit["df"] == 12 and hit["p"] < 1e-6
    assert miss["p"] > 0.05
    assert hit["fit_full"][0]["method"] == "lbfgs"
    assert any("ga" in k and "C2" in k for k in hit["interaction_coefficients"])


def test_bootstrap_reproducible():
    d = np.random.default_rng(1).normal(0, 1, 25)
    assert st.bootstrap_ci(d) == st.bootstrap_ci(d)
    assert st.bootstrap_ci(d) != st.bootstrap_ci(d, seed=1)
    assert st.effect(d) == st.effect(d)


@pytest.mark.parametrize("d,expected", [
    (np.array([0.1, -0.2, 0.3, 0.45, -0.05, 0.6, 0.7, 0.81, 0.9, 1.0, 1.1, -1.2, 1.3, 1.4]), "exact"),
    (np.array([0.1, 0.1, -0.2, 0.3, 0.4, 0.5]), "permutation"),       # ties, n <= 13
    (np.array([0.0, 0.1, -0.2, 0.3, 0.4, 0.5, 0.6]), "permutation"),  # zero, n <= 13
    (np.r_[np.arange(1, 20) / 10, 0.1], "asymptotic"),                 # ties, n > 13
    (np.random.default_rng(3).normal(0, 1, 60), "asymptotic"),         # n > 50
])
def test_wilcoxon_method_recorded_matches_scipy(d, expected):
    res = st.wilcoxon(d, "two-sided")
    assert res["method"] == expected
    named = {"exact": "exact", "asymptotic": "asymptotic", "permutation": sps.PermutationMethod()}[expected]
    assert res["p"] == sps.wilcoxon(d, zero_method="wilcox", method=named).pvalue
    assert res["n_after_zero_removal"] == int(np.count_nonzero(d))


def test_rank_biserial_and_stability():
    assert st.rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert st.rank_biserial(np.array([1.0, -1.0])) == 0.0
    same = [["a", "b"], ["a", "b"], ["a", "b"]]
    assert st.nogueira_stability(same, 10) == pytest.approx(1.0)
    assert st.nogueira_stability([["a"], ["a"]], 1) is None  # k = d: undefined
    disjoint = [["a"], ["b"]]
    assert st.nogueira_stability(disjoint, 2) is None or st.nogueira_stability(disjoint, 2) < 0


def test_control_values_required_when_a_control_has_several():
    from analysis.load import Refused
    df = pd.DataFrame({"control": ["C1", "C1", "C0"], "control_value": [3.0, 5.0, None],
                       "config": ["ga__C1", "ga__C1", "ga__C0"]})
    with pytest.raises(Refused):
        an.select_control_values(df, {})
    kept, excluded = an.select_control_values(df, {"C1": 5})
    assert list(kept["control_value"].fillna(-1)) == [5.0, -1] and excluded == ["ga__C1 (C1 = 3)"]
    with pytest.raises(Refused):
        an.select_control_values(df, {"C1": 10})
