"""Study pipeline components on synthetic data only (no real dataset is loaded).

Covers: partitions; section 3 pipeline fitted on the outer-training part only; baselines return
subsets inside the pool; C1 and C3 budget accounting; outer-test independence (permuting the
outer-test labels changes nothing but the logged outer AUC and Brier score); determinism.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fsbench import baselines, controls, experiment, partitions, pipeline
from fsbench.evaluator import Evaluator
from fsbench.fitness import WarningCounter
from fsbench.searchers import run_config

N, P = 120, 40


def synthetic(seed=0, n=N, p=P):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:5] = [1.5, -1.2, 1.0, 0.8, -0.8]
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    df = pd.DataFrame(X, columns=[f"feat_{j:03d}" for j in range(p)])
    return df, y


@pytest.fixture(scope="module")
def fold_data(tmp_path_factory):
    X, y = synthetic()
    parts = partitions.get("SyntheticSet", y, directory=tmp_path_factory.mktemp("partitions"))
    return experiment.prepare_fold("SyntheticSet", X, y, partitions.fold(parts, 0, 0))


# ---------------- partitions ----------------

def test_partitions_structure_and_reuse(tmp_path):
    _, y = synthetic()
    p1 = partitions.get("SyntheticSet", y, directory=tmp_path)
    p2 = partitions.get("SyntheticSet", y, directory=tmp_path)  # reloaded and verified
    assert p1 == p2 and (tmp_path / "SyntheticSet.json").exists()
    for rep in p1["repeats"]:
        tests = sorted(i for f in rep for i in f["outer_test"])
        assert tests == list(range(N))  # outer test folds partition the rows
        for f in rep:
            tr = set(f["outer_train"])
            assert not tr & set(f["outer_test"])
            for s in f["inner"] + f["c2"] + f["c1"]["inner"]:
                assert set(s["train"]) | set(s["test"]) <= tr
            assert set(f["c1"]["fit"]) | set(f["c1"]["selection"]) == tr
            assert not set(f["c1"]["fit"]) & set(f["c1"]["selection"])
            assert abs(len(f["c1"]["selection"]) - 0.2 * len(tr)) <= 1
            assert len(f["c2"]) == 9 and len(f["inner"]) == 3
    assert p1["repeats"][0][0]["outer_test"] != p1["repeats"][1][0]["outer_test"]


def test_partitions_file_mismatch_raises(tmp_path):
    _, y = synthetic()
    partitions.get("SyntheticSet", y, directory=tmp_path)
    path = tmp_path / "SyntheticSet.json"
    path.write_text(path.read_text().replace('"n_inner":3', '"n_inner":4'), newline="\n")
    with pytest.raises(RuntimeError):
        partitions.get("SyntheticSet", y, directory=tmp_path)


def test_seeds_are_derived_from_names():
    assert partitions.derive_seed("A", 0) == partitions.derive_seed("A", 0)
    assert partitions.derive_seed("A", 0) != partitions.derive_seed("A", 1)
    assert 0 <= partitions.derive_seed("A", 0, 1, "inner") < 2 ** 32


# ---------------- pipeline (section 3) ----------------

def test_pipeline_fitted_on_train_only():
    X, y = synthetic()
    tr, te = np.arange(0, 90), np.arange(90, N)
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    Xtr["const_in_train"] = 1.0
    Xte["const_in_train"] = np.arange(len(te), dtype=float)  # varies only in the test part
    Xtr.iloc[0, 0] = np.nan
    Xte.iloc[0, 0] = np.nan
    t = pipeline.fit(Xtr, y[tr])
    assert "const_in_train" not in t.kept_columns
    A = np.asarray(Xtr[t.kept_columns], dtype=float)
    med = np.nanmedian(A, axis=0)
    assert np.allclose(t.medians, med)
    A = np.where(np.isnan(A), med, A)
    assert np.allclose(t.means, A.mean(axis=0)) and np.allclose(t.scales, A.std(axis=0))
    # the test row's missing value is imputed with the TRAIN median
    j = t.kept_columns.index(X.columns[0])
    out = Xte.iloc[[0]][t.kept_columns].to_numpy(dtype=float)
    expected = (med[j] - t.means[j]) / t.scales[j]
    pool_pos = list(t.pool_order).index(j) if j in t.pool_order else None
    if pool_pos is not None:
        assert np.isclose(t.transform(Xte.iloc[[0]])[0, pool_pos], expected)
    # changing the test part changes nothing that was fitted
    Xte2 = Xte * 100 + 7
    a1 = pipeline.fit_transform(Xtr, y[tr], Xte)
    a2 = pipeline.fit_transform(Xtr, y[tr], Xte2)
    assert a1[2] == a2[2] and np.array_equal(a1[0], a2[0])


def test_pipeline_pool_size_and_order():
    X, y = synthetic(p=250)
    Xtr, Xte, names, t = pipeline.fit_transform(X.iloc[:90], y[:90], X.iloc[90:])
    assert Xtr.shape[1] == Xte.shape[1] == len(names) == 200
    assert np.all(np.diff(t.f_scores) <= 0)  # decreasing F
    X2, y2 = synthetic(p=30)
    assert pipeline.fit_transform(X2.iloc[:90], y2[:90], X2.iloc[90:])[0].shape[1] == 30


# ---------------- baselines (section 6) ----------------

@pytest.mark.parametrize("name", baselines.SELECTORS)
def test_baselines_return_subset_in_pool(fold_data, name):
    seed = 11 if name in baselines.NEEDS_MODEL_SEED else None
    sel = baselines.run(name, fold_data.X_train, fold_data.y_train, fold_data.inner, WarningCounter(), model_seed=seed)
    assert sel["mask"].dtype == bool and sel["mask"].shape == (fold_data.K,)
    if name in ("ftest", "mrmr", "rf"):
        assert sel["mask"].sum() == sel["hyperparameter"]["k"] and sel["hyperparameter"]["k"] in baselines.k_grid(fold_data.K)
    if name in ("l1", "enet"):
        assert len(sel["grid"]) == 20 and sel["hyperparameter"]["C"] in sel["grid"]


def test_rf_needs_model_seed(fold_data):
    with pytest.raises(ValueError):
        baselines.run("rf", fold_data.X_train, fold_data.y_train, fold_data.inner, WarningCounter())


def test_empty_penalized_selection_is_flagged(fold_data, monkeypatch):
    monkeypatch.setattr(baselines, "_penalized_selection", lambda X, y, C, kind, counter: np.array([], dtype=int))
    res = experiment.run_baseline(fold_data, "l1")
    assert res["flags"] == ["empty_subset"] and res["returned"]["size"] == 0
    assert res["returned"]["outer_auc"] == 0.5 and res["returned"]["standard_inner_auc"] == 0.5


def test_mrmr_first_feature_has_largest_f(fold_data):
    from sklearn.feature_selection import f_classif
    F, _ = f_classif(fold_data.X_train, fold_data.y_train)
    order = baselines.mrmr_ranking(fold_data.X_train, fold_data.y_train, 10)
    assert order[0] == int(np.argmax(F)) and len(set(order)) == 10


# ---------------- controls: budget accounting ----------------

def test_c1_budget_and_early_stopping(fold_data):
    counter = WarningCounter()
    Xf, yf = fold_data.X_train[fold_data.c1_fit], fold_data.y_train[fold_data.c1_fit]
    flat = lambda m: 0.6  # inner fitness never improves -> best-so-far never changes
    ev = controls.C1Evaluator(flat, fold_data.K, Xf, yf, fold_data.X_train[fold_data.c1_selection],
                              fold_data.y_train[fold_data.c1_selection], patience=3, counter=counter, budget=2000)
    res = run_config("ga", ev, fold_data.K, np.random.default_rng(0))
    # selection checkpoints after 100, 200, 300, 400 search evaluations: first improves, then 3 without
    assert ev.stopped_early and len(ev.selection_log) == 4
    assert ev.n_evals == ev.n_search + len(ev.selection_log) == 404
    assert res.n_evals == 404 and np.array_equal(res.mask, ev.sel_best_mask)


def test_c1_full_budget_counts_selection_evaluations(fold_data):
    counter = WarningCounter()
    Xf, yf = fold_data.X_train[fold_data.c1_fit], fold_data.y_train[fold_data.c1_fit]
    calls = {"n": 0}

    def improving(m):
        calls["n"] += 1
        return calls["n"] / 1e6  # every new subset is better -> best-so-far keeps changing
    ev = controls.C1Evaluator(improving, fold_data.K, Xf, yf, fold_data.X_train[fold_data.c1_selection],
                              fold_data.y_train[fold_data.c1_selection], patience=10 ** 6, counter=counter,
                              budget=2000, cache=False)
    run_config("random", ev, fold_data.K, np.random.default_rng(0))
    assert ev.n_evals == 2000 == ev.n_search + len(ev.selection_log)
    assert len(ev.selection_log) == 19  # 1981 search evaluations -> checkpoints at 100, ..., 1900


def test_c3_each_call_costs_one_plus_r(fold_data):
    r = 4
    ev = controls.C3Evaluator(lambda m: float(m.sum()) / fold_data.K, fold_data.K, r=r,
                              rng=np.random.default_rng(5), budget=2000)
    m = np.zeros(fold_data.K, dtype=bool)
    m[:3] = True
    for k in range(1, 6):
        ev(m)
        assert ev.n_evals == k * (1 + r)
    res = run_config("hill_climb", controls.C3Evaluator(lambda m: float(m.sum()), fold_data.K, r=r,
                                                        rng=np.random.default_rng(5), budget=2000),
                     fold_data.K, np.random.default_rng(0))
    assert res.n_evals == 2000


def test_c3_neighbour_stream_is_separate_from_searcher(fold_data):
    """C3 must not consume the searcher's random numbers: the searcher's first proposals match C0."""
    seen = {"C0": [], "C3": []}
    f = lambda m: float(m[:5].sum())
    ev0 = Evaluator(lambda m: (seen["C0"].append(m.copy()), f(m))[1], fold_data.K, budget=50)
    run_config("hill_climb", ev0, fold_data.K, np.random.default_rng(9))
    ev3 = controls.C3Evaluator(f, fold_data.K, r=2, rng=np.random.default_rng(1), budget=60)
    orig = ev3.score
    ev3.score = lambda m: (seen["C3"].append(np.asarray(m, bool).copy()), orig(m))[1]
    run_config("hill_climb", ev3, fold_data.K, np.random.default_rng(9))
    assert np.array_equal(seen["C0"][0], seen["C3"][0])  # same initial mask from the same search seed


# ---------------- outer-test independence and determinism ----------------

CONFIGS = [("ga", "C0", None), ("gwo_S", "C1", 3), ("de_S", "C2", 1.0), ("bpso_S", "C3", 2), ("woa_V", "C0", None)]
BUDGET = 200  # past the first C1 selection checkpoint (100 search evaluations); keeps the suite short
OUTER_ONLY = {"outer_auc", "outer_brier"}


def _strip(obj):
    """Drop timings and outer-test results, keep everything a search or selection decided."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in OUTER_ONLY and k != "timings"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


@pytest.mark.parametrize("searcher,control,param", CONFIGS)
def test_outer_test_labels_do_not_influence_search(fold_data, searcher, control, param):
    # flipping every outer-test label reverses the outer AUC; nothing else may change
    flipped = dataclasses.replace(fold_data, y_test=1 - fold_data.y_test)
    a = experiment.run_search(fold_data, searcher, control, param, search_seed=3, budget=BUDGET)
    c = experiment.run_search(flipped, searcher, control, param, search_seed=3, budget=BUDGET)
    assert _strip(a) == _strip(c)
    assert a["returned"]["outer_auc"] != c["returned"]["outer_auc"]  # the outer test labels are used for logging


def test_permuted_outer_test_labels_change_only_outer_results(fold_data):
    permuted = dataclasses.replace(fold_data, y_test=np.random.default_rng(42).permutation(fold_data.y_test))
    a = experiment.run_search(fold_data, "ga", "C0", None, search_seed=3, budget=BUDGET)
    b = experiment.run_search(permuted, "ga", "C0", None, search_seed=3, budget=BUDGET)
    assert _strip(a) == _strip(b)


@pytest.mark.parametrize("name", baselines.SELECTORS)
def test_outer_test_labels_do_not_influence_baselines(fold_data, name):
    seed = 11 if name in baselines.NEEDS_MODEL_SEED else None
    flipped = dataclasses.replace(fold_data, y_test=1 - fold_data.y_test)
    a = experiment.run_baseline(fold_data, name, model_seed=seed)
    b = experiment.run_baseline(flipped, name, model_seed=seed)
    assert _strip(a) == _strip(b)


@pytest.mark.parametrize("searcher,control,param", CONFIGS)
def test_run_is_deterministic(fold_data, searcher, control, param):
    a = experiment.run_search(fold_data, searcher, control, param, search_seed=8, budget=BUDGET)
    b = experiment.run_search(fold_data, searcher, control, param, search_seed=8, budget=BUDGET)
    a.pop("timings"), b.pop("timings")
    assert a == b


def test_baseline_rf_is_deterministic_for_a_model_seed(fold_data):
    a = experiment.run_baseline(fold_data, "rf", model_seed=5)
    b = experiment.run_baseline(fold_data, "rf", model_seed=5)
    a.pop("timings"), b.pop("timings")
    assert a == b


def test_search_result_contents(fold_data):
    r = experiment.run_search(fold_data, "ga", "C0", None, search_seed=1, budget=2000)
    assert r["n_evals"] == 2000 and sorted(r["checkpoints"], key=int) == ["250", "500", "1000", "2000"]
    assert len(r["archive"]["fitness"]) == len(r["archive"]["size"]) == 2000
    cp = r["checkpoints"]["2000"]
    assert cp["subset"] == r["returned"]["subset"] and cp["size"] == len(cp["subset"])
    assert all(name in fold_data.pool_names for name in cp["subset"])
    assert 0 <= cp["outer_brier"] <= 1 and 0 <= cp["outer_auc"] <= 1
    # C0: the best inner fitness is the standard inner AUC of the same subset
    assert np.isclose(cp["best_inner_fitness"], cp["standard_inner_auc"])


def test_c4_knee_records(fold_data):
    """C0 runs log the knee of the archive prefix at every checkpoint; controlled runs do not."""
    from fsbench.archive import knee_point
    r = experiment.run_search(fold_data, "ga", "C0", None, search_seed=1, budget=600)
    assert sorted(r["c4"], key=int) == ["250", "500"]
    fit, size = np.array(r["archive"]["fitness"]), np.array(r["archive"]["size"])
    for c, rec in r["c4"].items():
        i = knee_point(fit[:int(c)], size[:int(c)])
        assert rec["archive_index"] == i < int(c)
        assert rec["size"] == size[i] == len(rec["subset"]) and rec["best_inner_fitness"] == fit[i]
        assert np.isclose(rec["standard_inner_auc"], fit[i])  # C0 inner fitness is the standard inner AUC
    assert "c4_s" in r["timings"]
    assert "c4" not in experiment.run_search(fold_data, "ga", "C3", 2, search_seed=1, budget=BUDGET)


def test_c4_knee_choice_ignores_outer_test_labels(fold_data):
    """The knee is chosen from the archive's inner fitness and size only; the outer fold is scored afterwards."""
    flipped = dataclasses.replace(fold_data, y_test=1 - fold_data.y_test)
    a = experiment.run_search(fold_data, "ga", "C0", None, search_seed=4, budget=600)
    c = experiment.run_search(flipped, "ga", "C0", None, search_seed=4, budget=600)
    assert sorted(a["c4"], key=int) == ["250", "500"] and _strip(a["c4"]) == _strip(c["c4"])
    assert all(set(rec) >= {"standard_inner_auc", "outer_auc", "outer_brier"} for rec in a["c4"].values())
    assert any(a["c4"][k]["outer_auc"] != c["c4"][k]["outer_auc"] for k in a["c4"])
