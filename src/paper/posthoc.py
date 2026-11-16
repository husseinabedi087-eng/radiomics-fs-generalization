"""Post hoc analyses requested by a pre-submission review (not pre-registered).

    python src/paper/posthoc.py

Reads only committed outputs:
  - results/analysis/confirmation/results.json and tables/per_dataset_{auc,gap,brier}_2000.csv
    (per-dataset means at 2,000 evaluations over outer folds and seeds, as in the frozen analysis);
  - results/paper/sources.json (log(p/n) per dataset, computed from the run files as the analysis does);
  - docs/VERIFICATION.md (section H).
No run file is opened and nothing under src/analysis/, src/fsbench/ or any committed output is changed.

Analyses A-H as requested; natural logarithm throughout. Bootstrap: 10,000 resamples over datasets,
a fresh numpy default_rng(20261117) for each quantity (so results do not depend on the order of
computation), percentile intervals. Wilcoxon signed-rank tests: scipy.stats.wilcoxon with
zero_method="wilcox", method="auto". Hodges-Lehmann estimate: median of the Walsh averages; its 90%
confidence interval uses the exact null distribution of the signed-rank statistic. Mixed model (G):
statsmodels MixedLM fitted by maximum likelihood, lbfgs then powell if lbfgs does not converge.

Outputs in results/paper/posthoc/ (header "POST HOC (not pre-registered; confirmation data)"):
posthoc.json (full precision), A-H as CSV, posthoc.md (all tables, rounded), posthoc_manifest.csv.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
CONF = REPO / "results/analysis/confirmation"
OUT = REPO / "results/paper/posthoc"
HEADER = "POST HOC (not pre-registered; confirmation data)"
SEED = 20261117
N_BOOT = 10_000
C0 = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V", "gwo_S", "gwo_V",
      "woa_S", "woa_V"]
CTRL = ["ga", "bpso_S", "de_S", "gwo_S"]
BASELINES = ["all_k", "l1", "enet", "ftest", "mrmr", "rf"]
METAPHOR = ["gwo_S", "gwo_V", "woa_S", "woa_V"]
CLASSICAL = ["ga", "bpso_S", "bpso_V", "de_S", "de_V"]


def f(x):
    x = float(x)
    return x if math.isfinite(x) else None


def read_table(name):
    lines = (CONF / "tables" / name).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# ")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
    return {r["dataset"]: {k: float(v) for k, v in r.items() if k != "dataset"} for r in rows}


# ---------------- statistics ----------------

def boot_ci(values, level=0.95, stat=np.mean):
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return [None, None]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, v.size, size=(N_BOOT, v.size))
    s = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.percentile(s, [100 * a, 100 * (1 - a)])
    return [f(lo), f(hi)]


def summary(values, level=0.95):
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "mean": f(v.mean()) if v.size else None, "median": f(np.median(v)) if v.size else None,
            f"ci{int(level * 100)}_mean": boot_ci(v, level)}


def spearman_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    rs = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    rs = rs[np.isfinite(rs)]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return {"rho": f(rho), "ci95": [f(lo), f(hi)], "n": int(x.size), "bootstrap_resamples_used": int(rs.size)}


def signed_rank_null_cdf(n):
    """P(T+ <= t) for t = 0..n(n+1)/2 under H0, exact, no ties."""
    m = n * (n + 1) // 2
    counts = np.zeros(m + 1, dtype=object)
    counts[0] = 1
    for r in range(1, n + 1):
        counts[r:] = counts[r:] + counts[:-r].copy()
    total = 2 ** n
    return np.array([float(c) for c in np.cumsum(counts)]) / total


def hodges_lehmann(d, level=0.90):
    d = np.asarray(d, float)
    n = d.size
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    est = float(np.median(walsh))
    cdf = signed_rank_null_cdf(n)
    a = (1 - level) / 2
    k = int(np.max(np.flatnonzero(cdf <= a))) + 1 if np.any(cdf <= a) else 0  # C_alpha (1-based) = t + 1
    m = walsh.size
    lo, hi = walsh[k - 1] if k >= 1 else walsh[0], walsh[m - k] if k >= 1 else walsh[-1]
    achieved = 1 - 2 * cdf[k - 1] if k >= 1 else None
    return {"estimate": f(est), f"ci{int(level * 100)}": [f(lo), f(hi)], "walsh_averages": int(m),
            "order_statistic_k": k, "achieved_level": f(achieved) if achieved is not None else None}


def wilcoxon(d, alternative):
    d = np.asarray(d, float)
    r = stats.wilcoxon(d, zero_method="wilcox", method="auto", alternative=alternative)
    nz = d[d != 0]
    ties = np.unique(np.abs(nz)).size != nz.size
    method = "asymptotic" if d.size > 50 else ("exact" if not ties and nz.size == d.size else
                                                ("permutation" if d.size <= 13 else "asymptotic"))
    return {"statistic": f(r.statistic), "p": f(r.pvalue), "alternative": alternative, "n": int(d.size),
            "n_nonzero": int(nz.size), "method": method}


# ---------------- analyses ----------------

def analyse():
    res = json.loads((CONF / "results.json").read_text(encoding="utf-8"))
    assert res["header"] == "CONFIRMATORY (confirmation data)"
    auc, gap, brier = read_table("per_dataset_auc_2000.csv"), read_table("per_dataset_gap_2000.csv"), \
        read_table("per_dataset_brier_2000.csv")
    logpn = json.loads((REPO / "results/paper/sources.json").read_text(encoding="utf-8"))["log_pn_per_dataset"]
    ds = sorted(auc)
    assert len(ds) == 31 and set(ds) == set(logpn) == set(res["confirmatory"]["H1"]["d_per_dataset"])
    c = res["confirmatory"]
    mean = lambda row, cols: sum(row[k] for k in cols) / len(cols)
    out = {"header": HEADER, "datasets": ds, "bootstrap": {"resamples": N_BOOT, "seed": SEED, "fresh_generator_per_quantity": True}}

    # consistency with the frozen analysis
    h1 = {d: mean(gap[d], [f"{s}__C0" for s in C0]) for d in ds}
    h5 = {d: mean(auc[d], [f"{s}__C0" for s in METAPHOR]) - mean(auc[d], [f"{s}__C0" for s in CLASSICAL]) for d in ds}
    for d in ds:
        assert math.isclose(h1[d], c["H1"]["d_per_dataset"][d], abs_tol=1e-12)
        assert math.isclose(h5[d], c["H5"]["d_per_dataset"][d], abs_tol=1e-12)
    h2, h3 = c["H2"]["d_per_dataset"], c["H3"]["d_per_dataset"]

    # A
    allk = {d: gap[d]["baseline__all_k"] for d in ds}
    dA = {d: h1[d] - allk[d] for d in ds}
    out["A_search_attributable_optimism"] = {
        "d_A": {**summary([dA[d] for d in ds]), "n_positive": sum(dA[d] > 0 for d in ds)},
        "c0_mean_gap": summary([h1[d] for d in ds]), "all_k_gap": summary([allk[d] for d in ds]),
        "per_dataset": {d: {"c0_mean_gap": h1[d], "all_k_gap": allk[d], "d_A": dA[d]} for d in ds}}

    # B
    x = [logpn[d] for d in ds]
    out["B_spearman_log_pn"] = {"c0_mean_gap": spearman_ci(x, [h1[d] for d in ds]),
                                "all_k_gap": spearman_ci(x, [allk[d] for d in ds]),
                                "d_A": spearman_ci(x, [dA[d] for d in ds])}

    # C
    best = {d: max(auc[d][f"baseline__{b}"] for b in BASELINES) for d in ds}
    best_by = {d: max(BASELINES, key=lambda b: auc[d][f"baseline__{b}"]) for d in ds}
    guided = {d: mean(auc[d], [f"{s}__C0" for s in C0 if s != "random"]) - auc[d]["random__C0"] for d in ds}
    bv = np.array([best[d] for d in ds])
    strata = {}
    for name, sel in (("best_lt_0.75", [d for d in ds if best[d] < 0.75]), ("best_ge_0.75", [d for d in ds if best[d] >= 0.75])):
        strata[name] = {"n": len(sel), "datasets": sel,
                        "H2_d": summary([h2[d] for d in sel]), "H3_d": summary([h3[d] for d in sel]),
                        "guided_minus_random_C0": summary([guided[d] for d in sel])}
    out["C_dataset_difficulty"] = {
        "best_baseline_auc": {"min": f(bv.min()), "q1": f(np.percentile(bv, 25)), "median": f(np.median(bv)),
                              "q3": f(np.percentile(bv, 75)), "max": f(bv.max()), "mean": f(bv.mean())},
        "n_best_lt_0.60": int((bv < 0.60).sum()), "n_best_ge_0.75": int((bv >= 0.75).sum()),
        "best_baseline_counts": {b: sum(best_by[d] == b for d in ds) for b in BASELINES},
        "strata": strata,
        "per_dataset": {d: {"best_baseline_auc": best[d], "best_baseline": best_by[d],
                            "guided_minus_random_C0": guided[d]} for d in ds}}

    # D
    out["D_equivalence_details"] = {}
    for h, d_, margin in (("H3", h3, 0.02), ("H5", h5, 0.01)):
        v = [d_[d] for d in ds]
        out["D_equivalence_details"][h] = {"hodges_lehmann": hodges_lehmann(v, 0.90), "ci90_mean_bootstrap": boot_ci(v, 0.90),
                                           "mean": f(np.mean(v)), "margin": margin,
                                           "n_abs_d_gt_margin": int(sum(abs(x) > margin for x in v))}

    # E
    dS = {d: mean(auc[d], ["gwo_S__C0", "woa_S__C0"]) - mean(auc[d], ["ga__C0", "bpso_S__C0", "de_S__C0"]) for d in ds}
    dV = {d: mean(auc[d], ["gwo_V__C0", "woa_V__C0"]) - mean(auc[d], ["ga__C0", "bpso_V__C0", "de_V__C0"]) for d in ds}
    out["E_H5_by_transfer"] = {}
    for name, dd in (("S_shaped", dS), ("V_shaped", dV)):
        v = np.array([dd[d] for d in ds])
        out["E_H5_by_transfer"][name] = {**summary(v), "hodges_lehmann": hodges_lehmann(v, 0.90)["estimate"],
                                         "post_hoc_test_d_minus_0.01_less": wilcoxon(v - 0.01, "less"),
                                         "per_dataset": {d: dd[d] for d in ds}}

    # F
    dF = {d: mean(brier[d], ["gwo_S__C0", "gwo_V__C0", "woa_V__C0"]) - mean(brier[d], [f"{s}__C0" for s in CLASSICAL])
          for d in ds}
    v = np.array([dF[d] for d in ds])
    out["F_brier_H5_without_woa_S"] = {**summary(v), "two_sided_wilcoxon": wilcoxon(v, "two-sided"),
                                       "per_dataset": {d: dF[d] for d in ds}}

    # G
    out["G_mixed_model_aggregated"] = {resp: mixed(src, ds, logpn) for resp, src in (("outer_auc", auc), ("optimism_gap", gap))}

    # H
    out["H_structural_bias_runs"] = verification_runs()
    return out


def mixed(table, ds, logpn):
    import pandas as pd
    import statsmodels.formula.api as smf
    rows = [{"dataset": d, "searcher": s, "control": c, "y": table[d][f"{s}__{c}"], "log_pn": logpn[d]}
            for d in ds for s in CTRL for c in ("C0", "C1", "C2", "C3", "C4")]
    df = pd.DataFrame(rows)

    def fit(formula):
        tried = []
        for method in ("lbfgs", "powell"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = smf.mixedlm(f"y ~ {formula}", df, groups=df["dataset"]).fit(reml=False, method=method)
            tried.append({"method": method, "converged": bool(r.converged),
                          "warnings": [f"{w.category.__name__}: {w.message}" for w in caught
                                       if not issubclass(w.category, (DeprecationWarning, FutureWarning))]})
            if r.converged:
                return r, tried
        return r, tried

    full, ft = fit("C(searcher) * C(control) + log_pn")
    red, rt = fit("C(searcher) + C(control) + log_pn")
    ci = full.conf_int()
    fe = {k: {"estimate": f(full.fe_params[k]), "se": f(full.bse_fe[k]), "ci95": [f(ci.loc[k, 0]), f(ci.loc[k, 1])]}
          for k in full.fe_params.index}
    df_lrt = int(len(full.fe_params) - len(red.fe_params))
    lr = 2 * (full.llf - red.llf)
    return {"rows": int(len(df)), "datasets": len(ds), "formula": "y ~ C(searcher) * C(control) + log_pn, random intercept for dataset",
            "reference_levels": {"searcher": "bpso_S", "control": "C0"}, "fixed_effects": fe,
            "dataset_variance": f(full.cov_re.iloc[0, 0]), "residual_variance": f(full.scale),
            "llf_full": f(full.llf), "llf_reduced": f(red.llf), "lr_statistic": f(lr), "df": df_lrt,
            "p": f(stats.chi2.sf(lr, df_lrt)), "fit_full": ft, "fit_reduced": rt}


def verification_runs():
    text = (REPO / "docs/VERIFICATION.md").read_text(encoding="utf-8")
    sec3 = text.split("## 3. Structural bias")[1].split("## 3b.")[0]
    sec3b = text.split("## 3b.")[1].split("## 4.")[0]
    seeds3 = int(re.search(r"(\d+) seeds", sec3).group(1))
    seeds3b = int(re.search(r"(\d+) seeds", sec3b).group(1))
    rows = re.findall(r"^\| (\w+) \|", sec3, re.M)
    searchers = [r for r in rows if r != "Searcher"]
    return {"location_test_runs_per_encoding": seeds3, "encodings": 2, "location_test_runs_per_searcher": 2 * seeds3,
            "dispersion_runs_per_searcher": seeds3b, "searchers": searchers,
            "source": "docs/VERIFICATION.md sections 3 and 3b (original and inverted encoding, 30 seeds each; "
                      "dispersion uses the original-encoding runs)"}


# ---------------- outputs ----------------

def write_outputs(out):
    if OUT.exists():
        for p in sorted(OUT.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)
    files = []

    def wcsv(name, header, rows):
        p = OUT / f"{name}.csv"
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# {HEADER}\n")
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            for r in rows:
                w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
        files.append(p)

    clean = lambda o: json.loads(json.dumps(o, allow_nan=False))
    p = OUT / "posthoc.json"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(clean(out), indent=2, sort_keys=True) + "\n")
    files.append(p)

    A = out["A_search_attributable_optimism"]
    wcsv("A_per_dataset", ["dataset", "c0_mean_gap", "all_k_gap", "d_A"],
         [[d, v["c0_mean_gap"], v["all_k_gap"], v["d_A"]] for d, v in A["per_dataset"].items()])
    B = out["B_spearman_log_pn"]
    wcsv("B_spearman", ["quantity", "rho", "ci95_low", "ci95_high", "n"],
         [[k, v["rho"], v["ci95"][0], v["ci95"][1], v["n"]] for k, v in B.items()])
    C = out["C_dataset_difficulty"]
    wcsv("C_per_dataset", ["dataset", "best_baseline_auc", "best_baseline", "guided_minus_random_C0"],
         [[d, v["best_baseline_auc"], v["best_baseline"], v["guided_minus_random_C0"]] for d, v in C["per_dataset"].items()])
    wcsv("C_strata", ["stratum", "n", "quantity", "mean", "ci95_low", "ci95_high", "median"],
         [[s, v["n"], q, v[q]["mean"], v[q]["ci95_mean"][0], v[q]["ci95_mean"][1], v[q]["median"]]
          for s, v in C["strata"].items() for q in ("H2_d", "H3_d", "guided_minus_random_C0")])
    D = out["D_equivalence_details"]
    wcsv("D_equivalence", ["d", "hodges_lehmann", "hl_ci90_low", "hl_ci90_high", "mean", "boot_ci90_low", "boot_ci90_high",
                           "margin", "n_abs_d_gt_margin"],
         [[h, v["hodges_lehmann"]["estimate"], *v["hodges_lehmann"]["ci90"], v["mean"], *v["ci90_mean_bootstrap"],
           v["margin"], v["n_abs_d_gt_margin"]] for h, v in D.items()])
    E = out["E_H5_by_transfer"]
    wcsv("E_H5_by_transfer", ["transfer", "n", "mean", "ci95_low", "ci95_high", "median", "hodges_lehmann", "W", "p", "method"],
         [[k, v["n"], v["mean"], *v["ci95_mean"], v["median"], v["hodges_lehmann"],
           v["post_hoc_test_d_minus_0.01_less"]["statistic"], v["post_hoc_test_d_minus_0.01_less"]["p"],
           v["post_hoc_test_d_minus_0.01_less"]["method"]] for k, v in E.items()])
    F = out["F_brier_H5_without_woa_S"]
    wcsv("F_brier_H5_without_woa_S", ["n", "mean", "ci95_low", "ci95_high", "median", "W", "p", "method"],
         [[F["n"], F["mean"], *F["ci95_mean"], F["median"], F["two_sided_wilcoxon"]["statistic"],
           F["two_sided_wilcoxon"]["p"], F["two_sided_wilcoxon"]["method"]]])
    G = out["G_mixed_model_aggregated"]
    wcsv("G_fixed_effects", ["response", "term", "estimate", "se", "ci95_low", "ci95_high"],
         [[r, t, v["estimate"], v["se"], *v["ci95"]] for r, m in G.items() for t, v in m["fixed_effects"].items()])
    wcsv("G_model_summary", ["response", "rows", "dataset_variance", "residual_variance", "llf_full", "llf_reduced",
                             "lr_statistic", "df", "p", "warnings"],
         [[r, m["rows"], m["dataset_variance"], m["residual_variance"], m["llf_full"], m["llf_reduced"],
           m["lr_statistic"], m["df"], m["p"], "; ".join(w for x in m["fit_full"] + m["fit_reduced"] for w in x["warnings"])]
          for r, m in G.items()])
    H = out["H_structural_bias_runs"]
    wcsv("H_structural_bias_runs", ["runs_per_encoding", "encodings", "location_runs_per_searcher",
                                    "dispersion_runs_per_searcher", "searchers"],
         [[H["location_test_runs_per_encoding"], H["encodings"], H["location_test_runs_per_searcher"],
           H["dispersion_runs_per_searcher"], len(H["searchers"])]])
    files.append(write_md(out))
    rows = [[p.relative_to(OUT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size] for p in sorted(files)]
    m = OUT / "posthoc_manifest.csv"
    with open(m, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {HEADER}\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "sha256", "bytes"])
        w.writerows(rows)
    return rows


def write_md(out):
    r4 = lambda x: "-" if x is None else f"{x:.4f}"
    r3 = lambda x: "-" if x is None else f"{x:.3f}"
    ci = lambda c, fn=r4: f"[{fn(c[0])}, {fn(c[1])}]"
    pv = lambda p: "-" if p is None else (lambda s: s[:-1] if s.endswith(".") else s)(f"{p:#.2g}")
    L = [f"# {HEADER}", ""]
    A = out["A_search_attributable_optimism"]
    L += ["## A. Search-attributable optimism", "", "| Quantity | n | Mean | Median | 95% CI of mean | Datasets > 0 |",
          "|---|---|---|---|---|---|"]
    for k, lab in (("c0_mean_gap", "Mean gap, 12 C0"), ("all_k_gap", "Gap of all_k"), ("d_A", "d_A = C0 mean gap - all_k gap")):
        v = A[k]
        L.append(f"| {lab} | {v['n']} | {r4(v['mean'])} | {r4(v['median'])} | {ci(v['ci95_mean'])} | "
                 f"{v.get('n_positive', '-')} |")
    B = out["B_spearman_log_pn"]
    L += ["", "## B. Spearman correlation with log(p/n)", "", "| Quantity | rho | 95% bootstrap CI | n |", "|---|---|---|---|"]
    for k, lab in (("c0_mean_gap", "Mean gap, 12 C0"), ("all_k_gap", "Gap of all_k"), ("d_A", "d_A")):
        L.append(f"| {lab} | {r3(B[k]['rho'])} | {ci(B[k]['ci95'], r3)} | {B[k]['n']} |")
    C = out["C_dataset_difficulty"]
    b = C["best_baseline_auc"]
    L += ["", "## C. Dataset difficulty (best mean outer AUC among the six baselines)", "",
          f"Min {r3(b['min'])}, Q1 {r3(b['q1'])}, median {r3(b['median'])}, Q3 {r3(b['q3'])}, max {r3(b['max'])}, "
          f"mean {r3(b['mean'])}. Datasets with best AUC < 0.60: {C['n_best_lt_0.60']}; >= 0.75: {C['n_best_ge_0.75']}. "
          "Best baseline counts: " + ", ".join(f"{k} {v}" for k, v in C["best_baseline_counts"].items()) + ".", "",
          "| Stratum | n | H2 d mean [95% CI] | H3 d mean [95% CI] | Guided C0 - random C0 mean [95% CI] |", "|---|---|---|---|---|"]
    for s, v in C["strata"].items():
        L.append(f"| {s.replace('best_lt_', 'best < ').replace('best_ge_', 'best >= ')} | {v['n']} | "
                 + " | ".join(f"{r4(v[q]['mean'])} {ci(v[q]['ci95_mean'])}" for q in ("H2_d", "H3_d", "guided_minus_random_C0"))
                 + " |")
    D = out["D_equivalence_details"]
    L += ["", "## D. Equivalence details", "",
          "| d | Hodges-Lehmann | HL 90% CI (exact signed-rank) | Mean | 90% bootstrap CI of mean | Margin | Datasets with abs(d) > margin |",
          "|---|---|---|---|---|---|---|"]
    for h, v in D.items():
        hl = v["hodges_lehmann"]
        L.append(f"| {h} | {r4(hl['estimate'])} | {ci(hl['ci90'])} | {r4(v['mean'])} | {ci(v['ci90_mean_bootstrap'])} | "
                 f"{v['margin']} | {v['n_abs_d_gt_margin']} |")
    E = out["E_H5_by_transfer"]
    L += ["", "## E. H5 by transfer function (post hoc test: one-sided Wilcoxon of d - 0.01 < 0)", "",
          "| Transfer | n | Mean | 95% CI | Hodges-Lehmann | W | p (post hoc) |", "|---|---|---|---|---|---|---|"]
    for k, v in E.items():
        t = v["post_hoc_test_d_minus_0.01_less"]
        L.append(f"| {k.replace('_', '-')} | {v['n']} | {r4(v['mean'])} | {ci(v['ci95_mean'])} | {r4(v['hodges_lehmann'])} | "
                 f"{t['statistic']:g} | {pv(t['p'])} |")
    F = out["F_brier_H5_without_woa_S"]
    t = F["two_sided_wilcoxon"]
    L += ["", "## F. Brier score, H5 without WOA-S (metaphor = GWO-S, GWO-V, WOA-V; descriptive)", "",
          "| n | Mean | 95% CI | Median | W | p (two-sided) |", "|---|---|---|---|---|---|",
          f"| {F['n']} | {r4(F['mean'])} | {ci(F['ci95_mean'])} | {r4(F['median'])} | {t['statistic']:g} | {pv(t['p'])} |"]
    G = out["G_mixed_model_aggregated"]
    for resp, m in G.items():
        L += ["", f"## G. Mixed model on per-dataset means ({resp}): {m['rows']} rows, {m['datasets']} datasets", "",
              f"Dataset variance {m['dataset_variance']:.3g}; residual variance {m['residual_variance']:.3g}; "
              f"interaction LRT {m['lr_statistic']:.2f}, df {m['df']}, p {pv(m['p'])}; log-likelihood full "
              f"{m['llf_full']:.2f}, reduced {m['llf_reduced']:.2f}. Fits: "
              + "; ".join(f"{x['method']} converged={x['converged']}" for x in m["fit_full"] + m["fit_reduced"])
              + ". Warnings: " + ("; ".join(sorted({w for x in m["fit_full"] + m["fit_reduced"] for w in x["warnings"]})) or "none")
              + ".", "", "| Term | Estimate | SE | 95% CI |", "|---|---|---|---|"]
        for term, v in m["fixed_effects"].items():
            L.append(f"| {term} | {v['estimate']:+.5f} | {v['se']:.5f} | {ci(v['ci95'], lambda x: f'{x:+.5f}')} |")
    H = out["H_structural_bias_runs"]
    L += ["", "## H. Runs per searcher in the pure-noise structural-bias test (docs/VERIFICATION.md)", "",
          f"{H['location_test_runs_per_encoding']} runs per encoding x {H['encodings']} encodings = "
          f"{H['location_test_runs_per_searcher']} runs per searcher for the location tests; the dispersion diagnostic uses "
          f"the {H['dispersion_runs_per_searcher']} original-encoding runs; {len(H['searchers'])} searchers.", ""]
    p = OUT / "posthoc.md"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    return p


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="for testing only; default results/paper/posthoc")
    OUT = Path(ap.parse_args(argv).out)
    rows = write_outputs(analyse())
    print(f"written {len(rows)} files to {OUT.name}/")
    for path, h, b in rows:
        print(f"  {path:<34} {b:>8}  {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
