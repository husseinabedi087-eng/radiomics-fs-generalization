"""Post hoc: standard errors and variance components of the frozen run-level H4 mixed models.

    python src/paper/posthoc_runlevel.py

The frozen analysis (src/analysis at 0b3da97) stored the fixed effects, log-likelihoods and fit warnings
of the run-level mixed models, but not their standard errors or variance components. This script refits
the same two models (outer AUC and optimism gap; 74,400 rows each) by running the frozen code path itself:
analysis.load.load_config / load_plans, analysis.analyses.run_table / select_control_values / h4_frame and
analysis.stats.mixed_lrt, with configs/analysis_confirmation.yaml. It does not re-implement the model:
analysis.stats._fit_mixed is wrapped only to keep the fitted statsmodels result objects that mixed_lrt
discards.

Before writing anything it checks that every refitted fixed effect matches results/analysis/confirmation/
results.json to within 1e-6, and that both interaction likelihood ratios match to within 1e-6; on any
mismatch it prints the differences and exits with code 1 without writing.

Outputs (header "POST HOC (not pre-registered; confirmation data)") in results/paper/posthoc/:
I_runlevel_fixed_effects.csv, I_runlevel_variance_components.csv, I_runlevel_fits.csv; their SHA-256
are added to posthoc_manifest.csv (other rows kept). Run after src/paper/posthoc.py, which rewrites that
folder. Variance-component standard errors use statsmodels' bse_re formula (the "Std.Err." of its summary);
statsmodels gives no standard error for the residual variance and no confidence interval for variances.
In statsmodels 0.14.6 that value is sqrt(scale) x SE(psi), where psi = variance / residual variance; the
file also gives SE(psi) and the approximate SE on the variance scale, scale x SE(psi) (delta method,
residual variance treated as known). statsmodels' "MLE may be on the boundary" warning is issued whenever
any random-effect variance is below an absolute 0.01 (mixed_linear_model.py, fit), not only at zero.
"""

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from analysis import analyses as an  # noqa: E402
from analysis import stats as st  # noqa: E402
from analysis.load import load_config, load_plans, resolve  # noqa: E402

OUT = REPO / "results/paper/posthoc"
HEADER = "POST HOC (not pre-registered; confirmation data)"
TOL = 1e-6
RESPONSES = {"auc": ("outer_auc", "outer AUC"), "gap": ("optimism_gap_descriptive", "optimism gap")}


def refit():
    cfg = load_config(resolve("configs/analysis_confirmation.yaml"))
    cfg["control_values"] = cfg.get("control_values") or {}
    records, _ = load_plans(cfg["plans"], REPO / "results")
    df, _ = an.select_control_values(an.run_table(records), cfg["control_values"])
    h4 = an.h4_frame(df)
    captured = []
    frozen_fit = st._fit_mixed

    def keep(formula, data, response):
        res, tried = frozen_fit(formula, data, response)
        captured.append((formula, res))
        return res, tried

    st._fit_mixed = keep
    out = {}
    try:
        for response in RESPONSES:
            captured.clear()
            summary = st.mixed_lrt(h4, response)
            full = next(r for f, r in captured if "*" in f)
            red = next(r for f, r in captured if "*" not in f)
            out[response] = (summary, full, red)
    finally:
        st._fit_mixed = frozen_fit
    return out


def check(out, frozen):
    problems = []
    for response, (summary, full, red) in out.items():
        key, _ = RESPONSES[response]
        fz = frozen[key]
        for term, v in fz["fe_params_full"].items():
            d = abs(float(full.fe_params[term]) - v)
            if not d <= TOL:
                problems.append(f"{response} {term}: refit {full.fe_params[term]!r} vs frozen {v!r} (diff {d:.3g})")
        if set(full.fe_params.index) != set(fz["fe_params_full"]):
            problems.append(f"{response}: term sets differ")
        d = abs(summary["lr_statistic"] - fz["lr_statistic"])
        if not d <= TOL:
            problems.append(f"{response} LR: refit {summary['lr_statistic']!r} vs frozen {fz['lr_statistic']!r}")
    return problems


def var_rows(response, which, res):
    p = res.model.exog.shape[1]
    # statsmodels' own bse_re formula, unwrapped: res.bse_re fails in statsmodels 0.14.6 when a model has both a
    # random intercept and a variance component (it attaches one name to two values); summary() uses the same values
    se = np.sqrt(res.scale * np.diag(np.asarray(res.cov_params()))[p:])
    names = ["dataset (random intercept)"] + [f"{n} within dataset (variance component)" for n in res.model.exog_vc.names]
    values = [float(res.cov_re.iloc[0, 0])] + [float(v) for v in res.vcomp]
    # statsmodels' variance-parameter covariance refers to psi = variance / residual variance, and its bse_re is
    # sqrt(scale) * SE(psi); on the variance scale SE ~= scale * SE(psi) = sqrt(scale) * bse_re (delta method,
    # residual variance treated as known)
    rows = [[response, which, n, v, float(s), float(np.sqrt(res.scale) * s), float(s / np.sqrt(res.scale))]
            for n, v, s in zip(names, values, se)]
    rows.append([response, which, "residual", float(res.scale), None, None, None])
    assert len(se) == len(values) and res.model.exog.shape[1] == p
    return rows


def write(out):
    def wcsv(name, header, rows):
        path = OUT / name
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# {HEADER}\n")
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            for r in rows:
                w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
        return path

    fe, vc, fits = [], [], []
    for response, (summary, full, red) in out.items():
        label = RESPONSES[response][1]
        ci = full.conf_int()
        for term in full.fe_params.index:
            fe.append([label, term, float(full.fe_params[term]), float(full.bse_fe[term]), float(ci.loc[term, 0]),
                       float(ci.loc[term, 1])])
        vc += var_rows(label, "full", full) + var_rows(label, "reduced", red)
        for which in ("fit_full", "fit_reduced"):
            for f in summary[which]:
                fits.append([label, which.replace("fit_", ""), f["method"], f["converged"], f["llf"],
                             "; ".join(f"{w['category']}: {w['message']}" for w in f["warnings"])])
        fits.append([label, "LRT", "", "", summary["lr_statistic"], f"df {summary['df']}, p {summary['p']!r}"])
    files = [wcsv("I_runlevel_fixed_effects.csv", ["response", "term", "estimate", "se", "ci95_low", "ci95_high"], fe),
             wcsv("I_runlevel_variance_components.csv", ["response", "model", "component", "variance", "se_statsmodels",
                                                         "se_variance_scale_approx", "se_of_ratio_to_residual"], vc),
             wcsv("I_runlevel_fits.csv", ["response", "model", "optimizer", "converged", "log_likelihood_or_LR", "warnings_or_test"],
                  fits)]
    man = OUT / "posthoc_manifest.csv"
    lines = man.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# {HEADER}"
    rows = [r for r in csv.DictReader(io.StringIO("\n".join(lines[1:]))) if not r["path"].startswith("I_runlevel_")]
    rows += [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": str(p.stat().st_size)}
             for p in files]
    with open(man, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {HEADER}\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "sha256", "bytes"])
        w.writerows([[r["path"], r["sha256"], r["bytes"]] for r in sorted(rows, key=lambda r: r["path"])])
    return files


def main():
    frozen = json.loads((REPO / "results/analysis/confirmation/results.json").read_text(encoding="utf-8"))["confirmatory"]["H4"]
    out = refit()
    problems = check(out, frozen)
    for response, (summary, full, red) in out.items():
        key = RESPONSES[response][0]
        diffs = [abs(float(full.fe_params[t]) - v) for t, v in frozen[key]["fe_params_full"].items()]
        print(f"{response}: {len(diffs)} fixed effects, max abs diff vs frozen {max(diffs):.3g}; "
              f"LR refit {summary['lr_statistic']:.6f} vs frozen {frozen[key]['lr_statistic']:.6f}; rows {summary['n_rows']}")
    if problems:
        print("MISMATCH - nothing written:")
        for p in problems:
            print("  " + p)
        return 1
    for p in write(out):
        print(f"written {p.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
