"""Analysis entry point.

    python -m analysis.run --config configs/analysis_<label>.yaml [--results-dir results] [--output-dir DIR]

Steps: validate the named plans exactly as src/validate_release.py (refuse on any failure); load the run
files listed in results/MANIFEST.csv for those plans (applying 'replaces'); keep the configured control
values; compute per-dataset scores, the confirmatory family H1-H5 with Holm, the sensitivity analyses S1
(without the sensitivity set) and S2 (if an S2 plan is named), and the secondary analyses; write
results/analysis/<label>/: results.json (full precision), tables/*.csv, figures/*.pdf|png and
analysis_manifest.csv (SHA-256 of every output). Every output carries the data header
(e.g. "EXPLORATORY (development data)"). Outputs are byte-identical for identical inputs and code.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import analyses as an
from . import figures as fg
from .load import (HEADERS, REPO_ROOT, Refused, load_config, load_plans, plan_group, read_list, resolve,
                   validate)

S1_DEFAULT = ["Arita2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Keek2020", "LNDb", "UCSF-PDGM", "Wang2024"]


def clean(obj):
    """JSON-safe copy: numpy scalars to Python, nan/inf to None, tuple to list."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if np.isfinite(x) else None
    return obj


def write_csv(df, path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {header}\n")
        df.to_csv(f, index=False, lineterminator="\n")
    return path


def versions():
    import matplotlib
    import patsy
    import scipy
    import statsmodels
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "patsy": patsy.__version__,
            "matplotlib": matplotlib.__version__}


def code_digest():
    h = hashlib.sha256()
    for f in sorted((REPO_ROOT / "src" / "analysis").glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def confirmatory_rows(name, res):
    rows = []
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        r = res[h]
        eff = r.get("effect", {})
        test = r.get("test") or (r.get("tost") and {"method": r["tost"]["upper"]["method"]}) or {}
        rows.append({"analysis": name, "hypothesis": h, "skipped": bool(r.get("skipped", False)),
                     "n_datasets": eff.get("n_datasets"), "mean_d": eff.get("mean"), "median_d": eff.get("median"),
                     "ci95_low": (eff.get("ci95_mean") or [None, None])[0],
                     "ci95_high": (eff.get("ci95_mean") or [None, None])[1], "rank_biserial": eff.get("rank_biserial"),
                     "p_raw": r.get("p_raw"), "p_holm": r.get("p_holm"), "reject_holm": r.get("reject_holm"),
                     "wilcoxon_method": test.get("method"), "wording": r.get("wording")})
    return rows


def _prepare_output(out_dir):
    """Create the output folder; an existing one is cleared only if it is empty or holds a previous
    analysis output (results.json or analysis_manifest.csv), never an arbitrary folder."""
    if out_dir.exists():
        entries = list(out_dir.iterdir())
        if entries and not ((out_dir / "results.json").exists() or (out_dir / "analysis_manifest.csv").exists()):
            raise Refused(f"output folder exists and is not an analysis output: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "tables").mkdir(parents=True)
    (out_dir / "figures").mkdir(parents=True)


def analyse(cfg, results_dir, output_dir=None):
    results_dir = Path(results_dir).resolve()
    plan_files = [resolve(e["file"]) for e in cfg["plans"]]
    s2_file = resolve(cfg["s2_plan"]) if cfg.get("s2_plan") else None
    report = validate(plan_files + ([s2_file] if s2_file else []), results_dir)
    group = plan_group(cfg["plans"])
    header = HEADERS[group]

    cfg["control_values"] = cfg.get("control_values") or {}
    records, inputs = load_plans(cfg["plans"], results_dir)
    df_all = an.run_table(records)
    df, excluded = an.select_control_values(df_all, cfg["control_values"])
    kept_configs = set(df["config"].unique())
    records_kept = [r for r in records if r["config"] in kept_configs and
                    (r["control"] not in ("C1", "C2", "C3") or
                     float(r["control_value"]) == float(cfg["control_values"].get(r["control"], r["control_value"])))]

    s1_list = read_list(cfg["sensitivity_list"]) if cfg.get("sensitivity_list") else S1_DEFAULT
    out_dir = Path(output_dir).resolve() if output_dir else results_dir / "analysis" / cfg["label"]
    _prepare_output(out_dir)
    written = []

    # per-dataset tables
    for metric in ("auc", "gap", "brier"):
        written.append(write_csv(an.per_dataset(df, metric).reset_index(),
                                 out_dir / "tables" / f"per_dataset_{metric}_2000.csv", header))

    main = an.confirmatory(df)
    s1_df = df[~df["dataset"].isin(s1_list)]
    s1 = an.confirmatory(s1_df) if s1_df["dataset"].nunique() >= 1 else {"skipped": True}
    rows = confirmatory_rows("main", main) + (confirmatory_rows("S1", s1) if not s1.get("skipped") else [])

    s2 = {"skipped": True, "reason": "no S2 plan named in the config"}
    if s2_file:
        s2_entry = [{"file": cfg["s2_plan"]}]
        rec2, inputs2 = load_plans(s2_entry, results_dir)
        inputs += inputs2
        auc2 = an.per_dataset(an.run_table(rec2), "auc")
        if {"ga__C4", "baseline__l1"} <= set(auc2.columns):
            from . import stats as st
            d = (auc2["ga__C4"] - auc2["baseline__l1"]).dropna()
            s2 = {"plan": Path(cfg["s2_plan"]).name, "d_per_dataset": an._series(d),
                  "tost": st.tost(d, an.H3_MARGIN), "two_sided": st.wilcoxon(d, "two-sided"), "effect": st.effect(d),
                  "sensitivity_only": True}
            written.append(write_csv(pd.DataFrame({"dataset": d.index, "d_ga_C4_minus_l1": d.to_numpy()}),
                                     out_dir / "tables" / "s2_h3_per_dataset.csv", header))
        else:
            s2 = {"skipped": True, "reason": "ga__C4 or baseline__l1 missing in the S2 plan"}
    written.append(write_csv(pd.DataFrame(rows), out_dir / "tables" / "confirmatory_summary.csv", header))

    any_t = an.anytime(df)
    written.append(write_csv(any_t, out_dir / "tables" / "anytime.csv", header))
    rvg, rvg_t = an.random_vs_guided(df)
    if len(rvg_t):
        written.append(write_csv(rvg_t, out_dir / "tables" / "random_vs_guided.csv", header))
    stab_ds, stab_sum = an.size_stability(records_kept, kept_configs)
    written.append(write_csv(stab_ds, out_dir / "tables" / "size_stability_per_dataset.csv", header))
    written.append(write_csv(stab_sum, out_dir / "tables" / "size_stability_summary.csv", header))
    brier = an.brier_paired(df)
    fr, fr_t = an.friedman(df)
    if len(fr_t):
        written.append(write_csv(fr_t, out_dir / "tables" / "friedman_pairwise.csv", header))

    # figures
    figs = out_dir / "figures"
    written += fg.anytime(any_t, header, figs / "anytime")
    log_pn = df.groupby("dataset")["log_pn"].first()
    if not main["H1"].get("skipped"):
        d1 = pd.Series(main["H1"]["d_per_dataset"]).dropna()
        written += fg.gap_vs_logpn(d1, log_pn, header, figs / "optimism_gap_vs_log_pn")
    if not main["H2"].get("skipped"):
        written += fg.differences(pd.Series(main["H2"]["d_per_dataset"]).dropna(),
                                  "H2: per-dataset mean (C4 - C0) over the controlled searchers",
                                  "difference in outer AUC", an.H3_MARGIN, header, figs / "h2_differences")
    if not main["H3"].get("skipped"):
        written += fg.differences(pd.Series(main["H3"]["d_per_dataset"]).dropna(),
                                  "H3: per-dataset (ga + C4) - L1 logistic regression",
                                  "difference in outer AUC", an.H3_MARGIN, header, figs / "h3_differences")

    digest = hashlib.sha256("\n".join(f"{p} {h}" for p, h in sorted(inputs)).encode()).hexdigest()
    results = {
        "header": header, "label": cfg["label"], "exploratory": group == "development", "group": group,
        "versions": versions(),
        "code": {"head_at_run": subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                               capture_output=True, text=True).stdout.strip(),
                 "analysis_code_sha256": code_digest()},
        "config": {"plans": cfg["plans"], "s2_plan": cfg.get("s2_plan"), "control_values": cfg.get("control_values"),
                   "sensitivity_list": cfg.get("sensitivity_list"), "sensitivity_set": s1_list},
        "inputs": {"n_run_files": len(inputs), "sha256_of_sorted_path_hash_list": digest,
                   "n_records": len(records), "datasets": sorted(df["dataset"].unique())},
        "validation_report": report.splitlines(),
        "configurations": {**an.missing_report(df), "excluded_control_values": excluded},
        "definitions": {
            "run_level_outer_auc": "outer AUC of the subset returned at the checkpoint; C4: knee record; baseline: returned subset at 2000",
            "optimism_gap": "standard inner AUC (unpenalized mean over the fixed 3-fold inner split) minus outer AUC",
            "per_dataset_score": "mean over runs (outer folds x search seeds; rf: outer folds x model seeds)",
            "primary_checkpoint": an.PRIMARY,
            "wilcoxon": "scipy.stats.wilcoxon(zero_method='wilcox', method='auto'); method actually used recorded",
            "bootstrap": "95% percentile CI of the mean over datasets, 10,000 resamples, numpy default_rng(20261113)",
            "holm": "statsmodels multipletests(method='holm'), alpha 0.05, over H1, H2, H3 (p_TOST), H4 (LRT), H5"},
        "confirmatory": main,
        "sensitivity": {"S1": {"excluded_datasets": s1_list, "datasets_remaining": sorted(s1_df["dataset"].unique()),
                               "results": s1, "label": "sensitivity analysis; not in the Holm family"},
                        "S2": s2},
        "secondary": {"label": "secondary analyses; not in the Holm family",
                      "anytime": "tables/anytime.csv", "random_vs_guided": rvg,
                      "size_stability": {"method": "Nogueira et al. (2018) across outer folds per dataset x "
                                                   "configuration x seed index, averaged over seed indices; features "
                                                   "identified by original column name (stored in the run files); "
                                                   "d = n_features after the data-handling rules",
                                         "summary": stab_sum.to_dict(orient="records")},
                      "brier": brier, "friedman": fr},
    }
    rj = out_dir / "results.json"
    with open(rj, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(clean(results), indent=2, sort_keys=True, allow_nan=False) + "\n")
    written.append(rj)

    rows = [{"path": p.relative_to(out_dir).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "bytes": p.stat().st_size} for p in sorted(written)]
    write_csv(pd.DataFrame(rows), out_dir / "analysis_manifest.csv", header)
    return out_dir, results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--output-dir", default=None, help="default: <results-dir>/analysis/<label>")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(resolve(args.config))
        out_dir, res = analyse(cfg, args.results_dir, args.output_dir)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"{res['header']}: analysis '{cfg['label']}' written to {out_dir}")
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        r = res["confirmatory"][h]
        print(f"  {h}: p_raw {r.get('p_raw')}, p_holm {r.get('p_holm')}: {r.get('wording')}")
    h4 = res["confirmatory"]["H4"]
    for part in ("outer_auc", "optimism_gap_descriptive"):
        for model in ("fit_full", "fit_reduced"):
            for fit in h4.get(part, {}).get(model, []):
                for w in fit.get("warnings", []):
                    print(f"  H4 {part} {model} {fit['method']}: {w['category']}: {w['message']}")
    print(f"  missing configurations: {len(res['configurations']['missing'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
