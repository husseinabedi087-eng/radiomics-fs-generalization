"""Supplementary tables S1-S10 from committed files only.

    python src/paper/make_supplementary.py

Reads only committed files: results/analysis/confirmation/, results/analysis/development_dryrun/,
results/stage2/, results/verification/, results/health/ (written by src/save_health.py, which runs the
unchanged src/run_health.py), docs/, data/meta/, data/split/ and configs/. No run file and no dataset is
opened. Computes no test and no p-value: values are read, reordered, counted and averaged (S8 averages
the full-precision per-dataset values of the analysis tables). No patient or sample identifier is read
or written.

Outputs in results/paper/supplementary/: each table as CSV (full precision) and Markdown (rounded),
the three pre-registration documents copied byte for byte, and supplementary_manifest.csv (SHA-256 of
every output). Deterministic: sorted inputs, no timestamps.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import estimate_compute  # noqa: E402  (n and p after the data-handling rules, from metadata)
from fsbench.run import load_plan  # noqa: E402

CONF = REPO / "results/analysis/confirmation"
DEV = REPO / "results/analysis/development_dryrun"
OUT = REPO / "results/paper/supplementary"
HEALTH = REPO / "results/health"
EXPLORATORY = "EXPLORATORY (development data)"
C0 = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V", "gwo_S", "gwo_V",
      "woa_S", "woa_V"]
CTRL = ["ga", "bpso_S", "de_S", "gwo_S"]
FROZEN = {"docs/PREREG_STAGE1.md": "8fa82c7", "docs/PREREG_STAGE1_AMENDMENT1.md": "485ee76",
          "docs/PREREG_STAGE2.md": "862a7ea"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fx(v, n=4):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def pv(p):
    if p is None:
        return "-"
    s = f"{p:#.2g}"
    return s[:-1] if s.endswith(".") else s


def read_csv(path, comment=False):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if comment:
        assert lines[0].startswith("# ")
        lines = lines[1:]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


class Writer:
    def __init__(self):
        self.files = []

    def csv(self, name, header, rows, label=None):
        path = OUT / f"{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            if label:
                f.write(f"# {label}\n")
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            for r in rows:
                w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
        self.files.append(path)

    def md(self, name, blocks):
        """blocks: list of (title, header, rows, notes)."""
        lines = []
        for title, header, rows, notes in blocks:
            lines += [f"**{title}**", ""]
            if header:
                lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
                lines.append("")
            lines += [n for n in notes] + ([""] if notes else [])
        path = OUT / f"{name}.md"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines).rstrip("\n") + "\n")
        self.files.append(path)


# ---------------- S1 datasets ----------------

def s1(w):
    audit = {r["dataset"]: r for r in csv.DictReader(open(REPO / "data/meta/audit.csv", encoding="utf-8"))}
    lic = {r["dataset"]: r for r in csv.DictReader(open(REPO / "data/meta/license_decisions.csv", encoding="utf-8"))}
    rules = json.loads((REPO / "data/meta/data_handling_rules.json").read_text(encoding="utf-8"))
    sizes = estimate_compute.dataset_sizes()
    s1_set = {l.strip() for l in (REPO / "configs/sensitivity_s1.txt").read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#")}
    s2_set = set(load_plan(REPO / "configs/confirmation_s2.yaml")["datasets"])
    rows = []
    for d in sorted(sizes, key=lambda d: (sizes[d][0] != "development", d)):
        g, n, p = sizes[d]
        rows.append([d, g, n, p, math.log(p / n), int(audit[d]["minority_class_count"]), audit[d]["modality"],
                     lic[d]["license_category"], "yes" if d in s1_set else "no",
                     ("yes" if d in s2_set else "no") if g == "confirmation" else "n/a"])
    excl = rules["excluded_datasets"]
    assert len(excl) == 7 and len(rows) == 43 and len(audit) == 50
    erows = [[d, audit[d]["modality"], excl[d]] for d in sorted(excl)]
    h = ["Dataset", "Group", "n", "p", "log(p/n)", "Minority-class count", "Modality", "Licence category",
         "Flagged for S1", "Included in S2"]
    w.csv("S1_datasets", h, rows)
    w.csv("S1_excluded", ["Dataset", "Modality", "Reason"], erows)
    w.md("S1_datasets", [
        ("Table S1. Included datasets (43).", h,
         [[*r[:4], f"{r[4]:.3f}", *r[5:]] for r in rows],
         ["n and p after the data-handling rules (data/meta/audit.csv, data/meta/data_handling_rules.json); "
          "log(p/n) natural logarithm. Minority-class count from the audit, before the data-handling rules. "
          "Licence category from data/meta/license_decisions.csv. S1: sensitivity-exclusion set "
          "(configs/sensitivity_s1.txt). S2: confirmation datasets with p <= 1,000 (configs/confirmation_s2.yaml)."]),
        ("Excluded datasets (7).", ["Dataset", "Modality", "Reason"], erows,
         ["Source: data/meta/data_handling_rules.json (excluded_datasets)."])])


# ---------------- S2 pre-registration ----------------

def s2(w):
    rows = []
    for rel, commit in FROZEN.items():
        src = REPO / rel
        dst = OUT / "preregistration" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert sha(dst) == sha(src)
        recorded = (src.with_suffix(".sha256")).read_text(encoding="utf-8").split()[0]
        assert recorded == sha(src), rel
        added = subprocess.run(["git", "-C", str(REPO), "log", "--diff-filter=A", "--format=%H", "--", rel],
                               capture_output=True, text=True).stdout.split()
        assert added and added[-1].startswith(commit), (rel, added)
        rows.append([src.name, sha(src), commit])
        w.files.append(dst)
    h = ["Document", "SHA-256", "Freezing commit"]
    w.csv("S2_preregistration", h, rows)
    w.md("S2_preregistration", [("Table S2. Pre-registration documents (copied unchanged in preregistration/).",
                                  h, rows, ["Each SHA-256 equals the hash committed in docs/<name>.sha256."])])


# ---------------- S3 verification ----------------

def md_tables(text):
    """(section heading, header, rows) for every Markdown table, in order."""
    out, heading, i = [], None, 0
    lines = text.splitlines()
    while i < len(lines):
        if lines[i].startswith("## "):
            heading = lines[i][3:].strip()
        if lines[i].startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            header = [c.strip() for c in lines[i].strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append((heading, header, rows))
            continue
        i += 1
    return out


def s3(w):
    text = (REPO / "docs/VERIFICATION.md").read_text(encoding="utf-8")
    tables = md_tables(text)
    names = ["S3a_mealpy", "S3b_onemax", "S3c_bias_location", "S3d_dispersion", "S3e_dispersion_mechanism",
             "S3f_determinism", "S3g_planted_subset"]
    assert len(tables) == len(names), [t[0] for t in tables]
    blocks = []
    for name, (heading, header, rows) in zip(names, tables):
        w.csv(name, header, rows)
        suffix = " - per-iteration mechanism (woa_S, woa_V)" if name == "S3e_dispersion_mechanism" else ""
        blocks.append((f"{name.split('_')[0]}. {heading}{suffix}", header, rows, []))
    raw = []
    for f in sorted((REPO / "results/verification").glob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8")).get("meta", {})
        raw.append([f.name, sha(f), meta.get("git_commit", "")[:7], meta.get("utc", "")])
    w.csv("S3h_raw_files", ["File", "SHA-256", "Code commit", "UTC"], raw)
    blocks.append(("S3h. Budget accounting (stage 1, section 9)", None, None,
                   ["Checked by the tests in tests/test_budget.py (every scored subset, including control "
                    "evaluations, increments the counter). These are pass/fail tests without a result file; "
                    "they are part of the fast suite, which passed at commit 0b3da97 (155 passed; "
                    "docs/PREREG_STAGE2.md, section 7.1)."]))
    blocks.append(("Sources", None, None,
                   [f"Tables S3a-S3g are copied from docs/VERIFICATION.md (SHA-256 {sha(REPO / 'docs/VERIFICATION.md')}); "
                    f"the {len(raw)} raw result files in results/verification/ are listed with their SHA-256 and "
                    "code commit in S3h_raw_files.csv."]))
    w.md("S3_verification", blocks)


# ---------------- S4 per-dataset results ----------------

def s4(w, res, sizes):
    c = res["confirmatory"]
    s2d = res["sensitivity"]["S2"]["d_per_dataset"]
    ds = sorted(c["H1"]["d_per_dataset"])
    rows = []
    for d in ds:
        g, n, p = sizes[d]
        rows.append([d, math.log(p / n), c["H1"]["d_per_dataset"][d], c["H2"]["d_per_dataset"][d],
                     c["H3"]["d_per_dataset"][d], c["H5"]["d_per_dataset"][d], s2d.get(d)])
    h = ["Dataset", "log(p/n)", "H1 d (optimism gap)", "H2 d (C4 - C0)", "H3 d (GA+C4 - L1)",
         "H5 d (metaphor - classical)", "S2 d (GA+C4 - L1, no pre-filter)"]
    w.csv("S4_per_dataset", h, rows)
    w.md("S4_per_dataset", [("Table S4. Per-dataset differences, 31 confirmation datasets.", h,
                             [[r[0], f"{r[1]:.3f}", *[fx(v) for v in r[2:]]] for r in rows],
                             ["Source: results/analysis/confirmation/results.json. d values to 4 decimals; "
                              "S2 only for the 17 datasets with p <= 1,000."])])


# ---------------- S5 mixed models ----------------

def s5(w, res):
    models = [("main", "outer AUC (H4)", res["confirmatory"]["H4"]["outer_auc"]),
              ("main", "optimism gap (descriptive)", res["confirmatory"]["H4"]["optimism_gap_descriptive"]),
              ("main", "Brier score (descriptive)", res["secondary"]["brier"]["H4"]),
              ("S1", "outer AUC", res["sensitivity"]["S1"]["results"]["H4"]["outer_auc"]),
              ("S1", "optimism gap (descriptive)", res["sensitivity"]["S1"]["results"]["H4"]["optimism_gap_descriptive"])]
    fe_rows, fit_rows = [], []
    for analysis, resp, m in models:
        for term in sorted(m["fe_params_full"]):
            fe_rows.append([analysis, resp, term, m["fe_params_full"][term]])
        for which in ("fit_full", "fit_reduced"):
            for f in m[which]:
                fit_rows.append([analysis, resp, which.replace("fit_", ""), f["method"], f["converged"],
                                 f["llf"], "; ".join(f"{x['category']}: {x['message']}" for x in f["warnings"])])
        fit_rows.append([analysis, resp, "LRT", "", "", m["lr_statistic"], f"df {m['df']}, p {m['p']!r}, "
                                                                           f"rows {m['n_rows']}, partitions {m['n_partitions']}"])
    w.csv("S5_fixed_effects", ["Analysis", "Response", "Term", "Estimate"], fe_rows)
    w.csv("S5_fits", ["Analysis", "Response", "Model", "Optimizer", "Converged", "Log-likelihood or LR",
                      "Warnings or test"], fit_rows)
    blocks = []
    for analysis, resp, m in models:
        p = m["fe_params_full"]
        rows = [[t, f"{p[t]:+.5f}"] for t in sorted(p)]
        warn = sorted({x["message"] for k in ("fit_full", "fit_reduced") for f in m[k] for x in f["warnings"]})
        notes = [f"Log-likelihood full {m['llf_full']:.2f}, reduced {m['llf_reduced']:.2f}; LR {m['lr_statistic']:.2f}, "
                 f"df {m['df']}, p {pv(m['p'])}; {m['n_rows']:,} rows, {m['n_datasets']} datasets, "
                 f"{m['n_partitions']} partitions. Fits: " +
                 "; ".join(f"{k.replace('fit_', '')} {f['method']} converged={f['converged']}"
                           for k in ("fit_full", "fit_reduced") for f in m[k]) +
                 f". Warnings: {'; '.join(warn) if warn else 'none'}."]
        blocks.append((f"Table S5 ({analysis}, {resp}): fixed effects of the full model", ["Term", "Estimate"], rows,
                       notes))
    blocks.append(("Variance components", None, None,
                   ["Not recorded: the frozen analysis code (src/analysis at 0b3da97) stores the fixed effects, "
                    "log-likelihoods and fit warnings of each mixed model, but not the estimated variance "
                    "components (dataset intercept and outer-partition variance). They are not refitted here."]))
    w.md("S5_mixed_models", blocks)


# ---------------- S6 secondary tests ----------------

def s6(w, res):
    rvg = res["secondary"]["random_vs_guided"]["comparisons"]
    h = ["Guided C0 searcher", "Datasets", "Mean d", "Median d", "p", "p Holm", "Rank-biserial", "Method"]
    rows = [[r["guided"], r["n_datasets"], r["mean_d"], r["median_d"], r["p"], r["p_holm"], r["rank_biserial"],
             r["method"]] for r in rvg]
    w.csv("S6a_random_vs_guided", h, rows)
    br = res["secondary"]["brier"]
    bh = ["Comparison", "Datasets", "Statistic", "p (two-sided)", "Mean d", "Median d", "CI low", "CI high",
          "Rank-biserial"]
    names = {"H2": "searcher + C4 minus C0", "H3": "GA + C4 minus L1", "H5": "metaphor minus classical"}
    brows = []
    for k in ("H2", "H3", "H5"):
        e, t = br[k]["effect"], br[k]["two_sided"]
        brows.append([f"{k}: {names[k]}", e["n_datasets"], t["statistic"], t["p"], e["mean"], e["median"],
                      e["ci95_mean"][0], e["ci95_mean"][1], e["rank_biserial"]])
    w.csv("S6b_brier", bh, brows)
    fr = res["secondary"]["friedman"]
    frow = [[fr["n_datasets"], len(fr["selectors"]), fr["statistic"], fr["p"], fr["n_pairs"], fr["n_pairs_rejected_holm"]]]
    fh = ["Datasets", "Selectors", "Friedman statistic", "p", "Pairwise comparisons", "Rejected after Holm"]
    w.csv("S6c_friedman", fh, frow)
    pairs = read_csv(CONF / "tables/friedman_pairwise.csv", comment=True)
    w.csv("S6d_friedman_pairwise", list(pairs[0].keys()),
          [[r[k] for k in pairs[0]] for r in pairs])  # CSV only, values copied as written by the analysis
    w.md("S6_secondary", [
        ("Table S6a. Random search against each guided C0 searcher (two-sided Wilcoxon, Holm within the family).",
         h, [[r[0], r[1], fx(r[2]), fx(r[3]), pv(r[4]), pv(r[5]), f"{r[6]:.2f}", r[7]] for r in rows], []),
        ("Table S6b. Brier-score differences (lower is better; two-sided, descriptive).", bh,
         [[r[0], r[1], f"{r[2]:g}", pv(r[3]), fx(r[4]), fx(r[5]), fx(r[6]), fx(r[7]), f"{r[8]:.2f}"] for r in brows], []),
        ("Table S6c. Friedman test across all selectors at 2,000 evaluations (context only).", fh,
         [[frow[0][0], frow[0][1], f"{frow[0][2]:.2f}", pv(frow[0][3]), frow[0][4], frow[0][5]]],
         [f"The {fr['n_pairs']} pairwise Wilcoxon comparisons are in S6d_friedman_pairwise.csv (CSV only)."])])


# ---------------- S7 size and stability ----------------

def s7(w):
    per = read_csv(CONF / "tables/size_stability_per_dataset.csv", comment=True)
    summ = read_csv(CONF / "tables/size_stability_summary.csv", comment=True)
    w.csv("S7_size_stability_per_dataset", list(per[0].keys()), [[r[k] for k in per[0]] for r in per])
    w.csv("S7_size_stability_summary", list(summ[0].keys()), [[r[k] for k in summ[0]] for r in summ])
    f = lambda v, n: "-" if v == "" else f"{float(v):.{n}f}"
    w.md("S7_size_stability", [
        ("Table S7a. Subset size and stability per configuration (mean over 31 datasets).",
         ["Configuration", "Datasets", "Mean size", "Mean stability", "Datasets with undefined stability"],
         [[r["config"], r["n_datasets"], f(r["mean_size"], 1), f(r["mean_stability"], 3), r["n_undefined_stability"]]
          for r in summ], []),
        ("Table S7b. Per dataset and configuration.",
         ["Dataset", "Configuration", "Seeds", "Folds", "d", "Mean size", "Stability"],
         [[r["dataset"], r["config"], r["n_seeds"], r["n_folds"], r["d"], f(r["mean_size"], 1), f(r["stability"], 3)]
          for r in per],
         ["Nogueira et al. (2018) across outer folds per dataset, configuration and seed, averaged over seeds; "
          "undefined values left out. Source: results/analysis/confirmation/tables."])])


# ---------------- S8 control summary ----------------

def s8(w):
    auc = read_csv(CONF / "tables/per_dataset_auc_2000.csv", comment=True)
    gap = read_csv(CONF / "tables/per_dataset_gap_2000.csv", comment=True)
    out = {}
    for ctrl in ("C0", "C1", "C2", "C3", "C4"):
        per_auc = [sum(float(r[f"{s}__{ctrl}"]) for s in CTRL) / len(CTRL) for r in auc]
        per_gap = [sum(float(r[f"{s}__{ctrl}"]) for s in CTRL) / len(CTRL) for r in gap]
        out[ctrl] = (per_auc, per_gap)
    rows = []
    mean = lambda v: sum(v) / len(v)
    for ctrl in ("C0", "C1", "C2", "C3", "C4"):
        a, g = out[ctrl]
        a0, g0 = out["C0"]
        rows.append([ctrl, len(a), mean(g), mean(a), mean([x - y for x, y in zip(g, g0)]),
                     mean([x - y for x, y in zip(a, a0)])])
    h = ["Control", "Datasets", "Mean optimism gap", "Mean outer AUC", "Gap minus C0", "AUC minus C0"]
    w.csv("S8_control_summary", h, rows)
    labels = {"C0": "C0 (no control)", "C1": "C1 early stopping (P = 5)", "C2": "C2 variance penalty (lambda = 1)",
              "C3": "C3 flatness (r = 8)", "C4": "C4 knee"}
    w.md("S8_control_summary", [("Table S8. Controls averaged over GA, BPSO-S, DE-S and GWO-S at 2,000 evaluations.",
                                 h, [[labels[r[0]], r[1], fx(r[2]), fx(r[3]), f"{r[4]:+.4f}", f"{r[5]:+.4f}"]
                                     for r in rows],
                                 ["For each dataset, the mean over the four searchers of the per-dataset mean "
                                  "(outer folds x seeds); then the mean over the 31 confirmation datasets. "
                                  "Differences are means over datasets of the per-dataset difference from C0. "
                                  "Computed from the full-precision per_dataset_{gap,auc}_2000.csv tables of the "
                                  "frozen analysis. Descriptive; no test."])])


# ---------------- S9 development (exploratory) ----------------

def s9(w):
    ch = json.loads((REPO / "results/stage2/stage2_choices.json").read_text(encoding="utf-8"))
    r1 = []
    for ctrl, rule in ch["rule1_hyperparameters"].items():
        for c in rule["candidates"]:
            r1.append([ctrl, c["value"], c["score"], "yes" if c["value"] == rule["chosen"] else ""])
    r2 = [[c["control"], c["value"], c["score"], "yes" if c["control"] == ch["choices"]["primary_control"] else ""]
          for c in ch["rule2_primary_control"]["candidates"]]
    pw = ch["choices"]["primary_wrapper"]
    r3 = sorted([[c["searcher"], c["control"], c["value"], c["score"],
                  "yes" if (c["searcher"], c["control"]) == (pw["searcher"], pw["control"]) else ""]
                 for c in ch["rule3_primary_wrapper"]["candidates"]], key=lambda r: -r[3])
    w.csv("S9a_stage2_hyperparameters", ["Control", "Value", "Score (mean outer AUC)", "Chosen"], r1, EXPLORATORY)
    w.csv("S9b_stage2_primary_control", ["Control", "Value", "Score (mean outer AUC)", "Chosen"], r2, EXPLORATORY)
    w.csv("S9c_stage2_primary_wrapper", ["Searcher", "Control", "Value", "Score (mean outer AUC)", "Chosen"], r3,
          EXPLORATORY)
    dev = json.loads((DEV / "results.json").read_text(encoding="utf-8"))
    assert dev["header"] == EXPLORATORY
    c = dev["confirmatory"]
    rows = []
    for h in ("H1", "H2", "H3", "H4", "H5"):
        e = c[h].get("effect", {})
        rows.append([h, e.get("n_datasets", c[h].get("outer_auc", {}).get("n_datasets")), c[h]["p_raw"],
                     c[h]["p_holm"], e.get("mean"), (e.get("ci95_mean") or [None, None])[0],
                     (e.get("ci95_mean") or [None, None])[1], e.get("rank_biserial"), c[h]["wording"]])
    hh = ["Hypothesis", "Datasets", "p raw", "p Holm", "Mean d", "CI low", "CI high", "Rank-biserial", "Wording"]
    w.csv("S9d_development_dryrun", hh, rows, EXPLORATORY)
    w.md("S9_development", [
        (f"{EXPLORATORY}. Table S9a. Stage-2 hyperparameter candidates (12 development datasets).",
         ["Control", "Value", "Score", "Chosen"], [[r[0], r[1], f"{r[2]:.4f}", r[3]] for r in r1], []),
        (f"{EXPLORATORY}. Table S9b. Primary-control candidates.", ["Control", "Value", "Score", "Chosen"],
         [[r[0], "-" if r[1] is None else r[1], f"{r[2]:.4f}", r[3]] for r in r2], []),
        (f"{EXPLORATORY}. Table S9c. Primary-wrapper candidates (ranked).", ["Searcher", "Control", "Value", "Score",
                                                                            "Chosen"],
         [[r[0], r[1], "-" if r[2] is None else r[2], f"{r[3]:.4f}", r[4]] for r in r3],
         ["Source: results/stage2/stage2_choices.json (development data; per-dataset scores are in that file)."]),
        (f"{EXPLORATORY}. Table S9d. H1-H5 in the development dry run (configurations missing by design).", hh,
         [[r[0], r[1], pv(r[2]), pv(r[3]), fx(r[4]), fx(r[5]), fx(r[6]), "-" if r[7] is None else f"{r[7]:.2f}", r[8]]
          for r in rows],
         ["Source: results/analysis/development_dryrun/results.json. The development plan ran 4 of the 12 C0 "
          "searchers, so H1 and H5 use fewer configurations than the confirmatory analysis."])])


# ---------------- S10 data health ----------------

def s10(w):
    blocks, all_rows, empty_rows, c1_rows, plan_rows = [], [], [], [], []
    for name in ("development", "confirmation"):
        path = HEALTH / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing: run src/save_health.py first")
        h = json.loads(path.read_text(encoding="utf-8"))
        for p in h["plans"]:
            plan_rows.append([name, p["plan"], p["units"], p["missing"], json.dumps(p["status"], sort_keys=True),
                              ", ".join(sorted(p["commits"])), p["runs_before_stage2"],
                              "; ".join(f"{t['runs']} runs {t['setting']}" for t in p["threads"])])
            for cfg, v in sorted(p["warnings"].items()):
                all_rows.append([name, p["plan"], cfg, v["total"], v["runs_with_warnings"], v["runs"]])
            for e in p["empty_subset"]:
                empty_rows.append([name, p["plan"], e["configuration"], e["runs"], e["datasets"],
                                   ", ".join(f"{d} ({c})" for d, c in sorted(e["per_dataset"].items()))])
            if "c1" in p:
                c1_rows.append([name, p["plan"], p["c1"]["runs"], p["c1"]["stopped_early"],
                                p["c1"]["ended_before_first_checkpoint"]])
    w.csv("S10a_plans", ["Health report", "Plan", "Units", "Missing", "Status", "Commits",
                         "Runs before stage-2 commit", "Threads"], plan_rows)
    w.csv("S10b_convergence_warnings", ["Health report", "Plan", "Configuration", "Warnings", "Runs with warnings",
                                        "Runs"], all_rows)
    w.csv("S10c_empty_subsets", ["Health report", "Plan", "Configuration", "Runs", "Datasets", "Per dataset"],
          empty_rows)
    w.csv("S10d_c1_stops", ["Health report", "Plan", "C1 runs", "Stopped early", "Ended before 250 evaluations"],
          c1_rows)
    warn_md = [r for r in all_rows if r[3] > 0]
    w.md("S10_data_health", [
        ("Table S10a. Plans checked.", ["Report", "Plan", "Units", "Missing", "Status", "Commits",
                                        "Before stage 2", "Threads"], plan_rows, []),
        ("Table S10b. Convergence warnings (configurations with any warning; all configurations in the CSV).",
         ["Report", "Plan", "Configuration", "Warnings", "Runs with warnings", "Runs"], warn_md,
         [f"{sum(1 for r in all_rows if r[3] == 0)} of {len(all_rows)} plan x configuration cells had no warning."]),
        ("Table S10c. Baseline runs that returned an empty subset.",
         ["Report", "Plan", "Configuration", "Runs", "Datasets", "Per dataset"], empty_rows, []),
        ("Table S10d. C1 early stops.", ["Report", "Plan", "C1 runs", "Stopped early", "Ended before 250 evaluations"],
         c1_rows,
         ["Source: results/health/{development,confirmation}.json, written by src/save_health.py from the unchanged "
          "src/run_health.py (no AUC, Brier score or subset read). The development report covers the development "
          "plan (elastic net at max_iter 2,000) and its elastic-net rerun dev_enet_fix (max_iter 10,000); the "
          "analysis uses the rerun."])])


def main(argv=None):
    global OUT, HEALTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="for testing only; default results/paper/supplementary")
    ap.add_argument("--health-dir", default=str(HEALTH), help="for testing only; default results/health")
    args = ap.parse_args(argv)
    OUT, HEALTH = Path(args.out), Path(args.health_dir)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    res = json.loads((CONF / "results.json").read_text(encoding="utf-8"))
    assert res["header"] == "CONFIRMATORY (confirmation data)"
    sizes = estimate_compute.dataset_sizes()
    w = Writer()
    s1(w)
    s2(w)
    s3(w)
    s4(w, res, sizes)
    s5(w, res)
    s6(w, res)
    s7(w)
    s8(w)
    s9(w)
    s10(w)
    rows = [[p.relative_to(OUT).as_posix(), sha(p), p.stat().st_size] for p in sorted(w.files)]
    with open(OUT / "supplementary_manifest.csv", "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["path", "sha256", "bytes"])
        cw.writerows(rows)
    print(f"written {len(rows)} files to supplementary/")
    for path, h, b in rows:
        print(f"  {path:<44} {b:>9}  {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
