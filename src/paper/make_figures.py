"""Paper figures and tables (Elsevier format) from the committed confirmatory outputs.

    python src/paper/make_figures.py
    (options --analysis-dir, --results-dir, --plan, --choices, --out exist only so that the script can be
    tested on synthetic data; the defaults are the committed confirmation outputs)

Reads only:
  - results/analysis/confirmation/results.json and results/analysis/confirmation/tables/*.csv
    (the frozen analysis, code 0b3da97);
  - results/stage2/stage2_choices.json (stage-2 choices) and configs/confirmation.yaml;
  - from the confirmation run files listed in results/MANIFEST.csv, only configuration, dataset,
    timings (cost column) and n_features_raw, n_train, n_test (p and n for log(p/n), exactly as the
    analysis computes them in src/analysis/load.py: p = n_features_raw, n = n_train + n_test).
    As a cross-check, p and n are also taken from data/meta/audit.csv after the data-handling rules
    and the agreement is reported in results/paper/sources.json.
Computes no test and no p-value. It reads values, ranks and orders them, and takes medians and
quartiles of run times. Table 5's mean outer AUC and optimism gap come from tables/anytime.csv for
the 36 search configurations; for the six baselines they are the mean over datasets of
tables/per_dataset_{auc,gap}_2000.csv, the same definition (checked here: recomputing it for the
search configurations reproduces anytime.csv exactly).

Outputs, deterministic (fixed matplotlib metadata, sorted inputs), in results/paper/:
  figures/fig{1,2,3}.pdf (vector, fonttype 42) and .png (600 dpi); widths 190 mm; sans-serif 8 pt;
  Okabe-Ito colour-blind-safe palette; no titles inside figures.
  tables/table{1..5}.csv (full precision) and .md (rounded); sources.json; paper_manifest.csv.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import estimate_compute  # noqa: E402  (metadata cross-check only)
from fsbench.run import expand, load_plan  # noqa: E402

ANALYSIS = REPO / "results/analysis/confirmation"
RESULTS = REPO / "results"
PLAN = REPO / "configs/confirmation.yaml"
CHOICES = REPO / "results/stage2/stage2_choices.json"
OUT = REPO / "results/paper"
MM = 1 / 25.4
DOUBLE, SINGLE = 190 * MM, 90 * MM
DPI = 600
PDF_META = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}
PNG_META = {"Software": None}
# Okabe-Ito palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
C0 = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V", "gwo_S", "gwo_V",
      "woa_S", "woa_V"]
CTRL = ["ga", "bpso_S", "de_S", "gwo_S"]
BASELINES = ["all_k", "l1", "enet", "ftest", "mrmr", "rf"]
LABEL = {"random": "Random search", "hill_climb": "Hill climbing", "one_plus_one": "(1+1)-EA", "ga": "GA",
         "bpso_S": "BPSO-S", "bpso_V": "BPSO-V", "de_S": "DE-S", "de_V": "DE-V", "gwo_S": "GWO-S",
         "gwo_V": "GWO-V", "woa_S": "WOA-S", "woa_V": "WOA-V"}
STYLE = {"random": (OI["black"], "--", None), "hill_climb": (OI["orange"], "-", "o"),
         "one_plus_one": ("#555555", "-", "s"), "ga": (OI["blue"], "-", "D"),
         "bpso_S": (OI["sky"], "-", "^"), "bpso_V": (OI["sky"], ":", "v"),
         "de_S": (OI["green"], "-", "^"), "de_V": (OI["green"], ":", "v"),
         "gwo_S": (OI["vermillion"], "-", "^"), "gwo_V": (OI["vermillion"], ":", "v"),
         "woa_S": (OI["purple"], "-", "^"), "woa_V": (OI["purple"], ":", "v")}
BASELINE_LABEL = {"all_k": "All K features", "l1": "L1 logistic regression", "enet": "Elastic net",
                  "ftest": "F-test top-k", "mrmr": "mRMR top-k", "rf": "Random-forest importance top-k"}
CTRL_LABEL = {"C0": "none", "C1": "C1 early stopping", "C2": "C2 variance penalty", "C3": "C3 flatness",
              "C4": "C4 knee"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(name):
    """Analysis CSV (first line is the data header comment)."""
    text = (ANALYSIS / "tables" / name).read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("# "), name
    return list(csv.DictReader(io.StringIO("\n".join(text[1:]))))


def style():
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "paper",
                         "savefig.dpi": DPI, "figure.dpi": 100})


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", metadata=PDF_META)
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=DPI, metadata=PNG_META)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return path


def write_md(path, title, header, rows, notes=()):
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    lines += [""] + [f"{n}" for n in notes]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")
    return path


def p2(p):
    """p to 2 significant figures, keeping trailing zeros (0.5954 -> 0.60, 4.66e-10 -> 4.7e-10)."""
    if p is None:
        return "-"
    s = f"{p:#.2g}"  # '#' keeps trailing zeros
    return s.rstrip(".") if s.endswith(".") else s


def f4(x):
    return "-" if x is None else f"{x:.4f}"


# ---------------- inputs ----------------

def load_runs():
    """Per run: configuration, dataset, method-phase seconds; per dataset: p and n as in the analysis."""
    with open(RESULTS / "MANIFEST.csv", encoding="utf-8", newline="") as f:
        latest = {r["path"]: r for r in csv.DictReader(f)}
    times = defaultdict(list)
    pn = {}
    for u in sorted(expand(load_plan(PLAN)), key=lambda u: u["path"]):
        row = latest[u["path"]]
        data = (RESULTS / u["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"] or row["status"] != "ok":
            raise SystemExit(f"run file does not match the manifest: {u['path']}")
        r = json.loads(gzip.decompress(data))
        t = r["timings"]
        times[r["configuration"]].append(t.get("search_s", t.get("selection_s")))
        key = (r["n_features_raw"], r["n_train"] + r["n_test"])
        if pn.setdefault(r["dataset"], key) != key:
            raise SystemExit(f"p and n differ between runs of {r['dataset']}")
    return times, pn


def config_key(name):
    """Analysis label (e.g. ga__C1) from the run configuration (e.g. ga__C1_P5)."""
    s, c = name.split("__")
    return f"{s}__{c.split('_')[0]}" if not s == "baseline" else name


# ---------------- figures ----------------

def fig1(res, n_core):
    style()
    fig, ax = plt.subplots(figsize=(DOUBLE, 62 * MM))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    n_ds = len(res["inputs"]["datasets"])
    n_s2 = res["inputs"]["n_run_files"] - n_core
    boxes = [
        "Audit and inclusion\n50 radMLBench datasets,\n43 included",
        "Sealed dataset-level split\n12 development\n31 confirmation",
        "Stage 1 frozen\n(commit 8fa82c7)\n+ amendment 1 (H5)",
        "Code verification\nOneMax, MEALPY checks,\nstructural bias, budget",
        "Pilot and development runs\n480 pilot + 2,880 development\n+ 60 elastic-net rerun",
        "Stage 2 frozen\n(commit 862a7ea)\nconfirmation unsealed",
        f"Confirmation run\n{n_ds} datasets\n{n_core:,} + {n_s2:,} (S2) runs",
        "Confirmatory analysis\n(code 0b3da97, frozen\nbefore unsealing)\nH1-H5, S1, S2",
    ]
    w, h = 0.86, 0.66
    centres = [(0.5 + i, 1.5) for i in range(4)] + [(0.5 + i, 0.5) for i in range(4)]
    for (x, y), text in zip(centres, boxes):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    linewidth=0.6, edgecolor=OI["black"], facecolor="#F2F2F2"))
        ax.text(x, y, text, ha="center", va="center", fontsize=7, linespacing=1.25)
    arrow = dict(arrowstyle="-|>", mutation_scale=7, linewidth=0.6, color=OI["black"])
    for i in (0, 1, 2, 4, 5, 6):
        (x0, y0), (x1, y1) = centres[i], centres[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + w / 2, y0), (x1 - w / 2, y1), **arrow))
    # row 1 end -> row 2 start: down, left, down
    (x3, y3), (x4, y4) = centres[3], centres[4]
    ymid = (y3 - h / 2 + y4 + h / 2) / 2
    ax.plot([x3, x3, x4, x4], [y3 - h / 2, ymid, ymid, ymid], color=OI["black"], linewidth=0.6)
    ax.add_patch(FancyArrowPatch((x4, ymid), (x4, y4 + h / 2), **arrow))
    fig.tight_layout(pad=0.2)
    return save(fig, OUT / "figures/fig1")


def fig2(any_rows, h1_d, logpn):
    style()
    fig = plt.figure(figsize=(DOUBLE, 110 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.08, wspace=0.28)
    ax_gap = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[1, 0], sharex=ax_gap)
    ax_b = fig.add_subplot(gs[:, 1])
    by = {(r["config"], int(r["checkpoint"])): r for r in any_rows}
    cps = [250, 500, 1000, 2000]
    for s in C0:
        colour, ls, marker = STYLE[s]
        gap = [float(by[(f"{s}__C0", c)]["mean_optimism_gap"]) for c in cps]
        auc = [float(by[(f"{s}__C0", c)]["mean_outer_auc"]) for c in cps]
        kw = dict(color=colour, linestyle=ls, marker=marker, markersize=3, linewidth=1.3 if s == "random" else 0.9,
                  label=LABEL[s], zorder=3 if s == "random" else 2)
        ax_gap.plot(cps, gap, **kw)
        ax_auc.plot(cps, auc, **kw)
    ax_auc.set_xscale("log")
    ax_auc.minorticks_off()
    ax_auc.set_xticks(cps)
    ax_auc.set_xticklabels([str(c) for c in cps])
    plt.setp(ax_gap.get_xticklabels(), visible=False)
    ax_auc.set_xlabel("Evaluations (log scale)")
    ax_gap.set_ylabel("Mean optimism gap (AUC units)")
    ax_auc.set_ylabel("Mean outer AUC")
    # same y-axis span in both panels (same AUC scale), each centred on its own data
    gap_vals = [float(by[(f"{s}__C0", c)]["mean_optimism_gap"]) for s in C0 for c in cps]
    auc_vals = [float(by[(f"{s}__C0", c)]["mean_outer_auc"]) for s in C0 for c in cps]
    span = 1.1 * max(max(gap_vals) - min(gap_vals), max(auc_vals) - min(auc_vals))
    for ax, vals in ((ax_gap, gap_vals), (ax_auc, auc_vals)):
        mid = (max(vals) + min(vals)) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
    # one legend above panel (a), outside the plotting area
    ax_gap.legend(ncol=4, frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
                  mode="expand", handlelength=2.2, borderaxespad=0.0)
    ds = sorted(h1_d)
    ax_b.scatter([logpn[d] for d in ds], [h1_d[d] for d in ds], s=12, color=OI["blue"], edgecolor="none")
    ax_b.axhline(0, color="0.6", linewidth=0.5)
    ax_b.set_xlabel("log(p/n) (natural logarithm)")
    ax_b.set_ylabel("Per-dataset mean optimism gap, 12 C0 configurations (AUC units)")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.08)
    # panel labels in the top margin, clear of tick labels, axis labels and the legend
    for ax, lab in ((ax_gap, "(a)"), (ax_b, "(b)")):
        x0 = ax.get_position().x0
        fig.text(max(x0 - 0.075, 0.003), 0.985, lab, fontsize=8, fontweight="bold", va="top", ha="left")
    return save(fig, OUT / "figures/fig2")


def fig3(h2_d, h3_d, s2_d, logpn):
    style()
    order = sorted(h2_d, key=lambda d: (logpn[d], d))
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 120 * MM), sharey=True)
    a, b = axes
    a.axvline(0, color="0.4", linewidth=0.6)
    a.scatter([h2_d[d] for d in order], y, s=12, color=OI["blue"], edgecolor="none", zorder=3)
    a.set_xlabel("Difference in outer AUC, searcher + C4 minus C0\n(mean of GA, BPSO-S, DE-S, GWO-S)")
    b.axvline(0, color="0.4", linewidth=0.6)
    for i, m in enumerate((-0.02, 0.02)):
        b.axvline(m, color="0.5", linewidth=0.7, linestyle="--",
                  label="Equivalence margin (+/-0.02)" if i == 0 else None)
    b.scatter([h3_d[d] for d in order], y, s=12, color=OI["blue"], edgecolor="none", zorder=3,
              label="Pre-filtered pool (H3)")
    s2y = [i for i, d in enumerate(order) if d in s2_d]
    b.scatter([s2_d[order[i]] for i in s2y], s2y, s=16, facecolor="none", edgecolor=OI["orange"], linewidth=0.8,
              zorder=4, label="No pre-filter (S2)")
    b.set_xlabel("Difference in outer AUC,\nGA + C4 minus L1 logistic regression")
    b.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.5, borderaxespad=0.2,
             handlelength=2.0)
    for ax in (a, b):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    a.set_yticks(y)
    a.set_yticklabels(order, fontsize=6)
    a.set_ylim(-0.7, len(order) - 0.3)
    a.set_ylabel("Confirmation dataset (ordered by log(p/n), lowest at bottom)")
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        ax.text(0.0, 1.01, lab, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.grid(axis="y", color="0.92", linewidth=0.4)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.92, bottom=0.11, wspace=0.08)
    return save(fig, OUT / "figures/fig3")


# ---------------- tables ----------------

def table1(plan):
    settings = {  # PREREG_STAGE1.md, section 5 (verbatim settings)
        "random": "2,000 masks; subset size drawn uniformly from 1 to K, then features drawn uniformly",
        "hill_climb": "One random bit flipped per step; move accepted if fitness is greater than or equal",
        "one_plus_one": "Standard bit mutation with rate 1/K; offspring accepted if fitness is greater than or equal",
        "ga": "Population 20, 100 generations; binary tournament; uniform crossover with probability 0.9; "
              "bit-flip mutation 1/K; elitism of 1",
        "bpso": "20 particles, 100 iterations; inertia 0.9 to 0.4 linearly; c1 = c2 = 2; velocity clamp 6",
        "de": "DE/rand/1/bin; population 20, 100 generations; F = 0.5; CR = 0.9; continuous positions in [-6, 6]",
        "gwo": "20 wolves, 100 iterations; a decreases linearly from 2 to 0",
        "woa": "20 whales, 100 iterations; a decreases linearly from 2 to 0; spiral constant b = 1",
    }
    names = {"random": "Random subset search", "hill_climb": "Bit-flip hill climbing", "one_plus_one": "(1+1)-EA",
             "ga": "Genetic algorithm", "bpso": "Binary PSO", "de": "Differential evolution",
             "gwo": "Grey wolf optimizer", "woa": "Whale optimization algorithm"}
    c0 = plan["c0_searchers"]
    rows = []
    for fam in ["random", "hill_climb", "one_plus_one", "ga", "bpso", "de", "gwo", "woa"]:
        confs = [s for s in c0 if s == fam or s.startswith(fam + "_")]
        transfer = "S-shaped (sigmoid) and V-shaped (|tanh|)" if len(confs) == 2 else "-"
        rows.append([names[fam], ", ".join(LABEL[s] for s in confs), transfer, settings[fam]])
    common = (f"All searchers: binary mask over the pool of K = min(200, p) features; budget {plan['budget']:,} "
              f"evaluations; checkpoints 250, 500, 1,000, 2,000; {len(plan['repeats'])} x {len(plan['folds'])} "
              f"outer folds; {len(plan['search_seed_indices'])} search seeds per outer fold; initial masks drawn "
              "bit by bit with probability 0.5 (configs/confirmation.yaml; PREREG_STAGE1 sections 4 and 5).")
    header = ["Searcher", "Configurations", "Transfer functions", "Settings"]
    return header, rows, [common]


def table2(choices, plan):
    r1 = choices["rule1_hyperparameters"]
    rows, md = [], []

    def margin(rule):
        cands = sorted(rule["candidates"], key=lambda c: -c["score"])
        chosen = rule["chosen"]
        best = next(c for c in cands if c["value"] == chosen)
        runner = next(c for c in cands if c["value"] != chosen)
        return best["score"] - runner["score"], runner["value"]

    ctrl_plan = plan["controls"]
    spec = [
        ("C1", "Early stopping on a sealed 20% selection fold",
         "1 selection-fold scoring after every 100 search evaluations (101 per round)", "P", [3, 5, 10]),
        ("C2", "Variance-penalized fitness (mean - lambda x SD over a 3 x 3 split)",
         "1 per scored subset (9 model fits)", "lambda", [0.5, 1.0, 2.0]),
        ("C3", "Flatness fitness (mean over the subset and r one-bit neighbours)",
         "1 + r per scored subset", "r", [2, 4, 8]),
    ]
    for c, desc, cost, par, grid in spec:
        m, runner = margin(r1[c])
        chosen = r1[c]["chosen"]
        assert float(chosen) == float(ctrl_plan[c])
        rows.append([c, desc, cost, par, "{" + ", ".join(f"{g:g}" for g in grid) + "}", chosen, runner, m])
    r2 = {x["control"]: x["score"] for x in choices["rule2_primary_control"]["candidates"]}
    runner_c = max((k for k in r2 if k != "C4"), key=lambda k: r2[k])
    rows.append(["C4", "Knee of the archive's (inner fitness, size) front, from the C0 run",
                 "0 (computed from the C0 archive)", "-", "-",
                 "primary control (H2)" if choices["choices"]["primary_control"] == "C4" else "-",
                 runner_c, r2["C4"] - r2[runner_c]])
    header = ["Control", "Description", "Cost in evaluations", "Parameter", "Grid", "Chosen", "Runner-up",
              "Development margin (mean outer AUC)"]
    notes = ["Development margin: score of the chosen value minus that of the runner-up (results/stage2/"
             "stage2_choices.json; 12 development datasets, 5 runs per cell). For C4 the margin is over the "
             "best other control at its chosen value (primary control for H2)."]
    return header, rows, notes


def table3(res):
    c = res["confirmatory"]
    rows = []
    tests = {"H1": "Wilcoxon signed-rank, one-sided (d > 0)", "H2": "Wilcoxon signed-rank, one-sided (d > 0)",
             "H3": "TOST, Wilcoxon form, margin 0.02", "H4": "Likelihood-ratio test of searcher x control (mixed model)",
             "H5": "Wilcoxon signed-rank, one-sided (d - 0.01 < 0)"}
    dlabel = {"H1": "optimism gap, mean of 12 C0", "H2": "searcher + C4 minus C0 (4 searchers)",
              "H3": "GA + C4 minus L1", "H4": "-", "H5": "GWO/WOA minus GA/BPSO/DE (C0)"}
    for h in ("H1", "H2", "H3", "H4", "H5"):
        r = c[h]
        if h == "H4":
            m = r["outer_auc"]
            stat = f"LR = {m['lr_statistic']:.2f}, df = {m['df']}"
            e = {"mean": None, "ci95_mean": [None, None], "rank_biserial": None, "n_datasets": m["n_datasets"]}
        elif h == "H3":
            t = r["tost"]
            stat = f"W_low = {t['lower']['statistic']:g}, W_up = {t['upper']['statistic']:g}"
            e = r["effect"]
        else:
            stat = f"W = {r['test']['statistic']:g}"
            e = r["effect"]
        rows.append([h, tests[h], dlabel[h], e["n_datasets"], stat, r["p_raw"], r["p_holm"], e["mean"],
                     e["ci95_mean"][0], e["ci95_mean"][1], e["rank_biserial"], r["wording"]])
    header = ["Hypothesis", "Test", "d", "Datasets", "Statistic", "p raw", "p Holm", "Mean d", "CI low", "CI high",
              "Rank-biserial", "Wording"]
    md = [[r[0], r[1], r[2], r[3], r[4], p2(r[5]), p2(r[6]),
           "-" if r[7] is None else f"{f4(r[7])} [{f4(r[8])}, {f4(r[9])}]", "-" if r[10] is None else f"{r[10]:.2f}",
           r[11]] for r in rows]
    md_header = ["Hypothesis", "Test", "d", "Datasets", "Statistic", "p raw", "p Holm", "Mean d [95% CI]",
                 "Rank-biserial", "Wording"]
    notes = ["Holm correction across H1-H5 at alpha = 0.05 (H3 enters with p_TOST, H4 with the likelihood-ratio p). "
             "95% CI: percentile bootstrap of the mean over 31 confirmation datasets (10,000 resamples). "
             "Exact Wilcoxon distribution in every test. Source: results/analysis/confirmation/results.json."]
    return header, rows, md_header, md, notes


def table4(res):
    h4 = res["confirmatory"]["H4"]
    a, g = h4["outer_auc"], h4["optimism_gap_descriptive"]
    pa, pg = a["fe_params_full"], g["fe_params_full"]
    rows = []
    for c in ("C1", "C2", "C3", "C4"):
        k = f"C(control)[T.{c}]"
        rows.append([f"{CTRL_LABEL[c]} vs C0 (main effect)", pa[k], pg[k]])
    for s in ("de_S", "ga", "gwo_S"):
        for c in ("C1", "C2", "C3", "C4"):
            k = f"C(searcher)[T.{s}]:C(control)[T.{c}]"
            rows.append([f"Interaction {LABEL[s]} x {c}", pa[k], pg[k]])
    rows.append(["log(evaluations)", pa["log_budget"], pg["log_budget"]])
    rows.append(["log(p/n)", pa["log_pn"], pg["log_pn"]])
    lrt = [["Interaction LRT: statistic", a["lr_statistic"], g["lr_statistic"]],
           ["Interaction LRT: df", a["df"], g["df"]],
           ["Interaction LRT: p", a["p"], g["p"]],
           ["Rows (runs x checkpoints)", a["n_rows"], g["n_rows"]]]
    header = ["Term", "Outer AUC", "Optimism gap (descriptive)"]
    md = [[r[0], f"{r[1]:+.4f}", f"{r[2]:+.4f}"] for r in rows]
    md += [[lrt[0][0], f"{lrt[0][1]:.2f}", f"{lrt[0][2]:.2f}"], [lrt[1][0], lrt[1][1], lrt[1][2]],
           [lrt[2][0], p2(lrt[2][1]), p2(lrt[2][2])], [lrt[3][0], f"{lrt[3][1]:,}", f"{lrt[3][2]:,}"]]
    notes = ["Linear mixed model fitted by maximum likelihood on run-level values at 250, 500, 1,000 and 2,000 "
             "evaluations for GA, BPSO-S, DE-S and GWO-S under C0-C4; random intercept for dataset and a variance "
             "component for outer partition within dataset. Treatment coding: control main effects are for the "
             "reference searcher BPSO-S; for another searcher the control effect is the main effect plus its "
             "interaction term. Every fit converged with lbfgs; each logged 'The MLE may be on the boundary of the "
             "parameter space.' H4 is the outer-AUC test; the optimism-gap model is descriptive. "
             "Source: results/analysis/confirmation/results.json."]
    return header, rows + lrt, md, notes


def table5(times, choices_plan):
    anyt = {(r["config"], int(r["checkpoint"])): r for r in read_table("anytime.csv")}
    auc = read_table("per_dataset_auc_2000.csv")
    gap = read_table("per_dataset_gap_2000.csv")
    stab = {r["config"]: r for r in read_table("size_stability_summary.csv")}

    def col_mean(rows, cfg):
        return float(np.mean([float(r[cfg]) for r in rows]))

    run_times = defaultdict(list)
    for cfg, ts in times.items():
        run_times[config_key(cfg)].extend(ts)
    configs = ([f"{s}__C0" for s in C0] + [f"{s}__{c}" for c in ("C1", "C2", "C3") for s in CTRL]
               + [f"{s}__C4" for s in C0] + [f"baseline__{b}" for b in BASELINES])
    rows = []
    for cfg in configs:
        if cfg.startswith("baseline__"):
            m_auc, m_gap = col_mean(auc, cfg), col_mean(gap, cfg)
        else:
            m_auc = float(anyt[(cfg, 2000)]["mean_outer_auc"])
            m_gap = float(anyt[(cfg, 2000)]["mean_optimism_gap"])
            # same definition: the mean over datasets of the per-dataset table reproduces anytime.csv
            assert math.isclose(m_auc, col_mean(auc, cfg), rel_tol=0, abs_tol=1e-12), cfg
            assert math.isclose(m_gap, col_mean(gap, cfg), rel_tol=0, abs_tol=1e-12), cfg
        s, c = cfg.split("__")
        src = f"{s}__C0" if c == "C4" else cfg  # C4 is computed inside the C0 run
        ts = np.array(run_times[src], dtype=float)
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        name = (BASELINE_LABEL[c] if s == "baseline" else
                f"{LABEL[s]}, {CTRL_LABEL[c] if c != 'C0' else 'C0 (no control)'}")
        st = stab[cfg]
        rows.append([cfg, name, m_auc, m_gap, float(st["mean_size"]),
                     None if st["mean_stability"] == "" else float(st["mean_stability"]),
                     int(st["n_undefined_stability"]), float(med), float(q1), float(q3), len(ts),
                     "shared with C0" if c == "C4" else ""])
    header = ["Configuration", "Name", "Mean outer AUC", "Mean optimism gap", "Mean subset size",
              "Mean stability", "Datasets with undefined stability", "Median CPU s per run", "CPU s Q1",
              "CPU s Q3", "Runs timed", "Cost note"]
    md = [[r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.1f}", "-" if r[5] is None else f"{r[5]:.3f}"
           + (f" ({r[6]} undef.)" if r[6] else ""), f"{r[7]:.1f} [{r[8]:.1f}, {r[9]:.1f}]" + (" a" if r[11] else "")]
          for r in rows]
    md_header = ["Configuration", "Mean outer AUC", "Mean optimism gap", "Mean subset size", "Mean stability",
                 "CPU s per run, median [IQR]"]
    notes = ["At 2,000 evaluations; means over the 31 confirmation datasets of per-dataset means (outer folds x "
             "seeds). Stability: Nogueira et al. (2018), undefined values left out. CPU seconds: the method phase of "
             "each run (search for searchers, selection for baselines), single-threaded; median and interquartile "
             "range over all runs of the configuration, pooled over datasets, folds and seeds. "
             "a: shared with C0 - C4 adds no evaluations and is computed inside the C0 run, so its row shows the C0 "
             "run time. Sources: results/analysis/confirmation/tables (anytime.csv; per_dataset_auc_2000.csv and "
             "per_dataset_gap_2000.csv for the baselines; size_stability_summary.csv) and the timings of the "
             "confirmation run files."]
    return header, rows, md_header, md, notes


def main(argv=None):
    global ANALYSIS, RESULTS, PLAN, CHOICES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=str(ANALYSIS))
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--choices", default=str(CHOICES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    ANALYSIS, RESULTS, PLAN, CHOICES, OUT = (Path(args.analysis_dir), Path(args.results_dir), Path(args.plan),
                                              Path(args.choices), Path(args.out))
    res = json.loads((ANALYSIS / "results.json").read_bytes())
    if res["group"] == "confirmation":
        assert res["header"] == "CONFIRMATORY (confirmation data)"
        assert res["code"]["analysis_code_sha256"] == "b68f3b7181ff89af271022ad67b837be0c112597d547c30f96a7a17d0d396d58"
    choices = json.loads(CHOICES.read_bytes())
    plan = load_plan(PLAN)
    times, pn = load_runs()

    sizes = estimate_compute.dataset_sizes()
    meta_pn = {d: (p, n) for d, (g, n, p) in sizes.items() if g == "confirmation"}
    agree = {d: meta_pn.get(d) == pn[d] for d in pn}
    logpn = {d: math.log(p / n) for d, (p, n) in pn.items()}

    files = []
    files += fig1(res, len(expand(plan)))
    h1 = res["confirmatory"]["H1"]["d_per_dataset"]
    files += fig2(read_table("anytime.csv"), h1, logpn)
    files += fig3(res["confirmatory"]["H2"]["d_per_dataset"], res["confirmatory"]["H3"]["d_per_dataset"],
                  res["sensitivity"]["S2"]["d_per_dataset"], logpn)

    tdir = OUT / "tables"
    h, rows, notes = table1(plan)
    files += [write_csv(tdir / "table1.csv", h, rows),
              write_md(tdir / "table1.md", "Table 1. Searchers and settings.", h, rows, notes)]
    h, rows, notes = table2(choices, plan)
    md = [[*r[:7], f"{r[7]:.4f}"] for r in rows]
    files += [write_csv(tdir / "table2.csv", h, rows),
              write_md(tdir / "table2.md", "Table 2. Generalization controls and stage-2 choices.", h, md, notes)]
    h, rows, mh, md, notes = table3(res)
    files += [write_csv(tdir / "table3.csv", h, rows),
              write_md(tdir / "table3.md", "Table 3. Confirmatory tests (31 confirmation datasets).", mh, md, notes)]
    h, rows, md, notes = table4(res)
    files += [write_csv(tdir / "table4.csv", h, rows),
              write_md(tdir / "table4.md", "Table 4. Mixed-model control effects relative to C0 and interaction tests.",
                       h, md, notes)]
    h, rows, mh, md, notes = table5(times, plan)
    files += [write_csv(tdir / "table5.csv", h, rows),
              write_md(tdir / "table5.md", "Table 5. Configurations at 2,000 evaluations.", mh, md, notes)]

    sources = {
        "analysis_results_json_sha256": sha(ANALYSIS / "results.json"),
        "analysis_code_sha256": res["code"]["analysis_code_sha256"],
        "stage2_choices_sha256": sha(CHOICES),
        "confirmation_plan_sha256": sha(PLAN),
        "analysis_tables_sha256": {p.name: sha(p) for p in sorted((ANALYSIS / "tables").glob("*.csv"))},
        "run_files_read": sum(len(v) for v in times.values()),
        "p_n_source": "run files: p = n_features_raw, n = n_train + n_test (as src/analysis/load.py)",
        "p_n_audit_metadata_agrees": all(agree.values()),
        "p_n_disagreements": sorted(d for d, ok in agree.items() if not ok),
        "log_pn_per_dataset": {d: logpn[d] for d in sorted(logpn)},
    }
    sp = OUT / "sources.json"
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    files.append(sp)
    mrows = [[p.relative_to(OUT).as_posix(), sha(p), p.stat().st_size] for p in sorted(files)]
    write_csv(OUT / "paper_manifest.csv", ["path", "sha256", "bytes"], mrows)
    print(f"written {len(files)} files to {OUT.name}/; p and n agree with audit metadata: {all(agree.values())}")
    for path, h_, b in mrows:
        print(f"  {path:<26} {b:>9}  {h_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
