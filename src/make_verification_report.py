"""Build docs/VERIFICATION.md (supplement) from the raw result files in results/verification/.

Every number in the document is computed here from the raw JSON files written by the tests
(tests/conftest.py, save_result), never from printed or rounded values.

Usage (from the repository root, with .venv activated):
    python src/make_verification_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "verification"
OUT = REPO_ROOT / "docs" / "VERIFICATION.md"

ORDER = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V",
         "gwo_S", "gwo_V", "woa_S", "woa_V"]
ALPHA = 0.01


def load(prefix):
    out = {}
    for p in sorted(RESULTS_DIR.glob(f"{prefix}_*.json")):
        r = json.loads(p.read_bytes().decode("utf-8"))
        out[p.stem[len(prefix) + 1:]] = r
    return out


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fp(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    sections, metas = [], []

    mealpy = load("mealpy")
    rows = []
    for algo in ["bpso", "de", "gwo", "woa"]:
        for fn in ["sphere", "rastrigin"]:
            r = mealpy.get(f"{algo}_{fn}")
            if not r:
                continue
            d = r["data"]
            metas.append(r["meta"])
            rows.append([algo.upper() if algo != "bpso" else "PSO core (BPSO)", fn, d["mealpy_model"],
                         f"{np.median(d['final_best_ours']):.4g}", f"{np.median(d['final_best_mealpy']):.4g}",
                         fp(d["p_mannwhitney"]), "flag" if d["flag"] else "ok"])
    sections.append(
        "## 1. Continuous cores vs MEALPY 3.0.3\n\n"
        "Sphere (bounds [-100, 100]) and Rastrigin ([-5.12, 5.12]), d = 30, population 20, 100 iterations "
        "including the initial population (2,000 evaluations), 20 seeds per implementation. Final best "
        "values (lower is better); two-sided Mann-Whitney U; a difference with p < 0.01 is flagged.\n\n"
        + (table(["Algorithm", "Function", "MEALPY model", "Median ours", "Median MEALPY", "p", "Result"], rows)
           if rows else "_Not run._"))

    onemax = load("onemax")
    rows = []
    for name in ORDER:
        r = onemax.get(name)
        if not r:
            continue
        d = r["data"]
        metas.append(r["meta"])
        seeds = "; ".join(f"{s['best_250']:.3f} -> {s['best_2000']:.3f}" for s in d["per_seed"])
        rows.append([name, seeds, f"{d['mean_250']:.3f}", f"{d['mean_2000']:.3f}",
                     "yes" if d["all_optimal_at_250"] else "no",
                     "pass" if d["criterion_a"] else "FAIL", "pass" if d["criterion_b"] else "FAIL"])
    sections.append(
        "## 2. OneMax (K = 200, budget 2,000, seeds 0-4)\n\n"
        "Best fitness at 250 -> 2,000 evaluations per seed. Criterion (a): for every seed, best at 2,000 >= "
        "best at 250. Criterion (b): mean best at 2,000 > mean best at 250, or every seed reached the "
        "optimum (1.0) by 250. bpso_S can stagnate (seed 0): once a bit agrees with pbest and gbest its "
        "velocity decays to 0 and sigmoid(0) = 0.5 re-randomizes it, the documented behaviour of the "
        "original binary PSO (Kennedy & Eberhart, 1997).\n\n"
        + (table(["Searcher", "Per seed (250 -> 2,000)", "Mean 250", "Mean 2,000", "All optimal at 250",
                  "(a)", "(b)"], rows) if rows else "_Not run._"))

    bias = load("bias")
    rows = []
    for name in ORDER:
        r = bias.get(name)
        if not r:
            continue
        d = r["data"]
        metas.append(r["meta"])
        o, i = np.array(d["sizes_original"]), np.array(d["sizes_inverted"])
        rows.append([name, f"{o.mean():.1f} ({o.std(ddof=1):.1f})", f"{i.mean():.1f} ({i.std(ddof=1):.1f})",
                     fp(d["p_mannwhitney"]), fp(d["p_wilcoxon_vs_half"]),
                     "FAIL" if d["flag_mannwhitney"] else "pass",
                     "flag" if d["flag_wilcoxon"] else "-"])
    sections.append(
        "## 3. Structural bias (Kononova et al., 2015, adapted to binary masks)\n\n"
        "Pure-noise fitness, deterministic per mask (uniform in [0, 1] from a BLAKE2b hash of the mask bytes "
        "and the seed); K = 200, budget 2,000, 30 seeds. Returned subset size under the original and the "
        "inverted encoding (mean, SD); two-sided Mann-Whitney U between encodings (must be p > 0.01); "
        "one-sample two-sided Wilcoxon of the original sizes against K/2 = 100 (reported; flagged if p <= 0.01).\n\n"
        + (table(["Searcher", "Size original", "Size inverted", "p (MW)", "p (Wilcoxon vs 100)",
                  "Encoding test", "Wilcoxon"], rows) if rows else "_Not run._"))

    det = load("determinism")
    rows = []
    for name in ORDER:
        r = det.get(name)
        if not r:
            continue
        d = r["data"]
        metas.append(r["meta"])
        rows.append([name, "yes" if d["same_seed_identical"] else "NO",
                     "yes" if d["different_seeds_differ"] else "NO"])
    sections.append(
        "## 4. Determinism\n\n"
        "NK landscape (K = 60, k = 4, fixed tables), budget 2,000. Same seed (7) run twice: identical "
        "returned mask. Seeds 1 and 2: different returned masks.\n\n"
        + (table(["Searcher", "Same seed identical", "Different seeds differ"], rows) if rows else "_Not run._"))

    planted = load("planted")
    rows = []
    for name in ORDER:
        r = planted.get(name)
        if not r:
            continue
        d = r["data"]
        metas.append(r["meta"])
        ps = d["per_seed"]
        rec = np.array([s["recovery"] for s in ps])
        rows.append([name, ", ".join(f"{s['informative_selected']}/{d['n_informative']}" for s in ps),
                     f"{rec.mean():.2f}", ", ".join(str(s["subset_size"]) for s in ps),
                     ", ".join(f"{s['inner_auc']:.3f}" for s in ps),
                     f"{np.mean([s['seconds'] for s in ps]):.1f}"])
    sections.append(
        "## 5. Planted subset (reported, not a gate)\n\n"
        "Synthetic logistic data: n = 150, p = 200, 10 informative features (coefficients +-1), data seed "
        "2026. Fitness: the inner fitness of section 4 (mean AUC over a fixed stratified 3-fold split, "
        "L2 logistic regression, C = 1, lbfgs, max_iter = 2,000). Budget 2,000; search seeds 0-2.\n\n"
        + (table(["Searcher", "Informative selected (per seed)", "Mean recovery", "Subset size",
                  "Inner AUC", "Seconds per run"], rows) if rows else "_Not run._"))

    commits = sorted({m["git_commit"][:7] + (" (dirty)" if m["git_dirty"] else "") for m in metas})
    m0 = metas[0] if metas else {}
    header = (
        "# Verification of the searchers and the evaluator (supplement)\n\n"
        "PREREG_STAGE1 section 9, with the operational decisions recorded in docs/DECISIONS.md "
        "(2026-11-10, part 5 implementation decisions). All checks use synthetic problems; no real dataset "
        "is loaded. Generated by src/make_verification_report.py from the raw files in "
        "results/verification/.\n\n"
        f"- Code commit(s) of the runs: {', '.join(commits) if commits else 'n/a'}\n"
        f"- Run dates (UTC): {min(m['utc'] for m in metas) if metas else 'n/a'} to {max(m['utc'] for m in metas) if metas else 'n/a'}\n"
        f"- Python {m0.get('python')}, numpy {m0.get('numpy')}, scipy {m0.get('scipy')}, "
        f"scikit-learn {m0.get('sklearn')}, MEALPY {m0.get('mealpy')}\n"
    )
    text = header + "\n" + "\n\n".join(sections) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} from {len(metas)} result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
