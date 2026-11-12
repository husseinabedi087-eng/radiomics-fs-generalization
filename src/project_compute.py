"""Project the compute of the confirmation core tier from development-run timings (no confirmation data).

    python src/project_compute.py

Reads from the run files under results/runs/development/ (plans "development" and "pilot") ONLY the fields
configuration, n_train, K, n_features_raw, git_commit and timings; never an AUC, Brier score or subset.
Confirmation-dataset sizes (n, p after the data-handling rules) come from data/meta/audit.csv and
data/meta/data_handling_rules.json via estimate_compute.dataset_sizes(); no confirmation dataset is loaded.

Model: per configuration and phase, least squares on the development runs of
    log t = a + b log n_train + c log K.
C0 configurations not run in development (random, hill_climb, one_plus_one, bpso_V, de_V, gwo_V, woa_S,
woa_V) use the ga__C0 model times the pilot ratio (mean time of the configuration / mean time of ga__C0,
per phase). Development n_train reaches at most DEV_NMAX; above it three growth cases are projected:
    flat    time stays at its n_train = DEV_NMAX value (as measured by src/bench_scaling.py),
    fit     the fitted exponent b continues,
    linear  time grows linearly in n_train above DEV_NMAX.

Core tier: 31 confirmation datasets x 10 outer folds; 12 C0 configurations x 3 search seeds (C4 in-run,
its c4_s time counted); C1 (P = 5), C2 (lambda = 1), C3 (r = 8) on ga, bpso_S, de_S, gwo_S x 3 seeds; the
baselines once per fold, random forest with 3 model seeds; K = min(200, p).
S2: ga C0 (C4 from the same runs) x 3 seeds and L1 logistic regression, 10 folds, without the pre-filter
(K = p), on confirmation datasets with p <= 1000; once with the fitted K exponent, once with time scaled
linearly in K from the K = 200 prediction.

Output: printed summary and results/compute/projection.json.
"""

import gzip
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
import estimate_compute as ec  # noqa: E402

OUT = root / "results" / "compute" / "projection.json"
W = 6
C0 = ec.C0
CS = ["ga", "bpso_S", "de_S", "gwo_S"]
BASE = ec.BASELINES
CHOSEN = {"C1": "C1_P5", "C2": "C2_lambda1", "C3": "C3_r8"}
PH = ["main", "standard_inner_auc_s", "outer_scoring_s", "c4_s", "fold_preparation_s"]
MODES = ("flat", "fit", "linear")
FOLDS = 10
commits = defaultdict(set)


def load(sub):
    out = []
    for f in sorted((root / "results/runs/development" / sub).rglob("*.json.gz")):
        r = json.loads(gzip.decompress(f.read_bytes()))
        t = r["timings"]
        commits[sub].add(r["git_commit"])
        out.append((r["configuration"], r["n_train"], r["K"], r["n_features_raw"],
                    {"main": t.get("search_s", t.get("selection_s")), "standard_inner_auc_s": t["standard_inner_auc_s"],
                     "outer_scoring_s": t["outer_scoring_s"], "c4_s": t.get("c4_s", 0.0),
                     "fold_preparation_s": t["fold_preparation_s"]}))
    return out


dev, pilot = load("development"), load("pilot")
print("dev runs", len(dev), "n_train", min(r[1] for r in dev), "-", max(r[1] for r in dev), "K values", sorted({r[2] for r in dev}))


def fit(pts):
    n = np.log([p[0] for p in pts])
    K = np.log([p[1] for p in pts])
    t = np.log([max(p[2], 1e-4) for p in pts])
    A = np.column_stack([np.ones_like(n), n, K])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return coef, float(np.std(t - A @ coef))


models = {}
by = defaultdict(lambda: defaultdict(list))
for cfg, n, K, praw, t in dev:
    for ph in PH:
        by[cfg][ph].append((n, K, t[ph]))
for cfg in by:
    models[cfg] = {ph: fit(by[cfg][ph]) for ph in PH}
# C0 configurations not in the development run: ga__C0 model x pilot ratio of mean time per phase
pil = defaultdict(lambda: defaultdict(list))
for cfg, n, K, praw, t in pilot:
    for ph in PH:
        pil[cfg][ph].append(t[ph])
ratio = {c: {ph: (np.mean(pil[f"{c}__C0"][ph]) / max(np.mean(pil["ga__C0"][ph]), 1e-9)) for ph in PH} for c in C0}

DEV_NMAX = max(r[1] for r in dev)


def pred(cfg, ph, n, K, mode):
    """mode 'flat': time at DEV_NMAX beyond it; 'fit': fitted b beyond the dev range; 'linear': b = 1 above DEV_NMAX."""
    src = cfg if cfg in models else "ga__C0"
    (a, b, c), _ = models[src][ph]
    if mode == "flat" and n > DEV_NMAX:
        t = math.exp(a + b * math.log(DEV_NMAX) + c * math.log(K))
    elif mode == "linear" and n > DEV_NMAX:
        t = math.exp(a + b * math.log(DEV_NMAX) + c * math.log(K)) * (n / DEV_NMAX)
    else:
        t = math.exp(a + b * math.log(n) + c * math.log(K))
    if cfg not in models:
        t *= ratio[cfg.split("__")[0]][ph]
    return t


print("\nDevelopment fit, main phase (t = exp(a) n^b K^c), chosen configurations:")
for cfg in [f"{s}__C0" for s in CS] + [f"{s}__{v}" for v in CHOSEN.values() for s in CS] + [f"baseline__{b}" for b in BASE]:
    (a, b, c), sd = models[cfg]["main"]
    print(f"  {cfg:<20} b {b:5.2f}  c {c:5.2f}  resid SD {sd:4.2f}")
print("C0 configurations from pilot ratio to ga__C0:", {c: round(float(ratio[c]['main']), 2) for c in C0 if f'{c}__C0' not in models})

sizes = ec.dataset_sizes()
conf = {d: (n, p) for d, (g, n, p) in sizes.items() if g == "confirmation"}
units = [(f"{s}__C0", 3) for s in C0] + [(f"{s}__{v}", 3) for v in CHOSEN.values() for s in CS] + \
        [(f"baseline__{b}", 3 if b == "rf" else 1) for b in BASE]

result = {
    "description": "Confirmation core-tier compute projection from development-run timings (src/project_compute.py)",
    "head_at_run": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "run_commits": {k: sorted(v) for k, v in commits.items()},
    "n_runs": {"development": len(dev), "pilot": len(pilot)},
    "dev_n_train_max": int(DEV_NMAX),
    "dev_K_values": sorted(int(k) for k in {r[2] for r in dev}),
    "workers": W, "outer_folds": FOLDS, "search_seeds": 3,
    "chosen_controls": CHOSEN,
    "units_per_fold": [{"configuration": c, "repeats": r} for c, r in units],
    "fit": {cfg: {ph: {"a": float(m[0][0]), "b": float(m[0][1]), "c": float(m[0][2]), "resid_sd": m[1]}
                  for ph, m in models[cfg].items()} for cfg in sorted(models)},
    "pilot_ratio_to_ga_C0": {c: {ph: float(v) for ph, v in ratio[c].items()} for c in C0 if f"{c}__C0" not in models},
    "cases": {},
}

for mode in MODES:
    per_ds, per_ph = {}, defaultdict(float)
    for d, (n, p) in conf.items():
        nt, K = 0.8 * n, min(200, p)
        tot = 0.0
        for cfg, reps in units:
            for ph in PH:
                if ph == "c4_s" and not cfg.endswith("__C0"):
                    continue
                if ph == "fold_preparation_s":
                    continue
                t = pred(cfg, ph, nt, K, mode) * reps * FOLDS
                per_ph[ph] += t
                tot += t
        prep = pred("ga__C0", "fold_preparation_s", nt, K, mode) * FOLDS
        per_ph["fold_preparation_s"] += prep
        per_ds[d] = tot + prep
    T = sum(per_ds.values())
    print(f"\nCore tier [{mode}]: {T/3600:.0f} CPU h -> {T/3600/W:.1f} h ({T/3600/W/24:.2f} days) at {W} workers; "
          + ", ".join(f"{k.removesuffix('_s')} {v/3600:.1f} h" for k, v in per_ph.items()))
    print("  5 slowest datasets (CPU h; n; n_train; p; K):")
    slowest = sorted(per_ds, key=per_ds.get, reverse=True)[:5]
    for d in slowest:
        n, p = conf[d]
        print(f"    {d:<24} {per_ds[d]/3600:6.1f}   n {n}  n_train ~{0.8*n:.0f}  p {p}  K {min(200,p)}")
    # S2: ga C0 (C4 from its archive) and L1 without pre-filter, K = p, datasets with p <= 1000
    s2 = 0.0
    s2_ds = []
    for d, (n, p) in conf.items():
        if p > 1000:
            continue
        nt = 0.8 * n
        t = sum(pred("ga__C0", ph, nt, p, mode) for ph in ("main", "standard_inner_auc_s", "outer_scoring_s", "c4_s")) * 3 * FOLDS
        t += sum(pred("baseline__l1", ph, nt, p, mode) for ph in ("main", "standard_inner_auc_s", "outer_scoring_s")) * FOLDS
        s2 += t
        s2_ds.append(d)
    # K-linear bracket for S2: scale K=200 prediction linearly to K=p
    s2_lin = 0.0
    for d in s2_ds:
        n, p = conf[d]
        nt = 0.8 * n
        Kb = min(200, p)
        t = sum(pred("ga__C0", ph, nt, Kb, mode) for ph in ("main", "standard_inner_auc_s", "outer_scoring_s", "c4_s")) * 3 * FOLDS
        t += sum(pred("baseline__l1", ph, nt, Kb, mode) for ph in ("main", "standard_inner_auc_s", "outer_scoring_s")) * FOLDS
        s2_lin += t * (p / Kb)
    print(f"  S2 ({len(s2_ds)} datasets with p <= 1000; ga C0 x 3 seeds + L1, 10 folds, K = p): "
          f"{s2/3600:.1f} CPU h ({s2/3600/W:.1f} h at {W}) with the fitted K exponent; "
          f"{s2_lin/3600:.1f} CPU h ({s2_lin/3600/W:.1f} h at {W}) if time is linear in K above 200")
    result["cases"][mode] = {
        "core_cpu_h": T / 3600, "core_wall_h_at_workers": T / 3600 / W, "core_days_at_workers": T / 3600 / W / 24,
        "per_phase_cpu_h": {k: v / 3600 for k, v in per_ph.items()},
        "per_dataset": {d: {"cpu_h": per_ds[d] / 3600, "n": conf[d][0], "n_train": 0.8 * conf[d][0], "p": conf[d][1],
                            "K": min(200, conf[d][1])} for d in sorted(per_ds)},
        "slowest_5": slowest,
        "s2": {"datasets": sorted(s2_ds), "cpu_h_fitted_K": s2 / 3600, "wall_h_fitted_K": s2 / 3600 / W,
               "cpu_h_linear_K": s2_lin / 3600, "wall_h_linear_K": s2_lin / 3600 / W},
    }
conf_ntr = sorted(0.8 * n for n, p in conf.values())
n_above = sum(x > DEV_NMAX for x in conf_ntr)
result["confirmation_datasets_above_dev_n_train_max"] = n_above
print(f"\nConfirmation datasets with n_train > {DEV_NMAX} (dev max): {n_above} of {len(conf_ntr)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
import hashlib  # noqa: E402
print(f"written: {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()})")
