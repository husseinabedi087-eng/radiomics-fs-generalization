"""Sealed dataset-level split into development and confirmation groups.

Inputs (only): data/meta/audit.csv and data/meta/data_handling_rules.json. No parquet file is opened.

Procedure
1. Datasets: the 50 audited datasets minus rules["excluded_datasets"] (43 datasets).
2. Post-rule size per dataset, from the integer counts in audit.csv:
       n = n_samples - n_exact_duplicate_rows_with_target   (datasets in drop_duplicate_rows; keep first)
       p = n_features - len(drop_columns[dataset]["columns"])
       log_p_over_n = ln(p / n)   (computed here from the counts, not taken from the rounded column)
3. Units: each split cluster is one unit (name = sorted member names joined by "+"), with
   log_p_over_n = mean of its members; every other dataset is its own unit.
4. Tertiles of log_p_over_n over units: cut points q1, q2 = numpy.quantile(values, [1/3, 2/3]) with
   the default method; tertile 1: v <= q1, tertile 2: q1 < v <= q2, tertile 3: v > q2.
5. One generator rng = numpy.random.default_rng(20261108). For tertiles 1, 2, 3 in this order:
   sort the units by name (Python string order), rng.shuffle the list, put the first
   max(1, floor(k/3)) units into development and the rest into confirmation.
6. Write data/split/development.txt and confirmation.txt (dataset names, one per line, sorted,
   cluster members listed individually), data/split/split_record.json and data/split/split.sha256.

Refuses to run if any split output already exists, or if make_split.py, audit.csv or
data_handling_rules.json have uncommitted changes (so the recorded commit hash is meaningful).

Prints counts per tertile and per group and the development list. The confirmation list is
never printed.

Usage (from the repository root, with .venv activated):
    python src/make_split.py
"""

import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = REPO_ROOT / "data/meta/audit.csv"
RULES_JSON = REPO_ROOT / "data/meta/data_handling_rules.json"
SPLIT_DIR = REPO_ROOT / "data/split"
DEV_TXT = SPLIT_DIR / "development.txt"
CONF_TXT = SPLIT_DIR / "confirmation.txt"
RECORD_JSON = SPLIT_DIR / "split_record.json"
SPLIT_SHA = SPLIT_DIR / "split.sha256"

SEED = 20261108
SELF_REL = "src/make_split.py"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True).stdout


def write_text(path, text):
    assert "\r" not in text
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    outputs = [DEV_TXT, CONF_TXT, RECORD_JSON, SPLIT_SHA]
    if any(p.exists() for p in outputs):
        print("Split outputs already exist in data/split; refusing to overwrite a sealed split.")
        return 1
    rels = [SELF_REL, "data/meta/audit.csv", "data/meta/data_handling_rules.json"]
    dirty = git("status", "--porcelain", "--", *rels).strip()
    if dirty:
        print(f"Uncommitted changes in inputs or in make_split.py; commit them first:\n{dirty}")
        return 1
    commit = git("log", "-1", "--format=%H", "--", SELF_REL).strip()

    rules = json.loads(RULES_JSON.read_bytes().decode("utf-8"))
    with open(AUDIT_CSV, encoding="utf-8", newline="") as f:
        audit = {r["dataset"]: r for r in csv.DictReader(f)}

    excluded = set(rules["excluded_datasets"])
    included = sorted(d for d in audit if d not in excluded)
    drop_cols = {d: len(v["columns"]) for d, v in rules.get("drop_columns", {}).items()}
    drop_dup = set(rules.get("drop_duplicate_rows", {}))

    size = {}
    for d in included:
        n = int(audit[d]["n_samples"]) - (int(audit[d]["n_exact_duplicate_rows_with_target"]) if d in drop_dup else 0)
        p = int(audit[d]["n_features"]) - drop_cols.get(d, 0)
        size[d] = {"n": n, "p": p, "log_p_over_n": math.log(p / n)}

    # units
    clusters = [sorted(c) for c in rules.get("split_clusters", {}).get("clusters", [])]
    in_cluster = {m for c in clusters for m in c}
    missing = [m for c in clusters for m in c if m not in size]
    if missing:
        print(f"Cluster members not among the included datasets: {missing}")
        return 1
    units = {}
    for c in clusters:
        units["+".join(c)] = {"members": c, "log_p_over_n": float(np.mean([size[m]["log_p_over_n"] for m in c]))}
    for d in included:
        if d not in in_cluster:
            units[d] = {"members": [d], "log_p_over_n": size[d]["log_p_over_n"]}

    values = np.array([u["log_p_over_n"] for u in units.values()])
    q1, q2 = (float(x) for x in np.quantile(values, [1 / 3, 2 / 3]))

    def tertile(v):
        return 1 if v <= q1 else (2 if v <= q2 else 3)

    rng = np.random.default_rng(SEED)
    tertiles, dev_units, conf_units = {}, [], []
    for t in (1, 2, 3):
        names = sorted(u for u, info in units.items() if tertile(info["log_p_over_n"]) == t)
        order = list(names)
        rng.shuffle(order)
        k = len(order)
        n_dev = max(1, k // 3) if k else 0
        tertiles[t] = {"units_sorted": names, "units_shuffled": order, "k": k,
                       "development_units": sorted(order[:n_dev]), "n_development_units": n_dev,
                       "n_confirmation_units": k - n_dev}
        dev_units += order[:n_dev]
        conf_units += order[n_dev:]

    dev = sorted(m for u in dev_units for m in units[u]["members"])
    conf = sorted(m for u in conf_units for m in units[u]["members"])
    assert not set(dev) & set(conf) and set(dev) | set(conf) == set(included)

    write_text(DEV_TXT, "\n".join(dev) + "\n")
    write_text(CONF_TXT, "\n".join(conf) + "\n")
    write_text(SPLIT_SHA, f"{sha256_file(DEV_TXT)}  development.txt\n{sha256_file(CONF_TXT)}  confirmation.txt\n")

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "rng": "numpy.random.default_rng(seed); one generator, tertiles shuffled in order 1, 2, 3",
        "tertile_variable": "log_p_over_n = ln(p / n) after the data-handling rules; cluster = mean of members",
        "cut_points": {"method": "numpy.quantile, default method, probabilities [1/3, 2/3]", "q1": q1, "q2": q2,
                       "rule": "tertile 1: v <= q1; tertile 2: q1 < v <= q2; tertile 3: v > q2"},
        "allocation": "within each tertile: sort units by name, shuffle, first max(1, floor(k/3)) to development",
        # Unit names and tertile membership only; group membership of confirmation units is not listed here.
        "units_per_tertile": {str(t): {"k": v["k"], "units": v["units_sorted"],
                                       "n_development_units": v["n_development_units"],
                                       "n_confirmation_units": v["n_confirmation_units"]}
                              for t, v in tertiles.items()},
        "cluster_handling": {"clusters": clusters,
                             "rule": "each cluster is one unit; its members are assigned to the same group"},
        "n_units": len(units),
        "dataset_counts": {"included": len(included), "development": len(dev), "confirmation": len(conf),
                           "excluded": len(excluded)},
        "inputs": {"audit.csv_sha256": sha256_file(AUDIT_CSV),
                   "data_handling_rules.json_sha256": sha256_file(RULES_JSON)},
        "make_split_py": {"git_commit": commit, "sha256": sha256_file(REPO_ROOT / SELF_REL)},
        "lists_sha256": {"development.txt": sha256_file(DEV_TXT), "confirmation.txt": sha256_file(CONF_TXT)},
    }
    write_text(RECORD_JSON, json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print(f"Included datasets: {len(included)}  |  units: {len(units)}  |  seed {SEED}")
    print(f"Cut points: q1 = {q1:.6f}, q2 = {q2:.6f}")
    for t, v in tertiles.items():
        n_dev_ds = sum(len(units[u]["members"]) for u in v["units_shuffled"][:v["n_development_units"]])
        n_all_ds = sum(len(units[u]["members"]) for u in v["units_shuffled"])
        print(f"Tertile {t}: {v['k']} units ({n_all_ds} datasets) -> development {v['n_development_units']} units "
              f"({n_dev_ds} datasets), confirmation {v['n_confirmation_units']} units ({n_all_ds - n_dev_ds} datasets)")
    print(f"Development: {len(dev)} datasets  |  Confirmation: {len(conf)} datasets")
    print("\nDevelopment datasets:")
    for d in dev:
        print(f"  {d}")
    print(f"\nWrote {DEV_TXT.relative_to(REPO_ROOT)}, {CONF_TXT.relative_to(REPO_ROOT)}, "
          f"{RECORD_JSON.relative_to(REPO_ROOT)}, {SPLIT_SHA.relative_to(REPO_ROOT)}")
    print(f"make_split.py commit: {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
