"""ID-only structure checks.

Reads only the ID column of a few datasets (never features or Target) and prints counts.
No individual ID is printed.

Usage (from the repository root, with .venv activated):
    python src/id_checks.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
ID = "ID"


def read_ids(name):
    return pd.read_parquet(RAW_DIR / f"{name}.parquet", columns=[ID])[ID]


def prefix_suffix_report(name):
    ids = read_ids(name)
    s = ids.dropna().astype(str).str.strip()
    has_us = s.str.contains("_", regex=False)
    parts = s[has_us].str.rsplit("_", n=1)
    prefixes = parts.str[0]
    suffixes = parts.str[1]
    print(f"{name}")
    print(f"  rows: {len(ids)}")
    print(f"  missing IDs: {int(ids.isna().sum())}")
    print(f"  IDs without an underscore: {int((~has_us).sum())}")
    print(f"  distinct prefixes before the last underscore: {prefixes.nunique()}")
    print(f"  distinct suffixes: {sorted(suffixes.unique(), key=lambda x: (len(x), x))}")
    counts = suffixes.value_counts()
    print(f"  rows per suffix: {dict(sorted(counts.items(), key=lambda kv: (len(kv[0]), kv[0])))}")
    per_prefix = prefixes.value_counts()
    print(f"  prefixes with more than one row: {int((per_prefix > 1).sum())} "
          f"(max rows per prefix: {int(per_prefix.max()) if len(per_prefix) else 0})")


KNBTG = re.compile(r"^([A-Za-z]+)0*(\d+)$")


def normalise_knbtg(ids):
    """Strip whitespace, remove '-', remove leading zeros from the numeric part."""
    out, unmatched = set(), 0
    for v in ids.dropna().astype(str):
        v = v.strip().replace("-", "")
        m = KNBTG.match(v)
        if m:
            out.add(f"{m.group(1).upper()}{int(m.group(2))}")
        else:
            unmatched += 1
    return out, unmatched


def knbtg_overlap(a, b):
    ids_a, ids_b = read_ids(a), read_ids(b)
    norm_a, un_a = normalise_knbtg(ids_a)
    norm_b, un_b = normalise_knbtg(ids_b)
    print(f"{a} vs {b}")
    print(f"  {a}: {len(ids_a)} rows, {len(norm_a)} distinct normalised IDs, {un_a} not matching the pattern")
    print(f"  {b}: {len(ids_b)} rows, {len(norm_b)} distinct normalised IDs, {un_b} not matching the pattern")
    print(f"  intersection size: {len(norm_a & norm_b)}")


def main():
    prefix_suffix_report("Hunter2023")
    print()
    prefix_suffix_report("UPENN-GBM")
    print()
    knbtg_overlap("Arita2018", "Sasaki2019")
    return 0


if __name__ == "__main__":
    sys.exit(main())
