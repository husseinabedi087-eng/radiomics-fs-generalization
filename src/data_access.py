"""The only entry point for loading radMLBench data in this project.

load_dataset(name) -> (X, y, ids)
    X    pandas DataFrame of features (ID and Target removed)
    y    numpy int64 array with values 0/1
    ids  numpy array of str (missing IDs become "")

Before loading, in this order:
    1. the dataset must be known (listed in data/meta/radmlbench_index.csv) and not excluded
       (data_handling_rules.json -> excluded_datasets);
    2. seal: datasets in data/split/confirmation.txt are refused unless docs/PREREG_STAGE2.sha256
       exists AND is tracked by git;
    3. the parquet SHA-256 must match radmlbench_index.csv.
The rules file and the split lists are themselves checked against their stored SHA-256 values.

Then every rule in data_handling_rules.json is applied, in this order:
    drop_columns, drop_duplicate_rows (exact duplicates over features + Target, keep first),
    strip_id_whitespace, target_to_int.
No other transformation: no imputation, scaling or filtering.

list_datasets(group) returns the dataset names of "development" or "confirmation" (no loading).
"""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
META_DIR = REPO_ROOT / "data" / "meta"
SPLIT_DIR = REPO_ROOT / "data" / "split"
INDEX_CSV = META_DIR / "radmlbench_index.csv"
RULES_JSON = META_DIR / "data_handling_rules.json"
RULES_SHA = META_DIR / "data_handling_rules.sha256"
SPLIT_SHA = SPLIT_DIR / "split.sha256"
STAGE2_SHA_REL = "docs/PREREG_STAGE2.sha256"

TARGET = "Target"
ID = "ID"
GROUPS = ("development", "confirmation")


class DataAccessError(Exception):
    """Base class for refusals by data_access."""


class DatasetExcludedError(DataAccessError):
    pass


class SealedDatasetError(DataAccessError):
    pass


class HashMismatchError(DataAccessError):
    pass


class IntegrityError(DataAccessError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stored_hashes(sha_file):
    out = {}
    for line in Path(sha_file).read_bytes().decode("utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def _rules():
    expected = _stored_hashes(RULES_SHA).get(RULES_JSON.name)
    if expected != _sha256(RULES_JSON):
        raise IntegrityError("data_handling_rules.json does not match data_handling_rules.sha256")
    return json.loads(RULES_JSON.read_bytes().decode("utf-8"))


def _index():
    with open(INDEX_CSV, encoding="utf-8", newline="") as f:
        return {r["dataset_name"]: r for r in csv.DictReader(f)}


def list_datasets(group):
    """Dataset names in 'development' or 'confirmation' (names only; nothing is loaded)."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")
    path = SPLIT_DIR / f"{group}.txt"
    if not path.exists() or not SPLIT_SHA.exists():
        raise IntegrityError("the split has not been created (run src/make_split.py)")
    if _stored_hashes(SPLIT_SHA).get(path.name) != _sha256(path):
        raise IntegrityError(f"{path.name} does not match split.sha256")
    return [l.strip() for l in path.read_bytes().decode("utf-8").splitlines() if l.strip()]


def stage2_frozen():
    """True only if docs/PREREG_STAGE2.sha256 exists and is tracked by git."""
    if not (REPO_ROOT / STAGE2_SHA_REL).is_file():
        return False
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", STAGE2_SHA_REL],
                       capture_output=True)
    return r.returncode == 0


def load_dataset(name):
    rules = _rules()
    index = _index()
    if name not in index:
        raise DataAccessError(f"unknown dataset: {name}")
    if name in rules.get("excluded_datasets", {}):
        raise DatasetExcludedError(f"{name} is excluded: {rules['excluded_datasets'][name]}")
    if name in list_datasets("confirmation") and not stage2_frozen():
        raise SealedDatasetError(f"{name} is sealed until stage 2 of the pre-registration is frozen "
                                 f"({STAGE2_SHA_REL} must exist and be tracked by git)")

    path = RAW_DIR / f"{name}.parquet"
    if not path.is_file():
        raise DataAccessError(f"missing file {path}; run src/download_data.py")
    if _sha256(path) != index[name]["parquet_sha256"]:
        raise HashMismatchError(f"SHA-256 of {path.name} does not match radmlbench_index.csv")

    df = pd.read_parquet(path)

    # drop_columns
    rule = rules.get("drop_columns", {}).get(name)
    if rule:
        missing = [c for c in rule["columns"] if c not in df.columns]
        if missing:
            raise IntegrityError(f"{name}: columns to drop not found: {missing}")
        df = df.drop(columns=rule["columns"])

    # drop_duplicate_rows: exact duplicates over features + Target (ID excluded), keep first
    rule = rules.get("drop_duplicate_rows", {}).get(name)
    if rule:
        subset = [c for c in df.columns if c != ID]
        df = df.loc[~df.duplicated(subset=subset, keep=rule.get("keep", "first"))]

    # strip_id_whitespace
    if name in rules.get("strip_id_whitespace", []):
        df[ID] = df[ID].where(df[ID].isna(), df[ID].astype(str).str.strip())

    # target_to_int
    tti = rules.get("target_to_int", {})
    if name in tti.get("datasets", []):
        mapping = {True: tti["mapping"]["True"], False: tti["mapping"]["False"]}
        if not set(df[TARGET].dropna().unique()) <= set(mapping):
            raise IntegrityError(f"{name}: Target has values outside True/False")
        df[TARGET] = df[TARGET].map(mapping)

    y = df[TARGET].to_numpy()
    if pd.isna(y).any() or not set(np.unique(y)) <= {0, 1}:
        raise IntegrityError(f"{name}: Target must contain only 0 and 1")
    y = y.astype(np.int64)
    ids = df[ID].astype("string").fillna("").to_numpy(dtype=str)
    X = df.drop(columns=[ID, TARGET]).reset_index(drop=True)
    return X, y, ids
