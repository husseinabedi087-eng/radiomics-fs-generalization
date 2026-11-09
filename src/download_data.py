"""Download every dataset listed by radMLBench and save it unchanged.

Outputs
-------
data/raw/<dataset_name>.parquet        dataset exactly as returned by radMLBench.loadData
data/meta/<dataset_name>.json          metadata from radMLBench.getMetaData (bundled metadata.yaml)
data/meta/radmlbench_index.csv         one row per downloaded dataset
data/meta/download_failures.csv        one row per failed dataset (header only if none failed)

No cleaning, no filtering, no statistics beyond row and column counts.
Text outputs are written with LF line endings; hashes are computed with hashlib.

Usage (from the repository root, with .venv activated):
    python src/download_data.py
    python src/download_data.py --overwrite   # replace existing parquet files
"""

import argparse
import csv
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import radMLBench

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
META_DIR = REPO_ROOT / "data" / "meta"
INDEX_CSV = META_DIR / "radmlbench_index.csv"
FAILURES_CSV = META_DIR / "download_failures.csv"

# Column names documented in radMLBench.loadData ("with columns ID and Target").
TARGET_COLUMN = "Target"
ID_COLUMN = "ID"

INDEX_FIELDS = [
    "dataset_name",
    "n_rows",
    "n_columns",
    "target_column",
    "id_column",
    "parquet_sha256",
    "download_timestamp_utc",
    "radmlbench_version",
]
FAILURE_FIELDS = ["dataset_name", "error_type", "error_message", "timestamp_utc"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_csv(path: Path, fields: list, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace parquet files that already exist in data/raw")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(RAW_DIR.glob("*.parquet"))
    if existing and not args.overwrite:
        print(f"data/raw already contains {len(existing)} parquet file(s). "
              "Refusing to overwrite; rerun with --overwrite to replace them.")
        return 1

    rmb_version = version("radMLBench")
    datasets = radMLBench.listDatasets()
    print(f"radMLBench {rmb_version}: {len(datasets)} datasets listed")

    index_rows = []
    failures = []
    written_files = []

    for i, name in enumerate(datasets, start=1):
        print(f"[{i}/{len(datasets)}] {name} ... ", end="", flush=True)
        parquet_path = RAW_DIR / f"{name}.parquet"
        try:
            df = radMLBench.loadData(name)
            timestamp = utc_now()
            df.to_parquet(parquet_path)

            meta = radMLBench.getMetaData(name)
            meta_path = META_DIR / f"{name}.json"
            write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n")

            index_rows.append({
                "dataset_name": name,
                "n_rows": df.shape[0],
                "n_columns": df.shape[1],
                "target_column": TARGET_COLUMN if TARGET_COLUMN in df.columns else "",
                "id_column": ID_COLUMN if ID_COLUMN in df.columns else "",
                "parquet_sha256": sha256_file(parquet_path),
                "download_timestamp_utc": timestamp,
                "radmlbench_version": rmb_version,
            })
            written_files += [parquet_path, meta_path]
            print(f"ok ({df.shape[0]} rows, {df.shape[1]} columns)")
        except Exception as e:  # record and continue with the rest
            if parquet_path.exists():
                parquet_path.unlink()
            failures.append({
                "dataset_name": name,
                "error_type": type(e).__name__,
                "error_message": " ".join(str(e).split()),
                "timestamp_utc": utc_now(),
            })
            print(f"FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    write_csv(INDEX_CSV, INDEX_FIELDS, index_rows)
    write_csv(FAILURES_CSV, FAILURE_FIELDS, failures)
    written_files += [INDEX_CSV, FAILURES_CSV]

    total_mb = sum(p.stat().st_size for p in written_files) / 2**20
    print()
    print(f"radMLBench version:   {rmb_version}")
    print(f"Datasets listed:      {len(datasets)}")
    print(f"Datasets downloaded:  {len(index_rows)}")
    print(f"Datasets failed:      {len(failures)}")
    print(f"Total size on disk:   {total_mb:.2f} MB")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
