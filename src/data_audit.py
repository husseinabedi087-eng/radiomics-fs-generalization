"""Descriptive audit of the downloaded radMLBench datasets.

Reads data/raw/<name>.parquet and data/meta/<name>.json and writes
    data/meta/audit.csv            one row per dataset
    data/meta/audit_notes.md       one paragraph per dataset
    data/meta/license_review.csv   empty license fields, filled in by hand

Forbidden here: any model, any performance metric, any feature-target statistic
(class counts are allowed), any sample-level train/test split.

Usage (from the repository root, with .venv activated):
    python src/data_audit.py
"""

import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
META_DIR = REPO_ROOT / "data" / "meta"
INDEX_CSV = META_DIR / "radmlbench_index.csv"
AUDIT_CSV = META_DIR / "audit.csv"
NOTES_MD = META_DIR / "audit_notes.md"
LICENSE_CSV = META_DIR / "license_review.csv"

TARGET = "Target"
ID = "ID"

CORR_THRESHOLD = 0.99
CORR_MAX_FEATURES = 2000
CORR_SEED = 0

SUSPICIOUS_TOKENS = ["target", "label", "outcome", "class", "survival", "time", "date", "status",
                     "patient", "lesion", "split", "set", "fold", "train", "test", "center", "site", "id"]
# Remarks keywords that make crit4 a flag ("time point" added as a spelling variant of "timepoint").
REMARK_FLAG_TOKENS = ["lesion", "scan", "timepoint", "time point", "multiple"]

CRITERIA = ["crit1_binary", "crit2_minority20", "crit3_license", "crit4_patient", "crit5_features"]

AUDIT_FIELDS = [
    "dataset", "modality", "pathology", "outcome", "year", "publication_doi", "download_source",
    "license",
    "n_samples", "n_features", "n_classes", "class_counts", "minority_class_count", "minority_fraction",
    "meta_nInstances", "meta_nFeatures", "meta_matches",
    "log_p_over_n",
    "missing_total_fraction", "n_features_with_missing", "n_features_with_inf",
    "n_non_numeric_features",
    "n_constant_features", "n_duplicate_feature_columns",
    "n_feature_pairs_abs_corr_above_0.99", "corr_sampled",
    "n_exact_duplicate_rows", "n_exact_duplicate_rows_with_target",
    "id_unique", "n_unique_ids", "id_pattern",
    "suspicious_columns",
    "patient_structure_proposed", "patient_structure_reason",
    *CRITERIA,
    "included", "needs_review",
    "target_dtype", "meta_missings_reported", "meta_missings_mismatch",
    "id_missing_count", "id_has_whitespace",
]
LICENSE_FIELDS = ["dataset", "download_source", "publication_doi", "publication_title", "authors", "year",
                  "license", "license_source", "decision"]


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yn(flag):
    return "yes" if flag else "no"


def one_line(text):
    return " ".join(str(text).split()) if text is not None else ""


def mask_id(value):
    """Mask an ID for privacy: letters -> 'A', digits -> '9', everything else unchanged
    (e.g. 'EXAMPLE-123' -> 'AAAAAAA-999'). Missing IDs become ''."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\d", "9", re.sub(r"[^\W\d_]", "A", str(value)))


def count_high_corr_pairs(num):
    """Feature-feature pairs with |r| > threshold (pairwise-complete, inf treated as missing)."""
    cols = list(num.columns)
    sampled = len(cols) > CORR_MAX_FEATURES
    if sampled:
        rng = np.random.default_rng(CORR_SEED)
        idx = np.sort(rng.choice(len(cols), size=CORR_MAX_FEATURES, replace=False))
        cols = [cols[i] for i in idx]
    if len(cols) < 2:
        return 0, sampled
    x = num[cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    if x.isna().any().any():
        c = x.corr().to_numpy()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.corrcoef(x.to_numpy(), rowvar=False)
    upper = np.triu_indices(len(cols), k=1)
    vals = np.abs(c[upper])
    return int(np.sum(vals > CORR_THRESHOLD)), sampled  # NaN (constant columns) never counts


def suspicious(columns):
    out = []
    for col in columns:
        low = str(col).lower()
        if any(tok in low for tok in SUSPICIOUS_TOKENS):
            out.append(str(col))
    return out


def id_prefix_grouping(ids):
    """Return (suspected, n_prefixes): does stripping the last separator-delimited token of the
    ID group rows into plausible patients (more than one group, on average at most 5 rows each)?"""
    s = ids.astype(str)
    if not s.str.contains(r"[_\-\. ]", regex=True).all():
        return False, 0
    prefixes = s.str.replace(r"[_\-\. ][^_\-\. ]*$", "", regex=True)
    n_pref = prefixes.nunique()
    n = len(s)
    suspected = 1 < n_pref < s.nunique() and n_pref >= n / 5
    return suspected, n_pref


def patient_structure(df, remarks):
    """Propose a patient structure, crit4 value and a one-sentence reason."""
    remarks_l = (remarks or "").lower()
    hits = [t for t in REMARK_FLAG_TOKENS if t in remarks_l]
    remark_txt = (f'remarks mention {", ".join(repr(h) for h in hits)} ("{one_line(remarks)}")'
                  if hits else "remarks do not mention lesions, scans, timepoints or multiple")

    if ID not in df.columns:
        return "unknown", "flag", f"No ID column is present; {remark_txt}."
    ids = df[ID]
    if ids.isna().any():
        return "unknown", "flag", f"The ID column has {int(ids.isna().sum())} missing values; {remark_txt}."
    n, n_unique = len(ids), ids.nunique()
    if n_unique < n:
        crit4 = "flag" if hits else "pass"
        return ("multiple_rows_with_id", crit4,
                f"ID is not unique ({n_unique} unique IDs for {n} rows), so rows repeat per ID; {remark_txt}.")
    grouped, n_pref = id_prefix_grouping(ids)
    if grouped:
        return ("multiple_rows_with_id", "flag",
                f"ID is unique ({n} rows) but dropping its last token leaves {n_pref} groups, "
                f"suggesting several rows per patient; {remark_txt}.")
    if hits:
        return ("unknown", "flag",
                f"ID is unique ({n} rows) and shows no grouping pattern, but {remark_txt}.")
    return ("one_row_per_patient", "pass",
            f"ID is unique ({n} rows), shows no grouping pattern, and {remark_txt}.")


def audit_dataset(name, df, meta, index_sha):
    notes = []
    remarks = meta.get("remarks")

    if index_sha is None:
        notes.append("Not listed in radmlbench_index.csv.")
    elif sha256_file(RAW_DIR / f"{name}.parquet") != index_sha:
        notes.append("WARNING: parquet SHA-256 does not match radmlbench_index.csv.")

    features = [c for c in df.columns if c not in (ID, TARGET)]
    X = df[features]
    n_samples, n_features = df.shape[0], len(features)

    # target: class counts only
    if TARGET in df.columns:
        y = df[TARGET]
        counts = y.value_counts(dropna=True).sort_index()
        n_classes = int(len(counts))
        class_counts = ";".join(f"{k}:{v}" for k, v in counts.items())
        minority = int(counts.min()) if n_classes else 0
        minority_fraction = round(minority / n_samples, 6) if n_samples else ""
        if y.isna().any():
            notes.append(f"Target has {int(y.isna().sum())} missing values.")
        odd = [v for v in counts.index if v not in (0, 1)]
        if odd:
            notes.append(f"Target values other than 0/1: {odd}.")
        target_dtype = str(y.dtype)
        if pd.api.types.is_bool_dtype(y):
            notes.append(f"Target is stored as boolean (True/False, dtype {target_dtype}), not 0/1.")
    else:
        n_classes, class_counts, minority, minority_fraction = 0, "", 0, ""
        target_dtype = ""
        notes.append("No Target column.")

    # metadata agreement
    m_inst, m_feat = meta.get("nInstances"), meta.get("nFeatures")
    meta_matches = (m_inst == n_samples) and (m_feat == n_features)
    if not meta_matches:
        notes.append(f"Counts differ from metadata (computed {n_samples}x{n_features}, "
                     f"metadata {m_inst}x{m_feat}).")

    # feature types
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    non_numeric = [c for c in features if c not in numeric_cols]
    if non_numeric:
        examples = {c: X[c].dropna().astype(str).unique()[:3].tolist() for c in non_numeric}
        notes.append(f"Non-numeric features ({len(non_numeric)}): {examples}.")
    num = X[numeric_cols]

    # missing / inf
    na = X.isna()
    missing_total_fraction = round(float(na.to_numpy().mean()), 6) if n_features and n_samples else 0.0
    n_feat_missing = int(na.any().sum())
    n_missing_cells = int(na.to_numpy().sum())
    meta_missings = meta.get("Missings")
    meta_missings_mismatch = meta_missings is not None and meta_missings != n_missing_cells
    if meta_missings_mismatch:
        notes.append(f"Metadata reports {meta_missings} missing values, but the features contain "
                     f"{n_missing_cells} missing cells.")
    if numeric_cols:
        inf_mask = np.isinf(num.astype("float64").to_numpy())
        n_feat_inf = int(inf_mask.any(axis=0).sum())
    else:
        n_feat_inf = 0
    if n_feat_inf:
        notes.append(f"{n_feat_inf} features contain infinite values.")

    # constant and duplicate columns
    n_constant = int((X.nunique(dropna=False) <= 1).sum())
    n_dup_cols = int(X.T.duplicated().sum()) if n_features else 0

    # feature-feature correlation
    n_corr, corr_sampled = count_high_corr_pairs(num)

    # duplicate rows
    n_dup_rows = int(X.duplicated().sum()) if n_features else 0
    with_target = features + ([TARGET] if TARGET in df.columns else [])
    n_dup_rows_t = int(df[with_target].duplicated().sum()) if with_target else 0
    if n_dup_rows or n_dup_rows_t:
        notes.append(f"{n_dup_rows} exact duplicate rows over features only, "
                     f"{n_dup_rows_t} including the target.")

    # ID
    if ID in df.columns:
        n_unique_ids = int(df[ID].nunique())
        id_unique = n_unique_ids == n_samples and not df[ID].isna().any()
        # privacy: no real ID is written; the first 5 IDs are masked (letters -> A, digits -> 9)
        id_pattern = "; ".join(mask_id(v) for v in df[ID].head(5).tolist())
        id_missing = int(df[ID].isna().sum())
        id_str = df[ID].dropna().astype(str)
        n_ws = int(id_str.str.contains(r"\s", regex=True).sum())
        id_has_ws = n_ws > 0
        if id_missing:
            notes.append(f"ID has {id_missing} missing value(s).")
        if id_has_ws:
            n_edge = int((id_str != id_str.str.strip()).sum())
            notes.append(f"{n_ws} IDs contain whitespace ({n_edge} with leading or trailing whitespace), "
                         f"masked e.g. {id_str[id_str.str.contains(r'\s', regex=True)].head(3).map(lambda v: repr(mask_id(v))).tolist()}.")
    else:
        n_unique_ids, id_unique, id_pattern = 0, False, ""
        id_missing, id_has_ws = "", False

    susp = suspicious(features)
    if susp:
        shown = ", ".join(susp[:20]) + (f", ... ({len(susp) - 20} more)" if len(susp) > 20 else "")
        notes.append(f"Suspicious column names ({len(susp)}): {shown}.")

    proposed, crit4, reason = patient_structure(df, remarks)

    crit = {
        "crit1_binary": "pass" if n_classes == 2 else "fail",
        "crit2_minority20": "pass" if minority >= 20 else "fail",
        "crit3_license": "flag",
        "crit4_patient": crit4,
        "crit5_features": "pass" if n_features >= 1 else "fail",
    }

    row = {
        "dataset": name,
        "modality": meta.get("modality", ""),
        "pathology": meta.get("pathology", ""),
        "outcome": meta.get("outcome", ""),
        "year": meta.get("year", ""),
        "publication_doi": meta.get("publication_doi", ""),
        "download_source": meta.get("download_source", ""),
        "license": "unknown",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "class_counts": class_counts,
        "minority_class_count": minority,
        "minority_fraction": minority_fraction,
        "meta_nInstances": m_inst,
        "meta_nFeatures": m_feat,
        "meta_matches": yn(meta_matches),
        "log_p_over_n": round(math.log(n_features / n_samples), 6) if n_features and n_samples else "",
        "missing_total_fraction": missing_total_fraction,
        "n_features_with_missing": n_feat_missing,
        "n_features_with_inf": n_feat_inf,
        "n_non_numeric_features": len(non_numeric),
        "n_constant_features": n_constant,
        "n_duplicate_feature_columns": n_dup_cols,
        "n_feature_pairs_abs_corr_above_0.99": n_corr,
        "corr_sampled": yn(corr_sampled),
        "n_exact_duplicate_rows": n_dup_rows,
        "n_exact_duplicate_rows_with_target": n_dup_rows_t,
        "id_unique": yn(id_unique),
        "n_unique_ids": n_unique_ids,
        "id_pattern": id_pattern,
        "suspicious_columns": ";".join(susp),
        "patient_structure_proposed": proposed,
        "patient_structure_reason": reason,
        **crit,
        "included": yn(all(v == "pass" for v in crit.values())),
        "needs_review": yn(any(v == "flag" for v in crit.values())),
        "target_dtype": target_dtype,
        "meta_missings_reported": "" if meta_missings is None else meta_missings,
        "meta_missings_mismatch": yn(meta_missings_mismatch),
        "id_missing_count": id_missing,
        "id_has_whitespace": yn(id_has_ws),
    }
    remarks_text = f"\"{remarks}\"" if remarks is not None else "(no remarks in metadata)"

    paragraph = (
        f"## {name}\n\n"
        f"{meta.get('modality', '')}, {meta.get('pathology', '')}, outcome: {meta.get('outcome', '')}. "
        f"{n_samples} samples, {n_features} features, classes {class_counts or 'n/a'}. "
        f"Metadata remarks (verbatim): {remarks_text}. "
        + (" ".join(notes) + " " if notes else "No odd values found. ")
        + f"Patient structure proposed: {proposed} (crit4 {crit4}). {reason}\n"
    )
    return row, paragraph


def main():
    t0 = time.perf_counter()
    index = {}
    if INDEX_CSV.exists():
        with open(INDEX_CSV, encoding="utf-8", newline="") as f:
            index = {r["dataset_name"]: r["parquet_sha256"] for r in csv.DictReader(f)}

    names = sorted(p.stem for p in RAW_DIR.glob("*.parquet"))
    if not names:
        print("No parquet files in data/raw.")
        return 1

    rows, paragraphs, licenses = [], [], []
    for i, name in enumerate(names, start=1):
        print(f"[{i}/{len(names)}] {name}", flush=True)
        df = pd.read_parquet(RAW_DIR / f"{name}.parquet")
        with open(META_DIR / f"{name}.json", encoding="utf-8") as f:
            meta = json.load(f)
        row, para = audit_dataset(name, df, meta, index.get(name))
        rows.append(row)
        paragraphs.append(para)
        licenses.append({
            "dataset": name,
            "download_source": meta.get("download_source", ""),
            "publication_doi": meta.get("publication_doi", ""),
            "publication_title": one_line(meta.get("publication_title", "")),
            "authors": meta.get("authors", ""),
            "year": meta.get("year", ""),
            "license": "", "license_source": "", "decision": "",
        })

    write_csv(AUDIT_CSV, AUDIT_FIELDS, rows)
    write_csv(LICENSE_CSV, LICENSE_FIELDS, licenses)
    with open(NOTES_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("# radMLBench audit notes\n\n"
                "Descriptive audit only: no models, no performance metrics, no feature-target statistics.\n\n")
        f.write("\n".join(paragraphs))

    print()
    print(f"Datasets audited: {len(rows)}")
    for c in CRITERIA:
        passed = [r["dataset"] for r in rows if r[c] == "pass"]
        failed = [r["dataset"] for r in rows if r[c] == "fail"]
        print(f"{c}: pass {len(passed)}, fail {len(failed)}" + (f" -> {', '.join(failed)}" if failed else ""))
    flagged = [r["dataset"] for r in rows if r["crit4_patient"] == "flag"]
    print(f"crit4_patient flag: {len(flagged)}" + (f" -> {', '.join(flagged)}" if flagged else ""))
    print(f"Included (pass on all five): {sum(r['included'] == 'yes' for r in rows)}")
    print(f"Needs review (any flag): {sum(r['needs_review'] == 'yes' for r in rows)}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
