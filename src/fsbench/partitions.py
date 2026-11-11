"""Partitions (PREREG_STAGE1 section 4 and section 7).

All seeds come from SHA-256 of a "|"-joined key, reduced to an integer below 2**32:
    outer folds of repeat r:          "<dataset>|<r>"
    inner 3-fold split of fold (r,f): "<dataset>|<r>|<f>|inner"
    C1 selection split (20%):         "<dataset>|<r>|<f>|c1_selection"
    C1 inner 3-fold of the other 80%: "<dataset>|<r>|<f>|c1_inner"
    C2 3 x 3 repeated split:          "<dataset>|<r>|<f>|c2"
Outer: StratifiedKFold(5, shuffle=True), 2 repeats. All other splits are stratified and drawn from the
outer-training part only. Every index stored is an absolute row index of the dataset as returned by
src/data_access.py (after the data-handling rules). Partitions are saved to
results/partitions/<dataset>.json and reused by every method; an existing file is verified against
a fresh derivation and must match exactly.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit

N_OUTER, N_REPEATS, N_INNER = 5, 2, 3
C1_SELECTION_FRACTION = 0.2
C2_SPLITS, C2_REPEATS = 3, 3
REPO_ROOT = Path(__file__).resolve().parents[2]
PARTITION_DIR = REPO_ROOT / "results" / "partitions"


def derive_seed(*parts):
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2 ** 32)


def _splits(splitter, idx, y):
    """Run a splitter on the subset idx; return absolute (train, test) index lists."""
    out = []
    for tr, te in splitter.split(np.zeros(len(idx)), y[idx]):
        out.append({"train": sorted(int(i) for i in idx[tr]), "test": sorted(int(i) for i in idx[te])})
    return out


def derive(dataset, y):
    """Full partition structure for one dataset (pure function of the dataset name and labels)."""
    y = np.asarray(y)
    all_idx = np.arange(len(y))
    repeats = []
    for r in range(N_REPEATS):
        outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=derive_seed(dataset, r))
        folds = []
        for f, (tr, te) in enumerate(outer.split(all_idx, y)):
            tr = all_idx[tr]
            inner = _splits(StratifiedKFold(N_INNER, shuffle=True,
                                            random_state=derive_seed(dataset, r, f, "inner")), tr, y)
            sss = StratifiedShuffleSplit(1, test_size=C1_SELECTION_FRACTION,
                                         random_state=derive_seed(dataset, r, f, "c1_selection"))
            (fit_pos, sel_pos), = sss.split(np.zeros(len(tr)), y[tr])
            c1_fit, c1_sel = np.sort(tr[fit_pos]), np.sort(tr[sel_pos])
            c1_inner = _splits(StratifiedKFold(N_INNER, shuffle=True,
                                               random_state=derive_seed(dataset, r, f, "c1_inner")), c1_fit, y)
            c2 = _splits(RepeatedStratifiedKFold(n_splits=C2_SPLITS, n_repeats=C2_REPEATS,
                                                 random_state=derive_seed(dataset, r, f, "c2")), tr, y)
            folds.append({"repeat": r, "fold": f,
                          "outer_train": sorted(int(i) for i in tr), "outer_test": sorted(int(i) for i in all_idx[te]),
                          "inner": inner,
                          "c1": {"fit": [int(i) for i in c1_fit], "selection": [int(i) for i in c1_sel],
                                 "inner": c1_inner},
                          "c2": c2})
        repeats.append(folds)
    seeds = {"outer": {str(r): derive_seed(dataset, r) for r in range(N_REPEATS)}}
    return {"dataset": dataset, "n_samples": int(len(y)), "n_outer": N_OUTER, "n_repeats": N_REPEATS,
            "n_inner": N_INNER, "seed_rule": "SHA-256 of '|'-joined key, first 8 bytes little-endian, mod 2**32",
            "seeds": seeds, "repeats": repeats}


def get(dataset, y, directory=PARTITION_DIR):
    """Load results/partitions/<dataset>.json (verified against a fresh derivation) or create it."""
    fresh = derive(dataset, y)
    path = Path(directory) / f"{dataset}.json"
    text = json.dumps(fresh, separators=(",", ":")) + "\n"
    if path.exists():
        if path.read_bytes().decode("utf-8") != text:
            raise RuntimeError(f"{path} does not match the derived partitions for {dataset}")
        return fresh
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)
    return fresh


def fold(partitions, repeat, fold_index):
    return partitions["repeats"][repeat][fold_index]


def to_positions(absolute, reference):
    """Positions of absolute row indices inside the sorted array `reference` (e.g. outer_train)."""
    reference = np.asarray(reference)
    pos = np.searchsorted(reference, absolute)
    if not np.array_equal(reference[pos], np.asarray(absolute)):
        raise ValueError("indices are not a subset of the reference")
    return pos
