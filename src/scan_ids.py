"""Scan the repository and its full git history for real dataset IDs.

Reads only the ID column of every parquet file in data/raw. Keeps IDs that, after stripping
whitespace, are at least 5 characters long and contain at least one letter, underscore or hyphen.
Searches for them as whole tokens (not preceded or followed by a letter, digit, "_" or "-") in
    (a) every tracked file at HEAD, read from the working tree
    (b) every blob in the git history (git rev-list --all, git ls-tree, git cat-file)
    (c) every commit message
It prints only: file path, commit hash(es), dataset name and the MASKED form of each match
(letters -> A, digits -> 9). It never prints a real ID.

Usage (from the repository root, with .venv activated):
    python src/scan_ids.py
Exit code 0 if nothing is found, 1 if any match is found.
"""

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
ID = "ID"
MIN_LEN = 5
TOKEN = re.compile(r"[\w\-]+")  # whole-token boundaries: not adjacent to [A-Za-z0-9_-]


def mask(s):
    return re.sub(r"\d", "9", re.sub(r"[^\W\d_]", "A", s))


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True).stdout


def load_ids():
    """{id_string: set(dataset names)} for qualifying IDs."""
    ids = defaultdict(set)
    per_dataset = {}
    for p in sorted(RAW_DIR.glob("*.parquet")):
        col = pd.read_parquet(p, columns=[ID])[ID].dropna().astype(str).str.strip()
        keep = {v for v in col.unique()
                if len(v) >= MIN_LEN and re.search(r"[A-Za-z_\-]", v)}
        per_dataset[p.stem] = len(keep)
        for v in keep:
            ids[v].add(p.stem)
    return ids, per_dataset


class Matcher:
    def __init__(self, ids):
        self.ids = ids
        # IDs made only of token characters are found by token lookup;
        # the few others (containing ".", spaces, etc.) by a boundary-checked regex.
        self.simple = {i for i in ids if TOKEN.fullmatch(i)}
        self.complex = [(i, re.compile(rf"(?<![\w\-]){re.escape(i)}(?![\w\-])"))
                        for i in ids if i not in self.simple]

    def find(self, text):
        found = {t for t in TOKEN.findall(text) if t in self.simple}
        for i, rx in self.complex:
            if i in text and rx.search(text):
                found.add(i)
        return found


def main():
    ids, per_dataset = load_ids()
    print(f"IDs loaded: {len(ids)} qualifying IDs from {len(per_dataset)} datasets "
          f"(datasets with qualifying IDs: {sum(1 for v in per_dataset.values() if v)})")
    m = Matcher(ids)
    hits = []  # (where, path, commits, id)

    # (a) tracked files at HEAD, from the working tree
    for rel in git("ls-files", "-z").decode("utf-8").split("\0"):
        if not rel:
            continue
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        text = p.read_bytes().decode("utf-8", errors="replace")
        for i in m.find(text):
            hits.append(("HEAD (working tree)", rel, ["HEAD"], i))

    # (b) every blob in history: map blob -> {path: [commits]}
    commits = git("rev-list", "--all").decode().split()
    blob_paths = defaultdict(lambda: defaultdict(list))
    for c in commits:
        for line in git("ls-tree", "-r", "-z", c).decode("utf-8").split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            _, otype, sha = meta.split()
            if otype == "blob":
                blob_paths[sha][path].append(c[:7])
    n_blobs = 0
    for sha, paths in blob_paths.items():
        n_blobs += 1
        text = git("cat-file", "blob", sha).decode("utf-8", errors="replace")
        for i in m.find(text):
            for path, cs in paths.items():
                hits.append(("history", path, cs, i))

    # (c) commit messages
    for c in commits:
        msg = git("log", "-1", "--format=%B", c).decode("utf-8", errors="replace")
        for i in m.find(msg):
            hits.append(("history", "(commit message)", [c[:7]], i))

    print(f"Tracked files scanned at HEAD; {n_blobs} distinct blobs in {len(commits)} commits scanned; "
          f"{len(commits)} commit messages scanned")
    print()
    if not hits:
        print("No matches.")
        return 0
    print(f"{'where':<20}  {'path':<40}  {'dataset(s)':<28}  {'masked match':<20}  commits")
    for where, path, cs, i in sorted(hits, key=lambda h: (h[0], h[1], mask(h[3]))):
        ds = ",".join(sorted(ids[i]))
        print(f"{where:<20}  {path:<40}  {ds:<28}  {mask(i):<20}  {','.join(cs)}")
    print(f"\nMatches: {len(hits)} (distinct IDs: {len({h[3] for h in hits})})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
