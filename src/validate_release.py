"""Validate the run files of a release against MANIFEST.csv.

    python src/validate_release.py --plan configs/pilot.yaml [--plan ...] [--results-dir results]

Checks
  1. every manifest entry (latest row per path) re-hashes to its SHA-256, and the file's own commit and
     status match the manifest row;
  2. for each plan, every expected unit has an "ok" run (counts per dataset and configuration);
  3. every run's commit exists in the git history;
  4. no confirmation-group run started before the commit that added docs/PREREG_STAGE2.sha256 (and,
     if that file is not committed, there is no confirmation-group run at all).
Exit code 0 only if every check passes.
"""

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from fsbench.run import expand, load_plan  # noqa: E402

STAGE2 = "docs/PREREG_STAGE2.sha256"


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="append", default=[])
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args(argv)
    root = Path(args.results_dir)
    manifest = root / "MANIFEST.csv"
    problems = []

    rows = []
    if manifest.exists():
        with open(manifest, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    latest = {}
    for r in rows:
        latest[r["path"]] = r
    print(f"manifest rows: {len(rows)}; distinct run files: {len(latest)}")

    # 1. hashes, commit and status inside each file
    run_meta = {}
    for p, r in latest.items():
        f = root / p
        if not f.exists():
            problems.append(f"missing file: {p}")
            continue
        data = f.read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            problems.append(f"hash mismatch: {p}")
            continue
        payload = json.loads(gzip.decompress(data))
        if payload.get("git_commit") != r["commit"] or payload.get("status") != r["status"]:
            problems.append(f"file content disagrees with manifest: {p}")
        run_meta[p] = payload
    ok_paths = {p for p, r in latest.items() if r["status"] == "ok" and p in run_meta}
    n_err = sum(1 for r in latest.values() if r["status"] != "ok")
    print(f"check 1: {len(run_meta)} files re-hashed; ok {len(ok_paths)}, not ok {n_err}")
    if n_err:
        problems.append(f"{n_err} runs with status other than ok")

    # 2. expected counts per plan
    for plan_path in args.plan:
        plan = load_plan(plan_path)
        units = expand(plan)
        expected = Counter((u["dataset"], u["configuration"]) for u in units)
        found = Counter((u["dataset"], u["configuration"]) for u in units if u["path"] in ok_paths)
        missing = {k: expected[k] - found[k] for k in expected if found[k] < expected[k]}
        print(f"check 2: plan {plan['name']}: {sum(found.values())}/{len(units)} expected runs present "
              f"({len(expected)} dataset x configuration cells)")
        for (ds, cfg), n in sorted(missing.items()):
            problems.append(f"plan {plan['name']}: {ds} / {cfg}: {n} runs missing")

    # 3. commits exist
    commits = {r["commit"] for r in latest.values()}
    for c in sorted(commits):
        if git("cat-file", "-e", f"{c}^{{commit}}").returncode != 0:
            problems.append(f"commit not in history: {c}")
    print(f"check 3: {len(commits)} distinct commits checked")

    # 4. confirmation runs only after stage 2 was frozen
    conf = [p for p, r in latest.items() if r["group"] == "confirmation"]
    added = git("log", "--diff-filter=A", "--format=%cI", "--", STAGE2).stdout.strip().splitlines()
    if conf:
        if not added:
            problems.append(f"{len(conf)} confirmation runs exist but {STAGE2} is not committed")
        else:
            frozen = datetime.fromisoformat(added[-1])
            early = [p for p in conf if p in run_meta
                     and datetime.fromisoformat(run_meta[p]["utc_start"]) < frozen]
            for p in early:
                problems.append(f"confirmation run started before stage 2 was frozen: {p}")
    print(f"check 4: {len(conf)} confirmation runs; stage 2 committed: {'yes' if added else 'no'}")

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems[:200]:
            print(f"  - {p}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
