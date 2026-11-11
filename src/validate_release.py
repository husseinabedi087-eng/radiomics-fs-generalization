"""Validate the run files of a release against MANIFEST.csv.

    python src/validate_release.py --plan configs/pilot.yaml [--plan ...] [--results-dir results]

Checks
  1. every manifest entry (latest row per path) re-hashes to its SHA-256, and the file's own commit and
     status match the manifest row;
  2. for each plan, separately: every expected unit has an "ok" run (counts per dataset and configuration),
     every file lies under results/runs/<group>/<plan>/, and its payload names that plan and group and the
     unit's dataset, configuration, repeat, fold and seed index; the plan's files are re-hashed and its
     commits listed on their own. Across plans: no two plans expect the same path (no collisions).
     Manifest entries that belong to none of the given plans are counted and reported (not a failure).
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

    # 2. each plan separately
    plan_paths = {}
    for plan_path in args.plan:
        plan = load_plan(plan_path)
        units = expand(plan)
        name = plan["name"]
        prefix = f"runs/{plan['group']}/{name}/"
        expected = Counter((u["dataset"], u["configuration"]) for u in units)
        found = Counter((u["dataset"], u["configuration"]) for u in units if u["path"] in ok_paths)
        missing = {k: expected[k] - found[k] for k in expected if found[k] < expected[k]}
        bad_place, bad_payload = 0, 0
        for u in units:
            if not u["path"].startswith(prefix):
                bad_place += 1
                problems.append(f"plan {name}: path outside {prefix}: {u['path']}")
            r = run_meta.get(u["path"])
            if r is None:
                continue
            ident = (r.get("plan"), r.get("group"), r.get("dataset"), r.get("configuration"), r.get("repeat"),
                     r.get("fold"), r.get("seed_index"))
            want = (name, plan["group"], u["dataset"], u["configuration"], u["repeat"], u["fold"], u["seed_index"])
            if ident != want:
                bad_payload += 1
                problems.append(f"plan {name}: payload identifiers {ident} do not match unit {want}")
        plan_commits = sorted({latest[u["path"]]["commit"] for u in units if u["path"] in latest})
        n_hashed = sum(u["path"] in run_meta for u in units)
        print(f"check 2: plan {name}: {sum(found.values())}/{len(units)} expected runs present "
              f"({len(expected)} dataset x configuration cells); {n_hashed} files re-hashed; "
              f"path or payload mismatches {bad_place + bad_payload}; commits {[c[:7] for c in plan_commits]}")
        for (ds, cfg), n in sorted(missing.items()):
            problems.append(f"plan {name}: {ds} / {cfg}: {n} runs missing")
        if name in plan_paths:
            problems.append(f"plan name {name} given twice")
        plan_paths[name] = {u["path"] for u in units}
    names = sorted(plan_paths)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = plan_paths[a] & plan_paths[b]
            if shared:
                problems.append(f"plans {a} and {b} share {len(shared)} paths (collision)")
            print(f"check 2: plans {a} / {b}: {len(shared)} shared paths")
    if args.plan:
        other = set(latest) - set().union(*plan_paths.values())
        print(f"check 2: manifest entries outside the given plans: {len(other)}")

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
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
