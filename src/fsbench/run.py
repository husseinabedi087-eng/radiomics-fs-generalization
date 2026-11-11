"""Study runner (part 6b).

    python -m fsbench.run --plan configs/pilot.yaml --n-jobs 6      (with src on PYTHONPATH)

One unit of work = (dataset, repeat, fold, configuration, seed index).
  Search configurations: the 12 C0 searchers; C1, C2, C3 on the plan's control searchers with the plan's
    hyperparameter values. Search seed = SHA-256 of "dataset|repeat|fold|seed_index" (fsbench.partitions
    .derive_seed), shared by every configuration (common random numbers).
  Baselines: one run per outer fold; the random forest once per model seed index, model seed =
    SHA-256 of "dataset|repeat|fold|rf|model_seed_index".
Output: results/runs/<group>/<dataset>/<config>/r<repeat>_f<fold>_s<seed index>.json.gz, written atomically
(temporary file, then rename); then one row in results/MANIFEST.csv (path, SHA-256, identifiers, seeds,
commit, status) and one in results/progress.csv. Only the main process writes the CSV files.
Resume: a unit is skipped if its file exists and its SHA-256 matches an "ok" manifest row.
Refuses to start if git has uncommitted changes in src/ or configs/ (except the test-only "synthetic"
group, whose datasets SYNTH_<n>_<p>_<seed> are generated in memory and never touch real data).
Workers: process pool with spawn; OMP/MKL/OPENBLAS threads set to 1 in the environment inherited by
every worker (set before the pool starts) and enforced with threadpoolctl.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
C0_SEARCHERS = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V",
                "gwo_S", "gwo_V", "woa_S", "woa_V"]
CONTROL_KEYS = {"C1": "P", "C2": "lambda", "C3": "r"}
GROUPS = ("development", "confirmation", "synthetic")
MANIFEST_FIELDS = ["path", "sha256", "dataset", "group", "configuration", "repeat", "fold", "seed_index",
                   "search_seed", "model_seed", "commit", "status", "utc_finished"]
PROGRESS_FIELDS = ["utc", "done", "total", "path", "status", "wall_s", "eta_s"]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


# ---------------- plan ----------------

def load_plan(path):
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "group", "datasets", "repeats", "folds", "search_seed_indices"):
        if key not in plan:
            raise ValueError(f"plan is missing '{key}'")
    if plan["group"] not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")
    if plan["group"] == "synthetic" and not all(d.startswith("SYNTH_") for d in plan["datasets"]):
        raise ValueError("the synthetic group accepts only SYNTH_* datasets")
    if plan["group"] != "synthetic" and any(d.startswith("SYNTH_") for d in plan["datasets"]):
        raise ValueError("SYNTH_* datasets are allowed only in the synthetic group")
    return plan


def expand(plan):
    """All units of a plan, ordered by dataset, repeat, fold."""
    from fsbench.partitions import derive_seed

    budget = int(plan.get("budget", 2000))
    searchers = plan.get("c0_searchers", C0_SEARCHERS)
    controls = plan.get("controls", {}) or {}
    ctrl_searchers = plan.get("control_searchers", [])
    selectors = plan.get("baselines", [])
    rf_seeds = plan.get("rf_model_seed_indices", [0])
    units = []
    for ds in plan["datasets"]:
        for r in plan["repeats"]:
            for f in plan["folds"]:
                def add(config, seed_index, **kw):
                    rel = Path("runs") / plan["group"] / ds / config / f"r{r}_f{f}_s{seed_index}.json.gz"
                    units.append({"plan": plan["name"], "group": plan["group"], "dataset": ds, "repeat": r,
                                  "fold": f, "configuration": config, "seed_index": seed_index,
                                  "budget": budget, "path": rel.as_posix(), **kw})
                for k in plan["search_seed_indices"]:
                    seed = derive_seed(ds, r, f, k)
                    for s in searchers:
                        add(f"{s}__C0", k, kind="search", searcher=s, control="C0", param=None,
                            search_seed=seed, model_seed=None)
                    for ctrl, values in controls.items():
                        vals = values if isinstance(values, list) else [values]
                        for v in vals:
                            for s in ctrl_searchers:
                                add(f"{s}__{ctrl}_{CONTROL_KEYS[ctrl]}{fmt(v)}", k, kind="search", searcher=s,
                                    control=ctrl, param=v, search_seed=seed, model_seed=None)
                for sel in selectors:
                    if sel == "rf":
                        for m in rf_seeds:
                            add("baseline__rf", m, kind="baseline", selector="rf", search_seed=None,
                                model_seed=derive_seed(ds, r, f, "rf", m))
                    else:
                        add(f"baseline__{sel}", 0, kind="baseline", selector=sel, search_seed=None, model_seed=None)
    return units


# ---------------- data ----------------

def synthetic_dataset(name):
    """SYNTH_<n>_<p>_<seed>: logistic data with 10 informative features (test-only)."""
    import numpy as np
    import pandas as pd
    _, n, p, seed = name.split("_")
    n, p, seed = int(n), int(p), int(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:min(10, p)] = rng.choice([-1.0, 1.0], size=min(10, p))
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    return pd.DataFrame(X, columns=[f"f{j:04d}" for j in range(p)]), y


def load_data(name, group):
    if group == "synthetic":
        return synthetic_dataset(name)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import data_access
    X, y, _ = data_access.load_dataset(name)
    return X, y


# ---------------- worker ----------------

_CACHE = {"data": {}, "parts": {}, "folds": {}}


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _versions():
    import numpy, scipy, sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def _threads():
    from threadpoolctl import threadpool_info
    return sorted({(i.get("internal_api"), i.get("num_threads")) for i in threadpool_info()})


def _write_atomic(path, payload):
    data = gzip.compress(json.dumps(payload, separators=(",", ":"), default=float).encode("utf-8"), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def execute(unit, results_root, commit, partitions_dir):
    """Run one unit, write its file, return its manifest row."""
    from threadpoolctl import threadpool_limits
    from fsbench import experiment, partitions

    t_start, utc_start = time.perf_counter(), utc_now()
    out = {"plan": unit["plan"], "group": unit["group"], "configuration": unit["configuration"],
           "seed_index": unit["seed_index"], "git_commit": commit, "versions": _versions(),
           "utc_start": utc_start, "pid": os.getpid()}
    try:
        with threadpool_limits(1):
            out["blas_threads"] = _threads()
            ds = unit["dataset"]
            if ds not in _CACHE["data"]:
                _CACHE["data"][ds] = load_data(ds, unit["group"])
            X, y = _CACHE["data"][ds]
            if ds not in _CACHE["parts"]:
                _CACHE["parts"][ds] = partitions.get(ds, y, directory=partitions_dir)
            key = (ds, unit["repeat"], unit["fold"])
            fold_cached = key in _CACHE["folds"]
            if not fold_cached:
                _CACHE["folds"] = {key: experiment.prepare_fold(ds, X, y, partitions.fold(_CACHE["parts"][ds], *key[1:]))}
            fd = _CACHE["folds"][key]
            if unit["kind"] == "search":
                res = experiment.run_search(fd, unit["searcher"], unit["control"], unit["param"],
                                            search_seed=unit["search_seed"], budget=unit["budget"])
            else:
                res = experiment.run_baseline(fd, unit["selector"], model_seed=unit["model_seed"])
            res["timings"]["fold_preparation_cached"] = fold_cached
            out.update(res)
    except Exception as e:  # recorded, never silently dropped
        out.update({"dataset": unit["dataset"], "repeat": unit["repeat"], "fold": unit["fold"],
                    "search_seed": unit.get("search_seed"), "model_seed": unit.get("model_seed"),
                    "status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    out["utc_finished"] = utc_now()
    out["wall_s"] = time.perf_counter() - t_start
    sha = _write_atomic(Path(results_root) / unit["path"], out)
    return {"path": unit["path"], "sha256": sha, "dataset": unit["dataset"], "group": unit["group"],
            "configuration": unit["configuration"], "repeat": unit["repeat"], "fold": unit["fold"],
            "seed_index": unit["seed_index"], "search_seed": unit.get("search_seed"),
            "model_seed": unit.get("model_seed"), "commit": commit, "status": out["status"],
            "utc_finished": out["utc_finished"], "_wall_s": out["wall_s"]}


# ---------------- main process ----------------

def read_manifest(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, row):
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def completed_paths(results_root, manifest_rows):
    latest = {}
    for row in manifest_rows:
        latest[row["path"]] = row
    done = set()
    for p, row in latest.items():
        f = Path(results_root) / p
        if row["status"] == "ok" and f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == row["sha256"]:
            done.add(p)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a study plan.")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"),
                    help="results root (only the synthetic group may use another directory)")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    results_root = Path(args.results_dir).resolve()
    synthetic = plan["group"] == "synthetic"
    if not synthetic:
        if results_root != (REPO_ROOT / "results").resolve():
            sys.exit("only the synthetic group may write outside results/")
        dirty = git("status", "--porcelain", "--", "src", "configs")
        if dirty:
            sys.exit(f"refusing to start: uncommitted changes in src/ or configs/:\n{dirty}")
    commit = git("rev-parse", "HEAD")
    partitions_dir = results_root / "partitions"
    manifest_path, progress_path = results_root / "MANIFEST.csv", results_root / "progress.csv"

    units = expand(plan)
    done = completed_paths(results_root, read_manifest(manifest_path))
    todo = [u for u in units if u["path"] not in done]
    print(f"plan {plan['name']} ({plan['group']}): {len(units)} units, {len(units) - len(todo)} already done, "
          f"{len(todo)} to run, {args.n_jobs} workers, commit {commit[:7]}", flush=True)
    if not todo:
        return 0

    # partitions are created here, once per dataset, so that workers only read and verify them
    from fsbench import partitions
    for ds in plan["datasets"]:
        _, y = load_data(ds, plan["group"])
        partitions.get(ds, y, directory=partitions_dir)

    t0 = time.perf_counter()
    n_done, n_err = 0, 0

    def finished(row):
        nonlocal n_done, n_err
        n_done += 1
        n_err += row["status"] != "ok"
        elapsed = time.perf_counter() - t0
        eta = elapsed / n_done * (len(todo) - n_done)
        append_csv(manifest_path, MANIFEST_FIELDS, row)
        append_csv(progress_path, PROGRESS_FIELDS, {"utc": utc_now(), "done": n_done, "total": len(todo),
                                                     "path": row["path"], "status": row["status"],
                                                     "wall_s": f"{row['_wall_s']:.1f}", "eta_s": f"{eta:.0f}"})
        print(f"[{n_done}/{len(todo)}] {row['status']:<5} {row['path']}  {row['_wall_s']:.1f}s  "
              f"elapsed {elapsed / 3600:.2f} h  ETA {eta / 3600:.2f} h", flush=True)

    if args.n_jobs == 1:
        _init_worker()
        for u in todo:
            finished(execute(u, str(results_root), commit, str(partitions_dir)))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker) as pool:
            futures = [pool.submit(execute, u, str(results_root), commit, str(partitions_dir)) for u in todo]
            for fut in as_completed(futures):
                finished(fut.result())
    print(f"finished: {n_done} runs, {n_err} errors, {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
