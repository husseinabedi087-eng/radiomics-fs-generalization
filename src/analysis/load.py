"""Configuration, validation and run loading for the analysis.

Config (YAML):
  label: development_dryrun
  plans:                          # in order; a later plan may replace configurations of earlier ones
    - file: configs/development.yaml
    - file: configs/dev_enet_fix.yaml
      replaces: [baseline__enet]
  s2_plan: null                   # optional plan file for sensitivity analysis S2
  control_values: {C1: 5, C2: 1.0, C3: 8}   # the stage-2 values; other grid values are excluded
  sensitivity_list: configs/sensitivity_s1.txt

Runs are read only from results/MANIFEST.csv (latest row per path) for the units of the named plans;
each file must be "ok" and match its SHA-256. A C0 run yields two records: the C0 record (its
checkpoints) and a C4 record (its "c4" knee records). A baseline yields one record at checkpoint 2000
(its returned subset).
"""

import contextlib
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fsbench.run import expand, load_plan  # noqa: E402
import validate_release  # noqa: E402

CHECKPOINTS = (250, 500, 1000, 2000)
CONTROL_PARAM = {"C1": "P", "C2": "lambda", "C3": "r"}
HEADERS = {"development": "EXPLORATORY (development data)",
           "confirmation": "CONFIRMATORY (confirmation data)",
           "synthetic": "SYNTHETIC TEST DATA"}


class Refused(RuntimeError):
    """The analysis refuses to run (failed validation, missing or inconsistent run files)."""


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("label", "plans"):
        if key not in cfg:
            raise Refused(f"config is missing '{key}'")
    cfg.setdefault("s2_plan", None)
    cfg.setdefault("control_values", {})
    cfg.setdefault("sensitivity_list", None)
    return cfg


def validate(plan_files, results_dir):
    """Run src/validate_release.py on the named plans; return its printed report or raise Refused."""
    args = []
    for f in plan_files:
        args += ["--plan", str(f)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = validate_release.main(args + ["--results-dir", str(results_dir)])
    report = buf.getvalue()
    if code != 0:
        raise Refused("validation failed:\n" + report)
    return report


def _manifest(results_dir):
    path = Path(results_dir) / "MANIFEST.csv"
    if not path.exists():
        raise Refused("results/MANIFEST.csv not found")
    with open(path, encoding="utf-8", newline="") as f:
        latest = {}
        for r in csv.DictReader(f):
            latest[r["path"]] = r
    return latest


def _ckpt(rec):
    return {"auc": rec["outer_auc"], "std_auc": rec["standard_inner_auc"], "brier": rec["outer_brier"],
            "size": rec["size"], "subset": rec["subset"]}


def records_from_run(r):
    """Analysis records of one run payload."""
    base = {"dataset": r["dataset"], "repeat": r["repeat"], "fold": r["fold"], "seed_index": r["seed_index"],
            "K": r["K"], "n_train": r["n_train"], "n_test": r["n_test"], "p": r["n_features_raw"],
            "prefilter": r.get("prefilter", True)}
    if r["kind"] == "baseline":
        sel = r["selector"]
        return [{**base, "kind": "baseline", "config": f"baseline__{sel}", "family": sel, "control": "baseline",
                 "control_value": None, "checkpoints": {2000: _ckpt(r["returned"])}}]
    s, ctrl = r["searcher"], r["control"]
    value = None if ctrl == "C0" else float(next(iter(r["control_param"].values())))
    out = [{**base, "kind": "search", "config": f"{s}__{ctrl}", "family": s, "control": ctrl, "control_value": value,
            "checkpoints": {int(c): _ckpt(v) for c, v in r["checkpoints"].items()}}]
    if ctrl == "C0":
        if "c4" not in r:
            raise Refused(f"C0 run without C4 records: {r['dataset']} {r['configuration']} r{r['repeat']} f{r['fold']}")
        out.append({**base, "kind": "search", "config": f"{s}__C4", "family": s, "control": "C4", "control_value": None,
                    "checkpoints": {int(c): _ckpt(v) for c, v in r["c4"].items()}})
    return out


def load_plans(plan_entries, results_dir):
    """Records of the named plans, applying 'replaces'. Returns (records, inputs) where inputs lists
    (path, sha256) of every file read."""
    latest = _manifest(results_dir)
    records, inputs = [], []
    for entry in plan_entries:
        plan = load_plan(resolve(entry["file"]))
        replaces = set(entry.get("replaces", []))
        units = expand(plan)
        if replaces:
            have = {u["configuration"] for u in units}
            if not replaces <= have:
                raise Refused(f"plan {plan['name']} replaces {sorted(replaces - have)} but does not contain them")
            records = [x for x in records if x["run_configuration"] not in replaces]
        for u in units:
            row = latest.get(u["path"])
            f = Path(results_dir) / u["path"]
            if row is None or row["status"] != "ok" or not f.is_file():
                raise Refused(f"planned run missing or not ok: {u['path']}")
            data = f.read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise Refused(f"SHA-256 mismatch: {u['path']}")
            payload = json.loads(gzip.decompress(data))
            if payload.get("plan") != plan["name"] or payload.get("status") != "ok":
                raise Refused(f"run file does not belong to plan {plan['name']}: {u['path']}")
            inputs.append((u["path"], row["sha256"]))
            for rec in records_from_run(payload):
                rec["run_configuration"] = u["configuration"]
                rec["plan"] = plan["name"]
                records.append(rec)
    keys = [(x["dataset"], x["config"], x["control_value"], x["repeat"], x["fold"], x["seed_index"]) for x in records]
    if len(keys) != len(set(keys)):
        raise Refused("duplicate runs across plans (use 'replaces')")
    return records, inputs


def plan_group(plan_entries):
    groups = {load_plan(resolve(e["file"]))["group"] for e in plan_entries}
    if len(groups) != 1:
        raise Refused(f"plans of one analysis must share a group, got {sorted(groups)}")
    return groups.pop()


def read_list(path):
    return [l.strip() for l in resolve(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
