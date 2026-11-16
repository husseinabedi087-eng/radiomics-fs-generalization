"""Save the output of src/run_health.py (unchanged) as LF text and as JSON parsed from that text.

    python src/save_health.py --name confirmation --plan configs/confirmation.yaml --plan configs/confirmation_s2.yaml

run_health.py prints its report; this wrapper calls run_health.main() unmodified, captures the printed
report, writes it to results/health/<name>.txt (UTF-8, LF) and writes results/health/<name>.json with
the same content parsed into fields. No run-file field is read here beyond what run_health.py reads
(no AUC, Brier score or subset).
"""

import argparse
import ast
import contextlib
import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import run_health  # noqa: E402

OUT = REPO / "results" / "health"


def parse(text):
    plans, cur, section = [], None, None
    head = {}
    for line in text.splitlines():
        if line.startswith("stage 2 committed at:"):
            head["stage2_committed_at"] = line.split(":", 1)[1].strip()
            continue
        m = re.match(r"=== plan (\S+) \((\w+), prefilter (\w+)\): (\d+) units, (\d+) missing", line)
        if m:
            cur = {"plan": m[1], "group": m[2], "prefilter": m[3] == "True", "units": int(m[4]),
                   "missing": int(m[5]), "threads": [], "warnings": {}, "empty_subset": [], "c1_early": [],
                   "wall_s": {}}
            plans.append(cur)
            section = None
            continue
        if cur is None or not line.strip():
            continue
        if line.startswith("status:"):
            s, c = line[len("status: "):].split("; commits: ")
            cur["status"], cur["commits"] = ast.literal_eval(s), ast.literal_eval(c)
        elif line.startswith("runs started before the stage-2 commit:"):
            cur["runs_before_stage2"] = int(line.rsplit(":", 1)[1])
        elif line.startswith("thread settings"):
            section = "threads"
        elif line.startswith("convergence warnings per configuration"):
            section = "warnings"
        elif line.startswith("configurations with any warning:"):
            a, b = re.findall(r"\d+", line)
            cur["configurations_with_warnings"], cur["configurations"] = int(a), int(b)
            section = None
        elif line.startswith("baseline runs flagged empty_subset:"):
            cur["empty_subset_runs"] = int(line.rsplit(":", 1)[1])
            section = "empty"
        elif line.startswith("C1 runs:"):
            n = [int(x) for x in re.findall(r"(?<=: )\d+", line)]
            cur["c1"] = {"runs": n[0], "stopped_early": n[1], "ended_before_first_checkpoint": n[2]}
            section = "c1"
        elif line.startswith("K below the number of features:"):
            a, b = re.findall(r"(\d+) runs", line)
            cur["k_below_p_runs"], cur["k_equal_p_runs"] = int(a), int(b)
            section = None
        elif line.startswith("main-phase wall time"):
            section = "wall"
        elif section == "threads":
            n, spec = line.strip().split(None, 1)
            cur["threads"].append({"runs": int(n), "setting": json.loads(spec)})
        elif section == "warnings":
            cfg, total, ratio = line.split()
            k, n = ratio.split("/")
            cur["warnings"][cfg] = {"total": int(total), "runs_with_warnings": int(k), "runs": int(n)}
        elif section == "empty":
            m = re.match(r"\s+(\S+): (\d+) runs on (\d+) datasets: (.*)", line)
            per = {d: int(c) for d, c in re.findall(r"([^,]+?) \((\d+)\)(?:, |$)", m[4])}
            cur["empty_subset"].append({"configuration": m[1], "runs": int(m[2]), "datasets": int(m[3]),
                                        "per_dataset": per})
        elif section == "c1":
            cur["c1_early"].append(line.strip())
        elif section == "wall":
            cfg, lo, med, hi = line.split()
            cur["wall_s"][cfg] = {"min": float(lo), "median": float(med), "max": float(hi)}
    return {**head, "plans": plans}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--plan", action="append", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run_health.main([x for p in args.plan for x in ("--plan", p)])
    text = buf.getvalue()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.name}.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    data = {"name": args.name, "plans_requested": [Path(p).as_posix() for p in args.plan],
            "source": "src/run_health.py (unchanged), output parsed by src/save_health.py", **parse(text)}
    with open(out / f"{args.name}.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"written {out.name}/{args.name}.txt and {args.name}.json ({len(data['plans'])} plans)")
    return code


if __name__ == "__main__":
    sys.exit(main())
