"""Look up dataset licences from public source metadata APIs (no dataset download).

For each dataset, queries the source's public metadata endpoint, saves every raw response
(with request URL, HTTP status and UTC timestamp) to data/meta/license_evidence/<dataset>.json,
and writes data/meta/license_lookup.csv:
    dataset, source_type, url, license_found, http_status, timestamp

Sources
- GitHub:   GET https://api.github.com/repos/{owner}/{repo}/license  (404 -> "no license file")
- Zenodo:   GET https://zenodo.org/api/records/{id}                  (metadata.license)
- figshare: GET https://api.figshare.com/v2/articles/{id}            (license.name)
- Mendeley: tries the public API endpoints below in order; if none returns a licence field,
            saves the landing page HTML and extracts the licence text from it.

Usage (from the repository root, with .venv activated):
    python src/license_lookup.py
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "data" / "meta" / "license_evidence"
LOOKUP_CSV = REPO_ROOT / "data" / "meta" / "license_lookup.csv"

USER_AGENT = "radiomics-fs-generalization license check"
DELAY_S = 1.0
TIMEOUT_S = 30

GITHUB = {
    "Deng2023": "BboyT/BM_NSCLC_subpathology",
    "Hosny2018A": "modelhub-ai/deep-prognosis",
    "Hosny2018B": "modelhub-ai/deep-prognosis",
    "Hosny2018C": "modelhub-ai/deep-prognosis",
    "Keek2020": "SebastianSanduleanu/Peritumoral-HN-Radiomics",
    "Song2020": "salan668/FAE",
    "Veeraraghavan2020": "The-Veeraraghavan-Lab/SciRepEndometrial2020",
    "radMLBench": "aydindemircioglu/radMLBench",
}
ZENODO = {
    "Fusco2022": "6344730",
    "Granata2021": "5162861",
    "Petrillo2023": "8392800",
    "LNDb": "6613714",
    "PI-CAI": "6624726",
}
FIGSHARE = {
    "OcanaTienda2023": "22277638",
    "Wang2024": "25203245",
}
MENDELEY = {
    "Ahn2021": "9mpy8mc6hn",
    "Brancato2023": "dvysdynchh",
    "Hunter2023": "rxn95mp24d",
    "Lu2019": "4c5znk5m8t",
}
MENDELEY_API_TEMPLATES = [
    "https://data.mendeley.com/public-api/datasets/{id}",
    "https://data.mendeley.com/public-api/datasets/{id}/versions",
]
MENDELEY_LANDING = "https://data.mendeley.com/datasets/{id}"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
_last_request = [0.0]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get(url, accept=None):
    """Polite GET: at least DELAY_S between requests. Returns an attempt record."""
    wait = DELAY_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    headers = {"Accept": accept} if accept else {}
    ts = utc_now()
    try:
        r = session.get(url, headers=headers, timeout=TIMEOUT_S)
        _last_request[0] = time.monotonic()
        rec = {"request_url": url, "timestamp_utc": ts, "http_status": r.status_code,
               "content_type": r.headers.get("Content-Type", "")}
        try:
            rec["response_json"] = r.json()
        except ValueError:
            rec["response_text"] = r.text
        return rec
    except requests.RequestException as e:
        _last_request[0] = time.monotonic()
        return {"request_url": url, "timestamp_utc": ts, "http_status": "",
                "error": f"{type(e).__name__}: {e}"}


# Only keys that name a licence exactly ("license", "licence", "data_licence", ...).
# Keys that merely contain the word, such as "licenseKey" (a web-monitoring key), are ignored.
LICENCE_KEY = re.compile(r"^(?:data_|dataset_)?licen[cs]e$", re.I)


def find_licence_fields(obj, path=""):
    """Collect (path, value) for every licence key (see LICENCE_KEY) in a nested JSON object."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if LICENCE_KEY.match(str(k)):
                found.append((p, v))
            found += find_licence_fields(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += find_licence_fields(v, f"{path}[{i}]")
    return found


def licence_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "full_name", "fullName", "title", "short_name", "shortName", "text",
                    "spdx_id", "id", "url"):
            v = value.get(key)
            if isinstance(v, dict):  # e.g. Zenodo title: {"en": "..."}
                v = next(iter(v.values()), None)
            if v:
                return str(v).strip()
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def save_evidence(dataset, source_type, attempts):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset, "source_type": source_type, "attempts": attempts}
    with open(EVIDENCE_DIR / f"{dataset}.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def row(dataset, source_type, attempt, licence):
    return {"dataset": dataset, "source_type": source_type, "url": attempt["request_url"],
            "license_found": licence, "http_status": attempt.get("http_status", ""),
            "timestamp": attempt["timestamp_utc"]}


def failed(att):
    return f"lookup failed: {att.get('error') or att.get('http_status')}"


# ---- parsers: attempt record(s) -> licence text (used by live lookups and by --reparse) ----

def parse_github(att):
    if att.get("http_status") == 404:
        return "no license file"
    if att.get("http_status") == 200:
        lic_obj = att.get("response_json", {}).get("license") or {}
        lic = lic_obj.get("spdx_id") or lic_obj.get("name") or "license file present, type not identified"
        if lic_obj.get("name") and lic_obj.get("spdx_id") and lic_obj["name"] != lic_obj["spdx_id"]:
            lic = f"{lic_obj['spdx_id']} ({lic_obj['name']})"
        return lic
    return failed(att)


def parse_zenodo(att):
    if att.get("http_status") == 200:
        return licence_to_text(att.get("response_json", {}).get("metadata", {}).get("license")) or "no license field"
    return failed(att)


def parse_figshare(att):
    if att.get("http_status") == 200:
        return licence_to_text((att.get("response_json", {}) or {}).get("license")) or "no license field"
    return failed(att)


def parse_mendeley_api(att):
    """Licence texts from a Mendeley public-API attempt ([] if none)."""
    if att.get("http_status") == 200 and "response_json" in att:
        fields = find_licence_fields(att["response_json"])
        return list(dict.fromkeys(t for t in (licence_to_text(v) for _, v in fields) if t))
    return []


def parse_mendeley(attempts):
    """(source_type, decisive attempt, licence) from the saved Mendeley attempts."""
    for att in attempts:
        if "public-api" in att["request_url"]:
            texts = parse_mendeley_api(att)
            if texts:
                return "mendeley_api", att, " | ".join(texts)
    landing = [a for a in attempts if "public-api" not in a["request_url"]]
    if not landing:
        return "mendeley_api", attempts[-1], "no licence field in API responses"
    att = landing[-1]
    if att.get("http_status") == 200:
        html = att.get("response_text") or json.dumps(att.get("response_json", ""))
        texts = extract_licence_from_html(html)
        return "mendeley_landing", att, " | ".join(texts) if texts else "no licence text found in landing page"
    return "mendeley_landing", att, failed(att)


# ---- live lookups ----

def lookup_github(dataset, repo, cache):
    url = f"https://api.github.com/repos/{repo}/license"
    if url not in cache:  # Hosny2018A/B/C share one repository: query it once
        cache[url] = get(url, accept="application/vnd.github+json")
    att = cache[url]
    save_evidence(dataset, "github", [att])
    return row(dataset, "github", att, parse_github(att))


def lookup_zenodo(dataset, rec_id):
    att = get(f"https://zenodo.org/api/records/{rec_id}", accept="application/json")
    save_evidence(dataset, "zenodo", [att])
    return row(dataset, "zenodo", att, parse_zenodo(att))


def lookup_figshare(dataset, art_id):
    att = get(f"https://api.figshare.com/v2/articles/{art_id}", accept="application/json")
    save_evidence(dataset, "figshare", [att])
    return row(dataset, "figshare", att, parse_figshare(att))


# Plain-text licence names searched in landing-page HTML. Free text such as "Creative Commons ..."
# and key names such as "licenseKey" are not used.
LICENCE_PATTERNS = [
    r"\bCC[\s\-]?BY(?:[\s\-]NC)?(?:[\s\-]SA)?(?:[\s\-]ND)?[\s\-]?\d\.\d\b",
    r"\bCC0(?:\s*1\.0)?\b",
]
# JSON key/value pairs embedded in HTML whose key is exactly a licence key (see LICENCE_KEY).
LICENCE_JSON_PAIR = re.compile(r'"((?:data_|dataset_)?licen[cs]e)"\s*:\s*("[^"]*"|\{[^{}]*\})', re.I)


def extract_licence_from_html(html):
    hits = []
    # JSON-LD blocks often carry "license"
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            for _, v in find_licence_fields(json.loads(block)):
                hits.append(licence_to_text(v))
        except ValueError:
            pass
    for m in LICENCE_JSON_PAIR.finditer(html):
        try:
            hits.append(licence_to_text(json.loads(m.group(2))))
        except ValueError:
            hits.append(m.group(2).strip('"'))
    for pat in LICENCE_PATTERNS:
        hits += [h.strip() for h in re.findall(pat, html, re.I)]
    return list(dict.fromkeys(h for h in hits if h))


def lookup_mendeley(dataset, ds_id):
    attempts = []
    for tmpl in MENDELEY_API_TEMPLATES:
        att = get(tmpl.format(id=ds_id), accept="application/json")
        attempts.append(att)
        texts = parse_mendeley_api(att)
        if texts:
            att["licence_fields_found"] = [{"path": p, "value": v}
                                           for p, v in find_licence_fields(att["response_json"])]
            save_evidence(dataset, "mendeley_api", attempts)
            return row(dataset, "mendeley_api", att, " | ".join(texts))
    # fall back to the landing page HTML
    att = get(MENDELEY_LANDING.format(id=ds_id), accept="text/html")
    attempts.append(att)
    source_type, att, lic = parse_mendeley(attempts)
    if att.get("http_status") == 200:
        att["licence_text_extracted"] = extract_licence_from_html(
            att.get("response_text") or json.dumps(att.get("response_json", "")))
    save_evidence(dataset, source_type, attempts)
    return row(dataset, source_type, att, lic)


def reparse_rows():
    """Rebuild every row from the saved evidence files; no request is sent."""
    rows = []
    jobs = ([("github", d) for d in GITHUB] + [("zenodo", d) for d in ZENODO]
            + [("figshare", d) for d in FIGSHARE] + [("mendeley", d) for d in MENDELEY])
    for kind, dataset in jobs:
        path = EVIDENCE_DIR / f"{dataset}.json"
        if not path.exists():
            print(f"  {dataset}: no evidence file, skipped")
            continue
        ev = json.loads(path.read_bytes().decode("utf-8"))
        attempts = ev["attempts"]
        if kind == "mendeley":
            source_type, att, lic = parse_mendeley(attempts)
        else:
            att = attempts[-1]
            source_type = kind
            lic = {"github": parse_github, "zenodo": parse_zenodo, "figshare": parse_figshare}[kind](att)
        rows.append(row(dataset, source_type, att, lic))
    return rows


FIELDS = ["dataset", "source_type", "url", "license_found", "http_status", "timestamp"]


def read_lookup_csv():
    if not LOOKUP_CSV.exists():
        return []
    with open(LOOKUP_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Look up dataset licences from public source metadata APIs.")
    parser.add_argument("--only", metavar="DATASET",
                        help="query only this dataset; its evidence file and its row in "
                             "license_lookup.csv are replaced, all other rows are kept")
    parser.add_argument("--reparse", action="store_true",
                        help="rebuild license_lookup.csv from the saved evidence files; "
                             "no request is sent and the evidence files are not changed")
    args = parser.parse_args()
    if args.reparse and args.only:
        parser.error("--reparse and --only cannot be combined")

    t0 = time.perf_counter()
    rows, cache = [], {}
    if args.reparse:
        print("Reparsing saved evidence (no network requests)")
        rows = reparse_rows()
        write_and_print(rows, t0)
        return 0
    jobs = ([("github", d, x) for d, x in GITHUB.items()]
            + [("zenodo", d, x) for d, x in ZENODO.items()]
            + [("figshare", d, x) for d, x in FIGSHARE.items()]
            + [("mendeley", d, x) for d, x in MENDELEY.items()])
    if args.only:
        known = [d for _, d, _ in jobs]
        jobs = [j for j in jobs if j[1] == args.only]
        if not jobs:
            print(f"Unknown dataset for --only: {args.only}. Known: {', '.join(known)}")
            return 1
    for i, (kind, dataset, ident) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {dataset} ({kind})", flush=True)
        if kind == "github":
            rows.append(lookup_github(dataset, ident, cache))
        elif kind == "zenodo":
            rows.append(lookup_zenodo(dataset, ident))
        elif kind == "figshare":
            rows.append(lookup_figshare(dataset, ident))
        else:
            rows.append(lookup_mendeley(dataset, ident))

    if args.only:  # merge: replace only this dataset's row, keep the original row order
        new = {r["dataset"]: r for r in rows}
        old = read_lookup_csv()
        merged = [new.pop(r["dataset"], r) for r in old]
        rows = merged + list(new.values())

    write_and_print(rows, t0)
    return 0


def write_and_print(rows, t0):
    fields = FIELDS
    with open(LOOKUP_CSV, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print()
    table = [[str(r[k]) for k in fields] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(fields)]
    widths[2] = min(widths[2], 60)
    widths[3] = min(widths[3], 90)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*fields))
    for t in table:
        print(fmt.format(*[c if len(c) <= w else c[: w - 3] + "..." for c, w in zip(t, widths)]))
    print(f"\nEvidence files: {EVIDENCE_DIR.relative_to(REPO_ROOT)}  |  runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
