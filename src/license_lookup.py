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


def find_licence_fields(obj, path=""):
    """Collect (path, value) for every key containing 'licen' in a nested JSON object."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if "licen" in k.lower():
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
        for key in ("name", "full_name", "fullName", "title", "short_name", "shortName", "spdx_id", "id", "url"):
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


def lookup_github(dataset, repo, cache):
    url = f"https://api.github.com/repos/{repo}/license"
    if url not in cache:  # Hosny2018A/B/C share one repository: query it once
        cache[url] = get(url, accept="application/vnd.github+json")
    att = cache[url]
    if att.get("http_status") == 404:
        lic = "no license file"
    elif att.get("http_status") == 200:
        lic_obj = att.get("response_json", {}).get("license") or {}
        lic = lic_obj.get("spdx_id") or lic_obj.get("name") or "license file present, type not identified"
        if lic_obj.get("name") and lic_obj.get("spdx_id") and lic_obj["name"] != lic_obj["spdx_id"]:
            lic = f"{lic_obj['spdx_id']} ({lic_obj['name']})"
    else:
        lic = f"lookup failed: {att.get('error') or att.get('http_status')}"
    save_evidence(dataset, "github", [att])
    return row(dataset, "github", att, lic)


def lookup_zenodo(dataset, rec_id):
    att = get(f"https://zenodo.org/api/records/{rec_id}", accept="application/json")
    lic = ""
    if att.get("http_status") == 200:
        lic = licence_to_text(att.get("response_json", {}).get("metadata", {}).get("license"))
        lic = lic or "no license field"
    else:
        lic = f"lookup failed: {att.get('error') or att.get('http_status')}"
    save_evidence(dataset, "zenodo", [att])
    return row(dataset, "zenodo", att, lic)


def lookup_figshare(dataset, art_id):
    att = get(f"https://api.figshare.com/v2/articles/{art_id}", accept="application/json")
    if att.get("http_status") == 200:
        lic = licence_to_text((att.get("response_json", {}) or {}).get("license")) or "no license field"
    else:
        lic = f"lookup failed: {att.get('error') or att.get('http_status')}"
    save_evidence(dataset, "figshare", [att])
    return row(dataset, "figshare", att, lic)


LICENCE_PATTERNS = [
    r"CC[\s\-]?BY(?:[\s\-]NC)?(?:[\s\-]SA)?(?:[\s\-]ND)?[\s\-]?\d\.\d",
    r"CC0(?:\s*1\.0)?",
    r"Creative Commons[^<\"]{0,80}",
    r"MIT License|Apache License[^<\"]{0,20}|GNU [^<\"]{0,40}",
]


def extract_licence_from_html(html):
    hits = []
    # JSON-LD blocks often carry "license"
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            for _, v in find_licence_fields(json.loads(block)):
                hits.append(licence_to_text(v))
        except ValueError:
            pass
    for m in re.finditer(r'"[^"]*licen[cs]e[^"]*"\s*:\s*("(?:[^"\\]|\\.)*"|\{[^{}]*\})', html, re.I):
        hits.append(m.group(1).strip('"'))
    for pat in LICENCE_PATTERNS:
        hits += [h.strip() for h in re.findall(pat, html, re.I)]
    seen, out = set(), []
    for h in hits:
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def lookup_mendeley(dataset, ds_id):
    attempts = []
    for tmpl in MENDELEY_API_TEMPLATES:
        att = get(tmpl.format(id=ds_id), accept="application/json")
        attempts.append(att)
        if att.get("http_status") == 200 and "response_json" in att:
            fields = find_licence_fields(att["response_json"])
            texts = [t for t in (licence_to_text(v) for _, v in fields) if t]
            if texts:
                att["licence_fields_found"] = [{"path": p, "value": v} for p, v in fields]
                save_evidence(dataset, "mendeley_api", attempts)
                return row(dataset, "mendeley_api", att, " | ".join(dict.fromkeys(texts)))
    # fall back to the landing page HTML
    att = get(MENDELEY_LANDING.format(id=ds_id), accept="text/html")
    attempts.append(att)
    if att.get("http_status") == 200:
        html = att.get("response_text") or json.dumps(att.get("response_json", ""))
        texts = extract_licence_from_html(html)
        att["licence_text_extracted"] = texts
        lic = " | ".join(texts) if texts else "no licence text found in landing page"
    else:
        lic = f"lookup failed: {att.get('error') or att.get('http_status')}"
    save_evidence(dataset, "mendeley_landing", attempts)
    return row(dataset, "mendeley_landing", att, lic)


def main():
    t0 = time.perf_counter()
    rows, cache = [], {}
    jobs = ([("github", d, x) for d, x in GITHUB.items()]
            + [("zenodo", d, x) for d, x in ZENODO.items()]
            + [("figshare", d, x) for d, x in FIGSHARE.items()]
            + [("mendeley", d, x) for d, x in MENDELEY.items()])
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

    fields = ["dataset", "source_type", "url", "license_found", "http_status", "timestamp"]
    with open(LOOKUP_CSV, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print()
    table = [[str(r[k]) for k in fields] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(fields)]
    widths[2] = min(widths[2], 60)
    widths[3] = min(widths[3], 60)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*fields))
    for t in table:
        print(fmt.format(*[c if len(c) <= w else c[: w - 3] + "..." for c, w in zip(t, widths)]))
    print(f"\nEvidence files: {EVIDENCE_DIR.relative_to(REPO_ROOT)}  |  runtime {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
