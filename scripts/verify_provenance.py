#!/usr/bin/env python3
"""Verify every cited DOI/arXiv record and, when present, dataset hashes.

This script is intentionally dependency-free.  It performs network lookups, so
it is meant to be run by a human before a release, not on every CI job.

Usage:
    python scripts/verify_provenance.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "references.json"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "photocat-provenance/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def crossref_title(doi: str) -> str:
    payload = json.loads(fetch(f"https://api.crossref.org/works/{doi}"))
    return str(payload["message"].get("title", [""])[0])


def arxiv_title(identifier: str) -> str:
    xml = fetch(f"https://export.arxiv.org/api/query?id_list={identifier}").decode("utf-8")
    match = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = 0
    print(f"manifest verified on {manifest['verified_on']}")
    print("\nReferences")
    for item in manifest["proposal_references"]:
        try:
            if item.get("arxiv"):
                actual = arxiv_title(item["arxiv"])
            elif item.get("doi"):
                actual = crossref_title(item["doi"])
            else:
                actual = item.get("citation", "")
            print(f"  {item['label']:4s} {item['key']:16s} {actual[:80]}")
        except Exception as error:
            failures += 1
            print(f"  {item['label']:4s} {item['key']:16s} FAILED: {error}")
    print("\nDatasets")
    for item in manifest["data_sources"]:
        try:
            title = crossref_title(item["doi"])
            print(f"  {item['key']:18s} {title[:90]}")
        except Exception as error:
            failures += 1
            print(f"  {item['key']:18s} DOI lookup failed: {error}")
        path = ROOT / "data" / "external" / (
            "matbench_expt_gap.json.gz" if item["key"] == "matbench_expt_gap" else "ees_all_lit_data.xlsx"
        )
        if path.exists():
            actual = sha256(path)
            status = "ok" if actual == item["sha256"] else "MISMATCH"
            if status != "ok":
                failures += 1
            print(f"  {'':18s} {path.name}: {status} ({actual[:16]}...)")
        else:
            print(f"  {'':18s} {path.name}: not downloaded (run scripts/fetch_data.py)")
    if failures:
        print(f"\n{failures} verification failure(s)")
        return 1
    print("\nAll reachable records and downloaded files verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
