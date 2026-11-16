#!/usr/bin/env python3
"""Download and normalise the external datasets used by the real benchmarks.

Nothing is redistributed with this repository.  The script verifies SHA-256
hashes, writes a provenance manifest, and converts the EES workbook to CSV.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "external"

MATBENCH_URL = "https://ndownloader.figshare.com/files/17494814"
MATBENCH_SHA256 = "1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a"
EES_URL = (
    "https://raw.githubusercontent.com/yousof96/ligh_intensity_paper/master/"
    "paper_data/all_lit_data.xlsx"
)
EES_SHA256 = "9081eef866c4b14bc20a13ab0f6bccb5a4fe0d43474d21416175b7b9782beb03"

EES_COLUMNS = (
    "id",
    "paper",
    "description",
    "eg",
    "her",
    "cocatalyst",
    "cocatalyst_loading",
    "surface_area",
    "sacrificial_agent",
    "sacrificial_concentration",
    "light_intensity",
    "synthesis",
    "rate_unit",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected: str, force: bool) -> Path:
    if destination.exists() and not force:
        actual = sha256(destination)
        if actual == expected:
            print(f"cached  {destination.relative_to(ROOT)}")
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "photocat/0.2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    destination.write_bytes(data)
    actual = sha256(destination)
    if actual != expected:
        raise SystemExit(
            f"hash mismatch for {destination.name}: expected {expected}, got {actual}"
        )
    print(f"saved    {destination.relative_to(ROOT)} ({len(data)} bytes)")
    return destination


def ees_to_csv(workbook: Path, destination: Path) -> int:
    try:
        import openpyxl
    except ImportError:
        raise SystemExit(
            "openpyxl is required for the EES workbook: pip install openpyxl"
        )
    book = openpyxl.load_workbook(workbook, data_only=True, read_only=True)
    sheet = book.active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    index = {str(value).strip(): position for position, value in enumerate(header) if value}

    def value(row: tuple[object, ...], name: str) -> object:
        position = index.get(name)
        return row[position] if position is not None and position < len(row) else None

    written = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EES_COLUMNS)
        writer.writeheader()
        for row in rows:
            rate = value(row, "Rate")
            surface = value(row, "SA")
            light = value(row, "LI")
            gap = value(row, "BG")
            if any(v is None for v in (rate, surface, light, gap)):
                continue
            try:
                float(rate), float(surface), float(light), float(gap)
            except (TypeError, ValueError):
                continue
            writer.writerow(
                {
                    "id": value(row, "ID") or "",
                    "paper": value(row, "Paper_title") or "",
                    "description": value(row, "Description") or "",
                    "eg": gap,
                    "her": rate,
                    "cocatalyst": value(row, "MT") or "Bare",
                    "cocatalyst_loading": value(row, "ML") or 0.0,
                    "surface_area": surface,
                    "sacrificial_agent": value(row, "AT") or "unknown",
                    "sacrificial_concentration": value(row, "AC") or 0.0,
                    "light_intensity": light,
                    "synthesis": value(row, "Synthesis") or "unknown",
                    "rate_unit": "source Rate column; units not harmonised",
                }
            )
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    matbench = download(
        MATBENCH_URL, out / "matbench_expt_gap.json.gz", MATBENCH_SHA256, args.force
    )
    ees_workbook = download(EES_URL, out / "ees_all_lit_data.xlsx", EES_SHA256, args.force)
    ees_csv = out / "ees_her.csv"
    count = ees_to_csv(ees_workbook, ees_csv)
    print(f"wrote    {ees_csv.relative_to(ROOT)} ({count} complete rows)")

    manifest = {
        "retrieved": date.today().isoformat(),
        "files": {
            "matbench_expt_gap": {
                "path": str(matbench.relative_to(ROOT)),
                "url": MATBENCH_URL,
                "sha256": MATBENCH_SHA256,
                "license": "MIT",
                "doi": "10.6084/m9.figshare.9765779",
                "citation": (
                    "Dunn et al., Benchmarking materials property prediction methods: "
                    "the Matbench test set and Automatminer reference algorithm, "
                    "npj Comput. Mater. 6, 138 (2020). Data derived from Zhuo et al., "
                    "J. Phys. Chem. Lett. 9, 1668-1673 (2018), doi:10.1021/acs.jpclett.8b00124."
                ),
            },
            "ees_her": {
                "path": str(ees_csv.relative_to(ROOT)),
                "workbook": str(ees_workbook.relative_to(ROOT)),
                "url": EES_URL,
                "sha256": EES_SHA256,
                "license": "No licence declared upstream; not redistributed with photocat.",
                "doi": "10.1039/D3EY00246B",
                "citation": (
                    "Haghshenas et al., Predicting the rates of photocatalytic hydrogen "
                    "evolution over cocatalyst-deposited TiO2 using machine learning with "
                    "active photon flux as a unifying feature, EES Catalysis 2, 612-623 "
                    "(2024), doi:10.1039/D3EY00246B."
                ),
            },
        },
    }
    (out / "SOURCES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote    {(out / 'SOURCES.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
