#!/usr/bin/env python3
"""Download complete, real CMS hospital readmission aggregate tables via the official API.

The resulting CSVs contain hospital- or national-level quality measures, not
patient-level 30-day readmission labels. See docs/hcupnet_cms_external_feasibility.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from _paths import PROJECT_ROOT


DATASETS = {
    "hospital": "632h-zaca",
    "national": "cvcs-xecj",
}
API_ROOT = "https://data.cms.gov/provider-data/api/1/datastore/query"
CATALOG_ROOT = "https://data.cms.gov/provider-data/dataset"


def request_page(dataset_id: str, offset: int, limit: int) -> tuple[dict, str]:
    url = f"{API_ROOT}/{dataset_id}/0?{urlencode({'limit': limit, 'offset': offset})}"
    request = Request(url, headers={"User-Agent": "diabetes-project/1.0", "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError(f"Unexpected CMS API response at {url}")
            return payload, url
        except (OSError, TimeoutError) as error:
            if attempt == 3:
                raise RuntimeError(f"CMS download failed at offset {offset}: {error}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def validate_existing(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not any(name.lower().replace(" ", "_") == "measure_id" for name in reader.fieldnames):
            raise ValueError(f"CMS CSV is missing Measure ID: {path}")
        return sum(1 for _ in reader)


def download_dataset(kind: str, outdir: Path, page_size: int, force: bool) -> Path:
    dataset_id = DATASETS[kind]
    destination = outdir / f"unplanned_hospital_visits_{kind}.csv"
    metadata_path = outdir / f"unplanned_hospital_visits_{kind}.source.json"
    if destination.exists() and not force:
        count = validate_existing(destination)
        print(f"[{kind}] already present: {destination} ({count:,} rows)", flush=True)
        return destination

    temporary = outdir / f".{destination.name}.{os.getpid()}.part"
    temporary.unlink(missing_ok=True)
    offset = 0
    total = None
    fieldnames = None
    measure_counts: Counter[str] = Counter()
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = None
            while True:
                payload, _ = request_page(dataset_id, offset, page_size)
                rows = payload["results"]
                if total is None and payload.get("count") is not None:
                    total = int(payload["count"])
                if not rows:
                    break
                if not all(isinstance(row, dict) for row in rows):
                    raise ValueError(f"CMS API returned a non-object row at offset {offset}")
                if writer is None:
                    fieldnames = list(rows[0])
                    if not any(name.lower().replace(" ", "_") == "measure_id" for name in fieldnames):
                        raise ValueError(f"CMS API did not return Measure ID: {fieldnames}")
                    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
                    writer.writeheader()
                writer.writerows(rows)
                measure_key = next(name for name in fieldnames if name.lower().replace(" ", "_") == "measure_id")
                measure_counts.update(str(row.get(measure_key, "")) for row in rows)
                offset += len(rows)
                print(f"[{kind}] downloaded {offset:,}" + (f"/{total:,}" if total is not None else ""), flush=True)
                if total is not None and offset >= total:
                    break
                if len(rows) < page_size:
                    break
        if offset == 0 or (total is not None and offset != total):
            raise ValueError(f"Incomplete CMS {kind} download: {offset} rows, expected {total}")
        if validate_existing(temporary) != offset:
            raise ValueError(f"CSV row count changed while saving {kind}")
        with temporary.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        temporary.replace(destination)
        metadata = {
            "source": f"{CATALOG_ROOT}/{dataset_id}",
            "api": f"{API_ROOT}/{dataset_id}/0",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "unit": "hospital-by-measure" if kind == "hospital" else "national-by-measure",
            "rows": offset,
            "columns": fieldnames,
            "sha256": sha256,
            "measure_ids": dict(sorted(measure_counts.items())),
            "note": "Aggregate CMS quality measures; not patient-level readmission labels",
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[{kind}] saved {destination} ({offset:,} rows)", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "data/raw/cms")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.page_size <= 1500:
        parser.error("--page-size must be between 1 and 1500")
    args.outdir.mkdir(parents=True, exist_ok=True)
    for kind in args.datasets:
        download_dataset(kind, args.outdir, args.page_size, args.force)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"Download incomplete: {error}", file=sys.stderr)
        raise SystemExit(1) from None
