#!/usr/bin/env python3
"""Download official MEPS inpatient-stay and emergency-room event files."""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError

from _paths import PROJECT_ROOT


EVENT_PUF = {
    2008: 118, 2009: 126, 2010: 135, 2011: 144, 2012: 152,
    2013: 160, 2014: 168, 2015: 178, 2016: 188, 2017: 197,
    2018: 206, 2019: 213, 2020: 220, 2021: 229, 2022: 239,
    2023: 248, 2024: 254,
}
EVENTS = {"inpatient": "d", "er": "e"}
URLS = (
    "https://meps.ahrq.gov/mepsweb/data_files/pufs/{stem}/{stem}{suffix}.zip",
    "https://meps.ahrq.gov/mepsweb/data_files/pufs/{stem}{suffix}.zip",
)


def validate_zip(path, stem, suffix):
    extensions = (".ssp", ".xpt") if suffix == "ssp" else (".dta",)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(extensions)]
        if len(members) != 1 or archive.getinfo(members[0]).file_size <= 0:
            raise ValueError(f"Invalid MEPS event archive {path}: {archive.namelist()}")
        if archive.testzip() is not None:
            raise ValueError(f"CRC validation failed for {path}")
        if stem not in members[0].lower():
            raise ValueError(f"Unexpected event member in {path}: {members[0]}")


def download(year, event, destination, force=False):
    puf = EVENT_PUF[year]; letter = EVENTS[event]
    stem = f"h{puf}{letter}"
    # Old event pages offer XPORT; newer pages offer a directly readable Stata file.
    suffix = "ssp" if year <= 2016 else "dta"
    output = destination / f"{stem}{suffix}.zip"
    if output.exists() and not force:
        validate_zip(output, stem, suffix)
        print(f"[{year} {event}] already present and valid: {output}", flush=True)
        return output
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    last_error = None
    for template in URLS:
        url = template.format(stem=stem, suffix=suffix)
        print(f"[{year} {event}] downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "diabetes-project/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            validate_zip(temporary, stem, suffix)
            temporary.replace(output)
            break
        except HTTPError as error:
            temporary.unlink(missing_ok=True); last_error = error
            if error.code == 404:
                continue
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    else:
        raise last_error or FileNotFoundError(f"No official URL found for HC-{puf}{letter.upper()}")
    print(f"[{year} {event}] saved {output} ({output.stat().st_size / 1024**2:.2f} MiB)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2008, 2025)))
    parser.add_argument("--events", nargs="+", choices=EVENTS, default=list(EVENTS))
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "data/raw/meps/events_2008_2024"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.years).difference(EVENT_PUF))
    if unknown:
        parser.error(f"unsupported years: {unknown}; available: {min(EVENT_PUF)}-{max(EVENT_PUF)}")
    destination = Path(args.output_root); destination.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        for event in args.events:
            try:
                download(year, event, destination, args.force)
            except Exception as error:
                print(f"[{year} {event}] download failed: {error}", file=sys.stderr)
                raise SystemExit(1) from error


if __name__ == "__main__":
    main()
