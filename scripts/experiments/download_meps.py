#!/usr/bin/env python3
"""Download official AHRQ MEPS Full-Year Consolidated SAS transport files."""

import argparse
import shutil
import sys
import urllib.request
from urllib.error import HTTPError
import zipfile
from pathlib import Path

from _paths import PROJECT_ROOT


PUF = {
    2008: 121, 2009: 129, 2010: 138, 2011: 147, 2012: 155,
    2013: 163, 2014: 171, 2015: 181, 2016: 192, 2017: 201,
    2018: 209, 2019: 216, 2020: 224, 2021: 233, 2022: 243,
    2023: 251, 2024: 256,
}
URLS = (
    "https://meps.ahrq.gov/mepsweb/data_files/pufs/h{puf}/h{puf}{suffix}.zip",
    "https://meps.ahrq.gov/mepsweb/data_files/pufs/h{puf}{suffix}.zip",
)


def validate_zip(path, puf, suffix):
    with zipfile.ZipFile(path) as archive:
        expected_extensions = (".ssp", ".xpt") if suffix == "ssp" else (".dta",)
        members = [name for name in archive.namelist() if name.lower().endswith(expected_extensions)]
        if len(members) != 1 or archive.getinfo(members[0]).file_size <= 0:
            raise ValueError(f"Invalid MEPS archive {path}: {archive.namelist()}")
        if archive.testzip() is not None:
            raise ValueError(f"CRC validation failed for {path}")
        expected = f"h{puf}"
        if expected not in members[0].lower():
            raise ValueError(f"Unexpected transport member in {path}: {members[0]}")


def download(year, destination, force=False):
    puf = PUF[year]
    # HC-201 (2017) was released as SAS CPORT rather than XPORT and cannot be
    # read by pandas. Use the official Stata file from 2017 onward.
    suffix = "ssp" if year <= 2016 else "dta"
    output = destination / f"h{puf}{suffix}.zip"
    if output.exists() and not force:
        validate_zip(output, puf, suffix)
        print(f"[{year}] already present and valid: {output}", flush=True)
        return output
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    last_error = None
    for template in URLS:
        url = template.format(puf=puf, suffix=suffix)
        print(f"[{year}] downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "diabetes-project/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            validate_zip(temporary, puf, suffix)
            temporary.replace(output)
            break
        except HTTPError as error:
            temporary.unlink(missing_ok=True)
            last_error = error
            if error.code == 404:
                continue
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    else:
        raise last_error or FileNotFoundError(f"No official download URL found for HC-{puf}")
    print(f"[{year}] saved {output} ({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2025)))
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "data/raw/meps/MEPS_2013_2024"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.years).difference(PUF))
    if unknown:
        parser.error(f"unsupported years: {unknown}; available: {min(PUF)}-{max(PUF)}")
    destination = Path(args.output_root); destination.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        try:
            download(year, destination, args.force)
        except Exception as error:
            print(f"[{year}] download failed: {error}", file=sys.stderr)
            raise SystemExit(1) from error


if __name__ == "__main__":
    main()
