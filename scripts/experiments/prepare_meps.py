
import argparse
import io
import zipfile
from pathlib import Path
from _paths import PROJECT_ROOT

import numpy as np
import pandas as pd


FILES = {
    2008: "h121ssp.zip",
    2009: "h129ssp.zip",
    2010: "h138ssp.zip",
    2011: "h147ssp.zip",
    2012: "h155ssp.zip",
    2013: "h163ssp.zip",
    2014: "h171ssp.zip",
    2015: "h181ssp.zip",
    2016: "h192ssp.zip",
    2017: "h201dta.zip",
    2018: "h209dta.zip",
    2019: "h216dta.zip",
    2020: "h224dta.zip",
    2021: "h233dta.zip",
    2022: "h243dta.zip",
    2023: "h251dta.zip",
    2024: "h256dta.zip",
}

# Common variables across the five MEPS years.
COMMON = {
    "age": "AGE42X",
    "sex": "SEX",
    "high_bp": "HIBPDX",
    "high_chol": "CHOLDX",
    "stroke": "STRKDX",
    "heart_disease": "CHDDX",
    "mi": "MIDX",
    "physical_health": "PCS42",
    "mental_health": "MCS42",
    "general_health": "RTHLTH42",
    "activity_limitation": "ADCLIM42",
}

COLUMN_ALIASES = {
    "DIABDX": ("DIABDX", "DIABDX_M18"),
    "PCS42": ("PCS42", "VPCS42"),
    "MCS42": ("MCS42", "VMCS42"),
}

# yy is the two-digit year suffix.
YEARLY = {
    "er_visits": "ERTOT{yy}",
    "inpatient_nights": "IPNGTD{yy}",
    "inpatient_expense": "IPTEXP{yy}",
    "er_expense": "ERTEXP{yy}",
    "total_expense": "TOTEXP{yy}",
    "person_weight": "PERWT{yy}F",
}

TARGETS = [
    "er_visits",
    "inpatient_nights",
    "inpatient_expense",
    "er_expense",
    "total_expense",
]

BINARY = [
    "high_bp",
    "high_chol",
    "stroke",
    "heart_disease",
    "mi",
]

RACE_MAP = {
    1: "Hispanic",
    2: "Caucasian",
    3: "AfricanAmerican",
    4: "Asian",
    5: "Other",
}


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        matches = [
            name for name in z.namelist()
            if name.lower().endswith(
                (".ssp", ".xpt", ".xport", ".v9x", ".dta")
            )
        ]

        if len(matches) != 1:
            raise ValueError(
                f"{path.name}: unexpected files: "
                f"{z.namelist()}"
            )

        raw = z.read(matches[0])

    if matches[0].lower().endswith(".dta"):
        df = pd.read_stata(io.BytesIO(raw), convert_categoricals=False)
    else:
        df = pd.read_sas(
            io.BytesIO(raw),
            format="xport",
            encoding="latin-1",
        )

    df.columns = df.columns.str.upper()

    return df


def clean_numeric(series):
    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    # Some legacy SAS XPORT zeroes are decoded by pandas as the smallest
    # positive IBM float (~5.4e-79). Normalize them before deriving targets.
    x = x.mask(x.abs() < 1e-50, 0.0)

    # MEPS uses negative codes for missing,
    # not applicable, refused, etc.
    return x.mask(x < 0)


def clean_binary_diagnosis(series):
    # MEPS diagnosis variables commonly use:
    # 1 = Yes, 2 = No.
    # Negative codes must remain missing.
    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    return x.map({
        1: 1.0,
        2: 0.0,
    })


def make_person_key(df):
    panel = pd.to_numeric(
        df["PANEL"],
        errors="coerce",
    ).astype("Int64").astype("string")

    dupersid = (
        df["DUPERSID"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    # Beginning with the 2018 PUF, DUPERSID is commonly stored as a 10-digit
    # value prefixed by the two-digit PANEL. Older files use the underlying
    # 8-digit identifier. Remove only the verified redundant prefix so the
    # same panel member links across the 2017/2018 format boundary.
    redundant_panel = dupersid.str.len().eq(10) & dupersid.str[:2].eq(panel)
    dupersid = dupersid.where(~redundant_panel, dupersid.str[2:])

    if (
        panel.isna().any()
        or dupersid.isna().any()
        or dupersid.eq("").any()
    ):
        raise ValueError(
            "Missing PANEL or DUPERSID identifiers"
        )

    return panel + "_" + dupersid


def find_archive(root, year, filename):
    preferred = root / ("MEPS_2008_2012" if year <= 2012 else "MEPS_2013_2024") / filename
    if preferred.exists():
        return preferred
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Missing {year} MEPS archive {filename} below {root}")
    return matches[0]


def process_year(root, year, filename):
    archive = find_archive(root, year, filename)
    print(f"Source: {archive}", flush=True)
    df = read_zip(archive)

    race_column = next((name for name in ("RACETHX", "RACETHNX") if name in df.columns), None)
    if race_column is None:
        raise ValueError(f"{year} missing race/ethnicity column (RACETHX or RACETHNX)")

    def resolve(canonical):
        choices = COLUMN_ALIASES.get(canonical, (canonical,))
        return next((name for name in choices if name in df.columns), canonical)

    diabetes_column = resolve("DIABDX")
    resolved_common = {name: resolve(column) for name, column in COMMON.items()}

    required = [
        "PANEL",
        "DUPERSID",
        diabetes_column,
        *resolved_common.values(),
    ]

    for template in YEARLY.values():
        required.append(
            template.format(
                yy=str(year)[-2:]
            )
        )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{year} missing columns: {missing}"
        )

    diabetes = pd.to_numeric(
        df[diabetes_column],
        errors="coerce",
    )

    df = df.loc[
        diabetes == 1
    ].copy()

    out = pd.DataFrame(
        index=df.index
    )

    out["year"] = year
    out["person_key"] = make_person_key(df)

    for name, col in resolved_common.items():
        if name in BINARY:
            out[name] = clean_binary_diagnosis(
                df[col]
            )
        else:
            out[name] = clean_numeric(
                df[col]
            )

    race = pd.to_numeric(df[race_column], errors="coerce")
    out["race_common"] = race.map(RACE_MAP).astype("string")

    yy = str(year)[-2:]

    for name, template in YEARLY.items():
        col = template.format(yy=yy)

        out[name] = clean_numeric(
            df[col]
        )

    out["er_any"] = np.where(
        out["er_visits"].notna(),
        (out["er_visits"] > 0).astype(float),
        np.nan,
    )

    out["inpatient_any"] = np.where(
        out["inpatient_nights"].notna(),
        (
            out["inpatient_nights"] > 0
        ).astype(float),
        np.nan,
    )

    print(
        f"[{year}] Diabetes rows: {len(out):,}; "
        f"unique people: "
        f"{out['person_key'].nunique():,}",
        flush=True,
    )

    return out.reset_index(drop=True)


def summarize(df, root):
    print(
        "\n========== SAMPLE COUNTS ==========",
        flush=True,
    )

    print(
        "Person-year records:",
        f"{len(df):,}",
    )

    print(
        "Unique persons:",
        f"{df['person_key'].nunique():,}",
    )

    print(
        "\n========== MISSINGNESS ==========",
        flush=True,
    )

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print(
        missing.round(2).to_string(),
        flush=True,
    )

    missing.rename(
        "MissingPercent"
    ).to_csv(
        root / "meps_missingness.csv"
    )

    print(
        "\n========== UTILIZATION TARGETS ==========",
        flush=True,
    )

    rows = []

    for col in TARGETS:
        x = df[col].dropna()

        rows.append({
            "Feature": col,
            "ValidRows": len(x),
            "MissingPercent":
                100 * df[col].isna().mean(),
            "Mean": x.mean(),
            "Median": x.median(),
            "P90": x.quantile(0.90),
            "P99": x.quantile(0.99),
            "Max": x.max(),
            "PositivePercent":
                100 * (x > 0).mean(),
        })

    table = pd.DataFrame(rows)

    print(
        table.round(3).to_string(
            index=False
        ),
        flush=True,
    )

    table.to_csv(
        root / "meps_utilization_summary.csv",
        index=False,
    )

    print(
        "\n========== BY YEAR ==========",
        flush=True,
    )

    yearly = df.groupby(
        "year"
    ).agg(
        N=("person_key", "size"),
        UniquePersons=("person_key", "nunique"),
        MeanAge=("age", "mean"),
        ERAnyRate=("er_any", "mean"),
        InpatientAnyRate=("inpatient_any", "mean"),
        MeanERN=("er_visits", "mean"),
        MeanInpatientNights=(
            "inpatient_nights", "mean"
        ),
    )

    print(
        yearly.round(4).to_string(),
        flush=True,
    )

    yearly.to_csv(
        root / "meps_harmonized_yearly.csv"
    )

    print(
        "\n========== BRIDGE FEATURE PREVALENCE ==========",
        flush=True,
    )

    cols = [
        "high_bp",
        "high_chol",
        "stroke",
        "heart_disease",
    ]

    prevalence = df[cols].agg([
        "count",
        "mean",
    ]).T

    print(
        prevalence.round(4).to_string(),
        flush=True,
    )

    prevalence.to_csv(
        root / "meps_bridge_prevalence.csv"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        default=str(PROJECT_ROOT / "data/raw/meps"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data/processed/meps"),
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=sorted(FILES),
    )

    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    chunks = []

    unknown = sorted(set(args.years).difference(FILES))
    if unknown:
        parser.error(f"unsupported years: {unknown}")

    for year in args.years:
        filename = FILES[year]
        print(
            f"\n========== LOAD {year} ==========",
            flush=True,
        )

        chunks.append(
            process_year(
                input_root,
                year,
                filename,
            )
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df.to_csv(
        output_root / "meps_diabetes_harmonized.csv",
        index=False,
    )
    df.to_csv(
        output_root / "meps_diabetes_bridge8.csv",
        index=False,
    )

    summarize(df, output_root)

    print(
        "\n========== FINISHED ==========",
        flush=True,
    )

    print(
        "Saved:",
        output_root / "meps_diabetes_harmonized.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
