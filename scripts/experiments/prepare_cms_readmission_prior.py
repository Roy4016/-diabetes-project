#!/usr/bin/env python3
"""Build an auditable CCS lookup from real CMS national readmission measures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _paths import CCS_RAW, PROJECT_ROOT, UCI_RAW
from train_xgboost_ahrq_ccs_prefix import load_ccs, make_prefix_map, map_ccs, norm_icd9

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from diabetes_project.paper_comparison import load_comparison_cohort  # noqa: E402


# These are clinical concept approximations, not exact CMS measure eligibility.
# CABG and hip/knee measures are excluded because the UCI table has no procedure codes.
CCS_TO_CMS = {
    "CCS_100": ("READM_30_AMI", "Acute myocardial infarction"),
    "CCS_108": ("READM_30_HF", "Nonhypertensive congestive heart failure"),
    "CCS_122": ("READM_30_PN", "Pneumonia excluding TB/STI"),
    "CCS_127": ("READM_30_COPD", "COPD and bronchiectasis"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cms-national", type=Path, default=PROJECT_ROOT / "data/raw/cms/unplanned_hospital_visits_national.csv")
    parser.add_argument("--cms-hospital", type=Path, default=PROJECT_ROOT / "data/raw/cms/unplanned_hospital_visits_hospital.csv")
    parser.add_argument("--uci", type=Path, default=UCI_RAW)
    parser.add_argument("--ccs", type=Path, default=CCS_RAW)
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "data/processed/cms")
    args = parser.parse_args()

    national = pd.read_csv(args.cms_national, dtype=str)
    hospitals = pd.read_csv(args.cms_hospital, dtype=str)
    required_national = {"measure_id", "national_rate", "start_date", "end_date"}
    required_hospital = {"measure_id", "facility_id", "score"}
    if missing := required_national - set(national):
        raise ValueError(f"CMS national file missing: {sorted(missing)}")
    if missing := required_hospital - set(hospitals):
        raise ValueError(f"CMS hospital file missing: {sorted(missing)}")

    ccs_map, _ = load_ccs(args.ccs)
    prefixes, ambiguous = make_prefix_map(ccs_map)
    cohort = load_comparison_cohort(args.uci)
    diag = cohort.diag_1.map(norm_icd9).map(lambda code: map_ccs(code, ccs_map, prefixes, ambiguous)[0])
    rows = []
    for ccs, (measure, concept) in CCS_TO_CMS.items():
        match = national.loc[national.measure_id.eq(measure)]
        if len(match) != 1:
            raise ValueError(f"Expected one national row for {measure}; found {len(match)}")
        rate = pd.to_numeric(match.iloc[0].national_rate, errors="coerce")
        if pd.isna(rate) or not 0 <= rate <= 100:
            raise ValueError(f"Invalid national rate for {measure}: {match.iloc[0].national_rate}")
        hospital = hospitals.loc[hospitals.measure_id.eq(measure)]
        scores = pd.to_numeric(hospital.score, errors="coerce")
        rows.append({
            "ccs": ccs,
            "ccs_concept": concept,
            "cms_measure_id": measure,
            "cms_national_rate_pct": float(rate),
            "cms_hospital_score_available": int(scores.notna().sum()),
            "cms_hospital_score_total": int(len(hospital)),
            "cms_hospital_score_median_pct": float(scores.median()),
            "uci_principal_diagnosis_rows": int(diag.eq(ccs).sum()),
            "cms_start_date": match.iloc[0].start_date,
            "cms_end_date": match.iloc[0].end_date,
            "mapping_quality": "approximate clinical concept; CMS risk-standardized measure eligibility differs",
        })
    output = pd.DataFrame(rows)
    periods = output[["cms_start_date", "cms_end_date"]].drop_duplicates()
    if len(periods) != 1:
        raise ValueError("CMS readmission measures have inconsistent reporting periods")
    args.outdir.mkdir(parents=True, exist_ok=True)
    target = args.outdir / "cms_readmission_ccs_prior.csv"
    output.to_csv(target, index=False)
    audit = {
        "source_national": str(args.cms_national),
        "source_hospital": str(args.cms_hospital),
        "source_ahrq_ccs": str(args.ccs),
        "uci_rows": int(len(cohort)),
        "uci_principal_covered_rows": int(diag.isin(CCS_TO_CMS).sum()),
        "uci_principal_coverage": float(diag.isin(CCS_TO_CMS).mean()),
        "measure_period": {
            "start_date": str(periods.iloc[0].cms_start_date),
            "end_date": str(periods.iloc[0].cms_end_date),
        },
        "unit": "CCS concept to CMS national risk-standardized hospital readmission percent",
        "caveat": "These are real aggregate external data, not encounter labels or calibrated UCI probabilities.",
    }
    (args.outdir / "cms_readmission_ccs_prior.audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(output.to_string(index=False))
    print(f"Saved {target}; UCI diag_1 coverage={audit['uci_principal_covered_rows']:,}/{len(cohort):,} ({audit['uci_principal_coverage']:.1%})")


if __name__ == "__main__":
    main()
