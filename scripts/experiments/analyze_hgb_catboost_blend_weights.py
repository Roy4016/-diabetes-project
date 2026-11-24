#!/usr/bin/env python3
"""Evaluate prespecified HGB/CatBoost blend weights on the same five-fold OOF.

No model is retrained. Each probability is a fixed weighted average of the
corresponding held-out HGB and CatBoost probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_paper_fair_comparison import paired_patient_bootstrap, summarize  # noqa: E402

KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]
HGB = "ccs_only_hgb"
CATBOOST = {
    "baseline": ("catboost_raw_plus_ccs", "ccs_hgb_catboost_raw_plus_ccs_50"),
    "diabetes_meps": ("diabetes_meps_teacher_catboost", "diabetes_meps_teacher_hgb50"),
    "all_disease_meps": ("all_disease_meps_teacher_catboost", "all_disease_meps_teacher_hgb50"),
}


def run(args):
    if not args.oof.exists():
        raise FileNotFoundError(f"OOF file not found: {args.oof}")
    frame = pd.read_csv(args.oof)
    required = set(KEYS + [HGB])
    if missing := required - set(frame):
        raise ValueError(f"OOF is missing {sorted(missing)}")
    if len(frame) != 99343 or frame.encounter_id.duplicated().any():
        raise ValueError("Expected exactly 99,343 unique UCI encounters")
    if set(frame.Fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected all five original outer folds")
    if frame.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Patient appears in multiple outer folds")
    if frame.y_true.isna().any() or not frame.y_true.isin([0, 1]).all():
        raise ValueError("Labels are missing or nonbinary")
    if len(set(args.hgb_weights)) != len(args.hgb_weights):
        raise ValueError("Duplicate HGB weights")
    if any(not 0 <= weight <= 1 for weight in args.hgb_weights):
        raise ValueError("Every HGB weight must be between zero and one")
    if .5 not in args.hgb_weights:
        raise ValueError("Include 0.5 to retain the saved baseline comparison")
    if not 0 < args.alert_fraction < 1 or args.bootstrap < 0:
        raise ValueError("Invalid alert fraction or bootstrap count")

    available = {group: spec for group, spec in CATBOOST.items() if spec[0] in frame}
    if "baseline" not in available:
        raise ValueError("No saved CCS HGB + CCS CatBoost baseline probabilities")
    output = frame[KEYS].copy()
    output[HGB] = frame[HGB]
    model_columns = []
    for group, (cat_name, saved_50) in available.items():
        probabilities = frame[[HGB, cat_name]].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(f"Missing or invalid probabilities for {group}")
        if saved_50 in frame and not np.allclose(frame[saved_50], probabilities.mean(axis=1),
                                                 rtol=0, atol=1e-12):
            raise ValueError(f"Saved 50/50 probabilities do not match {group}")
        for weight in args.hgb_weights:
            pct = round(weight * 100)
            if abs(weight * 100 - pct) > 1e-9:
                raise ValueError("Use whole-percentage HGB weights for stable output names")
            name = f"{group}_hgb{pct:02d}_cat{100-pct:02d}"
            output[name] = weight * probabilities[:, 0] + (1 - weight) * probabilities[:, 1]
            model_columns.append(name)
    args.outdir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.outdir / "oof_blend_weights.csv.gz", index=False)
    summary = summarize(output, model_columns, args.outdir, args.alert_fraction)
    metadata = []
    for group in available:
        for weight in args.hgb_weights:
            pct = round(weight * 100)
            metadata.append({"Model": f"{group}_hgb{pct:02d}_cat{100-pct:02d}",
                             "Group": group, "HGB_Weight": weight, "CatBoost_Weight": 1 - weight})
    summary = summary.merge(pd.DataFrame(metadata), on="Model", validate="one_to_one")
    summary.to_csv(args.outdir / "blend_weight_metrics.csv", index=False)
    pairs = []
    for weight in args.hgb_weights:
        pct = round(weight * 100)
        suffix = f"hgb{pct:02d}_cat{100-pct:02d}"
        if weight != .5:
            pairs.append((f"baseline_{suffix}", "baseline_hgb50_cat50"))
        for group in ("diabetes_meps", "all_disease_meps"):
            if group in available:
                pairs.append((f"{group}_{suffix}", f"baseline_{suffix}"))
        if "diabetes_meps" in available and "all_disease_meps" in available:
            pairs.append((f"all_disease_meps_{suffix}", f"diabetes_meps_{suffix}"))
    paired_patient_bootstrap(output, pairs, args.outdir, args.bootstrap, args.seed)
    (args.outdir / "config.json").write_text(json.dumps({
        "source_oof": str(args.oof), "seed": args.seed,
        "weights_hgb": args.hgb_weights, "available_groups": list(available),
        "fusion": "HGB_Weight * saved HGB OOF + (1 - HGB_Weight) * saved CatBoost OOF",
        "threshold": f"top {args.alert_fraction:.0%} within each outer test fold",
        "bootstrap": args.bootstrap,
        "interpretation": "Prespecified descriptive weight sweep; choosing the best weight from these held-out folds requires new validation",
    }, indent=2) + "\n")
    print(f"Saved blend analysis to {args.outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=ROOT / "results/ccs_catboost_5fold_seed777/oof_all.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/hgb_catboost_blend_weights_seed777")
    parser.add_argument("--hgb-weights", type=float, nargs="+", default=[.3, .4, .5, .6, .7])
    parser.add_argument("--alert-fraction", type=float, default=.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
