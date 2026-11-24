#!/usr/bin/env python3
"""Refit existing models on each outer fold's saved inner-training patients.

Only inner validation predictions are saved. Original outer OOF probabilities
remain untouched. CDC models and the independent Rajpoot reproduction require
their original feature builders; they can be supplied separately to the
evaluation script using its --validation argument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from complete_final_presentation import KEYS, ROOT
from run_cms_group_prior_5fold import build_frames, selected_catboost
from run_dual_external_meps_nhanes import selected_hgb
from run_final_catboost_ablation import DROP
from run_nested_hgb_blend import encode_frames, hgb_model
from run_paper_fair_comparison import fit_catboost, prepare_our_meps_hgb
from run_meps_30d_joint_transfer import encoded_uci_private, fit_joint
from diabetes_project.meps_30d_transfer import common_matrix_meps, common_matrix_uci, load_events

HGB_SELECTION = ROOT / "results/meps_event_ccs_group_hgb_seed777/selected_hgb.csv"
CAT_SELECTION = ROOT / "results/paper_fair_comparison_seed777/catboost_inner_trials.csv"
ABLATION_SELECTION = ROOT / "results/final_presentation_completion"


def hgb_predict(frame, columns, labels, train, validation, selected, fold, args):
    x_train, x_val, _ = encode_frames(frame.loc[train], frame.loc[validation], frame.loc[validation],
                                     columns, 100)
    choice = selected.loc[fold]
    params = json.loads(choice.Parameters)
    params["max_iter"] = int(choice.Iterations)
    model = hgb_model(args.seed + fold, params, early_stopping=False)
    model.fit(x_train, labels[train])
    return model.predict_proba(x_val)[:, 1]


def cat_predict(frame, categorical, labels, train, validation, choice, args):
    _, p = fit_catboost(frame.loc[train].reset_index(drop=True), labels[train],
                        frame.loc[validation].reset_index(drop=True), categorical, choice,
                        SimpleNamespace(threads=args.threads))
    return p


def row_frame(reference, mask, fold, model, probability):
    part = reference.loc[mask, KEYS].copy()
    if len(part) != len(probability):
        raise ValueError(f"Prediction length mismatch: {model}, fold {fold}")
    return part.assign(Model=model, OuterFold=fold, probability=probability)[
        ["Model", "OuterFold", "encounter_id", "patient_nbr", "y_true", "probability"]]


def run(args):
    if args.seed != 777 or not 1 <= args.max_folds <= 5:
        raise ValueError("Use original seed 777 and 1..5 folds")
    reference = pd.read_csv(args.reference_oof, usecols=KEYS)
    assignments = pd.read_csv(args.assignments)
    assignments = assignments.set_index("encounter_id").loc[reference.encounter_id].reset_index()
    if len(reference) != 99343 or not assignments[KEYS].equals(reference):
        raise ValueError("Reference OOF and saved inner assignments differ")
    if reference.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("Outer folds are not patient-disjoint")
    features_args = SimpleNamespace(hospital=args.hospital, ccs_file=args.ccs_file,
                                    cms_prior=args.cms_prior)
    hospital, ccs_columns, cms_cat, categorical, _, _ = build_frames(features_args, reference)
    raw_cat = cms_cat.drop(columns=["cms_readmission_rate", "cms_mapped"])
    cms_hgb_columns = ccs_columns
    ccs_hgb_columns = ccs_columns[:-2]
    meps_args = SimpleNamespace(input=args.hospital, ccs_file=args.ccs_file,
                                event_data=args.event_data, event_ccs=args.event_ccs,
                                our_selected_hgb=HGB_SELECTION)
    meps_hospital, meps_columns, meps_choice = prepare_our_meps_hgb(meps_args, reference)
    ccs_choice = selected_hgb(HGB_SELECTION)
    cat_choice = selected_catboost(CAT_SELECTION)
    labels = reference.y_true.to_numpy(dtype=int)
    if args.include_nn:
        events, linked, groups = load_events(args.event_data, args.event_ccs, 64)
        common_uci = common_matrix_uci(hospital, groups)
        common_meps, private_meps = common_matrix_meps(events, linked, groups)
        source_labels = events.readmission_30d.to_numpy(dtype=int)
        nn_args = SimpleNamespace(seed=777, torch_threads=args.torch_threads,
                                  learning_rate=.001, epochs=12, batch_size=512,
                                  meps_interval=4, external_weight=.3)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, args.max_folds + 1):
        output = args.outdir / f"validation_fold_{fold}.csv.gz"
        if output.exists() and not args.overwrite:
            print(f"[Fold {fold}] already exists: {output}", flush=True)
            continue
        validation = assignments[f"inner_val_{fold}"].eq(1).to_numpy()
        outer_train = reference.Fold.ne(fold).to_numpy()
        if np.any(validation & ~outer_train):
            raise ValueError("Inner validation includes outer test")
        train = outer_train & ~validation
        if set(reference.loc[train, "patient_nbr"]) & set(reference.loc[validation, "patient_nbr"]):
            raise ValueError("Inner validation shares patients with inner training")
        print(f"[Fold {fold}] train={train.sum():,} validation={validation.sum():,}", flush=True)
        predictions = {}
        predictions["ccs_only_hgb"] = hgb_predict(hospital, ccs_hgb_columns, labels, train, validation,
                                                    ccs_choice, fold, args)
        predictions["ccs_meps_group_hgb"] = hgb_predict(meps_hospital, meps_columns, labels, train,
                                                          validation, meps_choice, fold, args)
        predictions["cms_hgb"] = hgb_predict(hospital, cms_hgb_columns, labels, train, validation,
                                             ccs_choice, fold, args)
        predictions["catboost_raw"] = cat_predict(raw_cat, categorical, labels, train, validation,
                                                    cat_choice.loc[fold], args)
        predictions["cms_catboost_raw"] = cat_predict(cms_cat, categorical, labels, train, validation,
                                                        cat_choice.loc[fold], args)
        for name, drop in DROP.items():
            path = ABLATION_SELECTION / f"{name}_fold_{fold}_selection.json"
            if not path.exists():
                print(f"[Fold {fold}] skip {name}: no original selection file", flush=True)
                continue
            frame = raw_cat.drop(columns=drop)
            choice = json.loads(path.read_text())
            predictions[name] = cat_predict(frame, [c for c in categorical if c in frame], labels,
                                             train, validation, choice, args)
        predictions["ccs_only_catboost_raw_50"] = (predictions["ccs_only_hgb"] + predictions["catboost_raw"]) / 2
        predictions["ccs_meps_catboost_raw_50"] = (predictions["ccs_meps_group_hgb"] + predictions["catboost_raw"]) / 2
        predictions["cms_hgb_catboost_raw_50"] = (predictions["cms_hgb"] + predictions["catboost_raw"]) / 2
        predictions["ccs_only_hgb_cms_catboost_50"] = (predictions["ccs_only_hgb"] + predictions["cms_catboost_raw"]) / 2
        predictions["cms_hgb_cms_catboost_50"] = (predictions["cms_hgb"] + predictions["cms_catboost_raw"]) / 2
        if args.include_nn:
            private_uci = encoded_uci_private(raw_cat, categorical, train)
            for external, name in ((False, "uci_only_nn"), (True, "joint_meps_30d")):
                p, _ = fit_joint(common_uci, private_uci, labels, common_meps, private_meps,
                                 source_labels, train, validation, nn_args, fold, external)
                predictions[name] = p
            predictions["joint_meps_30d_catboost50"] = (predictions["joint_meps_30d"] + predictions["catboost_raw"]) / 2
        pieces = [row_frame(reference, validation, fold, name, p) for name, p in predictions.items()]
        pd.concat(pieces, ignore_index=True).to_csv(output, index=False)
        print(f"[Fold {fold}] wrote {len(predictions)} models to {output}", flush=True)
    (args.outdir / "run.json").write_text(json.dumps({
        "seed": 777, "reference_oof": str(args.reference_oof), "assignments": str(args.assignments),
        "hospital": str(args.hospital), "ccs_file": str(args.ccs_file),
        "cms_prior": str(args.cms_prior), "event_data": str(args.event_data),
        "event_ccs": str(args.event_ccs),
        "method": "refit on original inner training patients, predict original inner validation patients",
        "hyperparameters": "reuse outer-fold selections; no outer-test labels used",
        "threads": args.threads,
        "include_nn": args.include_nn,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-oof", type=Path, default=ROOT / "results/paper_fair_comparison_seed777/oof_all.csv.gz")
    parser.add_argument("--assignments", type=Path, default=ROOT.parent / "Bhuvan_Rajpoot_SSRN/5-fold-reproduce/results/five_fold/fold_assignments.csv.gz")
    parser.add_argument("--hospital", type=Path, default=ROOT / "data/raw/uci/diabetic_data.csv")
    parser.add_argument("--ccs-file", type=Path, default=ROOT / "data/raw/ahrq/Single_Level_CCS_2015.zip")
    parser.add_argument("--cms-prior", type=Path, default=ROOT / "data/processed/cms/cms_readmission_ccs_prior.csv")
    parser.add_argument("--event-data", type=Path, default=ROOT / "data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--event-ccs", type=Path, default=ROOT / "data/processed/meps_events_2008_2012/meps_inpatient_event_ccs.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/validation_f1_threshold_seed777")
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--include-nn", action="store_true", help="Also refit UCI-only and MEPS joint neural networks")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
