#!/usr/bin/env python3
"""Compare HGB with CCS diagnoses and MEPS next-year CCS group statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from _paths import CCS_RAW, MEPS_BRIDGE, UCI_RAW, result_dir
from analyze_nested_hgb_blend import cluster_bootstrap
from run_cdc_encoder_control import prepare_hospital
from run_meps_hgb import raw_discharge_predictors
from run_nested_hgb_blend import choose_hgb, encode_frames, hgb_model
from train_xgboost_ahrq_ccs_prefix import load_ccs, make_prefix_map, map_ccs, norm_icd9


DIAG = ("diag_1", "diag_2", "diag_3")
STATS = ("meps_ccs_er_mean", "meps_ccs_ip_mean", "meps_ccs_er_max", "meps_ccs_ip_max")
EVENT_STATS = ("meps_ccs_readmit_mean", "meps_ccs_revisit_mean",
               "meps_ccs_readmit_max", "meps_ccs_revisit_max")
SOFT_STATS = tuple(
    f"meps_soft_{metric}_{diag}"
    for diag in DIAG
    for metric in ("readmit", "revisit", "log_support", "level")
)


def icd_prefix(value):
    code = norm_icd9(value)
    return code[:4] if code.startswith("E") else code[:3]


def add_soft_event_features(frame, events, event_ccs, rates, global_rates,
                            min_prefix=10, min_parent=30):
    """Use exact CCS, MEPS P(CCS|ICD prefix), then broad ICD group rates."""
    if "icd9_prefix" not in event_ccs:
        raise ValueError("Event CCS data lacks icd9_prefix; rerun prepare_meps_event_ccs.py")
    linked = event_ccs.copy()
    linked["prefix"] = linked.icd9_prefix.map(icd_prefix)
    linked = linked.loc[linked.prefix.ne("")].copy()
    linked["ccs"] = "CCS_" + linked.ccs.astype(int).astype(str)
    linked["event_id"] = linked.event_id.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    events = events.copy()
    events["event_id"] = events.event_id.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    linked = linked.merge(
        events[["year", "person_key", "event_id", "readmission_30d", "er_revisit_30d"]],
        on=["year", "person_key", "event_id"], how="inner", validate="many_to_one",
    )
    linked = linked.loc[linked[["readmission_30d", "er_revisit_30d"]].isin([0, 1]).all(axis=1)]
    linked = linked.drop_duplicates(["year", "person_key", "event_id", "prefix", "ccs"])
    if linked.empty:
        raise ValueError("No linked events have valid ICD prefixes and 30-day labels")

    ccs_rates = rates.set_index("ccs")
    ccs_risk = {
        ccs: (float(row.next_er), float(row.next_ip), int(row.n))
        for ccs, row in ccs_rates.iterrows()
    }
    # Each prefix may correspond to several CCS categories. Use the observed
    # MEPS distribution instead of forcing an ambiguous code into one category.
    prefix_risk = {}
    for prefix, group in linked.groupby("prefix"):
        counts = group.ccs.value_counts()
        total = int(counts.sum())
        posterior = np.array([counts[ccs] for ccs in counts.index], dtype=float)
        posterior /= posterior.sum()
        readmit = sum(p * ccs_risk.get(ccs, (global_rates["next_er"], 0, 0))[0]
                      for p, ccs in zip(posterior, counts.index))
        revisit = sum(p * ccs_risk.get(ccs, (0, global_rates["next_ip"], 0))[1]
                      for p, ccs in zip(posterior, counts.index))
        prefix_risk[prefix] = (readmit, revisit, total)

    linked["parent"] = linked.prefix.str[:2]
    parent_risk = {}
    for parent, group in linked.groupby("parent"):
        group = group.drop_duplicates(["year", "person_key", "event_id"])
        n = len(group)
        readmit = (group.readmission_30d.sum() + 50 * global_rates["next_er"]) / (n + 50)
        revisit = (group.er_revisit_30d.sum() + 50 * global_rates["next_ip"]) / (n + 50)
        parent_risk[parent] = (float(readmit), float(revisit), n)

    output = frame.copy()
    diagnostics = {}
    for diag in DIAG:
        ccs_col = f"ccs_{diag}"
        prefixes = output[diag].map(icd_prefix)
        values = np.zeros((len(output), 4), dtype=np.float32)
        levels = np.zeros(4, dtype=int)
        for index, (prefix, ccs) in enumerate(zip(prefixes, output[ccs_col])):
            if ccs in ccs_risk:
                readmit, revisit, n = ccs_risk[ccs]
                level = 3
            elif prefix in prefix_risk and prefix_risk[prefix][2] >= min_prefix:
                readmit, revisit, n = prefix_risk[prefix]
                level = 2
            elif prefix[:2] in parent_risk and parent_risk[prefix[:2]][2] >= min_parent:
                readmit, revisit, n = parent_risk[prefix[:2]]
                level = 1
            else:
                readmit, revisit, n = global_rates["next_er"], global_rates["next_ip"], 0
                level = 0
            values[index] = (readmit, revisit, np.log1p(n), level)
            levels[level] += 1
        for offset, metric in enumerate(("readmit", "revisit", "log_support", "level")):
            output[f"meps_soft_{metric}_{diag}"] = values[:, offset]
        diagnostics[diag] = {str(level): int(count) for level, count in enumerate(levels)}
    return output, diagnostics


def attach_uci_ccs(frame, mapping):
    prefixes, ambiguous = make_prefix_map(mapping)
    out = frame.copy()
    for source in DIAG:
        out[f"ccs_{source}"] = out[source].map(norm_icd9).map(
            lambda code: map_ccs(code, mapping, prefixes, ambiguous)[0]
        )
    return out


def make_meps_group_rates(annual, conditions, min_group, alpha):
    annual = annual.sort_values(["person_key", "year"]).copy()
    groups = annual.groupby("person_key", sort=False)
    annual["next_year"] = groups.year.shift(-1)
    annual["next_er"] = groups.er_any.shift(-1)
    annual["next_ip"] = groups.inpatient_any.shift(-1)
    pairs = annual.loc[
        annual.next_year.eq(annual.year + 1)
        & annual[["next_er", "next_ip"]].isin([0, 1]).all(axis=1)
    ].copy()
    if pairs.empty:
        raise ValueError("No consecutive-year MEPS pairs")
    merged = pairs.merge(conditions, on=["year", "person_key"], how="inner", validate="one_to_many")
    if merged.empty:
        raise ValueError("No MEPS CCS conditions matched consecutive-year pairs")
    merged = merged.drop_duplicates(["year", "person_key", "ccs"])
    merged["ccs"] = "CCS_" + merged.ccs.astype(int).astype(str)
    # One person can contribute multiple CCS groups; each group statistic is
    # computed from unique person-year observations, with survey weights.
    weight = pd.to_numeric(merged.person_weight, errors="coerce")
    merged["weight"] = weight.where(weight.gt(0), 1.).fillna(1.)
    global_rates = {
        target: float(np.average(pairs[target], weights=pairs.person_weight.clip(lower=1)))
        for target in ("next_er", "next_ip")
    }
    records = []
    for category, group in merged.groupby("ccs"):
        n = len(group)
        if n < min_group:
            continue
        row = {"ccs": category, "n": n}
        for target in ("next_er", "next_ip"):
            weights = group.weight.to_numpy(dtype=float)
            weighted_sum = float(np.dot(weights, group[target].to_numpy(dtype=float)))
            row[target] = (weighted_sum + alpha * global_rates[target] * weights.mean()) / (
                weights.sum() + alpha * weights.mean()
            )
        records.append(row)
    rates = pd.DataFrame(records)
    if rates.empty:
        raise ValueError("No MEPS CCS groups reached --min-group")
    return rates, global_rates, len(pairs), len(merged)


def make_event_group_rates(events, event_ccs, min_group, alpha):
    events = events.copy()
    events["event_id"] = events.event_id.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    event_ccs = event_ccs.copy()
    event_ccs["event_id"] = event_ccs.event_id.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    events = events.loc[
        events[["readmission_30d", "er_revisit_30d"]].isin([0, 1]).all(axis=1)
    ].copy()
    merged = event_ccs.merge(
        events[["year", "person_key", "event_id", "readmission_30d", "er_revisit_30d"]],
        on=["year", "person_key", "event_id"], how="inner", validate="many_to_one",
    ).drop_duplicates(["year", "person_key", "event_id", "ccs"])
    if merged.empty:
        raise ValueError("No linked MEPS event-CCS records with both 30-day outcomes")
    merged["ccs"] = "CCS_" + merged.ccs.astype(int).astype(str)
    global_rates = {
        "next_er": float(events.readmission_30d.mean()),
        "next_ip": float(events.er_revisit_30d.mean()),
    }
    rows = []
    for category, group in merged.groupby("ccs"):
        if len(group) < min_group:
            continue
        rows.append({
            "ccs": category, "n": len(group),
            "next_er": (group.readmission_30d.sum() + alpha * global_rates["next_er"]) / (len(group) + alpha),
            "next_ip": (group.er_revisit_30d.sum() + alpha * global_rates["next_ip"]) / (len(group) + alpha),
        })
    rates = pd.DataFrame(rows)
    if rates.empty:
        raise ValueError("No event CCS groups reached --min-group")
    return rates, global_rates, len(events), len(merged)


def attach_group_features(frame, rates, global_rates, stat_names=STATS):
    out = frame.copy()
    lookup_er = rates.set_index("ccs").next_er.to_dict()
    lookup_ip = rates.set_index("ccs").next_ip.to_dict()
    er = np.column_stack([out[f"ccs_{name}"].map(lookup_er).to_numpy(dtype=float) for name in DIAG])
    ip = np.column_stack([out[f"ccs_{name}"].map(lookup_ip).to_numpy(dtype=float) for name in DIAG])
    er = np.where(np.isfinite(er), er, global_rates["next_er"])
    ip = np.where(np.isfinite(ip), ip, global_rates["next_ip"])
    out[stat_names[0]] = er.mean(axis=1).astype(np.float32)
    out[stat_names[1]] = ip.mean(axis=1).astype(np.float32)
    out[stat_names[2]] = er.max(axis=1).astype(np.float32)
    out[stat_names[3]] = ip.max(axis=1).astype(np.float32)
    return out


def run(args):
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    hospital, _ = prepare_hospital(args.hospital)
    base_columns = raw_discharge_predictors(hospital)
    mapping, _ = load_ccs(args.ccs_file)
    hospital = attach_uci_ccs(hospital, mapping)
    if args.event_ccs:
        events = pd.read_csv(args.event_data, dtype={"person_key": str, "event_id": str})
        event_ccs = pd.read_csv(args.event_ccs, dtype={"person_key": str, "event_id": str})
        rates, global_rates, n_pairs, n_conditions = make_event_group_rates(
            events, event_ccs, args.min_group, args.alpha,
        )
        stat_names = EVENT_STATS
    else:
        annual = pd.read_csv(args.meps, dtype={"person_key": str})
        conditions = pd.read_csv(args.conditions, dtype={"person_key": str})
        rates, global_rates, n_pairs, n_conditions = make_meps_group_rates(
            annual, conditions, args.min_group, args.alpha,
        )
        stat_names = STATS
    rates.to_csv(output / "meps_ccs_group_rates.csv", index=False)
    hospital = attach_group_features(hospital, rates, global_rates, stat_names)
    soft_diagnostics = None
    if args.soft_prefix:
        if not args.event_ccs:
            raise ValueError("--soft-prefix requires --event-ccs")
        hospital, soft_diagnostics = add_soft_event_features(
            hospital, events, event_ccs, rates, global_rates,
            args.min_prefix, args.min_parent,
        )
    experiments = {
        "ccs_only_hgb": base_columns + [f"ccs_{name}" for name in DIAG],
        "ccs_meps_group_hgb": base_columns + [f"ccs_{name}" for name in DIAG] + list(stat_names),
    }
    if args.soft_prefix:
        experiments["ccs_meps_soft_hgb"] = base_columns + [f"ccs_{name}" for name in DIAG] + list(SOFT_STATS)
        if args.shuffled_control:
            experiments["ccs_meps_soft_shuffled"] = experiments["ccs_meps_soft_hgb"]
    reference = pd.read_csv(args.reference_oof, usecols=[
        "encounter_id", "patient_nbr", "Fold", "y_true", "p_hgb_tuned", "p_ensemble_fixed_50",
    ])
    if reference.encounter_id.duplicated().any():
        raise ValueError("Reference OOF contains duplicate encounters")
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    parts, choices = [], []
    for fold, (tr_ids, te_ids) in enumerate(
        splitter.split(hospital, hospital.target, groups=hospital.patient_nbr), 1,
    ):
        train, test = hospital.iloc[tr_ids], hospital.iloc[te_ids]
        inner_ids, val_ids = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=args.seed + fold,
        ).split(train, train.target, groups=train.patient_nbr))
        inner, val = train.iloc[inner_ids], train.iloc[val_ids]
        part = test[["encounter_id", "patient_nbr", "target"]].rename(columns={"target": "y_true"}).copy()
        part["Fold"] = fold
        for name, columns in experiments.items():
            exp_train, exp_test = train, test
            if name == "ccs_meps_soft_shuffled":
                rng = np.random.default_rng(args.seed + fold * 10000 + 97)
                exp_train, exp_test = train.copy(), test.copy()
                exp_train[list(SOFT_STATS)] = train.iloc[rng.permutation(len(train))][list(SOFT_STATS)].to_numpy()
                exp_test[list(SOFT_STATS)] = test.iloc[rng.permutation(len(test))][list(SOFT_STATS)].to_numpy()
            exp_inner, exp_val = exp_train.iloc[inner_ids], exp_train.iloc[val_ids]
            x_inner, x_val, _ = encode_frames(exp_inner, exp_val, exp_val, columns, args.min_frequency)
            val_ap, params, iterations, _ = choose_hgb(
                x_inner, x_val, exp_inner.target.to_numpy(), exp_val.target.to_numpy(), args.seed + fold,
            )
            x_train, _, x_test = encode_frames(exp_train, exp_train, exp_test, columns, args.min_frequency)
            model = hgb_model(
                args.seed + fold, {**params, "max_iter": iterations}, early_stopping=False,
            ).fit(x_train, exp_train.target.to_numpy())
            probability = model.predict_proba(x_test)[:, 1]
            part[f"p_{name}"] = probability
            choices.append({"Fold": fold, "Experiment": name, "ValidationAP": val_ap,
                            "Iterations": iterations, "Parameters": json.dumps(params)})
            print(f"[Fold {fold}] {name} AP={average_precision_score(test.target, probability):.6f}", flush=True)
        parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(output / "partial_oof.csv", index=False)
    oof = pd.concat(parts, ignore_index=True)
    oof = oof.merge(reference, on="encounter_id", how="left", validate="one_to_one", suffixes=("", "_ref"))
    if len(oof) != len(reference) or oof.p_hgb_tuned.isna().any():
        raise ValueError("Reference OOF encounters do not match")
    for col in ("patient_nbr", "Fold", "y_true"):
        if not np.array_equal(oof[col].to_numpy(), oof[f"{col}_ref"].to_numpy()):
            raise ValueError(f"Reference OOF {col} mismatch")
    oof = oof.drop(columns=[f"{col}_ref" for col in ("patient_nbr", "Fold", "y_true")])
    models = ["hgb_tuned", "ensemble_fixed_50", *experiments]
    rows = []
    for fold, part in oof.groupby("Fold", sort=True):
        for name in models:
            rows.append({"Fold": fold, "Experiment": name,
                         "PR_AUC_AP": average_precision_score(part.y_true, part[f"p_{name}"])})
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("Experiment").PR_AUC_AP.agg(["mean", "std"]).reset_index()
    bootstrap = cluster_bootstrap(oof, models, [
        ("ccs_meps_group_hgb", "ccs_only_hgb"),
        ("ccs_meps_group_hgb", "hgb_tuned"),
        ("ccs_meps_group_hgb", "ensemble_fixed_50"),
    ] + ([
        ("ccs_meps_soft_hgb", "ccs_only_hgb"),
        ("ccs_meps_soft_hgb", "ccs_meps_group_hgb"),
    ] if args.soft_prefix else []) + ([
        ("ccs_meps_soft_hgb", "ccs_meps_soft_shuffled"),
    ] if args.soft_prefix and args.shuffled_control else []), args.bootstrap, args.seed + 17000)
    oof.to_csv(output / "oof_all.csv", index=False)
    metrics.to_csv(output / "fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "fold_stratified_patient_bootstrap.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "selected_hgb.csv", index=False)
    if soft_diagnostics is not None:
        (output / "soft_prefix_coverage.json").write_text(
            json.dumps(soft_diagnostics, indent=2), encoding="utf-8",
        )
    (output / "config.json").write_text(json.dumps({**vars(args), "meps_pairs": n_pairs,
        "matched_condition_rows": n_conditions}, default=str, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.round(6).to_string(index=False))
    print("\nPatient bootstrap\n", bootstrap.round(6).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital", default=UCI_RAW)
    parser.add_argument("--meps", default=MEPS_BRIDGE)
    parser.add_argument("--ccs-file", default=CCS_RAW)
    parser.add_argument("--conditions", default="data/processed/meps/meps_conditions_ccs_2008_2012.csv")
    parser.add_argument("--event-ccs")
    parser.add_argument("--event-data", default="data/processed/meps_events_2008_2012/meps_diabetes_event_30d_model.csv")
    parser.add_argument("--soft-prefix", action="store_true")
    parser.add_argument("--shuffled-control", action="store_true")
    parser.add_argument("--min-prefix", type=int, default=10)
    parser.add_argument("--min-parent", type=int, default=30)
    parser.add_argument("--reference-oof", required=True)
    parser.add_argument("--outdir", default=result_dir("meps_ccs_group_hgb"))
    parser.add_argument("--min-group", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=50.)
    parser.add_argument("--min-frequency", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    if (args.folds < 2 or args.min_group < 1 or args.alpha < 0 or args.bootstrap < 1
            or args.min_prefix < 1 or args.min_parent < 1):
        parser.error("Invalid folds, min-group, alpha, or bootstrap")
    run(args)


if __name__ == "__main__":
    main()
