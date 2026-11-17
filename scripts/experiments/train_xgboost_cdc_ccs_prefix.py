#!/usr/bin/env python3
"""Compare UCI-only / +CDC / +CCS unique-prefix / +CDC+CCS XGBoost.

Uses original patient-disjoint CSV files and fitted UCI preprocessor.
CDC aggregate health indicators are age/sex group-level, not patient observations.
CCS ambiguous ICD-9 prefixes are kept as a separate category.
Requires train_xgboost.py, train_xgboost_cdc.py, and
train_xgboost_ahrq_ccs_prefix.py next to this script.
"""
import argparse
import json
from pathlib import Path
from _paths import CCS_RAW, CDC_RAW, UCI_HOLDOUT, result_dir
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from _shared import choose_f1_threshold, evaluate, draw_curves
from train_xgboost_cdc import build_cdc_statistics, add_cdc_features, ADDED_COLUMNS
from train_xgboost_ahrq_ccs_prefix import DIAG, norm_icd9, load_ccs, make_prefix_map, map_ccs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=UCI_HOLDOUT)
    ap.add_argument('--cdc-csv', default=CDC_RAW)
    ap.add_argument('--ccs-file', default=CCS_RAW)
    ap.add_argument('--outdir', default=result_dir('xgboost_cdc_ccs_prefix_results'))
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    ap.add_argument('--min-group-n', type=int, default=30)
    ap.add_argument('--n-estimators', type=int, default=500)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--early-stopping-rounds', type=int, default=30)
    ap.add_argument('--only-combined', action='store_true', help='Skip the three control arms (not recommended for the first run)')
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_csv(root / f'{s}.csv', low_memory=False, dtype={d:str for d in DIAG})
              for s in ('train','val','test')}
    ids = [set(frames[s]['patient_nbr']) for s in ('train','val','test')]
    if not (ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])):
        raise ValueError('Patient leakage: patient_nbr overlaps between splits')
    print('[Data]', ', '.join(f'{s}={len(df):,}' for s,df in frames.items()), flush=True)

    spec = joblib.load(root / 'preprocessor.joblib')
    preprocessor, cols, cats = spec['preprocessor'], spec['feature_columns'], spec['categorical_columns']
    def transform_original(frame):
        z = frame[cols].copy()
        for c in cats:
            z[c] = z[c].astype('object').where(z[c].notna(), np.nan)
        x = preprocessor.transform(z)
        return x.tocsr() if issparse(x) else csr_matrix(x)
    base = {s: transform_original(df) for s,df in frames.items()}
    y = {s: df['target'].astype(int).to_numpy() for s,df in frames.items()}
    print(f'[UCI] original encoded features={base["train"].shape[1]}', flush=True)

    # CDC: external population health statistics derived from diabetic respondents.
    cdc_stats = build_cdc_statistics(args.cdc_csv, args.min_group_n)
    cdc_raw, cdc_coverage = {}, {}
    for s, df in frames.items():
        cdc_raw[s], cdc_coverage[s] = add_cdc_features(df, cdc_stats, args.min_group_n)
        print(f'[CDC] {s}: age-sex match={cdc_coverage[s]["age_sex_match_rate"]:.1%}', flush=True)
    scaler = StandardScaler().fit(cdc_raw['train'])
    cdc_encoded = {s: csr_matrix(scaler.transform(cdc_raw[s]).astype(np.float32)) for s in frames}

    # CCS: exact match or unique category among all descendants of a code.
    exact, source_name = load_ccs(args.ccs_file)
    prefixes, ambiguous = make_prefix_map(exact)
    print(f'[CCS] source={source_name}; codes={len(exact):,}; unique_prefix={len(prefixes):,}; ambiguous_prefix={len(ambiguous):,}', flush=True)
    ccs_cols = ['ccs_' + d for d in DIAG]
    ccs_coverage = {}
    for s, df in frames.items():
        ccs_coverage[s] = {}
        for d in DIAG:
            codes = df[d].map(norm_icd9)
            result = codes.map(lambda c: map_ccs(c, exact, prefixes, ambiguous))
            df['ccs_' + d] = result.map(lambda a: a[0])
            status = result.map(lambda a: a[1])
            counts = {k: int((status == k).sum()) for k in ('exact','unique_prefix','ambiguous','unmapped','missing')}
            valid = int(codes.ne('').sum())
            coverage = (counts['exact']+counts['unique_prefix']) / max(valid,1)
            ccs_coverage[s][d] = {'coverage':coverage, **counts}
            print(f'[CCS] {s}/{d}: coverage={coverage:.2%} exact={counts["exact"]:,} unique_prefix={counts["unique_prefix"]:,} ambiguous={counts["ambiguous"]:,}', flush=True)
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32).fit(frames['train'][ccs_cols])
    ccs_encoded = {s: encoder.transform(df[ccs_cols]) for s,df in frames.items()}
    print(f'[Features] base={base["train"].shape[1]}, CDC={len(ADDED_COLUMNS)}, CCS={ccs_encoded["train"].shape[1]}', flush=True)

    matrices = {
        'xgb_uci_only': base,
        'xgb_uci_cdc': {s: hstack([base[s], cdc_encoded[s]], format='csr') for s in frames},
        'xgb_uci_ccs': {s: hstack([base[s], ccs_encoded[s]], format='csr') for s in frames},
        'xgb_uci_cdc_ccs': {s: hstack([base[s], cdc_encoded[s], ccs_encoded[s]], format='csr') for s in frames},
    }
    if args.only_combined:
        matrices = {'xgb_uci_cdc_ccs': matrices['xgb_uci_cdc_ccs']}

    rows = []
    for name, x in matrices.items():
        model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_lambda=1.0, objective='binary:logistic',
            eval_metric='aucpr', tree_method='hist', device=args.device,
            random_state=42, n_jobs=-1, scale_pos_weight=1,
            early_stopping_rounds=args.early_stopping_rounds)
        model.fit(x['train'], y['train'], eval_set=[(x['val'], y['val'])], verbose=False)
        pval = model.predict_proba(x['val'])[:, 1]
        ptest = model.predict_proba(x['test'])[:, 1]
        threshold = choose_f1_threshold(y['val'], pval)
        for split, true, proba in [('validation', y['val'], pval), ('test', y['test'], ptest)]:
            for policy, t in [('default_0.5',0.5), ('val_f1_optimal',threshold)]:
                rows.append({'model':name, 'split':split, 'threshold_policy':policy,
                             **evaluate(true, proba, t)})
        joblib.dump(model, out / f'{name}_model.joblib')
        pd.DataFrame({'patient_nbr':frames['test']['patient_nbr'],
                      'target':y['test'], 'predicted_probability':ptest}).to_csv(
                          out / f'{name}_test_predictions.csv', index=False)
        draw_curves(y['test'], ptest, out, name)
        print(f'[Done] {name}: best_iteration={model.best_iteration}; '
              f'val_threshold={threshold:.4f}; test_AUROC={roc_auc_score(y["test"],ptest):.4f}; '
              f'test_AP={average_precision_score(y["test"],ptest):.4f}', flush=True)
    joblib.dump({'cdc_scaler':scaler, 'cdc_feature_names':ADDED_COLUMNS,
                 'cdc_group_statistics':cdc_stats, 'ccs_encoder':encoder,
                 'ccs_exact_mapping':exact, 'ccs_unique_prefixes':prefixes,
                 'ccs_ambiguous_prefixes':ambiguous, 'ccs_columns':ccs_cols,
                 'feature_order':'UCI original -> CDC 4 -> CCS one-hot'},
                out/'augmentation_config.joblib')
    summary = {'data_dir':str(root), 'cdc_csv':str(args.cdc_csv),
        'ccs_file':str(args.ccs_file), 'split_sizes':{s:len(f) for s,f in frames.items()},
        'cdc_diabetes_respondents':cdc_stats[-1], 'cdc_coverage':cdc_coverage,
        'ccs_coverage':ccs_coverage, 'original_features':base['train'].shape[1],
        'cdc_features':len(ADDED_COLUMNS), 'ccs_features':ccs_encoded['train'].shape[1],
        'threshold_selection':'validation F1', 'early_stopping':'validation aucpr',
        'notes':['Only UCI labels are used. CDC are age-sex group aggregate indicators, not patient measurements.',
                 'CCS categories are deterministic mappings from existing ICD-9 codes.',
                 'Use validation, not test, for model choice.']}
    (out/'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    result = pd.DataFrame(rows)
    result.to_csv(out/'metrics.csv', index=False)
    print(result[['model','split','threshold_policy','AUROC','PR_AUC_AP','Precision','Recall','F1','Brier']].round(4).to_string(index=False))
    print('Saved to',out.resolve(), flush=True)

if __name__ == '__main__':
    main()
