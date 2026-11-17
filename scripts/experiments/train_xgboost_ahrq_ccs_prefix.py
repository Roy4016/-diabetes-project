#!/usr/bin/env python3
"""UCI-only vs UCI + AHRQ Single-Level CCS (ICD-9-CM diagnosis categories).

Usage:
  python train_xgboost_ahrq_ccs.py --data-dir uci_prepared_with_ids \
    --ccs-file Single_Level_CCS_2015.zip --outdir xgboost_ccs_results --device cuda

The zip is the original AHRQ official Single_Level_CCS_2015.zip.
Preprocessor is kept identical to the UCI-only baseline; CCS OHE fits TRAIN only.
Requires train_xgboost.py in the same directory for evaluation and plots.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path
from _paths import CCS_RAW, UCI_HOLDOUT, result_dir

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from _shared import choose_f1_threshold, evaluate, draw_curves

DIAG = ('diag_1', 'diag_2', 'diag_3')


def norm_icd9(value):
    """Normalize stored ICD-9-CM diagnosis code to AHRQ's dotless format.

    NEVER coerce ICD codes to floats: E/V codes and leading zeros must survive.
    """
    if pd.isna(value):
        return ''
    s = str(value).strip().strip("'\" ").upper()
    if s in ('', '?', 'NAN', 'NONE', 'UNKNOWN', 'NULL'):
        return ''
    s = re.sub(r'\s+', '', s).replace('.', '')
    if not re.fullmatch(r'(?:[0-9]+|[VE][0-9]+)', s):
        return ''
    # ICD-9-CM numeric categories are three digits minimum.
    # A code such as '8' can arise from reading '008' as a number.
    if s.isdigit() and len(s) < 3:
        s = s.zfill(3)
    return s


def parse_ccs_csv(file_obj):
    mapping = {}
    reader = csv.reader(file_obj)
    for row in reader:
        if len(row) < 2:
            continue
        code = norm_icd9(row[0])
        category = str(row[1]).strip().strip("'\" ")
        if code and re.fullmatch(r'\d+', category):
            mapping[code] = 'CCS_' + category
    if len(mapping) < 1000:
        raise ValueError(f'Only {len(mapping)} mappings found. Need AHRQ diagnosis file $DXREF 2015.csv (not DXLABEL or PRREF).')
    return mapping


def load_ccs(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'CCS file not found: {path}')
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            candidates = [n for n in z.namelist()
                          if 'dxref' in Path(n).name.lower() and n.lower().endswith('.csv')]
            if len(candidates) != 1:
                raise ValueError(f'Could not uniquely identify $DXREF CSV inside ZIP. Found: {z.namelist()}')
            with z.open(candidates[0]) as raw:
                txt = io.TextIOWrapper(raw, encoding='utf-8-sig', errors='replace', newline='')
                mapping = parse_ccs_csv(txt)
                selected = candidates[0]
    else:
        with open(path, encoding='utf-8-sig', errors='replace', newline='') as f:
            mapping = parse_ccs_csv(f)
        selected = path.name
    return mapping, selected


def make_prefix_map(mapping):
    """Only infer a CCS category when ALL descendants of a code agree.

    Prefixes are at least 3 characters; no arbitrary padding on the right.
    Return unique prefixes and ambiguous prefixes for reporting.
    """
    categories = {}
    for full_code, category in mapping.items():
        for prefix_len in range(3, len(full_code)):
            prefix = full_code[:prefix_len]
            categories.setdefault(prefix, set()).add(category)
    unique = {k: next(iter(v)) for k, v in categories.items() if len(v) == 1}
    ambiguous = {k for k, v in categories.items() if len(v) > 1}
    return unique, ambiguous


def map_ccs(code, exact_map, prefix_map, ambiguous_prefixes):
    if not code:
        return 'CCS_UNMAPPED', 'missing'
    if code in exact_map:
        return exact_map[code], 'exact'
    if code in prefix_map:
        return prefix_map[code], 'unique_prefix'
    if code in ambiguous_prefixes:
        return 'CCS_AMBIGUOUS', 'ambiguous'
    return 'CCS_UNMAPPED', 'unmapped'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=UCI_HOLDOUT)
    ap.add_argument('--ccs-file', default=CCS_RAW, help='AHRQ Single_Level_CCS_2015.zip OR extracted $DXREF 2015.csv')
    ap.add_argument('--outdir', default=result_dir('xgboost_ccs_results'))
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--n-estimators', type=int, default=500)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--early-stopping-rounds', type=int, default=30)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_dir)
    mapping, source_name = load_ccs(args.ccs_file)
    prefixes, ambiguous_prefixes = make_prefix_map(mapping)
    print(f'[AHRQ CCS] source={source_name}, codes={len(mapping):,}; '
          f'unique_prefixes={len(prefixes):,}, ambiguous_prefixes={len(ambiguous_prefixes):,}', flush=True)
    frames = {k: pd.read_csv(root / f'{k}.csv', low_memory=False, dtype={d: str for d in DIAG})
              for k in ('train', 'val', 'test')}
    train, val, test = frames['train'], frames['val'], frames['test']
    for label, frame in frames.items():
        if not set(DIAG).issubset(frame.columns):
            raise ValueError(f'{label}.csv missing diagnosis columns: {set(DIAG)-set(frame.columns)}')
    ids = [set(df['patient_nbr']) for df in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]), 'Patient leakage detected!'
    print(f'[Data] train={len(train):,}, val={len(val):,}, test={len(test):,}', flush=True)

    spec = joblib.load(root / 'preprocessor.joblib')
    prep = spec['preprocessor']
    feature_columns = spec['feature_columns']
    categorical_columns = spec['categorical_columns']

    def transform_original(df):
        features = df[feature_columns].copy()
        for col in categorical_columns:
            features[col] = features[col].astype('object').where(features[col].notna(), np.nan)
        transformed = prep.transform(features)
        return transformed.tocsr() if issparse(transformed) else csr_matrix(transformed)

    x_base = {k: transform_original(v) for k, v in frames.items()}
    y = {k: v['target'].astype(int).to_numpy() for k, v in frames.items()}
    print(f'[Original encoded features] {x_base["train"].shape[1]}', flush=True)

    stats = {}
    added_cols = ['ccs_' + d for d in DIAG]
    for split, df in frames.items():
        stats[split] = {}
        for col in DIAG:
            code = df[col].map(norm_icd9)
            resolved = code.map(lambda v: map_ccs(v, mapping, prefixes, ambiguous_prefixes))
            df['ccs_' + col] = resolved.map(lambda pair: pair[0])
            status = resolved.map(lambda pair: pair[1])
            nonempty = code.ne('')
            counts = {name: int(status.eq(name).sum())
                      for name in ('exact', 'unique_prefix', 'ambiguous', 'unmapped', 'missing')}
            matched_count = counts['exact'] + counts['unique_prefix']
            cov = matched_count / max(int(nonempty.sum()), 1)
            stats[split][col] = dict(nonempty=int(nonempty.sum()), matched=matched_count,
                                     coverage=cov, **counts)
            print(f'[CCS] {split}/{col}: mapped={matched_count:,}/{int(nonempty.sum()):,} ({cov:.2%}); '
                  f'exact={counts["exact"]:,}, unique_prefix={counts["unique_prefix"]:,}, '
                  f'ambiguous={counts["ambiguous"]:,}, unmapped={counts["unmapped"]:,}', flush=True)

    # Fit new encoder on TRAIN ONLY, preserve original fitted preprocessing.
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
    encoder.fit(train[added_cols])
    x_extra = {k: encoder.transform(frame[added_cols]) for k, frame in frames.items()}
    print(f'[Added encoded CCS features] {x_extra["train"].shape[1]}', flush=True)
    x_plus = {k: hstack([x_base[k], x_extra[k]], format='csr') for k in frames}

    rows = []
    for name, matrices in [('xgb_uci_only', x_base), ('xgb_uci_ccs', x_plus)]:
        model = XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_lambda=1.0, objective='binary:logistic',
            eval_metric='aucpr', tree_method='hist', device=args.device,
            random_state=42, n_jobs=-1, scale_pos_weight=1,
            early_stopping_rounds=args.early_stopping_rounds)
        model.fit(matrices['train'], y['train'], eval_set=[(matrices['val'], y['val'])], verbose=False)
        pval = model.predict_proba(matrices['val'])[:, 1]
        ptest = model.predict_proba(matrices['test'])[:, 1]
        threshold = choose_f1_threshold(y['val'], pval)
        for split, labels, probabilities in [('validation', y['val'], pval), ('test', y['test'], ptest)]:
            for policy, t in [('default_0.5', 0.5), ('val_f1_optimal', threshold)]:
                rows.append(dict(model=name, split=split, threshold_policy=policy,
                                 **evaluate(labels, probabilities, t)))
        joblib.dump(model, out / f'{name}_model.joblib')
        pd.DataFrame({'patient_nbr':test['patient_nbr'], 'target':y['test'],
                      'predicted_probability':ptest}).to_csv(out / f'{name}_test_predictions.csv', index=False)
        draw_curves(y['test'], ptest, out, name)
        print(f'[Done] {name}: best_iteration={model.best_iteration}; '
              f'val_threshold={threshold:.4f}; test_AUROC={roc_auc_score(y["test"],ptest):.4f}; '
              f'test_AP={average_precision_score(y["test"],ptest):.4f}', flush=True)

    joblib.dump({'onehot':encoder, 'ccs_source':str(args.ccs_file),
                 'feature_names':added_cols, 'mapping':mapping}, out / 'ccs_feature_config.joblib')
    summary = dict(ccs_file=str(args.ccs_file), ccs_mapping_count=len(mapping),
                   mapping_policy='exact -> unique-prefix only; ambiguous kept separate; leading zeros recovered for <3-digit numeric codes',
                   mapping_coverage=stats, base_encoded_features=x_base['train'].shape[1],
                   extra_encoded_features=x_extra['train'].shape[1], patient_disjoint=True,
                   selected_threshold='validation F1', early_stopping='validation PR-AUC',
                   note='CCS derived from existing UCI ICD-9 codes, not additional patient observations.')
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    result = pd.DataFrame(rows)
    result.to_csv(out / 'metrics.csv', index=False)
    print(result[['model', 'split', 'threshold_policy','AUROC','PR_AUC_AP','Precision','Recall','F1','Brier']].round(4).to_string(index=False))
    print('Saved to', out.resolve(), flush=True)

if __name__ == '__main__':
    main()
