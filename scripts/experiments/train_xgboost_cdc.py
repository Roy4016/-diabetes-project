#!/usr/bin/env python3
"""UCI-only vs UCI + CDC BRFSS 2015 group-level feature augmentation.

CDC features are demographic-group summaries, NOT individual patient measurements.
Uses preexisting patient-disjoint splits and saved training-fitted UCI preprocessor.
Requires train_xgboost.py in same directory (for metrics and plotting functions).
"""
import argparse
import json
from pathlib import Path
from _paths import CDC_RAW, UCI_HOLDOUT, result_dir

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from _shared import choose_f1_threshold, evaluate, draw_curves

# 2015 CDC Diabetes Health Indicators Age codes 1..13:
# 18-24,25-29,30-34,...,75-79,80+
CDC_AGE_MID = {1:21, 2:27, 3:32, 4:37, 5:42, 6:47, 7:52,
               8:57, 9:62, 10:67, 11:72, 12:77, 13:85}
CDC_COLUMNS = ['HighBP', 'HeartDiseaseorAttack', 'BMI', 'GenHlth']
ADDED_COLUMNS = ['cdc_highbp_rate', 'cdc_heart_rate',
                 'cdc_mean_bmi', 'cdc_poor_health_rate']


def broad_age(value):
    """Broad common age buckets, reducing mismatch of original age bins."""
    try:
        age = float(value)
    except (ValueError, TypeError):
        return 'unknown'
    if age < 20:
        return 'under20'
    if age < 40:
        return '20-39'
    if age < 60:
        return '40-59'
    if age < 80:
        return '60-79'
    return '80+'


def uci_age_bucket(value):
    if pd.isna(value):
        return 'unknown'
    import re
    match = re.search(r'\[\s*(\d+)\s*-\s*(\d+)\s*\)', str(value))
    if not match:
        return 'unknown'
    low, high = int(match.group(1)), int(match.group(2))
    return broad_age((low + high - 1) / 2)


def uci_sex(value):
    v = str(value).strip().lower()
    return 'male' if v == 'male' else ('female' if v == 'female' else 'unknown')


def build_cdc_statistics(path, min_group_n):
    cdc = pd.read_csv(path, low_memory=False)
    needed = ['Diabetes_012', 'Age', 'Sex'] + CDC_COLUMNS
    missing = set(needed) - set(cdc.columns)
    if missing:
        raise ValueError(f'CDC CSV missing columns: {sorted(missing)}. Use diabetes_012_health_indicators_BRFSS2015.csv')
    # Diabetes_012 == 2 means diagnosed diabetes (not pre-diabetes).
    cdc = cdc.loc[pd.to_numeric(cdc['Diabetes_012'], errors='coerce') == 2].copy()
    if cdc.empty:
        raise ValueError('No rows with Diabetes_012=2: check CDC source CSV')
    for col in CDC_COLUMNS:
        cdc[col] = pd.to_numeric(cdc[col], errors='coerce')
    cdc['age_bucket'] = pd.to_numeric(cdc['Age'], errors='coerce').map(CDC_AGE_MID).map(broad_age)
    cdc['sex_bucket'] = pd.to_numeric(cdc['Sex'], errors='coerce').map({0:'female', 1:'male'}).fillna('unknown')
    cdc['cdc_highbp_rate'] = cdc['HighBP']
    cdc['cdc_heart_rate'] = cdc['HeartDiseaseorAttack']
    cdc['cdc_mean_bmi'] = cdc['BMI']
    cdc['cdc_poor_health_rate'] = (cdc['GenHlth'] >= 4).astype(float)
    global_means = cdc[ADDED_COLUMNS].mean().to_dict()
    groups = cdc.groupby(['age_bucket', 'sex_bucket'], dropna=False)
    group_means = groups[ADDED_COLUMNS].mean()
    counts = groups.size()
    # Fallback: by age; if missing then overall CDC diabetic-population mean
    age_means = cdc.groupby('age_bucket')[ADDED_COLUMNS].mean()
    return group_means, counts, age_means, global_means, len(cdc)


def add_cdc_features(frame, stats, min_group_n):
    group_means, counts, age_means, global_means, _ = stats
    buckets = pd.DataFrame(index=frame.index)
    buckets['age_bucket'] = frame['age'].map(uci_age_bucket)
    buckets['sex_bucket'] = frame['gender'].map(uci_sex)
    pairs = pd.MultiIndex.from_frame(buckets[['age_bucket','sex_bucket']])
    group = group_means.reindex(pairs)
    group.index = frame.index
    ns = counts.reindex(pairs, fill_value=0).to_numpy()
    group.loc[ns < min_group_n, :] = np.nan
    age_fallback = age_means.reindex(buckets['age_bucket'])
    age_fallback.index = frame.index
    group = group.fillna(age_fallback).fillna(pd.Series(global_means))
    if group.isna().any().any():
        raise RuntimeError('CDC features contain missing values after fallbacks')
    return group[ADDED_COLUMNS].astype('float32'), {
        'age_sex_match_rate': float((ns >= min_group_n).mean()),
        'fallback_count': int((ns < min_group_n).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=UCI_HOLDOUT)
    ap.add_argument('--cdc-csv', default=CDC_RAW)
    ap.add_argument('--outdir', default=result_dir('xgboost_cdc_results'))
    ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--min-group-n', type=int, default=30)
    ap.add_argument('--n-estimators', type=int, default=500)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--early-stopping-rounds', type=int, default=30)
    args = ap.parse_args()
    source = Path(args.data_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    train, val, test = [pd.read_csv(source / f'{split}.csv', low_memory=False)
                        for split in ['train', 'val', 'test']]
    spec = joblib.load(source / 'preprocessor.joblib')
    preprocessor = spec['preprocessor']
    features = spec['feature_columns']
    cats = spec['categorical_columns']
    assert set(train.patient_nbr).isdisjoint(val.patient_nbr)
    assert set(train.patient_nbr).isdisjoint(test.patient_nbr)
    assert set(val.patient_nbr).isdisjoint(test.patient_nbr)
    print(f'[Data] train={len(train):,}, val={len(val):,}, test={len(test):,}', flush=True)

    def original_matrix(frame):
        x = frame[features].copy()
        for col in cats:
            x[col] = x[col].astype('object').where(x[col].notna(), np.nan)
        a = preprocessor.transform(x)
        return a.tocsr() if issparse(a) else csr_matrix(a)

    x_orig = [original_matrix(frame) for frame in (train, val, test)]
    y = [frame['target'].astype(int).to_numpy() for frame in (train, val, test)]
    print(f'[Original encoded features] {x_orig[0].shape[1]}', flush=True)

    stats = build_cdc_statistics(args.cdc_csv, args.min_group_n)
    cdc_feats = []
    coverage = {}
    for name, frame in zip(['train','validation','test'], [train,val,test]):
        f, info = add_cdc_features(frame, stats, args.min_group_n)
        cdc_feats.append(f)
        coverage[name] = info
        print(f'[CDC] {name}: age-sex group matches={info["age_sex_match_rate"]:.1%}', flush=True)
    # Scale using TRAIN only; do not fit on validation/test.
    from sklearn.preprocessing import StandardScaler
    cdc_scaler = StandardScaler()
    scaled = [csr_matrix(cdc_scaler.fit_transform(cdc_feats[0]))]
    scaled += [csr_matrix(cdc_scaler.transform(f)) for f in cdc_feats[1:]]
    augmented = [hstack([orig, ext], format='csr') for orig, ext in zip(x_orig,scaled)]

    results = []
    for name, matrices in [('xgb_uci_only', x_orig), ('xgb_uci_cdc', augmented)]:
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='aucpr',
            tree_method='hist',
            device=args.device,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=1,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model.fit(matrices[0], y[0], eval_set=[(matrices[1], y[1])], verbose=False)
        p_val = model.predict_proba(matrices[1])[:, 1]
        p_test = model.predict_proba(matrices[2])[:, 1]
        threshold = choose_f1_threshold(y[1], p_val)
        for split, truth, probs in [('validation',y[1],p_val), ('test',y[2],p_test)]:
            for policy, t in [('default_0.5', 0.5), ('val_f1_optimal', threshold)]:
                results.append(dict(model=name, split=split, threshold_policy=policy,
                                    **evaluate(truth, probs, t)))
        joblib.dump(model, out / f'{name}_model.joblib')
        pd.DataFrame({'patient_nbr':test['patient_nbr'], 'target':y[2],
                      'predicted_probability':p_test}).to_csv(
                          out / f'{name}_test_predictions.csv', index=False)
        draw_curves(y[2], p_test, out, name)
        print(f'[Done] {name}: best_iteration={model.best_iteration}; '
              f'val_threshold={threshold:.4f}; '
              f'test_AUROC={roc_auc_score(y[2],p_test):.4f}; '
              f'test_AP={average_precision_score(y[2],p_test):.4f}', flush=True)

    joblib.dump({'scaler':cdc_scaler, 'feature_names':ADDED_COLUMNS,
                 'age_mapping':CDC_AGE_MID, 'min_group_n':args.min_group_n},
                out / 'cdc_feature_config.joblib')
    pd.DataFrame(results).to_csv(out / 'metrics.csv', index=False)
    summary = {'cdc_diabetic_respondents':stats[-1], 'additional_features':ADDED_COLUMNS,
               'cdc_age_sex_coverage':coverage,
               'source':'CDC diabetes_012_health_indicators_BRFSS2015.csv, Diabetes_012 == 2',
               'disclaimer':'CDC age-sex group statistics, not individual UCI patient measurements. No readmission labels in CDC.',
               'model_selection':'validation aucpr early stopping, validation F1 threshold',
               'same_patient_disjoint_split':True}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(results)[['model','split','threshold_policy','AUROC','PR_AUC_AP',
           'Precision','Recall','F1','Brier']].round(4).to_string(index=False))
    print('Saved to', out.resolve())

if __name__ == '__main__':
    main()
