"""UCI Diabetes 130-US Hospitals: reproducible preparation, Steps 1-6.

Run: python prepare_uci_steps_1_to_6.py --input 'diabetic_data(1).csv' --outdir uci_prepared
Dependencies: pandas numpy scikit-learn scipy joblib
Data unit: hospital encounter; patient_nbr used only for group splits.
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUSION_IDS = {11, 13, 14, 19, 20, 21}
DROP_FEATURES = {
    'encounter_id', 'patient_nbr', 'readmitted', 'target',
    'discharge_disposition_id', 'weight', 'examide', 'citoglipton',
}
MAPPING_IDS = ['admission_type_id', 'admission_source_id']


def load_ids_mapping(path):
    """Read the three stacked sections of official UCI IDS_mapping.csv."""
    maps = {key: {} for key in [*MAPPING_IDS, 'discharge_disposition_id']}
    current = None
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            first = row[0].strip()
            if first in maps:
                current = first
                continue
            if current is not None and first.isdigit():
                description = ','.join(row[1:]).strip()
                maps[current][int(first)] = description
    if not all(maps.values()):
        raise ValueError('IDS mapping has missing or unrecognized sections')
    return maps


def clean_category(description):
    if description is None or description.strip().lower() in {
        'null', 'not mapped', 'not available', 'unknown/invalid', ''
    }:
        return 'Unknown'
    return description.strip()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='diabetic_data.csv')
    ap.add_argument('--outdir', default='uci_prepared')
    ap.add_argument('--mapping', default='IDS_mapping.csv', help='Official UCI IDS_mapping.csv')
    args = ap.parse_args()
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)

    # Treat '?' as missing; keep ICD diagnosis codes as text (including leading zeros).
    df = pd.read_csv(args.input, na_values=['?'], low_memory=False,
                     dtype={f'diag_{n}': 'string' for n in (1, 2, 3)})
    id_maps = load_ids_mapping(args.mapping)
    original_rows = len(df)
    assert {'patient_nbr', 'readmitted', 'discharge_disposition_id'}.issubset(df.columns)

    # Step 1: Define cohort using official discharge mapping to verify IDs.
    for ident in EXCLUSION_IDS:
        description = id_maps['discharge_disposition_id'].get(ident, '').lower()
        if not ('expired' in description or 'hospice' in description):
            raise ValueError(f'Unexpected discharge mapping for ID {ident}: {description}')
    df = df.loc[~df['discharge_disposition_id'].isin(EXCLUSION_IDS)].copy()
    excluded_rows = original_rows - len(df)
    # Step 2: Define binary labels; greater than 30 days and NO are both negative.
    assert set(df['readmitted'].dropna().unique()) <= {'<30', '>30', 'NO'}
    df['target'] = df['readmitted'].eq('<30').astype('int8')

    # Remove invalid gender, keep missing demographics as Unknown in preprocessing.
    invalid_gender = df['gender'].eq('Unknown/Invalid').sum()
    df = df.loc[~df['gender'].eq('Unknown/Invalid')].copy()
    df.reset_index(drop=True, inplace=True)

    # Step 3: feature selection; discharge disposition only defines exclusion,
    # never input. Drop two constant medication columns from original dataset.
    feature_cols = [c for c in df.columns if c not in DROP_FEATURES]
    # Step 3/5/6: external official ID descriptions for interpretation.
    # Preserve ID labels only for audit; replace original model features by text.
    unmapped = {}
    for col in MAPPING_IDS:
        source = pd.to_numeric(df[col], errors='coerce')
        unmapped[col] = int((source.notna() & ~source.isin(id_maps[col])).sum())
        description_col = col.removesuffix('_id')
        df[description_col] = source.map(id_maps[col]).map(clean_category).astype('object')
        df.drop(columns=[col], inplace=True)
    feature_cols = [c for c in df.columns if c not in DROP_FEATURES]
    # Number-like ICD strings are deliberately kept categorical, not numeric.

    # Step 4: approximately 70/15/15 encounter splits, disjoint by patient.
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    t_idx, r_idx = next(first.split(df, df['target'], groups=df['patient_nbr']))
    train = df.iloc[t_idx].copy()
    remaining = df.iloc[r_idx].copy()
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43)
    v_idx, s_idx = next(second.split(remaining, remaining['target'], groups=remaining['patient_nbr']))
    val = remaining.iloc[v_idx].copy()
    test = remaining.iloc[s_idx].copy()
    parts = {'train': train, 'val': val, 'test': test}
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        assert set(parts[a].patient_nbr).isdisjoint(set(parts[b].patient_nbr))
    assert sum(len(x) for x in parts.values()) == len(df)

    # Step 5 + 6: only learn imputation and category vocab from training.
    X_train = train[feature_cols].copy()
    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in feature_cols if c not in numerical]
    # Missing A1C/max glucose should not be imputed as 'normal': category Unknown.
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=False)),
        ('scaler', StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=np.float32))
    ])
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, numerical),
        ('categorical', cat_pipe, categorical)
    ], sparse_threshold=1.0)
    # Convert pandas extension-string columns to ordinary Python string or np.nan
    # for compatibility across sklearn versions.
    def model_x(frame):
        xx = frame[feature_cols].copy()
        for col in categorical:
            xx[col] = xx[col].astype('object').where(xx[col].notna(), np.nan)
        return xx
    xformed = preprocessor.fit_transform(model_x(train))
    check_val = preprocessor.transform(model_x(val))
    check_test = preprocessor.transform(model_x(test))
    assert xformed.shape[1] == check_val.shape[1] == check_test.shape[1]

    # Unencoded patient-level split files are kept for model iteration and CDC/CCS joins.
    # IDs remain only for auditing group independence; DO NOT use as model input.
    for name, frame in parts.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    joblib.dump({'preprocessor': preprocessor, 'feature_columns': feature_cols,
                 'categorical_columns': categorical, 'numeric_columns': numerical,
                 'excluded_columns': sorted(DROP_FEATURES)}, output / 'preprocessor.joblib')

    report = {
        'original_encounters': original_rows,
        'excluded_death_hospice': excluded_rows,
        'excluded_invalid_gender': int(invalid_gender),
        'remaining_encounters': len(df),
        'unique_patients': int(df.patient_nbr.nunique()),
        'target_definition': '1 if readmitted == <30, else 0',
        'official_id_mapping_file': str(args.mapping),
        'mapped_feature_names': ['admission_type', 'admission_source'],
        'unmapped_ids_count': unmapped,
        'feature_count_raw': len(feature_cols),
        'encoded_feature_count': int(xformed.shape[1]),
        'numerical_features': numerical, 'categorical_features': categorical,
        'discharge_disposition_as_predictor': False,
        'split': {name: {
            'encounters': len(frame),
            'unique_patients': int(frame.patient_nbr.nunique()),
            'positive_cases': int(frame.target.sum()),
            'positive_rate': round(float(frame.target.mean()), 5)
        } for name, frame in parts.items()},
        'missing_fraction_by_feature': {
            c: round(float(df[c].isna().mean()), 5) for c in feature_cols
        },
        'notes': [
            'Hospital encounter is the unit of analysis; repeated encounters from the same patient stay in one split.',
            'GroupShuffleSplit gives approximate encounter proportions, not class stratification.',
            'Original admission IDs were replaced by official description labels; missing, Not Mapped, NULL and Not Available become Unknown.',
            'Official IDS_mapping.csv is from the same UCI dataset and does not count as external augmentation.',
            'No external augmentation, resampling or calibration has been applied yet.',
            'Predictors correspond to discharge-time risk prediction; review use of all medication and procedure variables if changing target time.',
            'Missing A1Cresult/max_glu_serum are retained as separate Unknown categorical values.',
            'When adding CDC/CCS columns, rebuild and refit this preprocessor using TRAIN only.'
        ]
    }
    (output / 'quality_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ['missing_fraction_by_feature', 'categorical_features','numerical_features']}, indent=2, ensure_ascii=False))
    print('Saved output to:', output.resolve())

if __name__ == '__main__':
    main()
