#!/usr/bin/env python3
"""Random Forest baseline for UCI Diabetes prepared data.

Examples:
  python -u train_rf.py --data-dir uci_prepared_with_ids --outdir rf_results
  python -u train_rf.py --data-dir uci_prepared_with_ids --outdir rf_results --n-estimators 300 --max-depth 16

Uses scikit-learn RandomForestClassifier (CPU-only), same patient-disjoint
splits and saved training-fitted preprocessor as the XGBoost baseline.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


def choose_f1_threshold(y, prob):
    precision, recall, thresholds = precision_recall_curve(y, prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'AUROC': float(roc_auc_score(y, prob)),
        'PR_AUC_AP': float(average_precision_score(y, prob)),
        'Accuracy': float(accuracy_score(y, pred)),
        'Precision': float(precision_score(y, pred, zero_division=0)),
        'Recall': float(recall_score(y, pred, zero_division=0)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'Brier': float(brier_score_loss(y, prob)),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


def draw_curves(y, prob, out, prefix):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'AUROC = {roc_auc_score(y, prob):.3f}')
    ax.plot([0, 1], [0, 1], '--', linewidth=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f'{prefix}_roc.png', dpi=170)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, label=f'AP = {average_precision_score(y, prob):.3f}')
    ax.axhline(y.mean(), linestyle='--', linewidth=1, label=f'Prevalence = {y.mean():.3f}')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f'{prefix}_pr.png', dpi=170)
    plt.close(fig)

    frac, mean = calibration_curve(y, prob, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    ax.plot(mean, frac, 'o-', label=prefix)
    ax.set(xlabel='Mean predicted probability', ylabel='Observed positive fraction',
           title='Calibration Curve', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f'{prefix}_calibration.png', dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default='uci_prepared_with_ids')
    ap.add_argument('--outdir', default='rf_results')
    ap.add_argument('--n-estimators', type=int, default=300)
    ap.add_argument('--max-depth', type=int, default=16,
                    help='Use 0 for unlimited depth')
    ap.add_argument('--min-samples-leaf', type=int, default=2)
    ap.add_argument('--max-features', default='sqrt', choices=['sqrt', 'log2', 'none'])
    ap.add_argument('--n-jobs', type=int, default=8,
                    help='CPU workers; use -1 for all available cores')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.n_estimators < 1 or args.min_samples_leaf < 1:
        ap.error('n-estimators and min-samples-leaf must be positive')

    src, out = Path(args.data_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(src / 'train.csv', low_memory=False)
    val = pd.read_csv(src / 'val.csv', low_memory=False)
    test = pd.read_csv(src / 'test.csv', low_memory=False)

    spec = joblib.load(src / 'preprocessor.joblib')
    prep = spec['preprocessor']
    features = spec['feature_columns']
    categoricals = spec['categorical_columns']

    for left, right, label in [(train, val, 'train/val'),
                               (train, test, 'train/test'),
                               (val, test, 'val/test')]:
        if not set(left['patient_nbr']).isdisjoint(right['patient_nbr']):
            raise ValueError(f'Patient overlap detected: {label}')

    def build_x(frame):
        x = frame[features].copy()
        for column in categoricals:
            x[column] = x[column].astype('object').where(x[column].notna(), np.nan)
        return prep.transform(x)

    X_train, X_val, X_test = [build_x(f) for f in (train, val, test)]
    y_train = train['target'].astype(int).to_numpy()
    y_val = val['target'].astype(int).to_numpy()
    y_test = test['target'].astype(int).to_numpy()
    print(f'[Data] train={len(train):,}, val={len(val):,}, test={len(test):,}', flush=True)
    print(f'[Features] raw={len(features)}, encoded={X_train.shape[1]:,}', flush=True)
    print(f'[Class ratio] positive={y_train.sum():,}, negative={(y_train == 0).sum():,}', flush=True)
    print('[Device] CPU (scikit-learn RandomForestClassifier does not use CUDA)', flush=True)

    rows = []
    for name, weight in [('rf_standard', None), ('rf_balanced', 'balanced')]:
        print(f'\n[Training] {name} ...', flush=True)
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None if args.max_depth == 0 else args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=None if args.max_features == 'none' else args.max_features,
            class_weight=weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        threshold = choose_f1_threshold(y_val, p_val)
        for split, labels, probs in [('validation', y_val, p_val), ('test', y_test, p_test)]:
            for policy, t in [('default_0.5', 0.5), ('val_f1_optimal', threshold)]:
                rows.append({'model': name, 'split': split, 'threshold_policy': policy,
                             **evaluate(labels, probs, t)})
        joblib.dump(model, out / f'{name}_model.joblib', compress=3)
        pd.DataFrame({'patient_nbr': test['patient_nbr'], 'target': y_test,
                      'predicted_probability': p_test}).to_csv(
            out / f'{name}_test_predictions.csv', index=False)
        draw_curves(y_test, p_test, out, name)
        print(f'[Done] {name}; validation threshold={threshold:.4f}; '
              f'test AUROC={roc_auc_score(y_test, p_test):.4f}; '
              f'test AP={average_precision_score(y_test, p_test):.4f}', flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'rf_metrics.csv', index=False)
    summary = {
        'train': len(train), 'validation': len(val), 'test': len(test),
        'input_features': len(features), 'encoded_features': int(X_train.shape[1]),
        'train_positive_rate': float(y_train.mean()),
        'class_weight_versions': {'rf_standard': None, 'rf_balanced': 'balanced'},
        'parameters': vars(args), 'threshold_selection': 'Validation F1 only',
        'device': 'CPU',
        'notes': 'Saved preprocessor reused without refitting; patient-disjoint sets; no SMOTE or external feature augmentation.',
    }
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n' + metrics[['model', 'split', 'threshold_policy', 'AUROC', 'PR_AUC_AP',
                          'Precision', 'Recall', 'F1', 'Brier']].round(4).to_string(index=False), flush=True)
    print('Saved to', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
