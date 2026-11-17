#!/usr/bin/env python3
"""UCI Diabetes XGBoost baseline, compatible with Step 1-6 + IDS Mapping.

python train_xgboost.py --data-dir uci_prepared_with_ids --outdir xgboost_results

Requires: pandas numpy scikit-learn xgboost joblib matplotlib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def choose_f1_threshold(y, prob):
    precision, recall, thresholds = precision_recall_curve(y, prob)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=float(threshold),
        AUROC=float(roc_auc_score(y, prob)),
        PR_AUC_AP=float(average_precision_score(y, prob)),
        Accuracy=float(accuracy_score(y, pred)),
        Precision=float(precision_score(y, pred, zero_division=0)),
        Recall=float(recall_score(y, pred, zero_division=0)),
        F1=float(f1_score(y, pred, zero_division=0)),
        Brier=float(brier_score_loss(y, prob)),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
    )


def draw_curves(y, prob, out, prefix):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'AUROC = {roc_auc_score(y, prob):.3f}')
    ax.plot([0, 1], [0, 1], '--', linewidth=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(out / f'{prefix}_roc.png', dpi=170); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f'AP = {average_precision_score(y, prob):.3f}')
    ax.axhline(y.mean(), linestyle='--', linewidth=1, label=f'Prevalence = {y.mean():.3f}')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–Recall Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(out / f'{prefix}_pr.png', dpi=170); plt.close(fig)

    frac, mean = calibration_curve(y, prob, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    ax.plot(mean, frac, 'o-', label=prefix)
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction of positives', title='Calibration Curve', xlim=(0,1), ylim=(0,1))
    ax.legend(); fig.tight_layout(); fig.savefig(out / f'{prefix}_calibration.png', dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='uci_prepared_with_ids')
    ap.add_argument('--outdir', default='xgboost_results')
    ap.add_argument('--n-estimators', type=int, default=500)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--early-stopping-rounds', type=int, default=30)
    args = ap.parse_args()

    src = Path(args.data_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(src / 'train.csv', low_memory=False)
    val = pd.read_csv(src / 'val.csv', low_memory=False)
    test = pd.read_csv(src / 'test.csv', low_memory=False)
    spec = joblib.load(src / 'preprocessor.joblib')
    prep = spec['preprocessor']
    features = spec['feature_columns']
    categoricals = spec['categorical_columns']

    assert set(train.patient_nbr).isdisjoint(val.patient_nbr)
    assert set(train.patient_nbr).isdisjoint(test.patient_nbr)
    assert set(val.patient_nbr).isdisjoint(test.patient_nbr)

    def build_x(frame):
        x = frame[features].copy()
        for column in categoricals:
            x[column] = x[column].astype('object').where(x[column].notna(), np.nan)
        return prep.transform(x)

    X_train, X_val, X_test = map(build_x, [train, val, test])
    y_train = train['target'].astype(int).to_numpy()
    y_val = val['target'].astype(int).to_numpy()
    y_test = test['target'].astype(int).to_numpy()
    positive = int(y_train.sum())
    negative = len(y_train) - positive
    ratio = negative / positive
    rows = []

    for name, pos_weight in [('xgb_standard', 1.0), ('xgb_balanced', ratio)]:
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
            device="cuda",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=pos_weight,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        # Validation is used for early stopping and F1-threshold selection ONLY.
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        threshold = choose_f1_threshold(y_val, p_val)

        for split, labels, probabilities in [
            ('validation', y_val, p_val), ('test', y_test, p_test)
        ]:
            for policy, t in [('default_0.5', 0.5), ('val_f1_optimal', threshold)]:
                rows.append(dict(model=name, split=split, threshold_policy=policy,
                                 **evaluate(labels, probabilities, t)))

        joblib.dump(model, out / f'{name}_model.joblib')
        pd.DataFrame(dict(patient_nbr=test.patient_nbr, target=y_test,
                          predicted_probability=p_test)).to_csv(
            out / f'{name}_test_predictions.csv', index=False)
        draw_curves(y_test, p_test, out, name)
        print(f'{name}: best iteration={model.best_iteration}; '
              f'validation threshold={threshold:.4f}; '
              f'test AUROC={roc_auc_score(y_test,p_test):.4f}; '
              f'test AP={average_precision_score(y_test,p_test):.4f}')

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'xgb_metrics.csv', index=False)
    summary = dict(train=len(train), validation=len(val), test=len(test),
                   input_features=len(features), encoded_features=int(X_train.shape[1]),
                   train_positive_rate=float(y_train.mean()),
                   positive_class_weight=float(ratio),
                   threshold_selection='Validation F1 only',
                   early_stopping='Validation average precision (aucpr)',
                   note='Independent patient test split; no CDC augmentation or probability calibration fitted.')
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n', metrics[['model','split','threshold_policy','AUROC','PR_AUC_AP','Precision','Recall','F1','Brier']].round(4).to_string(index=False))
    print('Saved to', out.resolve())

if __name__ == '__main__':
    main()


