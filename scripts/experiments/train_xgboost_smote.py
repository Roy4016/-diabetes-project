#!/usr/bin/env python3
"""Training-only SMOTENC + GPU XGBoost, compatible with UCI Step 1-6 files.

Requires: numpy pandas scipy scikit-learn==1.8.0 imbalanced-learn xgboost joblib matplotlib

SMOTENC operates on numeric + categorically-encoded raw training columns BEFORE
OneHotEncoding. The saved original preprocessor is then used to transform the
resampled training frame and untouched validation/test frames.
"""
import argparse
import json
from pathlib import Path
from _paths import UCI_HOLDOUT, result_dir

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, brier_score_loss,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve


def choose_f1_threshold(y, prob):
    p, r, thresholds = precision_recall_curve(y, prob)
    if len(thresholds) == 0:
        return .5
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), AUROC=float(roc_auc_score(y, prob)),
                PR_AUC_AP=float(average_precision_score(y, prob)),
                Accuracy=float(accuracy_score(y, pred)),
                Precision=float(precision_score(y, pred, zero_division=0)),
                Recall=float(recall_score(y, pred, zero_division=0)),
                F1=float(f1_score(y, pred, zero_division=0)),
                Brier=float(brier_score_loss(y, prob)),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


def draw_curves(y, prob, out, prefix):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(fpr, tpr, label=f'AUROC={roc_auc_score(y, prob):.3f}')
    ax.plot([0, 1], [0, 1], '--'); ax.set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(out / f'{prefix}_roc.png', dpi=170); plt.close(fig)
    p, r, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(r, p, label=f'AP={average_precision_score(y, prob):.3f}')
    ax.axhline(np.mean(y), ls='--', label=f'Prevalence={np.mean(y):.3f}')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(out / f'{prefix}_pr.png', dpi=170); plt.close(fig)
    frac, mean = calibration_curve(y, prob, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    ax.plot(mean, frac, 'o-', label=prefix)
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction of positives',
           xlim=(0, 1), ylim=(0, 1), title='Calibration Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(out / f'{prefix}_calibration.png', dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Training-only SMOTENC + XGBoost')
    ap.add_argument('--data-dir', default=UCI_HOLDOUT)
    ap.add_argument('--outdir', default=result_dir('xgboost_smote_results'))
    ap.add_argument('--sampling-strategy', type=float, default=0.35,
                    help='target positive/negative ratio after SMOTENC (e.g. 0.35, 0.5, 1.0)')
    ap.add_argument('--k-neighbors', type=int, default=5)
    ap.add_argument('--n-estimators', type=int, default=500)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--early-stopping-rounds', type=int, default=30)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.data_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(src / 'train.csv', low_memory=False)
    val = pd.read_csv(src / 'val.csv', low_memory=False)
    test = pd.read_csv(src / 'test.csv', low_memory=False)
    saved = joblib.load(src / 'preprocessor.joblib')
    preprocessor = saved['preprocessor']
    cols = list(saved['feature_columns'])
    cat_cols = list(saved['categorical_columns'])
    num_cols = list(saved['numeric_columns'])
    assert set(train.patient_nbr).isdisjoint(set(val.patient_nbr))
    assert set(train.patient_nbr).isdisjoint(set(test.patient_nbr))
    assert set(val.patient_nbr).isdisjoint(set(test.patient_nbr))
    assert set(cols) == set(cat_cols + num_cols), 'Feature groups do not match!'

    y_train = train.target.astype(int).to_numpy()
    y_val = val.target.astype(int).to_numpy()
    y_test = test.target.astype(int).to_numpy()
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    original_ratio = n_pos / n_neg
    if args.sampling_strategy <= original_ratio:
        raise ValueError(f'--sampling-strategy must exceed original minority/majority ratio {original_ratio:.4f}')

    # SMOTENC expects raw numeric + categorical variables, not one-hot vectors.
    # Fit all temporary encoding/imputation decisions on TRAIN only.
    data = train[cols].copy()
    num_medians = {}
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors='coerce')
        median = data[c].median()
        num_medians[c] = float(median) if pd.notna(median) else 0.0
        data[c] = data[c].fillna(num_medians[c]).astype('float64')

    categories = {}
    for c in cat_cols:
        s = data[c].astype('object').where(data[c].notna(), 'Unknown').astype(str)
        levels = pd.Index(s.unique()).tolist()
        categories[c] = levels
        data[c] = pd.Categorical(s, categories=levels).codes.astype('int32')

    print(f'[Data] train={len(train):,}, val={len(val):,}, test={len(test):,}', flush=True)
    print(f'[Before SMOTENC] negative={n_neg:,}, positive={n_pos:,}, pos/neg={original_ratio:.4f}', flush=True)
    print(f'[SMOTENC] sampling_strategy={args.sampling_strategy}, k_neighbors={args.k_neighbors}', flush=True)
    cat_indices = [cols.index(c) for c in cat_cols]
    sampler = SMOTENC(categorical_features=cat_indices,
                      sampling_strategy=args.sampling_strategy,
                      k_neighbors=args.k_neighbors, random_state=args.seed)
    resampled, y_resampled = sampler.fit_resample(data[cols], y_train)
    resampled = pd.DataFrame(resampled, columns=cols)
    y_resampled = np.asarray(y_resampled, dtype='int32')
    for c in num_cols:
        resampled[c] = pd.to_numeric(resampled[c], errors='raise').astype(float)
    for c in cat_cols:
        levels = categories[c]
        codes = np.rint(pd.to_numeric(resampled[c], errors='raise').to_numpy(dtype=float)).astype(int)
        if codes.min() < 0 or codes.max() >= len(levels):
            raise RuntimeError(f'Invalid generated category codes in {c}')
        resampled[c] = pd.Series(pd.Categorical.from_codes(codes, categories=levels)).astype(object)
    new_pos = int(y_resampled.sum()); new_neg = len(y_resampled) - new_pos
    print(f'[After SMOTENC] negative={new_neg:,}, positive={new_pos:,}, pos/neg={new_pos/new_neg:.4f}', flush=True)

    def transform(frame):
        x = frame[cols].copy()
        for c in cat_cols:
            x[c] = x[c].astype('object').where(x[c].notna(), np.nan)
        return preprocessor.transform(x)

    # Original train-fitted preprocessor; NEVER refit on SMOTE/val/test.
    X_train = transform(resampled)
    X_val = transform(val)
    X_test = transform(test)
    print(f'[Encoded features] {X_train.shape[1]:,}; device={args.device}', flush=True)

    model = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        random_state=args.seed, n_jobs=-1, scale_pos_weight=1.0,
        device=args.device, early_stopping_rounds=args.early_stopping_rounds,
    )
    print('[Training XGBoost...]', flush=True)
    model.fit(X_train, y_resampled, eval_set=[(X_val, y_val)], verbose=False)
    # CPU sparse matrices vs CUDA model can print an innocuous inference device warning.
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    threshold = choose_f1_threshold(y_val, p_val)
    rows = []
    name = f'xgb_smotenc_{str(args.sampling_strategy).replace(".", "p")}'
    for split, labels, probs in [('validation', y_val, p_val), ('test', y_test, p_test)]:
        for policy, t in [('default_0.5', 0.5), ('val_f1_optimal', threshold)]:
            rows.append(dict(model=name, split=split, threshold_policy=policy,
                             **metrics(labels, probs, t)))
    pd.DataFrame(rows).to_csv(out / 'xgb_metrics.csv', index=False)
    joblib.dump(model, out / f'{name}_model.joblib')
    pd.DataFrame({'patient_nbr': test.patient_nbr, 'target': y_test,
                  'predicted_probability': p_test}).to_csv(out / f'{name}_test_predictions.csv', index=False)
    draw_curves(y_test, p_test, out, name)
    summary = {
        'method': 'SMOTENC on original train only, followed by existing train-fitted preprocessor',
        'sampling_strategy_positive_over_negative': args.sampling_strategy,
        'train_original': {'negative': n_neg, 'positive': n_pos},
        'train_resampled': {'negative': new_neg, 'positive': new_pos},
        'validation': {'rows': len(val), 'positive_rate': float(y_val.mean())},
        'test': {'rows': len(test), 'positive_rate': float(y_test.mean())},
        'feature_count': X_train.shape[1], 'best_iteration': int(model.best_iteration),
        'actual_training_rounds': int(model.get_booster().num_boosted_rounds()),
        'device': args.device, 'validation_selected_threshold': threshold,
        'note': 'CCS/CCI/CDC not included; test not used for tuning; no class weighting combined with SMOTENC.',
    }
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[Done] best_iteration={model.best_iteration}; threshold={threshold:.4f}', flush=True)
    print(pd.DataFrame(rows)[['model','split','threshold_policy','AUROC','PR_AUC_AP','Precision','Recall','F1','Brier']].round(4).to_string(index=False), flush=True)
    print(f'Saved to {out.resolve()}', flush=True)

if __name__ == '__main__':
    main()
