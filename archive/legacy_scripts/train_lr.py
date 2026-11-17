#!/usr/bin/env python3
"""Train a patient-disjoint LR baseline using outputs from UCI Step 1–6.

Usage: python train_lr.py --data-dir uci_prepared_with_ids --outdir lr_results
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, accuracy_score, brier_score_loss,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': round(float(threshold), 6),
        'AUROC': float(roc_auc_score(y, p)),
        'PR_AUC_AP': float(average_precision_score(y, p)),
        'Accuracy': float(accuracy_score(y, pred)),
        'Precision': float(precision_score(y, pred, zero_division=0)),
        'Recall': float(recall_score(y, pred, zero_division=0)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'Brier': float(brier_score_loss(y, p)),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return 0.5
    f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1s)])


def plot_all(y, p, output, name):
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'AUROC = {roc_auc_score(y,p):.3f}')
    ax.plot([0,1],[0,1], linestyle='--', linewidth=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(output/f'{name}_roc.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc, pr, label=f'AP = {average_precision_score(y,p):.3f}')
    ax.axhline(np.mean(y), linestyle='--', linewidth=1, label=f'Prevalence = {np.mean(y):.3f}')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–Recall Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(output/f'{name}_pr.png', dpi=180); plt.close(fig)
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0,1],[0,1], linestyle='--', linewidth=1, label='Perfect calibration')
    ax.plot(mean_pred,frac, marker='o', label='LR')
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction of positives', title='Calibration Curve', xlim=(0,1), ylim=(0,1))
    ax.legend(); fig.tight_layout(); fig.savefig(output/f'{name}_calibration.png', dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='uci_prepared_with_ids')
    parser.add_argument('--outdir', default='lr_results')
    parser.add_argument('--max-iter', type=int, default=1500)
    args=parser.parse_args()
    src, out = Path(args.data_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    train=pd.read_csv(src/'train.csv', low_memory=False)
    val=pd.read_csv(src/'val.csv', low_memory=False)
    test=pd.read_csv(src/'test.csv', low_memory=False)
    spec=joblib.load(src/'preprocessor.joblib')
    prep=spec['preprocessor']
    features=spec['feature_columns']
    cats=spec['categorical_columns']
    for a, b in [(train,val),(train,test),(val,test)]:
        assert set(a.patient_nbr).isdisjoint(set(b.patient_nbr)), 'Patient overlap!'
    def input_frame(df):
        x=df[features].copy()
        for c in cats:
            x[c]=x[c].astype('object').where(x[c].notna(), np.nan)
        return x
    xtr=prep.transform(input_frame(train))
    xv=prep.transform(input_frame(val))
    xt=prep.transform(input_frame(test))
    ytr=train.target.astype(int).to_numpy()
    yv=val.target.astype(int).to_numpy()
    yt=test.target.astype(int).to_numpy()
    all_results=[]
    for title, weight in [('lr_standard',None),('lr_balanced','balanced')]:
        clf=LogisticRegression(solver='lbfgs',max_iter=args.max_iter,
                               class_weight=weight, random_state=42, C=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            clf.fit(xtr,ytr)
        convergence_warnings=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        if convergence_warnings:
            print('WARNING:',title,'did not converge:',convergence_warnings)
        pv=clf.predict_proba(xv)[:,1]
        pt=clf.predict_proba(xt)[:,1]
        chosen=best_f1_threshold(yv,pv)
        for split, ys, ps in [('validation',yv,pv),('test',yt,pt)]:
            for name,threshold in [('default_0.5',0.5),('val_f1_optimal',chosen)]:
                row={'model':title,'split':split,'threshold_policy':name,**get_metrics(ys,ps,threshold)}
                all_results.append(row)
        plot_all(yt,pt,out,title)
        joblib.dump(clf,out/f'{title}_model.joblib')
        pd.DataFrame({'patient_nbr':test.patient_nbr,'target':yt,'predicted_probability':pt}).to_csv(out/f'{title}_test_predictions.csv',index=False)
        print(f'{title}: val-F1 threshold={chosen:.4f}; test AUROC={roc_auc_score(yt,pt):.4f}; AP={average_precision_score(yt,pt):.4f}')
    results=pd.DataFrame(all_results)
    results.to_csv(out/'lr_metrics.csv',index=False)
    (out/'run_summary.json').write_text(json.dumps({'data_dir':str(src),'features':len(features),
        'encoded_features':int(xtr.shape[1]), 'counts':{'train':len(train),'val':len(val),'test':len(test)},
        'positive_rates':{'train':float(ytr.mean()),'val':float(yv.mean()),'test':float(yt.mean())},
        'threshold_selection':'max F1 on validation only; test remains held out',
        'note':'PR_AUC_AP is sklearn average_precision_score (AP). No calibration fitting, SMOTE, or CDC augmentation.'},indent=2))
    print('\n',results[['model','split','threshold_policy','threshold','AUROC','PR_AUC_AP','Precision','Recall','F1','Brier']].round(4).to_string(index=False))
    print('Saved:',out.resolve())

if __name__=='__main__':
    main()
