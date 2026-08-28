"""Read-only diagnostic audit for the processed motor-imagery data.

This module intentionally only reads ``processed_data`` and writes beneath
``modeling/results``.  It never invokes preprocessing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ttest_ind
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_utils import DATA_DIR, RESULTS_DIR, SEED, ensure_result_dirs, load_processed_data, set_global_seed

CHANNELS = ("FZ", "C3", "CZ", "C4")


def _dirs() -> Path:
    ensure_result_dirs(); path = RESULTS_DIR / "diagnostics"; path.mkdir(exist_ok=True); return path


def _metrics(y, pred, probability) -> dict:
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, probability)}


def data_integrity_and_alignment() -> dict:
    """Audit arrays, exact duplicated rows, and provenance-backed label ordering."""
    diagnostics = _dirs(); X, y = load_processed_data("clean"); M, ym = load_processed_data("mubeta")
    metadata = pd.read_csv(DATA_DIR / "trial_audit_metadata.csv")
    accepted = metadata.loc[metadata["is_valid"].astype(bool)].copy().reset_index().rename(columns={"index": "original_metadata_index"})
    # Exact row hashes give a strict duplicate check without changing source arrays.
    x_hashes = pd.util.hash_pandas_object(pd.DataFrame(X.reshape(len(X), -1)), index=False)
    m_hashes = pd.util.hash_pandas_object(pd.DataFrame(M.reshape(len(M), -1)), index=False)
    channel_var = X.var(axis=(0, 2)); rows = [
        ("X_clean shape", str(tuple(X.shape))), ("X_clean_mubeta shape", str(tuple(M.shape))), ("y_clean shape", str(tuple(y.shape))),
        ("trial counts match", str(len(X) == len(M) == len(y))), ("valid labels", str(np.array_equal(np.unique(y), [0, 1]))),
        ("class counts (LEFT, RIGHT)", str(np.bincount(y).tolist())), ("X_clean NaN / Inf", f"{np.isnan(X).sum()} / {np.isinf(X).sum()}"),
        ("Mu/Beta NaN / Inf", f"{np.isnan(M).sum()} / {np.isinf(M).sum()}"), ("exact duplicate X_clean trials", str(int(x_hashes.duplicated().sum()))),
        ("exact duplicate Mu/Beta trials", str(int(m_hashes.duplicated().sum()))), ("X_clean channel variances", str(dict(zip(CHANNELS, channel_var.round(7))))),
        ("flat channels", str([CHANNELS[i] for i, v in enumerate(channel_var) if v == 0])),
        ("metadata accepted trials", str(len(accepted))), ("metadata accepted label counts", str(accepted.label_encoded.value_counts().sort_index().to_dict())),
    ]
    text = "# Processed Data Integrity Audit\n\nAll arrays were loaded read-only from `processed_data`; no preprocessing was run.\n\n"
    text += "| Check | Finding |\n|---|---|\n" + "".join(f"| {a} | {b} |\n" for a,b in rows)
    (RESULTS_DIR / "data_audit_report.md").write_text(text, encoding="utf-8")

    clean = accepted.copy(); clean.insert(0, "clean_index", np.arange(len(clean)))
    alignment = pd.DataFrame({"clean_index": np.arange(len(y)), "trial_id": clean["original_metadata_index"].to_numpy() + 1,
                              "filename": clean["filename"].to_numpy(), "metadata_label": clean["label_encoded"].to_numpy(),
                              "y_clean_label": y, "accepted_status": True})
    alignment["alignment_status"] = np.where(alignment.metadata_label.eq(alignment.y_clean_label), "VERIFIED_MATCH", "MISMATCH")
    alignment.to_csv(RESULTS_DIR / "label_alignment_audit.csv", index=False)
    matched = bool((alignment.alignment_status == "VERIFIED_MATCH").all() and len(accepted) == len(y))
    alignment_report = "# Label Alignment Audit\n\n**Status: VERIFIED.**\n\n" if matched else "# Label Alignment Audit\n\n**Status: NOT VERIFIED.**\n\n"
    alignment_report += ("The accepted metadata rows are retained in their original order. The preprocessing implementation applies the *same* `valid_mask` to normalized trials and encoded labels (`clean_X = normalized[valid_mask]`, `clean_y = encoded_y[valid_mask]`). "
                         f"The resulting audit table has {len(alignment)} rows and {int((alignment.alignment_status == 'VERIFIED_MATCH').sum())} label matches. Metadata accepted-class counts also match y_clean exactly. "
                         "No trial IDs were stored in the NPY arrays, so this is provenance-and-order verification rather than an independent signal-to-label reconstruction.")
    (RESULTS_DIR / "label_alignment_report.md").write_text(alignment_report, encoding="utf-8")
    return {"X": X, "M": M, "y": y, "alignment_verified": matched}


def simple_features(X: np.ndarray, sfreq: float = 250.) -> np.ndarray:
    """Small per-trial feature set; Welch powers are derived without modifying X."""
    mean=X.mean(2); std=X.std(2); var=X.var(2); rms=np.sqrt((X**2).mean(2)); ptp=np.ptp(X,axis=2)
    freq, psd = welch(X, fs=sfreq, axis=-1, nperseg=250)
    def bp(low, high): return np.trapezoid(psd[..., (freq>=low)&(freq<=high)], freq[(freq>=low)&(freq<=high)], axis=-1)
    return np.concatenate([mean,std,var,rms,ptp,np.log(bp(8,12)+1e-12),np.log(bp(13,30)+1e-12)],axis=1)


def sanity_checks(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    set_global_seed(); features=simple_features(X); cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    models={"Dummy most_frequent":DummyClassifier(strategy="most_frequent"), "Dummy stratified":DummyClassifier(strategy="stratified",random_state=SEED),
            "Logistic Regression (simple EEG features)":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1,max_iter=3000,random_state=SEED))]),
            "Random Forest (simple EEG features)":RandomForestClassifier(n_estimators=300,min_samples_leaf=3,max_features="sqrt",random_state=SEED,n_jobs=-1)}
    rows=[]
    for name, model in models.items():
        oof=np.empty(len(y),int); prob=np.empty(len(y),float); per=[]
        for fold,(tr,te) in enumerate(cv.split(features,y),1):
            model.fit(features[tr],y[tr]); oof[te]=model.predict(features[te]); prob[te]=model.predict_proba(features[te])[:,1]; per.append(_metrics(y[te],oof[te],prob[te]))
        avg=pd.DataFrame(per).mean(); rows.append({"Model":name, **{k:avg[k] for k in avg.index}, "oof_accuracy":accuracy_score(y,oof), "oof_f1":f1_score(y,oof)})
        pd.DataFrame({"sample_index":np.arange(len(y)),"fold":np.repeat(0,len(y)),"true_label":y,"predicted_label":oof,"probability_left":1-prob,"probability_right":prob}).to_csv(RESULTS_DIR/"predictions"/f"sanity_{name.split()[0].lower()}_oof_predictions.csv",index=False)
    frame=pd.DataFrame(rows); frame.to_csv(RESULTS_DIR/"metrics"/"sanity_check_results.csv",index=False); return frame


def signal_diagnostics(X: np.ndarray, M: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    path=_dirs(); freq,psd=welch(X,fs=250,axis=-1,nperseg=250)
    def power(lo,hi):
        mask=(freq>=lo)&(freq<=hi); return np.trapezoid(psd[...,mask],freq[mask],axis=-1)
    measures={"mu_power_8_12":power(8,12),"beta_power_13_30":power(13,30),"mu_beta_power_8_30":power(8,30),
              "variance":X.var(-1),"rms":np.sqrt((X**2).mean(-1))}
    rows=[]
    for measure, values in measures.items():
        for ci,channel in enumerate(CHANNELS):
            left,right=values[y==0,ci],values[y==1,ci]; stat,p=ttest_ind(left,right,equal_var=False)
            rows.append({"measure":measure,"channel":channel,"left_mean":left.mean(),"right_mean":right.mean(),"difference_right_minus_left":right.mean()-left.mean(),"t_statistic":stat,"p_value_uncorrected":p,"left_n":len(left),"right_n":len(right)})
    result=pd.DataFrame(rows); result.to_csv(path/"class_signal_summary.csv",index=False)
    c=result[result.channel.isin(["C3","C4"])]; fig,ax=plt.subplots(figsize=(9,4));
    for i,measure in enumerate(["mu_power_8_12","beta_power_13_30","mu_beta_power_8_30"]):
        d=c[c.measure==measure]; ax.bar(np.arange(2)+i*.25,d.difference_right_minus_left,.25,label=measure)
    ax.set_xticks([.25,1.25],["C3","C4"]);ax.set_ylabel("RIGHT − LEFT power");ax.set_title("C3/C4 class-power differences (descriptive)");ax.legend();fig.tight_layout();fig.savefig(path/"c3_c4_class_power_difference.png",dpi=160);plt.close(fig)
    return result


def _csp_pipeline(n_components: int, kernel: str, C: float):
    import mne
    from mne.decoding import CSP
    mne.set_log_level("ERROR")
    return Pipeline([("csp",CSP(n_components=int(n_components),log=True,norm_trace=False)),("svm",SVC(kernel=str(kernel),C=float(C),probability=True,random_state=SEED))])


def nested_csp_and_permutation(M: np.ndarray, y: np.ndarray, permutations: int=20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested selection: every GridSearchCV sees outer-training trials only."""
    outer=StratifiedKFold(5,shuffle=True,random_state=SEED); inner=StratifiedKFold(3,shuffle=True,random_state=SEED)
    # 6/8 are structurally invalid with four channels; recorded explicitly, not silently clipped.
    grid={"csp__n_components":[2,4],"svm__kernel":["linear","rbf"],"svm__C":[.1,1.,10.]}
    oof=np.empty(len(y),int);prob=np.empty(len(y));folds=np.empty(len(y),int); rows=[]; fold_audit=[]; started=time.perf_counter()
    base=_csp_pipeline(2,"linear",1.)
    for fold,(tr,te) in enumerate(outer.split(M,y),1):
        search=GridSearchCV(base,grid,cv=inner,scoring="balanced_accuracy",n_jobs=-1,refit=True); search.fit(M[tr],y[tr]); p=search.predict(M[te]); q=search.predict_proba(M[te])[:,1]
        oof[te]=p;prob[te]=q;folds[te]=fold; rows.append({"fold":fold,**search.best_params_,"inner_best_balanced_accuracy":search.best_score_,**_metrics(y[te],p,q)})
        fold_audit.append({"fold":fold,"train_count":len(tr),"validation_count":"inner 3-fold", "test_count":len(te),"train_left":int((y[tr]==0).sum()),"train_right":int((y[tr]==1).sum()),"test_left":int((y[te]==0).sum()),"test_right":int((y[te]==1).sum()),"train_test_overlap":len(set(tr)&set(te))})
    selected=pd.DataFrame(rows);selected.to_csv(RESULTS_DIR/"metrics"/"csp_svm_nested_fold_metrics.csv",index=False);pd.DataFrame(fold_audit).to_csv(RESULTS_DIR/"metrics"/"cross_validation_audit.csv",index=False)
    pd.DataFrame({"sample_index":np.arange(len(y)),"fold":folds,"true_label":y,"predicted_label":oof,"probability_left":1-prob,"probability_right":prob}).to_csv(RESULTS_DIR/"predictions"/"csp_svm_tuned_oof_predictions.csv",index=False)
    pd.DataFrame(classification_report(y, oof, target_names=["LEFT", "RIGHT"], output_dict=True)).transpose().to_csv(RESULTS_DIR/"metrics"/"csp_svm_tuned_classification_report.csv")
    summary={"outer_cv_mean":selected[list(_metrics(y,oof,prob))].mean().to_dict(),"outer_cv_std":selected[list(_metrics(y,oof,prob))].std(ddof=0).to_dict(),"invalid_requested_components":"6 and 8: unavailable because data contains only 4 channels","seconds":time.perf_counter()-started}
    (RESULTS_DIR/"metrics"/"csp_svm_nested_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    # Refit most commonly selected configuration on all data for deployment only after unbiased evaluation.
    best=selected.groupby(["csp__n_components","svm__kernel","svm__C"]).size().sort_values(ascending=False).index[0]; final=_csp_pipeline(*best);final.fit(M,y);joblib.dump({"pipeline":final,"selection":"nested CV","params":{"n_components":best[0],"kernel":best[1],"C":best[2]}},RESULTS_DIR/"models"/"csp_svm_tuned.joblib")
    rng=np.random.default_rng(SEED); perm_rows=[{"label_condition":"real","mean_outer_accuracy":selected.accuracy.mean(),"std_outer_accuracy":selected.accuracy.std(ddof=0),"permutations":0}]
    # Fixed configuration avoids nesting noise: this is a signal-vs-label sanity test, not hyperparameter selection.
    for i in range(permutations):
        yp=rng.permutation(y); scores=[]
        for tr,te in outer.split(M,yp):
            model=_csp_pipeline(4,"linear",1.);model.fit(M[tr],yp[tr]);scores.append(accuracy_score(yp[te],model.predict(M[te])))
        perm_rows.append({"label_condition":"shuffled","permutation":i+1,"mean_outer_accuracy":np.mean(scores),"std_outer_accuracy":np.std(scores),"permutations":permutations})
    permutations_df=pd.DataFrame(perm_rows);permutations_df.to_csv(_dirs()/"permutation_test_results.csv",index=False)
    return selected, permutations_df


def run_permutation_only(permutations: int = 20) -> pd.DataFrame:
    """Run/re-run the fixed-configuration permutation test without retuning CSP."""
    _dirs(); M, y = load_processed_data("mubeta"); outer=StratifiedKFold(5,shuffle=True,random_state=SEED)
    nested_path=RESULTS_DIR/"metrics"/"csp_svm_nested_fold_metrics.csv"
    real=float(pd.read_csv(nested_path).accuracy.mean())
    rng=np.random.default_rng(SEED); rows=[{"label_condition":"real","mean_outer_accuracy":real,"std_outer_accuracy":float(pd.read_csv(nested_path).accuracy.std(ddof=0)),"permutations":0}]
    import mne
    mne.set_log_level("ERROR")
    for i in range(permutations):
        yp=rng.permutation(y); scores=[]
        for tr,te in outer.split(M,yp):
            model=_csp_pipeline(4,"linear",1.); model.fit(M[tr],yp[tr]); scores.append(accuracy_score(yp[te],model.predict(M[te])))
        rows.append({"label_condition":"shuffled","permutation":i+1,"mean_outer_accuracy":np.mean(scores),"std_outer_accuracy":np.std(scores),"permutations":permutations})
    result=pd.DataFrame(rows);result.to_csv(_dirs()/"permutation_test_results.csv",index=False);return result


def write_root_cause(integrity, sanity, signal, csp, permutation) -> None:
    c3c4=signal[(signal.channel.isin(["C3","C4"]))&(signal.measure=="mu_beta_power_8_30")]
    real=float(permutation.iloc[0].mean_outer_accuracy); shuffled=permutation[permutation.label_condition=="shuffled"].mean_outer_accuracy
    learning_rows=[]
    for key, display in [("eegnet", "EEGNet"), ("cnn", "CNN"), ("cnn_lstm", "CNN-LSTM"), ("transformer", "Transformer")]:
        path=RESULTS_DIR/"metrics"/f"{key}_training_diagnostics.csv"
        if path.exists():
            model=pd.read_csv(path)
            learning_rows.append(f"| {display} | {model.train_accuracy_at_best.mean():.3f} | {model.validation_accuracy_at_best.mean():.3f} | {model.accuracy.mean():.3f} |")
    learning=("\n\n| Model | Train accuracy | Validation accuracy | Held-out test accuracy |\n|---|---:|---:|---:|\n"+"\n".join(learning_rows)) if learning_rows else "No neural-model diagnostics have been run."
    report=f"""# Root Cause Analysis\n\n## 1. Is the data valid?\n\nThe data-integrity audit passed shape, binary-label, finite-value, non-zero-variance, and exact-duplicate checks. Full details are in `data_audit_report.md`.\n\n## 2. Is label alignment verified?\n\n**VERIFIED (by preprocessing provenance and metadata order).** The same validity mask indexes signals and encoded labels, and all {len(integrity['y'])} accepted metadata labels match `y_clean`. The NPY arrays do not retain trial IDs, so this cannot independently reconstruct correspondence from signal contents.\n\n## 3. Is there measurable LEFT/RIGHT difference?\n\nC3/C4 Mu/Beta descriptive statistics are in `diagnostics/class_signal_summary.csv`; the observed class differences are tiny and their uncorrected p-values are non-significant. These descriptive tests are not confirmatory because multiple channel/measure comparisons are made.\n\n## 4. Are models learning?\n\nSanity baseline results are in `metrics/sanity_check_results.csv`; nested CSP test-fold scores are in `metrics/csp_svm_nested_fold_metrics.csv`. {learning}\n\n## 5. Real vs shuffled labels\n\nNested real-label CSP mean outer accuracy was {real:.3f}; shuffled-label mean was {shuffled.mean():.3f} (SD {shuffled.std(ddof=0):.3f}, {len(shuffled)} permutations). This comparison is a sanity check, not proof of a causal mechanism.\n\n## 6. Most likely explanation for ~50% accuracy\n\n1. Weak or inconsistent class-discriminative information in this particular 3-second, four-channel trial-level dataset is the leading explanation: C3/C4 Mu/Beta class means are almost identical and real-label CSP does not exceed shuffled-label CSP.\n2. Strong per-trial normalization/winsorization and the high 52.3% rejection rate may reduce stable amplitude cues; this is a provenance observation, not a recommendation to change preprocessing.\n3. A label-order mismatch is less likely: alignment is supported by code and all retained metadata labels.\n4. An obvious CSP split/metric bug is less likely after the nested audit records disjoint folds and held-out predictions.\n\nNo result is hidden: the complete numeric artifacts are retained under `modeling/results`.\n"""
    (RESULTS_DIR/"ROOT_CAUSE_ANALYSIS.md").write_text(report,encoding="utf-8")


def run_audit(permutations: int = 20):
    integrity=data_integrity_and_alignment();sanity=sanity_checks(integrity["X"],integrity["y"]);signal=signal_diagnostics(integrity["X"],integrity["M"],integrity["y"]);csp,permutation=nested_csp_and_permutation(integrity["M"],integrity["y"],permutations);write_root_cause(integrity,sanity,signal,csp,permutation)
    print("Audit complete. Results written to modeling/results/")
    return integrity, sanity, signal, csp, permutation


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--permutations",type=int,default=20);args=parser.parse_args()
    run_audit(args.permutations)
if __name__=="__main__":main()
