"""Leakage-free CSP + linear SVM cross-validation baseline."""
from __future__ import annotations
import time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from .data_utils import RESULTS_DIR, SEED, ensure_result_dirs, load_processed_data, set_global_seed
from .evaluation import compute_metrics, save_evaluation


def run_csp_svm() -> dict:
    set_global_seed(); ensure_result_dirs(); X, y = load_processed_data("mubeta")
    try:
        from mne.decoding import CSP
    except ImportError as error:
        raise ImportError("CSP + SVM requires MNE. Install modeling/requirements_modeling.txt") from error
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred, prob, folds = np.empty(len(y), int), np.empty(len(y), float), np.empty(len(y), int); metrics = []
    started = time.perf_counter()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        csp = CSP(n_components=4, log=True, norm_trace=False)
        svm = SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)
        svm.fit(csp.fit_transform(X[train_idx], y[train_idx]), y[train_idx])
        pred[test_idx] = svm.predict(csp.transform(X[test_idx])); prob[test_idx] = svm.predict_proba(csp.transform(X[test_idx]))[:, 1]; folds[test_idx] = fold
        metrics.append({"fold": fold, **compute_metrics(y[test_idx], pred[test_idx], prob[test_idx])})
    seconds = time.perf_counter() - started
    # Final refit produces the deployable artifact; CV scores above remain unbiased.
    csp = CSP(n_components=4, log=True, norm_trace=False); svm = SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)
    svm.fit(csp.fit_transform(X, y), y); joblib.dump({"csp": csp, "svm": svm, "seed": SEED, "input": "X_clean_mubeta.npy"}, RESULTS_DIR / "models" / "csp_svm.joblib")
    return save_evaluation("CSP + SVM", y, pred, prob, folds, metrics, None, seconds)
