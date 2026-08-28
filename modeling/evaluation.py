"""Metrics, artifact writing, and figures shared by every model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # File-only figures; works on servers and headless CI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from .data_utils import RESULTS_DIR, ensure_result_dirs

METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")


def compute_metrics(y_true, y_pred, probabilities) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else np.nan
    return result


def save_evaluation(model_name: str, y_true, y_pred, probabilities, folds, fold_metrics: list[dict],
                    parameters: int | None, training_seconds: float) -> dict:
    """Persist OOF predictions, fold metrics, final metrics, and model figures."""
    ensure_result_dirs()
    stem = {"CSP + SVM": "csp_svm", "EEGNet": "eegnet", "CNN": "cnn",
            "CNN-LSTM": "cnn_lstm", "Transformer": "transformer"}[model_name]
    prediction_df = pd.DataFrame({"true_label": y_true, "predicted_label": y_pred,
                                  "prediction_probability": probabilities, "fold_number": folds})
    prediction_df.to_csv(RESULTS_DIR / "predictions" / f"{stem}_oof_predictions.csv", index=False)
    oof = compute_metrics(y_true, y_pred, probabilities)
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(RESULTS_DIR / "metrics" / f"{stem}_fold_metrics.csv", index=False)
    summary = {f"{metric}_mean": float(fold_df[metric].mean()) for metric in METRIC_NAMES}
    summary.update({f"{metric}_std": float(fold_df[metric].std(ddof=0)) for metric in METRIC_NAMES})
    summary["oof_metrics"] = {key: float(value) for key, value in oof.items()}
    summary["number_of_parameters"] = parameters
    summary["training_time_seconds"] = training_seconds
    with open(RESULTS_DIR / "metrics" / f"{stem}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _save_confusion_matrix(stem, y_true, y_pred)
    _save_roc_curve(stem, y_true, probabilities)
    _update_comparison(model_name, summary)
    return summary


def _save_confusion_matrix(stem: str, y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["LEFT", "RIGHT"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{stem.replace('_', ' ').upper()} Confusion Matrix")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "figures" / f"{stem}_confusion_matrix.png", dpi=160); plt.close(fig)


def _save_roc_curve(stem: str, y_true, probabilities) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities); auc = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{stem.replace('_', ' ').upper()} ROC")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(RESULTS_DIR / "figures" / f"{stem}_roc_curve.png", dpi=160); plt.close(fig)


def save_training_curves(model_name: str, histories: list[dict]) -> None:
    """Plot per-fold histories together, retaining train/validation loss and accuracy."""
    stem = model_name.lower().replace("-", "_")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for fold_index, history in enumerate(histories):
        epochs = range(1, len(history["train_loss"]) + 1)
        labels = ("Training", "Validation") if fold_index == 0 else (None, None)
        axes[0].plot(epochs, history["train_loss"], alpha=.35, color="tab:blue", label=labels[0])
        axes[0].plot(epochs, history["val_loss"], alpha=.35, color="tab:orange", label=labels[1])
        axes[1].plot(epochs, history["train_accuracy"], alpha=.35, color="tab:blue", label=labels[0])
        axes[1].plot(epochs, history["val_accuracy"], alpha=.35, color="tab:orange", label=labels[1])
    for ax, title, ylabel in zip(axes, ("Loss", "Accuracy"), ("Cross-entropy", "Accuracy")):
        ax.set(title=f"{model_name} {title}", xlabel="Epoch", ylabel=ylabel); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "figures" / f"{stem}_training_curves.png", dpi=160); plt.close(fig)


def _update_comparison(model_name: str, summary: dict) -> None:
    path = RESULTS_DIR / "metrics" / "model_comparison.csv"
    row = {"Model": model_name, "Accuracy Mean": summary["accuracy_mean"], "Accuracy Std": summary["accuracy_std"],
           "Balanced Accuracy": summary["balanced_accuracy_mean"], "Precision": summary["precision_mean"],
           "Recall": summary["recall_mean"], "F1 Score": summary["f1_mean"], "ROC-AUC": summary["roc_auc_mean"],
           "Number of Parameters": summary["number_of_parameters"], "Training Time": summary["training_time_seconds"]}
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=row.keys())
    table = table[table["Model"] != model_name]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    if not table.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); table.plot.bar(x="Model", y=["Accuracy Mean", "F1 Score"], ax=ax, rot=20)
        ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison"); fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "model_performance_comparison.png", dpi=160); plt.close(fig)
