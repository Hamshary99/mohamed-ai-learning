"""
Module: evaluate.py
Single Responsibility Principle (SRP):
Performance evaluation, leaderboard generation, confusion matrix diagnostics,
feature importance extraction, and held-out test set reporting.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def build_leaderboard(
    arena_results: Dict[str, Dict[str, float]],
    sort_by: str = "Test F1",
    overfit_threshold: float = 0.08
) -> pd.DataFrame:
    """
    Step 1: Construct competitive model leaderboard and analyze the Overfit Gap.
    Overfit Gap (Δ) = Train F1 - Val F1.
    """
    rows = []
    for name, metrics in arena_results.items():
        train_f1 = metrics["train_f1"]
        test_f1 = metrics["test_f1"]
        train_acc = metrics["train_accuracy"]
        test_acc = metrics["test_accuracy"]
        train_auc = metrics["train_roc_auc"]
        test_auc = metrics["test_roc_auc"]

        f1_gap = train_f1 - test_f1
        status = "OVERFITTING" if f1_gap > overfit_threshold else "HEALTHY"

        rows.append({
            "Model": name,
            "Test F1": round(test_f1, 4),
            "Train F1": round(train_f1, 4),
            "F1 Gap (Delta)": round(f1_gap, 4),
            "Test Accuracy": round(test_acc, 4),
            "Train Accuracy": round(train_acc, 4),
            "Test ROC-AUC": round(test_auc, 4),
            "Status": status
        })

    leaderboard = pd.DataFrame(rows)
    sort_col = "Test F1" if sort_by in ["val_f1", "Test F1", "test_f1"] else sort_by
    if sort_col in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return leaderboard


def compare_tuned_models(
    tuning_results: Dict[str, Dict[str, Any]],
    baseline_results: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Step 1B: Multi-Model Parameter & Performance Comparison Leaderboard.
    """
    rows = []
    for name, data in tuning_results.items():
        best_params = data["best_params"]
        param_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in best_params.items())
        tuned_score = data["best_score"]

        if baseline_results and name in baseline_results:
            baseline_score = baseline_results[name]["test_f1"]
            delta_gain = tuned_score - baseline_score
        else:
            baseline_score = np.nan
            delta_gain = np.nan

        rows.append({
            "Model": name,
            "Tuned CV F1": round(tuned_score, 4),
            "Baseline F1": round(baseline_score, 4) if not np.isnan(baseline_score) else "-",
            "Delta Gain": round(delta_gain, 4) if not np.isnan(delta_gain) else "-",
            "Best Parameters": param_str
        })

    comparison = pd.DataFrame(rows).sort_values(by="Tuned CV F1", ascending=False).reset_index(drop=True)
    comparison.insert(0, "Rank", range(1, len(comparison) + 1))
    return comparison


def evaluate_confusion_matrix(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Step 2: Cross-validated Confusion Matrix & Error Breakdown.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, X_train, y_train, cv=skf)
    cm = confusion_matrix(y_train, y_pred)
    tn, fp, fn, tp = cm.ravel()

    breakdown = {
        "True Negatives (Correct Perished)": int(tn),
        "False Positives (Type I Error)": int(fp),
        "False Negatives (Type II Error)": int(fn),
        "True Positives (Correct Survived)": int(tp),
    }
    return cm, breakdown


def extract_feature_importances(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: int = 10,
    save_fig_path: Optional[str] = None
) -> pd.Series:
    """
    Step 3: Extract Gini feature importances to interpret model decision boundaries.
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        return pd.Series(dtype=float)

    series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        series.head(top_n).sort_values().plot(kind="barh", color="#1f77b4")
        plt.title(f"Top {top_n} Predictive Features")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=300)
        plt.close()
        print(f"[*] Saved feature importance plot to: {save_fig_path}")

    return series


def evaluate_held_out_set(
    model: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Any]:
    """
    Step 4: Strict Held-Out Evaluation on untouched validation data.
    """
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    report = classification_report(y_val, y_pred)

    print("\n" + "=" * 55)
    print("         HELD-OUT VALIDATION SET EVALUATION")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")
    print("Classification Report:")
    print(report)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "classification_report": report
    }
