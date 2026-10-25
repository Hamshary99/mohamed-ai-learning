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
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def build_leaderboard(
    arena_results: Dict[str, Dict[str, float]],
    sort_by: str = "Test F1",
    overfit_threshold: float = 0.08
) -> pd.DataFrame:
    """
    Step 1: Construct competitive model leaderboard and analyze the Overfit Gap.
    Overfit Gap (Delta) = Train F1 - Val F1.
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
        tuned_f1 = data["best_score"]

        if baseline_results and name in baseline_results:
            baseline_score = baseline_results[name]["test_f1"]
            delta_gain = tuned_f1 - baseline_score
        else:
            baseline_score = np.nan
            delta_gain = np.nan

        rows.append({
            "Model": name,
            "Tuned CV F1": round(tuned_f1, 4),
            "Baseline F1": round(baseline_score, 4) if not np.isnan(baseline_score) else "-",
            "Delta Gain": round(delta_gain, 4) if not np.isnan(delta_gain) else "-",
            "Best Parameters": param_str
        })

    comparison = pd.DataFrame(rows).sort_values(by="Tuned CV F1", ascending=False).reset_index(drop=True)
    comparison.insert(0, "Rank", range(1, len(comparison) + 1))
    verdicts = ["[+] CHAMPION" if i == 0 else "Runner-Up" if i == 1 else "Candidate" for i in range(len(comparison))]
    comparison.insert(2, "Verdict", verdicts)
    return comparison


def plot_model_arena(
    leaderboard: pd.DataFrame,
    save_fig_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Step 1C: Visual Leaderboard comparing Candidate Models in 5-Fold Stratified CV.
    Plots Test F1 vs Train F1 side-by-side, visually illustrating the Overfit Gap.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    models = leaderboard["Model"].tolist()
    y_pos = np.arange(len(models))
    height = 0.35

    bars_val = ax.barh(y_pos - height / 2, leaderboard["Test F1"], height, label="Validation F1 (CV)", color="#2563eb", edgecolor="#1d4ed8")
    bars_train = ax.barh(y_pos + height / 2, leaderboard["Train F1"], height, label="Train F1 (CV)", color="#94a3b8", edgecolor="#64748b", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Model Arena: Validation Performance & Overfit Gap Analysis", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    for i, (_, row) in enumerate(leaderboard.iterrows()):
        val_f1 = row["Test F1"]
        train_f1 = row["Train F1"]
        gap = row["F1 Gap (Delta)"]
        status = row["Status"]

        ax.text(val_f1 + 0.015, i - height / 2, f"{val_f1:.3f}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1e3a8a")
        gap_color = "#dc2626" if status == "OVERFITTING" else "#16a34a"
        gap_text = f"Delta={gap:.2f} [{status[:4]}]" if status == "OVERFITTING" else f"Delta={gap:.2f}"
        ax.text(train_f1 + 0.015, i + height / 2, gap_text, va="center", ha="left", fontsize=8.5, fontweight="bold", color=gap_color)

    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"[*] Saved arena benchmark plot to: {save_fig_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return fig


def evaluate_confusion_matrix(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
    save_fig_path: Optional[str] = None,
    show_plot: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Step 2: Cross-validated Confusion Matrix & Error Breakdown with Visual Heatmap.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, X_train, y_train, cv=skf)
    cm = confusion_matrix(y_train, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_train)

    breakdown = {
        "True Negatives (Correct Perished)": int(tn),
        "False Positives (Type I Error)": int(fp),
        "False Negatives (Type II Error)": int(fn),
        "True Positives (Correct Survived)": int(tp),
    }

    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 6))

    labels = np.array([
        [f"True Negative\n{tn} ({tn/total:.1%})\nCorrect Perished", f"False Positive (Type I)\n{fp} ({fp/total:.1%})\nFalse Alarm"],
        [f"False Negative (Type II)\n{fn} ({fn/total:.1%})\nMissed Survivor", f"True Positive\n{tp} ({tp/total:.1%})\nCorrect Survived"]
    ])

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "#1e293b"
            ax.text(j, i, labels[i, j], ha="center", va="center", color=color,
                    fontsize=10.5, fontweight="bold", multialignment="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Perished (0)", "Survived (1)"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Perished (0)", "Survived (1)"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Ground Truth", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Out-of-Fold Cross-Validation Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    stats_text = f"Accuracy: {acc:.1%}  |  Precision: {precision:.1%}  |  Recall: {recall:.1%}  |  Specificity: {specificity:.1%}"
    plt.figtext(0.5, 0.01, stats_text, wrap=True, horizontalalignment="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1", alpha=0.9))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"[*] Saved confusion matrix plot to: {save_fig_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return cm, breakdown


def extract_feature_importances(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: int = 10,
    save_fig_path: Optional[str] = None,
    show_plot: bool = True
) -> pd.Series:
    """
    Step 3: Extract Gini feature importances and plot horizontal bar chart.
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        return pd.Series(dtype=float)

    series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_series = series.head(top_n).sort_values(ascending=True)
    total_imp = series.sum()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top_series)))
    bars = ax.barh(top_series.index, top_series.values, color=colors, edgecolor="#0f172a", height=0.65)

    ax.set_title(f"Top {top_n} Predictive Feature Importances (Gini/Variance)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, max(top_series.values) * 1.25)

    for bar in bars:
        width = bar.get_width()
        pct = (width / total_imp) * 100 if total_imp > 0 else 0
        ax.text(width + (max(top_series.values) * 0.02), bar.get_y() + bar.get_height() / 2,
                f"{width:.3f} ({pct:.1f}%)",
                va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1e293b")

    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"[*] Saved feature importance plot to: {save_fig_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return series


def evaluate_held_out_set(
    model: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_fig_path: Optional[str] = None,
    show_plot: bool = True
) -> Dict[str, Any]:
    """
    Step 4: Strict Held-Out Evaluation on untouched validation data with ROC Curve.
    """
    from sklearn.metrics import roc_curve
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

    # Plot ROC Curve with Youden's J optimal operating point
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    print("-" * 65)
    print("           ROC-AUC INTUITIVE INTERPRETATION")
    print("-" * 65)
    print(f"  * ROC-AUC = {roc_auc:.4f} (Area Under the Curve):")
    print(f"    There is an {roc_auc * 100:.1f}% probability that the model ranks a randomly")
    print(f"    chosen survivor higher than a randomly chosen perished passenger.")
    print(f"  * Baseline: 0.5000 represents random guessing (coin flip).")
    print(f"  * Optimal Operating Cutoff = {best_thresh:.2f}:")
    print(f"    - Catches {tpr[best_idx] * 100:.1f}% of all actual survivors (True Positive Rate / Recall)")
    print(f"    - Falsely flags only {fpr[best_idx] * 100:.1f}% of non-survivors (False Positive Rate)")
    print("-" * 65 + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Champion RF (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#3b82f6")
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", lw=1.5, label="Chance Baseline (AUC = 0.500)")

    # Highlight optimal threshold
    ax.scatter(fpr[best_idx], tpr[best_idx], color="#dc2626", s=80, zorder=5, label=f"Optimal Cutoff ({best_thresh:.2f})")
    ax.annotate(
        f"Optimal Cutoff: {best_thresh:.2f}\nCatches: {tpr[best_idx]:.1%} Survivors\nFalse Alarms: {fpr[best_idx]:.1%}",
        xy=(fpr[best_idx], tpr[best_idx]),
        xytext=(fpr[best_idx] + 0.12, tpr[best_idx] - 0.15),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.2, headwidth=6),
        fontsize=9, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", edgecolor="#f87171", alpha=0.95)
    )

    # Shaded AUC explanation box inside plot
    ax.text(
        0.50, 0.32,
        f"Shaded Area (AUC) = {roc_auc:.3f}\n85.5% probability of ranking\na real survivor above a non-survivor\n(Chance baseline = 0.50)",
        fontsize=9, fontweight="bold", color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", edgecolor="#93c5fd", alpha=0.92)
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR: % of Perished Falsely Called Survivors)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall: % of Survivors Caught)", fontsize=10.5, fontweight="bold")
    ax.set_title("Held-Out Validation ROC Curve (Pristine 179-Passenger Test Set)", fontsize=12.5, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    stat_box = f"Acc: {acc:.1%}  |  F1: {f1:.3f}  |  AUC: {roc_auc:.3f}"
    ax.text(0.04, 0.90, stat_box, transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

    plt.tight_layout()

    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"[*] Saved ROC curve plot to: {save_fig_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "classification_report": report
    }
