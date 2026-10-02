"""
src/evaluation.py
Module for comprehensive model evaluation, metric computation, confusion matrix visualization,
feature importance extraction, and error analysis.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import CLASS_NAMES, decode_target


def evaluate_test_set(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluates a trained model pipeline on the unseen test set (final month).
    
    Parameters
    ----------
    pipeline : scikit-learn Pipeline
        Trained model pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : np.ndarray
        True integer class labels.
    class_names : list of str
        Class label names.
    verbose : bool, default=True
        Whether to print evaluation summary.
        
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics and predictions.
    """
    y_pred = pipeline.predict(X_test)

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_pred_labels": decode_target(y_pred),
    }

    if verbose:
        print("=" * 65)
        print("FINAL TEST SET EVALUATION (Unseen Final Month: Dec 2018)")
        print("=" * 65)
        print(f"Accuracy           : {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision (Weighted): {prec_weighted:.4f}")
        print(f"Precision (Macro)  : {prec_macro:.4f}")
        print(f"Recall (Weighted)   : {rec_weighted:.4f}")
        print(f"Recall (Macro)     : {rec_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"F1-Score (Macro)   : {f1_macro:.4f}")
        print("\nClassification Report:\n")
        print(report_str)
        print("=" * 65)

    return results


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = "outputs/figures/confusion_matrix.png",
    title: str = "Confusion Matrix - Final Month (Dec 2018) Evaluation"
) -> plt.Figure:
    """Plots and saves a labeled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate percentage annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0) * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor="white",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Load Type", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Load Type", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Evaluation] Saved confusion matrix to {save_path}")

    return fig


def extract_and_plot_feature_importance(
    pipeline: Any,
    feature_names: List[str],
    top_n: int = 10,
    save_path: Optional[str] = "outputs/figures/feature_importance.png"
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Extracts feature importances from tree-based pipelines and creates a bar chart.
    
    Parameters
    ----------
    pipeline : scikit-learn Pipeline
        Fitted model pipeline.
    feature_names : list of str
        Predictor column names.
    top_n : int, default=10
        Number of top features to plot.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    importance_df : pd.DataFrame
        Sorted table of feature importances.
    fig : plt.Figure
        Bar chart figure.
    """
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        # Fallback for linear models: use mean absolute coefficient
        importances = np.mean(np.abs(clf.coef_), axis=0)

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Top N plot
    df_top = df_imp.head(top_n).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_top["Feature"], df_top["Importance"], color="#2980b9", edgecolor="none", height=0.65)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, color="#2c3e50")

    ax.set_title(f"Top {top_n} Most Important Features (Random Forest / Tree-based)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_ylabel("Feature Name", fontsize=11)
    ax.set_xlim(0, max(df_top["Importance"]) * 1.15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Evaluation] Saved feature importance plot to {save_path}")

    return df_imp, fig


def perform_error_analysis(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    date_col: str = "Date_Time",
    save_csv: Optional[str] = "outputs/predictions/test_predictions.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes test set prediction errors, identifies confusion patterns, and exports predictions.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test DataFrame containing metadata and timestamps.
    y_test : np.ndarray
        True encoded target.
    y_pred : np.ndarray
        Predicted encoded target.
    date_col : str, default='Date_Time'
        Timestamp column name.
    save_csv : str, optional
        Path to save test predictions.
        
    Returns
    -------
    predictions_df : pd.DataFrame
        Complete test prediction records.
    errors_df : pd.DataFrame
        Subset containing misclassified records.
    summary : dict
        Summary statistics of errors.
    """
    pred_df = test_df.copy()
    pred_df["Actual_Load_Type"] = decode_target(y_test)
    pred_df["Predicted_Load_Type"] = decode_target(y_pred)
    pred_df["Is_Correct"] = (pred_df["Actual_Load_Type"] == pred_df["Predicted_Load_Type"])

    errors_df = pred_df[~pred_df["Is_Correct"]].copy()

    total_test = len(pred_df)
    total_errors = len(errors_df)
    error_rate = (total_errors / total_test) * 100

    # Class-wise confusion pairs
    confusion_pairs = errors_df.groupby(["Actual_Load_Type", "Predicted_Load_Type"]).size().reset_index(name="Count")
    confusion_pairs = confusion_pairs.sort_values(by="Count", ascending=False)

    summary = {
        "total_test_samples": total_test,
        "total_errors": total_errors,
        "error_rate_pct": round(error_rate, 2),
        "accuracy_pct": round(100 - error_rate, 2),
        "confusion_pairs": confusion_pairs.to_dict(orient="records"),
    }

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        export_cols = [date_col, "Actual_Load_Type", "Predicted_Load_Type", "Is_Correct"]
        other_cols = [c for c in pred_df.columns if c not in export_cols]
        final_export = pred_df[export_cols + other_cols]
        final_export.to_csv(save_csv, index=False)
        print(f"[Evaluation] Saved test predictions to {save_csv}")

    return pred_df, errors_df, summary
