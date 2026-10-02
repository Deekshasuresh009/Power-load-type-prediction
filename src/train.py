"""
src/train.py
End-to-end training pipeline for Power System Load Type Prediction.
Executes data loading, EDA generation, feature engineering, time-aware cross-validation,
hyperparameter tuning, model training, test set evaluation, and artifact saving.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data, get_dataset_quality_report
from src.eda import generate_all_eda_figures
from src.feature_engineering import (
    engineer_features,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.preprocessing import (
    split_time_series_by_last_month,
    encode_target,
    decode_target,
    CLASS_NAMES,
)
from src.models import (
    get_model_catalog,
    evaluate_models_cross_validation,
    tune_hyperparameters,
)
from src.evaluation import (
    evaluate_test_set,
    plot_confusion_matrix_heatmap,
    extract_and_plot_feature_importance,
    perform_error_analysis,
)


def run_training_pipeline(
    data_path: str = None,
    output_dir: str = "outputs",
    model_dir: str = "models",
    generate_eda: bool = True,
    tune_best: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Executes the full end-to-end machine learning pipeline.
    
    Returns
    -------
    dict
        Pipeline execution results including models, metrics, and artifact paths.
    """
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Dataset
    print("\n" + "=" * 75)
    print("STEP 1: DATA LOADING & QUALITY VALIDATION")
    print("=" * 75)
    raw_df = load_data(data_path, verbose=verbose)
    quality_report = get_dataset_quality_report(raw_df)
    with open(os.path.join(output_dir, "metrics", "data_quality_report.json"), "w") as f:
        json.dump(quality_report, f, indent=2)

    # 2. Exploratory Data Analysis
    if generate_eda:
        print("\n" + "=" * 75)
        print("STEP 2: GENERATING EXPLORATORY DATA ANALYSIS (EDA) FIGURES")
        print("=" * 75)
        generate_all_eda_figures(raw_df, output_dir=os.path.join(output_dir, "figures"))

    # 3. Feature Engineering (Row-wise, zero leakage)
    print("\n" + "=" * 75)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 75)
    df_featured = engineer_features(raw_df, is_training=False)
    print(f"[FeatureEngineering] Features created. Total columns: {len(df_featured.columns)}")
    print(f"[FeatureEngineering] Predictor features ({len(ALL_FEATURE_COLUMNS)}): {ALL_FEATURE_COLUMNS}")

    # 4. Strict Time-Based Train/Test Split (Final Month = Test Set)
    print("\n" + "=" * 75)
    print("STEP 4: TIME-BASED TRAIN/TEST SPLIT (Last Month Reserved for Testing)")
    print("=" * 75)
    train_df, test_df, split_meta = split_time_series_by_last_month(df_featured, date_col="Date_Time", verbose=verbose)
    with open(os.path.join(output_dir, "metrics", "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    # Prepare feature matrices and target vectors
    X_train = train_df[ALL_FEATURE_COLUMNS]
    y_train = encode_target(train_df[TARGET_COLUMN])

    X_test = test_df[ALL_FEATURE_COLUMNS]
    y_test = encode_target(test_df[TARGET_COLUMN])

    # 5. Model Catalog & Time-Series Cross-Validation
    print("\n" + "=" * 75)
    print("STEP 5: MODEL SELECTION VIA TIME-SERIES CROSS-VALIDATION (Training Data Only)")
    print("=" * 75)
    catalog = get_model_catalog(ALL_FEATURE_COLUMNS)
    cv_comparison = evaluate_models_cross_validation(catalog, X_train, y_train, n_splits=5, verbose=verbose)
    cv_csv_path = os.path.join(output_dir, "metrics", "model_comparison_cv.csv")
    cv_comparison.to_csv(cv_csv_path, index=False)
    print("\n[Cross-Validation Comparison Table]:")
    print(cv_comparison.to_string(index=False))

    # Identify top performing base model
    best_candidate_name = cv_comparison.iloc[0]["Model"]
    print(f"\n[Model Selection] Top Candidate based on CV F1-Weighted: {best_candidate_name}")

    # 6. Hyperparameter Tuning (Training Data Only)
    final_pipeline = catalog[best_candidate_name]
    best_params = {}
    if tune_best:
        print("\n" + "=" * 75)
        print(f"STEP 6: HYPERPARAMETER TUNING FOR {best_candidate_name.upper()}")
        print("=" * 75)
        tuned_pipeline, best_params, best_cv_score = tune_hyperparameters(
            best_candidate_name,
            catalog[best_candidate_name],
            X_train,
            y_train,
            n_splits=5,
            verbose=verbose
        )
        final_pipeline = tuned_pipeline
        with open(os.path.join(output_dir, "metrics", "best_hyperparameters.json"), "w") as f:
            json.dump({k: str(v) for k, v in best_params.items()}, f, indent=2)

    # 7. Final Model Retraining on Full Training Period (Jan - Nov 2018)
    print("\n" + "=" * 75)
    print("STEP 7: FINAL MODEL TRAINING ON COMPLETE TRAINING SET (Jan - Nov 2018)")
    print("=" * 75)
    final_pipeline.fit(X_train, y_train)
    print(f"[Training] Model successfully fitted on {len(X_train):,} training records.")

    # Save final pipeline and preprocessor
    final_model_path = os.path.join(model_dir, "final_load_type_model.joblib")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    joblib.dump(final_pipeline, final_model_path)
    joblib.dump(final_pipeline.named_steps["preprocessor"], preprocessor_path)
    print(f"[Artifacts] Saved trained model pipeline to: {final_model_path}")
    print(f"[Artifacts] Saved preprocessor artifact to: {preprocessor_path}")

    # 8. Final Test Evaluation (Unseen Test Month: Dec 2018)
    print("\n" + "=" * 75)
    print("STEP 8: FINAL TEST EVALUATION (Unseen Month: Dec 2018)")
    print("=" * 75)
    test_metrics = evaluate_test_set(final_pipeline, X_test, y_test, class_names=CLASS_NAMES, verbose=verbose)

    # Save test metrics table
    test_summary_df = pd.DataFrame([{
        "Model": f"{best_candidate_name} (Final Tuned)",
        "Test_Accuracy": test_metrics["accuracy"],
        "Test_Precision_Weighted": test_metrics["precision_weighted"],
        "Test_Precision_Macro": test_metrics["precision_macro"],
        "Test_Recall_Weighted": test_metrics["recall_weighted"],
        "Test_Recall_Macro": test_metrics["recall_macro"],
        "Test_F1_Weighted": test_metrics["f1_weighted"],
        "Test_F1_Macro": test_metrics["f1_macro"],
    }])
    test_metrics_path = os.path.join(output_dir, "metrics", "final_test_metrics.csv")
    test_summary_df.to_csv(test_metrics_path, index=False)

    # Save consolidated model comparison table
    model_comp_path = os.path.join(output_dir, "metrics", "model_comparison.csv")
    cv_comparison.to_csv(model_comp_path, index=False)

    # 9. Confusion Matrix
    print("\n" + "=" * 75)
    print("STEP 9: GENERATING CONFUSION MATRIX")
    print("=" * 75)
    cm_fig_path = os.path.join(output_dir, "figures", "confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        test_metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=cm_fig_path,
        title=f"Confusion Matrix: {best_candidate_name} on Dec 2018 Test Set"
    )

    # 10. Feature Importance
    print("\n" + "=" * 75)
    print("STEP 10: EXTRACTING FEATURE IMPORTANCE")
    print("=" * 75)
    feat_imp_path = os.path.join(output_dir, "figures", "feature_importance.png")
    feat_imp_df, _ = extract_and_plot_feature_importance(
        final_pipeline,
        feature_names=ALL_FEATURE_COLUMNS,
        top_n=10,
        save_path=feat_imp_path
    )
    feat_imp_df.to_csv(os.path.join(output_dir, "metrics", "feature_importance.csv"), index=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp_df.head(10).to_string(index=False))

    # 11. Error Analysis
    print("\n" + "=" * 75)
    print("STEP 11: ERROR ANALYSIS & TEST PREDICTIONS EXPORT")
    print("=" * 75)
    pred_path = os.path.join(output_dir, "predictions", "test_predictions.csv")
    pred_df, error_df, error_summary = perform_error_analysis(
        test_df,
        y_test,
        test_metrics["y_pred"],
        date_col="Date_Time",
        save_csv=pred_path
    )
    with open(os.path.join(output_dir, "metrics", "error_analysis_summary.json"), "w") as f:
        json.dump(error_summary, f, indent=2)

    print(f"\nTotal Test Samples: {error_summary['total_test_samples']:,}")
    print(f"Total Errors: {error_summary['total_errors']:,} ({error_summary['error_rate_pct']}%)")
    print("Confusion Pairs Breakdown:")
    for pair in error_summary["confusion_pairs"]:
        print(f"  Actual: {pair['Actual_Load_Type']:15s} -> Predicted: {pair['Predicted_Load_Type']:15s} : {pair['Count']} errors")

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

    return {
        "best_model_name": best_candidate_name,
        "best_params": best_params,
        "cv_comparison": cv_comparison,
        "test_metrics": test_metrics,
        "feat_imp_df": feat_imp_df,
        "error_summary": error_summary,
    }


if __name__ == "__main__":
    run_training_pipeline()
