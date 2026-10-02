"""
src/models.py
Model catalog, time-series cross-validation, and hyperparameter tuning module.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.preprocessing import build_preprocessor_pipeline

# Check optional XGBoost availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_model_catalog(feature_columns: list) -> Dict[str, Pipeline]:
    """
    Constructs a catalog of classification models wrapped in self-contained
    scikit-learn Pipeline objects with appropriate imputation and scaling.
    
    Parameters
    ----------
    feature_columns : list of str
        List of predictor feature names.
        
    Returns
    -------
    dict of {str: Pipeline}
        Dictionary mapping model names to configured Pipeline instances.
    """
    catalog = {}

    # 1. Logistic Regression (Requires standard scaling)
    lr_preprocessor = build_preprocessor_pipeline(feature_columns, with_scaling=True)
    catalog["Logistic Regression"] = Pipeline([
        ("preprocessor", lr_preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            random_state=42,
            solver="lbfgs"
        )),
    ])

    # 2. Decision Tree Classifier (Tree-based, scaling optional, imputation included)
    tree_preprocessor = build_preprocessor_pipeline(feature_columns, with_scaling=False)
    catalog["Decision Tree"] = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )),
    ])

    # 3. Random Forest Classifier
    rf_preprocessor = build_preprocessor_pipeline(feature_columns, with_scaling=False)
    catalog["Random Forest"] = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
    ])

    # 4. Gradient Boosting Classifier
    gb_preprocessor = build_preprocessor_pipeline(feature_columns, with_scaling=False)
    catalog["Gradient Boosting"] = Pipeline([
        ("preprocessor", gb_preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )),
    ])

    # 5. XGBoost Classifier (if installed)
    if XGBOOST_AVAILABLE:
        xgb_preprocessor = build_preprocessor_pipeline(feature_columns, with_scaling=False)
        catalog["XGBoost"] = Pipeline([
            ("preprocessor", xgb_preprocessor),
            ("classifier", xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1
            )),
        ])

    return catalog


def evaluate_models_cross_validation(
    models_dict: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluates each candidate model using TimeSeriesSplit on training data.
    
    Parameters
    ----------
    models_dict : dict of {str: Pipeline}
        Dictionary of models to evaluate.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : np.ndarray
        Encoded training target array.
    n_splits : int, default=5
        Number of time-series splits.
    verbose : bool, default=True
        Whether to log progress.
        
    Returns
    -------
    pd.DataFrame
        Table summarizing CV performance across all candidate algorithms.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
    }

    results = []
    if verbose:
        print("=" * 70)
        print(f"RUNNING TIME-SERIES CROSS-VALIDATION ({n_splits} Splits on Train Data)")
        print("=" * 70)

    for name, pipeline in models_dict.items():
        if verbose:
            print(f"[Cross-Validation] Evaluating: {name}...")

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        acc_mean = scores["test_accuracy"].mean()
        acc_std = scores["test_accuracy"].std()
        prec_mean = scores["test_precision_weighted"].mean()
        rec_mean = scores["test_recall_weighted"].mean()
        f1_w_mean = scores["test_f1_weighted"].mean()
        f1_w_std = scores["test_f1_weighted"].std()
        f1_m_mean = scores["test_f1_macro"].mean()

        results.append({
            "Model": name,
            "CV_Accuracy_Mean": round(acc_mean, 4),
            "CV_Accuracy_Std": round(acc_std, 4),
            "CV_Precision_Weighted": round(prec_mean, 4),
            "CV_Recall_Weighted": round(rec_mean, 4),
            "CV_F1_Weighted_Mean": round(f1_w_mean, 4),
            "CV_F1_Weighted_Std": round(f1_w_std, 4),
            "CV_F1_Macro_Mean": round(f1_m_mean, 4),
        })

        if verbose:
            print(f"  -> Accuracy: {acc_mean:.4f} (+/- {acc_std:.4f}) | F1 (Weighted): {f1_w_mean:.4f} (+/- {f1_w_std:.4f})")

    results_df = pd.DataFrame(results).sort_values(by="CV_F1_Weighted_Mean", ascending=False).reset_index(drop=True)
    return results_df


def tune_hyperparameters(
    model_name: str,
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    n_splits: int = 5,
    verbose: bool = True
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Conducts GridSearchCV using TimeSeriesSplit on training data only.
    
    Parameters
    ----------
    model_name : str
        Name of candidate model.
    base_pipeline : Pipeline
        Candidate pipeline instance.
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Encoded training target.
    param_grid : dict, optional
        Hyperparameter grid. If None, uses default tailored grid.
    n_splits : int, default=5
        Number of time-series cross-validation folds.
    verbose : bool, default=True
        Whether to print search details.
        
    Returns
    -------
    best_pipeline : Pipeline
        Fitted best pipeline.
    best_params : dict
        Selected best hyperparameters.
    best_score : float
        Best cross-validation F1-weighted score.
    """
    if param_grid is None:
        if "Random Forest" in model_name:
            param_grid = {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [12, 18, 25],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
            }
        elif "XGBoost" in model_name:
            param_grid = {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [4, 6, 8],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
            }
        elif "Gradient Boosting" in model_name:
            param_grid = {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [4, 6],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
            }
        else:
            param_grid = {
                "classifier__C": [0.1, 1.0, 10.0],
            }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if verbose:
        print("=" * 70)
        print(f"HYPERPARAMETER TUNING FOR: {model_name}")
        print(f"Parameter Grid: {param_grid}")
        print("=" * 70)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1 if verbose else 0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    if verbose:
        print(f"[Tuning] Best F1_Weighted Score (CV): {grid_search.best_score_:.4f}")
        print(f"[Tuning] Best Hyperparameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
