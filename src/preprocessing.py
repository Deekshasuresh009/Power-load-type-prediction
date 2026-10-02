"""
src/preprocessing.py
Module for dataset splitting, missing-value imputation, feature scaling pipelines,
and target encoding with zero data leakage.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Canonical target class mappings
CLASS_NAMES = ["Light_Load", "Medium_Load", "Maximum_Load"]
TARGET_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INT_TO_TARGET = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def split_time_series_by_last_month(
    df: pd.DataFrame,
    date_col: str = "Date_Time",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs a strict chronological train/test split by reserving the entire
    last month of data exclusively for testing and using all prior months for training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing timestamp and features.
    date_col : str, default='Date_Time'
        Timestamp column name.
    verbose : bool, default=True
        Whether to print split statistics.
        
    Returns
    -------
    train_df : pd.DataFrame
        Training records preceding the final month.
    test_df : pd.DataFrame
        Testing records belonging to the final month.
    metadata : dict
        Split metadata including date boundaries, row counts, and class distributions.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in dataframe columns: {list(df.columns)}")

    # Ensure chronological sort
    data = df.sort_values(by=date_col).reset_index(drop=True).copy()

    # Extract monthly periods
    month_periods = data[date_col].dt.to_period("M")
    unique_months = sorted(month_periods.unique())

    if len(unique_months) < 2:
        raise ValueError(
            f"Dataset contains only {len(unique_months)} month(s): {unique_months}. "
            "At least 2 months are required for time-based train/test splitting."
        )

    last_month = unique_months[-1]
    train_months = unique_months[:-1]

    # Perform strict mask split
    train_mask = month_periods < last_month
    test_mask = month_periods == last_month

    train_df = data[train_mask].reset_index(drop=True)
    test_df = data[test_mask].reset_index(drop=True)

    metadata = {
        "all_months": [str(m) for m in unique_months],
        "training_months": [str(m) for m in train_months],
        "test_month": str(last_month),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start_date": str(train_df[date_col].min()),
        "train_end_date": str(train_df[date_col].max()),
        "test_start_date": str(test_df[date_col].min()),
        "test_end_date": str(test_df[date_col].max()),
        "train_target_dist": train_df["Load_Type"].value_counts().to_dict() if "Load_Type" in train_df.columns else {},
        "test_target_dist": test_df["Load_Type"].value_counts().to_dict() if "Load_Type" in test_df.columns else {},
    }

    if verbose:
        print("=" * 60)
        print("TIME-BASED TRAIN / TEST SPLIT SUMMARY (Zero Leakage)")
        print("=" * 60)
        print(f"Total Records: {len(data):,}")
        print(f"Months Found : {[str(m) for m in unique_months]}")
        print(f"\n[TRAIN SET] (Months: {[str(m) for m in train_months]})")
        print(f"  Start Date : {metadata['train_start_date']}")
        print(f"  End Date   : {metadata['train_end_date']}")
        print(f"  Row Count  : {metadata['train_rows']:,} ({len(train_df)/len(data)*100:.2f}%)")
        print(f"  Target Dist:\n{train_df['Load_Type'].value_counts()}")
        print(f"\n[TEST SET] (Final Month: {metadata['test_month']})")
        print(f"  Start Date : {metadata['test_start_date']}")
        print(f"  End Date   : {metadata['test_end_date']}")
        print(f"  Row Count  : {metadata['test_rows']:,} ({len(test_df)/len(data)*100:.2f}%)")
        print(f"  Target Dist:\n{test_df['Load_Type'].value_counts()}")
        print("=" * 60)

    return train_df, test_df, metadata


def build_preprocessor_pipeline(
    feature_columns: List[str],
    with_scaling: bool = True,
    impute_strategy: str = "median"
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that encapsulates numerical
    imputation and optional standard scaling.
    
    Parameters
    ----------
    feature_columns : list of str
        List of all feature column names to be processed.
    with_scaling : bool, default=True
        Whether to include standard scaling in the transformer.
    impute_strategy : str, default='median'
        Imputation strategy for missing values ('median' or 'mean').
        
    Returns
    -------
    ColumnTransformer
        Configured preprocessor ready to be fit exclusively on training data.
    """
    steps = [
        ("imputer", SimpleImputer(strategy=impute_strategy))
    ]
    if with_scaling:
        steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feature_columns)
        ],
        remainder="drop"
    )

    return preprocessor


def encode_target(y: pd.Series) -> np.ndarray:
    """
    Encodes string target labels into numeric integers (0, 1, 2).
    
    Parameters
    ----------
    y : pd.Series or array-like
        Series of Load_Type strings.
        
    Returns
    -------
    np.ndarray
        Array of integer labels.
    """
    return np.array([TARGET_TO_INT[label] for label in y])


def decode_target(y_encoded: np.ndarray) -> np.ndarray:
    """
    Decodes numeric integer labels back to human-readable Load_Type strings.
    
    Parameters
    ----------
    y_encoded : array-like
        Array of integer class labels.
        
    Returns
    -------
    np.ndarray
        Array of Load_Type strings.
    """
    return np.array([INT_TO_TARGET[int(code)] for code in y_encoded])
