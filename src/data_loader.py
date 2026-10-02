"""
src/data_loader.py
Module for loading, validating, and chronologically ordering power system load data.
"""

import os
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "Date_Time",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "Load_Type",
]


def find_data_file(filepath: Optional[str] = None) -> str:
    """
    Locates the dataset file by checking multiple potential candidate paths.
    
    Parameters
    ----------
    filepath : str, optional
        Custom path provided by caller.
        
    Returns
    -------
    str
        Valid absolute or relative file path to the dataset.
        
    Raises
    ------
    FileNotFoundError
        If load_data.csv cannot be located.
    """
    candidates = []
    if filepath:
        candidates.append(filepath)
    candidates.extend([
        os.path.join("data", "raw", "load_data.csv"),
        os.path.join("data", "load_data.csv"),
        "load_data.csv",
        os.path.join("..", "data", "raw", "load_data.csv"),
        os.path.join("..", "data", "load_data.csv"),
    ])

    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find 'load_data.csv' in candidate locations: {candidates}"
    )


def load_data(filepath: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Loads the raw load dataset, verifies structure, parses date timestamps,
    and sorts the records chronologically.
    
    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. If None, default search paths are checked.
    verbose : bool, default=True
        Whether to print loading summary statistics.
        
    Returns
    -------
    pd.DataFrame
        Cleaned, chronologically sorted DataFrame with parsed datetime.
    """
    resolved_path = find_data_file(filepath)
    if verbose:
        print(f"[DataLoader] Loading raw data from: {resolved_path}")

    # Load CSV
    df = pd.read_csv(resolved_path)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. Found: {list(df.columns)}"
        )

    # Parse Date_Time column
    # Format in dataset: DD-MM-YYYY HH:MM (e.g. 01-01-2018 00:15)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], format="%d-%m-%Y %H:%M", errors="coerce")
    
    if df["Date_Time"].isnull().any():
        null_dates = df["Date_Time"].isnull().sum()
        raise ValueError(f"Failed to parse {null_dates} date entries in 'Date_Time' column.")

    # Chronological sort
    df = df.sort_values(by="Date_Time").reset_index(drop=True)

    if verbose:
        print(f"[DataLoader] Successfully loaded {len(df):,} records.")
        print(f"[DataLoader] Date range: {df['Date_Time'].min()} to {df['Date_Time'].max()}")
        print(f"[DataLoader] Columns: {list(df.columns)}")

    return df


def get_dataset_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a dictionary containing comprehensive data quality metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Loaded power dataset.
        
    Returns
    -------
    dict
        Data quality metrics including row counts, missing values, duplicates, and types.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = ((df.isnull().sum() / total_rows) * 100).round(2).to_dict()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_timestamps = int(df["Date_Time"].duplicated().sum()) if "Date_Time" in df.columns else 0

    target_dist = df["Load_Type"].value_counts().to_dict() if "Load_Type" in df.columns else {}
    target_pct = (df["Load_Type"].value_counts(normalize=True) * 100).round(2).to_dict() if "Load_Type" in df.columns else {}

    report = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "date_min": str(df["Date_Time"].min()) if "Date_Time" in df.columns else None,
        "date_max": str(df["Date_Time"].max()) if "Date_Time" in df.columns else None,
        "target_distribution": target_dist,
        "target_distribution_pct": target_pct,
    }
    return report
