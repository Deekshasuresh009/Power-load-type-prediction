"""
src/feature_engineering.py
Module for engineering time-based, cyclic, and power-domain features without data leakage.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


# Base numerical features present in the raw dataset
RAW_NUMERICAL_FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]

# Engineered temporal and cyclic features
ENGINEERED_TEMPORAL_FEATURES = [
    "hour",
    "minute",
    "day",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

# Domain-specific power features
DOMAIN_POWER_FEATURES = [
    "Total_Reactive_Power_kVarh",
    "Apparent_Power_Approx_kVA",
    "Reactive_to_Active_Ratio",
]

ALL_FEATURE_COLUMNS = (
    RAW_NUMERICAL_FEATURES
    + ENGINEERED_TEMPORAL_FEATURES
    + DOMAIN_POWER_FEATURES
)

TARGET_COLUMN = "Load_Type"


def engineer_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Constructs temporal, cyclic, and power-domain interaction features
    from raw dataset columns in a purely deterministic, row-wise manner.
    
    Guarantees:
    - No future information leakage.
    - No target leakage (target column is never used to construct features).
    - Fully reproducible transformation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'Date_Time' and raw power features.
    is_training : bool, default=False
        Flag indicating if the dataset is training data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame augmented with engineered features.
    """
    data = df.copy()

    # Ensure Date_Time is parsed
    if not pd.api.types.is_datetime64_any_dtype(data["Date_Time"]):
        data["Date_Time"] = pd.to_datetime(data["Date_Time"], format="%d-%m-%Y %H:%M", errors="coerce")

    # 1. Temporal calendar features
    dt = data["Date_Time"].dt
    data["hour"] = dt.hour
    data["minute"] = dt.minute
    data["day"] = dt.day
    data["month"] = dt.month
    data["day_of_week"] = dt.dayofweek  # 0=Monday, 6=Sunday
    data["day_of_year"] = dt.dayofyear
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Exact decimal hour (e.g. 14:30 -> 14.5)
    time_decimal = data["hour"] + (data["minute"] / 60.0)

    # 2. Cyclic representations for periodic variables
    # 24-hour cycle
    data["hour_sin"] = np.sin(2 * np.pi * time_decimal / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * time_decimal / 24.0)

    # 7-day weekly cycle
    data["day_of_week_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["day_of_week_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    # 12-month annual cycle (0-indexed month)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)

    # 3. Domain-specific power interactions
    # Active power approx (kWh in 15 min -> 4 * kWh = kW)
    lag_q = data["Lagging_Current_Reactive.Power_kVarh"]
    lead_q = data["Leading_Current_Reactive_Power_kVarh"]
    p_kwh = data["Usage_kWh"]

    data["Total_Reactive_Power_kVarh"] = lag_q + lead_q
    
    # Apparent Energy ~ sqrt(P^2 + Q^2)
    # Using numpy safe sqrt
    data["Apparent_Power_Approx_kVA"] = np.sqrt(
        np.maximum(0, p_kwh ** 2 + data["Total_Reactive_Power_kVarh"] ** 2)
    )
    
    # Reactive to active ratio (with small epsilon to avoid div by zero)
    data["Reactive_to_Active_Ratio"] = data["Total_Reactive_Power_kVarh"] / (p_kwh + 1e-4)

    return data


def get_feature_and_target_columns() -> Tuple[List[str], str]:
    """
    Returns the list of final predictor feature column names and the target column name.
    
    Returns
    -------
    tuple of (list of str, str)
        (feature_column_names, target_column_name)
    """
    return ALL_FEATURE_COLUMNS, TARGET_COLUMN
