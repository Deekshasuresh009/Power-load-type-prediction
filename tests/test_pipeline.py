"""
tests/test_pipeline.py
Unit and integration tests for data loading, feature engineering,
preprocessing pipeline, time-based splitting, and prediction inference.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data, get_dataset_quality_report, REQUIRED_COLUMNS
from src.feature_engineering import (
    engineer_features,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ENGINEERED_TEMPORAL_FEATURES,
)
from src.preprocessing import (
    split_time_series_by_last_month,
    build_preprocessor_pipeline,
    encode_target,
    decode_target,
    CLASS_NAMES,
)
from src.predict import predict_load_type, load_model_pipeline


@pytest.fixture(scope="session")
def raw_df():
    """Loads dataset once for session."""
    return load_data(verbose=False)


@pytest.fixture(scope="session")
def featured_df(raw_df):
    """Generates features once for session."""
    return engineer_features(raw_df)


def test_data_loading(raw_df):
    """Verify data loading, structure, date parsing, and sorting."""
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) > 0
    for col in REQUIRED_COLUMNS:
        assert col in raw_df.columns, f"Required column {col} missing"

    # Verify datetime parsing
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Date_Time"])

    # Verify chronological ordering
    assert raw_df["Date_Time"].is_monotonic_increasing, "Dataset must be chronologically ordered"


def test_data_quality_report(raw_df):
    """Verify data quality report generation."""
    report = get_dataset_quality_report(raw_df)
    assert "total_rows" in report
    assert "missing_counts" in report
    assert "target_distribution" in report
    assert report["total_rows"] == len(raw_df)


def test_feature_engineering(featured_df):
    """Verify temporal, cyclic, and power-domain feature engineering."""
    for feat in ALL_FEATURE_COLUMNS:
        assert feat in featured_df.columns, f"Engineered feature {feat} missing"

    # Verify cyclic feature bounds
    assert featured_df["hour_sin"].between(-1.0, 1.0).all()
    assert featured_df["hour_cos"].between(-1.0, 1.0).all()
    assert featured_df["day_of_week_sin"].between(-1.0, 1.0).all()
    assert featured_df["day_of_week_cos"].between(-1.0, 1.0).all()
    assert featured_df["month_sin"].between(-1.0, 1.0).all()
    assert featured_df["month_cos"].between(-1.0, 1.0).all()

    # Verify is_weekend binary
    assert set(featured_df["is_weekend"].unique()).issubset({0, 1})


def test_time_based_split(featured_df):
    """Verify time-based train/test splitting on the last month."""
    train_df, test_df, meta = split_time_series_by_last_month(featured_df, verbose=False)

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(featured_df)

    # Verify chronological non-overlap
    max_train_date = train_df["Date_Time"].max()
    min_test_date = test_df["Date_Time"].min()
    assert max_train_date < min_test_date, "Train max date must precede test min date"

    # Verify test set month is the final month
    test_months = test_df["Date_Time"].dt.to_period("M").unique()
    assert len(test_months) == 1, "Test set must contain exactly 1 month"
    assert str(test_months[0]) == meta["test_month"]


def test_preprocessor_pipeline_and_no_leakage(featured_df):
    """Verify preprocessor fits only on train without leakage."""
    train_df, test_df, _ = split_time_series_by_last_month(featured_df, verbose=False)
    X_train = train_df[ALL_FEATURE_COLUMNS]
    X_test = test_df[ALL_FEATURE_COLUMNS]

    preprocessor = build_preprocessor_pipeline(ALL_FEATURE_COLUMNS, with_scaling=True)
    
    # Fit only on train
    X_train_trans = preprocessor.fit_transform(X_train)
    assert not np.isnan(X_train_trans).any(), "Transformed train data must contain no NaNs"

    # Transform test
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_test_trans).any(), "Transformed test data must contain no NaNs"


def test_target_encoding():
    """Verify target encoding and roundtrip decoding."""
    series = pd.Series(["Light_Load", "Medium_Load", "Maximum_Load", "Light_Load"])
    encoded = encode_target(series)
    assert (encoded == np.array([0, 1, 2, 0])).all()

    decoded = decode_target(encoded)
    assert (decoded == series.values).all()


def test_end_to_end_predict_pipeline(featured_df):
    """Verify prediction pipeline runs on sample inputs."""
    train_df, test_df, _ = split_time_series_by_last_month(featured_df, verbose=False)
    X_train = train_df[ALL_FEATURE_COLUMNS]
    y_train = encode_target(train_df[TARGET_COLUMN])

    # Fit a lightweight pipeline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor_pipeline(ALL_FEATURE_COLUMNS, with_scaling=False)),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Save to models directory temporarily if not yet trained
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "final_load_type_model.joblib")
    if not os.path.exists(model_path):
        import joblib
        joblib.dump(pipeline, model_path)

    # Test single dictionary input with raw features
    sample_raw = {
        "Date_Time": "15-06-2018 14:30",
        "Usage_kWh": 45.2,
        "Lagging_Current_Reactive.Power_kVarh": 18.5,
        "Leading_Current_Reactive_Power_kVarh": 0.0,
        "CO2(tCO2)": 0.02,
        "Lagging_Current_Power_Factor": 92.5,
        "Leading_Current_Power_Factor": 100.0,
        "NSM": 52200,
    }
    pred = predict_load_type(sample_raw)
    assert len(pred) == 1
    assert pred[0] in CLASS_NAMES

    # Test batch DataFrame input
    sample_batch = test_df.head(5)
    pred_batch = predict_load_type(sample_batch)
    assert len(pred_batch) == 5
    assert all(p in CLASS_NAMES for p in pred_batch)
