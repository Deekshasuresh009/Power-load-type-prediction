"""
src/predict.py
Production-ready inference module for predicting power system load types.
"""

import os
from typing import Union, Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import engineer_features, ALL_FEATURE_COLUMNS
from src.preprocessing import decode_target, CLASS_NAMES


DEFAULT_MODEL_PATH = os.path.join("models", "final_load_type_model.joblib")


def load_model_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> Any:
    """
    Loads the persisted joblib model pipeline.
    
    Parameters
    ----------
    model_path : str, default='models/final_load_type_model.joblib'
        Path to the saved pipeline artifact.
        
    Returns
    -------
    pipeline : scikit-learn Pipeline
        Loaded pipeline.
    """
    # Check alternate fallback paths
    candidates = [
        model_path,
        os.path.join("..", model_path),
        os.path.join(os.path.dirname(__file__), "..", "models", "final_load_type_model.joblib"),
    ]

    for cand in candidates:
        if os.path.exists(cand) and os.path.isfile(cand):
            return joblib.load(cand)

    raise FileNotFoundError(
        f"Model file not found at '{model_path}'. Please train the model first by running `python src/train.py`."
    )


def predict_load_type(
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: str = DEFAULT_MODEL_PATH,
    return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
    """
    Generates Load_Type predictions for new, unseen power system records.
    
    Parameters
    ----------
    input_data : pd.DataFrame, dict, or list of dicts
        Input record(s) containing either raw columns ('Date_Time', 'Usage_kWh', etc.)
        or pre-engineered feature columns.
    model_path : str, default='models/final_load_type_model.joblib'
        Path to the trained pipeline.
    return_proba : bool, default=False
        Whether to return prediction probabilities alongside class labels.
        
    Returns
    -------
    predictions : np.ndarray
        Array of predicted Load_Type string labels ('Light_Load', 'Medium_Load', 'Maximum_Load').
    probabilities : pd.DataFrame, optional
        Class probability DataFrame (if return_proba=True).
    """
    pipeline = load_model_pipeline(model_path)

    # Convert dictionary or list of dictionaries to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input_data type: {type(input_data)}. Expected DataFrame, dict, or list of dicts.")

    # Apply feature engineering if engineered features are missing and Date_Time is present
    missing_feats = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing_feats and "Date_Time" in df.columns:
        df = engineer_features(df, is_training=False)

    # Ensure all required features are available in DataFrame
    available_feats = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    if len(available_feats) < len(ALL_FEATURE_COLUMNS):
        missing = set(ALL_FEATURE_COLUMNS) - set(df.columns)
        raise ValueError(f"Missing required feature columns for prediction: {missing}")

    X = df[ALL_FEATURE_COLUMNS]

    # Predict integer labels and decode
    y_pred_encoded = pipeline.predict(X)
    y_pred_labels = decode_target(y_pred_encoded)

    if return_proba and hasattr(pipeline, "predict_proba"):
        proba_matrix = pipeline.predict_proba(X)
        proba_df = pd.DataFrame(proba_matrix, columns=CLASS_NAMES)
        return y_pred_labels, proba_df

    return y_pred_labels


if __name__ == "__main__":
    # Quick self-test demonstration
    sample_input = {
        "Date_Time": "15-06-2018 14:30",
        "Usage_kWh": 45.2,
        "Lagging_Current_Reactive.Power_kVarh": 18.5,
        "Leading_Current_Reactive_Power_kVarh": 0.0,
        "CO2(tCO2)": 0.02,
        "Lagging_Current_Power_Factor": 92.5,
        "Leading_Current_Power_Factor": 100.0,
        "NSM": 52200,
    }
    print("[predict.py] Inference module loaded.")
