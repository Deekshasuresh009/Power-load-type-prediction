# Power System Load Type Prediction

This repository contains my machine learning project for the USEReady AI/ML internship assignment. The goal is to predict the operational load type of an industrial power system using 15-minute smart meter readings.

---

## Overview

Industrial facilities and electrical grids experience changing electricity demands throughout the day. Categorizing power usage into distinct load regimes helps grid operators manage peak demand, schedule equipment maintenance, and optimize electricity costs.

This project implements an end-to-end classification pipeline that predicts the load category from power telemetry features and time-based indicators.

---

## Objective

Develop and evaluate a multi-class classification model to predict the target variable `Load_Type`, which consists of three classes:

1. `Light_Load`
2. `Medium_Load`
3. `Maximum_Load`

---

## Dataset

The provided dataset (`load_data.csv`) contains 35,041 observations recorded at 15-minute intervals across the entire calendar year 2018 (January 1, 2018 to December 31, 2018).

Key features in the dataset:
- `Date_Time`: 15-minute timestamp
- `Usage_kWh`: Active energy consumption per 15-minute window
- `Lagging_Current_Reactive.Power_kVarh`: Reactive power under inductive loads
- `Leading_Current_Reactive_Power_kVarh`: Reactive power under capacitive loads
- `CO2(tCO2)`: Greenhouse gas emissions (metric tons)
- `Lagging_Current_Power_Factor`: Power factor percentage during lagging current
- `Leading_Current_Power_Factor`: Power factor percentage during leading current
- `NSM`: Number of Seconds from Midnight (0 to 86,400)
- `Load_Type` (Target): `Light_Load`, `Medium_Load`, `Maximum_Load`

---

## Approach

The implementation follows a structured machine learning workflow:

1. **Data Preprocessing:**
   - Parsed timestamps into standard datetime format and sorted chronologically.
   - Performed data quality analysis (checked missing values, data types, and duplicates).
   - Applied median imputation within scikit-learn pipelines fitted strictly on training data to prevent data leakage.

2. **Exploratory Data Analysis (EDA):**
   - Analyzed target class distribution and class balance.
   - Evaluated feature distributions and correlation heatmaps.
   - Investigated daily, weekly, and hourly consumption patterns and power factor relationships.

3. **Feature Engineering:**
   - Extracted calendar features: `hour`, `minute`, `day`, `month`, `day_of_week`, `day_of_year`, and `is_weekend`.
   - Created sine and cosine cyclic transformations for periodic time variables (`hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`) to preserve continuity across 23:45 and 00:00.
   - Derived electrical domain features: `Total_Reactive_Power_kVarh`, `Apparent_Power_Approx_kVA`, and `Reactive_to_Active_Ratio`.

4. **Time-Based Train/Test Split:**
   - Reserving the final month (December 2018) as the test set.
   - Training on all prior months (January to November 2018).

5. **Model Comparison & Validation:**
   - Evaluated candidate algorithms using 5-fold `TimeSeriesSplit` cross-validation on the training set.
   - Used weighted F1-score as the primary metric to account for class distribution.

6. **Hyperparameter Tuning:**
   - Tuned the top candidate model using `GridSearchCV` with `TimeSeriesSplit` on training data only.

7. **Final Evaluation:**
   - Evaluated the final trained model on the unseen December 2018 test set.
   - Computed classification metrics, confusion matrix, feature importance, and error analysis.

---

## Test Strategy

As required by the assignment, the final evaluation uses a strict chronological split:

- **Training Set:** January 1, 2018 to November 30, 2018 (32,064 records, ~91.5% of data)
- **Test Set:** December 1, 2018 to December 31, 2018 (2,977 records, ~8.5% of data)

### Why the last month is used as the test set
In power systems, machine learning models are deployed to forecast future load states based on historical operating data. Standard random shuffling (`train_test_split`) causes temporal data leakage by allowing the model to train on future timestamps and predict past ones. Using the complete final month as an unseen test set provides a realistic assessment of how the model generalizes to future operational conditions.

---

## Models

Five classification algorithms were trained and compared:

1. **Logistic Regression** (with standard scaling)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**
5. **XGBoost Classifier**

---

## Evaluation Metrics

To evaluate multi-class performance with class imbalance, the following metrics were used:

- **Accuracy:** Overall proportion of correct predictions
- **Precision (Weighted & Macro):** Accuracy of positive predictions per class
- **Recall (Weighted & Macro):** Proportion of actual class instances correctly identified
- **F1-Score (Weighted & Macro):** Harmonic mean of precision and recall

---

## Results

### Cross-Validation Comparison (5-Fold TimeSeriesSplit on Training Set)

| Model | CV Accuracy (Mean ± Std) | CV F1-Score (Weighted) | CV F1-Score (Macro) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **0.8622 ± 0.1126** | **0.8562 ± 0.1239** | **0.8336** |
| **XGBoost** | 0.8598 ± 0.1050 | 0.8546 ± 0.1153 | 0.8322 |
| **Gradient Boosting** | 0.8568 ± 0.1048 | 0.8514 ± 0.1154 | 0.8290 |
| **Decision Tree** | 0.8358 ± 0.0978 | 0.8317 ± 0.1065 | 0.8060 |
| **Logistic Regression** | 0.7222 ± 0.1021 | 0.6895 ± 0.1413 | 0.6524 |

### Final Test Evaluation (Unseen Month: December 2018)

Random Forest was selected as the final model and tuned via `GridSearchCV` (`max_depth=12`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=100`). After fitting on the full training period (January–November), it was evaluated on the December test set:

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.9197** | **91.97%** |
| **Weighted Precision** | **0.9256** | **92.56%** |
| **Weighted Recall** | **0.9197** | **91.97%** |
| **Weighted F1-Score** | **0.9197** | **91.97%** |
| **Macro Precision** | **0.9109** | **91.09%** |
| **Macro Recall** | **0.8940** | **89.40%** |
| **Macro F1-Score** | **0.8975** | **89.75%** |

### Per-Class Performance on Test Set

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Light_Load` | 0.9611 | 0.9484 | 0.9547 | 1,745 |
| `Medium_Load` | 0.8133 | 0.9531 | 0.8777 | 704 |
| `Maximum_Load` | 0.9581 | 0.7803 | 0.8601 | 528 |

---

## Project Structure

```
power-load-type-prediction/
│
├── data/
│   └── raw/
│       └── load_data.csv                    # Dataset (35,041 records)
│
├── notebooks/
│   └── Power_Load_Type_Prediction.ipynb     # Main Jupyter notebook with executed outputs
│
├── src/
│   ├── __init__.py                          # Package initializer
│   ├── data_loader.py                       # Data loading and validation
│   ├── preprocessing.py                     # Split logic and preprocessing pipelines
│   ├── feature_engineering.py               # Time, cyclic, and power features
│   ├── eda.py                               # Reusable visualization functions
│   ├── models.py                            # Model catalog, cross-validation, tuning
│   ├── evaluation.py                        # Metric computation and evaluation plots
│   ├── train.py                             # End-to-end training pipeline
│   └── predict.py                           # Inference function
│
├── models/
│   ├── final_load_type_model.joblib         # Saved final trained model pipeline
│   └── preprocessor.joblib                  # Saved preprocessor
│
├── outputs/
│   ├── figures/                             # Saved EDA and evaluation charts
│   ├── metrics/                             # Saved comparison tables and reports
│   └── predictions/
│       └── test_predictions.csv             # December test set predictions
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                     # Pytest suite (7 tests)
│
├── README.md                                # Project documentation
├── requirements.txt                         # Required dependencies
├── .gitignore                               # Git ignore rules
└── LICENSE                                  # MIT License
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/power-load-type-prediction.git
cd power-load-type-prediction
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Project

To run the complete training pipeline and generate all outputs:
```bash
python src/train.py
```

To run prediction inference on new sample data:
```python
from src.predict import predict_load_type

sample = {
    "Date_Time": "15-12-2018 14:30",
    "Usage_kWh": 68.4,
    "Lagging_Current_Reactive.Power_kVarh": 28.5,
    "Leading_Current_Reactive_Power_kVarh": 0.0,
    "CO2(tCO2)": 0.03,
    "Lagging_Current_Power_Factor": 92.4,
    "Leading_Current_Power_Factor": 100.0,
    "NSM": 52200,
}

prediction = predict_load_type(sample)
print("Predicted Load Type:", prediction[0])
```

---

## Running Tests

Run the test suite using pytest:
```bash
pytest tests/test_pipeline.py -v
```

All 7 unit and integration tests cover data loading, quality reporting, feature engineering, time-based splitting, preprocessor fitting, target encoding, and inference.

---

## Notebook

The primary assessment notebook is located at:
```
notebooks/Power_Load_Type_Prediction.ipynb
```

The notebook is fully executed and organized into 27 structured sections with Markdown commentary, code cells, data tables, and inline plots.

---

## Future Improvements

1. **Lagged Power Features:** Add rolling and lagged features ($t-1$, $t-4$, $t-96$) to capture short-term memory in load transitions.
2. **Deep Learning:** Test sequential models such as LSTM or Temporal Convolutional Networks (TCN).
3. **API Deployment:** Wrap the inference function in a lightweight REST API using FastAPI.
4. **Drift Monitoring:** Implement data drift detection to monitor power consumption shifts over time.

---

## License

This project is released under the [MIT License](LICENSE).
