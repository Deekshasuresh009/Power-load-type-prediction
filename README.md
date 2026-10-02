# Power System Load Type Prediction

Machine learning project developed for the USEReady AI/ML Internship Assignment.

The objective is to predict the operational load type of an industrial power system using historical 15-minute smart meter readings.

## Objective

Predict the `Load_Type` for each observation using a multi-class classification approach.

The target variable contains three classes:

- `Light_Load`
- `Medium_Load`
- `Maximum_Load`

## Dataset

The dataset contains 35,041 observations recorded at 15-minute intervals throughout 2018.

### Input Features

| Feature | Description |
|---|---|
| `Date_Time` | Timestamp of the reading |
| `Usage_kWh` | Active energy consumption |
| `Lagging_Current_Reactive.Power_kVarh` | Reactive power under lagging current |
| `Leading_Current_Reactive_Power_kVarh` | Reactive power under leading current |
| `CO2(tCO2)` | CO2 emissions |
| `Lagging_Current_Power_Factor` | Power factor for lagging current |
| `Leading_Current_Power_Factor` | Power factor for leading current |
| `NSM` | Number of seconds from midnight |
| `Load_Type` | Target variable |

## Methodology

The project follows a chronological machine learning workflow:

1. **Data Loading and Validation**
   - Load and validate the raw data.
   - Parse timestamps and sort records chronologically.
   - Check data types, missing values, and duplicates.

2. **Data Preprocessing**
   - Handle missing values using median imputation.
   - Fit preprocessing steps only on training data to avoid data leakage.

3. **Exploratory Data Analysis**
   - Analyze target class distribution.
   - Examine feature distributions and correlations.
   - Study hourly, daily, and weekly consumption patterns.

4. **Feature Engineering**
   - Extract time-based features such as hour, minute, day, month, day of week, day of year, and weekend indicator.
   - Create cyclic features using sine and cosine transformations.
   - Create electrical features including total reactive power, approximate apparent power, and reactive-to-active power ratio.

5. **Time-Based Train/Test Split**
   - January–November 2018: training data.
   - December 2018: final unseen test data.

6. **Model Training and Comparison**
   - Train multiple classification algorithms.
   - Compare their performance using 5-fold TimeSeriesSplit cross-validation on the training data.

7. **Hyperparameter Tuning**
   - Tune the selected model using `GridSearchCV`.
   - Perform hyperparameter tuning only on the training period.

8. **Final Evaluation**
   - Evaluate the final model on the unseen December 2018 test set.
   - Calculate accuracy, precision, recall, and F1-score.
   - Generate confusion matrix, feature importance, and error analysis.

## Train/Test Strategy

Since the data is time-dependent, a chronological split is used instead of random train-test splitting.

| Period | Records | Usage |
|---|---:|---|
| January–November 2018 | 32,064 | Training |
| December 2018 | 2,977 | Final Testing |

December 2018 is kept completely unseen during model training and hyperparameter tuning.

This provides a more realistic evaluation of how the model performs on future observations.

## Models

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

## Model Comparison

The models were evaluated using 5-fold `TimeSeriesSplit` cross-validation on the training data.

| Model | CV Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 0.8622 ± 0.1126 | 0.8562 ± 0.1239 | 0.8336 |
| XGBoost | 0.8598 ± 0.1050 | 0.8546 ± 0.1153 | 0.8322 |
| Gradient Boosting | 0.8568 ± 0.1048 | 0.8514 ± 0.1154 | 0.8290 |
| Decision Tree | 0.8358 ± 0.0978 | 0.8317 ± 0.1065 | 0.8060 |
| Logistic Regression | 0.7222 ± 0.1021 | 0.6895 ± 0.1413 | 0.6524 |

## Final Results

The final Random Forest model was evaluated on the completely unseen December 2018 test set.

| Metric | Score |
|---|---:|
| Accuracy | 91.97% |
| Weighted Precision | 92.56% |
| Weighted Recall | 91.97% |
| Weighted F1 | 91.97% |
| Macro Precision | 91.09% |
| Macro Recall | 89.40% |
| Macro F1 | 89.75% |

### Per-Class Performance

| Load Type | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Light_Load | 96.11% | 94.84% | 95.47% | 1,745 |
| Medium_Load | 81.33% | 95.31% | 87.77% | 704 |
| Maximum_Load | 95.81% | 78.03% | 86.01% | 528 |

## Key Findings

- The final model achieved 91.97% accuracy on the unseen December 2018 test set.
- Time-based and electrical features were included to capture temporal and electrical patterns associated with different load conditions.
- Random Forest provided the strongest cross-validation performance among the evaluated models.
- The model achieved high recall for Light_Load and Medium_Load, while Maximum_Load had comparatively lower recall.
