# Data Leakage Prevention Guide

## 1. What is Data Leakage?
Data leakage (or data contamination) occurs when information from outside the training dataset (such as the validation set, test set, or future data) accidentally influences model training or preprocessing. This leads to artificially high validation metrics during development, followed by failure in production or on unseen data.

## 2. Common Causes & Types of Leakage
1. **Data Snooping Bias**: Analyzing or inspecting the test set prior to completing feature engineering and model design choices [26].
2. **Preprocessing & Feature Scaling Leakage**: Computing statistics (e.g., mean, std, median, min/max) on the whole dataset before splitting into train and test sets [35, 48].
3. **Validation & Hyperparameter Tuning Leakage**: Evaluating multiple hyperparameter choices on the test set instead of a separate validation fold or CV setup [26].
4. **Time-Series / Temporal Leakage**: Randomly shuffling chronological data, causing future data to leak into the past.

---

## 3. How to Prevent Data Leakage (5 Golden Rules)

### Rule 1: Split Your Data First
Always perform the train/test split before applying any imputations, scalers, encoders, or feature engineering transformers [26, 48].

```python
from sklearn.model_selection import train_test_split

# Split dataset before ANY preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Rule 2: Strictly Distinguish `fit_transform()` vs `transform()`
- **Training Set**: Call `fit_transform()` to learn statistics (means, standard deviations, medians) and transform training features [38, 39, 48].
- **Test Set**: Call **`transform()` only** to apply the learned statistics from the training set. Never fit scalers/imputers on test data [35, 39, 48].

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Learn mean/std from train set & scale train set
X_train_scaled = scaler.fit_transform(X_train)

# Transform test set using learned train mean/std (NO fitting!)
X_test_scaled = scaler.transform(X_test)
```

### Rule 3: Use Scikit-Learn Pipelines & ColumnTransformers
Wrapping transformations and estimators in a Scikit-Learn `Pipeline` ensures that feature scaling and imputation are computed independently on each training fold during cross-validation, completely isolating validation folds [18, 49, 51].

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("one_hot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

full_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Fit ONLY on X_train
full_pipeline.fit(X_train, y_train)
```

### Rule 4: Hyperparameter Tuning via Cross-Validation
Pass the full `Pipeline` to `GridSearchCV` or `RandomizedSearchCV`. During each fold of cross-validation, Scikit-Learn automatically re-runs `.fit()` strictly on the training folds [18, 49].

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "preprocessing__num__imputer__strategy": ["mean", "median"],
    "classifier__n_estimators": [100, 200]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Rule 5: Respect Chronology for Time-Series
For time-dependent data, avoid random row shuffling [65]. Use `TimeSeriesSplit` so evaluation folds only evaluate on future instances chronologically.

---

## 4. Quick Checklist
- [ ] Split dataset into `train` and `test` before data cleaning.
- [ ] No `fit()` or `fit_transform()` called on test or validation sets.
- [ ] Wrapped feature transformers and model in a Scikit-Learn `Pipeline`.
- [ ] Performed hyperparameter tuning (`GridSearchCV`) using cross-validation on `X_train` only.
- [ ] Evaluated on `X_test` strictly once as the final checkpoint.
