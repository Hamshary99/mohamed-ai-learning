# Week 1 Deliverable: Production Classical ML Pipeline

This folder contains the production-grade modular codebase for the **Titanic Survival Prediction Pipeline**, structured strictly around **Clean Code** and **SOLID Principles** (specifically SRP & OCP) as outlined in `StudentGator_Learning_Plan_Mohamed.pdf` (Week 1).

---

## Architecture & SOLID Principles

```
week1/deliverable/
├── main.py                  # Pipeline Orchestrator (Coordinates the full workflow)
├── src/
│   ├── __init__.py          # Package initialization
│   ├── preprocessing.py     # SRP: Ingestion, cleaning, imputation, feature engineering, splitting
│   ├── train.py             # SRP & OCP: Model registry, Cross-Validation Arena, GridSearchCV tuning
│   └── evaluate.py          # SRP: Leaderboard, Overfit Gap (Δ), Confusion Matrix, Held-out tests
└── notebook.ipynb           # Exploratory & scratch notebook
```

### 1. Single Responsibility Principle (SRP)
- **`src/preprocessing.py`**: ONLY handles data transformations. It knows nothing about machine learning models or evaluation metrics.
- **`src/train.py`**: ONLY handles model construction, cross-validation, hyperparameter tuning, and model persistence. It knows nothing about data cleaning or plotting.
- **`src/evaluate.py`**: ONLY handles metric computation, leaderboard generation, confusion matrix analysis, and diagnostic reporting.
- **`main.py`**: Serves as the conductor/orchestrator. It connects the modules without implementing their internal algorithms.

### 2. Open/Closed Principle (OCP)
- **`src/train.py` (`get_model_registry`)**: The model registry is open for extension but closed for modification. If you want to benchmark a new algorithm (e.g. `LightGBMClassifier` or `CatBoostClassifier`), you only add one line to the registry. The downstream benchmarking arena, cross-validation loops, leaderboard generation, and tuning functions will automatically work without editing a single line of their code.

---

## Module Map & Implementation Steps

| Module | Function | Responsibilities | Notebook Reference |
| :--- | :--- | :--- | :--- |
| **`preprocessing.py`** | `load_raw_data` | Ingest raw train & test CSVs | Section 1 |
| | `drop_uninformative_columns` | Drop `PassengerId`, `Ticket`, `Cabin` | Section 2 |
| | `impute_missing_values` | Median for continuous, Mode for categorical | Section 2 |
| | `engineer_domain_features` | Extract `Title`, `FamilySize`, `IsAlone` | Section 3 |
| | `encode_categorical_features` | One-Hot / Binary encode categorical columns | Section 3 |
| | `perform_stratified_split` | 80/20 train/val split preserving class ratios | Section 4 |
| | `run_full_preprocessing_pipeline` | Chains preprocessing functions | Section 1-4 |
| **`train.py`** | `get_model_registry` | Instantiates Dummy, LogReg, DT, RF, XGB | Section 5 |
| | `run_model_arena` | 5-Fold Stratified CV with train scores | Section 5 |
| | `get_hyperparameter_grids` | Defines parameter search spaces per model | Section 7 |
| | `tune_candidate_models` | Multi-model parallel `GridSearchCV` optimization | Section 7 |
| | `tune_champion_model` | Single champion `GridSearchCV` optimization | Section 7 |
| | `save_model_artifact` | Serializes model via `joblib.dump` | Section 8 |
| **`evaluate.py`** | `build_leaderboard` | Calculates Overfit Gap $\Delta = \text{Train} - \text{Val}$ | Section 5 |
| | `compare_tuned_models` | Compares best outcome parameters & $\Delta$ gain | Section 7 |
| | `evaluate_confusion_matrix` | Out-of-fold `cross_val_predict` & Type I/II errors | Section 6 |
| | `extract_feature_importances` | Gini importances from tree-based model | Section 6 |
| | `evaluate_held_out_set` | Strict held-out evaluation on $X_{val}$ (179 rows) | Section 8 |
| **`main.py`** | `main` | Runs the full pipeline end-to-end | Section 1-8 |

