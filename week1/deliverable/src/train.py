"""
Module: train.py
Single Responsibility Principle (SRP):
Model construction, model registry (OCP), cross-validation benchmarking,
hyperparameter tuning, and model artifact persistence.
"""

from typing import Dict, Any, Tuple
import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from xgboost import XGBClassifier


def get_model_registry(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Step 1: Open/Closed Principle (OCP) model registry.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))
    decision_tree = DecisionTreeClassifier(random_state=random_state)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state)
    xgb = XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss", n_jobs=1)

    return {
        "Dummy": dummy,
        "Logistic Regression": logistic,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "XGBoost": xgb
    }


def run_model_arena(
    models: Dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Step 2: 5-Fold Stratified Cross-Validation Benchmarking.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1", "roc_auc"]
    results = {}

    for name, model in models.items():
        print(f"[*] Benchmarking {name}...")
        cv = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1
        )

        results[name] = {
            "train_accuracy": float(cv["train_accuracy"].mean()),
            "test_accuracy": float(cv["test_accuracy"].mean()),
            "train_f1": float(cv["train_f1"].mean()),
            "test_f1": float(cv["test_f1"].mean()),
            "train_roc_auc": float(cv["train_roc_auc"].mean()),
            "test_roc_auc": float(cv["test_roc_auc"].mean()),
        }

    return results


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """
    Step 3A: Parameter search spaces for candidate algorithms.
    """
    return {
        "Logistic Regression": {
            "logisticregression__C": [0.01, 0.1, 1.0, 10.0]
        },
        "Decision Tree": {
            "max_depth": [3, 4, 5, 6],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Random Forest": {
            "n_estimators": [100, 150],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "XGBoost": {
            "n_estimators": [50, 100],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0]
        }
    }


def tune_candidate_models(
    models: Dict[str, BaseEstimator],
    param_grids: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    Step 3B: Multi-model parallel GridSearchCV optimization.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    tuning_results = {}

    for name, model in models.items():
        if name not in param_grids:
            continue

        print(f"[*] Tuning {name} with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=skf,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True
        )
        grid_search.fit(X_train, y_train)

        tuning_results[name] = {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "search_object": grid_search
        }
        print(f"    - {name} Best F1: {grid_search.best_score_:.4f} | Params: {grid_search.best_params_}")

    return tuning_results


def select_champion_model(
    tuning_results: Dict[str, Dict[str, Any]]
) -> Tuple[str, BaseEstimator, Dict[str, Any], float]:
    """
    Step 3C (Dynamic): Automatically selects top-performing model from tuning results.
    """
    champion_name = max(tuning_results, key=lambda k: tuning_results[k]["best_score"])
    champion_data = tuning_results[champion_name]
    best_params = champion_data["best_params"]
    best_score = float(champion_data["best_score"])

    print("\n" + "=" * 65)
    print(f"      [CHAMPION] OFFICIAL CHAMPION MODEL: {champion_name.upper()}")
    print("=" * 65)
    print(f"  * Cross-Validated F1-Score: {best_score:.4f}")
    print("  * Winning Hyperparameters:")
    for param, val in best_params.items():
        print(f"      - {param}: {val}")
    print("=" * 65 + "\n")

    return (
        champion_name,
        champion_data["best_estimator"],
        best_params,
        best_score
    )


def tune_champion_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Random Forest",
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Step 3C (Dedicated): Configurable single-model tuning (defaults to Random Forest).
    """
    models = get_model_registry(random_state)
    grids = get_hyperparameter_grids()

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=models[model_name],
        param_grid=grids[model_name],
        cv=skf,
        scoring="f1",
        n_jobs=n_jobs,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, float(grid_search.best_score_)


def save_model_artifact(model: BaseEstimator, output_filepath: str) -> None:
    """
    Step 4: Serialize and save the trained model artifact.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    joblib.dump(model, output_filepath)
    print(f"[*] Successfully saved model artifact to: {output_filepath}")
