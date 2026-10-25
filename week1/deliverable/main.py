import argparse
import os
import sys
from typing import Tuple

# Ensure local src/ package is discoverable when executing main.py directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports from src modules (following SRP and OCP)
from src.preprocessing import (
    load_raw_data,
    drop_uninformative_columns,
    impute_missing_values,
    engineer_domain_features,
    encode_categorical_features,
    perform_stratified_split,
    run_full_preprocessing_pipeline
)

from src.train import (
    get_model_registry,
    run_model_arena,
    get_hyperparameter_grids,
    tune_candidate_models,
    select_champion_model,
    tune_champion_model,
    save_model_artifact
)

from src.evaluate import (
    build_leaderboard,
    plot_model_arena,
    compare_tuned_models,
    evaluate_confusion_matrix,
    extract_feature_importances,
    evaluate_held_out_set
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for configurable pipeline execution.
    """
    default_train = "data/train.csv" if os.path.exists("data/train.csv") else "../../data/train.csv"
    default_test = "data/test.csv" if os.path.exists("data/test.csv") else "../../data/test.csv"

    parser = argparse.ArgumentParser(description="Titanic Classical ML Pipeline")
    parser.add_argument("--train-path", type=str, default=default_train, help="Path to training CSV")
    parser.add_argument("--test-path", type=str, default=default_test, help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save figures and reports")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=True, help="Display interactive plot popups")
    return parser.parse_args()


def main() -> None:
    """
    Execute the full end-to-end Machine Learning pipeline step-by-step.
    """
    # =========================================================================
    # STEP 0: Configuration & Argument Parsing
    # =========================================================================
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    print("[*] Pipeline Configured:")
    print(f"    - Train path: {args.train_path}")
    print(f"    - Test path:  {args.test_path}")
    print(f"    - Output dir: {args.output_dir}")
    print(f"    - Reports:    {args.reports_dir}")
    print(f"    - Seed:       {args.random_state}\n")

    # =========================================================================
    # STEP 1: Data Ingestion & Preprocessing Pipeline (src.preprocessing)
    # =========================================================================
    print("[*] Running Preprocessing Pipeline...")
    X_train, X_val, y_train, y_val, test_passenger_ids = run_full_preprocessing_pipeline(
        train_path=args.train_path,
        test_path=args.test_path
    )
    print(f"    - Preprocessing complete! Features: {list(X_train.columns)}\n")

    # =========================================================================
    # STEP 2: Model Benchmarking & Cross-Validation Arena (src.train)
    # =========================================================================
    print("[*] Instantiating candidate model registry...")
    models = get_model_registry(args.random_state)

    print("[*] Running 5-Fold Stratified CV Arena across models...")
    arena_results = run_model_arena(models, X_train, y_train, cv_splits=5)

    leaderboard = build_leaderboard(arena_results, sort_by="Test F1")
    print("\n[*] Baseline Arena Leaderboard:")
    print(leaderboard.to_string(index=False))
    print()

    # Visual Leaderboard Plot with Overfit Gap analysis
    plot_model_arena(
        leaderboard,
        save_fig_path=os.path.join(args.reports_dir, "arena_benchmark.png"),
        show_plot=args.show_plots
    )

    # =========================================================================
    # STEP 3: In-Depth Diagnostic Analysis (src.evaluate)
    # =========================================================================
    print("[*] Running Out-of-Fold Error Diagnostics on Random Forest...")
    cm, breakdown = evaluate_confusion_matrix(
        models["Random Forest"],
        X_train,
        y_train,
        cv_splits=5,
        save_fig_path=os.path.join(args.reports_dir, "confusion_matrix.png"),
        show_plot=args.show_plots
    )
    print(f"    - True Negatives (Perished):  {breakdown['True Negatives (Correct Perished)']}")
    print(f"    - True Positives (Survived):  {breakdown['True Positives (Correct Survived)']}")
    print(f"    - Type I Error (False Pos):   {breakdown['False Positives (Type I Error)']}")
    print(f"    - Type II Error (False Neg):  {breakdown['False Negatives (Type II Error)']}\n")

    # =========================================================================
    # STEP 4: Hyperparameter Optimization & Model Comparison (src.train & src.evaluate)
    # =========================================================================
    print("[*] Retrieving hyperparameter search grids...")
    param_grids = get_hyperparameter_grids()

    print("[*] Tuning candidate models across grids...")
    tuning_results = tune_candidate_models(
        models=models,
        param_grids=param_grids,
        X_train=X_train,
        y_train=y_train,
        cv_splits=5,
        scoring="f1",
        random_state=args.random_state
    )

    champion_name, champion_estimator, best_params, best_score = select_champion_model(tuning_results)

    comparison = compare_tuned_models(tuning_results, baseline_results=arena_results)
    print("\n[*] Tuned Models Comparison Leaderboard:")
    print(comparison.to_string(index=False))
    print()

    # =========================================================================
    # STEP 5: Held-Out Test Evaluation & Model Persistence (src.evaluate & src.train)
    # =========================================================================
    print(f"[*] Fitting overall champion ({champion_name}) on full training data...")
    champion_estimator.fit(X_train, y_train)

    # Export feature importance visualization
    extract_feature_importances(
        champion_estimator,
        feature_names=list(X_train.columns),
        top_n=10,
        save_fig_path=os.path.join(args.reports_dir, "feature_importance.png"),
        show_plot=args.show_plots
    )

    # Evaluate on pristine held-out validation set (179 rows) and export ROC curve
    evaluate_held_out_set(
        champion_estimator,
        X_val,
        y_val,
        save_fig_path=os.path.join(args.reports_dir, "roc_curve.png"),
        show_plot=args.show_plots
    )

    # Save model artifact
    artifact_path = os.path.join(args.output_dir, "champion_model.joblib")
    save_model_artifact(champion_estimator, artifact_path)
    print("\n[+] Full pipeline executed successfully!")


if __name__ == "__main__":
    main()
