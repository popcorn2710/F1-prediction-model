"""
F1 Prediction - Baseline Model
--------------------------------
Purpose:
    Train a simple Ridge Regression baseline for finishing position
    using only core pre-race-safe features.

Validation:
    Walk-forward by season:
        2018-2022 -> 2023
        2018-2023 -> 2024
        2018-2024 -> 2025
        2018-2025 -> 2026

Important:
    - No random train/test split.
    - No target columns are used as predictors.
    - Driver/constructor/circuit IDs are NOT used as numeric predictors.
    - Missing values are imputed using training-fold medians only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "f1_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline"

TARGET = "finish_position"

# These are intentionally conservative.
# They are available after qualifying and before the race.
CORE_FEATURES = [
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "q3_gap_to_pole",

    "driver_avg_finish_L3",
    "driver_avg_finish_L5",
    "driver_avg_finish_L10",
    "driver_points_L5",
    "driver_dnf_rate_L10",
    "driver_podium_rate_L10",
    "driver_win_rate_L10",

    "constructor_avg_finish_L5",
    "constructor_points_L5",
    "constructor_dnf_rate_L10",

    "driver_points_before_race",
    "driver_championship_position",
    "constructor_points_before_race",
    "constructor_championship_position",

    "driver_circuit_avg_finish",
    "driver_circuit_races",

    "season_progress",

    "qualifying_vs_teammate",
    "grid_vs_teammate",
]

# Walk-forward folds.
FOLDS = [
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
    {"train_end": 2024, "test_year": 2025},
    {"train_end": 2025, "test_year": 2026},
]


# ============================================================
# HELPERS
# ============================================================

def load_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find feature file:\n{FEATURE_FILE}\n\n"
            "Run build_features.py first."
        )

    df = pd.read_csv(FEATURE_FILE)

    required = set(CORE_FEATURES + ["year", "raceId", TARGET])
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "The feature file is missing required columns:\n"
            + "\n".join(missing)
        )

    # Keep only races from the intended modeling era.
    df = df[df["year"] >= 2018].copy()

    # Target must be numeric.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Drop rows where the target itself is unavailable.
    df = df.dropna(subset=[TARGET])

    # Sort chronologically.
    df = df.sort_values(["year", "raceId"]).reset_index(drop=True)

    return df


def build_model():
    """
    Baseline pipeline:
        median imputation -> standardization -> Ridge regression

    The imputer is inside the pipeline, so medians are learned
    separately for each training fold and never from validation data.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=10.0),
            ),
        ]
    )


def ranking_metrics(actual, predicted):
    """
    Compare the predicted ordering with the actual ordering.

    Higher is better for Spearman/Kendall.
    """
    if len(actual) < 2:
        return np.nan, np.nan

    spearman = spearmanr(actual, predicted).statistic
    kendall = kendalltau(actual, predicted).statistic

    return spearman, kendall


def evaluate_fold(model, train_df, test_df):
    X_train = train_df[CORE_FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[CORE_FEATURES]
    y_test = test_df[TARGET]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Finish positions cannot be below P1.
    predictions = np.maximum(predictions, 1.0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    spearman, kendall = ranking_metrics(
        y_test.to_numpy(),
        predictions,
    )

    results = test_df[
        ["raceId", "year", "round", "driverId", TARGET]
    ].copy()

    results["predicted_finish"] = predictions
    results["absolute_error"] = np.abs(
        results[TARGET] - results["predicted_finish"]
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "kendall": kendall,
        "predictions": results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("F1 FINISHING POSITION BASELINE")
    print("=" * 70)

    df = load_data()

    print(f"\nFeature file: {FEATURE_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Features used: {len(CORE_FEATURES)}")

    print("\nCore features:")
    for feature in CORE_FEATURES:
        print(f"  - {feature}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    all_predictions = []

    for fold in FOLDS:
        train_end = fold["train_end"]
        test_year = fold["test_year"]

        train_df = df[df["year"] <= train_end].copy()
        test_df = df[df["year"] == test_year].copy()

        if train_df.empty:
            print(f"\nSkipping {test_year}: no training data.")
            continue

        if test_df.empty:
            print(f"\nSkipping {test_year}: no test data.")
            continue

        print("\n" + "-" * 70)
        print(f"FOLD: train <= {train_end}  |  test = {test_year}")
        print("-" * 70)
        print(f"Train rows: {len(train_df):,}")
        print(f"Test rows:  {len(test_df):,}")
        print(f"Test races: {test_df['raceId'].nunique()}")

        model = build_model()

        result = evaluate_fold(
            model=model,
            train_df=train_df,
            test_df=test_df,
        )

        fold_metrics.append(
            {
                "train_through": train_end,
                "test_year": test_year,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_races": test_df["raceId"].nunique(),
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "Spearman": result["spearman"],
                "Kendall": result["kendall"],
            }
        )

        all_predictions.append(result["predictions"])

        print(f"MAE:       {result['mae']:.4f}")
        print(f"RMSE:      {result['rmse']:.4f}")
        print(f"Spearman:  {result['spearman']:.4f}")
        print(f"Kendall:   {result['kendall']:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(fold_metrics)

    if metrics_df.empty:
        raise RuntimeError("No validation folds were completed.")

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    predictions_path = OUTPUT_DIR / "baseline_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)

    print(
        metrics_df[
            [
                "train_through",
                "test_year",
                "MAE",
                "RMSE",
                "Spearman",
                "Kendall",
            ]
        ].to_string(index=False)
    )

    print("\nAverage across folds:")
    print(f"MAE:       {metrics_df['MAE'].mean():.4f}")
    print(f"RMSE:      {metrics_df['RMSE'].mean():.4f}")
    print(f"Spearman:  {metrics_df['Spearman'].mean():.4f}")
    print(f"Kendall:   {metrics_df['Kendall'].mean():.4f}")

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")

    print("\nBaseline complete.")
    print("Do NOT tune XGBoost yet.")
    print("First record these baseline numbers.")


if __name__ == "__main__":
    main()
