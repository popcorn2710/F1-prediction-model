from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "f1_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "src" / "models" / "main"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAME 24 FEATURES USED BY RIDGE BENCHMARK
# ============================================================

FEATURES = [
    # Qualifying
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "q3_gap_to_pole",

    # Driver form
    "driver_avg_finish_L3",
    "driver_avg_finish_L5",
    "driver_avg_finish_L10",
    "driver_points_L5",
    "driver_dnf_rate_L10",
    "driver_podium_rate_L10",
    "driver_win_rate_L10",

    # Constructor form
    "constructor_avg_finish_L5",
    "constructor_points_L5",
    "constructor_dnf_rate_L10",

    # Championship
    "driver_points_before_race",
    "driver_championship_position",
    "constructor_points_before_race",
    "constructor_championship_position",

    # Circuit history
    "driver_circuit_avg_finish",
    "driver_circuit_races",

    # Season
    "season_progress",

    # Teammate comparison
    "qualifying_vs_teammate",
    "grid_vs_teammate",

    # Practice
    "fp1_position",
    "fp1_best_lap_seconds",
    "fp1_laps",
    "practice_avg_position",
    "practice_best_position",
    "practice_avg_lap_seconds",

    # Sprint
    "sprint_grid",
    "sprint_position",
    "sprint_points",
    "sprint_laps",
    "sprint_is_dnf",
    "has_sprint",
    "sprint_vs_teammate",
]

TARGET = "finish_position"


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

FOLDS = [
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
    (2025, 2026),
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

df = df.sort_values(
    ["year", "round", "raceId", "driverId"]
).reset_index(drop=True)

df = df[df[TARGET].notna()].copy()

print("=" * 70)
print("XGBOOST MAIN RACE MODEL")
print("=" * 70)

print(f"Feature file: {FEATURE_FILE}")
print(f"Rows: {len(df):,}")
print(f"Features: {len(FEATURES)}")
print(f"Target: {TARGET}")
print()


# ============================================================
# STORAGE
# ============================================================

all_predictions = []
metrics = []


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

for train_end, test_year in FOLDS:

    train = df[df["year"] <= train_end].copy()
    test = df[df["year"] == test_year].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    model = XGBRegressor(
        objective="reg:squarederror",

        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,

        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=1.0,

        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(X_test)
    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    spearman = spearmanr(
        y_test,
        predictions
    ).statistic

    kendall = kendalltau(
        y_test,
        predictions
    ).statistic

    print(f"FOLD train<={train_end} test={test_year}")
    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")
    print()

    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"Kendall:  {kendall:.4f}")
    print()
    print("-" * 70)

    metrics.append({
        "train_end_year": train_end,
        "test_year": test_year,
        "train_rows": len(train),
        "test_rows": len(test),
        "MAE": mae,
        "RMSE": rmse,
        "Spearman": spearman,
        "Kendall": kendall,
    })

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    fold_predictions = test[
        [
            "raceId",
            "year",
            "round",
            "driverId",
            TARGET,
        ]
    ].copy()

    fold_predictions["predicted_finish"] = predictions
    fold_predictions["predicted_finish_rounded"] = (
        predictions.round().astype(int)
    )

    fold_predictions["train_end_year"] = train_end

    all_predictions.append(fold_predictions)


# ============================================================
# SUMMARY
# ============================================================

metrics_df = pd.DataFrame(metrics)

summary = pd.DataFrame([{
    "MAE": metrics_df["MAE"].mean(),
    "RMSE": metrics_df["RMSE"].mean(),
    "Spearman": metrics_df["Spearman"].mean(),
    "Kendall": metrics_df["Kendall"].mean(),
}])

print()
print("=" * 70)
print("FINAL XGBOOST RESULTS")
print("=" * 70)

print(f"MAE:      {summary.iloc[0]['MAE']:.4f}")
print(f"RMSE:     {summary.iloc[0]['RMSE']:.4f}")
print(f"Spearman: {summary.iloc[0]['Spearman']:.4f}")
print(f"Kendall:  {summary.iloc[0]['Kendall']:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_path = OUTPUT_DIR / "xgboost_metrics.csv"
predictions_path = OUTPUT_DIR / "xgboost_predictions.csv"
summary_path = OUTPUT_DIR / "xgboost_summary.csv"

metrics_df.to_csv(metrics_path, index=False)

pd.concat(
    all_predictions,
    ignore_index=True
).to_csv(
    predictions_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)

print()
print(f"Metrics saved:     {metrics_path}")
print(f"Predictions saved: {predictions_path}")
print(f"Summary saved:     {summary_path}")