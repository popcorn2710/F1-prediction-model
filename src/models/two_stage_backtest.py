"""
F1 Prediction - Two-Stage Qualifying -> Race Backtest
======================================================

Purpose
-------
Compare the existing one-stage race benchmark against a realistic two-stage
pipeline:

    pre-qualifying information -> predicted qualifying position
                              -> race finish prediction

The backtest is walk-forward by season, so every validation race is treated
as if its future qualifying/race outcome were unknown.

Models
------
1. BENCHMARK_ORACLE
   The existing Ridge race benchmark. It is given the actual qualifying
   information (including Q1/Q2/Q3 times and grid), matching baseline.py.

2. QUALIFYING_MODEL
   Ridge model that predicts qualifying_position using only information that
   exists BEFORE qualifying. It does not use qualifying_position, q-times,
   grid, sprint results from the same weekend, or race targets.

3. TWO_STAGE_MODEL
   Stage 1 predicts qualifying position. Stage 2 predicts finish position
   using the Stage-1 predicted qualifying position plus the same pre-race
   historical features. This is the end-to-end model we care about.

4. ORACLE_POSITION_MODEL
   Same Stage-2 race model, but fed the ACTUAL qualifying position. This
   isolates how much accuracy is lost because qualifying itself was predicted.

Important
---------
The existing benchmark is intentionally NOT changed. This experiment creates
new output files under models/two_stage/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "f1_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "two_stage"

TARGET_RACE = "finish_position"
TARGET_QUAL = "qualifying_position"

# Information available before qualifying.
# Deliberately excludes all current-weekend qualifying and grid fields.
PRE_QUAL_FEATURES = [
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
]

# Race features that exist before/at the point where we have a predicted
# qualifying result. The predicted qualifying position is added separately.
RACE_BASE_FEATURES = PRE_QUAL_FEATURES

# Exact feature set from the existing baseline.py.
BENCHMARK_FEATURES = [
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

FOLDS = [
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
    {"train_end": 2024, "test_year": 2025},
    {"train_end": 2025, "test_year": 2026},
]


def ridge_model(alpha=10.0):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def ranking_metrics(actual, predicted):
    if len(actual) < 2:
        return np.nan, np.nan
    return (
        spearmanr(actual, predicted).statistic,
        kendalltau(actual, predicted).statistic,
    )


def load_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}\n\n"
            "Run src/features/build_features.py first."
        )

    df = pd.read_csv(FEATURE_FILE)
    required = set(
        PRE_QUAL_FEATURES
        + BENCHMARK_FEATURES
        + ["year", "round", "raceId", "driverId", TARGET_RACE, TARGET_QUAL]
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required feature columns:\n" + "\n".join(missing)
        )

    df = df[df["year"] >= 2018].copy()
    df[TARGET_RACE] = pd.to_numeric(df[TARGET_RACE], errors="coerce")
    df[TARGET_QUAL] = pd.to_numeric(df[TARGET_QUAL], errors="coerce")
    df = df.sort_values(["year", "round", "raceId", "driverId"]).reset_index(drop=True)
    return df


def fit_predict(model, train_df, test_df, features, target):
    train = train_df.dropna(subset=[target])
    if train.empty:
        raise RuntimeError(f"No training rows available for target {target}")

    model.fit(train[features], train[target])
    pred = model.predict(test_df[features])
    return np.maximum(pred, 1.0), model


def rank_within_race(df, raw_col, output_col):
    """Convert continuous model scores into a valid 1..N race order."""
    out = df.copy()
    out[output_col] = (
        out.groupby("raceId")[raw_col]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return out


def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    spearman, kendall = ranking_metrics(
        np.asarray(actual), np.asarray(predicted)
    )
    return mae, rmse, spearman, kendall


def main():
    print("=" * 78)
    print("F1 TWO-STAGE QUALIFYING -> RACE BACKTEST")
    print("=" * 78)

    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Feature file: {FEATURE_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Pre-qualifying features: {len(PRE_QUAL_FEATURES)}")
    print(f"Benchmark race features: {len(BENCHMARK_FEATURES)}")

    all_metrics = []
    all_predictions = []
    qualifying_metrics = []

    for fold in FOLDS:
        train_end = fold["train_end"]
        test_year = fold["test_year"]

        train_df = df[df["year"] <= train_end].copy()
        test_df = df[df["year"] == test_year].copy()

        # For a clean end-to-end race comparison, evaluate only drivers that
        # have both a qualifying position and an eventual finish position.
        test_eval = test_df.dropna(subset=[TARGET_QUAL, TARGET_RACE]).copy()
        train_race = train_df.dropna(subset=[TARGET_RACE]).copy()
        train_qual = train_df.dropna(subset=[TARGET_QUAL]).copy()

        print("\n" + "-" * 78)
        print(f"FOLD: train <= {train_end} | test = {test_year}")
        print("-" * 78)
        print(f"Train race rows: {len(train_race):,}")
        print(f"Train qualifying rows: {len(train_qual):,}")
        print(f"Test rows: {len(test_eval):,}")
        print(f"Test races: {test_eval['raceId'].nunique()}")

        # ================================================================
        # STAGE 1: PREDICT QUALIFYING
        # ================================================================
        qual_model = ridge_model()
        qual_model.fit(train_qual[PRE_QUAL_FEATURES], train_qual[TARGET_QUAL])

        test_eval["predicted_qualifying_raw"] = qual_model.predict(
            test_eval[PRE_QUAL_FEATURES]
        )
        test_eval["predicted_qualifying_raw"] = np.maximum(
            test_eval["predicted_qualifying_raw"], 1.0
        )

        test_eval = rank_within_race(
            test_eval,
            "predicted_qualifying_raw",
            "predicted_qualifying_position",
        )

        q_mae, q_rmse, q_spearman, q_kendall = evaluate(
            test_eval[TARGET_QUAL],
            test_eval["predicted_qualifying_position"],
        )

        qualifying_metrics.append(
            {
                "train_through": train_end,
                "test_year": test_year,
                "MAE": q_mae,
                "RMSE": q_rmse,
                "Spearman": q_spearman,
                "Kendall": q_kendall,
            }
        )

        # ================================================================
        # STAGE 2A: ORACLE RACE MODEL (ACTUAL QUALIFYING POSITION)
        # ================================================================
        # This tells us the race model's performance when qualifying is known,
        # but we deliberately use only qualifying POSITION here. It is not the
        # old benchmark; it isolates the information content of qualifying order.
        oracle_train = train_race.dropna(subset=[TARGET_QUAL]).copy()
        oracle_features = RACE_BASE_FEATURES + [TARGET_QUAL]

        oracle_model = ridge_model()
        oracle_model.fit(oracle_train[oracle_features], oracle_train[TARGET_RACE])
        test_eval["oracle_predicted_finish"] = np.maximum(
            oracle_model.predict(test_eval[oracle_features]), 1.0
        )

        oracle_mae, oracle_rmse, oracle_spearman, oracle_kendall = evaluate(
            test_eval[TARGET_RACE],
            test_eval["oracle_predicted_finish"],
        )

        # ================================================================
        # STAGE 2B: END-TO-END TWO-STAGE RACE MODEL
        # ================================================================
        # IMPORTANT: stage 2 receives the same type of input it will receive
        # in production: a predicted qualifying position.
        two_stage_train = train_race.dropna(subset=[TARGET_QUAL]).copy()
        two_stage_features = RACE_BASE_FEATURES + ["predicted_qualifying_position"]

        # To prevent train/test distribution mismatch, generate out-of-fold
        # qualifying predictions for the training races. Each training race's
        # predicted qualifying position comes from a model trained only on
        # earlier seasons.
        train_aug = two_stage_train.copy()
        train_aug["predicted_qualifying_position"] = np.nan

        inner_years = sorted(train_aug["year"].unique())
        for inner_test_year in inner_years:
            if inner_test_year <= 2018:
                continue

            inner_train = train_aug[train_aug["year"] < inner_test_year].copy()
            inner_test_mask = train_aug["year"] == inner_test_year
            inner_test = train_aug[inner_test_mask].copy()

            if inner_train.empty or inner_test.empty:
                continue

            inner_qual = inner_train.dropna(subset=[TARGET_QUAL])
            if inner_qual.empty:
                continue

            inner_model = ridge_model()
            inner_model.fit(inner_qual[PRE_QUAL_FEATURES], inner_qual[TARGET_QUAL])
            raw = np.maximum(
                inner_model.predict(inner_test[PRE_QUAL_FEATURES]), 1.0
            )
            inner_test = inner_test.copy()
            inner_test["_raw_q"] = raw
            inner_test = rank_within_race(
                inner_test, "_raw_q", "predicted_qualifying_position"
            )
            train_aug.loc[inner_test.index, "predicted_qualifying_position"] = (
                inner_test["predicted_qualifying_position"].to_numpy()
            )

        train_aug = train_aug.dropna(subset=["predicted_qualifying_position"])

        if train_aug.empty:
            raise RuntimeError(
                f"Could not create out-of-fold qualifying predictions for fold {test_year}."
            )

        two_stage_model = ridge_model()
        two_stage_model.fit(
            train_aug[two_stage_features],
            train_aug[TARGET_RACE],
        )

        test_eval["two_stage_predicted_finish"] = np.maximum(
            two_stage_model.predict(test_eval[two_stage_features]), 1.0
        )

        two_mae, two_rmse, two_spearman, two_kendall = evaluate(
            test_eval[TARGET_RACE],
            test_eval["two_stage_predicted_finish"],
        )

        # ================================================================
        # CURRENT BENCHMARK: ACTUAL Q1/Q2/Q3 + GRID
        # ================================================================
        benchmark_train = train_race.dropna(subset=[TARGET_RACE]).copy()
        benchmark_model = ridge_model()
        benchmark_model.fit(
            benchmark_train[BENCHMARK_FEATURES],
            benchmark_train[TARGET_RACE],
        )
        test_eval["benchmark_predicted_finish"] = np.maximum(
            benchmark_model.predict(test_eval[BENCHMARK_FEATURES]), 1.0
        )

        b_mae, b_rmse, b_spearman, b_kendall = evaluate(
            test_eval[TARGET_RACE],
            test_eval["benchmark_predicted_finish"],
        )

        fold_rows = [
            ("benchmark_oracle", b_mae, b_rmse, b_spearman, b_kendall),
            ("oracle_position_race", oracle_mae, oracle_rmse, oracle_spearman, oracle_kendall),
            ("two_stage", two_mae, two_rmse, two_spearman, two_kendall),
        ]

        for model_name, mae, rmse, sp, ke in fold_rows:
            all_metrics.append(
                {
                    "train_through": train_end,
                    "test_year": test_year,
                    "model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "Spearman": sp,
                    "Kendall": ke,
                }
            )

        prediction_cols = [
            "raceId", "year", "round", "driverId",
            TARGET_QUAL, "predicted_qualifying_position",
            TARGET_RACE,
            "benchmark_predicted_finish",
            "oracle_predicted_finish",
            "two_stage_predicted_finish",
        ]
        all_predictions.append(test_eval[prediction_cols].copy())

        print("\nQualifying model:")
        print(f"  MAE:      {q_mae:.4f}")
        print(f"  RMSE:     {q_rmse:.4f}")
        print(f"  Spearman: {q_spearman:.4f}")
        print(f"  Kendall:  {q_kendall:.4f}")

        print("\nRace models:")
        print(f"  Existing benchmark: MAE={b_mae:.4f}  Spearman={b_spearman:.4f}")
        print(f"  Oracle actual Q-pos: MAE={oracle_mae:.4f}  Spearman={oracle_spearman:.4f}")
        print(f"  Two-stage predicted Q: MAE={two_mae:.4f}  Spearman={two_spearman:.4f}")

    # ================================================================
    # SAVE
    # ================================================================
    metrics_df = pd.DataFrame(all_metrics)
    qualifying_df = pd.DataFrame(qualifying_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = OUTPUT_DIR / "two_stage_metrics.csv"
    qualifying_path = OUTPUT_DIR / "qualifying_metrics.csv"
    predictions_path = OUTPUT_DIR / "two_stage_predictions.csv"
    summary_path = OUTPUT_DIR / "two_stage_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    qualifying_df.to_csv(qualifying_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    summary = (
        metrics_df.groupby("model")[["MAE", "RMSE", "Spearman", "Kendall"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("FINAL COMPARISON — AVERAGE WALK-FORWARD PERFORMANCE")
    print("=" * 78)
    print(summary.to_string(index=False))

    print("\nFiles saved:")
    print(f"  {metrics_path}")
    print(f"  {qualifying_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")

    print("\nInterpretation:")
    print("  benchmark_oracle     = your existing benchmark with actual Q1/Q2/Q3 + grid")
    print("  oracle_position_race = race model using actual qualifying POSITION only")
    print("  two_stage            = predicted qualifying POSITION -> race prediction")
    print("  qualifying_metrics   = how well Stage 1 predicts qualifying")


if __name__ == "__main__":
    main()
