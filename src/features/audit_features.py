from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "features"
    / "f1_features.csv"
)


df = pd.read_csv(INPUT_FILE)


print("=" * 60)
print("F1 FEATURE / LEAKAGE AUDIT")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Races: {df['raceId'].nunique()}")
print()


# ============================================================
# TARGET COLUMNS
# ============================================================

target_columns = [
    "finish_position",
    "positionOrder",
    "points",
    "laps",
    "is_classified",
    "is_dnf",
]


# ============================================================
# DEFINITELY SAFE PREDICTORS
# ============================================================

safe_features = [
    "raceId",
    "year",
    "round",
    "circuitId",
    "driverId",
    "constructorId",

    # Practice
    "fp1_position",
    "fp2_position",
    "fp3_position",
    "fp1_best_lap_seconds",
    "fp2_best_lap_seconds",
    "fp3_best_lap_seconds",
    "fp1_laps",
    "fp2_laps",
    "fp3_laps",
    "practice_avg_position",
    "practice_best_position",
    "practice_avg_lap_seconds",

    # Qualifying
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "q3_gap_to_pole",
    "made_q2",
    "made_q3",

    # Grid
    "grid",

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

    # Circuit
    "driver_circuit_avg_finish",
    "driver_circuit_races",

    # Season
    "season_progress",

    # Sprint
    "sprint_grid",
    "sprint_position",
    "sprint_points",
    "sprint_laps",
    "sprint_is_dnf",
    "has_sprint",

    # Teammate
    "qualifying_vs_teammate",
    "grid_vs_teammate",
    "practice_vs_teammate",
    "sprint_vs_teammate",
]


# ============================================================
# CHECK MISSING COLUMNS
# ============================================================

missing_safe = [
    col for col in safe_features
    if col not in df.columns
]

missing_targets = [
    col for col in target_columns
    if col not in df.columns
]


print("Missing safe features:")
print(missing_safe)

print()

print("Missing targets:")
print(missing_targets)

print()


# ============================================================
# TARGETS ACCIDENTALLY INCLUDED IN PREDICTORS?
# ============================================================

overlap = set(safe_features) & set(target_columns)

print("Target/predictor overlap:")
print(overlap)

print()


# ============================================================
# MISSING VALUES
# ============================================================

print("Top missing-value columns:")

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing.head(20).to_string())

print()


# ============================================================
# SINGAPORE / LATEST RACE
# ============================================================

print("Latest races:")

print(
    df[
        [
            "year",
            "round",
            "raceId"
        ]
    ]
    .drop_duplicates()
    .sort_values(["year", "round"])
    .tail(10)
    .to_string(index=False)
)

print()


# ============================================================
# FINAL RESULT
# ============================================================

if missing_safe:
    print("WARNING: Some safe features are missing!")

if missing_targets:
    print("WARNING: Some target columns are missing!")

if overlap:
    print("WARNING: TARGET LEAKAGE IN SAFE FEATURE LIST!")

if not missing_safe and not missing_targets and not overlap:
    print("✓ FEATURE DEFINITIONS LOOK CLEAN")