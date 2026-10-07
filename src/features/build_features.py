from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "f1_model_base.csv"
)

OUTPUT_DIR = ROOT / "data" / "features"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "f1_features.csv"
)
PRACTICE_FILE = (
    ROOT
    / "data"
    / "raw"
    / "archive"
    / "practice_results.csv"
)
SPRINT_FILE = ROOT / "data" / "raw" / "archive" / "sprint_results.csv"


# ============================================================
# SETTINGS
# ============================================================

# We start our main modeling era here.
# The original model base remains untouched.
START_YEAR = 2018


# ============================================================
# LOAD DATA
# ============================================================

print("Loading model base...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df):,}")


# ============================================================
# FILTER MODERN F1 ERA
# ============================================================

df = df[df["year"] >= START_YEAR].copy()

print(
    f"Rows from {START_YEAR} onward: "
    f"{len(df):,}"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    [
        "year",
        "round",
        "raceId",
        "driverId"
    ]
).reset_index(drop=True)


# ============================================================
# DRIVER RECENT FORM
# ============================================================

print("\nCreating driver form features...")


# IMPORTANT:
#
# shift(1) means:
#
# Race N
#   ↓
# only races BEFORE Race N
#
# Therefore the current race can NEVER enter
# its own rolling average.

driver_history = (
    df
    .sort_values(["driverId", "year", "round"])
    .groupby("driverId", group_keys=False)
)


# ------------------------------------------------------------
# Average finishing position - last 3 races
# ------------------------------------------------------------

df["driver_avg_finish_L3"] = (
    driver_history["finish_position"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )
)


# ------------------------------------------------------------
# Average finishing position - last 5 races
# ------------------------------------------------------------

df["driver_avg_finish_L5"] = (
    driver_history["finish_position"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )
)


# ------------------------------------------------------------
# Average finishing position - last 10 races
# ------------------------------------------------------------

df["driver_avg_finish_L10"] = (
    driver_history["finish_position"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )
)


# ============================================================
# DRIVER POINTS FORM
# ============================================================

df["driver_points_L5"] = (
    driver_history["points"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(5, min_periods=1)
        .sum()
    )
)


# ============================================================
# DRIVER DNF RATE
# ============================================================

df["driver_dnf_rate_L10"] = (
    driver_history["is_dnf"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )
)


# ============================================================
# DRIVER PODIUM RATE
# ============================================================

# Whether the driver finished P1-P3
df["_podium"] = (
    df["finish_position"]
    .between(1, 3)
    .astype(int)
)

# Re-create history because _podium was added after
driver_history = (
    df
    .sort_values(["driverId", "year", "round"])
    .groupby("driverId", group_keys=False)
)

df["driver_podium_rate_L10"] = (
    driver_history["_podium"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )
)


# ============================================================
# DRIVER WIN RATE
# ============================================================

df["_win"] = (
    df["finish_position"]
    == 1
).astype(int)

driver_history = (
    df
    .sort_values(["driverId", "year", "round"])
    .groupby("driverId", group_keys=False)
)

df["driver_win_rate_L10"] = (
    driver_history["_win"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )
)


# ============================================================
# CONSTRUCTOR FORM
# ============================================================

print("Creating constructor form features...")


# ------------------------------------------------------------
# STEP 1: ONE ROW PER CONSTRUCTOR PER RACE
# ------------------------------------------------------------

constructor_race = (
    df.groupby(
        [
            "constructorId",
            "raceId",
            "year",
            "round"
        ],
        as_index=False
    )
    .agg(
        constructor_avg_finish_race=(
            "finish_position",
            "mean"
        ),

        constructor_points_race=(
            "points",
            "sum"
        ),

        constructor_dnf_race=(
            "is_dnf",
            "mean"
        )
    )
)


# ------------------------------------------------------------
# STEP 2: SORT CHRONOLOGICALLY
# ------------------------------------------------------------

constructor_race = constructor_race.sort_values(
    [
        "constructorId",
        "year",
        "round"
    ]
)


# ------------------------------------------------------------
# STEP 3: PREVIOUS-RACE ROLLING FEATURES
# ------------------------------------------------------------

constructor_group = (
    constructor_race
    .groupby("constructorId", group_keys=False)
)


# Average finishing position - previous 5 races
constructor_race[
    "constructor_avg_finish_L5"
] = (
    constructor_group[
        "constructor_avg_finish_race"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            5,
            min_periods=1
        )
        .mean()
    )
)


# Points - previous 5 races
constructor_race[
    "constructor_points_L5"
] = (
    constructor_group[
        "constructor_points_race"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            5,
            min_periods=1
        )
        .sum()
    )
)


# DNF rate - previous 10 races
constructor_race[
    "constructor_dnf_rate_L10"
] = (
    constructor_group[
        "constructor_dnf_race"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            10,
            min_periods=1
        )
        .mean()
    )
)


# ------------------------------------------------------------
# STEP 4: MERGE BACK INTO DRIVER DATA
# ------------------------------------------------------------

constructor_features = constructor_race[
    [
        "constructorId",
        "raceId",
        "constructor_avg_finish_L5",
        "constructor_points_L5",
        "constructor_dnf_rate_L10"
    ]
]


df = df.merge(
    constructor_features,
    on=[
        "constructorId",
        "raceId"
    ],
    how="left",
    validate="many_to_one"
)
# ============================================================
# CHAMPIONSHIP FEATURES
# ============================================================

print("Creating championship features...")


# ------------------------------------------------------------
# DRIVER CHAMPIONSHIP
# ------------------------------------------------------------

# Sort races chronologically
df = df.sort_values(
    ["year", "round", "raceId", "driverId"]
).reset_index(drop=True)


# Points earned in each race are already in the dataset.
# We calculate cumulative points, then shift by one race.
#
# This means:
#
# Race 1 → 0 previous points
# Race 2 → points from Race 1
# Race 3 → points from Race 1 + Race 2
# etc.

df["driver_points_before_race"] = (
    df
    .groupby(["year", "driverId"])["points"]
    .transform(
        lambda x:
        x.shift(1)
        .fillna(0)
        .cumsum()
    )
)


# ------------------------------------------------------------
# DRIVER CHAMPIONSHIP POSITION
# ------------------------------------------------------------

# Rank drivers based on points BEFORE the current race.

df["driver_championship_position"] = (
    df
    .groupby(["year", "raceId"])[
        "driver_points_before_race"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


# ------------------------------------------------------------
# CONSTRUCTOR CHAMPIONSHIP
# ------------------------------------------------------------

# First calculate constructor points per race.

constructor_points = (
    df.groupby(
        [
            "year",
            "raceId",
            "round",
            "constructorId"
        ],
        as_index=False
    )["points"]
    .sum()
    .rename(
        columns={
            "points":
            "constructor_race_points"
        }
    )
)


# Sort chronologically

constructor_points = (
    constructor_points
    .sort_values(
        [
            "constructorId",
            "year",
            "round"
        ]
    )
)


# Cumulative constructor points BEFORE current race

constructor_points[
    "constructor_points_before_race"
] = (
    constructor_points
    .groupby(
        ["year", "constructorId"]
    )["constructor_race_points"]
    .transform(
        lambda x:
        x.shift(1)
        .fillna(0)
        .cumsum()
    )
)


# Constructor championship position

constructor_points[
    "constructor_championship_position"
] = (
    constructor_points
    .groupby(
        ["year", "raceId"]
    )["constructor_points_before_race"]
    .rank(
        method="min",
        ascending=False
    )
)


# ------------------------------------------------------------
# MERGE CONSTRUCTOR STANDINGS BACK
# ------------------------------------------------------------

constructor_standings_features = constructor_points[
    [
        "year",
        "raceId",
        "constructorId",
        "constructor_points_before_race",
        "constructor_championship_position"
    ]
]


df = df.merge(
    constructor_standings_features,
    on=[
        "year",
        "raceId",
        "constructorId"
    ],
    how="left",
    validate="many_to_one"
)

# ============================================================
# PRACTICE SESSION FEATURES
# ============================================================

print("Creating practice session features...")


# ------------------------------------------------------------
# LOAD PRACTICE DATA
# ------------------------------------------------------------

practice = pd.read_csv(
    PRACTICE_FILE,
    na_values=["\\N"]
)


# ------------------------------------------------------------
# CONVERT TYPES
# ------------------------------------------------------------

practice["raceId"] = pd.to_numeric(
    practice["raceId"],
    errors="coerce"
)

practice["driverId"] = pd.to_numeric(
    practice["driverId"],
    errors="coerce"
)

practice["position"] = pd.to_numeric(
    practice["position"],
    errors="coerce"
)

practice["laps"] = pd.to_numeric(
    practice["laps"],
    errors="coerce"
)


# ------------------------------------------------------------
# CONVERT LAP TIME TO SECONDS
# ------------------------------------------------------------

practice["bestLapTime_seconds"] = (
    pd.to_timedelta(
        practice["bestLapTime"],
        errors="coerce"
    )
    .dt.total_seconds()
)


# ------------------------------------------------------------
# KEEP ONLY THE INFORMATION WE NEED
# ------------------------------------------------------------

practice = practice[
    [
        "raceId",
        "driverId",
        "session",
        "position",
        "bestLapTime_seconds",
        "laps"
    ]
].copy()


# ------------------------------------------------------------
# REMOVE DUPLICATES
# ------------------------------------------------------------

practice = practice.drop_duplicates(
    subset=[
        "raceId",
        "driverId",
        "session"
    ],
    keep="first"
)


# ------------------------------------------------------------
# PIVOT FP1 / FP2 / FP3
# ------------------------------------------------------------

practice_position = (
    practice
    .pivot(
        index=[
            "raceId",
            "driverId"
        ],
        columns="session",
        values="position"
    )
    .reset_index()
)


practice_time = (
    practice
    .pivot(
        index=[
            "raceId",
            "driverId"
        ],
        columns="session",
        values="bestLapTime_seconds"
    )
    .reset_index()
)


practice_laps = (
    practice
    .pivot(
        index=[
            "raceId",
            "driverId"
        ],
        columns="session",
        values="laps"
    )
    .reset_index()


)


# ------------------------------------------------------------
# RENAME COLUMNS
# ------------------------------------------------------------

practice_position = practice_position.rename(
    columns={
        "FP1": "fp1_position",
        "FP2": "fp2_position",
        "FP3": "fp3_position"
    }
)


practice_time = practice_time.rename(
    columns={
        "FP1": "fp1_best_lap_seconds",
        "FP2": "fp2_best_lap_seconds",
        "FP3": "fp3_best_lap_seconds"
    }
)


practice_laps = practice_laps.rename(
    columns={
        "FP1": "fp1_laps",
        "FP2": "fp2_laps",
        "FP3": "fp3_laps"
    }
)


# ------------------------------------------------------------
# MERGE PRACTICE TABLES
# ------------------------------------------------------------

practice_features = practice_position.merge(
    practice_time,
    on=[
        "raceId",
        "driverId"
    ],
    how="outer"
)


practice_features = practice_features.merge(
    practice_laps,
    on=[
        "raceId",
        "driverId"
    ],
    how="outer"
)


# ------------------------------------------------------------
# PRACTICE SUMMARY FEATURES
# ------------------------------------------------------------

practice_features["practice_avg_position"] = (
    practice_features[
        [
            "fp1_position",
            "fp2_position",
            "fp3_position"
        ]
    ]
    .mean(axis=1)
)


practice_features["practice_best_position"] = (
    practice_features[
        [
            "fp1_position",
            "fp2_position",
            "fp3_position"
        ]
    ]
    .min(axis=1)
)


practice_features["practice_avg_lap_seconds"] = (
    practice_features[
        [
            "fp1_best_lap_seconds",
            "fp2_best_lap_seconds",
            "fp3_best_lap_seconds"
        ]
    ]
    .mean(axis=1)
)


# ------------------------------------------------------------
# MERGE INTO MAIN DATASET
# ------------------------------------------------------------

df = df.merge(
    practice_features,
    on=[
        "raceId",
        "driverId"
    ],
    how="left",
    validate="one_to_one"
)

# ---------------------------------------------------------
# Sprint features
# ---------------------------------------------------------

sprint = pd.read_csv(SPRINT_FILE, na_values="\\N")

# Convert numeric columns
sprint["raceId"] = pd.to_numeric(sprint["raceId"], errors="coerce")
sprint["driverId"] = pd.to_numeric(sprint["driverId"], errors="coerce")
sprint["grid"] = pd.to_numeric(sprint["grid"], errors="coerce")
sprint["position"] = pd.to_numeric(sprint["position"], errors="coerce")
sprint["points"] = pd.to_numeric(sprint["points"], errors="coerce")
sprint["laps"] = pd.to_numeric(sprint["laps"], errors="coerce")

# Rename Sprint-specific columns
sprint_features = sprint[
    [
        "raceId",
        "driverId",
        "grid",
        "position",
        "points",
        "laps",
    ]
].rename(
    columns={
        "grid": "sprint_grid",
        "position": "sprint_position",
        "points": "sprint_points",
        "laps": "sprint_laps",
    }
)

# DNF / unclassified Sprint result
sprint_features["sprint_is_dnf"] = (
    sprint_features["sprint_position"].isna().astype(int)
)

# Identify Sprint weekends
sprint_races = sprint_features["raceId"].unique()

sprint_features["has_sprint"] = 1

# Merge Sprint data into the model dataset
df = df.merge(
    sprint_features,
    on=["raceId", "driverId"],
    how="left",
)

# A race is a Sprint weekend even if a particular driver
# has no Sprint result
df["has_sprint"] = df["raceId"].isin(sprint_races).astype(int)

# ============================================================
# QUALIFYING FEATURES
# ============================================================

print("Creating qualifying features...")


# Gap to pole in Q3
#
# Smaller = faster
#
# This is available after qualifying,
# so it is allowed for our prediction point.

df["q3_gap_to_pole"] = (
    df["q3_seconds"]
    - df.groupby(
        ["year", "round"]
    )["q3_seconds"].transform("min")
)


# Q2/Q3 participation
df["made_q2"] = (
    df["q2_seconds"]
    .notna()
    .astype(int)
)

df["made_q3"] = (
    df["q3_seconds"]
    .notna()
    .astype(int)
)

# ============================================================
# TEAMMATE-RELATIVE FEATURES
# ============================================================

print("Creating teammate-relative features...")


def teammate_difference(data, column):
    """
    Calculates how far ahead/behind a driver is relative
    to the other driver(s) from the same constructor
    in the same race.

    Positive = driver is ahead of teammate(s)
    Negative = driver is behind teammate(s)

    Example:
        Driver:   P4
        Teammate: P10

        10 - 4 = +6
    """

    group = data.groupby(
        ["raceId", "constructorId"]
    )[column]

    teammate_sum = group.transform("sum")
    teammate_count = group.transform("count")

    return (
        (teammate_sum - data[column])
        / (teammate_count - 1)
    )


# ------------------------------------------------------------
# Qualifying vs teammate
# ------------------------------------------------------------

df["qualifying_vs_teammate"] = teammate_difference(
    df,
    "qualifying_position"
)


# ------------------------------------------------------------
# Grid vs teammate
# ------------------------------------------------------------

df["grid_vs_teammate"] = teammate_difference(
    df,
    "grid"
)


# ------------------------------------------------------------
# Practice vs teammate
# ------------------------------------------------------------

df["practice_vs_teammate"] = teammate_difference(
    df,
    "practice_avg_position"
)


# ------------------------------------------------------------
# Sprint vs teammate
# ------------------------------------------------------------

df["sprint_vs_teammate"] = teammate_difference(
    df,
    "sprint_position"
)

# ============================================================
# CIRCUIT HISTORY
# ============================================================

print("Creating circuit history features...")


circuit_history = (
    df
    .sort_values(
        ["driverId", "circuitId", "year", "round"]
    )
    .groupby(
        ["driverId", "circuitId"],
        group_keys=False
    )
)


df["driver_circuit_avg_finish"] = (
    circuit_history["finish_position"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(min_periods=1)
        .mean()
    )
)


df["driver_circuit_races"] = (
    circuit_history["finish_position"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .count()
    )
)


# ============================================================
# SEASON PROGRESS
# ============================================================

# ============================================================
# SEASON PROGRESS
# ============================================================

# Number of scheduled Grands Prix in each season.
# This avoids changing historical feature values when
# new races are added to the dataset.

SEASON_ROUNDS = {
    2018: 21,
    2019: 21,
    2020: 17,
    2021: 22,
    2022: 22,
    2023: 22,
    2024: 24,
    2025: 24,
    2026: 24,
}

df["season_total_rounds"] = df["year"].map(SEASON_ROUNDS)

if df["season_total_rounds"].isna().any():
    missing_years = sorted(
        df.loc[
            df["season_total_rounds"].isna(),
            "year"
        ].unique()
    )

    raise ValueError(
        f"Missing season round count for years: {missing_years}"
    )

df["season_progress"] = (
    df["round"] / df["season_total_rounds"]
)

df = df.drop(columns=["season_total_rounds"])

# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "_podium",
        "_win"
    ]
)


# ============================================================
# DEFINE TARGET COLUMNS
# ============================================================

target_columns = [
    "finish_position",
    "positionOrder",
    "points",
    "laps",
    "is_classified",
    "is_dnf"
]


# ============================================================
# DEFINE FEATURE COLUMNS
# ============================================================

feature_columns = [
    # ========================================================
    # IDENTIFIERS
    # ========================================================

    "raceId",
    "year",
    "round",
    "circuitId",

    "driverId",
    "constructorId",


    # ========================================================
    # PRACTICE
    # ========================================================

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


    # ========================================================
    # QUALIFYING
    # ========================================================

    "qualifying_position",

    "q1_seconds",
    "q2_seconds",
    "q3_seconds",

    "q3_gap_to_pole",

    "made_q2",
    "made_q3",


    # ========================================================
    # GRID
    # ========================================================

    "grid",


    # ========================================================
    # DRIVER FORM
    # ========================================================

    "driver_avg_finish_L3",
    "driver_avg_finish_L5",
    "driver_avg_finish_L10",

    "driver_points_L5",

    "driver_dnf_rate_L10",

    "driver_podium_rate_L10",
    "driver_win_rate_L10",


    # ========================================================
    # CONSTRUCTOR FORM
    # ========================================================

    "constructor_avg_finish_L5",
    "constructor_points_L5",
    "constructor_dnf_rate_L10",


    # ========================================================
    # CHAMPIONSHIP
    # ========================================================

    "driver_points_before_race",
    "driver_championship_position",

    "constructor_points_before_race",
    "constructor_championship_position",


    # ========================================================
    # CIRCUIT HISTORY
    # ========================================================

    "driver_circuit_avg_finish",
    "driver_circuit_races",


    # ========================================================
    # SEASON
    # ========================================================

    "season_progress",


    # ========================================================
    # TARGET / OUTCOME
    # ========================================================

    "finish_position",
    "positionOrder",
    "points",
    "laps",

    "is_classified",
    "is_dnf",

    #SPRINT FEATURES
    "sprint_grid",
    "sprint_position",
    "sprint_points",
    "sprint_laps",
    "sprint_is_dnf",
    "has_sprint",
    # ========================================================
    # TEAMMATE-RELATIVE
    # ========================================================

    "qualifying_vs_teammate",
    "grid_vs_teammate",
    "practice_vs_teammate",
    "sprint_vs_teammate",
]
# ============================================================
# CHECK COLUMNS
# ============================================================

missing = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing:
    print("\nERROR: Missing columns:")

    for col in missing:
        print(" -", col)

    raise ValueError(
        "Feature construction failed."
    )


features = df[feature_columns].copy()


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n================================")
print("FEATURE ENGINEERING COMPLETE")
print("================================")

print(
    f"Rows: {len(features):,}"
)

print(
    f"Columns: {len(features.columns)}"
)

print(
    f"Years: "
    f"{features['year'].min()} - "
    f"{features['year'].max()}"
)

print(
    f"Races: "
    f"{features['raceId'].nunique()}"
)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print("\nFeature columns:")

for col in features.columns:
    print(" -", col)