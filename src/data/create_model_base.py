from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "driver_race_base.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "f1_model_base.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading driver_race_base.csv...")

df = pd.read_csv(
    INPUT_FILE,
    na_values=["\\N"]
)

print(f"Loaded {len(df):,} rows")
print(f"Loaded {len(df.columns)} columns")


# ============================================================
# 1. REMOVE DUPLICATES
# ============================================================

key = ["raceId", "driverId"]

duplicates = df.duplicated(subset=key).sum()

print(f"\nDuplicate race/driver rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates(
        subset=key,
        keep="first"
    )


# ============================================================
# 2. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "resultId",
    "raceId",
    "driverId",
    "constructorId",

    "grid",
    "position",
    "positionOrder",

    "points",
    "laps",

    "milliseconds",
    "fastestLap",
    "rank",
    "fastestLapSpeed",

    "statusId",

    "year",
    "round",
    "circuitId",

    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# 3. CREATE MODEL-FRIENDLY TARGET NAMES
# ============================================================

# Your dataset calls finishing position "position".
# For the ML table we will call it "finish_position".

df["finish_position"] = df["position"]


# Your processing pipeline already created is_classified
# and is_unclassified.

df["is_dnf"] = df["is_unclassified"].astype(int)


# ============================================================
# 4. CHECK FINISHING POSITIONS
# ============================================================

invalid_positions = df[
    df["finish_position"].notna()
    & (
        (df["finish_position"] < 1)
        | (df["finish_position"] > 30)
    )
]

print(
    f"\nInvalid finishing positions: "
    f"{len(invalid_positions)}"
)

if len(invalid_positions) > 0:

    print(
        invalid_positions[
            [
                "raceId",
                "driverId",
                "finish_position"
            ]
        ]
    )


# ============================================================
# 5. CHECK QUALIFYING POSITIONS
# ============================================================

invalid_qualifying = df[
    df["qualifying_position"].notna()
    & (
        (df["qualifying_position"] < 1)
        | (df["qualifying_position"] > 30)
    )
]

print(
    f"Invalid qualifying positions: "
    f"{len(invalid_qualifying)}"
)


# ============================================================
# 6. CHECK GRID POSITIONS
# ============================================================

invalid_grid = df[
    df["grid"].notna()
    & (
        (df["grid"] < 0)
        | (df["grid"] > 30)
    )
]

print(
    f"Invalid grid positions: "
    f"{len(invalid_grid)}"
)


# ============================================================
# 7. SELECT CANONICAL MODEL COLUMNS
# ============================================================

canonical_columns = [

    # Race information
    "raceId",
    "year",
    "round",
    "circuitId",

    # Driver / constructor
    "driverId",
    "constructorId",

    # Qualifying information
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",

    # Starting position
    "grid",

    # Race outcome
    "finish_position",
    "positionOrder",

    # Race statistics
    "points",
    "laps",

    # Classification
    "is_classified",
    "is_dnf",

    # Status
    "statusId",
]


# Make sure every expected column exists
missing_columns = [
    column
    for column in canonical_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR!")
    print("Missing columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise ValueError(
        "The input dataset does not contain "
        "all required modeling columns."
    )


# Select columns
model_df = df[canonical_columns].copy()


# ============================================================
# 8. SORT CHRONOLOGICALLY
# ============================================================

model_df = model_df.sort_values(
    by=[
        "year",
        "round",
        "raceId",
        "driverId"
    ]
).reset_index(drop=True)


# ============================================================
# 9. FINAL DUPLICATE CHECK
# ============================================================

duplicates_after = model_df.duplicated(
    subset=["raceId", "driverId"]
).sum()

print(
    f"\nDuplicates after cleaning: "
    f"{duplicates_after}"
)


# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print("\n==============================")
print("MODEL BASE SUMMARY")
print("==============================")

print(
    f"Rows: {len(model_df):,}"
)

print(
    f"Columns: {len(model_df.columns)}"
)

print(
    f"Races: {model_df['raceId'].nunique():,}"
)

print(
    f"Drivers: {model_df['driverId'].nunique():,}"
)

print(
    f"Constructors: {model_df['constructorId'].nunique():,}"
)

print(
    f"Years: "
    f"{model_df['year'].min()} - "
    f"{model_df['year'].max()}"
)

print(
    f"DNFs: {model_df['is_dnf'].sum():,}"
)


# ============================================================
# 11. CHECK ROUND 16
# ============================================================

round_16 = model_df[
    (model_df["year"] == 2026)
    & (model_df["round"] == 16)
]

print("\n==============================")
print("2026 ROUND 16 CHECK")
print("==============================")

print(
    f"Round 16 rows: {len(round_16)}"
)

print(
    f"Round 16 drivers: "
    f"{round_16['driverId'].nunique()}"
)


# ============================================================
# 12. SAVE
# ============================================================

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("SUCCESS")
print("==============================")

print(
    f"Saved:\n{OUTPUT_FILE}"
)

print("\nFinal columns:")

for column in model_df.columns:
    print(f"  - {column}")