from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw/archive")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_csv(filename):
    """
    Load a raw CSV and convert \\N into missing values.
    """
    path = RAW_DIR / filename

    print(f"Loading {filename}...")

    return pd.read_csv(
        path,
        na_values=r"\N",
        keep_default_na=True
    )


def save_csv(df, filename):
    """
    Save a processed dataframe.
    """
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)

    print(f"Saved → {path}")


# ============================================================
# TIME CONVERSION
# ============================================================

def lap_time_to_seconds(value):
    """
    Convert F1 lap time:

        1:35.130

    into:

        95.130 seconds
    """

    if pd.isna(value):
        return None

    value = str(value)

    try:
        parts = value.split(":")

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])

            return minutes * 60 + seconds

        return float(value)

    except (ValueError, TypeError):
        return None


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("F1 DATA PROCESSING")
print("=" * 70)

races = load_csv("races.csv")
results = load_csv("results.csv")
qualifying = load_csv("qualifying.csv")
drivers = load_csv("drivers.csv")
constructors = load_csv("constructors.csv")
circuits = load_csv("circuits.csv")
status = load_csv("status.csv")
driver_standings = load_csv("driver_standings.csv")
constructor_standings = load_csv("constructor_standings.csv")


# ============================================================
# RACES
# ============================================================

print("\nProcessing races...")

races["raceId"] = pd.to_numeric(
    races["raceId"],
    errors="coerce"
).astype("Int64")

races["year"] = pd.to_numeric(
    races["year"],
    errors="coerce"
).astype("Int64")

races["round"] = pd.to_numeric(
    races["round"],
    errors="coerce"
).astype("Int64")

races["circuitId"] = pd.to_numeric(
    races["circuitId"],
    errors="coerce"
).astype("Int64")

races["date"] = pd.to_datetime(
    races["date"],
    errors="coerce"
)

races = races.sort_values(
    ["year", "round"]
).reset_index(drop=True)


# ============================================================
# RESULTS
# ============================================================

print("Processing results...")

numeric_result_columns = [
    "resultId",
    "raceId",
    "driverId",
    "constructorId",
    "grid",
    "position",
    "positionOrder",
    "points",
    "laps",
    "statusId"
]

for column in numeric_result_columns:
    results[column] = pd.to_numeric(
        results[column],
        errors="coerce"
    )


# DNF / unclassified drivers have no numeric position
results["is_classified"] = results["position"].notna()

results["is_unclassified"] = results["position"].isna()

results["points"] = results["points"].fillna(0)

results["grid"] = results["grid"].fillna(0)

results["laps"] = results["laps"].fillna(0)


# Convert race time fields
results["milliseconds"] = pd.to_numeric(
    results["milliseconds"],
    errors="coerce"
)

results["fastestLap"] = pd.to_numeric(
    results["fastestLap"],
    errors="coerce"
)

results["rank"] = pd.to_numeric(
    results["rank"],
    errors="coerce"
)

results["fastestLapSpeed"] = pd.to_numeric(
    results["fastestLapSpeed"],
    errors="coerce"
)

# Sort chronologically
results = results.sort_values(
    ["raceId", "positionOrder"]
).reset_index(drop=True)


# ============================================================
# QUALIFYING
# ============================================================

print("Processing qualifying...")

numeric_quali_columns = [
    "qualifyId",
    "raceId",
    "driverId",
    "constructorId",
    "number",
    "position"
]

for column in numeric_quali_columns:
    qualifying[column] = pd.to_numeric(
        qualifying[column],
        errors="coerce"
    )


# Convert Q1/Q2/Q3 times into seconds

for column in ["q1", "q2", "q3"]:
    qualifying[f"{column}_seconds"] = (
        qualifying[column]
        .apply(lap_time_to_seconds)
    )


# ============================================================
# DRIVER STANDINGS
# ============================================================

print("Processing driver standings...")

for column in [
    "driverStandingsId",
    "raceId",
    "driverId",
    "position",
    "wins"
]:
    driver_standings[column] = pd.to_numeric(
        driver_standings[column],
        errors="coerce"
    )

driver_standings["points"] = pd.to_numeric(
    driver_standings["points"],
    errors="coerce"
)


# ============================================================
# CONSTRUCTOR STANDINGS
# ============================================================

print("Processing constructor standings...")

for column in [
    "constructorStandingsId",
    "raceId",
    "constructorId",
    "position",
    "wins"
]:
    constructor_standings[column] = pd.to_numeric(
        constructor_standings[column],
        errors="coerce"
    )

constructor_standings["points"] = pd.to_numeric(
    constructor_standings["points"],
    errors="coerce"
)


# ============================================================
# CREATE DRIVER-RACE BASE TABLE
# ============================================================

print("\nCreating driver-race base table...")

driver_race = results.merge(
    drivers[
        [
            "driverId",
            "driverRef",
            "forename",
            "surname"
        ]
    ],
    on="driverId",
    how="left"
)

driver_race = driver_race.merge(
    constructors[
        [
            "constructorId",
            "constructorRef",
            "name"
        ]
    ],
    on="constructorId",
    how="left"
)

driver_race = driver_race.merge(
    races[
        [
            "raceId",
            "year",
            "round",
            "circuitId",
            "name",
            "date"
        ]
    ],
    on="raceId",
    how="left",
    suffixes=("", "_race")
)

driver_race = driver_race.merge(
    circuits[
        [
            "circuitId",
            "circuitRef",
            "name"
        ]
    ],
    on="circuitId",
    how="left",
    suffixes=("", "_circuit")
)


# ============================================================
# ADD QUALIFYING INFORMATION
# ============================================================

quali_features = qualifying[
    [
        "raceId",
        "driverId",
        "position",
        "q1_seconds",
        "q2_seconds",
        "q3_seconds"
    ]
].rename(
    columns={
        "position": "qualifying_position"
    }
)

driver_race = driver_race.merge(
    quali_features,
    on=["raceId", "driverId"],
    how="left"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)


# Duplicate primary keys

print("\nDuplicate checks:")

print(
    "results:",
    results["resultId"].duplicated().sum()
)

print(
    "qualifying:",
    qualifying["qualifyId"].duplicated().sum()
)

print(
    "races:",
    races["raceId"].duplicated().sum()
)


# Foreign key checks

print("\nForeign key checks:")

invalid_result_races = ~results["raceId"].isin(
    races["raceId"]
)

invalid_result_drivers = ~results["driverId"].isin(
    drivers["driverId"]
)

invalid_result_constructors = ~results["constructorId"].isin(
    constructors["constructorId"]
)

print(
    "Invalid result race IDs:",
    invalid_result_races.sum()
)

print(
    "Invalid result driver IDs:",
    invalid_result_drivers.sum()
)

print(
    "Invalid result constructor IDs:",
    invalid_result_constructors.sum()
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)

save_csv(races, "races_clean.csv")
save_csv(results, "results_clean.csv")
save_csv(qualifying, "qualifying_clean.csv")
save_csv(driver_standings, "driver_standings_clean.csv")
save_csv(
    constructor_standings,
    "constructor_standings_clean.csv"
)
save_csv(driver_race, "driver_race_base.csv")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(f"Races:              {len(races):,}")
print(f"Results:            {len(results):,}")
print(f"Qualifying:         {len(qualifying):,}")
print(f"Driver-race rows:   {len(driver_race):,}")

print("\nLatest race:")
print(
    races.sort_values("date")
    .tail(1)[
        ["raceId", "year", "round", "name", "date"]
    ]
    .to_string(index=False)
)