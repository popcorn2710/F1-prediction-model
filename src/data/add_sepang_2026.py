from pathlib import Path
import requests
import pandas as pd
import shutil

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/raw/archive")

YEAR = 2026
ROUND = 16
RACE_ID = 1184

API_BASE = "https://api.jolpi.ca/ergast/f1"

HEADERS = {
    "User-Agent": "F1PredictionProject/1.0"
}


# ============================================================
# HELPERS
# ============================================================

def get_api(endpoint):
    url = f"{API_BASE}/{endpoint}"
    print(f"Downloading: {url}")

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return response.json()


def backup_file(path):
    backup = path.with_suffix(path.suffix + ".bak")

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")


# ============================================================
# LOAD EXISTING DATA
# ============================================================

print("=" * 70)
print("ADDING 2026 ROUND 16 - BAHRAIN GP / SEPANG")
print("=" * 70)

drivers = pd.read_csv(DATA_DIR / "drivers.csv")
constructors = pd.read_csv(DATA_DIR / "constructors.csv")
races = pd.read_csv(DATA_DIR / "races.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
qualifying = pd.read_csv(DATA_DIR / "qualifying.csv")
status = pd.read_csv(DATA_DIR / "status.csv")


# ============================================================
# CHECK WHETHER RACE ALREADY EXISTS
# ============================================================

existing_race = races[
    (races["year"] == YEAR) &
    (races["round"] == ROUND)
]

if not existing_race.empty:
    print("\nERROR: Round 16 already exists in races.csv")
    print(existing_race.to_string(index=False))
    raise SystemExit


# ============================================================
# DRIVER / CONSTRUCTOR LOOKUPS
# ============================================================

# API gives permanent driver identifiers such as:
# max_verstappen
# lewis_hamilton
#
# We use those ONLY to find the corresponding EXISTING
# driverId in our local drivers.csv.

driver_lookup = dict(
    zip(drivers["driverRef"], drivers["driverId"])
)

constructor_lookup = dict(
    zip(constructors["constructorRef"], constructors["constructorId"])
)


# ============================================================
# DOWNLOAD RACE RESULTS
# ============================================================

race_data = get_api(f"{YEAR}/{ROUND}/results/?limit=100")

race = race_data["MRData"]["RaceTable"]["Races"][0]

api_results = race["Results"]

print(f"\nRace: {race['raceName']}")
print(f"Drivers returned: {len(api_results)}")


# ============================================================
# DOWNLOAD QUALIFYING
# ============================================================

quali_data = get_api(f"{YEAR}/{ROUND}/qualifying/?limit=100")

quali_race = quali_data["MRData"]["RaceTable"]["Races"][0]

api_quali = quali_race["QualifyingResults"]

print(f"Qualifying rows: {len(api_quali)}")


# ============================================================
# VERIFY ALL DRIVERS / CONSTRUCTORS EXIST LOCALLY
# ============================================================

missing_drivers = []
missing_constructors = []

for row in api_results:

    driver_ref = row["Driver"]["driverId"]
    constructor_ref = row["Constructor"]["constructorId"]

    if driver_ref not in driver_lookup:
        missing_drivers.append(driver_ref)

    if constructor_ref not in constructor_lookup:
        missing_constructors.append(constructor_ref)


if missing_drivers:
    print("\nERROR: Missing drivers in local drivers.csv:")
    print(missing_drivers)
    raise SystemExit


if missing_constructors:
    print("\nERROR: Missing constructors in local constructors.csv:")
    print(missing_constructors)
    raise SystemExit


print("\n✓ All API drivers exist in local drivers.csv")
print("✓ All API constructors exist in local constructors.csv")


# ============================================================
# CREATE RACE ROW
# ============================================================

# Sepang already exists in your circuits.csv as circuitId = 2.

race_row = {
    "raceId": RACE_ID,
    "year": YEAR,
    "round": ROUND,
    "circuitId": 2,
    "name": race["raceName"],
    "date": race["date"],
    "time": race.get("time", r"\N"),
    "url": race.get("url", r"\N"),
    "fp1_date": r"\N",
    "fp1_time": r"\N",
    "fp2_date": r"\N",
    "fp2_time": r"\N",
    "fp3_date": r"\N",
    "fp3_time": r"\N",
    "quali_date": r"\N",
    "quali_time": r"\N",
    "sprint_date": r"\N",
    "sprint_time": r"\N",
}

new_race = pd.DataFrame([race_row])


# ============================================================
# CREATE RESULT ROWS
# ============================================================

next_result_id = int(results["resultId"].max()) + 1

new_results = []

for i, row in enumerate(api_results):

    driver_ref = row["Driver"]["driverId"]
    constructor_ref = row["Constructor"]["constructorId"]

    driver_id = driver_lookup[driver_ref]
    constructor_id = constructor_lookup[constructor_ref]

    # API status is text. Our local status table uses statusId.
    status_text = row["status"]

    matching_status = status[
        status["status"].str.lower() == status_text.lower()
    ]

    if matching_status.empty:
        print(f"\nERROR: Status not found locally: {status_text}")
        raise SystemExit

    status_id = int(matching_status.iloc[0]["statusId"])

    fastest_lap = row.get("FastestLap", {})

    new_results.append({
        "resultId": next_result_id + i,
        "raceId": RACE_ID,
        "driverId": driver_id,
        "constructorId": constructor_id,
        "number": row.get("number", r"\N"),
        "grid": int(row.get("grid", 0)),
        "position": row.get("position", r"\N"),
        "positionText": row.get("positionText", r"\N"),
        "positionOrder": int(row.get("position", 0))
            if str(row.get("position", "")).isdigit()
            else 99,
        "points": float(row.get("points", 0)),
        "laps": int(row.get("laps", 0)),
        "time": (
            row.get("Time", {}).get("time", r"\N")
            if isinstance(row.get("Time"), dict)
            else r"\N"
        ),
        "milliseconds": (
            row.get("Time", {}).get("millis", r"\N")
            if isinstance(row.get("Time"), dict)
            else r"\N"
        ),
        "fastestLap": fastest_lap.get("lap", r"\N"),
        "rank": fastest_lap.get("AverageSpeed", {}).get("speed", r"\N")
            if isinstance(fastest_lap, dict)
            else r"\N",
        "fastestLapTime": (
            fastest_lap.get("Time", {}).get("time", r"\N")
            if isinstance(fastest_lap.get("Time"), dict)
            else r"\N"
        ),
        "fastestLapSpeed": (
            fastest_lap.get("AverageSpeed", {}).get("speed", r"\N")
            if isinstance(fastest_lap.get("AverageSpeed"), dict)
            else r"\N"
        ),
        "statusId": status_id,
    })


new_results = pd.DataFrame(new_results)


# ============================================================
# CREATE QUALIFYING ROWS
# ============================================================

next_qualify_id = int(qualifying["qualifyId"].max()) + 1

new_qualifying = []

for i, row in enumerate(api_quali):

    driver_ref = row["Driver"]["driverId"]
    constructor_ref = row["Constructor"]["constructorId"]

    driver_id = driver_lookup[driver_ref]
    constructor_id = constructor_lookup[constructor_ref]

    new_qualifying.append({
        "qualifyId": next_qualify_id + i,
        "raceId": RACE_ID,
        "driverId": driver_id,
        "constructorId": constructor_id,
        "number": row.get("number", r"\N"),
        "position": int(row["position"]),
        "q1": row.get("Q1", r"\N"),
        "q2": row.get("Q2", r"\N"),
        "q3": row.get("Q3", r"\N"),
    })


new_qualifying = pd.DataFrame(new_qualifying)


# ============================================================
# SHOW BEFORE WRITING
# ============================================================

print("\n" + "=" * 70)
print("PREVIEW")
print("=" * 70)

print("\nRACE:")
print(new_race.to_string(index=False))

print("\nRESULTS:")
print(
    new_results[
        ["resultId", "driverId", "constructorId",
         "grid", "position", "positionText", "points"]
    ].to_string(index=False)
)

print("\nQUALIFYING:")
print(
    new_qualifying[
        ["qualifyId", "driverId",
         "constructorId", "position", "q1", "q2", "q3"]
    ].to_string(index=False)
)


# ============================================================
# SAFETY CHECKS
# ============================================================

assert len(new_results) == len(api_results)
assert len(new_qualifying) == len(api_quali)

assert not new_results["resultId"].isin(
    results["resultId"]
).any()

assert not new_qualifying["qualifyId"].isin(
    qualifying["qualifyId"]
).any()

assert new_results["driverId"].isin(
    drivers["driverId"]
).all()

assert new_results["constructorId"].isin(
    constructors["constructorId"]
).all()

print("\n✓ Result IDs are unique")
print("✓ Qualifying IDs are unique")
print("✓ Driver IDs exist")
print("✓ Constructor IDs exist")


# ============================================================
# BACKUP
# ============================================================

for filename in [
    "races.csv",
    "results.csv",
    "qualifying.csv",
]:
    backup_file(DATA_DIR / filename)


# ============================================================
# APPEND
# ============================================================

new_race.to_csv(
    DATA_DIR / "races.csv",
    mode="a",
    header=False,
    index=False
)

new_results.to_csv(
    DATA_DIR / "results.csv",
    mode="a",
    header=False,
    index=False
)

new_qualifying.to_csv(
    DATA_DIR / "qualifying.csv",
    mode="a",
    header=False,
    index=False
)


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"Added race: {RACE_ID}")
print(f"Added results: {len(new_results)}")
print(f"Added qualifying rows: {len(new_qualifying)}")