from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/archive")


def load_data():
    """Load all tables needed for validation."""

    data = {}

    files = [
        "races.csv",
        "results.csv",
        "drivers.csv",
        "constructors.csv",
        "circuits.csv",
        "status.csv",
        "qualifying.csv",
        "sprint_results.csv",
    ]

    for file in files:
        data[file.replace(".csv", "")] = pd.read_csv(DATA_DIR / file)

    return data


def check_foreign_key(df, column, reference_df, reference_column, table_name):
    """Check whether every foreign-key value exists in its reference table."""

    invalid = ~df[column].isin(reference_df[reference_column])

    count = invalid.sum()

    if count == 0:
        print(f"✓ {table_name}.{column} → valid")
    else:
        print(
            f"✗ {table_name}.{column} → "
            f"{count} invalid values"
        )

        print("  Examples:")
        print(df.loc[invalid, column].drop_duplicates().head(10).tolist())


def check_duplicates(df, primary_key, table_name):
    """Check whether a primary key contains duplicates."""

    duplicates = df[primary_key].duplicated().sum()

    if duplicates == 0:
        print(f"✓ {table_name}.{primary_key} → unique")
    else:
        print(
            f"✗ {table_name}.{primary_key} → "
            f"{duplicates} duplicates"
        )


def check_non_negative(df, column, table_name):
    """Check that numerical values are not negative."""

    invalid = df[column] < 0
    count = invalid.sum()

    if count == 0:
        print(f"✓ {table_name}.{column} → no negative values")
    else:
        print(
            f"✗ {table_name}.{column} → "
            f"{count} negative values"
        )


def main():

    print("=" * 70)
    print("F1 DATA VALIDATION")
    print("=" * 70)

    data = load_data()

    races = data["races"]
    results = data["results"]
    drivers = data["drivers"]
    constructors = data["constructors"]
    circuits = data["circuits"]
    status = data["status"]
    qualifying = data["qualifying"]
    sprint_results = data["sprint_results"]

    # ---------------------------------------------------------
    # PRIMARY KEY CHECKS
    # ---------------------------------------------------------

    print("\nPRIMARY KEY CHECKS")
    print("-" * 70)

    check_duplicates(races, "raceId", "races")
    check_duplicates(results, "resultId", "results")
    check_duplicates(drivers, "driverId", "drivers")
    check_duplicates(constructors, "constructorId", "constructors")
    check_duplicates(circuits, "circuitId", "circuits")
    check_duplicates(status, "statusId", "status")
    check_duplicates(qualifying, "qualifyId", "qualifying")

    # Sprint uses resultId as its primary key in this dataset.
    check_duplicates(
        sprint_results,
        "resultId",
        "sprint_results"
    )

    # ---------------------------------------------------------
    # FOREIGN KEY CHECKS
    # ---------------------------------------------------------

    print("\nFOREIGN KEY CHECKS")
    print("-" * 70)

    # races → circuits
    check_foreign_key(
        races,
        "circuitId",
        circuits,
        "circuitId",
        "races"
    )

    # results → races
    check_foreign_key(
        results,
        "raceId",
        races,
        "raceId",
        "results"
    )

    # results → drivers
    check_foreign_key(
        results,
        "driverId",
        drivers,
        "driverId",
        "results"
    )

    # results → constructors
    check_foreign_key(
        results,
        "constructorId",
        constructors,
        "constructorId",
        "results"
    )

    # results → status
    check_foreign_key(
        results,
        "statusId",
        status,
        "statusId",
        "results"
    )

    # qualifying → races
    check_foreign_key(
        qualifying,
        "raceId",
        races,
        "raceId",
        "qualifying"
    )

    # qualifying → drivers
    check_foreign_key(
        qualifying,
        "driverId",
        drivers,
        "driverId",
        "qualifying"
    )

    # qualifying → constructors
    check_foreign_key(
        qualifying,
        "constructorId",
        constructors,
        "constructorId",
        "qualifying"
    )

    # sprint → races
    check_foreign_key(
        sprint_results,
        "raceId",
        races,
        "raceId",
        "sprint_results"
    )

    # sprint → drivers
    check_foreign_key(
        sprint_results,
        "driverId",
        drivers,
        "driverId",
        "sprint_results"
    )

    # sprint → constructors
    check_foreign_key(
        sprint_results,
        "constructorId",
        constructors,
        "constructorId",
        "sprint_results"
    )

    # sprint → status
    check_foreign_key(
        sprint_results,
        "statusId",
        status,
        "statusId",
        "sprint_results"
    )

    # ---------------------------------------------------------
    # NUMERICAL SANITY CHECKS
    # ---------------------------------------------------------

    print("\nNUMERICAL SANITY CHECKS")
    print("-" * 70)

    check_non_negative(
        results,
        "grid",
        "results"
    )

    check_non_negative(
        results,
        "laps",
        "results"
    )

    check_non_negative(
        results,
        "points",
        "results"
    )

    check_non_negative(
        qualifying,
        "position",
        "qualifying"
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()