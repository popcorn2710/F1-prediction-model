from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/archive")


def audit_csv(file_path):
    df = pd.read_csv(file_path)

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  None")
    else:
        print(missing)

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files.")

    for file_path in csv_files:
        try:
            audit_csv(file_path)
        except Exception as e:
            print(f"\nERROR reading {file_path.name}: {e}")


if __name__ == "__main__":
    main()