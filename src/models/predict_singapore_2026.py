from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "f1_features.csv"
MODEL_BASE_FILE = PROJECT_ROOT / "data" / "processed" / "f1_model_base.csv"
DRIVER_RACE_FILE = PROJECT_ROOT / "data" / "processed" / "driver_race_base.csv"
OUTPUT_DIR = PROJECT_ROOT / "src" / "models" / "singapore_2026"
OUTPUT_FILE = OUTPUT_DIR / "singapore_2026_prediction.csv"

TARGET_QUAL = "qualifying_position"
TARGET_RACE = "finish_position"

SINGAPORE_ROUND = 17
SINGAPORE_CIRCUIT_ID = 15  # Marina Bay Street Circuit in the project dataset

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


def ridge_model(alpha=10.0):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def rank_within_race(df, raw_col, output_col):
    out = df.copy()
    out[output_col] = (
        out["raceId"].groupby(out["raceId"])
        if False else
        out.groupby("raceId")[raw_col].rank(method="first", ascending=True).astype(int)
    )
    return out


def add_current_features(model_base, current_drivers):
    """Rebuild the 17 pre-qualifying features for the Singapore prediction point.

    The current prediction point is after 2026 Round 16 and before Singapore
    FP1/qualifying. Therefore all calculations use completed races through
    Round 16 only.
    """
    df = model_base.copy()
    df = df[df["year"] >= 2018].copy()
    df = df.sort_values(["year", "round", "raceId", "driverId"])

    # Ensure numeric values.
    numeric = [
        "year", "round", "raceId", "driverId", "constructorId",
        "finish_position", "points", "is_dnf", "circuitId"
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Only completed races before Singapore.
    completed = df[(df["year"] < 2026) | ((df["year"] == 2026) & (df["round"] < SINGAPORE_ROUND))].copy()

    # Current driver/constructor identity from latest completed 2026 race.
    latest = completed[(completed["year"] == 2026) & (completed["round"] == SINGAPORE_ROUND - 1)].copy()
    latest = latest.drop_duplicates("driverId")
    latest = latest[latest["driverId"].isin(current_drivers)].copy()

    if latest.empty:
        raise RuntimeError("Could not find current 2026 drivers from Round 16.")

    # Driver recent form.
    driver_rows = []
    for _, row in latest.iterrows():
        driver_id = row["driverId"]
        hist = completed[completed["driverId"] == driver_id].sort_values(["year", "round"])
        finishes = hist["finish_position"].dropna()
        points = hist["points"].fillna(0)
        dnf = hist["is_dnf"].fillna(0)

        driver_rows.append({
            "driverId": driver_id,
            "constructorId": row["constructorId"],
            "driver_avg_finish_L3": finishes.tail(3).mean(),
            "driver_avg_finish_L5": finishes.tail(5).mean(),
            "driver_avg_finish_L10": finishes.tail(10).mean(),
            "driver_points_L5": points.tail(5).sum(),
            "driver_dnf_rate_L10": dnf.tail(10).mean(),
            "driver_podium_rate_L10": (finishes.tail(10) <= 3).mean(),
            "driver_win_rate_L10": (finishes.tail(10) == 1).mean(),
        })

    current = pd.DataFrame(driver_rows)

    # Constructor form: first aggregate to constructor/race exactly like
    # build_features.py, then use the previous five/ten races.
    constructor_race = (
        completed.groupby(["constructorId", "year", "round"], as_index=False)
        .agg(
            constructor_avg_finish_race=("finish_position", "mean"),
            constructor_points_race=("points", "sum"),
            constructor_dnf_race=("is_dnf", "mean"),
        )
        .sort_values(["constructorId", "year", "round"])
    )

    constructor_rows = []
    for cid in current["constructorId"].dropna().unique():
        hist = constructor_race[constructor_race["constructorId"] == cid].sort_values(["year", "round"])
        constructor_rows.append({
            "constructorId": cid,
            "constructor_avg_finish_L5": hist["constructor_avg_finish_race"].tail(5).mean(),
            "constructor_points_L5": hist["constructor_points_race"].tail(5).sum(),
            "constructor_dnf_rate_L10": hist["constructor_dnf_race"].tail(10).mean(),
        })
    current = current.merge(pd.DataFrame(constructor_rows), on="constructorId", how="left")

    # Championship state: same definition as build_features.py (main GP points only).
    driver_points = completed.groupby(["year", "driverId"])["points"].sum().reset_index(name="points_total")
    current_driver_points = (
        driver_points[driver_points["year"] == 2026]
        .set_index("driverId")["points_total"]
        .to_dict()
    )
    current["driver_points_before_race"] = current["driverId"].map(current_driver_points).fillna(0)
    current["driver_championship_position"] = current["driver_points_before_race"].rank(method="min", ascending=False)

    constructor_points = completed.groupby(["year", "constructorId"])["points"].sum().reset_index(name="points_total")
    current_constructor_points = (
        constructor_points[constructor_points["year"] == 2026]
        .set_index("constructorId")["points_total"]
        .to_dict()
    )
    current["constructor_points_before_race"] = current["constructorId"].map(current_constructor_points).fillna(0)
    current["constructor_championship_position"] = current["constructor_points_before_race"].rank(method="min", ascending=False)

    # Singapore circuit history. Circuit ID 15 = Marina Bay in the project data.
    circuit_hist = completed[completed["circuitId"] == SINGAPORE_CIRCUIT_ID]
    circuit_rows = []
    for driver_id in current["driverId"]:
        hist = circuit_hist[circuit_hist["driverId"] == driver_id].sort_values(["year", "round"])
        finishes = hist["finish_position"].dropna()
        circuit_rows.append({
            "driverId": driver_id,
            "driver_circuit_avg_finish": finishes.mean() if len(finishes) else np.nan,
            "driver_circuit_races": float(len(finishes)),
        })
    current = current.merge(pd.DataFrame(circuit_rows), on="driverId", how="left")

    # Singapore is Round 17 of a 24-round season.
    current["season_progress"] = SINGAPORE_ROUND / 24.0

    return current


def build_oof_qual_predictions(train_df):
    """Create historical out-of-fold qualifying predictions for Stage 2."""
    train = train_df.dropna(subset=[TARGET_RACE, TARGET_QUAL]).copy()
    train["predicted_qualifying_position"] = np.nan

    years = sorted(train["year"].unique())
    for test_year in years:
        if test_year <= 2018:
            continue
        inner_train = train[train["year"] < test_year].dropna(subset=[TARGET_QUAL])
        mask = train["year"] == test_year
        inner_test = train.loc[mask].copy()
        if inner_train.empty or inner_test.empty:
            continue

        model = ridge_model()
        model.fit(inner_train[PRE_QUAL_FEATURES], inner_train[TARGET_QUAL])
        raw = np.maximum(model.predict(inner_test[PRE_QUAL_FEATURES]), 1.0)
        inner_test["_raw_q"] = raw
        inner_test = rank_within_race(inner_test, "_raw_q", "predicted_qualifying_position")
        train.loc[inner_test.index, "predicted_qualifying_position"] = inner_test["predicted_qualifying_position"]

    return train.dropna(subset=["predicted_qualifying_position"])


def load_driver_names():
    if DRIVER_RACE_FILE.exists():
        d = pd.read_csv(DRIVER_RACE_FILE)
        cols = [c for c in ["driverId", "forename", "surname"] if c in d.columns]
        d = d[cols].drop_duplicates("driverId")
        d["driver_name"] = d["forename"].fillna("") + " " + d["surname"].fillna("")
        return d[["driverId", "driver_name"]]
    return None


def main():
    print("=" * 78)
    print("F1 SINGAPORE 2026 PRE-WEEKEND PREDICTION")
    print("=" * 78)
    print("Round: 17")
    print("Race: Singapore Grand Prix")
    print("Circuit: Marina Bay Street Circuit")
    print("Prediction point: AFTER 2026 Round 16, BEFORE Singapore FP1")
    print()
    print("NOTE: This is a pre-weekend forecast. Singapore FP1, Sprint, and GP")
    print("qualifying have not been used. The model can be rerun after qualifying")
    print("with actual qualifying information for the final race forecast.")

    features = pd.read_csv(FEATURE_FILE)
    model_base = pd.read_csv(MODEL_BASE_FILE)

    features = features[features["year"] >= 2018].copy()
    features[TARGET_QUAL] = pd.to_numeric(features[TARGET_QUAL], errors="coerce")
    features[TARGET_RACE] = pd.to_numeric(features[TARGET_RACE], errors="coerce")

    latest = model_base[(model_base["year"] == 2026) & (model_base["round"] == 16)].copy()
    current_drivers = sorted(latest["driverId"].dropna().unique())

    if len(current_drivers) == 0:
        raise RuntimeError("No 2026 Round 16 drivers found.")

    current = add_current_features(model_base, current_drivers)

    # ------------------------------------------------------------
    # STAGE 1: qualifying prediction
    # ------------------------------------------------------------
    qual_train = features.dropna(subset=[TARGET_QUAL]).copy()
    qual_model = ridge_model()
    qual_model.fit(qual_train[PRE_QUAL_FEATURES], qual_train[TARGET_QUAL])

    current["predicted_qualifying_raw"] = np.maximum(
        qual_model.predict(current[PRE_QUAL_FEATURES]), 1.0
    )

    # Rank all current drivers into a valid 1..N qualifying order.
    current["predicted_qualifying_position"] = (
        current["predicted_qualifying_raw"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    # ------------------------------------------------------------
    # STAGE 2: race model using OOF qualifying predictions
    # ------------------------------------------------------------
    race_train = features.dropna(subset=[TARGET_RACE, TARGET_QUAL]).copy()
    train_aug = build_oof_qual_predictions(race_train)

    two_stage_features = PRE_QUAL_FEATURES + ["predicted_qualifying_position"]
    race_model = ridge_model()
    race_model.fit(train_aug[two_stage_features], train_aug[TARGET_RACE])

    current["predicted_finish"] = np.maximum(
        race_model.predict(current[two_stage_features]), 1.0
    )

    # Turn continuous finish estimates into a valid predicted race order.
    current["predicted_finish_position"] = (
        current["predicted_finish"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    names = load_driver_names()
    if names is not None:
        current = current.merge(names, on="driverId", how="left")
    else:
        current["driver_name"] = current["driverId"].map(lambda x: f"Driver {int(x)}")

    current["driver_name"] = current["driver_name"].fillna(current["driverId"].map(lambda x: f"Driver {int(x)}"))
    current = current.sort_values("predicted_finish_position")

    # ------------------------------------------------------------
    # PRINT HUMAN-READABLE RESULT
    # ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("PREDICTED QUALIFYING ORDER")
    print("-" * 78)
    q_order = current.sort_values("predicted_qualifying_position")
    for _, r in q_order.iterrows():
        print(f"P{int(r['predicted_qualifying_position']):<2} {r['driver_name']:<25}  model score={r['predicted_qualifying_raw']:.2f}")

    print("\n" + "-" * 78)
    print("PREDICTED SINGAPORE 2026 RACE")
    print("-" * 78)
    for _, r in current.iterrows():
        print(f"P{int(r['predicted_finish_position']):<2} {r['driver_name']:<25}  expected={r['predicted_finish']:.2f}")

    podium = current.head(3)
    print("\n" + "=" * 78)
    print("PREDICTED SINGAPORE 2026 PODIUM")
    print("=" * 78)
    print(f"Round: {SINGAPORE_ROUND}")
    print(f"🥇 P1: {podium.iloc[0]['driver_name']}")
    print(f"🥈 P2: {podium.iloc[1]['driver_name']}")
    print(f"🥉 P3: {podium.iloc[2]['driver_name']}")
    print("=" * 78)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_cols = [
        "driverId", "driver_name", "constructorId",
        "predicted_qualifying_raw", "predicted_qualifying_position",
        "predicted_finish", "predicted_finish_position",
    ]
    current[save_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved prediction CSV: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
