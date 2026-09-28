"""
predict.py — Step 5b: Interactive Price Estimator
===================================================
Loads the exported model artifacts and provides an interactive CLI so any
user can enter a vehicle's characteristics and get an instant price estimate.

Usage
-----
    python src/predict.py                  # interactive mode (default)
    python src/predict.py --demo           # runs a built-in demo without prompts

How it works
------------
1. Load final_model.joblib, scaler.joblib, feature_columns.json, model_meta.json
2. Collect vehicle characteristics from the user (or use demo values)
3. Apply the EXACT same preprocessing used during training:
     - binary-encode transmission  (Automatic=0, Manual=1)
     - one-hot encode fuel and seller_type  (drop_first=True, same ref categories)
     - align columns to the saved feature list  (fill missing dummies with 0)
     - scale with the fitted StandardScaler
4. Call model.predict() and display the result with a confidence range
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

VALID_FUEL         = {"Petrol", "Diesel", "Cng", "Lpg", "Electric"}
VALID_SELLER       = {"Individual", "Dealer", "Trustmark Dealer"}
VALID_TRANSMISSION = {"Manual", "Automatic"}

# Transmission binary encoding must match training: sorted(['Manual', 'Automatic']) → ['Automatic', 'Manual'] → {0, 1}
TRANSMISSION_MAP = {"Automatic": 0, "Manual": 1}



def load_artifacts() -> tuple:
    """
    Load all model artifacts from the models/ directory.

    Returns
    -------
    model, scaler, feature_columns (list), meta (dict)

    Raises
    ------
    FileNotFoundError if models/ has not been populated yet (run run_all.py first).
    """
    required = [
        "final_model.joblib",
        "scaler.joblib",
        "feature_columns.json",
        "model_meta.json",
    ]
    for f in required:
        if not (MODELS_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing artifact: {MODELS_DIR / f}\n"
                "Run `python src/run_all.py` first to train and export the model."
            )

    model   = joblib.load(MODELS_DIR / "final_model.joblib")
    scaler  = joblib.load(MODELS_DIR / "scaler.joblib")
    columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    meta    = json.loads((MODELS_DIR / "model_meta.json").read_text())
    return model, scaler, columns, meta


def _prompt(label: str, valid_set: set = None, cast=str) -> str:
    """Helper: prompt until a valid value is entered."""
    while True:
        raw = input(f"  {label}: ").strip()
        value = cast(raw) if cast != str else raw.title() if valid_set else raw
        if valid_set is None:
            return value
        if value in valid_set:
            return value
        print(f"    Invalid input. Choose from: {', '.join(sorted(valid_set))}")


def get_user_input() -> dict:
    """Interactively collect all vehicle characteristics from the user."""
    print()
    print("  Enter vehicle details")
    print("  " + "─" * 40)

    while True:
        try:
            year = int(input("  Registration year   (e.g. 2018) : ").strip())
            if 1990 <= year <= 2026:
                break
            print("    Year must be between 1990 and 2026.")
        except ValueError:
            print("    Please enter a valid year (integer).")

    while True:
        try:
            km = int(input("  Kilometres driven   (e.g. 45000) : ").strip())
            if km >= 0:
                break
            print("    Kilometres cannot be negative.")
        except ValueError:
            print("    Please enter a whole number.")

    print("  Fuel options        : Petrol | Diesel | Cng | Lpg | Electric")
    fuel = _prompt("Fuel type", VALID_FUEL)

    print("  Seller options      : Individual | Dealer | Trustmark Dealer")
    while True:
        raw_seller = input("  Seller type         : ").strip().title()
        if raw_seller in VALID_SELLER:
            seller = raw_seller
            break
        print(f"    Invalid. Choose from: {', '.join(sorted(VALID_SELLER))}")

    print("  Transmission options: Manual | Automatic")
    transmission = _prompt("Transmission", VALID_TRANSMISSION)

    while True:
        try:
            owner = int(input("  Previous owners     (1-4, 4=4th+) : ").strip())
            if 0 <= owner <= 4:
                break
            print("    Enter a value between 0 and 4.")
        except ValueError:
            print("    Please enter a whole number.")

    return {
        "year":         year,
        "km_driven":    km,
        "fuel":         fuel,
        "seller_type":  seller,
        "transmission": transmission,
        "owner":        owner,
    }


def preprocess_input(raw: dict, feature_columns: list, scaler) -> pd.DataFrame:
    """
    Reproduce the exact preprocessing pipeline applied during training.

    Steps
    -----
    1. Build a single-row DataFrame from raw user input
    2. Binary-encode transmission  (must match sorted() mapping from training)
    3. One-hot encode fuel and seller_type with drop_first=True
    4. Add any dummy columns missing from user input (fill with 0)
    5. Reorder columns to match training feature list exactly
    6. Apply the fitted StandardScaler

    Why column alignment (step 4-5) is necessary
    ─────────────────────────────────────────────
    If the user enters "Petrol", pd.get_dummies won't create a "fuel_Petrol"
    column because Petrol may be the reference (dropped) category. Conversely,
    the other fuel dummy columns ARE expected by the model. We fill them with 0.
    """

    row = pd.DataFrame([{
        "year":         raw["year"],
        "km_driven":    raw["km_driven"],
        "owner":        raw["owner"],
        "transmission": raw["transmission"],
        "fuel":         raw["fuel"],
        "seller_type":  raw["seller_type"],
    }])

    row["transmission"] = row["transmission"].map(TRANSMISSION_MAP)

    row = pd.get_dummies(row, columns=["fuel", "seller_type"], drop_first=True)

    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_columns]

    row_scaled = pd.DataFrame(
        scaler.transform(row),
        columns=feature_columns,
    )
    return row_scaled


def predict_price(model, row_scaled: pd.DataFrame, rmse: float) -> tuple:
    """
    Return point estimate and a ±1 RMSE confidence range.

    The RMSE is the average prediction error measured on the held-out test set.
    Using it as a rough confidence interval gives the user a realistic sense of
    how much the actual price might differ from the estimate.
    """
    price = float(model.predict(row_scaled)[0])
    low   = max(0.0, price - rmse)
    high  = price + rmse
    return price, low, high

def display_result(raw: dict, price: float, low: float, high: float, meta: dict) -> None:
    sep = "=" * 52

    print(f"\n{sep}")
    print("  PRICE ESTIMATE")
    print(sep)
    print(f"  Estimated price  :  INR {price:>12,.0f}")
    print(f"  Confidence range :  INR {low:>12,.0f}  –  {high:>12,.0f}")
    print(sep)

    print("\n  Vehicle summary:")
    print(f"    Year         : {raw['year']}")
    print(f"    Km driven    : {raw['km_driven']:,}")
    print(f"    Fuel         : {raw['fuel']}")
    print(f"    Seller type  : {raw['seller_type']}")
    print(f"    Transmission : {raw['transmission']}")
    print(f"    Owners       : {raw['owner']}")

    print(f"\n  Model : {meta['model_name']}")
    print(f"  R²    : {meta['metrics']['R2']:.4f}   "
          f"(explains {meta['metrics']['R2']*100:.1f}% of price variance)")
    print(f"  RMSE  : {meta['metrics']['RMSE']:,.0f} INR  "
          f"(avg. prediction error on test set)")


DEMO_INPUTS = [
    {
        "label":        "2019 Diesel SUV, low mileage, 1st owner",
        "year":         2019,
        "km_driven":    38000,
        "fuel":         "Diesel",
        "seller_type":  "Individual",
        "transmission": "Manual",
        "owner":        1,
    },
    {
        "label":        "2015 Petrol hatchback, high mileage, 2nd owner",
        "year":         2015,
        "km_driven":    95000,
        "fuel":         "Petrol",
        "seller_type":  "Individual",
        "transmission": "Manual",
        "owner":        2,
    },
    {
        "label":        "2021 Automatic, dealer, first owner",
        "year":         2021,
        "km_driven":    12000,
        "fuel":         "Petrol",
        "seller_type":  "Dealer",
        "transmission": "Automatic",
        "owner":        1,
    },
]

def run_demo(model, scaler, feature_columns, meta) -> None:
    print("\n  Running demo predictions...")
    rmse = meta["metrics"]["RMSE"]
    for demo in DEMO_INPUTS:
        label = demo.pop("label")
        print(f"\n  {'─'*48}")
        print(f"  Case: {label}")
        row_scaled = preprocess_input(demo, feature_columns, scaler)
        price, low, high = predict_price(model, row_scaled, rmse)
        print(f"  Estimated price : INR {price:>12,.0f}")
        print(f"  Range           : INR {low:>12,.0f}  –  {high:>12,.0f}")
        demo["label"] = label   # restore for potential re-use

def main() -> None:
    demo_mode = "--demo" in sys.argv

    print("=" * 52)
    print("  Used Car Price Estimator")
    print("=" * 52)

    try:
        model, scaler, feature_columns, meta = load_artifacts()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"  Model loaded : {meta['model_name']}")
    print(f"  Exported at  : {meta['exported_at']}")
    print(f"  R2           : {meta['metrics']['R2']:.4f}")
    print(f"  RMSE         : {meta['metrics']['RMSE']:,.0f} INR")

    if demo_mode:
        run_demo(model, scaler, feature_columns, meta)
        return

    while True:
        raw = get_user_input()
        row_scaled = preprocess_input(raw, feature_columns, scaler)
        rmse = meta["metrics"]["RMSE"]
        price, low, high = predict_price(model, row_scaled, rmse)
        display_result(raw, price, low, high, meta)

        again = input("\n  Estimate another vehicle? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
