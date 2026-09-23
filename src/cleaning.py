import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nmissing values")
    missing_before = df.isna().sum()
    print("Missing values before imputation:")
    print(missing_before[missing_before > 0] if missing_before.sum() else "None found.")

    for col in ["selling_price","year","km_driven","owner"]:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(
                f"  - '{col}': filled {n_filled} missing value(s) with median ({median_val})."
            )

    for col in ["fuel", "seller_type", "transmission"]:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True)[0]
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            print(
                f"  - '{col}': filled {n_filled} missing value(s) with mode ('{mode_val}')."
            )

    return df


def remove_duplicates(df: pd.DataFrame):
    print("\nDuplicates")
    n_dupes = df.duplicated().sum()
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_dupes} exact duplicate row(s). New shape: {df_clean.shape}")
    return df_clean


def _iqr_bounds(series: pd.Series, multiplier: float):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr
