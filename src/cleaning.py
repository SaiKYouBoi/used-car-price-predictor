import numpy as np
import pandas as pd
from scipy import stats
import config

def handle_missing_values(df: pd.DataFrame):
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

def detect_outliers_report(df: pd.DataFrame, cols=("selling_price", "km_driven", "year")):
    rows = []
    for col in cols:
        low_flag, high_flag = _iqr_bounds(df[col], config.igr_flag_multiplier)
        low_rm, high_rm = _iqr_bounds(df[col], config.igr_remove_multiplier)
        n_moderate = ((df[col] < low_flag) | (df[col] > high_flag)).sum()
        n_extreme = ((df[col] < low_rm) | (df[col] > high_rm)).sum()
        z = np.abs(stats.zscore(df[col].astype(float)))
        n_zscore = (z > config.zscore_threshold).sum()
        rows.append({
            "column": col,
            "IQR_moderate_outliers": n_moderate,
            "IQR_extreme_outliers": n_extreme,
            "zscore_gt_3": n_zscore,
            "flag_bounds": f"[{low_flag:.1f}, {high_flag:.1f}]",
            "remove_bounds": f"[{low_rm:.1f}, {high_rm:.1f}]",
        })
    report = pd.DataFrame(rows)
    print("\nOutliers detection report")
    print(report.to_string(index=False))
    return report

def handle_outliers(df: pd.DataFrame):
    df = df.copy()
    n0 = len(df)

    impossible_mask = (
        (df["selling_price"] <= 0)
        | (df["km_driven"] < 0)
        | (df["year"] > config.current_year)
        | (df["year"] < config.min_plausible_year)
    )
    n_impossible = impossible_mask.sum()
    df = df.loc[~impossible_mask].reset_index(drop=True)
    print(
        f"\nRemoved {n_impossible} row(s) with impossible values "
        f"(price<=0, negative mileage, or year outside "
        f"[{config.min_plausible_year}, {config.current_year}])."
    )

    extreme_mask = pd.Series(False, index=df.index)
    for col in ("selling_price", "km_driven", "year"):
        low, high = _iqr_bounds(df[col], config.igr_remove_multiplier)
        extreme_mask |= (df[col] < low) | (df[col] > high)
    n_extreme = extreme_mask.sum()
    df = df.loc[~extreme_mask].reset_index(drop=True)
    print(
        f"Removed {n_extreme} row(s) as extreme statistical outliers "
        f"(beyond {config.igr_remove_multiplier}x IQR on Selling_Price, "
        f"Km_Driven, or Year)."
    )

    moderate_mask = pd.Series(False, index=df.index)
    for col in ("selling_price", "km_driven", "year"):
        low, high = _iqr_bounds(df[col], config.igr_flag_multiplier)
        moderate_mask |= (df[col] < low) | (df[col] > high)
    n_moderate = moderate_mask.sum()
    print(
        f"Kept {n_moderate} row(s) flagged as moderate outliers "
        f"(between {config.igr_flag_multiplier}x and {config.igr_remove_multiplier}x IQR) "
        f"-- plausible in a used-car market (e.g. high-mileage older cars, "
        f"low-mileage premium cars)."
    )

    print(
        f"\nTotal rows: {n0} -> {len(df)} after outlier handling "
        f"({n0 - len(df)} rows removed in total)."
    )
    return df


def run_cleaning(df: pd.DataFrame):
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    detect_outliers_report(df)
    df = handle_outliers(df)
    return df

if __name__ == "__main__":
    from data_loader import load_data,encode_owner
    data = load_data()
    data = encode_owner(data)
    data.describe()
    run_cleaning(data)