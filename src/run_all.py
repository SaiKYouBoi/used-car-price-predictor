import sys
from pathlib import Path
from data_loader import load_data, encode_owner, check_structure
from data_analysis import run_eda
from cleaning import run_cleaning
from preprocessing import encode_categoricals, split_data, scale_features


def main():
    print("  STEP 1 — DATA EXPLORATION & CLEANING PIPELINE")

    print("\n1.loading raw data...")
    df = load_data()

    print("\n2. encoding 'owner' column...")
    df = encode_owner(df)

    print("\n3.checking data structure...")
    check_structure(df)

    print("\n4. running exploratory data analysis...")
    run_eda(df)

    print("\n5. leaning data...")
    df = run_cleaning(df)

    print("\n6.encoding categorical variables...")
    df = encode_categoricals(df)

    print("\n7. splitting into train / test sets (80/20)...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n8. scaling features with StandardScaler...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    
    print("pipline complte")
    print(f"  X_train : {X_train_scaled.shape}")
    print(f"  X_test  : {X_test_scaled.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"  Features: {list(X_train_scaled.columns)}")
    print("\nReady for Step 2 — Model Training.")


if __name__ == "__main__":
    main()
