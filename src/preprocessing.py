import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    for col in ["transmission"]:
        categories = sorted(df[col].unique())
        if len(categories) != 2:
            raise ValueError(
                f"'{col}' expected exactly 2 categories, found {categories}"
            )
        mapping = {categories[0]: 0, categories[1]: 1}
        df[col] = df[col].map(mapping)
        print(f"Binary-encoded '{col}': {mapping}")

    df = pd.get_dummies(df, columns=["fuel", "seller_type"], drop_first=True)
    print(f"One-hot encoded: {['fuel', 'seller_type']}")
    print(f"Resulting columns: {list(df.columns)}")
    return df


def split_data(df: pd.DataFrame):
    cols_to_drop = ["selling_price"] + [c for c in ["name"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain/test split: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    cols = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=cols, index=X_test.index
    )

    print(f"\nStandardScaler fitted on training set ({len(cols)} features).")
    print(f"  mean (first 4): {np.round(scaler.mean_[:4], 4)}")
    print(f"  std  (first 4): {np.round(scaler.scale_[:4], 4)}")
    return X_train_scaled, X_test_scaled, scaler

