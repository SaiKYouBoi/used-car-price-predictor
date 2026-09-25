import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def build_models() -> dict:
    return {
        "LinearRegression": Pipeline([("model", LinearRegression())]),
        "RandomForest": Pipeline([("model", RandomForestRegressor(random_state=42))]),
        "XGBoost": Pipeline([("model", XGBRegressor(random_state=42, verbosity=0))]),
        "SVR": Pipeline([("model", SVR())]),
    }


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_and_evaluate(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    models = build_models()
    records = []

    for name, pipeline in models.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        records.append({"Model": name, **metrics})
        print(
            f"    RMSE={metrics['RMSE']:,.0f}  MAE={metrics['MAE']:,.0f}  R²={metrics['R2']:.4f}"
        )

    results_df = pd.DataFrame(records).set_index("Model")
    return results_df, models


def run_training(X_train, X_test, y_train, y_test):
    print("baseline model training")

    results_df, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nresult summary")
    print(results_df.to_string())

    best = results_df["R2"].idxmax()
    print(f"\nBest baseline model by R²: {best}  (R²={results_df.loc[best, 'R2']:.4f})")

    return results_df, fitted_models