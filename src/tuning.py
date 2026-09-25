import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor



rf_param_dist = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.5],
    "bootstrap":         [True, False],
}

xgb_param_dist = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "gamma":            [0, 0.1, 0.2, 0.5],
    "reg_alpha":        [0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}


def _metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":  mean_absolute_error(y_test, y_pred),
        "R2":   r2_score(y_test, y_pred),
    }

def tune_model(name: str,estimator,param_dist: dict,X_train,y_train,n_iter: int = 30,cv: int = 5,random_state: int = 42,) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV for a single estimator.

    Parameters
    ----------
    name        : human-readable label for printing
    estimator   : unfitted sklearn/XGBoost estimator
    param_dist  : dict of hyperparameter distributions
    X_train     : training features
    y_train     : training target
    n_iter      : number of random combinations to try  (default 30)
    cv          : number of cross-validation folds       (default 5)

    Returns
    -------
    Fitted RandomizedSearchCV object (best_estimator_ is already retrained
    on the full X_train by sklearn automatically via refit=True).
    """
    print(f"\n  [{name}] Starting RandomizedSearchCV  "
          f"(n_iter={n_iter}, cv={cv}-fold)...")

    search = RandomizedSearchCV(estimator=estimator,param_distributions=param_dist,n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,)
    search.fit(X_train, y_train)

    best_cv_rmse = -search.best_score_
    print(f"  [{name}] Best CV RMSE : {best_cv_rmse:,.0f}")
    print(f"  [{name}] Best params  : {search.best_params_}")
    return search


def run_tuning(baseline_results: pd.DataFrame, fitted_models: dict, X_train,X_test,y_train,y_test,top_n: int = 2,
n_iter: int = 30,) -> tuple:
    """
    Select the best top_n baseline models, tune them, and compare results.

    Parameters
    ----------
    baseline_results : DataFrame from run_training() with RMSE/MAE/R2 columns
    fitted_models    : dict of already-trained baseline pipelines
    X_train / X_test : feature matrices (scaled)
    y_train / y_test : target vectors
    top_n            : how many top models to tune
    n_iter           : random search iterations per model

    Returns
    -------
    comparison_df  : before/after metrics for each tuned model
    tuned_models   : dict of {model_name: best_estimator_}
    """

    top_models = baseline_results.nlargest(top_n, "R2").index.tolist()
    print(f"\nTop {top_n} models selected for tuning: {top_models}")

    tunable = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            rf_param_dist,
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, verbosity=0),
            xgb_param_dist,
        ),
    }

    records = []
    tuned_models = {}

    for name in top_models:
        if name not in tunable:
            print(f"  Skipping '{name}' -- no search space defined.")
            continue

        estimator, param_dist = tunable[name]

        before = {
            "RMSE": baseline_results.loc[name, "RMSE"],
            "MAE":  baseline_results.loc[name, "MAE"],
            "R2":   baseline_results.loc[name, "R2"],
        }

        search = tune_model(
            name=name,
            estimator=estimator,
            param_dist=param_dist,
            X_train=X_train,
            y_train=y_train,
            n_iter=n_iter,
        )

        best_model = search.best_estimator_
        tuned_models[name] = best_model

        after = _metrics(best_model, X_test, y_test)

        records.append({
            "Model":       name,
            "RMSE_before": before["RMSE"],
            "RMSE_after":  after["RMSE"],
            "RMSE_delta":  before["RMSE"] - after["RMSE"],
            "MAE_before":  before["MAE"],
            "MAE_after":   after["MAE"],
            "R2_before":   before["R2"],
            "R2_after":    after["R2"],
            "R2_delta":    after["R2"] - before["R2"],
        })

        print(f"\n  [{name}] Comparison")
        print(f"    RMSE : {before['RMSE']:>12,.0f}  ->  {after['RMSE']:>12,.0f}  "
              f"(delta {before['RMSE'] - after['RMSE']:+,.0f})")
        print(f"    MAE  : {before['MAE']:>12,.0f}  ->  {after['MAE']:>12,.0f}")
        print(f"    R2   : {before['R2']:>12.4f}  ->  {after['R2']:>12.4f}  "
              f"(delta {after['R2'] - before['R2']:+.4f})")

    comparison_df = pd.DataFrame(records).set_index("Model")
    return comparison_df, tuned_models
