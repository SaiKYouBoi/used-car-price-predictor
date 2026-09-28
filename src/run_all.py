import sys
from pathlib import Path
from data_loader import load_data, encode_owner, check_structure
from data_analysis import run_eda
from cleaning import run_cleaning
from preprocessing import encode_categoricals, split_data, scale_features
from training import run_training
from tuning import run_tuning
from evaluation import run_evaluation
from export import export_model

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

    print("\n9.training baseline models...")
    results_df, fitted_models = run_training(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\n10. hyperparameter tuning (RandomizedSearchCV, 5-fold CV)...")
    comparison_df, tuned_models = run_tuning(
        baseline_results=results_df,
        fitted_models=fitted_models,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        top_n=2,
        n_iter=30,
    )

    print("\n\nBefore / After Tuning Summary")
    print(comparison_df[["RMSE_before", "RMSE_after", "RMSE_delta",
                          "R2_before", "R2_after", "R2_delta"]].to_string())

    best_tuned = comparison_df["R2_after"].idxmax()
    print(f"\nBest tuned model: {best_tuned}  "
          f"(R2={comparison_df.loc[best_tuned, 'R2_after']:.4f})")

    print("\n11. comparison, visualization & final model selection...")
    save_dir = Path(__file__).resolve().parents[1] / "data" / "processed"

    winner_name, final_model, summary_df = run_evaluation(
        baseline_results=results_df,
        comparison_df=comparison_df,
        fitted_models=fitted_models,
        tuned_models=tuned_models,
        X_test=X_test_scaled,
        y_test=y_test,
        save_dir=save_dir,
    )

    print(f"\nFinal model ready: {winner_name}")

    print("\n12. exporting final model and artifacts...")

    winner_base  = winner_name.split(" (")[0] 
    winner_stage = "Tuned" if "Tuned" in winner_name else "Baseline"
    winner_row   = summary_df[
        (summary_df["Model"] == winner_base) & (summary_df["Stage"] == winner_stage)
    ]
    if winner_row.empty:
        winner_row = summary_df[summary_df["Stage"] == winner_stage].iloc[[0]]
    winner_metrics = winner_row.iloc[0][["RMSE", "MAE", "R2"]].to_dict()

    export_model(
        model=final_model,
        scaler=scaler,
        feature_columns=X_train_scaled.columns.tolist(),
        model_name=winner_name,
        metrics=winner_metrics,
    )

    print("\nAll steps complete.")
    print("  Run `python src/predict.py` to estimate a vehicle price.")
    print("  Run `python src/predict.py --demo` for sample predictions.")


if __name__ == "__main__":
    main()
