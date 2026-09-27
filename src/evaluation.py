import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":  mean_absolute_error(y_test, y_pred),
        "R2":   r2_score(y_test, y_pred),
    }

def build_summary_table(
    baseline_results: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine baseline and tuned metrics into a single summary DataFrame.

    Parameters
    ----------
    baseline_results : DataFrame indexed by model name with RMSE / MAE / R2
    comparison_df    : DataFrame from run_tuning() with *_before / *_after cols

    Returns
    -------
    summary : DataFrame with columns [Model, Stage, RMSE, MAE, R2]
    """

    base = baseline_results[["RMSE", "MAE", "R2"]].copy()
    base.index.name = "Model"
    base["Stage"] = "Baseline"

    tuned = comparison_df[["RMSE_after", "MAE_after", "R2_after"]].copy()
    tuned.columns = ["RMSE", "MAE", "R2"]
    tuned.index.name = "Model"
    tuned["Stage"] = "Tuned"

    summary = pd.concat([base, tuned]).reset_index()
    summary = summary[["Model", "Stage", "RMSE", "MAE", "R2"]]
    summary = summary.sort_values(["Model", "Stage"]).reset_index(drop=True)

    print("\nFull Model Performance Summary")
    print("=" * 65)
    print(summary.to_string(index=False,
          formatters={"RMSE": "{:,.0f}".format,
                      "MAE":  "{:,.0f}".format,
                      "R2":   "{:.4f}".format}))
    print("=" * 65)
    return summary

def plot_predictions_vs_actual(
    models: dict,
    X_test,
    y_test,
    save_dir: Path,
) -> None:
    """
    One scatter subplot per model: predicted price vs. actual price.
    The red dashed diagonal = perfect predictions.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Predicted vs. Actual Selling Price", fontsize=14, fontweight="bold")

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        ax.scatter(y_test, y_pred, alpha=0.35, s=12, color="steelblue", label="Predictions")

        # perfect-prediction line
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")

        ax.set_xlabel("Actual Price (₹)", fontsize=10)
        ax.set_ylabel("Predicted Price (₹)", fontsize=10)
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

    out = save_dir / "predictions_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def plot_residuals(
    models: dict,
    X_test,
    y_test,
    save_dir: Path,
) -> None:
    """
    Two-row plot per model:
      Row 1 — residuals vs. predicted values  (pattern check)
      Row 2 — histogram of residuals           (distribution check)
    """
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")

    for col, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        residuals = y_pred - y_test  # positive = over-predicted

        ax0 = axes[0, col]
        ax0.scatter(y_pred, residuals, alpha=0.35, s=12, color="steelblue")
        ax0.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax0.set_xlabel("Predicted Price (₹)", fontsize=9)
        ax0.set_ylabel("Residual (₹)", fontsize=9)
        ax0.set_title(f"{name}\nResiduals vs. Predicted", fontsize=9)
        ax0.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

        ax1 = axes[1, col]
        ax1.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax1.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax1.axvline(residuals.mean(), color="orange", linestyle="-",
                    linewidth=1.2, label=f"mean={residuals.mean():,.0f}")
        ax1.set_xlabel("Residual (₹)", fontsize=9)
        ax1.set_ylabel("Count", fontsize=9)
        ax1.set_title("Residual Distribution", fontsize=9)
        ax1.legend(fontsize=8)
        ax1.ticklabel_format(style="sci", axis="x", scilimits=(5, 5))

    out = save_dir / "residuals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def select_final_model(
    summary_df: pd.DataFrame,
    models: dict,
) -> tuple:
    """
    Pick the best model from the summary table.

    Selection criteria (in order):
      1. Highest R²  (primary — explains most variance)
      2. Lowest RMSE (tiebreaker — tighter error distribution)

    Parameters
    ----------
    summary_df : DataFrame with columns [Model, Stage, RMSE, MAE, R2]
    models     : dict of {display_name: fitted_estimator}

    Returns
    -------
    (winner_display_name, winner_estimator)
    """

    tuned_mask = summary_df["Stage"] == "Tuned"
    candidates = summary_df[tuned_mask] if tuned_mask.any() else summary_df

    best_row = (
        candidates
        .sort_values(["R2", "RMSE"], ascending=[False, True])
        .iloc[0]
    )

    model_name  = best_row["Model"]
    stage       = best_row["Stage"]
    display_key = f"{model_name} ({stage})"

    # find the matching key in the models dict
    matched_key = next(
        (k for k in models if model_name in k and stage in k),
        display_key,
    )
    winner = models.get(matched_key)

    sep = "─" * 52
    print(f"\n{sep}")
    print("  FINAL MODEL SELECTION")
    print(sep)
    print(f"  Winner : {display_key}")
    print(f"  R²     : {best_row['R2']:.4f}   ← explains {best_row['R2']*100:.1f}% of price variance")
    print(f"  RMSE   : {best_row['RMSE']:>12,.0f}   ← avg prediction error")
    print(f"  MAE    : {best_row['MAE']:>12,.0f}   ← median-like error")
    print(f"\n  Rationale:")
    print(f"    • Highest R² among {stage.lower()} models → best overall fit")
    print(f"    • RMSE used as tiebreaker → tighter error distribution")
    print(f"    • Model was retrained on full training set after tuning")
    print(sep)

    return display_key, winner

def run_evaluation(
    baseline_results: pd.DataFrame,
    comparison_df: pd.DataFrame,
    fitted_models: dict,
    tuned_models: dict,
    X_test,
    y_test,
    save_dir: Path,
) -> tuple:
    """
    Run the full Step 4 pipeline:
      1. Build summary table
      2. Plot predictions vs. actual
      3. Plot residuals
      4. Select and return the final model

    Parameters
    ----------
    baseline_results : DataFrame from run_training()
    comparison_df    : DataFrame from run_tuning()
    fitted_models    : dict of baseline Pipeline objects
    tuned_models     : dict of tuned estimator objects
    X_test / y_test  : held-out test set
    save_dir         : directory to write PNG files

    Returns
    -------
    (winner_name, final_model, summary_df)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    all_models = {}
    for name, pipeline in fitted_models.items():
        key = f"{name} (Baseline)"
        try:
            all_models[key] = pipeline.named_steps["model"]
        except AttributeError:
            all_models[key] = pipeline

    for name, estimator in tuned_models.items():
        all_models[f"{name} (Tuned)"] = estimator

    print("\n--- Step 4: Comparison & Final Model Selection ---")

    print("\nBuilding performance summary table...")
    summary_df = build_summary_table(baseline_results, comparison_df)

    print("\nGenerating predictions vs. actual plots...")
    plot_predictions_vs_actual(all_models, X_test, y_test, save_dir)

    print("\nGenerating residual plots...")
    plot_residuals(all_models, X_test, y_test, save_dir)

    winner_name, final_model = select_final_model(summary_df, all_models)

    return winner_name, final_model, summary_df
