"""
export.py — Step 5a: Model Export
==================================
Saves all artifacts needed to make predictions without retraining:
  - final_model.joblib     : the winning fitted estimator
  - scaler.joblib          : the fitted StandardScaler
  - feature_columns.json   : exact column list in training order (critical!)
  - model_meta.json        : model name, export date, test-set metrics
"""

import json
import joblib
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def export_model(
    model,
    scaler,
    feature_columns: list,
    model_name: str,
    metrics: dict,
) -> None:
    """
    Persist the final model and all artifacts required for prediction.

    Parameters
    ----------
    model           : fitted estimator (XGBRegressor, RandomForestRegressor, …)
    scaler          : fitted StandardScaler from preprocessing.scale_features()
    feature_columns : ordered list of column names the model was trained on
    model_name      : display name of the winner, e.g. "XGBoost (Tuned)"
    metrics         : dict with at least keys RMSE, MAE, R2

    Why joblib instead of pickle?
    ─────────────────────────────
    joblib is optimised for numpy arrays (the internal state of sklearn/XGBoost
    estimators). It is faster and produces smaller files than pickle for these
    objects, and is the format recommended by scikit-learn's own documentation.

    Why save feature_columns separately?
    ─────────────────────────────────────
    The model only sees numbers — it has no memory of column names. If the input
    DataFrame is reconstructed in a different column order at prediction time,
    the model will silently map the wrong values to the wrong features and produce
    garbage predictions. Saving the exact ordered list makes the predictor
    deterministic and safe.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(model, model_path, compress=3)
    print(f"  [export] Model saved       : {model_path}")

    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path, compress=3)
    print(f"  [export] Scaler saved      : {scaler_path}")

    cols_path = MODELS_DIR / "feature_columns.json"
    cols_path.write_text(json.dumps(feature_columns, indent=2, ensure_ascii=False))
    print(f"  [export] Columns saved     : {cols_path}  ({len(feature_columns)} features)")

    meta = {
        "model_name":      model_name,
        "exported_at":     datetime.now().isoformat(timespec="seconds"),
        "feature_columns": feature_columns,
        "metrics": {
            "RMSE": round(float(metrics.get("RMSE", 0)), 2),
            "MAE":  round(float(metrics.get("MAE",  0)), 2),
            "R2":   round(float(metrics.get("R2",   0)), 6),
        },
    }
    meta_path = MODELS_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  [export] Metadata saved    : {meta_path}")

    print(f"\n  Export complete.")
    print(f"  Winner : {model_name}")
    print(f"  R2     : {meta['metrics']['R2']:.4f}")
    print(f"  RMSE   : {meta['metrics']['RMSE']:,.0f} INR")
