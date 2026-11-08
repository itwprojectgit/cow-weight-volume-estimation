"""
Live Weight Model Training (Regression)
=======================================
Trains regression models directly on the 9 anatomical measurements from
Measurements.xlsx (103 cattle) to predict Live Weight (kg) without volume.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "withers height",
    "hip height",
    "chest depth",
    "chest width",
    "ilium width",
    "hip joint width",
    "oblique body length",
    "hip length",
    "heart girth",
]
TARGET_COL = "live weithg"


def load_dataset(excel_path: Path):
    """Loads dataset and corrects known data-entry typo for Cow #31."""
    df = pd.read_excel(excel_path)
    df.loc[df["oblique body length"] > 300, "oblique body length"] = 152
    return df


def main():
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "data" / "Measurements.xlsx"
    df = load_dataset(excel_path)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Linear Regression (OLS)": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        "Lasso Regression (L1)": Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=1.0, random_state=42))]),
        "Ridge Regression (L2)": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0, random_state=42))]),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    }

    best_name, best_mape, best_pipeline = None, float("inf"), None

    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=kf)
        mae = mean_absolute_error(y, preds)
        mape = mean_absolute_percentage_error(y, preds) * 100.0
        r2 = r2_score(y, preds)
        print(f"{name:<25} | MAE: {mae:5.2f} kg | MAPE: {mape:5.2f}% | R2: {r2:5.3f}")

        if mape < best_mape:
            best_mape, best_name, best_pipeline = mape, name, model

    # Fit best model on full dataset and serialize
    best_pipeline.fit(X, y)
    out_model = base_dir / "weight_model.joblib"
    joblib.dump({"model": best_pipeline, "feature_cols": FEATURE_COLS, "model_name": best_name, "best_mape": best_mape}, out_model)
    print(f"Saved best model ({best_name}, MAPE: {best_mape:.2f}%) to {out_model.name}")


if __name__ == "__main__":
    main()
