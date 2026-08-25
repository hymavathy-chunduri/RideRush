import sys
from sklearn.dummy import DummyRegressor

from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    load_data,
    clean_data,
    prepare_regression_data,
)
from src.features.build_features import build_features


DATA_PATH = "src/data/raw/ncr_ride_bookings.xlsx"


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a regression model."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Results")
    print("-" * 40)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Load and preprocess data
    # ---------------------------------------------------------

    df = load_data(DATA_PATH)

    df = clean_data(df)

    regression_df = prepare_regression_data(df)

    # ---------------------------------------------------------
    # 2. Build features
    # ---------------------------------------------------------

    features_df = build_features(regression_df)

    # ---------------------------------------------------------
    # 3. Separate features and target
    # ---------------------------------------------------------

    X = features_df.drop(columns=["Avg CTAT"])
    y = features_df["Avg CTAT"]

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # ---------------------------------------------------------
    # 4. Identify feature types
    # ---------------------------------------------------------

    categorical_features = [
        "Vehicle Type",
        "Pickup Location",
        "Drop Location",
        "Payment Method",
    ]

    numerical_features = [
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "Avg VTAT",
        "Ride Distance",
        "Booking Value",
    ]

    # ---------------------------------------------------------
    # 5. Encode categorical variables
    # ---------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    # ---------------------------------------------------------
    # 6. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")



    # ---------------------------------------------------------
    # 7. Baseline model
    # ---------------------------------------------------------

    baseline = DummyRegressor(strategy="mean")

    print("\nTraining baseline model...")

    baseline.fit(X_train, y_train)

    baseline_results = evaluate_model(
        baseline,
        X_test,
        y_test,
        "Mean Baseline",
    )

    # ---------------------------------------------------------
    # 8. Random Forest baseline
    # ---------------------------------------------------------

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest...")

    random_forest.fit(X_train, y_train)

    rf_results = evaluate_model(
        random_forest,
        X_test,
        y_test,
        "Random Forest",
    )

    joblib.dump(
        random_forest,
        "models/random_forest.pkl"
    )

    print("Random Forest model saved.")

    # ---------------------------------------------------------
    # 9. XGBoost model
    # ---------------------------------------------------------

    xgboost_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining XGBoost...")

    xgboost_model.fit(X_train, y_train)

    xgb_results = evaluate_model(
        xgboost_model,
        X_test,
        y_test,
        "XGBoost",
    )

    joblib.dump(
        xgboost_model,
        "models/xgboost.pkl"
    )

    print("XGBoost model saved.")

    # ---------------------------------------------------------
    # 10. Compare models
    # ---------------------------------------------------------

    results = pd.DataFrame(
        [baseline_results, rf_results, xgb_results]
    )

    print("\nModel Comparison")
    print("=" * 60)
    print(results.to_string(index=False))
