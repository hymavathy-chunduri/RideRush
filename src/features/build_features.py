import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    load_data,
    clean_data,
    prepare_regression_data,
)

DATA_PATH = "src/data/raw/ncr_ride_bookings.xlsx"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the RideRush CTAT prediction model.

    Existing preprocessing is kept separate from feature engineering.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Time-based features
    # ---------------------------------------------------------

    df["Hour"] = df["Time"].apply(
        lambda x: x.hour if pd.notna(x) else None
    )

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["Month"] = df["Date"].dt.month

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # Select useful features for the regression model
    # ---------------------------------------------------------

    feature_columns = [
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "Vehicle Type",
        "Pickup Location",
        "Drop Location",
        "Avg VTAT",
        "Ride Distance",
        "Booking Value",
        "Payment Method",
        "Avg CTAT",
    ]

    features_df = df[feature_columns].copy()

    print(
        f"Feature dataset shape: "
        f"{features_df.shape}"
    )

    return features_df


if __name__ == "__main__":

    # 1. Load raw data
    df = load_data(DATA_PATH)

    # 2. Perform existing preprocessing
    df = clean_data(df)

    # 3. Keep rows where CTAT is available
    regression_df = prepare_regression_data(df)

    # 4. Build ML features
    features_df = build_features(regression_df)

    print("\nFeature engineering completed successfully.")

    print("\nFeatures:")
    print(features_df.columns.tolist())

    print("\nFirst 5 rows:")
    print(features_df.head())

    print("\nMissing values:")
    print(features_df.isnull().sum())