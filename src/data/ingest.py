import os
import pandas as pd


def load_and_validate_data(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    print(f"Loading RideRush dataset from: {file_path}")

    df = pd.read_excel(file_path)

    required_columns = [
        'Date',
        'Time',
        'Booking ID',
        'Booking Status',
        'Customer ID',
        'Vehicle Type',
        'Pickup Location',
        'Drop Location',
        'Avg VTAT',
        'Avg CTAT',
        'Booking Value',
        'Ride Distance',
        'Driver Ratings',
        'Customer Rating',
        'Payment Method'
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Schema validation failed. "
            f"Missing columns: {missing_columns}"
        )

    print(
        f"Dataset loaded successfully: "
        f"{df.shape[0]} samples, {df.shape[1]} columns."
    )

    print(f"Duplicate rows: {df.duplicated().sum()}")

    return df


if __name__ == "__main__":

    DATA_PATH = "src/data/raw/ncr_ride_bookings.xlsx"

    try:
        raw_data = load_and_validate_data(DATA_PATH)

        print("\nDataset validation completed successfully.")

    except Exception as e:

        print(f"Data ingestion failed: {str(e)}")