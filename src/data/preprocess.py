import pandas as pd


DATA_PATH = "src/data/raw/ncr_ride_bookings.xlsx"


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw RideRush dataset."""

    print(f"Loading dataset from: {file_path}")

    df = pd.read_excel(file_path)

    print(f"Dataset shape: {df.shape}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning and data type conversion."""

    df = df.copy()

    # Convert Excel serial dates into proper dates
    df["Date"] = pd.to_datetime(
        df["Date"],
        unit="D",
        origin="1899-12-30"
    )

    # Convert Time column to datetime time
    df["Time"] = pd.to_datetime(
        df["Time"].astype(str),
        format="%H:%M:%S",
        errors="coerce"
    ).dt.time

    # Remove duplicate records
    df = df.drop_duplicates()

    print(f"Shape after basic cleaning: {df.shape}")

    return df


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for predicting Avg CTAT.

    Rows where Avg CTAT is missing cannot be used
    because Avg CTAT is the regression target.
    """

    regression_df = df.dropna(subset=["Avg CTAT"]).copy()

    print(
        f"Regression dataset: "
        f"{regression_df.shape[0]} samples"
    )

    return regression_df


if __name__ == "__main__":

    df = load_data(DATA_PATH)

    df = clean_data(df)

    regression_df = prepare_regression_data(df)

    print("\nPreprocessing completed successfully.")

    print("\nRegression target statistics:")
    print(regression_df["Avg CTAT"].describe())