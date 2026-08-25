import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the existing ingestion function
from src.data.ingest import load_and_validate_data


DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "data",
    "raw",
    "ncr_ride_bookings.xlsx"
)


def perform_eda():

    # ---------------------------------------------------------
    # 1. LOAD DATA USING EXISTING INGESTION PIPELINE
    # ---------------------------------------------------------

    print("=" * 60)
    print("LOADING RIDERUSH DATASET")
    print("=" * 60)

    df = load_and_validate_data(DATA_PATH)

    print("\nDataset loaded successfully.")

    # ---------------------------------------------------------
    # 2. DATASET DIMENSIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("1. DATASET DIMENSIONS")
    print("=" * 60)

    print(f"Total Rows    : {df.shape[0]}")
    print(f"Total Columns : {df.shape[1]}")

    # ---------------------------------------------------------
    # 3. FEATURE NAMES AND DATA TYPES
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("2. FEATURE NAMES & DATA TYPES")
    print("=" * 60)

    print(df.dtypes)

    # ---------------------------------------------------------
    # 4. MISSING VALUES AND DUPLICATES
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("3. MISSING VALUES & DUPLICATES")
    print("=" * 60)

    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        print("\nMissing Values per Column:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate Records: {duplicates}")

    # ---------------------------------------------------------
    # 5. STATISTICAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("4. SUMMARY STATISTICS")
    print("=" * 60)

    print(df.describe())

    # ---------------------------------------------------------
    # 6. CATEGORICAL FEATURE SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("5. CATEGORICAL FEATURE SUMMARY")
    print("=" * 60)

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        print(f"\n--- {column} ---")
        print(df[column].value_counts().head(10))

    # ---------------------------------------------------------
    # 7. BOOKING STATUS DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("6. BOOKING STATUS ANALYSIS")
    print("=" * 60)

    if "Booking Status" in df.columns:

        status_counts = df["Booking Status"].value_counts()

        status_percentages = (
            df["Booking Status"]
            .value_counts(normalize=True)
            * 100
        )

        print("\nBooking Status Counts:")
        print(status_counts)

        print("\nBooking Status Percentages:")
        print(status_percentages.round(2))

    # ---------------------------------------------------------
    # 8. CREATE FIGURE DIRECTORY
    # ---------------------------------------------------------

    figures_path = os.path.join(
        PROJECT_ROOT,
        "reports",
        "figures"
    )

    os.makedirs(figures_path, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------
    # 9. CORRELATION HEATMAP
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("7. GENERATING VISUALIZATIONS")
    print("=" * 60)

    numerical_df = df.select_dtypes(
        include=[np.number]
    )

    corr_matrix = numerical_df.corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("RideRush Numerical Feature Correlation Matrix")

    plt.tight_layout()

    output_path = os.path.join(
        figures_path,
        "correlation_heatmap.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"-> Saved: {output_path}")

    # ---------------------------------------------------------
    # 10. CTAT DISTRIBUTION
    # ---------------------------------------------------------

    if "Avg CTAT" in df.columns:

        plt.figure(figsize=(10, 6))

        sns.histplot(
            df["Avg CTAT"].dropna(),
            kde=True
        )

        plt.title("Distribution of Average Customer Time to Arrival")

        plt.xlabel("Avg CTAT")
        plt.ylabel("Frequency")

        plt.tight_layout()

        output_path = os.path.join(
            figures_path,
            "ctat_distribution.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        print(f"-> Saved: {output_path}")

    # ---------------------------------------------------------
    # 11. RIDE DISTANCE VS CTAT
    # ---------------------------------------------------------

    if (
        "Ride Distance" in df.columns
        and "Avg CTAT" in df.columns
    ):

        plot_df = df[
            ["Ride Distance", "Avg CTAT"]
        ].dropna()

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=plot_df,
            x="Ride Distance",
            y="Avg CTAT",
            alpha=0.4
        )

        plt.title(
            "Ride Distance vs Average Customer Time to Arrival"
        )

        plt.xlabel("Ride Distance")
        plt.ylabel("Avg CTAT")

        plt.tight_layout()

        output_path = os.path.join(
            figures_path,
            "distance_vs_ctat.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        print(f"-> Saved: {output_path}")

    # ---------------------------------------------------------
    # 12. VEHICLE TYPE VS CTAT
    # ---------------------------------------------------------

    if (
        "Vehicle Type" in df.columns
        and "Avg CTAT" in df.columns
    ):

        plot_df = df[
            ["Vehicle Type", "Avg CTAT"]
        ].dropna()

        plt.figure(figsize=(12, 6))

        sns.boxplot(
            data=plot_df,
            x="Vehicle Type",
            y="Avg CTAT"
        )

        plt.title(
            "Average CTAT Distribution by Vehicle Type"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        output_path = os.path.join(
            figures_path,
            "vehicle_type_ctat_boxplot.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        print(f"-> Saved: {output_path}")

    # ---------------------------------------------------------
    # 13. OUTLIER DETECTION USING IQR
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("8. OUTLIER DETECTION")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numerical_columns:

        values = df[column].dropna()

        if len(values) == 0:
            continue

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = values[
            (values < lower_bound)
            | (values > upper_bound)
        ]

        print(
            f"{column}: "
            f"{len(outliers)} outliers "
            f"({len(outliers) / len(values) * 100:.2f}%)"
        )

    # ---------------------------------------------------------
    # 14. OUTLIER BOXPLOTS
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 8))

    sns.boxplot(
        data=numerical_df,
        orient="h"
    )

    plt.title(
        "Outlier Detection Using Boxplots"
    )

    plt.tight_layout()

    output_path = os.path.join(
        figures_path,
        "outliers_boxplot.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"-> Saved: {output_path}")

    # ---------------------------------------------------------
    # 15. BOOKING STATUS VISUALIZATION
    # ---------------------------------------------------------

    if "Booking Status" in df.columns:

        plt.figure(figsize=(10, 6))

        sns.countplot(
            data=df,
            x="Booking Status"
        )

        plt.title(
            "RideRush Booking Status Distribution"
        )

        plt.xticks(rotation=25)

        plt.tight_layout()

        output_path = os.path.join(
            figures_path,
            "booking_status_distribution.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        print(f"-> Saved: {output_path}")

    print("\n" + "=" * 60)
    print("EDA EXECUTION COMPLETE")
    print("=" * 60)

    print(
        f"\nAll visualizations are stored in: "
        f"{figures_path}"
    )


if __name__ == "__main__":
    perform_eda()