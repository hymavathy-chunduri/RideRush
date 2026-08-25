import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    load_data,
    clean_data,
    prepare_regression_data,
)


DATA_PATH = "src/data/raw/ncr_ride_bookings.xlsx"


def train_linear_regression_ls():

    # ---------------------------------------------------------
    # 1. Load and preprocess data
    # ---------------------------------------------------------

    print("=" * 60)
    print("RIDERUSH LINEAR REGRESSION - LEAST SQUARES")
    print("=" * 60)

    df = load_data(DATA_PATH)

    df = clean_data(df)

    regression_df = prepare_regression_data(df)

    # ---------------------------------------------------------
    # 2. Select two numerical features and target
    # ---------------------------------------------------------

    feature_cols = [
        "Ride Distance",
        "Avg VTAT",
    ]

    target_col = "Avg CTAT"

    df_clean = regression_df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[feature_cols].values
    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print("\nDataset information")
    print("-" * 60)
    print(f"Samples (N): {N}")
    print(f"Input dimension (L): {X_raw.shape[1]}")
    print(f"Output dimension (M): {y.shape[1]}")

    # ---------------------------------------------------------
    # 3. Create design matrix
    # ---------------------------------------------------------
    #
    # X_design = [1, x1, x2]
    #
    # 1  -> intercept/bias
    # x1 -> Ride Distance
    # x2 -> Avg VTAT
    #

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw,
        ]
    )

    print(f"Design matrix shape: {X_design.shape}")

    # ---------------------------------------------------------
    # 4. Standard Least Squares / Normal Equation
    # ---------------------------------------------------------
    #
    # w = (X^T X)^(-1) X^T y
    #
    # Use pseudo-inverse for numerical stability.
    #

    print("\nCalculating optimal parameters...")

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    XT_y = np.dot(
        X_design.T,
        y
    )

    try:
        XT_X_inv = np.linalg.inv(XT_X)
    except np.linalg.LinAlgError:
        print("Matrix is singular. Using pseudo-inverse.")
        XT_X_inv = np.linalg.pinv(XT_X)

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # ---------------------------------------------------------
    # 5. Display parameters
    # ---------------------------------------------------------

    print("\nOptimal Model Parameters")
    print("-" * 60)

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.6f}"
    )

    print(
        f"Coefficient for Ride Distance (w1): "
        f"{w_optimal[1, 0]:.6f}"
    )

    print(
        f"Coefficient for Avg VTAT (w2): "
        f"{w_optimal[2, 0]:.6f}"
    )

    # ---------------------------------------------------------
    # 6. Predictions
    # ---------------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # ---------------------------------------------------------
    # 7. Calculate Least Squares Error
    # ---------------------------------------------------------

    errors = y_pred - y

    E_w = 0.5 * np.sum(
        errors ** 2
    )

    mse = np.mean(
        errors ** 2
    )

    rmse = np.sqrt(mse)

    print("\nModel Error")
    print("-" * 60)

    print(f"Sum of Squared Error / 2: {E_w:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # ---------------------------------------------------------
    # 8. R² calculation
    # ---------------------------------------------------------

    ss_res = np.sum(
        (y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    print(f"R²: {r2:.4f}")

    # ---------------------------------------------------------
    # 9. Create 3D regression plane
    # ---------------------------------------------------------

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        projection="3d"
    )

    # Actual data points

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.4,
        label="Actual Data"
    )

    # Create surface grid

    x1_surface = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        30
    )

    x2_surface = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        30
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surface,
        x2_surface
    )

    # Regression plane

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3
    )

    # ---------------------------------------------------------
    # 10. Plot labels
    # ---------------------------------------------------------

    ax.set_xlabel(
        "Ride Distance"
    )

    ax.set_ylabel(
        "Avg VTAT"
    )

    ax.set_zlabel(
        "Avg CTAT"
    )

    ax.set_title(
        "RideRush Linear Regression "
        "via Standard Least Squares"
    )

    ax.legend()

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\n-> 3D regression plot saved to:"
    )

    print(output_path)

    print("\n" + "=" * 60)
    print("LAB 4 EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_linear_regression_ls()