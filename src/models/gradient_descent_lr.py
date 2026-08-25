import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# ---------------------------------------------------------
# Cost Function
# ---------------------------------------------------------

def compute_cost(X, y, w):
    """
    Compute Mean Squared Error / 2.

    J(w) = (1 / 2m) * sum((Xw - y)^2)
    """

    m = len(y)

    predictions = np.dot(X, w)

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


# ---------------------------------------------------------
# Gradient Descent
# ---------------------------------------------------------

def gradient_descent(X, y, w, alpha, num_iters):
    """
    Perform Gradient Descent from scratch using NumPy.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        # Predictions
        predictions = np.dot(X, w)

        # Prediction errors
        errors = predictions - y

        # Gradient
        gradient = (1 / m) * np.dot(X.T, errors)

        # Update weights
        w = w - alpha * gradient

        # Calculate cost
        cost = compute_cost(X, y, w)

        cost_history.append(cost)

    return w, cost_history


# ---------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------

def run_gradient_descent_experiment():

    print("=" * 60)
    print("RIDERUSH LINEAR REGRESSION - GRADIENT DESCENT")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load and preprocess data
    # -----------------------------------------------------

    df = load_data(DATA_PATH)

    df = clean_data(df)

    regression_df = prepare_regression_data(df)

    # -----------------------------------------------------
    # 2. Select feature and target
    # -----------------------------------------------------

    feature_cols = ["Ride Distance"]
    target_col = "Avg CTAT"

    df_clean = regression_df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[feature_cols].values

    y_raw = df_clean[target_col].values.reshape(-1, 1)

    print("\nDataset information")
    print("-" * 60)
    print(f"Samples: {len(df_clean)}")
    print(f"Input feature: {feature_cols[0]}")
    print(f"Target: {target_col}")

    # -----------------------------------------------------
    # 3. Train-test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # -----------------------------------------------------
    # 4. Feature scaling
    # -----------------------------------------------------

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)

    # -----------------------------------------------------
    # 5. Add intercept column
    # -----------------------------------------------------

    X_train_design = np.hstack(
        [
            np.ones((X_train_scaled.shape[0], 1)),
            X_train_scaled
        ]
    )

    X_test_design = np.hstack(
        [
            np.ones((X_test_scaled.shape[0], 1)),
            X_test_scaled
        ]
    )

    # -----------------------------------------------------
    # 6. Learning rate experiment
    # -----------------------------------------------------

    learning_rates = [0.001, 0.01, 0.1, 0.5]

    num_iterations = 1000

    results = {}

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(10, 6))

    print("\nLearning Rate Experiment")
    print("-" * 60)

    for alpha in learning_rates:

        # Initialize weights
        w_init = np.zeros(
            (X_train_design.shape[1], 1)
        )

        # Run gradient descent
        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        results[alpha] = {
            "weights": w_opt,
            "history": cost_history
        }

        print(
            f"Alpha = {alpha:<6} "
            f"Initial Cost = {cost_history[0]:.6f} "
            f"Final Cost = {cost_history[-1]:.6f}"
        )

        # Plot cost history
        plt.plot(
            cost_history,
            label=f"α = {alpha}"
        )

    plt.xlabel("Iterations")
    plt.ylabel("Cost J(w)")
    plt.title(
        "Gradient Descent Convergence "
        "for Different Learning Rates"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(output_path)

    plt.close()

    print(
        f"\n-> Saved learning-rate graph to:\n"
        f"{output_path}"
    )

    # -----------------------------------------------------
    # 7. Select best learning rate
    # -----------------------------------------------------

    best_alpha = 0.1

    final_w = results[best_alpha]["weights"]

    print("\n" + "=" * 60)
    print(
        f"FINAL GRADIENT DESCENT MODEL "
        f"(α = {best_alpha})"
    )
    print("=" * 60)

    print(f"Intercept (w0): {final_w[0, 0]:.6f}")

    print(
        f"Coefficient for Ride Distance (w1): "
        f"{final_w[1, 0]:.6f}"
    )

    # -----------------------------------------------------
    # 8. Evaluate custom Gradient Descent
    # -----------------------------------------------------

    y_pred_scaled = np.dot(
        X_test_design,
        final_w
    )

    # Convert predictions back to original CTAT scale
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\nGradient Descent Results")
    print("-" * 60)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    # -----------------------------------------------------
    # 9. Scikit-learn comparison
    # -----------------------------------------------------

    sklearn_model = LinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    sklearn_pred_scaled = sklearn_model.predict(
        X_test_scaled
    )

    sklearn_pred = scaler_y.inverse_transform(
        sklearn_pred_scaled
    )

    sklearn_mae = mean_absolute_error(
        y_test,
        sklearn_pred
    )

    sklearn_mse = mean_squared_error(
        y_test,
        sklearn_pred
    )

    sklearn_rmse = np.sqrt(
        sklearn_mse
    )

    sklearn_r2 = r2_score(
        y_test,
        sklearn_pred
    )

    print("\nScikit-learn Linear Regression")
    print("-" * 60)

    print(
        f"Intercept: "
        f"{sklearn_model.intercept_[0]:.6f}"
    )

    print(
        f"Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.6f}"
    )

    print(f"MAE : {sklearn_mae:.4f}")
    print(f"RMSE: {sklearn_rmse:.4f}")
    print(f"R²  : {sklearn_r2:.4f}")

    # -----------------------------------------------------
    # 10. Final comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "NumPy Gradient Descent",
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            },
            {
                "Model": "Scikit-learn Linear Regression",
                "MAE": sklearn_mae,
                "RMSE": sklearn_rmse,
                "R2": sklearn_r2
            }
        ]
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(index=False)
    )

    print("\n")
    print("=" * 60)
    print("LAB 5 EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_gradient_descent_experiment()