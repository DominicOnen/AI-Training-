"""
Regression from first principles (assignment section 2).

Implements linear regression and batch gradient descent using only NumPy —
no sklearn estimator is used to fit the model itself (StandardScaler is
used only for feature scaling, which the assignment explicitly asks for
separately from "the NumPy gradient descent section").
"""

import time

import matplotlib
matplotlib.use("Agg")  # headless — never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import RANDOM_SEED, ensure_dir, save_json


class LinearRegressionGD:
    """
    Plain-NumPy linear regression trained with batch gradient descent.

    y_hat = X @ weights + bias

    Loss is mean squared error. Gradients are the standard closed-form
    derivatives of MSE with respect to weights and bias.
    """

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000, seed: int = RANDOM_SEED):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.seed = seed
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape

        # Small random init rather than zeros, so weights aren't symmetric
        # by coincidence — doesn't matter much for plain linear regression,
        # but keeps the implementation honest about "training" from a
        # non-trivial starting point.
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # MSE loss, recorded every iteration for regression_loss.png
            loss = float(np.mean(error ** 2))
            self.loss_history.append(loss)

            # Gradients of MSE w.r.t. weights and bias
            grad_weights = (2.0 / n_samples) * (X.T @ error)
            grad_bias = (2.0 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fit yet.")
        return X @ self.weights + self.bias


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def run_regression(X: np.ndarray, y: np.ndarray, output_dir: str, models_dir: str,
                    learning_rate: float = 0.05, n_iterations: int = 1000,
                    seed: int = RANDOM_SEED) -> dict:
    """
    Full regression workflow: split, scale (fit on train only), train with
    gradient descent, evaluate on the held-out test set, and save
    regression_metrics.json + regression_loss.png + the fitted model.
    """
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit on TRAIN only
    X_test_scaled = scaler.transform(X_test)          # test uses train's stats

    model = LinearRegressionGD(learning_rate=learning_rate, n_iterations=n_iterations, seed=seed)
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)

    metrics = {
        "random_seed": seed,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "final_training_loss_mse": model.loss_history[-1],
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test),
        "test_r_squared": r_squared(y_test, y_pred_test),
        "weights": model.weights,
        "bias": model.bias,
        "sample_predictions": {
            "y_true": y_test[:10],
            "y_pred": y_pred_test[:10],
        },
        "runtime_seconds": round(time.time() - start, 4),
    }

    ensure_dir(output_dir)
    save_json(metrics, f"{output_dir}/regression_metrics.json")

    # Loss curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(model.loss_history, color="#c0602a")
    plt.title("Regression Training Loss (MSE) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regression_loss.png", dpi=150)
    plt.close()

    ensure_dir(models_dir)
    import joblib
    joblib.dump(
        {"model": model, "scaler": scaler},
        f"{models_dir}/regression_model.joblib",
    )

    return metrics
