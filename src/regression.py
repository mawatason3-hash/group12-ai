from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .common import PipelineContext, write_json_atomic


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error for the provided targets and predictions."""
    return float(np.mean((y_pred - y_true) ** 2))


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.05,
    n_iterations: int = 2000,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], bool]:
    """Train a linear model via batch gradient descent with a divergence guard."""
    if X.size == 0:
        raise ValueError("Training data is empty.")
    weights = np.zeros(X.shape[1], dtype=float) if initial_weights is None else initial_weights.astype(float)
    losses: list[float] = []
    float_lr = float(learning_rate)
    converged = True
    previous_loss: float | None = None

    for _ in range(int(n_iterations)):
        predictions = X @ weights
        loss = mse_loss(y, predictions)
        if not np.isfinite(loss):
            converged = False
            break
        losses.append(float(loss))
        if previous_loss is not None and loss > previous_loss:
            float_lr *= 0.5
        previous_loss = loss
        gradient = (2.0 / len(X)) * (X.T @ (predictions - y))
        weights = weights - float_lr * gradient
        if not np.all(np.isfinite(weights)):
            converged = False
            break
    return weights, losses, converged


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination R^2."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return 1.0 - (ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict_yield(record_dict_or_array: dict[str, float] | np.ndarray, model_json: dict[str, Any]) -> float:
    """Predict yield for a single row using the saved model parameters."""
    feature_order = list(model_json.get("feature_order", []))
    weights = np.asarray(model_json.get("weights", []), dtype=float)
    bias = float(model_json.get("bias", 0.0))
    imputation_medians = np.asarray(model_json.get("imputation_medians", []), dtype=float)
    scaler_mean = np.asarray(model_json.get("scaler_mean", []), dtype=float)
    scaler_std = np.asarray(model_json.get("scaler_std", []), dtype=float)

    if isinstance(record_dict_or_array, dict):
        values = [float(record_dict_or_array[key]) for key in feature_order]
        row = np.asarray(values, dtype=float)
    else:
        row = np.asarray(record_dict_or_array, dtype=float)
        if row.shape[0] != len(feature_order):
            raise ValueError("Feature count mismatch for prediction input.")

    if imputation_medians.size:
        row = np.where(np.isnan(row), imputation_medians, row)
    if scaler_mean.size:
        row = (row - scaler_mean) / np.where(scaler_std == 0.0, 1.0, scaler_std)
    return float(np.dot(row, weights) + bias)


def run(ctx: PipelineContext) -> dict[str, Any]:
    """Train a NumPy-only linear regression model, then save the metrics and model files."""
    valid_rows = np.isfinite(ctx.y_reg)
    X_valid = ctx.X[valid_rows]
    y_valid = ctx.y_reg[valid_rows].astype(float)

    if X_valid.size == 0 or y_valid.size == 0:
        raise ValueError("No valid target rows remain for regression training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=0.2,
        random_state=ctx.seed,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_aug = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
    X_test_aug = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    weights, loss_history, converged = gradient_descent(
        X_train_aug,
        y_train,
        learning_rate=0.05,
        n_iterations=2000,
    )

    train_pred = X_train_aug @ weights
    test_pred = X_test_aug @ weights

    train_mae = _mae(y_train, train_pred)
    train_rmse = _rmse(y_train, train_pred)
    train_r2 = _r2_score(y_train, train_pred)
    test_mae = _mae(y_test, test_pred)
    test_rmse = _rmse(y_test, test_pred)
    test_r2 = _r2_score(y_test, test_pred)
    final_train_loss = mse_loss(y_train, train_pred)

    lstsq_coeffs, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    lstsq_r2 = _r2_score(y_train, X_train_aug @ lstsq_coeffs)

    coef_map = {}
    coef_scaled_map = {}
    for idx, feature in enumerate(ctx.feature_names):
        scaled = weights[idx + 1] / (scaler.scale_[idx] if scaler.scale_[idx] != 0.0 else 1.0)
        coef_map[feature] = float(scaled)
        coef_scaled_map[feature] = float(weights[idx + 1])
    bias = float(weights[0])
    bias_original = float(bias - np.sum(np.asarray([coef_map[f] * scaler.mean_[idx] for idx, f in enumerate(ctx.feature_names)])))

    model_payload = {
        "feature_order": list(ctx.feature_names),
        "imputation_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "weights": np.asarray(weights[1:]).tolist(),
        "bias": float(bias_original),
    }
    model_path = ctx.models_dir / "regression_model.json"
    write_json_atomic(model_path, model_payload)

    loss_path = ctx.output_dir / "regression_loss.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="tab:blue", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Regression Training Loss Curve")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    metrics = {
        "seed": ctx.seed,
        "learning_rate": 0.05,
        "n_iterations": 2000,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "final_train_loss": float(final_train_loss),
        "mae": {"train": float(train_mae), "test": float(test_mae)},
        "rmse": {"train": float(train_rmse), "test": float(test_rmse)},
        "r2": {"train": float(train_r2), "test": float(test_r2)},
        "coefficients": coef_map,
        "coefficients_scaled": coef_scaled_map,
        "bias": float(bias_original),
        "converged": bool(converged),
        "lstsq_r2": float(lstsq_r2),
        "comparison_note": "Gradient descent weights are compared with a least-squares fit on the same train split.",
        "warnings": [],
    }
    metrics_path = ctx.output_dir / "regression_metrics.json"
    write_json_atomic(metrics_path, metrics)

    return {
        "stage": "regression",
        "artifact_path": str(metrics_path),
        "status": "ok",
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
    }
