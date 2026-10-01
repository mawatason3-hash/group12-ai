"""Regression from first principles with NumPy only.

Trains a linear regressor with batch gradient descent. Fits imputer and
scaler on TRAIN ONLY. Saves both scaled and original-units weights so that
predict_yield can apply raw-input + original-units weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .common import PipelineContext, write_json_atomic


# ----------------------------------------------------------------------------
# Public helper used by predict.py
# ----------------------------------------------------------------------------
def predict_yield(record_dict_or_array: dict[str, float] | np.ndarray,
                  model_json: dict[str, Any]) -> float:
    """Predict yield using ONLY raw input + original-units weights."""
    feature_order = list(model_json["feature_order"])
    weights = np.asarray(model_json["weights"], dtype=float)
    bias = float(model_json["bias"])
    medians = np.asarray(model_json.get("imputation_medians", []), dtype=float)

    if isinstance(record_dict_or_array, dict):
        row = np.asarray([float(record_dict_or_array[k]) for k in feature_order],
                         dtype=float)
    else:
        row = np.asarray(record_dict_or_array, dtype=float)
        if row.shape[0] != len(feature_order):
            raise ValueError("Feature count mismatch for prediction input.")

    if medians.size:
        row = np.where(np.isnan(row), medians, row)

    return float(np.dot(row, weights) + bias)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    resid = X @ w - y
    return float(np.mean(resid * resid))


def _gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    resid = X @ w - y
    return (2.0 / n) * (X.T @ resid)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run(ctx: PipelineContext) -> dict[str, Any]:
    """Train regression on ALL valid rows. Save metrics and model bundle."""
    # ---- 1. Determine valid rows (finite target) ---------------------------
    y_all = np.asarray(ctx.y_reg, dtype=float)
    valid_mask = np.isfinite(y_all)
    X_all = np.asarray(ctx.X, dtype=float)

    # Make sure X has one row per df row
    if X_all.shape[0] != len(ctx.df):
        raise ValueError(
            f"ctx.X has {X_all.shape[0]} rows but df has {len(ctx.df)} rows"
        )

    X_valid = X_all[valid_mask]
    y_valid = y_all[valid_mask]

    n_valid = X_valid.shape[0]
    if n_valid == 0:
        raise ValueError("No rows with a finite regression target.")

    # ---- 2. Split (fit imputer+scaler on TRAIN ONLY) -----------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_valid, y_valid,
        test_size=ctx.seed and 0.2 or 0.2,
        random_state=ctx.seed,
    )

    # Impute medians from TRAIN
    train_medians = np.nanmedian(X_train_raw, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)

    def _impute(X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), train_medians, X)

    X_train_imp = _impute(X_train_raw)
    X_test_imp = _impute(X_test_raw)

    # Scaler stats from TRAIN
    scaler_mean = X_train_imp.mean(axis=0)
    scaler_std = X_train_imp.std(axis=0)
    scaler_std = np.where(scaler_std == 0.0, 1.0, scaler_std)

    def _scale(X: np.ndarray) -> np.ndarray:
        return (X - scaler_mean) / scaler_std

    X_train = _scale(X_train_imp)
    X_test = _scale(X_test_imp)

    # Augment with bias column
    X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # ---- 3. Batch gradient descent (NumPy only) ----------------------------
    learning_rate = 0.05
    n_iterations = 2000

    w = np.zeros(X_train_aug.shape[1], dtype=float)
    loss_history: list[float] = []
    converged = True

    for _ in range(n_iterations):
        loss = _mse(X_train_aug, y_train, w)
        loss_history.append(loss)
        if not np.isfinite(loss):
            converged = False
            break
        grad = _gradient(X_train_aug, y_train, w)
        w = w - learning_rate * grad

    final_loss = loss_history[-1] if loss_history else float("nan")

    # ---- 4. Metrics on scaled space ----------------------------------------
    y_train_pred_scaled = X_train_aug @ w
    y_test_pred_scaled = X_test_aug @ w

    train_metrics_scaled = _metrics(y_train, y_train_pred_scaled)
    test_metrics_scaled = _metrics(y_test, y_test_pred_scaled)

    # ---- 5. Normal-equation sanity check -----------------------------------
    try:
        lstsq_w, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
        y_train_lstsq = X_train_aug @ lstsq_w
        lstsq_r2 = _metrics(y_train, y_train_lstsq)["r2"]
    except np.linalg.LinAlgError:
        lstsq_r2 = float("nan")

    # ---- 6. Convert scaled weights -> original-units weights ---------------
    w_scaled = w[1:].copy()
    b_scaled = float(w[0])

    # y = b_scaled + sum(w_scaled_i * (x_i - mean_i)/std_i)
    #   = (b_scaled - sum(w_scaled_i * mean_i / std_i)) + sum(w_scaled_i/std_i * x_i)
    weights_original = w_scaled / scaler_std
    bias_original = b_scaled - float(np.sum(w_scaled * scaler_mean / scaler_std))

    # Sanity: original-units metrics should match scaled metrics
    def _predict_original(X_raw: np.ndarray) -> np.ndarray:
        Xi = _impute(X_raw)
        return bias_original + Xi @ weights_original

    y_train_pred = _predict_original(X_train_raw)
    y_test_pred = _predict_original(X_test_raw)
    train_metrics = _metrics(y_train, y_train_pred)
    test_metrics = _metrics(y_test, y_test_pred)

    # ---- 7. Write metrics JSON ---------------------------------------------
    metrics_payload = {
        "seed": ctx.seed,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
        "n_train": int(X_train_raw.shape[0]),
        "n_test": int(X_test_raw.shape[0]),
        "final_train_loss": float(final_loss),
        "mae": {"train": train_metrics["mae"], "test": test_metrics["mae"]},
        "rmse": {"train": train_metrics["rmse"], "test": test_metrics["rmse"]},
        "r2": {"train": train_metrics["r2"], "test": test_metrics["r2"]},
        "coefficients": {name: float(weights_original[i])
                         for i, name in enumerate(ctx.feature_names)},
        "coefficients_scaled": {name: float(w_scaled[i])
                                for i, name in enumerate(ctx.feature_names)},
        "bias": float(bias_original),
        "converged": bool(converged),
        "lstsq_r2": float(lstsq_r2),
        "comparison_note": (
            "Gradient descent weights are compared with a least-squares fit "
            "on the same train split."
        ),
        "warnings": [] if converged else ["gradient descent did not converge"],
    }

    metrics_path = ctx.output_dir / "regression_metrics.json"
    write_json_atomic(metrics_path, metrics_payload)

    # ---- 8. Loss curve -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(loss_history)), loss_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE Loss (scaled)")
        ax.set_title(f"Regression loss — lr={learning_rate}, n_iter={n_iterations}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "regression_loss.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        metrics_payload.setdefault("warnings", []).append(
            f"loss plot failed: {exc}"
        )

    # ---- 9. Save model bundle ---------------------------------------------
    model_payload = {
        "feature_order": list(ctx.feature_names),
        "imputation_medians": train_medians.tolist(),
        "scaler_mean": scaler_mean.tolist(),
        "scaler_std": scaler_std.tolist(),
        "weights": weights_original.tolist(),
        "bias": float(bias_original),
        "weights_scaled": w_scaled.tolist(),
        "bias_scaled": float(b_scaled),
    }

    model_path = ctx.models_dir / "regression_model.json"
    write_json_atomic(model_path, model_payload)

    return {
        "status": "ok",
        "artifact": str(metrics_path),
        "model": str(model_path),
        "n_train": int(X_train_raw.shape[0]),
        "n_test": int(X_test_raw.shape[0]),
        "r2_test": test_metrics["r2"],
        "rmse_test": test_metrics["rmse"],
    }


# ---------------------------------------------------------------------------
# Compatibility wrapper for tests/test_regression.py
# ---------------------------------------------------------------------------
def gradient_descent(X: "np.ndarray", y: "np.ndarray",
                     learning_rate: float = 0.05,
                     n_iterations: int = 2000,
                     seed: int = 42) -> tuple["np.ndarray", list]:
    """Run batch gradient descent on an already-augmented design matrix.

    Returns (weights, loss_history). Uses the same update rule as run():
        w := w - lr * (2/n) * X.T @ (Xw - y)
    """
    import numpy as _np

    X = _np.asarray(X, dtype=float)
    y = _np.asarray(y, dtype=float)

    n_features = X.shape[1]
    w = _np.zeros(n_features, dtype=float)
    loss_history: list = []

    for _ in range(n_iterations):
        resid = X @ w - y
        loss = float(_np.mean(resid * resid))
        loss_history.append(loss)
        if not _np.isfinite(loss):
            break
        grad = (2.0 / X.shape[0]) * (X.T @ resid)
        w = w - learning_rate * grad

    return w, loss_history
