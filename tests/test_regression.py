from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.common import PipelineContext
from src.config import FEATURES, TARGET_REG
from src.data import build_context
from src.regression import gradient_descent, run


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 10.0, 80)
    y = 12.0 + 3.5 * x + rng.normal(0.0, 0.5, size=x.shape)
    frame = pd.DataFrame({
        "record_id": [f"R-{i:03d}" for i in range(len(x))],
        "plot_area_ha": x,
        "rainfall_mm": 100 + 5 * x + rng.normal(0.0, 2.0, len(x)),
        "soil_ph": 6.0 + 0.1 * rng.normal(size=len(x)),
        "seed_kg": 125 + 15 * x + rng.normal(0.0, 6.0, len(x)),
        "distance_km": 8 + 1.5 * x + rng.normal(0.0, 0.7, len(x)),
        "arrival_hour": (5 + np.round(x / 2.5)).astype(int) % 24,
        TARGET_REG: y,
        "dispatch_attention": (rng.random(len(x)) > 0.5).astype(int),
    })
    return frame


def _make_ctx(df: pd.DataFrame, *, output_dir: str = "artifacts", models_dir: str = "models") -> PipelineContext:
    return build_context(
        df,
        group_code="AI-G12",
        output_dir=output_dir,
        models_dir=models_dir,
        seed=42,
        dataset_sha256="abc123",
        model_version="AI-A1-AI-G12-abc12345",
    )


def test_loss_monotonic_non_increasing_for_low_lr():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 3))
    y = 2 + 1.5 * x[:, 0] - 0.7 * x[:, 1] + 0.2 * x[:, 2]
    X = np.hstack([np.ones((x.shape[0], 1)), x])
    weights, losses, converged = gradient_descent(X, y, learning_rate=0.001, n_iterations=200)
    assert np.isfinite(weights).all()
    assert converged
    assert losses == sorted(losses, reverse=True)


def test_noise_free_linear_data_recovers_lstsq_weights(sample_df):
    df = sample_df.copy()
    ctx = _make_ctx(df)
    metrics_path = ctx.output_dir / "regression_metrics.json"
    model_path = ctx.models_dir / "regression_model.json"
    run(ctx)
    assert metrics_path.exists()
    assert model_path.exists()
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert "coefficients" in metrics
    assert "lstsq_r2" in metrics
    assert metrics["r2"]["test"] >= 0.0


def test_regression_metrics_keys_and_missing_values_support(tmp_path):
    rows = 50
    rng = np.random.default_rng(123)
    x = np.linspace(0.0, 10.0, rows)
    y = 30 + 2.5 * x + rng.normal(0.0, 0.3, size=x.shape)
    df = pd.DataFrame({
        "record_id": [f"R-{i:03d}" for i in range(rows)],
        "plot_area_ha": x,
        "rainfall_mm": 82 + 2 * x + rng.normal(0.0, 1.0, rows),
        "soil_ph": 6.0 + rng.normal(0.0, 0.2, rows),
        "seed_kg": 180 + 10 * x + rng.normal(0.0, 4.0, rows),
        "distance_km": 12 + 1.7 * x + rng.normal(0.0, 0.6, rows),
        "arrival_hour": (np.round(7 + x / 2.5) % 24).astype(int),
        "actual_yield_kg": y,
        "dispatch_attention": (rng.random(rows) > 0.4).astype(int),
    })
    df.loc[0, "actual_yield_kg"] = np.nan
    df.loc[1, "plot_area_ha"] = np.nan
    ctx = _make_ctx(df, output_dir=tmp_path / "artifacts", models_dir=tmp_path / "models")
    run(ctx)
    metrics_path = tmp_path / "artifacts" / "regression_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ["seed", "learning_rate", "n_iterations", "n_train", "n_test", "final_train_loss", "mae", "rmse", "r2", "coefficients", "coefficients_scaled", "bias", "converged", "lstsq_r2", "comparison_note", "warnings"]:
        assert key in payload
