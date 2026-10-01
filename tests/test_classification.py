from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.classification import predict_attention, run
from src.data import build_context


def _make_classification_df(rows: int, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plot_area_ha = rng.uniform(0.2, 2.5, size=rows)
    rainfall_mm = rng.uniform(50.0, 120.0, size=rows)
    soil_ph = rng.uniform(4.5, 7.5, size=rows)
    seed_kg = rng.uniform(50.0, 400.0, size=rows)
    distance_km = rng.uniform(2.0, 35.0, size=rows)
    arrival_hour = rng.integers(0, 24, size=rows)

    logits = (
        1.2 * plot_area_ha
        - 0.7 * rainfall_mm / 100
        + 0.8 * soil_ph
        - 0.2 * seed_kg / 100
        + 0.15 * distance_km / 10
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random(rows) < probs).astype(int)
    if labels.sum() < 2:
        labels[:2] = 1
    if (labels == 0).sum() < 2:
        labels[-2:] = 0

    return pd.DataFrame(
        {
            "record_id": [f"R-{index:03d}" for index in range(rows)],
            "plot_area_ha": plot_area_ha,
            "rainfall_mm": rainfall_mm,
            "soil_ph": soil_ph,
            "seed_kg": seed_kg,
            "distance_km": distance_km,
            "arrival_hour": arrival_hour,
            "actual_yield_kg": 1800 + 900 * plot_area_ha + rng.normal(0, 50, rows),
            "dispatch_attention": labels,
        }
    )


@pytest.mark.parametrize("n_rows", [20, 50, 100])
def test_classification_runs_for_multiple_row_counts(tmp_path, n_rows):
    df = _make_classification_df(n_rows, seed=123 + n_rows)
    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256=f"sha-{n_rows}",
        model_version=f"AI-A1-AI-G12-{n_rows}",
    )

    result = run(ctx)
    assert result["status"] == "ok"
    assert (tmp_path / "artifacts" / "classification_metrics.json").exists()
    assert (tmp_path / "models" / "classifier.joblib").exists()

    with (tmp_path / "artifacts" / "classification_metrics.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["n_test"] > 0
    assert payload["accuracy"] >= 0.0
    assert payload["roc_auc"] is None or 0.0 <= payload["roc_auc"] <= 1.0


def test_stratified_fallback_when_class_has_one_sample(tmp_path):
    labels = np.array([0] * 9 + [1])
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "record_id": [f"R-{index:03d}" for index in range(len(labels))],
            "plot_area_ha": rng.uniform(0.2, 2.5, size=len(labels)),
            "rainfall_mm": rng.uniform(50.0, 120.0, size=len(labels)),
            "soil_ph": rng.uniform(4.5, 7.5, size=len(labels)),
            "seed_kg": rng.uniform(50.0, 400.0, size=len(labels)),
            "distance_km": rng.uniform(2.0, 35.0, size=len(labels)),
            "arrival_hour": rng.integers(0, 24, size=len(labels)),
            "actual_yield_kg": 1800 + 900 * rng.uniform(0.2, 2.5, len(labels)),
            "dispatch_attention": labels,
        }
    )
    ctx = build_context(
        data,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256="sha-fallback",
        model_version="AI-A1-AI-G12-fallback",
    )

    result = run(ctx)
    with (tmp_path / "artifacts" / "classification_metrics.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert result["status"] == "ok"
    assert payload["stratified"] is False
    assert payload["n_test"] > 0


def test_confusion_matrix_sums_to_n_test(tmp_path):
    df = _make_classification_df(60, seed=99)
    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256="sha-cm",
        model_version="AI-A1-AI-G12-cm",
    )
    run(ctx)
    with (tmp_path / "artifacts" / "classification_metrics.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cm_total = sum(sum(row) for row in payload["confusion_matrix"])
    assert cm_total == payload["n_test"]


def test_scaler_does_not_use_test_data(tmp_path):
    df = _make_classification_df(80, seed=7)
    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256="sha-scaler",
        model_version="AI-A1-AI-G12-scaler",
    )

    X_valid = ctx.X
    y_valid = ctx.y_clf.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=ctx.seed, stratify=y_valid)

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    scaler.fit(train_imputed)
    result = run(ctx)
    assert result["status"] == "ok"

    import joblib

    fitted = joblib.load(tmp_path / "models" / "classifier.joblib")
    assert np.allclose(fitted.named_steps["scaler"].mean_, scaler.mean_)
    assert not np.allclose(fitted.named_steps["scaler"].mean_, X_valid.mean(axis=0))

    record = {
        "plot_area_ha": 1.5,
        "rainfall_mm": 80.0,
        "soil_ph": 5.9,
        "seed_kg": 220,
        "distance_km": 12.5,
        "arrival_hour": 9,
    }
    label, prob = predict_attention(record, fitted, threshold=0.5)
    assert label in {0, 1}
    assert 0.0 <= prob <= 1.0
