from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.clustering import predict_cluster, run
from src.data import build_context


def _make_clustering_df(rows: int, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plot_area_ha = rng.uniform(0.3, 2.8, size=rows)
    rainfall_mm = rng.uniform(40.0, 130.0, size=rows)
    soil_ph = rng.uniform(4.6, 7.4, size=rows)
    seed_kg = rng.uniform(80.0, 350.0, size=rows)
    distance_km = rng.uniform(5.0, 30.0, size=rows)
    arrival_hour = rng.integers(0, 24, size=rows)
    actual_yield_kg = 1400 + 1000 * plot_area_ha + 35 * rainfall_mm + 80 * soil_ph + rng.normal(0, 40, rows)
    dispatch_attention = (rng.random(rows) > 0.5).astype(int)
    return pd.DataFrame(
        {
            "record_id": [f"R-{index:03d}" for index in range(rows)],
            "plot_area_ha": plot_area_ha,
            "rainfall_mm": rainfall_mm,
            "soil_ph": soil_ph,
            "seed_kg": seed_kg,
            "distance_km": distance_km,
            "arrival_hour": arrival_hour,
            "actual_yield_kg": actual_yield_kg,
            "dispatch_attention": dispatch_attention,
        }
    )


@pytest.mark.parametrize("n_rows", [20, 50, 100])
def test_clustering_labels_every_record(tmp_path, n_rows):
    df = _make_clustering_df(n_rows, seed=101 + n_rows)
    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256=f"sha-cluster-{n_rows}",
        model_version=f"AI-A1-AI-G12-cluster-{n_rows}",
    )

    result = run(ctx)
    labels = pd.read_csv(tmp_path / "artifacts" / "clusters.csv")
    assert result["status"] == "ok"
    assert len(labels) == len(df)
    assert list(labels.columns) == ["record_id", "cluster"]
    assert labels["cluster"].between(0, result["selected_k"] - 1).all()

    model = joblib.load(tmp_path / "models" / "clustering.joblib")
    record = df.iloc[0].drop(labels=["actual_yield_kg", "dispatch_attention"]).to_dict()
    predicted = predict_cluster(record, model)
    assert predicted in set(labels["cluster"].unique())


def test_target_columns_are_not_used_as_features():
    df = _make_clustering_df(20, seed=11)
    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir="artifacts",
        models_dir="models",
        seed=42,
        dataset_sha256="sha-no-targets",
        model_version="AI-A1-AI-G12-no-targets",
    )
    assert len(ctx.feature_names) == 6
    assert "actual_yield_kg" not in ctx.feature_names
    assert "dispatch_attention" not in ctx.feature_names


def test_silhouette_scores_cover_k_2_to_5(tmp_path):
    df = _make_clustering_df(80, seed=7)
    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256="sha-silhouette",
        model_version="AI-A1-AI-G12-silhouette",
    )
    run(ctx)
    with (tmp_path / "artifacts" / "clustering_metrics.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {str(v) for v in [2, 3, 4, 5]}
    assert expected.issubset(set(payload["silhouette_per_k"].keys()))
