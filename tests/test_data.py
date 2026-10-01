from __future__ import annotations

import pandas as pd
import pytest

from src.config import FEATURES, ID_COL, TARGET_CLF, TARGET_REG
from src.data import build_context, load_and_validate


REQUIRED_COLUMNS = [ID_COL, *FEATURES, TARGET_REG, TARGET_CLF]


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: [f"R-{index:03d}" for index in range(1, 11)],
            "plot_area_ha": [1.2, 1.5, 2.1, 0.8, 1.9, 2.4, 1.1, 1.7, 2.0, 1.3],
            "rainfall_mm": [450, 520, 610, 430, 540, 600, 470, 580, 560, 490],
            "soil_ph": [5.8, 6.1, 6.0, 5.9, 6.2, 6.3, 5.7, 6.0, 6.1, 5.8],
            "seed_kg": [80, 95, 110, 70, 100, 120, 68, 105, 98, 75],
            "distance_km": [12.5, 18.0, 9.4, 14.2, 11.8, 16.9, 13.7, 15.1, 10.2, 17.5],
            "arrival_hour": [7, 8, 6, 9, 7, 10, 8, 7, 6, 9],
            TARGET_REG: [2000, 2100, 2600, 1800, 2300, 2750, 1750, 2400, 2250, 1950],
            TARGET_CLF: [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_load_and_validate_valid_file(tmp_path):
    csv_path = tmp_path / "valid.csv"
    _make_valid_df().to_csv(csv_path, index=False)

    df, report = load_and_validate(csv_path)

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 10
    assert report["row_count"] == 10
    assert report["duplicate_counts"]["record_id_duplicates"] == 0
    assert report["rows_dropped"] == 0

    ctx = build_context(
        df,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256="abc123",
        model_version="musanze-harvest-AI-G12-abc12345",
    )
    assert ctx.X.shape == (10, 6)
    assert ctx.y_reg.shape == (10,)


def test_load_and_validate_missing_column(tmp_path):
    csv_path = tmp_path / "missing_column.csv"
    df = _make_valid_df().drop(columns=["soil_ph"])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_and_validate(csv_path)


def test_load_and_validate_bad_type_values(tmp_path):
    csv_path = tmp_path / "bad_type.csv"
    df = _make_valid_df().copy()
    df["rainfall_mm"] = ["good", "bad", "700", "580", "610", "495", "530", "620", "unknown", "545"]
    df[TARGET_CLF] = [1, 0, 2, 0, 1, 0, 1, 0, 1, 1]
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="dispatch_attention"):
        load_and_validate(csv_path)


def test_load_and_validate_duplicate_record_ids(tmp_path):
    csv_path = tmp_path / "duplicates.csv"
    df = _make_valid_df()
    df.loc[0, ID_COL] = "R-001"
    df.loc[5, ID_COL] = "R-001"
    df.to_csv(csv_path, index=False)

    cleaned, report = load_and_validate(csv_path)

    assert report["duplicate_counts"]["record_id_duplicates"] == 2
    assert report["rows_dropped"] == 1
    assert cleaned[ID_COL].duplicated().sum() == 0


def test_load_and_validate_missing_values(tmp_path):
    csv_path = tmp_path / "missing_values.csv"
    df = _make_valid_df()
    df.loc[3, "rainfall_mm"] = None
    df.loc[7, "soil_ph"] = None
    df.to_csv(csv_path, index=False)

    cleaned, report = load_and_validate(csv_path)

    assert report["missing_values"]["rainfall_mm"] > 0
    assert report["missing_values"]["soil_ph"] > 0
    assert cleaned["rainfall_mm"].isna().sum() >= 1
    ctx = build_context(
        cleaned,
        group_code="AI-G12",
        output_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        seed=42,
        dataset_sha256="def456",
        model_version="musanze-harvest-AI-G12-def45678",
    )
    assert pd.isna(ctx.X).any()


def test_load_and_validate_preserves_row_count_reporting(tmp_path):
    csv_path = tmp_path / "different_row_count.csv"
    df = _make_valid_df().iloc[:8].copy()
    df.to_csv(csv_path, index=False)

    cleaned, report = load_and_validate(csv_path)

    assert len(cleaned) == 8
    assert report["row_count"] == 8
    assert report["feature_count"] == 6
