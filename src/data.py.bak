from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import PipelineContext, write_json_atomic
from .config import FEATURES, ID_COL, TARGET_CLF, TARGET_REG

REQUIRED_COLUMNS = [ID_COL, *FEATURES, TARGET_REG, TARGET_CLF]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the raw file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a CSV, validate required schema, and return a cleaned DataFrame with a report."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    unexpected_columns = [column for column in df.columns if column not in REQUIRED_COLUMNS]
    if missing_columns or unexpected_columns:
        message_parts = []
        if missing_columns:
            message_parts.append(f"Missing columns: {missing_columns}")
        if unexpected_columns:
            message_parts.append(f"Unexpected columns: {unexpected_columns}")
        raise ValueError("; ".join(message_parts))

    numeric_columns = [*FEATURES, TARGET_REG, TARGET_CLF]
    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "feature_count": len(FEATURES),
        "feature_names": list(FEATURES),
        "missing_values": {},
        "coerced_to_nan": {},
        "duplicate_counts": {
            "record_id_duplicates": 0,
            "full_row_duplicates": 0,
        },
        "rows_dropped": 0,
        "warnings": [],
    }

    for column in numeric_columns:
        original = df[column].copy()
        converted = pd.to_numeric(df[column], errors="coerce")
        report["missing_values"][column] = int(converted.isna().sum())
        report["coerced_to_nan"][column] = int((converted.isna() & original.notna()).sum())
        df[column] = converted

    record_id_dupes = df[ID_COL].duplicated(keep=False)
    full_row_dupes = df.duplicated(keep=False)
    report["duplicate_counts"]["record_id_duplicates"] = int(record_id_dupes.sum())
    report["duplicate_counts"]["full_row_duplicates"] = int(full_row_dupes.sum())
    duplicate_mask = df[ID_COL].duplicated(keep="first")
    report["rows_dropped"] = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    if report["duplicate_counts"]["record_id_duplicates"] > 0:
        report["warnings"].append(
            f"{report['duplicate_counts']['record_id_duplicates']} duplicate record_id values were detected; later records were dropped."
        )
    if any(report["coerced_to_nan"].values()):
        report["warnings"].append("Some values were coerced to NaN during numeric conversion.")
    if any(report["missing_values"].values()):
        report["warnings"].append("Missing values remain present and must be handled by the modeling stages.")

    invalid_mask = df[TARGET_CLF].notna() & ~df[TARGET_CLF].isin([0, 1])
    if invalid_mask.any():
        invalid_values = sorted({value for value in df.loc[invalid_mask, TARGET_CLF].tolist() if pd.notna(value)})
        raise ValueError(f"dispatch_attention must be binary 0/1 after coercion; found invalid values: {invalid_values}")

    return df, report


def build_context(
    df: pd.DataFrame,
    *,
    group_code: str,
    output_dir: str | Path,
    models_dir: str | Path,
    seed: int,
    dataset_sha256: str,
    model_version: str,
) -> PipelineContext:
    """Construct a shared PipelineContext for all downstream stages."""
    feature_names = list(FEATURES)
    X = df[feature_names].to_numpy(dtype=float, copy=True)
    assert X.shape[0] == len(df), "ctx.X must have one row per dataset row"
    ids = df[ID_COL].astype(str).to_numpy()
    y_reg = df[TARGET_REG].to_numpy(dtype=float, copy=True)
    y_clf = df[TARGET_CLF].to_numpy(dtype=float, copy=True)

    return PipelineContext(
        df=df.copy(),
        X=X,
        feature_names=feature_names,
        ids=ids,
        y_reg=y_reg,
        y_clf=y_clf,
        output_dir=output_dir,
        models_dir=models_dir,
        seed=seed,
        group_code=group_code,
        dataset_sha256=dataset_sha256,
        model_version=model_version,
    )


def write_data_report(ctx: PipelineContext, report: dict[str, Any]) -> Path:
    """Write the dataset report to artifacts/data_report.json."""
    numeric_columns = [*FEATURES, TARGET_REG, TARGET_CLF]
    descriptive_stats: dict[str, dict[str, Any]] = {}
    for column in numeric_columns:
        stats = ctx.df[column].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        descriptive_stats[column] = {
            key: None if pd.isna(value) else float(value)
            for key, value in stats.items()
        }

    payload = {
        "group_code": ctx.group_code,
        "dataset_sha256": ctx.dataset_sha256,
        "seed": ctx.seed,
        "row_count": int(len(ctx.df)),
        "feature_count": len(ctx.feature_names),
        "feature_names": ctx.feature_names,
        "missing_values": report.get("missing_values", {}),
        "duplicate_counts": report.get("duplicate_counts", {}),
        "rows_dropped": report.get("rows_dropped", 0),
        "dtypes": {column: str(ctx.df[column].dtype) for column in ctx.df.columns},
        "descriptive_statistics": descriptive_stats,
        "warnings": report.get("warnings", []),
    }
    output_path = Path(ctx.output_dir) / "data_report.json"
    write_json_atomic(output_path, payload)
    return output_path
