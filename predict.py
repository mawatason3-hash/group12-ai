from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib

from src.classification import predict_attention
from src.clustering import predict_cluster
from src.regression import predict_yield

REQUIRED_FIELDS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def _parse_record_string(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("record JSON must decode to an object")
    return payload


def _load_record(record: str | None, record_file: str | None) -> dict[str, Any]:
    if record_file:
        path = Path(record_file)
        if not path.exists():
            raise FileNotFoundError(f"record file {path} not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("record JSON must decode to an object")
        return payload

    if record is None:
        raise ValueError("provide --record or --record-file")
    return _parse_record_string(record)


def _is_valid_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    return True


def _validate_record(payload: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    for field in missing_fields:
        errors.append({"field": field, "message": "missing required field"})

    for field in REQUIRED_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if not _is_valid_number(value):
            errors.append({"field": field, "message": "must be a finite numeric value"})
            continue

        numeric_value = float(value)
        if field == "plot_area_ha" and numeric_value <= 0:
            errors.append({"field": field, "message": "must be > 0"})
        elif field == "rainfall_mm" and numeric_value < 0:
            errors.append({"field": field, "message": "must be >= 0"})
        elif field == "soil_ph" and not (0 <= numeric_value <= 14):
            errors.append({"field": field, "message": "must be between 0 and 14 inclusive"})
        elif field == "seed_kg" and numeric_value < 0:
            errors.append({"field": field, "message": "must be >= 0"})
        elif field == "distance_km" and numeric_value < 0:
            errors.append({"field": field, "message": "must be >= 0"})
        elif field == "arrival_hour" and not (0 <= numeric_value <= 23):
            errors.append({"field": field, "message": "must be between 0 and 23 inclusive"})

    return errors


def _ensure_model_exists(models_dir: Path, filename: str) -> None:
    candidate = models_dir / filename
    if not candidate.exists():
        raise FileNotFoundError(f"model file {filename} not found. Run run_all.py first.")


def _load_models(models_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Any, Any, str, str]:
    for filename in ["model_meta.json", "regression_model.json", "classifier.joblib", "clustering.joblib"]:
        _ensure_model_exists(models_dir, filename)

    model_meta = json.loads((models_dir / "model_meta.json").read_text(encoding="utf-8"))
    regression_model = json.loads((models_dir / "regression_model.json").read_text(encoding="utf-8"))
    classifier_model = joblib.load(models_dir / "classifier.joblib")
    clustering_model = joblib.load(models_dir / "clustering.joblib")
    group_code = str(model_meta.get("group_code", ""))
    model_version = str(model_meta.get("model_version", ""))
    return model_meta, regression_model, classifier_model, clustering_model, group_code, model_version


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict crop yield and decision labels for a single record.")
    parser.add_argument("--record", help="JSON object with the six model features.")
    parser.add_argument("--record-file", help="Path to a JSON file containing one record object.")
    parser.add_argument("--models-dir", default="models", help="Directory containing model artifacts.")
    args = parser.parse_args()

    try:
        record = _load_record(args.record, args.record_file)
        errors = _validate_record(record)
        if errors:
            _print_json({"status": "error", "errors": errors})
            return 1

        models_dir = Path(args.models_dir)
        model_meta, regression_model, classifier_model, clustering_model, group_code, model_version = _load_models(models_dir)
        feature_order = list(model_meta.get("feature_order", REQUIRED_FIELDS))
        ordered_record = {field: float(record[field]) for field in feature_order if field in record}

        regression_prediction = float(predict_yield(ordered_record, regression_model))
        classification_prediction, classification_probability = predict_attention(record, classifier_model)
        cluster_label = int(predict_cluster(record, clustering_model))

        payload = {
            "status": "ok",
            "regression_prediction": regression_prediction,
            "classification_prediction": int(classification_prediction),
            "classification_probability": float(classification_probability),
            "cluster_label": cluster_label,
            "group_code": group_code,
            "model_version": model_version,
            "feature_order": feature_order,
        }
        _print_json(payload)
        return 0
    except FileNotFoundError as exc:
        _print_json({"status": "error", "error": str(exc)})
        return 1
    except ValueError as exc:
        message = str(exc)
        if message.startswith("invalid JSON:"):
            _print_json({"status": "error", "error": message})
            return 1
        _print_json({"status": "error", "errors": [{"field": "record", "message": message}]})
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI fallback
        _print_json({"status": "error", "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
