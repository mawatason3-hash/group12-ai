from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.regression import predict_yield


def _emit_json(payload: dict[str, Any], *, stream: Any = None) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=stream or sys.stdout)


def _resolve_model_file(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "feature_order" in data and "weights" in data:
        return data
    if isinstance(data, dict) and "regression_model_path" in data:
        nested = path.parent / data["regression_model_path"]
        return _resolve_model_file(nested)
    raise ValueError("The provided model artifact does not contain a regression model payload.")


def _coerce_numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - exercised via CLI validation
        raise ValueError(f"Non-numeric value supplied: {value!r}") from exc


def _read_json_input(source: str) -> dict[str, Any]:
    payload = json.loads(source)
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Input JSON list is empty.")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must decode to a dictionary or a list of dictionaries.")
    return payload


def _read_file_input(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError("Input CSV is empty.")
        row = frame.iloc[0].to_dict()
        return {str(key): value for key, value in row.items() if str(key) != "record_id"}
    if suffix == ".json":
        return _read_json_input(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported input file type: {suffix or 'unknown'}")


def _parse_row_input(raw: str) -> dict[str, Any]:
    candidate = Path(raw)
    if candidate.exists():
        return _read_file_input(candidate)

    if raw.strip().startswith("{"):
        return _read_json_input(raw)

    pair_text = raw.replace(";", ",")
    parts = [item.strip() for item in pair_text.split(",") if item.strip()]
    if not parts:
        raise ValueError("Input row could not be parsed.")

    if any("=" in item for item in parts):
        payload: dict[str, Any] = {}
        for item in parts:
            key, value = item.split("=", 1)
            payload[key.strip()] = _coerce_numeric(value.strip())
        return payload

    numeric_values = [_coerce_numeric(value) for value in parts]
    return {f"value_{index}": value for index, value in enumerate(numeric_values)}


def _normalize_for_model(row: dict[str, Any], feature_order: list[str]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for feature in feature_order:
        if feature in row:
            normalized[feature] = _coerce_numeric(row[feature])
            continue
        if feature.lower() in {key.lower(): key for key in row}:
            match = feature.lower()
            normalized[feature] = _coerce_numeric(row[next(key for key in row if key.lower() == match)])
            continue
        if any(key.startswith("value_") for key in row):
            indexed = {key: value for key, value in row.items() if key.startswith("value_")}
            if len(indexed) == len(feature_order):
                ordered = [indexed[f"value_{index}"] for index in range(len(feature_order))]
                return {feature_name: _coerce_numeric(ordered[idx]) for idx, feature_name in enumerate(feature_order)}
        raise ValueError(f"Missing feature value for '{feature}' in the input row.")
    return normalized


def main() -> int:
    """Predict yield using the trained NumPy regression model and emit JSON results."""
    parser = argparse.ArgumentParser(description="Prediction entry point for the assignment model.")
    parser.add_argument("--model", required=True, help="Path to the trained model or model metadata JSON.")
    parser.add_argument("--input", required=True, help="Input row as a CSV/JSON path or comma-separated feature values.")
    args = parser.parse_args()

    try:
        model_payload = _resolve_model_file(args.model)
        feature_names = list(model_payload.get("feature_order", []))
        if not feature_names:
            raise ValueError("Model JSON does not define feature_order for prediction.")

        row = _parse_row_input(args.input)
        ordered = _normalize_for_model(row, feature_names)
        prediction = predict_yield(ordered, model_payload)
        payload = {
            "status": "ok",
            "prediction_kg": float(prediction),
            "feature_order": feature_names,
            "input_features": ordered,
        }
        _emit_json(payload)
        return 0
    except Exception as exc:  # pragma: no cover - command-line error contract
        _emit_json({"status": "error", "error": str(exc)}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
