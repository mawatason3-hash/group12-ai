from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.regression import predict_yield


def _resolve_model_file(model_path: str | Path) -> dict:
    path = Path(model_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "feature_order" in data and "weights" in data:
        return data
    if "regression_model_path" in data:
        model_file = path.parent / data["regression_model_path"]
        return json.loads(model_file.read_text(encoding="utf-8"))
    raise ValueError("The provided model artifact does not contain a regression model payload.")


def _resolve_row(input_value: str) -> dict[str, float]:
    candidate = Path(input_value)
    if candidate.exists():
        suffix = candidate.suffix.lower()
        if suffix in {".csv"}:
            frame = pd.read_csv(candidate)
            if frame.empty:
                raise ValueError("Input CSV is empty.")
            row = frame.iloc[0].to_dict()
            return {key: float(value) for key, value in row.items() if str(key) != "record_id"}
        if suffix in {".json"}:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                payload = payload[0]
            return {key: float(value) for key, value in payload.items()}

    if input_value.strip().startswith("{"):
        payload = json.loads(input_value)
        return {key: float(value) for key, value in payload.items()}

    pairs = [item.strip() for item in input_value.replace(";", ",").split(",") if item.strip()]
    if any("=" in item for item in pairs):
        payload: dict[str, float] = {}
        for item in pairs:
            key, value = item.split("=", 1)
            payload[key.strip()] = float(value.strip())
        return payload

    tokens = [value for value in pairs if value]
    if not tokens:
        raise ValueError("Input row could not be parsed.")
    return {f"value_{index}": float(value) for index, value in enumerate(tokens)}


def main() -> int:
    """Predict yield using the trained NumPy regression model."""
    parser = argparse.ArgumentParser(description="Prediction entry point for the assignment model.")
    parser.add_argument("--model", required=True, help="Path to the trained model or model metadata JSON.")
    parser.add_argument("--input", required=True, help="Input row as a CSV/JSON path or comma-separated feature values.")
    args = parser.parse_args()

    model_payload = _resolve_model_file(args.model)
    feature_names = list(model_payload.get("feature_order", []))
    if not feature_names:
        raise ValueError("Model JSON does not define feature_order for prediction.")

    row = _resolve_row(args.input)
    ordered = {name: float(row[name]) for name in feature_names if name in row}
    if len(ordered) != len(feature_names):
        missing = [name for name in feature_names if name not in ordered]
        raise ValueError(f"Missing model features for prediction: {missing}")

    prediction = predict_yield(ordered, model_payload)
    print(f"Predicted yield: {prediction:.4f} kg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
