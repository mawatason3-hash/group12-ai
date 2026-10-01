from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID_RECORD = {
    "plot_area_ha": 1.2,
    "rainfall_mm": 81,
    "soil_ph": 5.7,
    "seed_kg": 210,
    "distance_km": 14,
    "arrival_hour": 9,
}


def _run_predict(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "predict.py"), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_cli_predict_success_with_record() -> None:
    result = _run_predict("--record", json.dumps(VALID_RECORD))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["feature_order"] == [
        "plot_area_ha",
        "rainfall_mm",
        "soil_ph",
        "seed_kg",
        "distance_km",
        "arrival_hour",
    ]
    assert payload["classification_prediction"] in {0, 1}


def test_cli_predict_success_with_record_file(tmp_path) -> None:
    record_path = tmp_path / "valid.json"
    record_path.write_text(json.dumps(VALID_RECORD), encoding="utf-8")

    result = _run_predict("--record-file", str(record_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert isinstance(payload["regression_prediction"], (int, float))
    assert payload["regression_prediction"] == payload["regression_prediction"]


def test_cli_predict_missing_field() -> None:
    record = dict(VALID_RECORD)
    del record["arrival_hour"]
    result = _run_predict("--record", json.dumps(record))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert any(item["field"] == "arrival_hour" for item in payload["errors"])


def test_cli_predict_non_numeric_value() -> None:
    bad_record = dict(VALID_RECORD)
    bad_record["rainfall_mm"] = "bad"
    result = _run_predict("--record", json.dumps(bad_record))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert any(item["field"] == "rainfall_mm" for item in payload["errors"])


def test_cli_predict_invalid_json() -> None:
    result = _run_predict("--record", '{"plot_area_ha":1.2,')
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "invalid JSON" in payload["error"]
