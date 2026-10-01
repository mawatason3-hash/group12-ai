from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_meta.json"


def _run_predict(raw_input: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "predict.py"),
            "--model",
            str(MODEL_PATH),
            "--input",
            raw_input,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_cli_predict_success_with_key_value_input() -> None:
    result = _run_predict("plot_area_ha=1.6,rainfall_mm=550,soil_ph=6.1,seed_kg=100,distance_km=15,arrival_hour=8")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert float(payload["prediction_kg"]) > 0.0
    assert payload["feature_order"]


def test_cli_predict_success_with_json_file_input(tmp_path) -> None:
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps({
            "plot_area_ha": 1.6,
            "rainfall_mm": 550,
            "soil_ph": 6.1,
            "seed_kg": 100,
            "distance_km": 15,
            "arrival_hour": 8,
        }),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "predict.py"),
            "--model",
            str(MODEL_PATH),
            "--input",
            str(record_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert float(payload["prediction_kg"]) > 0.0


def test_cli_predict_rejects_invalid_input() -> None:
    result = _run_predict("plot_area_ha=bad,rainfall_mm=550,soil_ph=6.1,seed_kg=100,distance_km=15,arrival_hour=8")
    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert "Non-numeric value supplied" in payload["error"]
