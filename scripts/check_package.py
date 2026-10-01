from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
MODEL_PATH = REPO_ROOT / "models" / "model_meta.json"


def _read_requirements() -> list[str]:
    return [line.strip() for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def _assert_files_exist() -> None:
    required = [
        REPO_ROOT / "predict.py",
        REPO_ROOT / "run_all.py",
        REPO_ROOT / "src" / "data.py",
        REPO_ROOT / "src" / "regression.py",
        REPO_ROOT / "src" / "classification.py",
        REPO_ROOT / "src" / "clustering.py",
        REPO_ROOT / "README.md",
        REPO_ROOT / "requirements.txt",
        MODEL_PATH,
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def _check_cli() -> dict[str, object]:
    command = [
        sys.executable,
        str(REPO_ROOT / "predict.py"),
        "--model",
        str(MODEL_PATH),
        "--input",
        "plot_area_ha=1.6,rainfall_mm=550,soil_ph=6.1,seed_kg=100,distance_km=15,arrival_hour=8",
    ]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(f"CLI check failed with {completed.returncode}: {stderr or stdout}")
    payload = json.loads(stdout)
    if payload.get("status") != "ok":
        raise RuntimeError(f"Prediction CLI returned a non-ok status: {payload}")
    if "prediction_kg" not in payload:
        raise RuntimeError("Prediction CLI output missing prediction_kg.")
    return payload


def main() -> int:
    try:
        requirements = _read_requirements()
        required_packages = {
            "joblib==1.6.0",
            "matplotlib==3.8.4",
            "numpy==1.26.4",
            "pandas==2.2.3",
            "pytest==8.3.2",
            "scikit-learn==1.5.2",
            "seaborn==0.13.1",
        }
        if not required_packages.issubset(set(requirements)):
            missing = sorted(required_packages - set(requirements))
            raise RuntimeError(f"requirements.txt is missing pinned dependencies: {missing}")
        _assert_files_exist()
        payload = _check_cli()
        print(json.dumps({"status": "ok", "requirements": requirements, "prediction_kg": payload["prediction_kg"]}, sort_keys=True))
        return 0
    except Exception as exc:  # pragma: no cover - packaging check is intentionally strict
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
