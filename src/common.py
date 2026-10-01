from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    """Container for cleaned data and the model pipeline state."""

    df: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]
    ids: np.ndarray
    y_reg: np.ndarray
    y_clf: np.ndarray
    output_dir: Path
    models_dir: Path
    seed: int
    group_code: str
    dataset_sha256: str
    model_version: str

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.models_dir = Path(self.models_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: str | Path, payload: dict) -> None:
    """Write JSON to disk using a temporary file and atomic replace."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=str(output_path.parent),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, output_path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise
