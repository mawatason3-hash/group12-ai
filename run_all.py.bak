from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from src import classification, clustering, regression
from src.common import write_json_atomic
from src.config import MODEL_VERSION_PREFIX, SEED
from src.data import build_context, load_and_validate, sha256_file, write_data_report


def ensure_directories(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create the output, model, and evidence directories for the pipeline run."""
    base_dir = base_dir.expanduser().resolve()
    if base_dir.name.lower() == "artifacts":
        artifacts_dir = base_dir
        models_dir = base_dir.parent / "models"
        evidence_dir = base_dir.parent / "evidence"
    else:
        artifacts_dir = base_dir / "artifacts"
        models_dir = base_dir / "models"
        evidence_dir = base_dir / "evidence"
    for directory in (artifacts_dir, models_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return artifacts_dir, models_dir, evidence_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="Musanze Cooperative Harvest and Dispatch Decision Lab")
    parser.add_argument("--data", required=True, help="Path to the CSV input dataset.")
    parser.add_argument("--output", required=True, help="Target directory for artifacts and models.")
    parser.add_argument("--group", required=True, help="Group code for the assignment, for example AI-G12.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.group.startswith("AI-"):
        print("Group code must start with 'AI-'.", file=sys.stderr)
        return 2

    dataset_path = Path(args.data).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        return 2

    output_base = Path(args.output).expanduser().resolve()
    artifacts_dir, models_dir, evidence_dir = ensure_directories(output_base)
    dataset_sha256 = sha256_file(dataset_path)

    print("=" * 80)
    print(f"GROUP CODE: {args.group}")
    print(f"DATASET PATH: {dataset_path}")
    print(f"DATASET SHA256: {dataset_sha256}")
    print("=" * 80)

    try:
        df, report = load_and_validate(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2

    model_version = f"{MODEL_VERSION_PREFIX}-{args.group}-{dataset_sha256[:8]}"
    ctx = build_context(
        df,
        group_code=args.group,
        output_dir=artifacts_dir,
        models_dir=models_dir,
        seed=SEED,
        dataset_sha256=dataset_sha256,
        model_version=model_version,
    )

    write_data_report(ctx, report)
    written_files: list[str] = [str(artifacts_dir / "data_report.json")]

    stages = [
        ("regression", regression.run),
        ("classification", classification.run),
        ("clustering", clustering.run),
    ]
    for stage_name, stage_func in stages:
        try:
            result = stage_func(ctx)
        except Exception as exc:  # pragma: no cover - CLI contract for failed stages
            print(f"Stage failed: {stage_name}: {exc}", file=sys.stderr)
            return 1
        artifact_path = result.get("artifact_path")
        if artifact_path:
            written_files.append(str(Path(artifact_path)))

    model_meta = {
        "group_code": args.group,
        "seed": SEED,
        "model_version": model_version,
        "feature_order": list(ctx.feature_names),
        "dataset_sha256": dataset_sha256,
        "regression_model_path": "regression_model.json",
        "classification_model_path": "classification_model.json",
        "clustering_model_path": "clustering_model.json",
        "regression_metrics_path": "regression_metrics.json",
        "classification_summary_path": "classification_summary.json",
        "clustering_summary_path": "clustering_summary.json",
    }
    model_meta_path = models_dir / "model_meta.json"
    write_json_atomic(model_meta_path, model_meta)
    written_files.append(str(model_meta_path))

    print("\nFiles written:")
    for item in sorted(set(written_files)):
        print(f" - {item}")
    print(f"\nEvidence directory: {evidence_dir}")
    print(f"Model version: {model_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
