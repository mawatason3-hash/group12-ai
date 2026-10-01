# Musanze Cooperative Harvest and Dispatch Decision Lab

This repository implements the end-to-end AI workflow for Group G12 (AI-G12) in SWE 3513. It validates the provided agricultural dataset, trains a regression model for yield prediction, fits a dispatch-attention classifier, and profiles the dataset with K-means clustering.

## Project summary

- Dataset: `data/AI_A1_G12.csv`
- Objective: support yield forecasting and operational decision support for the Musanze farming dataset.
- Primary modules: `src/data.py`, `src/regression.py`, `src/classification.py`, `src/clustering.py`
- Output locations: `artifacts/`, `models/`, `evidence/`

## Quick start

1. Create and activate a virtual environment.
2. Install the pinned requirements:
   `python -m pip install -r requirements.txt`
3. Run the full pipeline from the repository root:
   `python run_all.py --data data/AI_A1_G12.csv --output . --group AI-G12`
4. Predict a single yield record with the saved metadata:
   `python predict.py --model models/model_meta.json --input "plot_area_ha=1.6,rainfall_mm=550,soil_ph=6.1,seed_kg=100,distance_km=15,arrival_hour=8"`

## Included stages

- `src/data.py`: schema validation, missing-value handling, duplicate detection, and shared pipeline context.
- `src/regression.py`: median-imputation + scaling pipeline with batch gradient descent and persisted model metadata.
- `src/classification.py`: logistic regression with balanced class weighting, metrics, and confusion matrix output.
- `src/clustering.py`: silhouette-based K-means search and PCA-based cluster visualization.
- `predict.py`: JSON-safe prediction entry point for a single record using the saved regression model.

## Validation and release workflow

- `pytest` validates the data loader, regression model, classification model, clustering model, and CLI contract.
- `run_all.py` writes model metadata and stage artifacts into the target output directory.
- `scripts/check_package.py` verifies the pinned dependency set and the expected release structure.

## Repository rules

- Do not modify the original lecturer dataset in `data/`.
- Keep all generated synthetic data inside temporary pytest paths only.
- Preserve the exact schema required by the assignment.
- Use the project environment for all Python execution.

## Evidence and contribution notes

Supporting evidence and contribution records are stored under `evidence/`.
