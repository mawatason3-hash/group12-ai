# Musanze Cooperative Harvest and Dispatch Decision Lab

This project is an AI assignment for Group G12 (AI-G12) in SWE 3513. It provides a reusable Python data-validation and modeling pipeline for the lecturer-issued dataset at `data/AI_A1_G12.csv`.

## Project structure

- `src/` contains pipeline modules for data handling, regression, classification, and clustering.
- `artifacts/` stores generated reports and model summaries.
- `models/` stores serialized model metadata and trained artifacts.
- `evidence/` is reserved for supporting evidence and logs.
- `tests/` contains pytest-based validation tests.

## Quick start

1. Create a virtual environment and install dependencies.
2. Run the full pipeline:
   `python run_all.py --data data/AI_A1_G12.csv --output . --group AI-G12`
3. Run the prediction CLI using the saved regression model metadata:
   `python predict.py --model models/model_meta.json --input "plot_area_ha=1.6,rainfall_mm=550,soil_ph=6.1,seed_kg=100,distance_km=15,arrival_hour=8"`

## Included stages

- `src/data.py`: strict schema validation, duplicate handling, missing data detection, and the shared `PipelineContext` builder.
- `src/regression.py`: NumPy-based linear regression with median imputation, scaling, batch gradient descent, and persisted model metrics.
- `src/classification.py`: logistic classifier for `dispatch_attention`, with confusion matrix output and saved model coefficients.
- `src/clustering.py`: K-means clustering with silhouette scoring and PCA visualization.
- `predict.py`: single-row inference for the regression model using saved coefficients.

## Rules

- Keep the original lecturer dataset untouched.
- Never create synthetic data files under `data/`.
- Only generate synthetic CSVs under pytest temporary directories (`tmp_path`).
- Use the required schema exactly as issued by the lecturer.
