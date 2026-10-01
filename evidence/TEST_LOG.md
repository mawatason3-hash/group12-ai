# Test Log

## Validation status

The project pipeline was validated with pytest using the repository's Python environment. The regression, classification, clustering, and CLI checks cover the required model and data behaviors.

## Commands used

- `pytest -q`
- `python run_all.py --data data/AI_A1_G12.csv --output . --group AI-G12`
- `python predict.py --model models/model_meta.json --input "plot_area_ha=1.6,rainfall_mm=550,soil_ph=6.1,seed_kg=100,distance_km=15,arrival_hour=8"`

## Outcome

The workflow completes successfully and produces reproducible artifacts alongside the final release package.
