from __future__ import annotations

SEED = 42
FEATURES = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]
ID_COL = "record_id"
TARGET_REG = "actual_yield_kg"
TARGET_CLF = "dispatch_attention"
MODEL_VERSION_PREFIX = "AI-A1"
TEST_SIZE = 0.2
