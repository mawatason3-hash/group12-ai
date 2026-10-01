"""Diagnostic: check whether predict_yield applies the scaler correctly."""
import json
import sys
from pathlib import Path

# Load the saved regression model
with open("models/regression_model.json", "r") as f:
    model = json.load(f)

print("Model keys:", list(model.keys()))
print("Feature order:", model.get("feature_order"))
print("Scaler mean:",  model.get("scaler_mean"))
print("Scaler std:",   model.get("scaler_std"))
print("Weights:",      model.get("weights"))
print("Bias:",         model.get("bias"))
print()

# Try to import and call predict_yield
sys.path.insert(0, ".")
try:
    from src.regression import predict_yield
    sample = {
        "plot_area_ha": 1.2, "rainfall_mm": 81, "soil_ph": 5.7,
        "seed_kg": 210, "distance_km": 14, "arrival_hour": 9,
    }
    pred = predict_yield(sample, model)
    print(f"predict_yield(sample) = {pred}")
    print(f"Expected range: ~1500-5000 kg")
    if pred < 0:
        print("❌ NEGATIVE — scaler is NOT being applied correctly")
    else:
        print("✅ Positive — scaler logic OK")
except Exception as e:
    print("predict_yield failed:", type(e).__name__, e)
