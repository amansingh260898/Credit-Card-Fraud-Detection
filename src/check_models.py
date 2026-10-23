
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

X_test = pd.read_csv(DATA_DIR / "test.csv").drop(
    columns=["Class"]
)

model_names = [
    "logistic_regression_baseline",
    "logistic_regression_balanced",
    "random_forest_balanced",
]

# Use a small sample just to verify loading and prediction.
sample = X_test.head(5)

for name in model_names:
    model_path = MODEL_DIR / f"{name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    model = joblib.load(model_path)
    probabilities = model.predict_proba(sample)[:, 1]

    print(f"\nModel: {name}")
    print("Fraud scores:", probabilities.round(4))
    print("Scores are finite:", pd.Series(probabilities).map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all())

print("\nAll saved models loaded and generated predictions.")
