
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

names = ["train", "validation", "test"]
splits = {}

for name in names:
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run python -m src.preprocessing first."
        )

    splits[name] = pd.read_csv(path)

    print(f"\n{name.upper()}")
    print("Shape:", splits[name].shape)
    print("Class counts:")
    print(splits[name]["Class"].value_counts().sort_index())
    print("Fraud percentage:",
          round(splits[name]["Class"].mean() * 100, 4), "%")

# Check that each split has the same columns
assert all(
    list(splits[name].columns) == list(splits["train"].columns)
    for name in names
), "Split columns do not match."

# Check that every original row is represented once.
# Resetting the index allows us to compare the saved row values.
original = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "creditcard.csv")

combined = pd.concat(
    [splits[name] for name in names],
    ignore_index=True,
)

assert len(combined) == len(original), "Total row count mismatch."

assert not combined.isnull().any().any(), "Missing values found."

print("\nAll split files passed basic integrity checks.")
