
from pathlib import Path

import pandas as pd


# Project root: the folder containing data/, src/, and app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Location of the original Kaggle dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

# Expected dataset columns
EXPECTED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)


def load_data(path=DATA_PATH):
    """
    Load the credit card transaction dataset
    and validate its basic structure.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download creditcard.csv and place it "
            "inside data/raw/."
        )

    df = pd.read_csv(path)

    # Remove accidental whitespace in column names
    df.columns = df.columns.str.strip()

    # Validate expected column names
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    # All expected fields should be numeric
    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )

    # The target must contain only legitimate/fraud labels
    if not df["Class"].isin([0, 1]).all():
        raise ValueError(
            "Class must contain only 0 (legitimate) "
            "and 1 (fraud)."
        )

    print("Dataset loaded successfully!")
    print(f"File: {path}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    return df


if __name__ == "__main__":
    df = load_data()
    print("\nFirst five transactions:")
    print(df.head())
