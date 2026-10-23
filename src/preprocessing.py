
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# 70% train, 15% validation, 15% test
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# --------------------------------------------------
# 2. VALIDATE THE DATA
# --------------------------------------------------

def validate_data(df):
    """Check that the dataset is suitable for splitting."""

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.isnull().any().any():
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        raise ValueError(
            f"Missing values found:\n{missing}"
        )

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' is missing.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    if df["Class"].nunique() != 2:
        raise ValueError(
            "Both legitimate and fraud cases are required."
        )

    feature_columns = [
        column for column in df.columns
        if column != "Class"
    ]

    if not all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in feature_columns
    ):
        raise ValueError("All features must be numeric.")

    if not all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in df.columns
    ):
        raise ValueError("All dataset columns must be numeric.")

    if not all(
        df[col].map(lambda value: pd.notna(value) and abs(value) != float("inf")).all()
        for col in df.columns
    ):
        raise ValueError("Infinite or invalid numeric values found.")

    print("Data validation passed.")


# --------------------------------------------------
# 3. SPLIT FEATURES AND TARGET
# --------------------------------------------------

def split_data(df):
    """Create stratified train, validation and test sets."""

    validate_data(df)

    # X = input features
    # y = correct class labels
    X = df.drop(columns=["Class"])
    y = df["Class"].astype("int64")

    # First split: 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split the remaining 30% equally:
    # 15% validation and 15% test overall
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Keep the target attached to each saved split
    train_df = X_train.copy()
    train_df["Class"] = y_train

    val_df = X_val.copy()
    val_df["Class"] = y_val

    test_df = X_test.copy()
    test_df["Class"] = y_test

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    return splits


# --------------------------------------------------
# 4. SAVE SPLITS
# --------------------------------------------------

def save_splits(splits):
    """Save each split as a separate CSV file."""

    for name, split_df in splits.items():
        output_path = PROCESSED_DIR / f"{name}.csv"

        split_df.to_csv(output_path, index=False)

        print(f"Saved {name}: {split_df.shape} -> {output_path}")


# --------------------------------------------------
# 5. REPORT THE RESULTS
# --------------------------------------------------

def report_splits(splits):
    """Display the class distribution in each split."""

    print("\n" + "=" * 65)
    print("TRAIN / VALIDATION / TEST SUMMARY")
    print("=" * 65)

    total_rows = sum(len(part) for part in splits.values())

    for name, part in splits.items():
        fraud_count = int((part["Class"] == 1).sum())
        legitimate_count = int((part["Class"] == 0).sum())
        fraud_rate = part["Class"].mean() * 100

        print(f"\n{name.upper()}")
        print(f"Rows: {len(part):,}")
        print(f"Legitimate: {legitimate_count:,}")
        print(f"Fraud: {fraud_count:,}")
        print(f"Fraud rate: {fraud_rate:.4f}%")

    print(f"\nTotal rows across splits: {total_rows:,}")


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():
    print("Loading original dataset...")
    df = load_data()

    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\nCreating stratified splits...")
    splits = split_data(df)

    report_splits(splits)
    save_splits(splits)

    print("\nPhase 3 data splitting completed successfully!")


if __name__ == "__main__":
    main()
