
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_loader import load_data


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_chart(filename):
    """Save the current Matplotlib figure."""
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {output_path}")


# --------------------------------------------------
# 2. LOAD AND INSPECT DATA
# --------------------------------------------------

def run_eda():
    df = load_data()

    print("\n" + "=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\nSummary statistics:")
    print(df[["Time", "Amount", "Class"]].describe().round(3))

    # --------------------------------------------------
    # 3. CLASS DISTRIBUTION
    # --------------------------------------------------

    class_counts = (
        df["Class"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    class_labels = ["Legitimate", "Fraud"]

    print("\nClass distribution:")
    print(class_counts.rename(index={0: "Legitimate", 1: "Fraud"}))

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=class_labels,
        y=class_counts.values,
        hue=class_labels,
        legend=False,
        ax=ax,
    )

    ax.set_title("Legitimate vs Fraudulent Transactions")
    ax.set_xlabel("Transaction class")
    ax.set_ylabel("Number of transactions")

    for i, count in enumerate(class_counts.values):
        ax.text(i, count, f"{count:,}", ha="center", va="bottom")

    save_chart("01_class_distribution.png")

    # --------------------------------------------------
    # 4. TRANSACTION AMOUNT DISTRIBUTION
    # --------------------------------------------------

    # Amount is highly skewed. log1p makes the distribution
    # easier to inspect without deleting large transactions.
    df["log_amount"] = np.log1p(df["Amount"])

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="log_amount",
        bins=60,
        ax=ax,
    )

    ax.set_title("Transaction Amount Distribution (log1p scale)")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Number of transactions")

    save_chart("02_amount_distribution.png")

    # --------------------------------------------------
    # 5. AMOUNT BY CLASS
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y="log_amount",
        hue="Class",
        legend=False,
        ax=ax,
    )

    ax.set_title("Transaction Amount by Class")
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("log(1 + Amount)")

    save_chart("03_amount_by_class.png")

    # --------------------------------------------------
    # 6. TRANSACTION TIMING
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=50,
        element="step",
        stat="count",
        common_norm=False,
        ax=ax,
    )

    ax.set_title("Transaction Timing by Class")
    ax.set_xlabel("Seconds since the first transaction")
    ax.set_ylabel("Number of transactions")

    save_chart("04_transaction_timing.png")

    # --------------------------------------------------
    # 7. FEATURE CORRELATION
    # --------------------------------------------------

    # Correlation is descriptive, not proof of causation.
    corr = df.drop(columns=["log_amount"], errors="ignore").corr()

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.0,
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap")

    save_chart("05_correlation_heatmap.png")

    # --------------------------------------------------
    # 8. SUMMARY FOR OUR PROJECT REPORT
    # --------------------------------------------------

    total = len(df)
    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    fraud_rate = (fraud_count / total * 100) if total else 0

    print("\n" + "=" * 55)
    print("EDA SUMMARY")
    print("=" * 55)
    print(f"Total transactions: {total:,}")
    print(f"Legitimate transactions: {legitimate_count:,}")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")
    print(f"Charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
