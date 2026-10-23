
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


# --------------------------------------------------
# 2. LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

def load_splits():
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Training or validation data not found. "
            "Run python -m src.preprocessing first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if "Class" not in train_df.columns:
        raise ValueError("Class column missing from training data.")

    if "Class" not in val_df.columns:
        raise ValueError("Class column missing from validation data.")

    if list(train_df.columns) != list(val_df.columns):
        raise ValueError(
            "Training and validation columns do not match."
        )

    if train_df.isnull().any().any() or val_df.isnull().any().any():
        raise ValueError("Missing values found in the saved splits.")

    if not train_df["Class"].isin([0, 1]).all():
        raise ValueError("Invalid training target labels.")

    if not val_df["Class"].isin([0, 1]).all():
        raise ValueError("Invalid validation target labels.")

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"].astype(int)

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"].astype(int)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Feature columns do not match.")

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Training fraud cases:", int(y_train.sum()))
    print("Validation fraud cases:", int(y_val.sum()))

    return X_train, y_train, X_val, y_val


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

def build_models():
    models = {
        "logistic_regression_baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "logistic_regression_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "random_forest_balanced": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# --------------------------------------------------
# 4. EVALUATE VALIDATION PREDICTIONS
# --------------------------------------------------

def evaluate_model(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]

    # Use 0.50 only for the initial comparison.
    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    metrics = {
        "threshold": DEFAULT_THRESHOLD,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_val, probabilities
        ),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "flagged_transactions": int(predictions.sum()),
        "actual_fraud_cases": int(y_val.sum()),
    }

    return metrics


# --------------------------------------------------
# 5. TRAIN, EVALUATE AND SAVE
# --------------------------------------------------

def train_models():
    X_train, y_train, X_val, y_val = load_splits()

    models = build_models()
    results = []

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        start_time = time.perf_counter()

        # Fit using training data only.
        model.fit(X_train, y_train)

        training_seconds = time.perf_counter() - start_time

        # Evaluate using validation data.
        metrics = evaluate_model(model, X_val, y_val)

        metrics["model"] = model_name
        metrics["training_seconds"] = round(
            training_seconds, 2
        )
        results.append(metrics)

        # Save the trained estimator.
        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"Saved model: {model_path}")
        print(f"Training time: {training_seconds:.2f} seconds")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(
            "Average precision (PR summary): "
            f"{metrics['average_precision']:.4f}"
        )
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(
            "Flagged transactions: "
            f"{metrics['flagged_transactions']:,}"
        )

    # Save a comparison table.
    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "model",
            "training_seconds",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "average_precision",
            "roc_auc",
            "flagged_transactions",
            "actual_fraud_cases",
        ]
    ]

    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.sort_values(
            "average_precision", ascending=False
        ).round(4).to_string(index=False)
    )

    print(f"\nComparison saved to: {results_path}")
    print("\nModel training completed!")


if __name__ == "__main__":
    train_models()
