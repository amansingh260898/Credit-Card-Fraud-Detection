
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

MODEL_NAMES = [
    "logistic_regression_baseline",
    "logistic_regression_balanced",
    "random_forest_balanced",
]


# --------------------------------------------------
# 2. LOAD VALIDATION AND TEST DATA
# --------------------------------------------------

def load_evaluation_data():
    validation_path = DATA_DIR / "validation.csv"
    test_path = DATA_DIR / "test.csv"

    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Validation/test data not found. "
            "Run: python -m src.preprocessing"
        )

    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    if "Class" not in validation_df or "Class" not in test_df:
        raise ValueError("The Class target column is missing.")

    if list(validation_df.columns) != list(test_df.columns):
        raise ValueError("Validation and test columns do not match.")

    if validation_df.isnull().any().any():
        raise ValueError("Missing values found in validation data.")

    if test_df.isnull().any().any():
        raise ValueError("Missing values found in test data.")

    if not validation_df["Class"].isin([0, 1]).all():
        raise ValueError("Invalid validation labels.")

    if not test_df["Class"].isin([0, 1]).all():
        raise ValueError("Invalid test labels.")

    X_val = validation_df.drop(columns=["Class"])
    y_val = validation_df["Class"].astype(int)

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].astype(int)

    print("Validation shape:", X_val.shape)
    print("Validation fraud cases:", int(y_val.sum()))
    print("Test shape:", X_test.shape)
    print("Test fraud cases:", int(y_test.sum()))

    return X_val, y_val, X_test, y_test


# --------------------------------------------------
# 3. LOAD TRAINED MODELS
# --------------------------------------------------

def load_models():
    models = {}

    for name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: python -m src.train"
            )

        models[name] = joblib.load(model_path)

    return models


# --------------------------------------------------
# 4. SELECT MODEL USING VALIDATION AVERAGE PRECISION
# --------------------------------------------------

def select_model(models, X_val, y_val):
    results = []
    validation_probabilities = {}

    for name, model in models.items():
        probabilities = model.predict_proba(X_val)[:, 1]

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid probability scores from {name}.")

        ap = average_precision_score(y_val, probabilities)

        validation_probabilities[name] = probabilities

        results.append({
            "model": name,
            "validation_average_precision": ap,
        })

    comparison = pd.DataFrame(results).sort_values(
        "validation_average_precision",
        ascending=False,
    )

    comparison.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    print("\nVALIDATION MODEL COMPARISON")
    print(comparison.round(5).to_string(index=False))

    # Select the model with the highest validation average precision.
    selected_name = comparison.iloc[0]["model"]

    print("\nSelected model:", selected_name)

    return selected_name, comparison, validation_probabilities


# --------------------------------------------------
# 5. FIND THE BEST THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

def optimize_threshold(y_val, probabilities):
    thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        results.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "flagged_transactions": int(predictions.sum()),
        })

    threshold_df = pd.DataFrame(results)

    threshold_df.to_csv(
        REPORT_DIR / "threshold_analysis.csv",
        index=False,
    )

    # Highest F1 wins. If tied, prefer higher precision.
    best_row = threshold_df.sort_values(
        ["f1_score", "precision"],
        ascending=[False, False],
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    print("\nBEST VALIDATION THRESHOLD")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall: {best_row['recall']:.4f}")
    print(f"F1-score: {best_row['f1_score']:.4f}")
    print(
        "Flagged transactions:",
        int(best_row["flagged_transactions"]),
    )

    # Save a threshold-versus-F1 chart.
    plt.figure(figsize=(9, 5))
    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1_score"],
        label="Validation F1-score",
    )
    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Selected threshold = {best_threshold:.2f}",
    )
    plt.xlabel("Probability threshold")
    plt.ylabel("F1-score")
    plt.title("Threshold Optimization on Validation Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "06_threshold_optimization.png",
        dpi=150,
    )
    plt.close()

    return best_threshold, threshold_df


# --------------------------------------------------
# 6. FINAL EVALUATION ON THE HELD-OUT TEST SET
# --------------------------------------------------

def evaluate_on_test(model, X_test, y_test, threshold):
    # Test labels are used only after model and threshold selection.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "test_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "test_precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "test_f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "test_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "actual_fraud_cases": int(y_test.sum()),
        "flagged_transactions": int(predictions.sum()),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    })

    print("\nFINAL TEST RESULTS")
    print("-" * 45)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.5f}")
        else:
            print(f"{name}: {value:,}")

    # Save confusion matrix.
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Final Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "07_test_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    # Save the test precision-recall curve.
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Test Data")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "08_test_precision_recall_curve.png",
        dpi=150,
    )
    plt.close()

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "final_test_metrics.csv",
        index=False,
    )

    return metrics


# --------------------------------------------------
# 7. SAVE THE FINAL MODEL AND ITS METADATA
# --------------------------------------------------

def save_final_artifacts(
    model,
    model_name,
    threshold,
    feature_columns,
    metrics,
):
    final_model_path = MODEL_DIR / "final_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"

    joblib.dump(model, final_model_path)

    metadata = {
        "selected_model": model_name,
        "threshold": threshold,
        "selection_metric": "validation_average_precision",
        "threshold_selection": "maximum validation F1-score",
        "feature_columns": list(feature_columns),
        "test_metrics": metrics,
        "risk_score_note": (
            "The model output is a predicted fraud score, "
            "not necessarily a calibrated real-world probability."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nSaved final model:", final_model_path)
    print("Saved model metadata:", metadata_path)


# --------------------------------------------------
# 8. MAIN WORKFLOW
# --------------------------------------------------

def main():
    X_val, y_val, X_test, y_test = load_evaluation_data()
    models = load_models()

    selected_name, comparison, validation_probabilities = (
        select_model(models, X_val, y_val)
    )

    selected_model = models[selected_name]
    val_probabilities = validation_probabilities[selected_name]

    threshold, threshold_df = optimize_threshold(
        y_val,
        val_probabilities,
    )

    # Do not change the model or threshold based on test results.
    test_metrics = evaluate_on_test(
        selected_model,
        X_test,
        y_test,
        threshold,
    )

    save_final_artifacts(
        selected_model,
        selected_name,
        threshold,
        X_val.columns,
        test_metrics,
    )

    print("\nPhase 5 completed successfully!")
    print("Check the reports/ and reports/figures/ folders.")


if __name__ == "__main__":
    main()
