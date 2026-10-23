# Credit Card Fraud Detection & Risk Scoring

A machine learning project that identifies potentially fraudulent
credit card transactions using classification models and a Streamlit
dashboard.

## Overview

Credit card fraud detection is an imbalanced classification problem:
fraudulent transactions represent only a small proportion of all
transactions. This project compares machine learning models, selects
a decision threshold using validation data, and evaluates the selected
configuration on a held-out test set.

## Features

- Exploratory data analysis and visualizations
- Data validation and stratified train/validation/test splits
- Logistic Regression baseline
- Class-weighted Logistic Regression
- Class-weighted Random Forest
- Model comparison using average precision and other metrics
- Validation-based decision threshold optimization
- Final test-set evaluation and confusion matrix
- Streamlit dashboard for individual and batch predictions
- CSV download of batch prediction results

## Technology Stack

- Python
- NumPy and pandas
- scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Joblib

## Dataset

Dataset: Credit Card Fraud Detection

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` from Kaggle and place it at:

`data/raw/creditcard.csv`

The dataset contains `Time`, `V1` through `V28`, `Amount`,
and the target column `Class`.

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

The `V1`–`V28` features are anonymized numerical features. They do
not represent directly interpretable cardholder information.

## Project Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-card-fraud-detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the data

Download the Kaggle dataset and place `creditcard.csv` in
`data/raw/`.

### 5. Run exploratory data analysis

```bash
python -m src.eda
```

### 6. Create data splits

```bash
python -m src.preprocessing
```

### 7. Train and compare models

```bash
python -m src.train
```

### 8. Evaluate and save the final model

```bash
python -m src.evaluate
```

### 9. Launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

## Evaluation Methodology

The project uses three data partitions:

- Training set: fits the candidate models.
- Validation set: compares models and selects the decision threshold.
- Test set: evaluates the selected model and threshold.

The test set should not be used repeatedly to tune the model.

Metrics include:

- Precision
- Recall
- F1-score
- Average precision
- ROC-AUC
- Confusion matrix counts

Accuracy is reported for completeness but should not be used alone
because the dataset is highly imbalanced.

## Results

Replace this section with the actual output from
`reports/final_test_metrics.csv`.

| Metric             | Final test result     |
| ------------------ | --------------------- |
| Selected model     | Add actual model name |
| Decision threshold | Add actual threshold  |
| Precision          | Add actual value      |
| Recall             | Add actual value      |
| F1-score           | Add actual value      |
| Average precision  | Add actual value      |
| ROC-AUC            | Add actual value      |
| False positives    | Add actual count      |
| False negatives    | Add actual count      |

## Limitations

- The model is an educational prototype, not a production banking system.
- The score is not necessarily a calibrated probability.
- The decision threshold represents a chosen trade-off between missed
  fraud and false alarms.
- The model depends on the original dataset's feature definitions.
- Real payment systems require further testing, monitoring, security,
  calibration, and operational controls.

## Future Improvements

- Calibrate predicted probabilities.
- Compare additional algorithms and resampling methods.
- Tune thresholds using explicit fraud and false-alarm costs.
- Add automated tests and data-drift monitoring.
- Deploy the dashboard to a suitable hosting platform.

## License and Dataset Terms

Review the dataset provider's terms before redistributing the dataset.
Add an appropriate license for your own code if you choose to publish it.

## Author

Add your name and relevant portfolio links here.
