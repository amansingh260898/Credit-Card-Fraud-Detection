# Exploratory Data Analysis — Credit Card Fraud Detection

## 1. Dataset Overview

The project uses the Credit Card Fraud Detection dataset from Kaggle. Each record represents a transaction, and the `Class` column identifies whether the transaction is legitimate or fraudulent.

## 2. Dataset Quality

- Total rows: To be filled from the Python output.
- Total columns: To be filled from the Python output.
- Missing values: To be filled from the Python output.
- Exact duplicate rows: To be filled from the Python output.

## 3. Class Imbalance

The dataset contains substantially more legitimate transactions than fraudulent transactions. This makes accuracy alone an unsuitable measure of model performance.

## 4. Transaction Amounts

The amount distribution is inspected using a logarithmic transformation for visualization. The original transaction amounts remain unchanged.

## 5. Transaction Timing

The `Time` feature represents elapsed seconds since the first transaction. Its distribution is compared between legitimate and fraudulent transactions.

## 6. Feature Relationships

A correlation heatmap is used to explore linear relationships between the numerical features and the target class. Correlation does not imply causation.

## 7. Conclusion

EDA establishes the data quality, feature structure and class imbalance. These findings will guide the preprocessing strategy and evaluation metrics in the next phase.
