# xbooster Examples

This directory contains example notebooks demonstrating various features of xbooster.

## Notebooks

### `xgboost-categorical-data.ipynb`
Demonstrates how to handle categorical data in XGBoost using the `DataPreprocessor`:
- Loading and preprocessing credit data
- Handling categorical features with one-hot encoding
- Generating interaction constraints
- Training an XGBoost model with categorical features

### `xgboost-scorecard.ipynb`
Shows how to create and use scorecards with XGBoost:
- Training an XGBoost model
- Creating a scorecard
- Making predictions using different methods (raw, WOE, PDO)
- Calculating Gini scores
- Visualizing feature importance
- Generating SQL queries for deployment

### `catboost-scorecard.ipynb`
Demonstrates the experimental CatBoost support:
- Training a CatBoost model
- Creating a scorecard
- Making predictions using different methods
- Calculating Gini scores
- Visualizing feature importance
- Tree visualization

## Data Files

### `test_data_01d9ab8b.csv`
Sample credit data used in the examples, containing:
- Numerical features (e.g., Gross_Annual_Income, Application_Score)
- Categorical features (e.g., Time_with_Bank)
- Target variable (Final_Decision)

### `train_u6lujuX_CVtuZ9i.csv`
Additional credit data used in the categorical data example, containing:
- Numerical features (e.g., ApplicantIncome, LoanAmount)
- Categorical features (e.g., Married, Education)
- Target variable (Loan_Status)

## Running the Examples

1. Install the required dependencies:
```bash
pip install xbooster
```

2. Run the notebooks using Jupyter:
```bash
jupyter notebook
```

3. Open the desired notebook and follow the instructions.

Note: The examples use sample data and may need to be adapted for your specific use case. 