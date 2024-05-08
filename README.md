# xbooster 🚀

A scorecard-format classificatory framework for logistic regression with XGBoost.
xbooster allows to convert an XGB logistic regression into a logarithmic (point) scoring system.

In addition, it provides a suite of interpretability tools to understand the model's behavior,
which can be instrumental for model testing and expert validation.

The interpretability suite includes:

- Granular boosted tree statistics, including metrics such as Weight of Evidence (WOE) and Information Value (IV) for splits 🌳
- Tree visualization with customizations 🎨
- Global and local feature importance 📊

xbooster also provides a scorecard deployment using SQL 📦.

## Installation ⤵

Install the package using pip:

```python
pip install xbooster
```

## Usage 📝
Here's a quick example of how to use xbooster to construct a scorecard for an XGBoost model:

```python
import pandas as pd
import xgboost as xgb
from xbooster.constructor import XGBScorecardConstructor
from sklearn.model_selection import train_test_split

# Load data and train XGBoost model
url = (
    "https://github.com/xRiskLab/xBooster/raw/main/examples/data/credit_data.parquet"
)
dataset = pd.read_parquet(url)

features = [
    "external_risk_estimate",
    "revolving_utilization_of_unsecured_lines",
    "account_never_delinq_percent",
    "net_fraction_revolving_burden",
    "num_total_cc_accounts",
    "average_months_in_file",
]

target = "is_bad"

X, y = dataset[features], dataset[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the XGBoost model
best_params = {
    'n_estimators': 100,
    'learning_rate': 0.55,
    'max_depth': 1,
    'min_child_weight': 10,
    'grow_policy': "lossguide",
    'early_stopping_rounds': 5
}
model = xgb.XGBClassifier(**best_params, random_state=62)
model.fit(X_train, y_train)

# Initialize XGBScorecardConstructor
scorecard_constructor = XGBScorecardConstructor(model, X_train, y_train)
scorecard_constructor.construct_scorecard()

# Print the scorecard
print(scorecard_constructor.scorecard)
```

After this, we can create a scorecard and test its Gini score:

```python
from sklearn.metrics import roc_auc_score

# Create scoring points
xgb_scorecard_with_points = scorecard_constructor.create_points(
    pdo=50, target_points=600, target_odds=50
)
# Make predictions using the scorecard
credit_scores = scorecard_constructor.predict_score(X_test)
gini = roc_auc_score(y_test, -credit_scores) * 2 - 1
print(f"Test Gini score: {gini:.2%}")
```

We can also visualize the score distribution between the events of interest.

```python
from xbooster import explainer

explainer.plot_score_distribution(
    y_test, 
    credit_scores,
    num_bins=30, 
    figsize=(8, 3),
    dpi=100
)
```

We can further examine feature importances.

Below, we can visualize the global feature importances using Points as our metric:

```python
from xbooster import explainer

explainer.plot_importance(
    scorecard_constructor,
    metric='Points',
    method='global',
    normalize=True,
    figsize=(3, 3)
)
```

Alternatively, we can calculate local feature importances, which are important for boosters with a depth greater than 1.

```python
explainer.plot_importance(
    scorecard_constructor,
    metric='Likelihood',
    method='local',
    normalize=True,
    color='#ffd43b',
    edgecolor='#1e1e1e',
    figsize=(3, 3)
)
```

Finally, we can generate a scorecard in SQL format.

```python
sql_query = scorecard_constructor.generate_sql_query(table_name='my_table')
print(sql_query)
```

# Parameters 🛠

## `xbooster.constructor` - XGBoost Scorecard Constructor

### Description

A class for generating a scorecard from a trained XGBoost model. The methodology is inspired by the NVIDIA GTC Talk "Machine Learning in Retail Credit Risk" by Paul Edwards.

### Methods

1. `extract_leaf_weights() -> pd.DataFrame`:
   - Extracts the leaf weights from the booster's trees and returns a DataFrame.
   - **Returns**:
     - `pd.DataFrame`: DataFrame containing the extracted leaf weights.

2. `extract_decision_nodes() -> pd.DataFrame`:
   - Extracts the split (decision) nodes from the booster's trees and returns a DataFrame.
   - **Returns**:
     - `pd.DataFrame`: DataFrame containing the extracted split (decision) nodes.

3. `construct_scorecard() -> pd.DataFrame`:
   - Constructs a scorecard based on a booster.
   - **Returns**:
     - `pd.DataFrame`: The constructed scorecard.

4. `create_points(pdo=50, target_points=600, target_odds=19, precision_points=0, score_type='XAddEvidence') -> pd.DataFrame`:
   - Creates a points card from a scorecard.
   - **Parameters**:
     - `pdo` (int, optional): The points to double the odds. Default is 50.
     - `target_points` (int, optional): The standard scorecard points. Default is 600.
     - `target_odds` (int, optional): The standard scorecard odds. Default is 19.
     - `precision_points` (int, optional): The points decimal precision. Default is 0.
     - `score_type` (str, optional): The log-odds to use for the points card. Default is 'XAddEvidence'.
   - **Returns**:
     - `pd.DataFrame`: The points card.

5. `predict_score(X: pd.DataFrame) -> pd.Series`:
   - Predicts the score for a given dataset using the constructed scorecard.
   - **Parameters**:
     - `X` (`pd.DataFrame`): Features of the dataset.
   - **Returns**:
     - `pd.Series`: Predicted scores.

6. `sql_query` (property):
   - Property that returns the SQL query for deploying the scorecard.
   - **Returns**:
     - `str`: The SQL query for deploying the scorecard.

7. `generate_sql_query(table_name: str = "my_table") -> str`:
   - Converts a scorecard into an SQL format.
   - **Parameters**:
     - `table_name` (str): The name of the input table in SQL.
   - **Returns**:
     - `str`: The final SQL query for deploying the scorecard.

## `xbooster.explainer` - XGBoost Scorecard Explainer

This module provides functionalities for explaining XGBoost scorecards, including methods to extract split information, build interaction splits, visualize tree structures, plot feature importances, and more.

### Methods:

1. `extract_splits_info(features: str) -> list`:
   - Extracts split information from the DetailedSplit feature.
   - **Inputs**:
     - `features` (str): A string containing split information.
   - **Outputs**:
     - Returns a list of tuples containing split information (feature, sign, value).

2. `build_interactions_splits(scorecard_constructor: Optional[XGBScorecardConstructor] = None, dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame`:
   - Builds interaction splits from the XGBoost scorecard.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing split information.
   - **Outputs**:
     - Returns a pandas DataFrame containing interaction splits.

3. `split_and_count(scorecard_constructor: Optional[XGBScorecardConstructor] = None, dataframe: Optional[pd.DataFrame] = None, label_column: Optional[str] = None) -> pd.DataFrame`:
   - Splits the dataset and counts events for each split.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing features and labels.
     - `label_column` (Optional[str]): The label column in the dataframe.
   - **Outputs**:
     - Returns a pandas DataFrame containing split information and event counts.

4. `plot_importance(scorecard_constructor: Optional[XGBScorecardConstructor] = None, metric: str = "Likelihood", normalize: bool = True, method: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, **kwargs: Any) -> None`:
   - Plots the importance of features based on the XGBoost scorecard.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `metric` (str): Metric to plot ("Likelihood" (default), "NegLogLikelihood", "IV", or "Points").
     - `normalize` (bool): Whether to normalize the importance values (default: True).
     - `method` (Optional[str]): The method to use for plotting the importance ("global" or "local").
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing features and labels.
     - `fontfamily` (str): The font family to use for the plot (default: "Monospace").
     - `fontsize` (int): The font size to use for the plot (default: 12).
     - `dpi` (int): The DPI of the plot (default: 100).
     - `title` (str): The title of the plot (default: "Feature Importance").
     - `**kwargs` (Any): Additional Matplotlib parameters.

5. `plot_score_distribution(y_true: pd.Series = None, y_pred: pd.Series = None, n_bins: int = 25, scorecard_constructor: Optional[XGBScorecardConstructor] = None, **kwargs: Any)`:
   - Plots the distribution of predicted scores based on actual labels.
   - **Inputs**:
     - `y_true` (pd.Series): The true labels.
     - `y_pred` (pd.Series): The predicted labels.
     - `n_bins` (int): Number of bins for histogram (default: 25).
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `**kwargs` (Any): Additional Matplotlib parameters.

6. `plot_local_importance(scorecard_constructor: Optional[XGBScorecardConstructor] = None, metric: str = "Likelihood", normalize: bool = True, dataframe: Optional[pd.DataFrame] = None, **kwargs: Any) -> None`:
   - Plots the local importance of features based on the XGBoost scorecard.
   - **Inputs**:
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `metric` (str): Metric to plot ("Likelihood" (default), "NegLogLikelihood", "IV", or "Points").
     - `normalize` (bool): Whether to normalize the importance values (default: True).
     - `dataframe` (Optional[pd.DataFrame]): The dataframe containing features and labels.
     - `fontfamily` (str): The font family to use for the plot (default: "Arial").
     - `fontsize` (int): The font size to use for the plot (default: 12).
     - `boxstyle` (str): The rounding box style to use for the plot (default: "round").
     - `title` (str): The title of the plot (default: "Local Feature Importance").
     - `**kwargs` (Any): Additional parameters to pass to the matplotlib function.

7. `plot_tree(tree_index: int, scorecard_constructor: Optional[XGBScorecardConstructor] = None, show_info: bool = True) -> None`:
   - Plots the tree structure.
   - **Inputs**:
     - `tree_index` (int): Index of the tree to plot.
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `show_info` (bool): Whether to show additional information (default: True).
     - `**kwargs` (Any): Additional Matplotlib parameters.

# Contributing 🤝
Contributions are welcome! For bug reports or feature requests, please open an issue.

For code contributions, please open a pull request.

## Version
Current version: 0.2.2

## Changelog

### [0.1.0] - 2024-02-14
- Initial release

### [0.2.0] - 2024-05-03
- Added tree visualization class (`explainer.py`)
- Updated the local explanation algorithm for models with a depth > 1 (`explainer.py`)
- Added a categorical preprocessor (`_utils.py`)

### [0.2.1] - 2024-05-03
- Updates of dependencies

### [0.2.2] - 2024-05-08
- Updates in `explainer.py` module to improve kwargs handling and minor changes.

# License 📄
This project is licensed under the MIT License - see the LICENSE file for details.