# xbooster 🚀

A scorecard-format famework for logistic regression tasks with gradient-boosted decision trees (XGBoost and CatBoost).
xbooster allows to convert a classification model into a logarithmic (point) scoring system.

In addition, it provides a suite of interpretability tools to understand the model's behavior.

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

### XGBoost Usage

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

### Interval Scorecards 📊

Convert complex tree-based scorecards into simplified interval-based rules. This feature requires `max_depth=1` models and follows industry standard practices (Siddiqi, 2017):

```python
# After creating a standard scorecard with points (see above)

# Build interval scorecard - simplifies complex rules into intervals
interval_scorecard = scorecard_constructor.construct_scorecard_by_intervals(add_stats=True)

print(f"Rule reduction: {len(xgb_scorecard_with_points)} → {len(interval_scorecard)} rules")
print("\nInterval format:")
print(interval_scorecard[['Feature', 'Bin', 'Points', 'WOE']].head())

# Add Points at Even Odds/Points to Double the Odds (PEO/PDO) 
peo_pdo_scorecard = scorecard_constructor.create_points_peo_pdo(peo=600, pdo=50)
print("\nPEO/PDO Points:")
print(peo_pdo_scorecard[['Feature', 'Bin', 'Points_PEO_PDO']].head())
```

**Key Benefits:**
- **Simplified Rules**: Transform complex tree conditions into simple intervals like `[70.8, 80.5)`
- **Rule Reduction**: Typically 60-80% fewer rules while maintaining accuracy
- **Industry Standard**: Follows credit scoring best practices
- **Interpretable**: Easy to understand and implement in production systems

### XGBoost Preprocessing

For handling categorical features in XGBoost, you can use the `DataPreprocessor`:

```python
from xbooster._utils import DataPreprocessor

# Define features and target
numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
categorical_features = [
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
target = "Loan_Status"

# Initialize preprocessor
preprocessor = DataPreprocessor(
    numerical_features, 
    categorical_features, 
    target
)

# Preprocess data
X, y = preprocessor.fit_transform(dataset)

# Get one-hot encoded feature names
features_ohe = [
    col for col in X.columns 
    if col not in numerical_features
]

# Generate interaction constraints for XGBoost
interaction_constraints = preprocessor.generate_interaction_constraints(features_ohe)
```

The `DataPreprocessor` provides:
1. Automatic one-hot encoding of categorical features
2. Proper handling of missing values
3. Generation of interaction constraints for XGBoost
4. Consistent feature naming for scorecard generation

### CatBoost Support 🐱 (Beta)

xbooster provides experimental support for CatBoost models with reduced functionality compared to XGBoost. Here's how to use it:

```python
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xbooster.constructor import CatBoostScorecardConstructor

# Load data and prepare features
data_path = "examples/data/test_data_01d9ab8b.csv"
credit_data = pd.read_csv(data_path)
num_features = ["Gross_Annual_Income", "Application_Score", "Bureau_Score"]
categorical_features = ["Time_with_Bank"]
features = num_features + categorical_features

# Prepare X and y
X = credit_data[features]
y = credit_data["Final_Decision"].replace({"Accept": 1, "Decline": 0})

# Create CatBoost Pool
pool = Pool(
    data=X,
    label=y,
    cat_features=categorical_features,
)

# Initialize and train CatBoost model
model = CatBoostClassifier(
    iterations=100,
    allow_writing_files=False,
    depth=1,
    learning_rate=0.1,
    verbose=0,
    one_hot_max_size=9999,  # Key for interpretability
)
model.fit(pool)

# Create and fit the scorecard constructor
constructor = CatBoostScorecardConstructor(model, pool)  # use_woe=False is the default, using raw LeafValue

# Alternatively, to use WOE values instead of raw leaf values:
# constructor = CatBoostScorecardConstructor(model, pool, use_woe=True)

# Construct the scorecard
scorecard = constructor.construct_scorecard()
print("\nScorecard:")
print(scorecard.head(3))

# Print raw leaf values
print("\nRaw Leaf Values:")
print(scorecard[["Tree", "LeafIndex", "LeafValue", "WOE"]].head(10))

# Make predictions using different methods - Do this BEFORE creating points
# Original CatBoost predictions
cb_preds = model.predict(X, prediction_type="RawFormulaVal")

# Get raw scores and WOE scores
raw_scores = constructor.predict_score(X, method="raw")
woe_scores = constructor.predict_score(X, method="woe")

# Now create points for the scorecard
scorecard_with_points = constructor.create_points(
    pdo=50, 
    target_points=600, 
    target_odds=19, 
    precision_points=0
)

# Calculate points-based scores
points_scores = constructor.predict_score(X, method="pdo")

# Even after creating points, raw and WOE scores remain consistent
# This is because the constructor maintains the original mappings
new_raw_scores = constructor.predict_score(X, method="raw")
new_woe_scores = constructor.predict_score(X, method="woe")

# Verify that raw scores still match CatBoost predictions
np.testing.assert_allclose(new_raw_scores, cb_preds, rtol=1e-2, atol=1e-2)

# Calculate Gini scores
from sklearn.metrics import roc_auc_score

raw_gini = 2 * roc_auc_score(y, raw_scores) - 1
woe_gini = 2 * roc_auc_score(y, woe_scores) - 1
points_gini = 2 * roc_auc_score(y, points_scores) - 1

print("\nGini Coefficients:")
print(f"Raw Scores: {raw_gini:.4f}")
print(f"WOE Scores: {woe_gini:.4f}")
print(f"Points Scores: {points_gini:.4f}")

# Get feature importance
feature_importance = constructor.get_feature_importance()
print("\nFeature Importance:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

# Visualize a tree
from xbooster._utils import CatBoostTreeVisualizer

visualizer = CatBoostTreeVisualizer(scorecard)
visualizer.plot_tree(tree_idx=0, title="CatBoost Tree Visualization")
```

### Limitations of CatBoost Support

The CatBoost implementation has some limitations compared to the XGBoost version:

1. Only supports depth=1 trees for interpretability
2. Limited support for categorical features
3. No SQL query generation
4. Reduced visualization options
5. No support for local feature importance
6. No support for score distribution plots

### CatBoost Preprocessing

For high-cardinality categorical features, you can use the `CatBoostPreprocessor`:

```python
from xbooster._utils import CatBoostPreprocessor

# Initialize preprocessor
preprocessor = CatBoostPreprocessor(max_categories=10)  # or top_p=0.9

# Fit and transform the data
X_processed = preprocessor.fit_transform(X, cat_features=categorical_features)

# Get the mapping of categories
category_maps = preprocessor.get_mapping()
```

### CatBoost Tree Visualization

The `CatBoostTreeVisualizer` class provides basic tree visualization with customizable settings:

```python
from xbooster._utils import CatBoostTreeVisualizer

# Initialize visualizer with custom configuration
plot_config = {
    "font_size": 12,
    "figsize": (12, 8),
    "level_distance": 8.0,
    "sibling_distance": 8.0,
    "fontfamily": "monospace",
    "yes_color": "#1f77b4",
    "no_color": "#ff7f0e",
    "leaf_color": "#2ca02c",
}

visualizer = CatBoostTreeVisualizer(scorecard, plot_config)
visualizer.plot_tree(tree_idx=0, title="Customized Tree Visualization")
```

## Parameters 🛠

### `xbooster.constructor` - XGBoost Scorecard Constructor

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

8. `construct_scorecard_by_intervals(add_stats=True) -> pd.DataFrame`:
   - Constructs a scorecard grouped by intervals of the type [a, b). Requires max_depth=1 models.
   - **Parameters**:
     - `add_stats` (bool, optional): Whether to include WOE, IV, and count statistics. Default is True.
   - **Returns**:
     - `pd.DataFrame`: The interval-based scorecard.

9. `create_points_peo_pdo(peo: int, pdo: int, precision_points: int = 0, scorecard: pd.DataFrame = None) -> pd.DataFrame`:
   - Creates Points at Even Odds/Points to Double the Odds (PEO/PDO) on interval scorecards.
   - **Parameters**:
     - `peo` (int): Points at Even Odds.
     - `pdo` (int): Points to Double the Odds.
     - `precision_points` (int, optional): Decimal precision for points. Default is 0.
     - `scorecard` (pd.DataFrame, optional): Specific scorecard to use. Default uses interval scorecard.
   - **Returns**:
     - `pd.DataFrame`: Scorecard with PEO/PDO points.

### `xbooster.explainer` - XGBoost Scorecard Explainer

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

## Contributing 🤝
Contributions are welcome! For bug reports or feature requests, please open an issue.

For code contributions, please open a pull request.

## Version
Current version: 0.2.6

## Changelog

### [0.2.6] - 2025-08-30
- Added interval scorecard functionality for XGBoost models with `max_depth=1`
- New methods: `construct_scorecard_by_intervals()` and `create_points_peo_pdo()`
- Simplifies complex tree rules into interpretable intervals following industry standards (Siddiqi, 2017)
- Typically achieves 60-80% rule reduction while maintaining accuracy

### [0.2.5] - 2025-04-19
- Minor changes in `catboost_wrapper.py` and `cb_constructor.py` to improve the scorecard generation.

### [0.2.4] - 2025-04-18
- Changed the build distribution in pyproject.toml.

### [0.2.3] - 2025-04-18
- Added support for CatBoost classification models and switch to `uv` for packaging.
- Python version requirement updated to 3.10-3.11.

### [0.2.2] - 2024-05-08
- Updates in `explainer.py` module to improve kwargs handling and minor changes.

### [0.2.1] - 2024-05-03
- Updates of dependencies

### [0.2.0] - 2024-05-03
- Added tree visualization class (`explainer.py`)
- Updated the local explanation algorithm for models with a depth > 1 (`explainer.py`)
- Added a categorical preprocessor (`_utils.py`)

### [0.1.0] - 2024-02-14
- Initial release

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.