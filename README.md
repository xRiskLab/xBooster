# xbooster 🚀

<div align="center">
  <img src="examples/ims/xbooster.png" alt="xbooster" width="600"/>
</div>

<div align="center">

[![PyPI version](https://badge.fury.io/py/xbooster.svg)](https://badge.fury.io/py/xbooster)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/xRiskLab/xBooster/actions/workflows/ci.yml/badge.svg)](https://github.com/xRiskLab/xBooster/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/xbooster.svg)](https://pypi.org/project/xbooster/)

</div>

A scorecard framework for credit scoring tasks with gradient-boosted decision trees (XGBoost, LightGBM, and CatBoost).
xbooster allows to convert a classification model into a logarithmic (point) scoring system.

In addition, it provides a suite of interpretability tools to understand the model's behavior.

The interpretability suite includes:

- Granular boosted tree statistics, including metrics such as Weight of Evidence (WOE) and Information Value (IV) for splits 🌳
- Tree visualization with customizations 🎨
- Global and local feature importance 📊
- SHAP-based scoring for models with `max_depth > 1` 🧩

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
    "n_estimators": 100,
    "learning_rate": 0.55,
    "max_depth": 1,
    "min_child_weight": 10,
    "grow_policy": "lossguide",
    "early_stopping_rounds": 5
}
model = xgb.XGBClassifier(**best_params, random_state=62)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Initialize XGBScorecardConstructor
scorecard_constructor = XGBScorecardConstructor(model, X_train, y_train)
scorecard_constructor.construct_scorecard()

# Print the scorecard
print(scorecard_constructor.xgb_scorecard)
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

### SHAP-Based Scoring 🎯

xbooster supports SHAP-based scoring for all three libraries (XGBoost, LightGBM, and CatBoost). This is particularly useful for models with `max_depth > 1` where traditional scorecard interpretability is challenging.

**Key Features:**
- **Native SHAP extraction** - No external `shap` package required
- **On-demand computation** - SHAP values are computed only when needed
- **Feature-level decomposition** - Understand individual feature contributions
- **Consistent API** - Same interface across all three libraries

**Usage:**

```python
# Predict scores using SHAP method (no binning table needed)
shap_scores = scorecard_constructor.predict_score(X_test, method="shap")

# Decompose scores by feature using SHAP
shap_decomposed = scorecard_constructor.predict_scores(X_test, method="shap")
print(shap_decomposed.head())
# Output: DataFrame with columns like 'age_score', 'income_score', ..., 'score'

# Compare with traditional scorecard-based scoring
traditional_scores = scorecard_constructor.predict_score(X_test)  # Default method
```

**How it works:**
- SHAP values are computed on-the-fly using native library methods:
  - XGBoost: `pred_contribs=True`
  - LightGBM: `pred_contrib=True`
  - CatBoost: `get_feature_importance(type='ShapValues')`
- Values are automatically scaled using PDO (Points to Double the Odds) formula
- No need to call `create_points()` first - SHAP scoring works independently
- SHAP values are **not** stored in the scorecard binning table (computed only when needed)

**Intercept and Offset Distribution:**

By default, xbooster distributes the intercept (base value) and offset across all features when computing feature-level scores, matching the behavior of SAS scorecard modeling. This ensures that:

1. Each feature score includes its proportional share of the intercept and offset
2. The sum of all feature scores equals the total score (accounting for rounding)
3. The decomposition is consistent with industry-standard scorecard practices

This approach follows the SAS Enterprise Miner methodology for scorecard construction, where the base score is distributed across features rather than applied as a single constant. For more details, see the [SAS Enterprise Miner documentation](https://documentation.sas.com/doc/en/emref/15.4/n181vl3wdwn89mn1pfpqm3w6oaz5.htm).

You can control this behavior using the `intercept_based` parameter:

```python
# Default: distribute intercept and offset across features (SAS-like behavior)
shap_decomposed = scorecard_constructor.predict_scores(X_test, method="shap", intercept_based=True)

# Alternative: apply intercept and offset once to the total score
shap_decomposed = scorecard_constructor.predict_scores(X_test, method="shap", intercept_based=False)
```

**Example with all three libraries:**

```python
# XGBoost
xgb_scores_shap = xgb_constructor.predict_score(X_test, method="shap")

# LightGBM
lgb_scores_shap = lgb_constructor.predict_score(X_test, method="shap")

# CatBoost
cb_scores_shap = cb_constructor.predict_score(X_test, method="shap")
```

For detailed examples, see the [SHAP Scorecard Examples notebook](examples/shap-scorecard-examples.ipynb).

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

### LightGBM Usage

xbooster provides support for LightGBM models with scorecard functionality. Here's how to use it:

```python
import pandas as pd
import lightgbm as lgb
from xbooster.constructor import LGBScorecardConstructor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
url = "https://github.com/xRiskLab/xBooster/raw/main/examples/data/credit_data.parquet"
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
    X, y, test_size=0.3, random_state=62, stratify=y
)

# Train LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=50,
    learning_rate=0.55,
    max_depth=1,
    num_leaves=2,
    min_child_samples=10,
    random_state=62,
    verbose=-1,
)
model.fit(X_train, y_train)

# Initialize LGBScorecardConstructor
constructor = LGBScorecardConstructor(model, X_train, y_train)

# Construct scorecard
scorecard = constructor.construct_scorecard()
print(scorecard.head())

# Create points with base score normalization (default)
scorecard_with_points = constructor.create_points(
    pdo=50,
    target_points=600,
    target_odds=19,
    precision_points=0,
    use_base_score=True  # Ensures proper tree contribution balancing
)

# Make predictions
credit_scores = constructor.predict_score(X_test)

# Calculate Gini
gini = roc_auc_score(y_test, -credit_scores) * 2 - 1
print(f"Scorecard Gini: {gini:.4f}")

# Compare with model predictions
model_gini = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) * 2 - 1
print(f"Model Gini: {model_gini:.4f}")
```

**Key Features:**
- **Scorecard Construction**: Implementation of `create_points()` and `predict_score()`
- **Base Score Normalization**: Proper handling of LightGBM's base score for balanced tree contributions
- **High Discrimination**: Scorecard Gini closely matches model Gini
- **Flexible**: `use_base_score` parameter for optional base score normalization

**Important Notes:**
- LightGBM's sklearn API handles base_score differently than XGBoost
- The `use_base_score=True` parameter (default) ensures proper normalization
- Only `XAddEvidence` score type is supported (WOE not applicable)

### CatBoost Usage

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
constructor = CatBoostScorecardConstructor(model, pool)
# Construct the scorecard
scorecard = constructor.construct_scorecard()
print("\nScorecard:")
print(scorecard.head(3))

# Print raw XAddEvidence values
print("\nRaw XAddEvidence Values:")
print(scorecard[["Tree", "LeafIndex", "XAddEvidence", "WOE"]].head(10))

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

1. **Depth recommendation**: While the code supports any tree depth (as long as trees are complete binary), `depth=1` is recommended for better interpretability. Deeper trees work but may be harder to interpret.
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

## Contributing 🤝
Contributions are welcome! For bug reports or feature requests, please open an issue.

For code contributions, please open a pull request.

## Changelog 📝
For a changelog, see [CHANGELOG](CHANGELOG.md).

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.
