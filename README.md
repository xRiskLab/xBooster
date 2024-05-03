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
data = pd.read_csv("data.csv")
X = data.drop(columns=["target"])
y = data["target"]
model = xgb.XGBClassifier()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# Initialize XGBScorecardConstructor
scorecard_constructor = XGBScorecardConstructor(model, X_train, y_train)
scorecard_constructor.construct_scorecard()

# Print the scorecard
print(scorecard_constructor.scorecard)
```

After this we can create a scorecard and test its discrimination skill (Gini score):

```python
from xbooster.constructor import XGBScorecardConstructor

# Create scoring points
xgb_scorecard_with_points = scorecard_constructor.create_points(
    pdo=50, target_points=600, target_odds=50
)
# Make predictions using the scorecard
credit_scores = scorecard_constructor.predict_score(X_test)
gini = roc_auc_score(y_test, -credit_scores) * 2 - 1
print(f"Test Gini score: {gini:.2%}")
```

We can also visualize the score distribution between the events of interest:

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

Below we can visualize the global feature importances using Points as our metric:

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

Alternatively, we can calculate local feature importances, which is important for booster with a depth larger than 1.

```python
from xbooster import explainer

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
print(scorecard_constructor.sql_query)
```

For more detailed examples of usage, please check out the `\examples` directory.

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
     - `**kwargs` (Any): Additional parameters to pass to the matplotlib function.

7. `plot_tree(tree_index: int, scorecard_constructor: Optional[XGBScorecardConstructor] = None, show_info: bool = True) -> None`:
   - Plots the tree structure.
   - **Inputs**:
     - `tree_index` (int): Index of the tree to plot.
     - `scorecard_constructor` (Optional[XGBScorecardConstructor]): The XGBoost scorecard constructor.
     - `show_info` (bool): Whether to show additional information (default: True).
     - `**kwargs` (Any): Additional Matplotlib parameters.

# Categorical features 📊

For risk modeling, the use of categorical features is very domain-specific. The existing gradient boosting frameworks, including XGBoost, perform grouping of categories based on the target variable. However, this grouping may not be economically warranted, since absolutely different categories can be grouped together based on the similarities of event rates.

In xbooster, an interaction constraints approach is used for processing categories. This approach involves creating one-hot encoded representations of categorical features, which are later allowed to interact only with categories belonging to the same feature. This feature may appear instrumental for scorecard developers who would like to maintain the original idea behind scorecards.

The approach was proposed by Paul Edwards (https://github.com/pedwardsada).

Below we provide an example of usage:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from constructor import XGBScorecardConstructor
from _utils import DataPreprocessor

# Load dataset
dataset = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv", index_col=False)

# Define features, numerical features, and target variable
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

preprocessor = DataPreprocessor(numerical_features, categorical_features, target)

# Preprocess data
X, y = preprocessor.fit_transform(dataset)

features_ohe = [col for col in X.columns if col not in numerical_features]
interaction_constraints = preprocessor.generate_interaction_constraints(features_ohe)

print(interaction_constraints)

# Split data into train and test sets
ix_train, ix_test = train_test_split(
    X.index, stratify=y, test_size=0.3, random_state=62
)

best_params = dict(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=2,
    subsample=0.8,
    min_child_weight=1,
    grow_policy="lossguide",
    interaction_constraints=interaction_constraints,
    early_stopping_rounds=5,
)

# Create an XGBoost model
xgb_model = xgb.XGBClassifier(**best_params, random_state=62)
evalset = [(X.loc[ix_train], y.loc[ix_train]), (X.loc[ix_test], y.loc[ix_test])]

# Fit the XGBoost model
xgb_model.fit(
    X.loc[ix_train],
    y.loc[ix_train],
    eval_set=evalset,
    verbose=False,
)

# Set up the scorecard constructor
scorecard_constructor = XGBScorecardConstructor(
    xgb_model, X.loc[ix_train], y.loc[ix_train]
)

# Construct the scorecard
xgb_scorecard = scorecard_constructor.construct_scorecard()

# Create a scorecard with points
xgb_scorecard_with_points = scorecard_constructor.create_points(
    pdo=50, target_points=600, target_odds=50
)

# Make predictions using the scorecard
credit_scores = scorecard_constructor.predict_score(X.loc[ix_test])
gini = roc_auc_score(y.loc[ix_test], -credit_scores) * 2 - 1  # type: ignore

print(f"Test Gini score: {gini:.2%}")
```

# Theoretical Background ❝❞

In recent years, gradient boosting algorithms have revolutionized machine learning with their exceptional performance. However, understanding their inner workings and interpretability has often been challenging.

The methodology for scorecard boosting, pioneered by Paul Edwards and Scotiabank's risk modeling teams, addresses this gap. xbooster builds upon this methodology, extending it with the inclusion of Weight of Evidence (WOE) as another method for building credit scorecards with XGBoost.

The underlying framework of xbooster is grounded in likelihood theory. This framework enables researchers and practitioners to build and validate scoring systems, assess prediction uncertainty, quantify feature importance, and perform model diagnostics effectively.

Boosting iterations in xbooster produce margins, which can be interpreted as likelihood ratios relative to the base score. By treating these margins as likelihoods, xbooster conceptualizes the boosting process as an iterative procedure aimed at maximizing the likelihood of the observed data (Maximum Likelihood Estimation).

Weight of Evidence (WOE) plays a crucial role in this framework, acting as a (log) likelihood ratio that measures deviations from the average event rate. By leveraging WOE, xbooster facilitates the derivation of final probabilities for leaf nodes in a gradient boosting machine's tree ensemble.

## Useful resources 📖

Below are the resources I used to develop xbooster:

### Gradient boosting
- [How to Explain Gradient Boosting](https://explained.ai/gradient-boosting/)
- [Understanding Gradient Boosting as a Gradient Descent](https://nicolas-hug.com/blog/gradient_boosting_descent)
- [Around Gradient Boosting: Classification, Missing Values, Second Order Derivatives, and Line Search](https://nicolas-hug.com/blog/around_gradient_boosting)
- [How Does Extreme Gradient Boosting (XGBoost) Work?](https://cengiz.me/posts/extreme-gradient-boosting/)
### Scorecard boosting
- [Boosting for Credit Scorecards and Similarity to WOE Logistic Regression](https://github.com/pedwardsada/real_adaboost/blob/master/real_adaboost.pptx.pdf)
- [Machine Learning in Retail Credit Risk: Algorithms, Infrastructure, and Alternative Data — Past, Present, and Future](https://www.nvidia.com/ko-kr/on-demand/session/gtcspring21-s31327/)
- [Building Credit Risk Scorecards with RAPIDS](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)
- [XGBoost for Interpretable Credit Models](https://wandb.ai/tim-w/credit_scorecard/reports/XGBoost-for-Interpretable-Credit-Models--VmlldzoxODI0NDgx)
- [`credit_scorecard` - Project](https://wandb.ai/morgan/credit_scorecard/overview)
- [`vehicle_loan_defaults` - Artifacts 📊](https://wandb.ai/morgan/credit_scorecard/artifacts/dataset/vehicle_loan_defaults/v1)

# Contributing 🤝
Contributions are welcome! For bug reports or feature requests, please open an issue.

For code contributions, please open a pull request.

## Version
Current version: 0.2.0

## Changelog

### [0.1.0] - 2024-02-14
- Initial release

### [0.2.0] - 2024-05-03
- Added tree visualization class (`explainer.py`)
- Updated the local explanation algorithm for models with a depth > 1 (`explainer.py`)
- Added a categorical preprocessor (`_utils.py`)

# License 📄
This project is licensed under the MIT License - see the LICENSE file for details.