# xBooster 🚀

xBooster is a Python package designed to enhance the interpretability and explainability of XGBoost models. 

It provides tools for constructing gradient boosted scorecards, generating local interpretations, and visualizing model explanations.

## Features ✨

    1️⃣ Construct (credit) scorecards for XGBoost models and make inference.

    2️⃣ Visualize feature importances using several metrics and two methods.

    3️⃣ Generate local explanations for model predictions.

    4️⃣ Generate SQL queries for boosted scorecards for easy deployment (e.g., with DuckDB).

> The methodology for explainers leverages the concepts of Weight-of-Evidence (WOE) and Fisher's Likelihood in calculating feature importances and local explanations. 🎲 For instance, booster's margins are seen as likelihoods and are conceptually similar to WOE. 📈 A scorecard can be constructed from WOE (natural logarithm of likelihood) based on booster's split information.<br><br>
> The results from explainer are highly consistent with SHAP values, but do not require significant computational resources, since all information is taken from the booster's model. 💡 This means that you can gain valuable insights into your model's behavior without the heavy computational overhead typically associated with SHAP computations. 🚀

## Installation 🛠️

You can install xBooster via pip:

```bash
pip install xbooster
```

## Usage 📝
Here's a quick example of how to use xBooster to construct a scorecard for an XGBoost model:

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

Below we can visualize the global feature importances using `Points` as our metric:

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

For more detailed examples and documentation, please refer to the [documentation](https://xbooster.readthedocs.io/en/latest/) and check out the `\notebooks` directory.

# Contributing 🤝
Contributions are welcome! For bug reports or feature requests, please open an issue. 

For code contributions, please open a pull request.

# License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.