import sys
from pathlib import Path

# Get the absolute path to the project root directory
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from xbooster.cb_constructor import CatBoostScorecardConstructor

# 1. Load and prepare the data
data_path = Path(__file__).parent / "data" / "test_data_01d9ab8b.csv"
credit_data = pd.read_csv(str(data_path))
num_features = ["Gross_Annual_Income", "Application_Score", "Bureau_Score"]
categorical_features = ["Time_with_Bank"]
features = num_features + categorical_features

# Prepare X and y
X = credit_data[features]
y = credit_data["Final_Decision"].replace({"Accept": 1, "Decline": 0})

# 2. Create CatBoost Pool
pool = Pool(
    data=X,
    label=y,
    cat_features=categorical_features,
)

# 3. Initialize and train CatBoost model
model = CatBoostClassifier(
    iterations=100,
    allow_writing_files=False,
    depth=1,
    learning_rate=0.1,
    verbose=0,
    one_hot_max_size=9999,  # Key for interpretability
)
model.fit(pool)

# 4. Create and fit the scorecard constructor
constructor = CatBoostScorecardConstructor(model, pool)

# 5. Construct the scorecard
scorecard = constructor.construct_scorecard()
print("\nScorecard:")
print(scorecard.head(3))

# Print raw leaf values
print("\nRaw Leaf Values:")
print(scorecard[["Tree", "LeafIndex", "LeafValue", "WOE"]].head(10))

# 6. Make predictions using different methods
raw_scores = constructor.predict_score(X, method="raw")
woe_scores = constructor.predict_score(X, method="woe")
points_scores = constructor.predict_score(X, method="pdo")

# Original CatBoost predictions
cb_preds = model.predict(X, prediction_type="RawFormulaVal").round(3)
cb_gini = 2 * roc_auc_score(y, cb_preds) - 1
print("\nGini Coefficients:")
print(f"CatBoost: {cb_gini:.4f}")

# Raw scores
raw_gini = 2 * roc_auc_score(y, raw_scores) - 1
print(f"Raw Scores: {raw_gini:.4f}")

# WOE scores
woe_gini = 2 * roc_auc_score(y, woe_scores) - 1
print(f"WOE Scores: {woe_gini:.4f}")

# Points scores
points_gini = 2 * roc_auc_score(y, points_scores) - 1
print(f"Points Scores: {points_gini:.4f}")

# 9. Get feature importance
feature_importance = constructor.get_feature_importance()
print("\nFeature Importance:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

# assert of raw scores and cb_preds
np.testing.assert_allclose(raw_scores, cb_preds, rtol=1e-2, atol=1e-2)
