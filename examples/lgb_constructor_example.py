"""
Example: Using LGBScorecardConstructor

This example demonstrates how to use the LightGBM scorecard constructor
and inspect its outputs. Currently implements extract_leaf_weights() and get_leafs().

Status: Alpha - Partial implementation
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from xbooster.lgb_constructor import LGBScorecardConstructor

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Create Sample Dataset
# ============================================================================
print("=" * 80)
print("1. Creating Sample Dataset")
print("=" * 80)

n_samples = 1000
X = pd.DataFrame(
    {
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.randint(20000, 150000, n_samples),
        "debt_ratio": np.random.uniform(0, 1, n_samples),
        "credit_history": np.random.uniform(0, 30, n_samples),
        "employment_years": np.random.randint(0, 40, n_samples),
    }
)

# Create target variable with some relationship to features
y = pd.Series(
    ((X["debt_ratio"] > 0.5) | (X["income"] < 40000) | (X["credit_history"] < 5)).astype(int)
)

# Add some noise
y = y ^ (np.random.random(n_samples) < 0.1).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Event rate: {y.mean():.2%}")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print()

# ============================================================================
# 2. Train LightGBM Model
# ============================================================================
print("=" * 80)
print("2. Training LightGBM Model")
print("=" * 80)

model = LGBMClassifier(
    n_estimators=5,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
)

model.fit(X_train, y_train)

# Check model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
print(f"Number of trees: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
print(f"Learning rate: {model.learning_rate}")
print()

# ============================================================================
# 3. Initialize LGBScorecardConstructor
# ============================================================================
print("=" * 80)
print("3. Initializing LGBScorecardConstructor")
print("=" * 80)

constructor = LGBScorecardConstructor(model, X_train, y_train)

print(f"Base score (log-odds): {constructor.base_score:.6f}")
print(f"Number of estimators: {constructor.n_estimators}")
print(f"Learning rate: {constructor.learning_rate}")
print(f"Max depth: {constructor.max_depth}")
print()

# ============================================================================
# 4. Extract Leaf Weights
# ============================================================================
print("=" * 80)
print("4. Extracting Leaf Weights")
print("=" * 80)

leaf_weights = constructor.extract_leaf_weights()

print(f"Leaf weights shape: {leaf_weights.shape}")
print(f"Columns: {leaf_weights.columns.tolist()}")
print()

print("First 10 leaf weights:")
print(leaf_weights.head(10))
print()

print("Summary statistics:")
print(leaf_weights[["Tree", "Split", "XAddEvidence"]].describe())
print()

# Group by tree
print("Leaf weights per tree:")
tree_counts = leaf_weights.groupby("Tree").size()
print(tree_counts)
print()

# Group by feature
print("Leaf weights per feature:")
feature_counts = leaf_weights.groupby("Feature").size().sort_values(ascending=False)
print(feature_counts)
print()

# Check value ranges
print("XAddEvidence range by tree:")
value_ranges = leaf_weights.groupby("Tree")["XAddEvidence"].agg(["min", "max", "mean"])
print(value_ranges)
print()

# ============================================================================
# 5. Get Leaf Indices for Test Data
# ============================================================================
print("=" * 80)
print("5. Getting Leaf Indices for Test Data")
print("=" * 80)

leaf_indices = constructor.get_leafs(X_test, output_type="leaf_index")

print(f"Leaf indices shape: {leaf_indices.shape}")
print(f"Columns: {leaf_indices.columns.tolist()}")
print()

print("First 10 observations:")
print(leaf_indices.head(10))
print()

print("Unique leaf indices per tree:")
for col in leaf_indices.columns:
    unique_leaves = leaf_indices[col].nunique()
    print(f"  {col}: {unique_leaves} unique leaves")
print()

# ============================================================================
# 6. Get Margins for Test Data
# ============================================================================
print("=" * 80)
print("6. Getting Margins (Raw Scores) for Test Data")
print("=" * 80)

margins = constructor.get_leafs(X_test, output_type="margin")

print(f"Margins shape: {margins.shape}")
print(f"Columns: {margins.columns.tolist()}")
print()

print("First 10 observations:")
print(margins.head(10))
print()

print("Margin statistics per tree:")
print(margins.describe())
print()

# Verify margins sum to raw predictions
print("Verification: Margins + Base Score = Raw Predictions")
margin_sum = margins.sum(axis=1) + constructor.base_score
raw_pred = model.predict(X_test, raw_score=True)

print(f"Margin sum (first 5): {margin_sum.head().values}")
print(f"Raw prediction (first 5): {raw_pred[:5]}")
print(f"Match (all close): {np.allclose(margin_sum, raw_pred)}")
print()

# ============================================================================
# 7. Detailed Analysis of a Single Tree
# ============================================================================
print("=" * 80)
print("7. Detailed Analysis of Tree 0")
print("=" * 80)

tree_0_weights = leaf_weights[leaf_weights["Tree"] == 0]
print("Tree 0 leaf weights:")
print(tree_0_weights)
print()

# Get the tree structure from LightGBM
tree_df = model.booster_.trees_to_dataframe()
tree_0_structure = tree_df[tree_df["tree_index"] == 0]

print("Tree 0 structure:")
print()

# Split nodes (decision nodes)
split_nodes = tree_0_structure[tree_0_structure["split_feature"].notna()]
print("Split/Decision Nodes (internal nodes):")
print(
    split_nodes[
        [
            "node_index",
            "split_feature",
            "threshold",
            "decision_type",
            "left_child",
            "right_child",
        ]
    ]
)
print()

# Leaf nodes
leaf_nodes = tree_0_structure[tree_0_structure["split_feature"].isna()]
print("Leaf Nodes (terminal nodes):")
print(leaf_nodes[["node_index", "value"]])
print()

print(f"Total nodes: {len(tree_0_structure)} ({len(split_nodes)} split + {len(leaf_nodes)} leaf)")
print()

# ============================================================================
# 8. Feature Importance Analysis
# ============================================================================
print("=" * 80)
print("8. Feature Importance from Leaf Weights")
print("=" * 80)

# Count how many times each feature appears in splits
feature_importance = (
    leaf_weights.groupby("Feature")
    .agg({"XAddEvidence": ["count", "mean", "std", "min", "max"]})
    .round(4)
)

feature_importance.columns = ["Count", "Mean_Value", "Std_Value", "Min_Value", "Max_Value"]
feature_importance = feature_importance.sort_values("Count", ascending=False)

print("Feature importance based on split frequency:")
print(feature_importance)
print()

# ============================================================================
# 9. Compare with Native LightGBM Predictions
# ============================================================================
print("=" * 80)
print("9. Comparing with Native LightGBM Predictions")
print("=" * 80)

# Get predictions
pred_proba = model.predict_proba(X_test)[:, 1]
pred_raw = model.predict(X_test, raw_score=True)

# Show comparison for first 5 samples
comparison_df = pd.DataFrame(
    {
        "BaseScore": constructor.base_score,
        "MarginSum": margins.sum(axis=1).values[:5],
        "RawScore": pred_raw[:5],
        "Probability": pred_proba[:5],
    }
)

print("Prediction comparison (first 5 samples):")
print(comparison_df)
print()

# Calculate probability from raw score manually
manual_proba = 1 / (1 + np.exp(-pred_raw))
print(f"Manual probability calculation matches: {np.allclose(manual_proba, pred_proba)}")
print()

# ============================================================================
# 10. Summary
# ============================================================================
print("=" * 80)
print("10. Summary")
print("=" * 80)

print("✓ Successfully created LGBScorecardConstructor")
print("✓ Extracted leaf weights from LightGBM model")
print("✓ Retrieved leaf indices for test data")
print("✓ Retrieved margins (raw scores) for test data")
print("✓ Verified margin calculations match LightGBM's raw predictions")
print()

print("Currently implemented methods:")
print("  - extract_leaf_weights(): Extract tree structure and leaf values")
print("  - get_leafs(): Get leaf indices or margins for new data")
print()

print("Methods still to be implemented:")
print("  - construct_scorecard(): Combine leaf weights with event statistics")
print("  - create_points(): Apply PDO (Points to Double Odds) scaling")
print("  - predict_score(): Score new data using the scorecard")
print()

print("Next steps for full implementation:")
print("  1. Implement construct_scorecard() to calculate WOE/IV")
print("  2. Implement create_points() for credit score calculation")
print("  3. Implement predict_score() for inference")
print("  4. Add SQL query generation for deployment")
print()

print("=" * 80)
print("Example completed successfully!")
print("=" * 80)
