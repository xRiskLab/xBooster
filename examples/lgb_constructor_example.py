"""
LGBScorecardConstructor - Minimal Example

Demonstrates the core functionality of LightGBM scorecard constructor.
Status: Alpha - 3 of 5 methods implemented (PR #8)

Implemented:
  ✅ extract_leaf_weights() - Extract tree structure and leaf values
  ✅ get_leafs() - Get leaf indices or margins for new data
  ✅ construct_scorecard() - Create scorecard with WOE/IV metrics

To be implemented:
  ⏳ create_points() - Apply PDO scaling
  ⏳ predict_score() - Score new data using scorecard
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split

from xbooster.lgb_constructor import LGBScorecardConstructor

np.random.seed(42)

# ============================================================================
# 1. Create Dataset
# ============================================================================
print("=" * 80)
print("LGBScorecardConstructor - Testing Progress")
print("=" * 80)
print()

n_samples = 500
X = pd.DataFrame(
    {
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.randint(20000, 150000, n_samples),
        "debt_ratio": np.random.uniform(0, 1, n_samples),
    }
)

y = pd.Series(((X["debt_ratio"] > 0.5) | (X["income"] < 40000)).astype(int))
y ^= (np.random.random(n_samples) < 0.1).astype(int)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
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
    print(f"{col}: {unique_leaves} unique leaves")
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
# 8. Construct Scorecard with WOE and IV
# ============================================================================
print("=" * 80)
print("8. Constructing Scorecard with WOE/IV")
print("=" * 80)

scorecard = constructor.construct_scorecard()

print(f"Scorecard shape: {scorecard.shape}")
print(f"Columns: {scorecard.columns.tolist()}")
print()

print("First 10 rows of scorecard:")
print(scorecard.head(10))
print()

print("Scorecard summary statistics:")
print(scorecard[["Count", "Events", "NonEvents", "EventRate", "WOE", "IV"]].describe())
print()

# ============================================================================
# 8.1. Correlation Analysis: XAddEvidence vs WOE
# ============================================================================
print("=" * 80)
print("8.1. Correlation Analysis: XAddEvidence vs WOE")
print("=" * 80)

# Calculate Spearman correlation
spearman_corr, spearman_pval = spearmanr(scorecard["XAddEvidence"], scorecard["WOE"])
print(f"Spearman correlation: {spearman_corr:.6f} (p-value: {spearman_pval:.6e})")

# Calculate Kendall's tau correlation
kendall_corr, kendall_pval = kendalltau(scorecard["XAddEvidence"], scorecard["WOE"])
print(f"Kendall's tau correlation: {kendall_corr:.6f} (p-value: {kendall_pval:.6e})")

# ============================================================================
# 8.1a. Verification: Tree Margins Sum Correctly
# ============================================================================
print("=" * 80)
print("8.1a. Verification: Tree Margins Sum to Total Prediction")
print("=" * 80)

# Get leaf margins for training data - each tree is a separate column
train_margins = constructor.get_leafs(X_train, output_type="margin")
print(f"Margin matrix shape: {train_margins.shape} (samples × trees)")
print(f"Columns (trees): {train_margins.columns.tolist()}")
print()

# Show first few rows
print("First 5 samples of per-tree margins:")
print(train_margins.head())
print()

# Calculate expected log-odds from training data
train_event_rate = y_train.mean()
expected_log_odds = np.log(train_event_rate / (1 - train_event_rate))

print("Base Score Information:")
print(f"Training event rate: {train_event_rate:.6f}")
print(
    f"Expected log-odds: log({train_event_rate:.6f} / {1 - train_event_rate:.6f}) = {expected_log_odds:.6f}"
)
print(f"LightGBM base_score (calculated): {constructor.base_score:.6f}")
print()

# Verify that margins sum to total prediction
margin_sum = train_margins.sum(axis=1)
raw_pred = model.predict(X_train, raw_score=True)

print("Verification:")
print(f"  Margin sum (first 5): {margin_sum.values[:5]}")
print(f"  Raw prediction (first 5): {raw_pred[:5]}")
print(f"  Margins sum correctly: {np.allclose(margin_sum, raw_pred)}")
print()

print("Note on LightGBM behavior:")
print("  • LightGBM tree values are absolute predictions (not deltas like XGBoost)")
print("  • Tree margins sum directly to total raw score")
print("  • First tree contains learned predictions, not just base_score")
print()

# ============================================================================
# 8.2. Tree Visualization and Scorecard Comparison
# ============================================================================
print("=" * 80)
print("8.2. Tree Visualization and Scorecard Comparison for Tree 0")
print("=" * 80)

# Create tree digraph
print("Generating tree visualization for Tree 0...")
tree_graph = lgb.create_tree_digraph(model, tree_index=0, precision=6)

# Display the graph (this will create a visual representation)
# In Jupyter notebooks, this would render inline
# In scripts, it can be saved to file
print("Tree digraph created (use tree_graph.render() to save to file)")
print()

# Show corresponding scorecard entries for Tree 0
print("Scorecard entries for Tree 0:")
tree_0_scorecard = scorecard[scorecard["Tree"] == 0].copy()
print(tree_0_scorecard.to_string(index=False))
print()

# Summary statistics for Tree 0
print("Tree 0 Summary:")
print(f"  Number of splits: {len(tree_0_scorecard)}")
print(f"  Total observations: {tree_0_scorecard['Count'].sum():.0f}")
print(f"  Total IV: {tree_0_scorecard['IV'].sum():.4f}")
print(f"  WOE range: [{tree_0_scorecard['WOE'].min():.4f}, {tree_0_scorecard['WOE'].max():.4f}]")
print(
    f"  XAddEvidence range: [{tree_0_scorecard['XAddEvidence'].min():.4f}, {tree_0_scorecard['XAddEvidence'].max():.4f}]"
)
print()

# ============================================================================
# 9. Feature Importance Analysis from Scorecard
# ============================================================================
print("=" * 80)
print("9. Feature Importance from Scorecard")
print("=" * 80)

# Count how many times each feature appears in splits
feature_importance = (
    scorecard.groupby("Feature")
    .agg(
        {
            "Count": "sum",  # Total observations
            "IV": "sum",  # Total information value
            "WOE": "mean",  # Average WOE
            "XAddEvidence": ["count", "mean"],  # Split count and avg contribution
        }
    )
    .round(4)
)

feature_importance.columns = ["Total_Obs", "Total_IV", "Avg_WOE", "Split_Count", "Avg_Contribution"]
feature_importance = feature_importance.sort_values("Total_IV", ascending=False)

print("Feature importance ranked by Information Value:")
print(feature_importance)
print()

# ============================================================================
# 10. Compare with Native LightGBM Predictions
# ============================================================================
print("=" * 80)
print("10. Comparing with Native LightGBM Predictions")
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
# 11. Summary
# ============================================================================

print("=" * 80)
print("Example completed successfully!")
print("=" * 80)
