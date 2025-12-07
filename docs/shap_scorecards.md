# SHAP Scorecards

> **Author:** Denis Burakov | **Date:** December 2025

This document explains how xBooster uses TreeSHAP to create scorecards for gradient-boosted trees, and the mathematical relationship between feature-based SHAP decomposition and tree-based margin decomposition.

## 1. SHAP-Based Scoring with `predict_score` and `predict_scores`

### 1.1 TreeSHAP Decomposition

For a gradient-boosted tree model, TreeSHAP decomposes the model's prediction into per-feature contributions. For an observation $x$, the model's log-odds (margin) can be written as:

$$
\text{margin}(x) = \phi_0 + \sum_{j=1}^{p} \phi_j(x)
$$

where:
- $\phi_0$ is the base value (expected value of the model output)
- $\phi_j(x)$ is the SHAP contribution for feature $j$
- $p$ is the number of features

The key property of SHAP values is that they sum to the difference between the prediction and the base value:

$$
\sum_{j=1}^{p} \phi_j(x) = \text{margin}(x) - \phi_0
$$

### 1.2 PDO Scaling

To convert log-odds to a scorecard scale, we use the Points to Double the Odds (PDO) method:

$$
\text{Score} = \text{Offset} - \text{Factor} \times \text{margin}
$$

where:
- $\text{Factor} = \frac{\text{PDO}}{\ln(2)}$
- $\text{Offset} = \text{Target Points} - \text{Factor} \times \ln(\text{Target Odds})$

With default parameters (PDO=50, Target Points=600, Target Odds=19):
- Factor ≈ 72.13
- Offset ≈ 387.60

### 1.3 Intercept Redistribution (SAS Method)

When computing feature-level scores, we follow the [SAS approach for scorecard development](https://documentation.sas.com/doc/en/emref/15.4/n181vl3wdwn89mn1pfpqm3w6oaz5.htm), which distributes the intercept and offset evenly across all features:

$$
\text{Score}_j = \text{Factor} \times (-\phi_j) + \frac{-\text{Intercept}_{\text{scaled}}}{p} + \frac{\text{Offset}}{p}
$$

where:
- $\text{Intercept}_{\text{scaled}} = \text{Factor} \times \phi_0$
- $p$ is the number of features

The total score is then:

$$
\text{Score}_{\text{total}} = \sum_{j=1}^{p} \text{round}(\text{Score}_j)
$$

**Important**: Each feature score is rounded first, then summed. This ensures that individual feature scores add up exactly to the total score, matching traditional scorecard behavior.

### 1.4 Implementation in xBooster

The `predict_score(method="shap")` method:

```python
# Extract SHAP values
shap_values = extract_shap_values_xgb(model, X, base_score)
feature_shap = shap_values[:, :-1]  # Per-feature contributions
base_value = shap_values[0, -1]     # Base value (φ₀)

# Scale to scorecard
intercept_scaled = factor * base_value
intercept_contribution = (-intercept_scaled) / n_features
offset_contribution = offset / n_features

for feature in features:
    feature_score = factor * (-shap[feature]) + intercept_contribution + offset_contribution
    feature_score = round(feature_score)

total_score = sum(feature_scores)
```

The `predict_scores(method="shap")` method returns the decomposed scores per feature, allowing interpretability at the feature level.

## 2. SHAP from the Scorecard Table

### 2.1 Tree-Based Margin Decomposition

An alternative decomposition is by tree rather than by feature. For an ensemble of $T$ trees:

$$
\text{margin}(x) = \phi_0 + \sum_{t=1}^{T} w_t(x)
$$

where $w_t(x)$ is the leaf weight (margin contribution) from tree $t$ for observation $x$.

**Key property**: All observations that fall into the same leaf of tree $t$ receive the same contribution $w_t$. This makes the per-tree contribution deterministic per leaf.

### 2.2 Storing SHAP in the Scorecard Table

When `construct_scorecard(shap=True)` is called, the scorecard table includes a SHAP column that stores the per-tree margin contribution for each (Tree, Node) combination:

| Tree | Node | Feature | Split | SHAP |
|------|------|---------|-------|------|
| 0 | 4 | debt_ratio | >= 0.47 | 0.456 |
| 0 | 6 | age | >= 30 | -0.119 |
| ... | ... | ... | ... | ... |

The SHAP value for each leaf is:
- **Deterministic**: All observations in the same leaf get the same value
- **Additive**: Summing across all trees gives the total margin contribution

### 2.3 Base Value Adjustment

There's a subtle difference between the base values used in different decompositions:
- **Constructor base_score**: The model's initial prediction (prior log-odds)
- **SHAP base_value**: TreeSHAP's expected value ($\phi_0$)

These can differ slightly. To ensure consistency, we adjust the table SHAP:

$$
\text{SHAP}^{(t)} = w_t + \frac{\text{base score} - \phi_0}{T}
$$

This adjustment distributes the base value difference across all trees, ensuring:

$$
\sum_{t=1}^{T} \text{SHAP}^{(t)} = \sum_{j=1}^{p} \phi_j
$$

### 2.4 Computing Scores from the Table

To compute a score for observation $x$ using the table:

1. **Find leaf indices**: For each tree $t$, determine which leaf $x$ falls into
2. **Sum SHAP values**: $\text{margin}_x = \sum_{t=1}^{T} \text{SHAP}_{\text{table}}^{(t)}$
3. **Apply PDO scaling**: $\text{Score} = \text{Factor} \times (-\text{margin}_x) - \text{Intercept}_{\text{scaled}} + \text{Offset}$
4. **Round**: $\text{Score} = \text{round}(\text{Score})$

```python
# Sum SHAP from table across all trees
total_shap = 0
for tree_idx in range(n_trees):
    node_idx = get_leaf_index(x, tree_idx)
    total_shap += scorecard[(Tree == tree_idx) & (Node == node_idx)]["SHAP"]

# Apply PDO scaling
score = factor * (-total_shap) - intercept_scaled + offset
score = round(score)
```

## 3. Equivalence of the Two Methods

### 3.1 The Fundamental Relationship

Both methods decompose the same model margin:

$$
\underbrace{\sum_{j=1}^{p} \phi_j(x)}_{\text{Feature SHAP}} = \underbrace{\sum_{t=1}^{T} w_t^{\text{adj}}(x)}_{\text{Table SHAP}} = \text{margin}(x) - \phi_0
$$

where $w_t^{\text{adj}}$ includes the base value adjustment.

### 3.2 Why Scores Match

When using the same scaling approach (no per-feature rounding):

$$
\text{Score}_{\text{feature}} = \text{Factor} \times \left(-\sum_{j=1}^{p} \phi_j\right) - \text{Intercept}_{\text{scaled}} + \text{Offset}
$$

$$
\text{Score}_{\text{table}} = \text{Factor} \times \left(-\sum_{t=1}^{T} w_t^{\text{adj}}\right) - \text{Intercept}_{\text{scaled}} + \text{Offset}
$$

Since $\sum_j \phi_j = \sum_t w_t^{\text{adj}}$, the scores are identical before rounding.

### 3.3 Rounding Differences

The only difference arises from rounding order:

| Method | Rounding | Result |
|--------|----------|--------|
| Feature SHAP (`intercept_based=True`) | Round each feature score, then sum | Integer feature scores that sum exactly |
| Table SHAP | Sum first, then round once | Single rounded total |

This can cause ±1 point differences, which is acceptable for scorecard applications.

### 3.4 Summary

| Aspect | Feature SHAP | Table SHAP |
|--------|--------------|------------|
| Decomposition | By feature | By tree |
| Deterministic per leaf? | No (varies by observation) | Yes (same for all in leaf) |
| Interpretability | Per-feature contributions | Per-tree contributions |
| Storage | Computed on-the-fly | Stored in scorecard table |
| Scores | Via `predict_score(method="shap")` | Sum SHAP from table, scale |

Both methods are mathematically equivalent and produce matching scores when using consistent base values and scaling approaches.

## 4. Example Code

A complete working example demonstrating the equivalence is available in [shap-in-leaf-weights.ipynb](../examples/shap-in-leaf-weights.ipynb).

```python
from xbooster.xgb_constructor import XGBScorecardConstructor
from xbooster.shap_scorecard import extract_shap_values_xgb

# Build scorecard with SHAP column
constructor = XGBScorecardConstructor(model, X_train, y_train)
scorecard = constructor.construct_scorecard(shap=True)

# Feature SHAP: sum across features
shap_full = extract_shap_values_xgb(model, X_test, constructor.base_score, False)
feature_shap_sum = shap_full[:, :-1].sum(axis=1)

# Table SHAP: sum across trees
leaf_indices = constructor.get_leafs(X_test, output_type="leaf_index")
table_shap_sum = [
    sum(scorecard[(scorecard["Tree"] == t) & (scorecard["Node"] == leafs.iloc[t])]["SHAP"].iloc[0]
        for t in range(n_trees))
    for leafs in leaf_indices.itertuples(index=False)
]

# Both sums are equal → scores match
assert np.allclose(feature_shap_sum, table_shap_sum)
```

Consult the example notebook [shap-in-leaf-weights.ipynb](../examples/shap-in-leaf-weights.ipynb) for a complete working example.

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. [SAS Scorecard Development Documentation](https://documentation.sas.com/doc/en/emref/15.4/n181vl3wdwn89mn1pfpqm3w6oaz5.htm)
