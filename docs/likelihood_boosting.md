---
title: "The Likelihoodist Interpretation of Gradient Boosting"
author: "Denis Burakov"
date: "November 2023"
geometry: "margin=1in"
fontsize: 12pt
colorlinks: true
linkcolor: blue
urlcolor: blue
toccolor: blue
header-includes:
  - \usepackage{titling}
  - \pretitle{\begin{center}\LARGE}
  - \posttitle{\end{center}}
  - \preauthor{\begin{center}\Large}
  - \postauthor{\end{center}}
  - \predate{\begin{center}\large}
  - \postdate{\end{center}}
  - \usepackage{listings}
  - \usepackage{xcolor}
  - \usepackage{fontspec}
  - \setmonofont{Menlo}
  - \lstset{
      basicstyle=\ttfamily\small,
      keywordstyle=\color{blue},
      commentstyle=\color{green!60!black},
      stringstyle=\color{red},
      showstringspaces=false,
      breaklines=true,
      frame=single,
      numbers=left,
      numberstyle=\tiny\ttfamily,
      numbersep=5pt
    }
  - \usepackage{graphicx}
  - \usepackage[most]{tcolorbox}
  - \usepackage{mdframed}
  - \usepackage{needspace}
  - \setlength{\parskip}{6pt plus 2pt minus 1pt}
  - \setlength{\parindent}{0pt}
  - \definecolor{infoboxbackground}{RGB}{240, 247, 255}
  - \definecolor{infoboxborder}{RGB}{187, 222, 251}
  - \definecolor{featureboxbackground}{RGB}{240, 247, 255}
  - \definecolor{featureboxborder}{RGB}{66, 133, 244}
  - \definecolor{warningboxbackground}{RGB}{255, 243, 205}
  - \definecolor{warningboxborder}{RGB}{243, 156, 18}
  - \definecolor{theoremboxbackground}{RGB}{232, 245, 232}
  - \definecolor{theoremboxborder}{RGB}{165, 214, 167}
  - \definecolor{definitionboxbackground}{RGB}{255, 249, 230}
  - \definecolor{definitionboxborder}{RGB}{255, 217, 102}
  - \newmdenv[backgroundcolor=theoremboxbackground,linecolor=theoremboxborder,linewidth=2pt,roundcorner=5pt,innerleftmargin=15pt,innerrightmargin=15pt,innertopmargin=15pt,innerbottommargin=15pt,skipabove=15pt,skipbelow=15pt,leftmargin=0pt,rightmargin=0pt]{theorembox}
  - \newmdenv[backgroundcolor=featureboxbackground,linecolor=featureboxborder,linewidth=2pt,roundcorner=5pt,innerleftmargin=15pt,innerrightmargin=15pt,innertopmargin=15pt,innerbottommargin=15pt,skipabove=15pt,skipbelow=15pt,leftmargin=0pt,rightmargin=0pt]{featurebox}
  - \newmdenv[backgroundcolor=warningboxbackground,linecolor=warningboxborder,leftline=true,rightline=false,topline=false,bottomline=false,linewidth=2pt,innerleftmargin=20pt,innerrightmargin=15pt,innertopmargin=15pt,innerbottommargin=15pt,skipabove=15pt,skipbelow=15pt,leftmargin=0pt,rightmargin=0pt]{warningbox}
---

# The Likelihoodist Interpretation of Gradient Boosting

This document explores a likelihood-based perspective on gradient boosting machines, conceptualizing tree margins as additive evidence in favor of an event hypothesis.

## 1. Introduction

Gradient boosting algorithms have emerged as powerful tools in machine learning, demonstrating exceptional performance across classification and regression tasks. While celebrated for their predictive prowess, their inner workings often remain elusive. This paper presents a *likelihoodist interpretation* of gradient boosting margins, providing insights into feature selection, model optimization, and interpretability.

The key insight is that gradient boosted trees can be viewed as aggregations of margins, where each boosting iteration contributes a **likelihood** relative to a base score. This perspective enables us to assess the importance of individual splits within tree interactions using **likelihood ratios**.

## 2. Background: Weight of Evidence

### 2.1 Definition

The Weight of Evidence (WOE), introduced by I.J. Good (1950), measures the evidence in favor of a hypothesis. For a binary classification problem, WOE quantifies how much a given split favors the event class:

$$
\text{WOE} = \ln\left(\frac{P(\text{Event} | \text{Split})}{P(\text{Non-Event} | \text{Split})} \cdot \frac{P(\text{Non-Event})}{P(\text{Event})}\right)
$$

This simplifies to:

$$
\text{WOE} = \ln\left(\frac{\text{Events} / \Sigma\text{Events}}{\text{NonEvents} / \Sigma\text{NonEvents}}\right)
$$

### 2.2 Relationship to Odds

The WOE can also be expressed in terms of odds:

$$
\text{WOE} = \ln\left(\frac{\text{Odds}_{\text{split}}}{\text{Odds}_{\text{prior}}}\right)
$$

where:
- $\text{Odds}_{\text{split}} = \frac{\text{EventRate}_{\text{split}}}{1 - \text{EventRate}_{\text{split}}}$
- $\text{Odds}_{\text{prior}} = \frac{\text{EventRate}_{\text{global}}}{1 - \text{EventRate}_{\text{global}}}$

### 2.3 Likelihood from WOE

The likelihood is simply the exponentiated WOE:

$$
\mathcal{L} = e^{\text{WOE}}
$$

This converts the log-odds ratio back to a probability ratio, representing how much more (or less) likely the event is given the split condition.

## 3. Gradient Boosting as Likelihood Aggregation

### 3.1 The Margin Decomposition

A gradient boosted tree ensemble produces a prediction as a sum of margins:

$$
\text{margin}(x) = \text{base\_score} + \sum_{t=1}^{T} w_t(x)
$$

where:
- $\text{base\_score}$ is the initial log-odds (prior)
- $w_t(x)$ is the leaf weight from tree $t$ for observation $x$
- $T$ is the number of trees

In xBooster, the leaf weight is stored as `XAddEvidence` (additive evidence):

```python
# From xbooster/xgb_constructor.py
scorecard["XAddEvidence"] = leaf_weights  # Per-tree margin contribution
```

### 3.2 Interpreting Margins as Likelihoods

Each margin $w_t$ can be interpreted as a likelihood update relative to the prior:

$$
\mathcal{L}_t = e^{w_t}
$$

The boosting process iteratively updates the previous likelihood by fitting new decision trees. The final prediction aggregates these likelihoods:

$$
\text{Odds}_{\text{final}} = \text{Odds}_{\text{prior}} \times \prod_{t=1}^{T} \mathcal{L}_t = \text{Odds}_{\text{prior}} \times e^{\sum_t w_t}
$$

## 4. Example: Analyzing a Split

### 4.1 Two-Feature Interaction

Consider a split in a gradient boosting tree:

| Field | Value |
|-------|-------|
| Tree | 0 |
| Node | 3 |
| Feature | revolving_utilization |
| Sign | < |
| Split | 0.60931 |
| Count | 1901 |
| CountPct | 27.16% |
| NonEvents | 1669 |
| Events | 232 |
| EventRate | 12.20% |
| WOE | 0.224 |
| XAddEvidence | -0.1801 |
| DetailedSplit | account_never_delinq_percent < 98, revolving_utilization < 0.609 |

This is a depth-2 tree forming a two-way interaction. The event rate (12.20%) is lower than the base rate (16%), indicating lower risk for this segment.

### 4.2 Calculating Likelihood from Event Rates

Given:
- Prior odds: $0.10 / (1 - 0.10) = 0.111$
- Split event rate: 12.20%
- Split odds: $0.122 / (1 - 0.122) = 0.139$

The likelihood is:

$$
\mathcal{L} = \frac{\text{Odds}_{\text{split}}}{\text{Odds}_{\text{prior}}} = \frac{0.139}{0.111} = 1.251
$$

Taking the natural logarithm:

$$
\text{WOE} = \ln(1.251) = 0.224
$$

This matches the WOE value in the scorecard table.

### 4.3 Base Score Difference

The XAddEvidence is negative (-0.1801) while WOE is positive (0.224). This is because XGBoost uses a different base score (16%) compared to the sample average (10%). The direction of the evidence depends on the reference point.

## 5. Likelihood Ratios for Feature Importance

### 5.1 The Problem with Interactions

When a tree has `max_depth > 1`, each leaf represents an interaction of multiple features. For example:

```
account_never_delinq_percent < 98 AND revolving_utilization < 0.609
```

The final likelihood is attributed to the last split (revolving_utilization), but the first split (account_never_delinq_percent) may have a larger impact.

### 5.2 Decomposing Split Contributions

To measure the relative importance of each split, we compute individual likelihoods:

| Feature | Split | EventRate | Odds | Likelihood |
|---------|-------|-----------|------|------------|
| account_never_delinq_percent | < 98 | 23.99% | 0.316 | 2.840 |
| revolving_utilization | < 0.609 | 4.56% | 0.048 | 0.430 |
| *Leaf (combined)* | — | 12.20% | 0.139 | 1.251 |

The first split (account_never_delinq_percent < 98) has a likelihood of 2.84, which is much larger than the second split (0.43).

### 5.3 Computing the Likelihood Ratio

The likelihood ratio compares each split's likelihood to the final leaf likelihood:

$$
\text{LR} = \frac{\mathcal{L}_{\text{leaf}}}{\mathcal{L}_{\text{split}}}
$$

Equivalently, using WOE:

$$
\text{LR} = e^{\text{WOE}_{\text{split}} - \text{WOE}_{\text{leaf}}}
$$

| Feature | $\mathcal{L}_{\text{leaf}}$ | $\mathcal{L}_{\text{split}}$ | LR |
|---------|------|------|------|
| account_never_delinq_percent | 1.251 | 2.840 | 0.441 |
| revolving_utilization | 1.251 | 0.430 | 2.913 |

**Interpretation**:
- A likelihood ratio < 1 means the split alone is *more* predictive than the leaf (the split condition alone has higher event rate than the combined condition)
- A likelihood ratio > 1 means the split alone is *less* predictive than the leaf

In this example, the revolving_utilization split has a higher LR (2.91), meaning its individual contribution deviates more from the final leaf—indicating it's more important for determining this specific segment's risk.

## 6. Implementation in xBooster

### 6.1 WOE Calculation

The `calculate_weight_of_evidence` function in [`xbooster/_utils.py`](../xbooster/_utils.py) implements Good's formula:

```python
def calculate_weight_of_evidence(xgb_scorecard: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WOE using Good's formula from Bayes factor.
    The use of event to non-event ratio aligns with XAddEvidence direction.
    """
    woe_table = xgb_scorecard.copy()

    # Calculate cumulative totals
    woe_table["CumNonEvents"] = woe_table.groupby("Tree")["NonEvents"].transform("sum")
    woe_table["CumEvents"] = woe_table.groupby("Tree")["Events"].transform("sum")

    # WOE = ln(Events/ΣEvents) - ln(NonEvents/ΣNonEvents)
    woe_table["WOE"] = np.log(
        (woe_table["Events"] / woe_table["CumEvents"]) /
        (woe_table["NonEvents"] / woe_table["CumNonEvents"])
    )
    return woe_table
```

### 6.2 Likelihood Calculation

```python
def calculate_likelihood(xgb_scorecard: pd.DataFrame) -> pd.Series:
    """Convert WOE to likelihood by exponentiating."""
    woe_table = calculate_information_value(xgb_scorecard)
    woe_table["Likelihood"] = np.exp(woe_table["WOE"])
    return pd.Series(woe_table["Likelihood"], name="Likelihood")
```

### 6.3 Scorecard Construction

The `construct_scorecard()` method produces a table with both WOE and XAddEvidence:

```python
from xbooster.xgb_constructor import XGBScorecardConstructor

# Build scorecard
constructor = XGBScorecardConstructor(model, X_train, y_train)
scorecard = constructor.construct_scorecard()

# Key columns available:
# - XAddEvidence: Leaf weight (margin contribution)
# - WOE: Weight of Evidence
# - IV: Information Value
# - DetailedSplit: Full path condition
print(scorecard[["Tree", "Node", "Feature", "XAddEvidence", "WOE"]].head())
```

## 7. Practical Applications

### 7.1 Feature Selection

The likelihood ratio provides a principled way to rank features within an interaction:
- Features with LR closer to 1 contribute proportionally to the final prediction
- Features with LR far from 1 are more/less important than their leaf context suggests

### 7.2 Model Interpretation

Viewing margins as likelihoods enables:
- **Uncertainty quantification**: Each tree contributes a likelihood update
- **Feature importance**: Likelihood ratios reveal which splits drive predictions
- **Model diagnostics**: Compare WOE direction with XAddEvidence direction

### 7.3 Comparison with SHAP

| Aspect | Likelihoodist View | SHAP View |
|--------|-------------------|-----------|
| Decomposition | By tree/split | By feature |
| Base value | Prior odds | Expected value |
| Contribution | Likelihood update | Additive attribution |
| Interactions | Natural via LR | Captured in feature values |
| Reference | Statistical tradition | Game theory |

Both approaches are complementary: the likelihoodist view excels at understanding tree structure and split importance, while SHAP provides feature-level attribution.

## 8. Mathematical Summary

### 8.1 Key Formulas

**Weight of Evidence:**
$$
\text{WOE} = \ln\left(\frac{P(\text{Event}|\text{Split}) / P(\text{Non-Event}|\text{Split})}{P(\text{Event}) / P(\text{Non-Event})}\right)
$$

**Likelihood:**
$$
\mathcal{L} = e^{\text{WOE}}
$$

**Gradient Boosting Prediction:**
$$
\text{margin}(x) = \text{base\_score} + \sum_{t=1}^{T} w_t(x)
$$

**Likelihood Ratio:**
$$
\text{LR} = \frac{\mathcal{L}_{\text{leaf}}}{\mathcal{L}_{\text{split}}} = e^{\text{WOE}_{\text{split}} - \text{WOE}_{\text{leaf}}}
$$

### 8.2 The Additivity Property

In log space, likelihoods are additive:

$$
\ln(\text{Odds}_{\text{final}}) = \ln(\text{Odds}_{\text{prior}}) + \sum_{t=1}^{T} w_t
$$

This is precisely what gradient boosting computes, making the likelihoodist interpretation natural.

## 9. Conclusion

The likelihoodist interpretation of gradient boosting margins offers a principled statistical framework for understanding these powerful algorithms. By viewing each tree's contribution as a likelihood update, we gain:

1. **Theoretical clarity**: Connect boosting to classical statistical inference
2. **Practical tools**: Likelihood ratios for feature importance within interactions
3. **Interpretability**: Natural probabilistic interpretation of model components

This perspective complements SHAP-based interpretability and enriches both the theoretical foundations and practical applications of gradient boosting.

## References

1. Good, I. J. (1950). *Probability and the Weighing of Evidence*. Griffin.
2. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
4. Siddiqi, N. (2017). *Intelligent Credit Scoring*. Wiley.
