"""
SHAP-based scorecard computation.

This module provides functions for computing scores directly from SHAP values
without using pre-computed binned scorecards. This is useful for models with
max_depth > 1 where interpretability is challenging.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_shap_scores(
    model=None,
    X: Optional[pd.DataFrame] = None,  # pylint: disable=C0103
    y: Optional[pd.Series] = None,
    shap_values: Optional[np.ndarray] = None,
    base_value: Optional[float] = None,
    scorecard_dict: Optional[Dict[str, float]] = None,
    feature_names: Optional[list] = None,
    negate_shap: bool = True,
) -> pd.DataFrame:
    """
    Convert SHAP values into a scorecard-like system.

    This function computes scores directly from SHAP values without using
    pre-computed binned scorecards. The approach is different from XAddEvidence-based
    scorecards which rely on binning tables.

    Parameters:
    -----------
    model: Trained ML model (optional, if shap_values and base_value are provided)
    X: Input dataset (required if model is provided)
    y: Target variable (optional, used to estimate base_score if not provided)
    shap_values: Precomputed SHAP values array of shape (n_samples, n_features)
    base_value: Base log-odds score (expected value). If None, will be estimated.
    scorecard_dict: Config for score scaling (PDO, target points, target odds)
    feature_names: List of feature names (required if shap_values is provided)

    Returns:
    --------
    pd.DataFrame: Scorecard with feature-wise contributions and final score.
        Columns: {feature}_score for each feature, and 'score' for final score.

    Example:
    --------
    >>> shap_values, base_value = extract_shap_values(model, X)
    >>> scorecard = compute_shap_scores(
    ...     shap_values=shap_values,
    ...     base_value=base_value,
    ...     feature_names=X.columns,
    ...     scorecard_dict={"pdo": 50, "target_points": 600, "target_odds": 19},
    ... )
    """
    if scorecard_dict is None:
        scorecard_dict = {
            "pdo": 50,
            "target_points": 600,
            "target_odds": 19,
        }

    pdo = scorecard_dict["pdo"]
    target_points = scorecard_dict["target_points"]
    target_odds = scorecard_dict["target_odds"]

    # Compute scaling factor and offset
    factor = pdo / np.log(2)
    offset = target_points - factor * np.log(target_odds)

    # Get SHAP values and base value
    if shap_values is not None and base_value is not None:
        # Use provided SHAP values
        if feature_names is None:
            raise ValueError("feature_names must be provided when using precomputed SHAP values")
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        intercept_ = base_value
    elif model is not None and X is not None:
        # Extract SHAP values from model
        # This is a placeholder - actual extraction should be done by the constructor
        raise NotImplementedError(
            "Direct model SHAP extraction not implemented. "
            "Please use extract_shap_values() from the constructor and pass shap_values and base_value."
        )
    else:
        raise ValueError(
            "Either (shap_values, base_value, feature_names) or (model, X) must be provided"
        )

    # Scale the intercept by factor (as per user requirement)
    intercept_scaled = factor * intercept_

    # Compute feature-level scores: factor * -shap_value (or factor * shap_value if negate_shap=False)
    # Note: We typically use -shap_value because higher SHAP (more positive) should reduce score
    # However, CatBoost's SHAP values may have different sign convention
    # The intercept is subtracted once from the total (not per feature)
    # Formula: prediction = sum(SHAP) + base_value (in log-odds)
    # Score = -factor * prediction + offset = -factor * (sum(SHAP) + base_value) + offset
    # = -factor * sum(SHAP) - factor * base_value + offset
    scorecard_df = pd.DataFrame()
    shap_multiplier = -1 if negate_shap else 1
    for feature in shap_df.columns:
        scorecard_df[f"{feature}_score"] = factor * (shap_multiplier * shap_df[feature])

    # Compute final score by summing feature-level scores, subtracting scaled intercept once, and adding offset
    # Formula: factor * sum(-shap) - factor * intercept + offset
    scorecard_df["score"] = scorecard_df.sum(axis=1) - intercept_scaled + offset

    # Return as integers (not floats) to avoid .0 display
    return scorecard_df.round(0).astype(int)


def compute_shap_scores_decomposed(
    shap_values: np.ndarray,
    base_value: float,
    feature_names: list,
    scorecard_dict: Optional[Dict[str, float]] = None,
    n_trees: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert SHAP values into decomposed scores (by feature and optionally by tree).

    This function computes scores directly from SHAP values and provides feature-level
    decomposition. For tree-level decomposition, SHAP values would need to be computed
    per tree, which is not directly supported by native SHAP implementations.

    Parameters:
    -----------
    shap_values: Precomputed SHAP values array of shape (n_samples, n_features)
    base_value: Base log-odds score (expected value)
    feature_names: List of feature names
    scorecard_dict: Config for score scaling (PDO, target points, target odds)
    n_trees: Number of trees (optional, for tree-level decomposition if available)

    Returns:
    --------
    pd.DataFrame: Scorecard with feature-wise contributions and final score.
        Columns: {feature}_score for each feature, and 'score' for final score.
    """
    if scorecard_dict is None:
        scorecard_dict = {
            "pdo": 50,
            "target_points": 600,
            "target_odds": 19,
        }

    pdo = scorecard_dict["pdo"]
    target_points = scorecard_dict["target_points"]
    target_odds = scorecard_dict["target_odds"]

    # Compute scaling factor and offset
    factor = pdo / np.log(2)
    offset = target_points - factor * np.log(target_odds)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    intercept_scaled = factor * base_value

    # Compute feature-level scores
    scorecard_df = pd.DataFrame()
    for feature in shap_df.columns:
        scorecard_df[f"{feature}_score"] = factor * (-shap_df[feature]) + intercept_scaled

    # Compute final score
    scorecard_df["score"] = scorecard_df.sum(axis=1) + offset

    return scorecard_df.round(0)
