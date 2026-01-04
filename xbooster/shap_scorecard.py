"""
shap_scorecard.py

This module provides functions for computing scores directly from SHAP values
without using pre-computed binned scorecards. This is useful for models with
max_depth > 1 where interpretability is challenging.

Author: Denis Burakov
Github: @deburky
License: MIT
This code is licensed under the MIT License.
Copyright (c) 2025 xRiskLab
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, TypeVar, overload

import numpy as np
import pandas as pd

T = TypeVar("T")


@overload
def _try_import(module_name: str) -> Any: ...


@overload
def _try_import(module_name: str, *, fromlist: list[str]) -> tuple[Any, ...] | None: ...


def _try_import(module_name: str, *, fromlist: list[str] | None = None) -> Any:
    """
    Attempt to import a module or attributes, returning None on failure.

    Args:
        module_name: Name of the module to import
        fromlist: Optional list of attribute names to import from the module

    Returns:
        Imported module, attribute(s), or None if import fails
    """
    try:
        module = importlib.import_module(module_name)
        if not fromlist:
            return module
        attrs = [getattr(module, name) for name in fromlist]
        return attrs[0] if len(attrs) == 1 else tuple(attrs)
    except (ImportError, AttributeError):
        return None


# Optional ML library imports
xgb = _try_import("xgboost")
LGBMClassifier = _try_import("lightgbm", fromlist=["LGBMClassifier"])

# Import multiple from same module efficiently
Pool, CatBoostClassifier = _try_import("catboost", fromlist=["Pool", "CatBoostClassifier"]) or (
    None,
    None,
)


def compute_shap_scores(
    shap_values: np.ndarray,
    base_value: float,
    feature_names: list,
    scorecard_dict: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Convert SHAP values into a scorecard-like system using intercept-based scoring.

    This function computes feature-level scores from SHAP values and maps them to a
    traditional scorecard scale (PDO, target points, target odds). The intercept and
    offset are distributed evenly across all features, ensuring that feature scores
    sum exactly to the final total score (SAS-style behavior).

    Each feature score includes:
        - A scaled SHAP contribution
        - An equal share of the intercept term
        - An equal share of the offset

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features).
    base_value : float
        SHAP expected value (log-odds). Required for correct scaling.
    feature_names : list of str
        Names of features corresponding to columns in shap_values.
    scorecard_dict : dict, optional
        Dictionary containing scoring scale parameters:
            - "pdo": points to double the odds (default=50)
            - "target_points": reference score (default=600)
            - "target_odds": reference odds (default=19)

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
            - {feature}_score columns for each feature
            - "score" column representing the total score (sum of feature scores)

    Example
    -------
    >>> from xbooster.shap import extract_shap_values_xgb, compute_shap_scores
    >>> shap_values_full = extract_shap_values_xgb(model, X, base_score)
    >>> shap_values = shap_values_full[:, :-1]
    >>> base_value = float(np.mean(shap_values_full[:, -1]))
    >>> scorecard = compute_shap_scores(
    ...     shap_values=shap_values,
    ...     base_value=base_value,
    ...     feature_names=X.columns.tolist(),
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

    # Scale the intercept by factor
    intercept_scaled = factor * base_value

    # Distribute intercept and offset across features (matches SAS behavior)
    n_features = shap_values.shape[1]

    # Distribute both intercept and offset
    intercept_contribution = (-intercept_scaled) / n_features
    offset_contribution = offset / n_features

    # Vectorized computation of all feature scores at once
    feature_scores = factor * (-shap_values) + intercept_contribution + offset_contribution

    # Create DataFrame with rounded integer scores
    feature_score_cols = [f"{f}_score" for f in feature_names]
    scorecard_df = pd.DataFrame(
        np.round(feature_scores).astype(np.int64),
        columns=feature_score_cols,
    )

    # Total score is the sum of rounded feature scores
    scorecard_df["score"] = scorecard_df[feature_score_cols].sum(axis=1).astype(int)

    return scorecard_df


def extract_shap_values_xgb(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,  # pylint: disable=C0103
    base_score: float,
    enable_categorical: bool = False,
) -> np.ndarray:
    """
    Extract SHAP values from XGBoost model using native pred_contribs.

    Args:
        model: Trained XGBoost classifier
        X: Input features DataFrame
        base_score: Base score from the model
        enable_categorical: Whether categorical features are enabled

    Returns:
        Array of shape (n_samples, n_features + 1) where last column is base_score.
        Feature SHAP values are in columns [:, :-1], base_score is in column [:, -1].
    """
    if xgb is None:
        raise ImportError("xgboost is required for XGBoost SHAP extraction")
    booster = model.get_booster()
    scores = np.full((X.shape[0],), base_score)
    if enable_categorical:
        dmatrix = xgb.DMatrix(X, base_margin=scores, enable_categorical=True)
    else:
        dmatrix = xgb.DMatrix(X, base_margin=scores)
    return booster.predict(dmatrix, pred_contribs=True)


def extract_shap_values_lgb(
    model: LGBMClassifier,
    X: pd.DataFrame,  # pylint: disable=C0103
) -> np.ndarray:
    """
    Extract SHAP values from LightGBM model using native pred_contrib.

    Args:
        model: Trained LightGBM classifier
        X: Input features DataFrame

    Returns:
        Array of shape (n_samples, n_features + 1) where last column is base_score.
        Feature SHAP values are in columns [:, :-1], base_score is in column [:, -1].
    """
    if LGBMClassifier is None:
        raise ImportError("lightgbm is required for LightGBM SHAP extraction")
    return model.predict(X, pred_contrib=True)


def extract_shap_values_cb(
    model: CatBoostClassifier,
    pool: "Pool",
) -> np.ndarray:
    """
    Extract SHAP values from CatBoost model using native get_feature_importance.

    Args:
        model: Trained CatBoost classifier
        pool: CatBoost Pool object

    Returns:
        Array of shape (n_samples, n_features + 1) where last column is base_score.
        Feature SHAP values are in columns [:, :-1], base_score is in column [:, -1].
        CatBoost SHAP format: [feature1, feature2, ..., featureN, expected_value]
    """
    if CatBoostClassifier is None or Pool is None:
        raise ImportError("catboost is required for CatBoost SHAP extraction")
    return model.get_feature_importance(type="ShapValues", data=pool)
