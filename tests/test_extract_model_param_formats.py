"""
Test module for extract_model_param with different value formats.

This test ensures that the extract_model_param method can handle different
string formats returned by XGBoost config, including values with brackets.

GitHub Issue: Value of base_score parameter is read as a string wrapped in
list brackets (e.g., '[9.9E-2]'), which caused a ValueError when attempting
to convert it directly to a float.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest
import xgboost as xgb

from xbooster.xgb_constructor import XGBScorecardConstructor


@pytest.fixture(scope="module")
def simple_model():
    """
    Creates and trains a simple XGBoost model.

    Returns:
        model (xgb.XGBClassifier): Trained XGBoost model.
    """
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    model = xgb.XGBClassifier(n_estimators=5, max_depth=1, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def simple_data():
    """
    Creates simple training data.

    Returns:
        tuple: X and y DataFrames.
    """
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_extract_model_param_with_brackets(simple_model, simple_data):
    """
    Test that extract_model_param handles values with brackets correctly.

    This test verifies the fix for GitHub issue where base_score came as '[9.9E-2]'
    instead of '9.9E-2', causing a ValueError.
    """
    X, y = simple_data
    constructor = XGBScorecardConstructor(simple_model, X, y)

    # Test with bracket format
    mock_config = {
        "learner": {
            "learner_model_param": {"base_score": "[9.9E-2]"},
            "gradient_booster": {
                "tree_train_param": {"learning_rate": "[0.3]", "max_depth": "[1]"}
            },
        }
    }

    with patch.object(constructor.booster_, "save_config", return_value=json.dumps(mock_config)):
        base_score = constructor.extract_model_param("base_score")
        learning_rate = constructor.extract_model_param("learning_rate")
        max_depth = constructor.extract_model_param("max_depth")

    assert abs(base_score - 0.099) < 1e-6
    assert abs(learning_rate - 0.3) < 1e-6
    assert abs(max_depth - 1.0) < 1e-6


def test_extract_model_param_without_brackets(simple_model, simple_data):
    """
    Test that extract_model_param handles values without brackets correctly.

    This ensures backward compatibility with the standard format.
    """
    X, y = simple_data
    constructor = XGBScorecardConstructor(simple_model, X, y)

    # Test without bracket format (standard)
    mock_config = {
        "learner": {
            "learner_model_param": {"base_score": "9.9E-2"},
            "gradient_booster": {"tree_train_param": {"learning_rate": "0.3", "max_depth": "1"}},
        }
    }

    with patch.object(constructor.booster_, "save_config", return_value=json.dumps(mock_config)):
        base_score = constructor.extract_model_param("base_score")
        learning_rate = constructor.extract_model_param("learning_rate")
        max_depth = constructor.extract_model_param("max_depth")

    assert abs(base_score - 0.099) < 1e-6
    assert abs(learning_rate - 0.3) < 1e-6
    assert abs(max_depth - 1.0) < 1e-6


def test_extract_model_param_various_formats(simple_model, simple_data):
    """
    Test that extract_model_param handles various numeric string formats.
    """
    X, y = simple_data
    constructor = XGBScorecardConstructor(simple_model, X, y)

    test_cases = [
        ("5E-1", 0.5),
        ("[5E-1]", 0.5),
        ("1.0", 1.0),
        ("[1.0]", 1.0),
        ("0.099", 0.099),
        ("[0.099]", 0.099),
    ]

    for value_str, expected in test_cases:
        mock_config = {
            "learner": {
                "learner_model_param": {"base_score": value_str},
                "gradient_booster": {"tree_train_param": {"learning_rate": "0.3"}},
            }
        }

        with patch.object(
            constructor.booster_, "save_config", return_value=json.dumps(mock_config)
        ):
            result = constructor.extract_model_param("base_score")

        assert abs(result - expected) < 1e-6, (
            f"Failed for {value_str}: got {result}, expected {expected}"
        )
