"""
File: test_utils.py
Description: This file contains the unit tests for the utility functions in the xbooster package.
"""

import pandas as pd
from xbooster import _utils  # pylint: disable=E0401

# Sample data for testing
xgb_scorecard = pd.DataFrame(
    {
        "Tree": [1, 1, 1],
        "NonEvents": [100, 150, 200],
        "Events": [50, 75, 100],
        "CumNonEvents": [250, 250, 250],
        "CumEvents": [150, 150, 150],
    }
)

dataset = pd.DataFrame(
    {
        "NumericalFeature1": [1, 2, 3],
        "NumericalFeature2": [4, 5, 6],
        "CategoricalFeature1": ["A", "B", "A"],
        "Target": [0, 1, 0],
    }
)


def test_data_preprocessor():
    """
    Test the DataPreprocessor class.
    """
    # Arrange
    numerical_features = [
        "NumericalFeature1",
        "NumericalFeature2",
    ]
    categorical_features = ["CategoricalFeature1"]
    target = "Target"

    preprocessor = _utils.DataPreprocessor(
        numerical_features, categorical_features, target
    )

    # pylint: disable=C0103
    preprocessor.fit(dataset)
    X, y = preprocessor.transform(
        dataset
    )

    transformed_columns = [
        col
        for col in X.columns
        if col not in numerical_features
    ]

    interaction_constraints = (
        preprocessor.generate_interaction_constraints(
            transformed_columns
        )
    )

    # Assert
    assert isinstance(
        X, pd.DataFrame
    ), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    assert len(X) == len(
        y
    ), "X and y should have the same length"
    assert (
        interaction_constraints is not None
    ), "Interaction constraints should not be empty"


# Test cases for calculate_weight_of_evidence
def test_calculate_weight_of_evidence():
    """
    Test case for the calculate_weight_of_evidence function.

    This function tests the calculate_weight_of_evidence function to ensure that it returns
    a DataFrame with the expected columns and numeric WOE values.

    Returns:
        None
    """
    # Act
    woe_table = _utils.calculate_weight_of_evidence(
        xgb_scorecard
    )

    # Assert
    assert isinstance(
        woe_table, pd.DataFrame
    ), "Result is not a DataFrame"
    assert all(
        col in woe_table.columns
        for col in [
            "Tree",
            "NonEvents",
            "Events",
            "CumNonEvents",
            "CumEvents",
            "WOE",
        ]
    ), "Columns are missing"
    assert all(
        isinstance(val, (int, float))
        for val in woe_table["WOE"]
    ), "WOE values are not numeric"


# Test cases for calculate_information_value
def test_calculate_information_value():
    """
    Test case for the calculate_information_value function.

    This function tests the calculate_information_value function to ensure that it
    returns a DataFrame with the expected columns and numeric IV values.

    Returns:
        None
    """
    # Act
    iv_table = _utils.calculate_information_value(
        xgb_scorecard
    )

    # Assert
    assert isinstance(
        iv_table, pd.DataFrame
    ), "Result is not a DataFrame"
    assert all(
        col in iv_table.columns
        for col in [
            "Tree",
            "NonEvents",
            "Events",
            "CumNonEvents",
            "CumEvents",
            "WOE",
            "IV",
        ]
    ), "Columns are missing"
    assert all(
        isinstance(val, (int, float))
        for val in iv_table["IV"]
    ), "IV values are not numeric"


# Test cases for calculate_likelihood
def test_calculate_likelihood():
    """
    Test the calculate_likelihood function.

    This function tests the calculate_likelihood function from the _utils module.
    It checks if the result is a pandas Series and if all the likelihood values are numeric.
    """
    # Act
    likelihood = _utils.calculate_likelihood(xgb_scorecard)

    # Assert
    assert isinstance(
        likelihood, pd.Series
    ), "Result is not a Series"
    assert all(
        isinstance(val, (int, float)) for val in likelihood
    ), "Likelihood values are not numeric"
