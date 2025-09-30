"""
Test module for xbooster.constructor.

The XGBScorecardConstructor class is responsible for constructing a scorecard from
a trained XGBoost model. This module provides test cases to ensure that the class
functions correctly.

Test cases:
- test_extract_model_param: Tests the extract_model_param function of the
  XGBScorecardConstructor class to assert the correctness of extracted model parameters.
- test_add_detailed_split: Tests the add_detailed_split method of the
  XGBScorecardConstructor class to ensure it returns a pandas DataFrame.
- test_extract_leaf_weights: Tests the extract_leaf_weights method to verify the
  returned DataFrame structure.
- test_extract_decision_nodes: Tests the extract_decision_nodes method to verify the
  returned DataFrame structure.
- test_construct_scorecard: Tests the construct_scorecard method to ensure it returns
  a non-empty pandas DataFrame.
- test_create_points: Tests the create_points method to verify it returns a non-empty
  pandas DataFrame.
- test_predict_score: Tests the predict_score method to ensure it returns a non-empty
  DataFrame with predicted scores.
- test_predict_scores: Tests the predict_scores method to ensure it returns a non-empty
  DataFrame with predicted scores.
- test_sql_query: Tests the `sql_query` attribute to verify it is a string.
- test_generate_sql_query: Tests the generate_sql_query method to ensure it returns a string.

"""

import pandas as pd
import pytest
import xgboost as xgb

from xbooster.xgb_constructor import XGBScorecardConstructor  # pylint: disable=E0401


@pytest.fixture(scope="module")
def xgb_model():
    """
    Creates and trains an XGBoost model.

    Returns:
        model (xgb.XGBClassifier): Trained XGBoost model.
    """
    X = pd.DataFrame(  # pylint: disable=C0103
        {
            "feature1": [
                37,
                61,
                51,
                92,
                49,
                35,
                7,
                8,
                58,
                27,
                38,
                15,
                66,
                11,
                20,
                21,
                40,
                25,
                95,
                3,
            ],
            "feature2": [
                26,
                76,
                27,
                90,
                4,
                46,
                27,
                23,
                49,
                57,
                49,
                5,
                6,
                48,
                92,
                81,
                52,
                56,
                86,
                46,
            ],
        }
    )
    y = pd.Series([1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    model = xgb.XGBClassifier()
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def scorecard_constructor(xgb_model):  # pylint: disable=W0621
    """
    Constructs an XGBScorecardConstructor object using the given XGBoost model,
    feature matrix (X), and target variable (y).

    Parameters:
    - xgb_model: The trained XGBoost model.

    Returns:
    - XGBScorecardConstructor: The initialized XGBScorecardConstructor object.
    """
    X = pd.DataFrame(  # pylint: disable=C0103
        {
            "feature1": [
                37,
                61,
                51,
                92,
                49,
                35,
                7,
                8,
                58,
                27,
                38,
                15,
                66,
                11,
                20,
                21,
                40,
                25,
                95,
                3,
            ],
            "feature2": [
                26,
                76,
                27,
                90,
                4,
                46,
                27,
                23,
                49,
                57,
                49,
                5,
                6,
                48,
                92,
                81,
                52,
                56,
                86,
                46,
            ],
        }
    )
    y = pd.Series([1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    return XGBScorecardConstructor(xgb_model, X, y)


@pytest.fixture(scope="module")
def X_test():  # pylint: disable=C0103
    """
    This function returns a sample input data frame.

    Returns:
        pd.DataFrame: A data frame with two features, feature1 and feature2.
    """
    return pd.DataFrame({"feature1": [55, 42, 30], "feature2": [35, 28, 18]})


def test_extract_model_param(scorecard_constructor):  # pylint: disable=W0621
    """
    Test the extract_model_param function of the XGBScorecardConstructor class.

    This function asserts the correctness of the extracted model parameters.

    Args:
        scorecard_constructor: An instance of the XGBScorecardConstructor class.

    Returns:
        None
    """
    assert abs(scorecard_constructor.extract_model_param("base_score") - 3.543437e-1) < 1e-2
    assert scorecard_constructor.extract_model_param("learning_rate") == 0.300000012
    assert scorecard_constructor.extract_model_param("max_depth") == 6


def test_add_detailed_split(scorecard_constructor):  # pylint: disable=W0621
    """
    Test the add_detailed_split method of the XGBScorecardConstructor class.

    Args:
        scorecard_constructor: An instance of the XGBScorecardConstructor class.

    Returns:
        None

    Raises:
        AssertionError: If the add_detailed_split method does not return a pd.DataFrame.
    """
    # Create a sample DataFrame with the provided data
    sample_dataframe = pd.DataFrame(
        {
            "Tree": [1],
            "Node": [1],
            "Feature": ["account_never_delinq_percent"],
            "Sign": ["<"],
            "Split": [98.0],
            "Count": [2810.0],
            "CountPct": [0.4014285714285714],
            "NonEvents": [2136.0],
            "Events": [674.0],
            "EventRate": [0.2398576512455516],
            "WOE": [1.0437644881684411],
            "IV": [0.6511102283336466],
            "XAddEvidence": [0.0512961857],
            "DetailedSplit": ["account_never_delinq_percent < 98"],
            "Points": [-1],
        }
    )

    # Call the add_detailed_split method with the sample DataFrame
    result_dataframe = scorecard_constructor.add_detailed_split(sample_dataframe)

    # Assert that the result is a DataFrame
    assert isinstance(result_dataframe, pd.DataFrame)

    scorecard_constructor.construct_scorecard()
    result_dataframe_none = scorecard_constructor.add_detailed_split(
        dataframe=scorecard_constructor.xgb_scorecard
    )
    # Assert that the result is a DataFrame
    assert result_dataframe_none is not None


def test_extract_leaf_weights(scorecard_constructor):  # pylint: disable=W0621
    """
    Test the extract_leaf_weights method of the XGBScorecardConstructor class.

    This test verifies that the extract_leaf_weights method returns a pandas DataFrame
    with the expected columns: 'Tree', 'Node', 'Feature', 'Sign', 'Split', and 'XAddEvidence'.
    """
    leaf_weights = scorecard_constructor.extract_leaf_weights()
    assert isinstance(leaf_weights, pd.DataFrame)
    assert set(leaf_weights.columns) == {
        "Tree",
        "Node",
        "Feature",
        "Sign",
        "Split",
        "XAddEvidence",
    }


def test_extract_decision_nodes(scorecard_constructor):  # pylint: disable=W0621
    """
    Test the extraction of decision nodes from the scorecard constructor.

    This function verifies that the extracted decision nodes are of type pd.DataFrame
    and have the expected columns: 'Tree', 'Node', 'Feature', 'Sign', 'Split', 'XAddEvidence'.
    """
    decision_nodes = scorecard_constructor.extract_decision_nodes()
    assert isinstance(decision_nodes, pd.DataFrame)
    assert set(decision_nodes.columns) == {
        "Tree",
        "Node",
        "Feature",
        "Sign",
        "Split",
        "XAddEvidence",
    }


def test_construct_scorecard(scorecard_constructor):  # pylint: disable=W0621
    """
    Test the construct_scorecard method of the XGBScorecardConstructor class.

    Parameters:
    - scorecard_constructor: An instance of the XGBScorecardConstructor class.

    Returns:
    - None

    Raises:
    - AssertionError: If the scorecard is not an instance of pd.DataFrame or empty.
    """
    # sourcery skip: no-conditionals-in-tests
    scorecard = scorecard_constructor.construct_scorecard()
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty


def test_create_points(scorecard_constructor):  # pylint: disable=W0621
    """
    Test case for the create_points method of XGBScorecardConstructor.

    This test verifies that the create_points method returns a non-empty pandas DataFrame.

    Args:
        scorecard_constructor (XGBScorecardConstructor): An instance of XGBScorecardConstructor.

    Returns:
        None
    """
    xgb_scorecard_with_points = scorecard_constructor.create_points()
    assert isinstance(xgb_scorecard_with_points, pd.DataFrame)
    assert not xgb_scorecard_with_points.empty


def test_predict_score(scorecard_constructor, X_test):  # pylint: disable=W0621, C0103
    """
    Test the predict_score method of the XGBScorecardConstructor class.

    Args:
        scorecard_constructor (XGBScorecardConstructor): An instance of XGBScorecardConstructor.
        X_test (pd.DataFrame): Input data for prediction.

    Raises:
        AssertionError: If the predict_score method returns None or an empty dataframe.

    Returns:
        None
    """
    result = scorecard_constructor.predict_score(X_test)
    assert result is not None, "Predicted score is None"
    assert not result.empty, "Predicted score dataframe is empty"


def test_predict_scores(scorecard_constructor, X_test):  # pylint: disable=W0621, C0103
    """
    Test the predict_scores method of the XGBScorecardConstructor class.

    Args:
        scorecard_constructor (XGBScorecardConstructor): An instance of XGBScorecardConstructor.
        X_test (pd.DataFrame): Input data for prediction.

    Raises:
        AssertionError: If the predict_scores method returns None or an empty dataframe.

    Returns:
        None
    """
    result = scorecard_constructor.predict_scores(X_test)
    assert result is not None, "Predicted scores are None"
    assert not result.empty, "Predicted scores dataframe is empty"


def test_sql_query(scorecard_constructor):  # pylint: disable=W0621
    """
    Test case to check the `sql_query` attribute of the scorecard_constructor.
    It verifies that the `sql_query` attribute is a string.
    """
    sql_query = scorecard_constructor.sql_query
    assert isinstance(sql_query, str)


def test_generate_sql_query(scorecard_constructor):  # pylint: disable=W0621
    """
    Test case for the generate_sql_query method of the XGBScorecardConstructor class.
    It checks if the return value is a string.
    """
    sql_query = scorecard_constructor.generate_sql_query()
    assert isinstance(sql_query, str)
