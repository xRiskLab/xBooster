"""
Test module for xbooster.catboost_scorecard.

This module provides test cases for the CatBoostScorecard class, which is responsible
for extracting and processing scorecard data from CatBoost models.
"""

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool  # type: ignore
from sklearn.datasets import make_classification  # type: ignore

from xbooster.catboost_scorecard import CatBoostScorecard


# pylint: disable=protected-access,redefined-outer-name
@pytest.fixture(scope="module")
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, y


@pytest.fixture(scope="module")
def trained_model(sample_data):  # noqa: F821
    """Create a trained CatBoost model."""
    X, y = sample_data
    model = CatBoostClassifier(iterations=10, depth=3, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def test_pool(sample_data):  # noqa: F821
    """Create a CatBoost Pool."""
    X, y = sample_data
    return Pool(X, y)


def test_trees_to_scorecard(trained_model: CatBoostClassifier, test_pool: Pool):
    """Test the trees_to_scorecard method."""
    scorecard = CatBoostScorecard.trees_to_scorecard(trained_model, test_pool)

    # Check basic structure
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty

    # Check required columns
    required_columns = {
        "Tree",
        "LeafIndex",
        "Feature",
        "Sign",
        "Split",
        "Count",
        "NonEvents",
        "Events",
        "EventRate",
        "LeafValue",
        "WOE",
        "IV",
        "xAddEvidence",
        "CountPct",
        "DetailedSplit",
    }
    # Using issuperset instead of equality to be more flexible
    assert set(scorecard.columns).issuperset(required_columns)

    # Check data types
    assert scorecard["Tree"].dtype == np.int64
    assert scorecard["LeafIndex"].dtype == np.int64
    assert scorecard["Count"].dtype == np.float64
    assert scorecard["NonEvents"].dtype == np.float64
    assert scorecard["Events"].dtype == np.float64
    assert scorecard["EventRate"].dtype == np.float64
    assert scorecard["LeafValue"].dtype == np.float64
    assert scorecard["WOE"].dtype == np.float64
    assert scorecard["IV"].dtype == np.float64
    assert scorecard["xAddEvidence"].dtype == np.float64
    assert scorecard["CountPct"].dtype == np.float64
    
    # Check for valid values
    assert scorecard["Count"].min() >= 0
    assert scorecard["NonEvents"].min() >= 0
    assert scorecard["Events"].min() >= 0
    assert scorecard["EventRate"].min() >= 0
    assert scorecard["EventRate"].max() <= 1
    assert scorecard["CountPct"].min() >= 0
    assert scorecard["CountPct"].max() <= 100


def test_extract_leaf_weights(trained_model: CatBoostClassifier):
    """Test the extract_leaf_weights method."""
    leaf_weights = CatBoostScorecard.extract_leaf_weights(trained_model)

    # Check basic structure
    assert isinstance(leaf_weights, pd.DataFrame)
    assert not leaf_weights.empty

    # Check required columns
    required_columns = {"Tree", "Node", "XAddEvidence"}
    assert set(leaf_weights.columns) == required_columns

    # Check data types
    assert leaf_weights["Tree"].dtype == np.int64
    assert leaf_weights["Node"].dtype == np.int64
    assert leaf_weights["XAddEvidence"].dtype == np.float64

    # Check for valid values (allow decimal values)
    assert leaf_weights["XAddEvidence"].between(-1, 1).all()


def test_parse_condition():  # pylint: disable=protected-access
    """Test the _parse_condition method."""
    parsed = _parse_conditions("feature_1, bin=0.5", "feature_1", "0.5")
    parsed = _parse_conditions("feature_2, value='category'", "feature_2", "category")
    parsed = CatBoostScorecard._parse_condition("3", "1", ["f0", "f1", "f2", "f3"])
    assert "f3" in parsed


def _parse_conditions(arg0, arg1, arg2):
    result = CatBoostScorecard._parse_condition(arg0, "1")
    assert arg1 in result
    assert arg2 in result

    return result


def test_get_leaf_conditions(trained_model: CatBoostClassifier, test_pool: Pool):
    """Test the _get_leaf_conditions method."""
    cb_obj = trained_model._object
    tree_idx = 0
    leaf_conditions = CatBoostScorecard._get_leaf_conditions(cb_obj, test_pool, tree_idx)

    # Check basic structure
    assert isinstance(leaf_conditions, dict)
    assert len(leaf_conditions) > 0
    assert all(isinstance(k, int) for k in leaf_conditions.keys())
    assert all(isinstance(v, str) for v in leaf_conditions.values())
    assert all("AND" in v or v == "" for v in leaf_conditions.values())


def test_is_numeric_only_condition():
    """Test the _is_numeric_only_condition method."""
    assert CatBoostScorecard._is_numeric_only_condition("4")
    assert CatBoostScorecard._is_numeric_only_condition("42")
    assert not CatBoostScorecard._is_numeric_only_condition("feature_1")
    assert not CatBoostScorecard._is_numeric_only_condition("feature_1 <= 0.5")


def test_get_split_feature_value():
    """Test the _get_split_feature_value method."""
    # Test numeric split
    feature, value, split_type = CatBoostScorecard._get_split_feature_value("feature_1, bin=0.5")
    assert feature == "feature_1"
    assert value == "0.5"
    assert split_type == "numerical"

    # Test categorical split
    feature, value, split_type = CatBoostScorecard._get_split_feature_value(
        "feature_2, value='category'"
    )
    assert feature == "feature_2"
    assert value == "'category'"
    assert split_type == "categorical_value"

    # Test index-based split
    feature, value, split_type = CatBoostScorecard._get_split_feature_value("3")
    assert feature == "Feature_3"
    assert value == 3
    assert split_type == "index_based"


def test_extract_feature_names(test_pool: Pool):
    """Test the _extract_feature_names method."""
    feature_names = CatBoostScorecard._extract_feature_names(test_pool)
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert all(isinstance(name, str) for name in feature_names)
