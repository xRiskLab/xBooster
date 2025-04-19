"""
Test module for xbooster.catboost_wrapper.

This module provides test cases for the CatBoostWOEMapper class, which is responsible
for mapping features to WOE (Weight of Evidence) values and calculating scores.
"""

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import make_classification

from xbooster.catboost_wrapper import CatBoostWOEMapper


@pytest.fixture(scope="module")
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, y


@pytest.fixture(scope="module")
def trained_model(sample_data):
    """Create a trained CatBoost model."""
    X, y = sample_data
    model = CatBoostClassifier(iterations=10, depth=3, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def catboost_pool(sample_data):
    """Create a CatBoost Pool."""
    X, y = sample_data
    return Pool(X, y)


@pytest.fixture(scope="module")
def scorecard_df(trained_model, catboost_pool):
    """Create a scorecard DataFrame."""
    from xbooster.catboost_scorecard import CatBoostScorecard

    return CatBoostScorecard.trees_to_scorecard(trained_model, catboost_pool)


@pytest.fixture(scope="module")
def woe_mapper(scorecard_df):
    """Create a WOE mapper instance."""
    return CatBoostWOEMapper(scorecard_df)


def test_initialization(scorecard_df):
    """Test initialization of CatBoostWOEMapper."""
    mapper = CatBoostWOEMapper(scorecard_df)
    assert hasattr(mapper, "scorecard")
    assert mapper.use_woe is False
    assert mapper.points_column is None
    assert hasattr(mapper, "feature_mappings")
    assert hasattr(mapper, "feature_importance")


def test_generate_feature_mappings(woe_mapper):
    """Test the generate_feature_mappings method."""
    woe_mapper.generate_feature_mappings()
    mappings = woe_mapper.feature_mappings

    # Check basic structure
    assert isinstance(mappings, dict)
    assert len(mappings) > 0

    # Check mapping format
    for feature, conditions in mappings.items():
        assert isinstance(feature, str)
        for condition, details in conditions.items():
            assert isinstance(condition, str)
            assert isinstance(details, dict)
            assert "value" in details
            assert "weight" in details
            assert "trees" in details
            assert "agg_value" in details
            assert "total_weight" in details
            assert "tree_count" in details
            assert isinstance(details["value"], list)
            assert isinstance(details["weight"], list)
            assert isinstance(details["trees"], list)
            assert isinstance(details["agg_value"], float)
            assert isinstance(details["total_weight"], (int, float))
            assert isinstance(details["tree_count"], int)


def test_calculate_feature_importance(woe_mapper):
    """Test the calculate_feature_importance method."""
    woe_mapper.calculate_feature_importance()
    importance = woe_mapper.feature_importance

    assert isinstance(importance, dict)
    assert len(importance) > 0

    assert all(isinstance(k, str) for k in importance.keys())
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0 for v in importance.values())


def test_predict_score(woe_mapper, sample_data):
    """Test the predict_score method."""
    X, _ = sample_data

    scores = woe_mapper.predict_score(X)
    assert isinstance(scores, (np.ndarray, pd.Series))
    assert len(scores) == len(X)
    assert not np.isnan(scores).any()

    sample_dict = X.iloc[0].to_dict()
    score_dict = woe_mapper.predict_score(sample_dict)
    assert isinstance(score_dict, float)
    assert not np.isnan(score_dict)


def test_transform_dataset(woe_mapper, sample_data):
    """Test the transform_dataset method."""
    X, _ = sample_data

    transformed_df = woe_mapper.transform_dataset(X)
    assert isinstance(transformed_df, pd.DataFrame)
    assert len(transformed_df) == len(X)
    assert not transformed_df.isna().any().any()

    sample_dict = X.iloc[0].to_dict()
    transformed_dict = woe_mapper.transform_dataset(pd.DataFrame([sample_dict]))
    assert isinstance(transformed_dict, pd.DataFrame)
    assert len(transformed_dict) == 1
    assert not transformed_dict.isna().any().any()


def test_plot_feature_importance(woe_mapper):
    """Test the plot_feature_importance method."""
    woe_mapper.plot_feature_importance()

    woe_mapper.plot_feature_importance(figsize=(12, 8), top_n=3)


def test_create_scorecard(woe_mapper):
    """Test the create_scorecard method."""
    scorecard = woe_mapper.create_scorecard()
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty
    assert "Points" in scorecard.columns

    pdo_params = {"pdo": 20, "target_points": 600, "target_odds": 1, "precision_points": 0}
    scorecard_pdo = woe_mapper.create_scorecard(pdo_params=pdo_params)
    assert isinstance(scorecard_pdo, pd.DataFrame)
    assert not scorecard_pdo.empty
    assert "Points" in scorecard_pdo.columns


def test_get_binned_feature_table(woe_mapper):
    """Test the get_binned_feature_table method."""
    table = woe_mapper.get_binned_feature_table()

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    required_columns = {"Feature", "Condition", "LeafValue", "Weight", "TreeCount"}
    assert set(table.columns) >= required_columns

    assert table["Feature"].dtype == object
    assert table["Condition"].dtype == object
    assert table["LeafValue"].dtype == np.float64
    assert table["Weight"].dtype == np.float64
    assert table["TreeCount"].dtype == np.int64


def test_get_value_column(woe_mapper):
    """Test the get_value_column method."""
    assert woe_mapper.get_value_column() == "LeafValue"

    woe_mapper.points_column = "Points"
    assert woe_mapper.get_value_column() == "Points"

    woe_mapper.points_column = None
