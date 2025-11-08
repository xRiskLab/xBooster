"""
Tests for LGBScorecardConstructor (alpha - reference implementation).

These tests verify the basic structure and instantiation of the LightGBM
constructor class. Full functionality tests will be added once methods
are implemented.
"""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from xbooster.lgb_constructor import LGBScorecardConstructor


@pytest.fixture
def sample_data():
    """Create sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y


@pytest.fixture
def trained_lgb_model(sample_data):
    """Create a trained LightGBM model."""
    X, y = sample_data
    model = LGBMClassifier(n_estimators=5, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)
    return model


def test_lgb_constructor_import():
    """Test that LGBScorecardConstructor can be imported."""
    assert LGBScorecardConstructor is not None


def test_lgb_constructor_instantiation(trained_lgb_model, sample_data):
    """Test that LGBScorecardConstructor can be instantiated."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)
    assert constructor is not None
    assert hasattr(constructor, "model")
    assert hasattr(constructor, "X")
    assert hasattr(constructor, "y")


def test_lgb_constructor_has_required_methods(trained_lgb_model, sample_data):
    """Test that LGBScorecardConstructor has all required methods."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)

    # Check for required methods
    assert hasattr(constructor, "extract_leaf_weights")
    assert hasattr(constructor, "construct_scorecard")
    assert hasattr(constructor, "create_points")
    assert hasattr(constructor, "predict_score")
    assert callable(constructor.extract_leaf_weights)
    assert callable(constructor.construct_scorecard)
    assert callable(constructor.create_points)
    assert callable(constructor.predict_score)


def test_lgb_constructor_methods_raise_not_implemented(trained_lgb_model, sample_data):
    """Test that stub methods raise NotImplementedError."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)

    # These methods are now implemented and should NOT raise
    try:
        leaf_weights = constructor.extract_leaf_weights()
        assert len(leaf_weights) > 0
        assert "XAddEvidence" in leaf_weights.columns
    except NotImplementedError:
        pytest.fail("extract_leaf_weights should be implemented")

    try:
        leaf_indices = constructor.get_leafs(X, output_type="leaf_index")
        assert leaf_indices.shape[0] == len(X)
    except NotImplementedError:
        pytest.fail("get_leafs should be implemented")

    # These methods are still stubs
    with pytest.raises(NotImplementedError):
        constructor.construct_scorecard()

    with pytest.raises(NotImplementedError):
        constructor.create_points()

    with pytest.raises(NotImplementedError):
        constructor.predict_score(X)


def test_lgb_constructor_model_type(trained_lgb_model, sample_data):
    """Test that constructor accepts LGBMClassifier."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)
    assert isinstance(constructor.model, LGBMClassifier)


def test_lgb_constructor_data_types(trained_lgb_model, sample_data):
    """Test that constructor properly stores data types."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)
    assert isinstance(constructor.X, pd.DataFrame)
    assert isinstance(constructor.y, pd.Series)


def test_extract_leaf_weights(trained_lgb_model, sample_data):
    """Test that extract_leaf_weights returns correct structure."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)

    leaf_weights = constructor.extract_leaf_weights()

    # Check DataFrame structure
    assert isinstance(leaf_weights, pd.DataFrame)
    expected_columns = ["Tree", "Node", "Feature", "Sign", "Split", "XAddEvidence"]
    assert list(leaf_weights.columns) == expected_columns

    # Check data types
    assert leaf_weights["Tree"].dtype in [np.int64, np.int32]
    assert leaf_weights["Feature"].dtype == object
    assert leaf_weights["Sign"].dtype == object
    assert leaf_weights["Split"].dtype in [np.float64, np.float32]
    assert leaf_weights["XAddEvidence"].dtype in [np.float64, np.float32]

    # Check Sign values are valid
    assert set(leaf_weights["Sign"].unique()).issubset({"<", ">="})

    # Check we have data
    assert len(leaf_weights) > 0


def test_get_leafs_leaf_index(trained_lgb_model, sample_data):
    """Test that get_leafs returns leaf indices correctly."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)

    leaf_indices = constructor.get_leafs(X, output_type="leaf_index")

    # Check shape
    assert leaf_indices.shape[0] == len(X)
    assert leaf_indices.shape[1] == trained_lgb_model.n_estimators

    # Check column names
    expected_cols = [f"tree_{i}" for i in range(trained_lgb_model.n_estimators)]
    assert list(leaf_indices.columns) == expected_cols


def test_get_leafs_margin(trained_lgb_model, sample_data):
    """Test that get_leafs returns margins correctly."""
    X, y = sample_data
    constructor = LGBScorecardConstructor(trained_lgb_model, X, y)

    margins = constructor.get_leafs(X, output_type="margin")

    # Check shape
    assert margins.shape[0] == len(X)
    assert margins.shape[1] == trained_lgb_model.n_estimators

    # Check that margins sum to raw predictions (accounting for base score)
    raw_pred = trained_lgb_model.predict(X, raw_score=True)
    margin_sum = margins.sum(axis=1).values + constructor.base_score

    assert np.allclose(margin_sum, raw_pred, rtol=1e-5)
