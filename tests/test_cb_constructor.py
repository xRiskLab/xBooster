"""
Test module for xbooster.cb_constructor.

The CatBoostScorecardConstructor class is responsible for constructing a scorecard from
a trained CatBoost model. This module provides test cases to ensure that the class
functions correctly.

Test cases:
- test_extract_leaf_weights: Tests the extract_leaf_weights method to verify the
  returned DataFrame structure.
- test_construct_scorecard: Tests the construct_scorecard method to ensure it returns
  a non-empty pandas DataFrame.
- test_create_points: Tests the create_points method to verify it returns a non-empty
  pandas DataFrame.
- test_predict_score: Tests the predict_score method to ensure it returns a non-empty
  DataFrame with predicted scores.
- test_predict_scores: Tests the predict_scores method to ensure it returns a non-empty
  DataFrame with predicted scores.
- test_woe_mapper: Tests the WOE mapper functionality and Gini score comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import os

from xbooster.constructor import CatBoostScorecardConstructor


@pytest.fixture(scope="module")
def cb_model():
    """
    Creates and trains a CatBoost model.

    Returns:
        model (CatBoostClassifier): Trained CatBoost model.
    """
    X = pd.DataFrame(
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
    model = CatBoostClassifier(iterations=10, depth=3, learning_rate=0.1, verbose=False)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def scorecard_constructor(cb_model):
    """
    Constructs a CatBoostScorecardConstructor object using the given CatBoost model,
    feature matrix (X), and target variable (y).

    Parameters:
    - cb_model: The trained CatBoost model.

    Returns:
    - CatBoostScorecardConstructor: The initialized CatBoostScorecardConstructor object.
    """
    X = pd.DataFrame(
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
    pool = Pool(X, y)
    return CatBoostScorecardConstructor(cb_model, pool)


@pytest.fixture(scope="module")
def X_test():
    """
    This function returns a sample input data frame.

    Returns:
        pd.DataFrame: A data frame with two features, feature1 and feature2.
    """
    return pd.DataFrame({"feature1": [55, 42, 30], "feature2": [35, 28, 18]})


@pytest.fixture(scope="module")
def credit_data():
    """
    Loads the credit data for testing WOE mapper and Gini score comparisons.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable.
    """
    credit_data = pd.read_csv("examples/data/test_data_01d9ab8b.csv")
    num_features = ["Gross_Annual_Income", "Time_with_Bank"]
    features = num_features

    X = credit_data[features]
    y = credit_data["Final_Decision"].replace({"Accept": 1, "Decline": 0})
    return X, y


@pytest.fixture(scope="module")
def credit_model(credit_data):
    """
    Creates and trains a CatBoost model on the credit data.

    Parameters:
        credit_data: Tuple of (X, y) from credit_data fixture

    Returns:
        tuple: (model, pool) where model is the trained CatBoost model and pool is the training pool
    """
    X, y = credit_data

    pool = Pool(
        data=X,
        label=y,
    )

    model = CatBoostClassifier(
        iterations=100,
        allow_writing_files=False,
        depth=1,
        learning_rate=0.1,
        verbose=0,
    )
    model.fit(pool)
    return model, pool


def test_extract_leaf_weights(scorecard_constructor):
    """
    Test the extract_leaf_weights method.

    This test verifies that the extract_leaf_weights method returns a DataFrame
    with the correct structure and data types.
    """
    leaf_weights = scorecard_constructor.extract_leaf_weights()

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


def test_construct_scorecard(scorecard_constructor):
    """
    Test the construct_scorecard method.

    This test verifies that the construct_scorecard method returns a DataFrame
    with the correct structure and data types.
    """
    scorecard = scorecard_constructor.construct_scorecard()

    # Check basic structure
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty

    # Check required columns (updated to match actual output)
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
        "WOE",
        "IV",
        "CountPct",
        "DetailedSplit",
        "LeafValue",
    }
    assert set(scorecard.columns).issuperset(required_columns)

    # Check data types
    assert scorecard["Tree"].dtype == np.int64
    assert scorecard["LeafIndex"].dtype == np.int64
    assert scorecard["Feature"].dtype == object
    assert scorecard["Sign"].dtype == object
    assert scorecard["Split"].dtype == object  # Can be float or string
    assert scorecard["Count"].dtype == np.float64
    assert scorecard["NonEvents"].dtype == np.float64
    assert scorecard["Events"].dtype == np.float64
    assert scorecard["EventRate"].dtype == np.float64
    assert scorecard["WOE"].dtype == np.float64
    assert scorecard["IV"].dtype == np.float64
    assert scorecard["CountPct"].dtype == np.float64
    assert scorecard["DetailedSplit"].dtype == object
    assert scorecard["LeafValue"].dtype == np.float64

    # Check for valid values
    assert scorecard["Count"].min() >= 0
    assert scorecard["NonEvents"].min() >= 0
    assert scorecard["Events"].min() >= 0
    assert scorecard["EventRate"].min() >= 0
    assert scorecard["EventRate"].max() <= 1
    assert not scorecard["WOE"].isnull().any()
    assert not scorecard["IV"].isnull().any()
    assert scorecard["CountPct"].min() >= 0
    assert scorecard["CountPct"].max() <= 100


def test_create_points(scorecard_constructor):
    """
    Test the create_points method of the CatBoostScorecardConstructor class.

    This test verifies that the create_points method returns a non-empty
    pandas DataFrame with the expected columns.
    """
    # First construct the scorecard
    scorecard_constructor.construct_scorecard()

    # Then create points
    points_card = scorecard_constructor.create_points()
    assert isinstance(points_card, pd.DataFrame)
    assert not points_card.empty
    assert "Points" in points_card.columns


def test_predict_score(scorecard_constructor, X_test):
    """
    Test the predict_score method of the CatBoostScorecardConstructor class.

    This test verifies that the predict_score method returns a non-empty
    pandas Series with predicted scores.
    """
    # First construct the scorecard and create points
    scorecard_constructor.construct_scorecard()
    scorecard_constructor.create_points()

    # Then predict scores
    scores = scorecard_constructor.predict_score(X_test)
    assert isinstance(scores, pd.Series)
    assert not scores.empty
    assert len(scores) == len(X_test)


def test_predict_scores(scorecard_constructor, X_test):
    """
    Test the predict_scores method of the CatBoostScorecardConstructor class.

    This test verifies that the predict_scores method returns a non-empty
    pandas DataFrame with predicted scores per tree and total score.
    """
    # First construct the scorecard and create points
    scorecard_constructor.construct_scorecard()
    scorecard_constructor.create_points()

    # Then predict scores
    scores = scorecard_constructor.predict_scores(X_test)
    assert isinstance(scores, pd.DataFrame)
    assert not scores.empty
    assert len(scores) == len(X_test)
    assert "Score" in scores.columns

def test_woe_mapper_and_gini_scores(credit_data, credit_model):
    """
    Test the WOE mapper functionality and Gini score comparisons.

    This test verifies that:
    1. The leaf scores match CatBoost raw predictions in terms of Gini score
    2. The WOE scores match the points scores in terms of Gini score
    """
    X, y = credit_data
    model, pool = credit_model

    # Create scorecard constructor
    constructor = CatBoostScorecardConstructor(model, pool)

    # Construct scorecard and create points
    constructor.construct_scorecard()
    constructor.create_points(pdo=50, target_points=600, target_odds=19, precision_points=0)

    # Get predictions from different methods
    leaf_scores = constructor.predict_score(X)
    woe_scores = constructor.predict_scores(X)["Score"]
    points_scores = constructor.predict_score(X)

    # Get CatBoost raw predictions
    cb_preds = model.predict_proba(X)[:, 1]

    # Calculate Gini scores
    cb_gini = 2 * roc_auc_score(y, cb_preds) - 1
    leaf_gini = 2 * roc_auc_score(y, leaf_scores) - 1
    woe_gini = 2 * roc_auc_score(y, woe_scores) - 1
    points_gini = 2 * roc_auc_score(y, points_scores) - 1

    # Verify that leaf scores match CatBoost predictions
    assert abs(cb_gini - leaf_gini) < 0.01, "Leaf scores Gini should match CatBoost Gini"

    # Verify that WOE scores match points scores
    assert abs(woe_gini - points_gini) < 0.01, "WOE scores Gini should match points Gini"


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained CatBoost model."""
    X, y = sample_data
    model = CatBoostClassifier(iterations=10, depth=3, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture
def catboost_pool(sample_data):
    """Create a CatBoost Pool."""
    X, y = sample_data
    return Pool(X, y)


def test_initialization():
    """Test initialization of CatBoostScorecardConstructor."""
    constructor = CatBoostScorecardConstructor()
    assert constructor.model is None
    assert constructor.pool is None
    assert constructor.use_woe is False
    assert constructor.points_column is None
    assert constructor.scorecard_df is None
    assert constructor.mapper is None


def test_fit(trained_model, catboost_pool):
    """Test fitting the constructor."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    assert constructor.model is trained_model
    assert constructor.pool is catboost_pool
    assert constructor.scorecard_df is not None
    assert constructor.mapper is not None


def test_get_scorecard(trained_model, catboost_pool):
    """Test getting the scorecard."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    scorecard = constructor.get_scorecard()
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty
    assert "Tree" in scorecard.columns
    assert "LeafIndex" in scorecard.columns
    assert "LeafValue" in scorecard.columns
    assert "DetailedSplit" in scorecard.columns


def test_get_feature_importance(trained_model, catboost_pool):
    """Test getting feature importance."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    importance = constructor.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    assert all(isinstance(k, str) for k in importance.keys())
    assert all(isinstance(v, float) for v in importance.values())


def test_predict(trained_model, catboost_pool, sample_data):
    """Test making predictions."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    X, _ = sample_data
    # Test with DataFrame
    predictions_df = constructor.predict(X)
    assert isinstance(predictions_df, np.ndarray)
    assert len(predictions_df) == len(X)

    # Test with dictionary
    sample_dict = X.iloc[0].to_dict()
    prediction_dict = constructor.predict(sample_dict)
    assert isinstance(prediction_dict, float)


def test_transform(trained_model, catboost_pool, sample_data):
    """Test transforming features."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    X, _ = sample_data
    # Test with DataFrame
    transformed_df = constructor.transform(X)
    assert isinstance(transformed_df, pd.DataFrame)
    assert not transformed_df.empty
    assert len(transformed_df) == len(X)

    # Test with dictionary
    sample_dict = X.iloc[0].to_dict()
    transformed_dict = constructor.transform(sample_dict)
    assert isinstance(transformed_dict, pd.DataFrame)
    assert len(transformed_dict) == 1


def test_create_scorecard(trained_model, catboost_pool):
    """Test creating a detailed scorecard."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    # Test without PDO parameters
    scorecard = constructor.create_scorecard()
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty

    # Test with PDO parameters
    pdo_params = {"pdo": 20, "odds": 1, "score": 600}
    scorecard_pdo = constructor.create_scorecard(pdo_params=pdo_params)
    assert isinstance(scorecard_pdo, pd.DataFrame)
    assert not scorecard_pdo.empty
    assert "Points" in scorecard_pdo.columns


def test_get_binned_feature_table(trained_model, catboost_pool):
    """Test getting binned feature table."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    binned_table = constructor.get_binned_feature_table()
    assert isinstance(binned_table, pd.DataFrame)
    assert not binned_table.empty
    assert "Feature" in binned_table.columns
    assert "Bin" in binned_table.columns
    assert "Value" in binned_table.columns


def test_plot_feature_importance(trained_model, catboost_pool):
    """Test plotting feature importance."""
    constructor = CatBoostScorecardConstructor()
    constructor.fit(trained_model, catboost_pool)

    # Test with default parameters
    constructor.plot_feature_importance()

    # Test with custom parameters
    constructor.plot_feature_importance(figsize=(12, 8), top_n=3)


def test_error_handling():
    """Test error handling for uninitialized constructor."""
    constructor = CatBoostScorecardConstructor()

    with pytest.raises(ValueError, match="Scorecard not built yet"):
        constructor.get_scorecard()

    with pytest.raises(ValueError, match="Mapper not initialized"):
        constructor.get_feature_importance()

    with pytest.raises(ValueError, match="Mapper not initialized"):
        constructor.predict({"feature_0": 0.5})

    with pytest.raises(ValueError, match="Mapper not initialized"):
        constructor.transform({"feature_0": 0.5})

    with pytest.raises(ValueError, match="Mapper not initialized"):
        constructor.plot_feature_importance()

    with pytest.raises(ValueError, match="Mapper not initialized"):
        constructor.create_scorecard()

    with pytest.raises(ValueError, match="Mapper not initialized"):
        constructor.get_binned_feature_table()
        constructor.get_binned_feature_table()
        constructor.get_binned_feature_table()


def test_feature_importance_and_multiple_create_points(credit_data, credit_model):
    """
    Test the feature importance calculation with the updated approach and
    verify that create_points can be called multiple times without errors.
    
    This test:
    1. Constructs a scorecard
    2. Checks feature importance calculation
    3. Creates points once
    4. Verifies different prediction methods work
    5. Creates points a second time (which should not fail)
    """
    X, y = credit_data
    model, pool = credit_model

    # Create and fit the scorecard constructor
    constructor = CatBoostScorecardConstructor(model, pool)
    
    # Construct scorecard
    scorecard = constructor.construct_scorecard()
    assert isinstance(scorecard, pd.DataFrame)
    assert not scorecard.empty
    assert "WOE" in scorecard.columns
    
    # Get feature importance before creating points
    feature_importance_before = constructor.get_feature_importance()
    assert isinstance(feature_importance_before, dict)
    assert len(feature_importance_before) > 0
    
    # Verify feature importance calculation uses WOE values
    # The importance should be a normalized value between 0 and 1
    assert all(0 <= v <= 1 for v in feature_importance_before.values())
    assert abs(sum(feature_importance_before.values()) - 1.0) < 1e-10
    
    # Get predictions before creating points
    raw_scores_before = constructor.predict_score(X, method="raw")
    woe_scores_before = constructor.predict_score(X, method="woe")
    
    # Get CatBoost raw predictions to compare
    cb_preds = model.predict(X, prediction_type="RawFormulaVal")
    
    # Verify that raw scores match CatBoost predictions
    np.testing.assert_allclose(raw_scores_before, cb_preds, rtol=1e-2, atol=1e-2)
    
    # Create points for the first time
    points_scorecard = constructor.create_points(
        pdo=50, target_points=600, target_odds=19, precision_points=0
    )
    assert isinstance(points_scorecard, pd.DataFrame)
    assert "Points" in points_scorecard.columns
    
    # Get predictions after creating points
    points_scores = constructor.predict_score(X, method="pdo")
    raw_scores_after = constructor.predict_score(X, method="raw")
    woe_scores_after = constructor.predict_score(X, method="woe")
    
    # Verify predictions are consistent
    np.testing.assert_allclose(raw_scores_after, cb_preds, rtol=1e-2, atol=1e-2)
    
    # Get feature importance after creating points
    feature_importance_after = constructor.get_feature_importance()
    assert isinstance(feature_importance_after, dict)
    
    # Create points a second time - this should not fail
    try:
        points_scorecard2 = constructor.create_points(
            pdo=40, target_points=500, target_odds=10, precision_points=0
        )
        assert isinstance(points_scorecard2, pd.DataFrame)
        assert "Points" in points_scorecard2.columns
        
        # The points values should be different with the new parameters
        assert not np.array_equal(
            points_scorecard["Points"].values, 
            points_scorecard2["Points"].values
        )
        
        # Get predictions after second points creation
        points_scores2 = constructor.predict_score(X, method="pdo")
        raw_scores_after2 = constructor.predict_score(X, method="raw")
        
        # Raw scores should still match CatBoost predictions
        np.testing.assert_allclose(raw_scores_after2, cb_preds, rtol=1e-2, atol=1e-2)
        
    except Exception as e:
        pytest.fail(f"create_points failed when called a second time: {str(e)}")
    
    # Calculate Gini coefficients to verify that the predictive power is maintained
    cb_gini = 2 * roc_auc_score(y, cb_preds) - 1
    raw_gini = 2 * roc_auc_score(y, raw_scores_after) - 1
    
    # Ensure Gini coefficient is consistent
    assert abs(cb_gini - raw_gini) < 0.01, "Raw scores Gini should match CatBoost Gini"


def test_pdo_scoring_produces_varied_scores():
    """Test that PDO scoring doesn't produce all constant values."""
    # Load test data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples/data/test_data_01d9ab8b.csv")
    credit_data = pd.read_csv(data_path)
    
    # Prepare data
    num_features = ["Gross_Annual_Income", "Application_Score", "Bureau_Score"]
    categorical_features = ["Time_with_Bank"]
    features = num_features + categorical_features
    
    X = credit_data[features]
    y = credit_data["Final_Decision"].replace({"Accept": 1, "Decline": 0})
    
    # Create pool and model
    pool = Pool(
        data=X,
        label=y,
        cat_features=categorical_features,
    )
    
    model = CatBoostClassifier(
        iterations=100,
        allow_writing_files=False,
        depth=3,
        learning_rate=0.1,
        verbose=0,
        one_hot_max_size=9999,
    )
    model.fit(pool)
    
    # Create scorecard constructor and generate points
    constructor = CatBoostScorecardConstructor(model, pool)
    scorecard = constructor.construct_scorecard()
    
    # Create points
    constructor.create_points(pdo=50, target_points=600, target_odds=19)
    
    # Get PDO scores
    points_scores = constructor.predict_score(X, method="pdo")
    
    # Check that scores have variance (not all constant)
    score_std = np.std(points_scores)
    print(f"PDO scores standard deviation: {score_std}")
    
    # Assert that standard deviation is greater than a small threshold
    assert score_std > 1.0, f"PDO scores have insufficient variance (std={score_std})"
    
    # Check that there are at least two different score values
    unique_scores = np.unique(points_scores)
    assert len(unique_scores) > 1, f"All PDO scores are identical: {unique_scores[0]}"
