"""
Regression tests for XGBScorecardConstructor

These tests verify that critical functionality maintains consistent behavior
across code changes, particularly for methods like get_leafs() and construct_scorecard().

Originally created for PR #6 refactoring validation (v0.2.7).

NOTE: These tests establish a baseline for expected behavior. If future changes
intentionally modify the behavior (e.g., changing precision, output format),
update these tests accordingly with clear documentation of why the change is expected.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from xbooster.xgb_constructor import XGBScorecardConstructor

# Regression test configuration
# Update this when intentional breaking changes are made
REGRESSION_VERSION = "0.2.7"
EXPECTED_PRECISION_TOLERANCE = 1e-6  # Allow for float precision differences


@pytest.fixture
def sample_data():
    """Create consistent sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        }
    )
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained XGBoost model."""
    X, y = sample_data
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model


class TestXGBRegression:
    """Regression test suite for XGBScorecardConstructor to prevent regressions."""

    def test_get_leafs_leaf_index_output(self, sample_data, trained_model):
        """Test that get_leafs with leaf_index returns correct shape and values."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Get leaf indices
        leaf_indices = constructor.get_leafs(X, output_type="leaf_index")

        # Verify structure
        assert isinstance(leaf_indices, pd.DataFrame)
        assert leaf_indices.shape[0] == len(X)
        assert leaf_indices.shape[1] == 10  # n_estimators

        # Verify column names
        expected_cols = [f"tree_{i}" for i in range(10)]
        assert list(leaf_indices.columns) == expected_cols

        # Verify values are numeric (leaf indices stored as float but represent integers)
        assert leaf_indices.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        # Verify they are whole numbers
        assert (leaf_indices % 1 == 0).all().all()

    def test_get_leafs_margin_output(self, sample_data, trained_model):
        """Test that get_leafs with margin returns correct shape and values."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Get margins (default behavior)
        margins = constructor.get_leafs(X, output_type="margin")

        # Verify structure
        assert isinstance(margins, pd.DataFrame)
        assert margins.shape[0] == len(X)
        assert margins.shape[1] == 10  # n_estimators

        # Verify column names
        expected_cols = [f"tree_{i}" for i in range(10)]
        assert list(margins.columns) == expected_cols

        # Verify values are floats (margins)
        assert margins.dtypes.apply(lambda x: np.issubdtype(x, np.floating)).all()

    def test_construct_scorecard_output_structure(self, sample_data, trained_model):
        """Test that construct_scorecard returns correct structure."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        scorecard = constructor.construct_scorecard()

        # Verify it's a DataFrame
        assert isinstance(scorecard, pd.DataFrame)
        assert not scorecard.empty

        # Verify expected columns exist
        expected_columns = [
            "Tree",
            "Node",
            "Feature",
            "Sign",
            "Split",
            "Count",
            "CountPct",
            "NonEvents",
            "Events",
            "EventRate",
            "WOE",
            "IV",
            "XAddEvidence",
            "DetailedSplit",
        ]
        assert all(col in scorecard.columns for col in expected_columns)

        # Verify data types
        assert pd.api.types.is_integer_dtype(scorecard["Tree"])
        assert pd.api.types.is_integer_dtype(scorecard["Node"])
        assert pd.api.types.is_integer_dtype(scorecard["Count"])
        assert pd.api.types.is_float_dtype(scorecard["EventRate"])

    def test_construct_scorecard_statistical_properties(self, sample_data, trained_model):
        """Test that construct_scorecard maintains correct statistical properties."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        scorecard = constructor.construct_scorecard()

        # Count and Events should sum correctly per tree
        # sourcery skip: no-loop-in-tests
        for tree_id in scorecard["Tree"].unique():
            tree_data = scorecard[scorecard["Tree"] == tree_id]

            # Total count should equal sample size
            total_count = tree_data["Count"].sum()
            assert abs(total_count - len(X)) < 1e-6, f"Tree {tree_id} count mismatch"

            # Events + NonEvents = Count
            for _, row in tree_data.iterrows():
                assert abs(row["Events"] + row["NonEvents"] - row["Count"]) < 1e-6

            # EventRate should be Events / Count
            for _, row in tree_data.iterrows():
                expected_rate = row["Events"] / row["Count"]
                assert abs(row["EventRate"] - expected_rate) < 1e-6

    def test_prediction_consistency(self, sample_data, trained_model):
        """Test that predictions are consistent between methods."""
        X, y = sample_data
        X_test = X.iloc[:20]  # Use first 20 rows for testing

        constructor = XGBScorecardConstructor(trained_model, X, y)
        constructor.construct_scorecard()
        constructor.create_points()

        # Get predictions
        scores = constructor.predict_score(X_test)
        detailed_scores = constructor.predict_scores(X_test)

        # Verify they match
        assert len(scores) == len(X_test)
        assert len(detailed_scores) == len(X_test)
        assert np.allclose(scores.values, detailed_scores["Score"].values)

    def test_leaf_indices_match_scorecard(self, sample_data, trained_model):
        """Test that leaf indices from get_leafs match scorecard construction."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Get leaf indices
        leaf_indices = constructor.get_leafs(X, output_type="leaf_index")

        # Construct scorecard
        scorecard = constructor.construct_scorecard()

        # Verify that all leaf nodes in scorecard exist in leaf_indices
        # sourcery skip: no-loop-in-tests
        for tree_id in scorecard["Tree"].unique():
            tree_nodes = scorecard[scorecard["Tree"] == tree_id]["Node"].unique()
            leaf_col = f"tree_{tree_id}"
            observed_nodes = leaf_indices[leaf_col].unique()

            # All nodes in scorecard should appear in leaf indices
            assert set(tree_nodes).issubset(set(observed_nodes))

    def test_points_calculation_consistency(self, sample_data, trained_model):
        """Test that points calculation produces valid results."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        constructor.construct_scorecard()
        points_card = constructor.create_points(
            pdo=50, target_points=600, target_odds=19, precision_points=0
        )

        # Verify Points column exists and is numeric
        assert "Points" in points_card.columns
        assert pd.api.types.is_numeric_dtype(points_card["Points"])

        # Verify no NaN or infinite values
        assert not points_card["Points"].isna().any()
        assert not np.isinf(points_card["Points"]).any()

        # Verify points are integers (precision_points=0)
        assert points_card["Points"].dtype in [np.int64, np.int32, np.float64]

    def test_margins_sum_to_model_prediction(self, sample_data, trained_model):
        """Test that margins from get_leafs sum to model predictions."""
        X, y = sample_data
        X_test = X.iloc[:20]

        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Get margins from get_leafs
        margins = constructor.get_leafs(X_test, output_type="margin")
        margin_sums = margins.sum(axis=1)

        # Get predictions from model (output_margin=True)
        dmatrix = xgb.DMatrix(X_test, base_margin=np.full(len(X_test), constructor.base_score))
        model_margins = trained_model.get_booster().predict(dmatrix, output_margin=True)
        model_margins_adjusted = model_margins - constructor.base_score

        # They should match closely (allowing for float32 vs float64 precision)
        assert np.allclose(margin_sums.values, model_margins_adjusted, rtol=1e-4, atol=1e-6)

    def test_woe_and_iv_calculations(self, sample_data, trained_model):
        """Test that WOE and IV are calculated correctly."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        scorecard = constructor.construct_scorecard()

        # WOE should be finite for valid event rates
        valid_woe = scorecard[scorecard["Count"] > 0]["WOE"]
        assert np.isfinite(valid_woe).all()

        # IV should be non-negative
        assert (scorecard["IV"] >= 0).all() or scorecard["IV"].isna().all()

    def test_extract_leaf_weights_consistency(self, sample_data, trained_model):
        """Test that extract_leaf_weights produces consistent results."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        leaf_weights = constructor.extract_leaf_weights()

        # Verify structure
        assert isinstance(leaf_weights, pd.DataFrame)
        assert not leaf_weights.empty

        expected_cols = {"Tree", "Node", "Feature", "Sign", "Split", "XAddEvidence"}
        assert expected_cols.issubset(set(leaf_weights.columns))

        # Verify each tree has entries
        trees = leaf_weights["Tree"].unique()
        assert len(trees) == 10  # n_estimators

        # Verify XAddEvidence values are finite
        assert np.isfinite(leaf_weights["XAddEvidence"]).all()

    def test_deterministic_results(self, sample_data, trained_model):
        """Test that running the same operations twice produces identical results."""
        X, y = sample_data

        # First run
        constructor1 = XGBScorecardConstructor(trained_model, X, y)
        scorecard1 = constructor1.construct_scorecard()

        # Second run
        constructor2 = XGBScorecardConstructor(trained_model, X, y)
        scorecard2 = constructor2.construct_scorecard()

        # Should be identical
        pd.testing.assert_frame_equal(scorecard1, scorecard2)

    def test_empty_prediction_handling(self, sample_data, trained_model):
        """Test that predictions work with single row."""
        X, y = sample_data
        X_single = X.iloc[[0]]  # Single row

        constructor = XGBScorecardConstructor(trained_model, X, y)
        constructor.construct_scorecard()
        constructor.create_points()

        # Should work with single row
        score = constructor.predict_score(X_single)
        assert len(score) == 1
        assert np.isfinite(score.iloc[0])

    def test_large_dataset_performance(self, trained_model):
        """Test that methods work efficiently with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        X_large = pd.DataFrame(
            {
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "feature3": np.random.randn(1000),
            }
        )
        y_large = np.random.randint(0, 2, 1000)

        constructor = XGBScorecardConstructor(trained_model, X_large, y_large)

        # Should complete without errors
        scorecard = constructor.construct_scorecard()
        assert not scorecard.empty

        leaf_indices = constructor.get_leafs(X_large, output_type="leaf_index")
        assert leaf_indices.shape[0] == 1000
