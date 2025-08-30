"""
Unit tests for interval scorecard functionality from PR #4.

This module tests the new methods:
- construct_scorecard_by_intervals()
- create_points_peo_pdo()
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.model_selection import train_test_split

from xbooster.constructor import XGBScorecardConstructor


class TestIntervalScorecard:
    """Test class for the new interval scorecard functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Use synthetic data for reproducible tests
        np.random.seed(42)
        n_samples = 1000

        dataset = pd.DataFrame(
            {
                "feature_1": np.random.normal(70, 10, n_samples),
                "feature_2": np.random.beta(2, 5, n_samples),
                "feature_3": np.random.beta(8, 2, n_samples),
                "target": np.random.binomial(1, 0.2, n_samples),
            }
        )

        features = ["feature_1", "feature_2", "feature_3"]
        target = "target"

        X, y = dataset[features], dataset[target]
        ix_train, ix_test = train_test_split(X.index, stratify=y, test_size=0.3, random_state=42)

        return X, y, ix_train, ix_test

    @pytest.fixture
    def trained_model_and_constructor_depth_1(self, sample_data):
        """Create a trained XGBoost model with max_depth=1 and constructor."""
        X, y, ix_train, ix_test = sample_data

        # Train XGBoost model with depth 1 (tree stumps)
        xgb_model = xgb.XGBClassifier(
            max_depth=1,  # Required for interval scorecard
            n_estimators=5,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X.loc[ix_train], y.loc[ix_train])

        # Create constructor
        constructor = XGBScorecardConstructor(xgb_model, X.loc[ix_train], y.loc[ix_train])

        return xgb_model, constructor, X, y, ix_train, ix_test

    @pytest.fixture
    def trained_model_and_constructor_depth_2(self, sample_data):
        """Create a trained XGBoost model with max_depth=2 for testing restrictions."""
        X, y, ix_train, ix_test = sample_data

        # Train XGBoost model with depth > 1
        xgb_model = xgb.XGBClassifier(
            max_depth=2, n_estimators=5, random_state=42, eval_metric="logloss"
        )
        xgb_model.fit(X.loc[ix_train], y.loc[ix_train])

        # Create constructor
        constructor = XGBScorecardConstructor(xgb_model, X.loc[ix_train], y.loc[ix_train])

        return xgb_model, constructor, X, y, ix_train, ix_test

    def test_interval_scorecard_requires_points(self, trained_model_and_constructor_depth_1):
        """Test that interval scorecard requires points to be calculated first."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        # Build standard scorecard but not points
        constructor.construct_scorecard()

        # Should raise ValueError if points not calculated
        with pytest.raises(
            ValueError, match="requires first having computed a scorecard with points"
        ):
            constructor.construct_scorecard_by_intervals()

    def test_interval_scorecard_requires_depth_one(self, trained_model_and_constructor_depth_2):
        # sourcery skip: class-extract-method
        """Test that interval scorecard requires max_depth=1."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_2

        constructor.construct_scorecard()
        constructor.create_points()

        # Should raise ValueError for depth > 1
        with pytest.raises(
            ValueError,
            match="can currently only be constructed for xgboost models with tree learners of depth one",
        ):
            constructor.construct_scorecard_by_intervals()

    def test_interval_scorecard_creation(self, trained_model_and_constructor_depth_1):
        """Test that interval scorecard is created successfully."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        # Build standard scorecard and points
        constructor.construct_scorecard()
        constructor.create_points()

        # Create interval scorecard
        interval_scorecard = constructor.construct_scorecard_by_intervals()

        # Verify structure
        assert isinstance(interval_scorecard, pd.DataFrame)
        assert len(interval_scorecard) > 0

        # Check required columns
        required_cols = ["Feature", "Bin", "Left", "Right", "Points", "XAddEvidence"]
        for col in required_cols:
            assert col in interval_scorecard.columns

        # Check that we have intervals for each feature
        features = constructor.xgb_scorecard_with_points.Feature.unique()
        interval_features = interval_scorecard.Feature.unique()
        assert set(features) == set(interval_features)

        # Verify the scorecard is stored as an attribute
        assert hasattr(constructor, "xgb_scorecard_intv")
        assert constructor.xgb_scorecard_intv is not None

    def test_peo_pdo_points_creation(self, trained_model_and_constructor_depth_1):
        """Test the PEO/PDO points creation method."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        # Build scorecard and interval scorecard
        constructor.construct_scorecard()
        constructor.create_points()
        interval_scorecard = constructor.construct_scorecard_by_intervals()

        # Create PEO/PDO points
        peo_pdo_scorecard = constructor.create_points_peo_pdo(peo=600, pdo=50)

        # Verify structure
        assert isinstance(peo_pdo_scorecard, pd.DataFrame)
        assert "Points_PEO_PDO" in peo_pdo_scorecard.columns
        assert len(peo_pdo_scorecard) == len(interval_scorecard)

        # Verify points are numeric
        assert peo_pdo_scorecard["Points_PEO_PDO"].dtype in [np.float64, np.int64]

        # Verify no NaN values in points (unless input had NaN)
        non_missing_mask = ~peo_pdo_scorecard["Bin"].str.contains("Missing", na=False)
        assert not peo_pdo_scorecard.loc[non_missing_mask, "Points_PEO_PDO"].isna().any()

    def test_interval_scorecard_with_stats(self, trained_model_and_constructor_depth_1):
        """Test interval scorecard with statistics."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        constructor.construct_scorecard()
        constructor.create_points()

        # Create with stats
        interval_scorecard = constructor.construct_scorecard_by_intervals(add_stats=True)

        # Check statistical columns are present
        stat_cols = ["Count", "Events", "NonEvents", "CountPct", "WOE", "IV"]
        # sourcery skip: no-loop-in-tests
        for col in stat_cols:
            assert col in interval_scorecard.columns

        # Verify stats make sense
        assert (interval_scorecard["Count"] >= 0).all()
        assert (interval_scorecard["Events"] >= 0).all()
        assert (interval_scorecard["NonEvents"] >= 0).all()
        assert (interval_scorecard["CountPct"] >= 0).all()
        assert (interval_scorecard["CountPct"] <= 1).all()

    def test_interval_scorecard_without_stats(self, trained_model_and_constructor_depth_1):
        """Test interval scorecard without statistics."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        constructor.construct_scorecard()
        constructor.create_points()

        # Create without stats
        interval_scorecard = constructor.construct_scorecard_by_intervals(add_stats=False)

        # Check statistical columns are NOT present
        stat_cols = ["Count", "Events", "NonEvents", "CountPct", "WOE", "IV"]
        for col in stat_cols:
            assert col not in interval_scorecard.columns

    def test_interval_scorecard_simplification(self, trained_model_and_constructor_depth_1):
        """Test that interval scorecard simplifies the number of rules."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        constructor.construct_scorecard()
        constructor.create_points()
        interval_scorecard = constructor.construct_scorecard_by_intervals()

        # Count rules per feature
        standard_rules = constructor.xgb_scorecard_with_points.groupby("Feature").size()
        interval_rules = interval_scorecard.groupby("Feature").size()

        # Interval scorecard should have fewer or equal rules per feature
        for feature in standard_rules.index:
            if feature in interval_rules.index:
                assert interval_rules[feature] <= standard_rules[feature]

    def test_bin_format(self, trained_model_and_constructor_depth_1):
        """Test that bins are formatted correctly."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        constructor.construct_scorecard()
        constructor.create_points()
        interval_scorecard = constructor.construct_scorecard_by_intervals()

        # Check bin formats
        for bin_str in interval_scorecard["Bin"]:
            if bin_str != "Missing":
                # Should be either (-inf, x) or [x, y) format
                assert (
                    bin_str.startswith("(-inf,") or bin_str.startswith("[") or "Missing" in bin_str
                )
                assert bin_str.endswith(")") or "Missing" in bin_str

    def test_peo_pdo_with_external_scorecard(self, trained_model_and_constructor_depth_1):
        """Test PEO/PDO points creation with external scorecard."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        constructor.construct_scorecard()
        constructor.create_points()
        interval_scorecard = constructor.construct_scorecard_by_intervals()

        # Use external scorecard
        external_scorecard = interval_scorecard.copy()
        peo_pdo_scorecard = constructor.create_points_peo_pdo(
            peo=700, pdo=40, scorecard=external_scorecard
        )

        # Should not modify the constructor's internal scorecard
        assert "Points_PEO_PDO" not in constructor.xgb_scorecard_intv.columns

        # Should return the modified external scorecard
        assert "Points_PEO_PDO" in peo_pdo_scorecard.columns
        assert len(peo_pdo_scorecard) == len(external_scorecard)

    def test_precision_points_parameter(self, trained_model_and_constructor_depth_1):
        """Test the precision_points parameter in PEO/PDO method."""
        _, constructor, _, _, _, _ = trained_model_and_constructor_depth_1

        constructor.construct_scorecard()
        constructor.create_points()
        constructor.construct_scorecard_by_intervals()

        # Test different precision levels
        peo_pdo_0 = constructor.create_points_peo_pdo(peo=600, pdo=50, precision_points=0)
        peo_pdo_2 = constructor.create_points_peo_pdo(peo=600, pdo=50, precision_points=2)

        # Check precision
        points_0 = peo_pdo_0["Points_PEO_PDO"].iloc[0]
        points_2 = peo_pdo_2["Points_PEO_PDO"].iloc[0]

        assert isinstance(points_0, (int, float))
        assert isinstance(points_2, (int, float))

        # Check that precision is respected (at least for the format)
        assert abs(points_2 - round(points_2, 2)) < 1e-10
