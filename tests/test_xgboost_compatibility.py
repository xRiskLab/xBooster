"""
XGBoost compatibility tests.

This module contains tests to verify compatibility across different XGBoost versions.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from xbooster.xgb_constructor import XGBScorecardConstructor


@pytest.mark.compatibility
class TestXGBoostCompatibility:
    """Test XGBoost compatibility across versions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
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
    def trained_model(self, sample_data):
        """Create a trained XGBoost model."""
        X, y = sample_data
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        return model

    def test_xgboost_version_compatibility(self):
        """Test that we can import and use XGBoost."""
        # Basic import test
        assert hasattr(xgb, "XGBClassifier")
        assert hasattr(xgb, "DMatrix")

        # Version check
        version = xgb.__version__
        assert version >= "2.0.0", f"XGBoost version {version} is too old"
        print(f"Testing with XGBoost version: {version}")

    def test_model_creation_and_training(self, sample_data):
        """Test basic model creation and training."""
        X, y = sample_data

        # Test model creation
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)

        # Test training
        model.fit(X, y)

        # Test prediction
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_scorecard_constructor_compatibility(self, sample_data, trained_model):
        """Test XGBScorecardConstructor compatibility."""
        X, y = sample_data

        # Test constructor initialization
        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Test basic methods
        base_score = constructor.extract_model_param("base_score")
        assert isinstance(base_score, (int, float))

        learning_rate = constructor.extract_model_param("learning_rate")
        assert isinstance(learning_rate, (int, float))

        max_depth = constructor.extract_model_param("max_depth")
        assert isinstance(max_depth, (int, float))

    def test_scorecard_construction_compatibility(self, sample_data, trained_model):
        """Test scorecard construction compatibility."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Test scorecard construction
        scorecard = constructor.construct_scorecard()
        assert isinstance(scorecard, pd.DataFrame)
        assert len(scorecard) > 0

        # Test points creation
        points_card = constructor.create_points()
        assert isinstance(points_card, pd.DataFrame)
        assert len(points_card) > 0

    def test_prediction_compatibility(self, sample_data, trained_model):
        """Test prediction compatibility."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)
        constructor.construct_scorecard()
        constructor.create_points()

        # Test score prediction
        scores = constructor.predict_score(X[:10])
        assert len(scores) == 10
        assert isinstance(scores, pd.Series)

        # Test detailed scores
        detailed_scores = constructor.predict_scores(X[:10])
        assert isinstance(detailed_scores, pd.DataFrame)
        assert len(detailed_scores) == 10

    def test_api_methods_compatibility(self, sample_data, trained_model):
        """Test that all API methods work across XGBoost versions."""
        X, y = sample_data
        constructor = XGBScorecardConstructor(trained_model, X, y)

        # Test all major methods exist and are callable
        methods_to_test = [
            "extract_model_param",
            "extract_leaf_weights",
            "extract_decision_nodes",
            "construct_scorecard",
            "create_points",
            "predict_score",
            "predict_scores",
            "get_leafs",
            "add_detailed_split",
        ]

        for method_name in methods_to_test:
            assert hasattr(constructor, method_name), f"Method {method_name} not found"
            method = getattr(constructor, method_name)
            assert callable(method), f"Method {method_name} is not callable"

    def test_dmatrix_compatibility(self, sample_data):
        """Test DMatrix creation compatibility."""
        X, y = sample_data

        # Test basic DMatrix creation
        dmatrix = xgb.DMatrix(X, label=y)
        assert dmatrix is not None

        # Test with base_margin
        scores = np.full((X.shape[0],), 0.5)
        dmatrix_with_margin = xgb.DMatrix(X, label=y, base_margin=scores)
        assert dmatrix_with_margin is not None

    def test_booster_methods_compatibility(self, trained_model):
        """Test booster methods compatibility."""
        booster = trained_model.get_booster()

        # Test key methods exist and work
        assert hasattr(booster, "save_config")
        assert hasattr(booster, "trees_to_dataframe")
        assert hasattr(booster, "predict")
        assert hasattr(booster, "num_boosted_rounds")

        # Test save_config
        config = booster.save_config()
        assert isinstance(config, str)

        # Test trees_to_dataframe
        tree_df = booster.trees_to_dataframe()
        assert isinstance(tree_df, pd.DataFrame)

        # Test num_boosted_rounds
        rounds = booster.num_boosted_rounds()
        assert isinstance(rounds, int)
        assert rounds > 0
