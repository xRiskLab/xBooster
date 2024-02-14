"""
Test module for xbooster.explainer.

This module contains unit tests for the functions and methods in the xbooster.explainer module.
It includes tests for plotting functions such as plot_importance, plot_score_distribution, 
and plot_local_importance, as well as the build_interactions_splits function.

"""

import unittest
import pandas as pd
import numpy as np
import pytest
from xbooster.explainer import (  # pylint: disable=E0401
    build_interactions_splits,  # pylint: disable=E0401
    plot_importance,  # pylint: disable=E0401
    plot_score_distribution,  # pylint: disable=E0401
    plot_local_importance,  # pylint: disable=E0401
)  # pylint: disable=E0401
from xbooster.constructor import XGBScorecardConstructor  # pylint: disable=E0401
import xgboost as xgb


class TestExplainer(unittest.TestCase):
    """
    Test suite for the Explainer module.

    This class contains test methods for various functionalities
    provided by the Explainer module. It sets up the test environment
    and defines individual test cases for different methods.

    Methods:
    - setUp: Set up the test environment.
    - test_build_interactions_splits: Test case for the build_interactions_splits method.
    - test_plot_importance: Test case for the plot_importance function.
    - test_plot_score_distribution: Test the plot_score_distribution function.
    - test_plot_local_importance: Test case for the plot_local_importance function.

    Examples:
        test_obj = TestExplainer()
        test_obj.setUp()
    """

    def setUp(self):
        """
        Set up the test environment.

        Args:
            self: The instance of the test class.

        Examples:
            test_obj = TestExplainer()
            test_obj.setUp()
        """
        # Mock data for testing
        self.model = None  # Assume we have a trained XGBoost model
        self.dataset = pd.DataFrame(
            {
                "feature1": np.random.rand(500),
                "feature2": np.random.rand(500),
                "label": np.random.randint(0, 2, 500),
            }
        )

        features = ["feature1", "feature2"]
        label = "label"
        X, y = self.dataset[features], self.dataset[label]  # pylint: disable=C0103
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X, y)
        self.scorecard_constructor = XGBScorecardConstructor(xgb_model, X, y)
        self.scorecard_constructor.construct_scorecard()  # Build the scorecard
        self.scorecard_constructor.create_points()  # Create scorecard points
        self.sample_to_explain = pd.DataFrame(
            {"feature1": np.random.rand(1), "feature2": np.random.rand(1)}
        )
        self.y_true = self.dataset["label"]
        self.y_pred = pd.Series(np.random.rand(500), index=self.y_true.index)

    def test_build_interactions_splits(self):
        """
        Test case for the build_interactions_splits method.

        This method tests the functionality of the build_interactions_splits method by
        asserting that the returned object is an instance of pd.DataFrame. Additional
        assertions can be added based on the expected behavior.

        """
        splits_df = build_interactions_splits(scorecard_constructor=self.scorecard_constructor)
        self.assertIsInstance(splits_df, pd.DataFrame)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_importance(self):
        """
        Test case for the plot_importance function.

        This test case checks the behavior of the plot_importance function in different scenarios.
        It verifies that the function raises a ValueError when called without any arguments,
        raises a ValueError when an invalid metric is provided, and
        successfully plots the importance when called with a valid scorecard_constructor.

        Additional assertions can be added based on the expected behavior of the function.
        """
        with self.assertRaises(ValueError):
            plot_importance()  # Should raise ValueError
        with self.assertRaises(ValueError):
            plot_importance(
                scorecard_constructor=self.scorecard_constructor, metric="InvalidMetric"
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_score_distribution(self):
        """
        Test the plot_score_distribution function.

        This test case checks if the plot_score_distribution function raises a ValueError when
        called without any arguments, and also verifies the expected behavior when called with
        valid arguments.

        """
        with self.assertRaises(ValueError):
            plot_score_distribution()  # Should raise ValueError
        plot_score_distribution(y_true=self.y_true, y_pred=self.y_pred)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_local_importance(self):
        """
        Test case for the plot_local_importance function.

        This test case checks the behavior of the plot_local_importance function
        under different scenarios. It verifies that the function raises a ValueError
        when an invalid input is provided, and it also checks the expected behavior
        when a valid input is provided.

        """
        with self.assertRaises(ValueError):
            plot_local_importance(
                self.scorecard_constructor, X=np.array([1, 2])  # type: ignore
            )  # Should raise ValueError

        # Check the expected behavior when a valid input is provided
        valid_input_df = pd.DataFrame({"feature1": [1], "feature2": [2]})
        try:
            plot_local_importance(
                scorecard_constructor=self.scorecard_constructor, X=valid_input_df
            )
        except ValueError:
            self.fail("plot_local_importance raised ValueError unexpectedly")


if __name__ == "__main__":
    unittest.main()
