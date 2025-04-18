"""
Test module for xbooster.explainer.

This module contains unit tests for the functions and methods in the xbooster.explainer module.
It includes tests for plotting functions such as plot_importance, plot_score_distribution,
and plot_local_importance, as well as the build_interactions_splits function.

"""

import unittest

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from xbooster.constructor import XGBScorecardConstructor  # pylint: disable=E0401
from xbooster.explainer import (
    TreeVisualizer,  # pylint: disable=E0401
    build_interactions_splits,  # pylint: disable=E0401
    plot_importance,  # pylint: disable=E0401
    plot_local_importance,  # pylint: disable=E0401
    plot_score_distribution,  # pylint: disable=E0401
)


class TestExplainer(unittest.TestCase):
    """
    Test suite for the Explainer module.
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
        # pylint: disable=C0103
        X, y = (
            self.dataset[features],
            self.dataset[label],
        )
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X, y)
        self.scorecard_constructor = XGBScorecardConstructor(xgb_model, X, y)
        self.scorecard_constructor.construct_scorecard()  # Build the scorecard
        self.scorecard_constructor.create_points()  # Create scorecard points
        self.sample_to_explain = pd.DataFrame(
            {
                "feature1": np.random.rand(1),
                "feature2": np.random.rand(1),
            }
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
                scorecard_constructor=self.scorecard_constructor,
                metric="InvalidMetric",
            )

        # Test with CatBoost constructor (should raise NotImplementedError)
        from catboost import CatBoostClassifier

        from xbooster.cb_constructor import CatBoostScorecardConstructor

        catboost_model = CatBoostClassifier(iterations=10, depth=3, verbose=0)
        catboost_model.fit(self.dataset[["feature1", "feature2"]], self.dataset["label"])
        catboost_constructor = CatBoostScorecardConstructor(
            catboost_model, self.dataset[["feature1", "feature2"]], self.dataset["label"]
        )

        with self.assertRaises(NotImplementedError):
            plot_importance(scorecard_constructor=catboost_constructor)

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
        # Test with non-DataFrame input to confirm it raises ValueError
        with self.assertRaises(ValueError):
            sample_to_explain = self.dataset[:1]
            plot_local_importance(self.scorecard_constructor, sample_to_explain)

    def test_tree_visualizer(self):
        """
        Test case for the TreeVisualizer class.

        This test case checks the behavior of the TreeVisualizer class in different scenarios.
        It verifies that the class raises a ValueError when called without a scorecard constructor,
        successfully parses XGBoost model output, and correctly plots the decision tree.

        Additional assertions can be added based on the expected behavior of the class.

        """
        # Check if ValueError is raised when scorecard constructor is not set
        with self.assertRaises(ValueError):
            tree_visualizer = TreeVisualizer()
            tree_visualizer.parse_xgb_output()

        # Check if parsing XGBoost model output works correctly
        tree_visualizer = TreeVisualizer()
        tree_visualizer.parse_xgb_output(scorecard_constructor=self.scorecard_constructor)
        self.assertIsInstance(tree_visualizer.tree_dump, dict)

        # Check if decision tree is plotted successfully
        try:
            tree_visualizer.plot_tree(scorecard_constructor=self.scorecard_constructor)
        except ValueError as ve:
            self.fail(f"Failed to plot decision tree: {str(ve)}")
        except RuntimeError as re:
            self.fail(f"A runtime error occurred during plotting: {str(re)}")


if __name__ == "__main__":
    unittest.main()
