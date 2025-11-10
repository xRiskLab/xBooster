"""
lgb_constructor.py

This module defines a class for generating a scorecard from a LightGBM model.
The methodology follows the same approach as XGBoost and CatBoost implementations,
adapted for LightGBM's specific tree structure and API.

Status: Alpha - Reference implementation for community contribution
Contributor: @RektPunk (initial request in issue #7)

The class provides methods for extracting leaf weights, constructing the
scorecard, creating points, and predicting scores based on the constructed
scorecard.

Example usage (to be implemented):

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    # Instantiate the LGBScorecardConstructor
    scorecard_constructor = LGBScorecardConstructor(
        lgb_model, X.loc[ix_train], y.loc[ix_train]
    )
    # Generate a scorecard
    scorecard_constructor.construct_scorecard()
    lgb_scorecard_with_points = scorecard_constructor.create_points(
        pdo=50, target_points=600, target_odds=50
    )
    # Make predictions using the scorecard
    credit_scores = scorecard_constructor.predict_score(X.loc[ix_test])
    gini = roc_auc_score(y.loc[ix_test], -credit_scores) * 2 - 1
    print(f"Test Gini score: {gini:.2%}")
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from ._utils import calculate_information_value, calculate_weight_of_evidence

# Note: These will be needed when implementing the methods:
# from typing import Optional
# from scipy.special import logit
# from ._utils import calculate_information_value, calculate_weight_of_evidence


class LGBScorecardConstructor:  # pylint: disable=R0902
    """
    A class for generating a scorecard from a trained LightGBM model.

    This implementation follows the same methodology as XGBScorecardConstructor,
    adapted for LightGBM's API and tree structure.

    Parameters:
    - model (lightgbm.LGBMClassifier): The trained LightGBM model.
    - X (pd.DataFrame): Features of the training data.
    - y (pd.Series): Labels of the training data.

    Methods to implement:
    - extract_leaf_weights: Extracts leaf weights based on the LightGBM model.
    - construct_scorecard: Combines leaf weights and an event summary.
    - create_points: Creates a pointscard from a scorecard.
    - predict_score: Predicts a score for a given dataset using the scorecard.

    Reference Implementation Notes:

    1. LightGBM Tree Structure Access:
       - Use model.booster_.dump_model() for tree structure
       - trees_to_dataframe() provides a convenient DataFrame representation

    2. Key Differences from XGBoost:
       - Tree indexing: tree_info[i]['tree_structure']
       - Leaf access: Direct dictionary access vs JSON parsing
       - Decision type: '<=' mapping instead of 'Sign'

    3. Base Score Calculation:
       base_score = np.log(y_train.mean() / (1 - y_train.mean()))

    4. Leaf Prediction:
       model.predict(X, pred_leaf=True) returns leaf indices
       model.predict(X, raw_score=True) returns margins
    """

    def __init__(self, model, X, y):  # pylint: disable=C0103
        """
        Initialize the LGBScorecardConstructor.

        Args:
            model: Trained LightGBM classifier
            X: Training features
            y: Training labels
        """
        if not isinstance(model, LGBMClassifier):
            raise TypeError("model must be an instance of lightgbm.LGBMClassifier")

        self.model = model
        self.X = X  # pylint: disable=C0103
        self.y = y

        # Extract model parameters
        self.booster_ = model.booster_
        self.n_estimators = model.n_estimators
        self.learning_rate = model.learning_rate
        self.max_depth = model.max_depth or -1

        # Calculate base score (prior log-odds)
        self.base_score = np.log(y.mean() / (1 - y.mean()))

        # Initialize scorecard storage
        self.lgb_scorecard = None
        self.lgb_scorecard_with_points = None
        self.pdo = None
        self.target_points = None
        self.target_odds = None
        self.precision_points = None
        self.score_type = None
        self._sql_query = None

    def extract_model_param(self, param):
        """
        Extracts a specific parameter from the LightGBM model configuration.

        Args:
            param (str): The name of the parameter to extract.

        Returns:
            float: The extracted parameter value.

        Note: LightGBM parameters can be accessed via model.get_params()
        """
        params = self.model.get_params()
        if param in params:
            value = params[param]
            if isinstance(value, str):
                value = value.strip("[]'\"")
            return float(value) if value is not None else None
        return None

    def get_leafs(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        output_type: str = "margin",
    ) -> pd.DataFrame:
        """
        Get leaf indices or margins for each tree.

        Args:
            X: Input features
            output_type: 'leaf_index' or 'margin'
                - 'leaf_index': Returns leaf node indices
                - 'margin': Returns leaf values (raw scores)

        Returns:
            DataFrame with columns [tree_0, tree_1, ..., tree_n]

        Implementation follows XGBoost pattern using LightGBM's predict API.
        """
        n_trees = self.booster_.num_trees()
        _colnames = [f"tree_{i}" for i in range(n_trees)]

        if output_type == "leaf_index":
            # Predict leaf indices for all trees
            tree_leaf_idx = self.model.predict(X, pred_leaf=True)
            return pd.DataFrame(tree_leaf_idx, columns=_colnames)

        # For margin output, we need to get raw scores per tree
        # LightGBM doesn't have direct iteration_range like XGBoost,
        # so we use num_iteration to get cumulative scores then compute differences
        df_leafs = pd.DataFrame()

        # For binary classification, get contributions per tree
        # Note: This is an approximation - pred_contrib gives feature contributions
        # For now, we'll use predict with num_iteration to get cumulative scores
        for i in range(n_trees):
            df_leafs[f"tree_{i}"] = self.model.predict(
                X, raw_score=True, start_iteration=i, num_iteration=1
            ) - self.base_score * (i == 0)

        return df_leafs

    def extract_leaf_weights(self) -> pd.DataFrame:
        """
        Extract leaf weights from the LightGBM model.

        Returns:
            DataFrame with columns: Tree, Node, Feature, Sign, Split, XAddEvidence

        Implementation follows XGBoost pattern:
        - Get tree structure from trees_to_dataframe()
        - Identify decision nodes and leaf nodes
        - Map left/right children to leaf values
        - Convert decision_type '<=' to Sign '<' and '>='

        Note: XAddEvidence represents the leaf weight (margin/log-odds contribution)
        """
        tree_df = self.booster_.trees_to_dataframe()

        # Extract decision nodes (non-leaf nodes with split features)
        decision_nodes = tree_df[tree_df["split_feature"].notna()][
            [
                "tree_index",
                "node_index",
                "split_feature",
                "threshold",
                "decision_type",
                "left_child",
                "right_child",
            ]
        ].copy()

        # Extract leaf nodes (nodes without split features)
        leaf_nodes = tree_df[tree_df["split_feature"].isna()][
            ["tree_index", "node_index", "value"]
        ].copy()
        # Make leaf index relative within each tree
        leaf_nodes["relative_leaf_index"] = leaf_nodes.groupby("tree_index").cumcount()

        # Helper function to merge decision nodes with leaf values
        def merge_and_format(decisions, leafs, child_column, sign):
            """Merge decision nodes with their corresponding leaf children."""
            merged = decisions.merge(
                leafs,
                left_on=["tree_index", child_column],
                right_on=["tree_index", "node_index"],
                how="inner",
            )
            result = merged.rename(
                columns={
                    "relative_leaf_index": "Node",  # Leaf node index
                    "split_feature": "Feature",
                    "threshold": "Split",
                    "value": "XAddEvidence",
                }
            )
            result["Sign"] = sign
            result["Tree"] = result["tree_index"]
            return result[["Tree", "Node", "Feature", "Sign", "Split", "XAddEvidence"]]

        # Map left children (decision_type '<=' means left child gets '<')
        left_leafs = merge_and_format(decision_nodes, leaf_nodes, "left_child", "<")

        # Map right children (decision_type '<=' means right child gets '>=')
        right_leafs = merge_and_format(decision_nodes, leaf_nodes, "right_child", ">=")

        # Combine and sort by tree
        leaf_weights_df = pd.concat([left_leafs, right_leafs], ignore_index=True)
        leaf_weights_df = leaf_weights_df.sort_values(by="Tree").reset_index(drop=True)

        return leaf_weights_df

    def construct_scorecard(self) -> pd.DataFrame:
        """
        Construct a scorecard by combining leaf weights with event statistics.

        Returns:
            DataFrame with scorecard including WOE and IV calculations

        Implementation notes:
        - Similar to XGBoost implementation
        - Use get_leafs() to map observations to leaf nodes
        - Calculate event rates per leaf
        - Apply WOE/IV calculations from _utils
        """
        n_trees = self.booster_.num_trees()
        labels = self.y
        tree_leaf_idx = self.booster_.predict(self.X, pred_leaf=True)
        if tree_leaf_idx.shape != (len(labels), n_trees):
            raise ValueError(
                f"Invalid leaf index shape {tree_leaf_idx.shape}. Expected {(len(labels), n_trees)}"
            )

        df_binning_table = pd.DataFrame()
        for i in range(n_trees):
            index_and_label = pd.concat(
                [
                    pd.Series(tree_leaf_idx[:, i], name="leaf_idx"),
                    pd.Series(labels, name="label"),
                ],
                axis=1,
            )
            # Create a binning table
            binning_table = (
                index_and_label.groupby("leaf_idx").agg(["sum", "count"]).reset_index()
            ).astype(float)
            binning_table.columns = ["leaf_idx", "Events", "Count"]  # type: ignore
            binning_table["tree"] = i
            binning_table["NonEvents"] = binning_table["Count"] - binning_table["Events"]
            binning_table["EventRate"] = binning_table["Events"] / binning_table["Count"]
            binning_table = binning_table[
                ["tree", "leaf_idx", "Events", "NonEvents", "Count", "EventRate"]
            ]
            # Aggregate indices, leafs, and counts of events and non-events
            df_binning_table = pd.concat([df_binning_table, binning_table], axis=0)
        # Extract leaf weights (XAddEvidence)
        df_x_add_evidence = self.extract_leaf_weights()
        self.lgb_scorecard = df_x_add_evidence.merge(
            df_binning_table,
            left_on=["Tree", "Node"],
            right_on=["tree", "leaf_idx"],
            how="left",
        ).drop(["tree", "leaf_idx"], axis=1)

        self.lgb_scorecard = self.lgb_scorecard[
            [
                "Tree",
                "Node",
                "Feature",
                "Sign",
                "Split",
                "Count",
                "NonEvents",
                "Events",
                "EventRate",
                "XAddEvidence",
            ]
        ]

        # Sort by Tree and Node
        self.lgb_scorecard = self.lgb_scorecard.sort_values(by=["Tree", "Node"]).reset_index(
            drop=True
        )
        # Get WOE and IV scores
        self.lgb_scorecard["WOE"] = calculate_weight_of_evidence(self.lgb_scorecard)["WOE"]
        self.lgb_scorecard["IV"] = calculate_information_value(self.lgb_scorecard)["IV"]

        # Get % of observation counts in a Split
        self.lgb_scorecard["CountPct"] = self.lgb_scorecard["Count"] / self.lgb_scorecard.groupby(
            "Tree"
        )["Count"].transform("sum")

        self.lgb_scorecard = self.lgb_scorecard[
            [
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
            ]
        ]
        return self.lgb_scorecard

    def create_points(
        self,
        pdo: int = 50,
        target_points: int = 600,
        target_odds: int = 19,
        precision_points: int = 0,
        score_type: str = "LGBValue",
    ) -> pd.DataFrame:
        """
        Create points from scorecard using PDO (Points to Double Odds) scaling.

        Args:
            pdo: Points to double the odds
            target_points: Points at target odds
            target_odds: Target odds ratio
            precision_points: Decimal precision for points
            score_type: Column name for score values

        Returns:
            DataFrame with Points column added

        Implementation notes:
        - Formula: Points = (target_points - (pdo/ln(2)) * (ln(odds) - ln(target_odds)))
        - Apply to leaf values or WOE scores

        TODO: Implement this method following XGBoost pattern
        """
        raise NotImplementedError("create_points() method needs to be implemented")

    def predict_score(self, X: pd.DataFrame) -> pd.Series:  # pylint: disable=C0103
        """
        Predict scores for new data using the constructed scorecard.

        Args:
            X: Input features

        Returns:
            Series of credit scores

        Implementation notes:
        - Map observations to leaf nodes using predict(pred_leaf=True)
        - Look up points from scorecard
        - Sum across trees

        TODO: Implement this method following XGBoost pattern
        """
        raise NotImplementedError("predict_score() method needs to be implemented")

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """
        Predict detailed scores showing contribution from each tree.

        Args:
            X: Input features

        Returns:
            DataFrame with tree-level score breakdowns

        TODO: Implement this method following XGBoost pattern
        """
        raise NotImplementedError("predict_scores() method needs to be implemented")

    @property
    def sql_query(self) -> str:
        """
        Generate SQL query for scorecard deployment.

        Returns:
            SQL query string for scoring

        TODO: Implement this method following XGBoost pattern
        """
        raise NotImplementedError("SQL query generation needs to be implemented")

    # Additional helper methods can be added here following the XGBoost pattern
    # Reference: xbooster/xgb_constructor.py for implementation examples
