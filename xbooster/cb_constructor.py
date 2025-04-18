"""
CatBoost Scorecard Constructor
=================================
This module provides a high-level interface for working with CatBoost scorecards.
It combines the functionality of CatBoostScorecard and CatBoostWOEMapper to provide
a streamlined workflow for creating and using scorecards.

Author: Denis Burakov
Github: @deburky
License: MIT
This code is licensed under the MIT License.
Copyright (c) 2025 Denis Burakov
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from xbooster.catboost_scorecard import CatBoostScorecard
from xbooster.catboost_wrapper import CatBoostWOEMapper


class CatBoostScorecardConstructor:
    """
    A high-level interface for working with CatBoost scorecards.
    This class combines the functionality of CatBoostScorecard and CatBoostWOEMapper
    to provide a streamlined workflow for creating and using scorecards.
    """

    def __init__(
        self,
        model: Optional[CatBoostClassifier] = None,
        pool: Optional[Pool] = None,
        use_woe: bool = True,
        points_column: Optional[str] = None,
    ) -> None:
        """
        Initialize the scorecard constructor.

        Args:
            model: Trained CatBoostClassifier
            pool: CatBoost Pool object used for training/validation
            use_woe: If True, use WOE values; if False, use LeafValue
            points_column: If provided, use this column for scoring
        """
        self.model = model
        self.pool = pool
        self.use_woe = use_woe
        self.points_column = points_column
        self.scorecard_df: Optional[pd.DataFrame] = None
        self.mapper: Optional[CatBoostWOEMapper] = None
        self.scorecard: Optional[pd.DataFrame] = None
        self.tree_indices: list = []
        self._sql_query: Optional[str] = None

        # Initialize pdo_params with default values
        self.pdo_params = {
            "pdo": 50,
            "target_points": 600,
            "target_odds": 19,
            "precision_points": 0,
        }

    def fit(self, model: CatBoostClassifier, pool: Pool) -> None:
        """
        Fit the scorecard constructor with a trained model and pool.

        Args:
            model: Trained CatBoostClassifier
            pool: CatBoost Pool object used for training/validation
        """
        self.model = model
        self.pool = pool
        self._build_scorecard()

    def _build_scorecard(self) -> None:
        """Build the scorecard from the model and pool."""
        if self.model is None or self.pool is None:
            raise ValueError("Model and pool must be set before building scorecard")

        # Create the scorecard
        self.scorecard_df = CatBoostScorecard.trees_to_scorecard(
            self.model, self.pool, output_format="pandas"
        )

        # Initialize the mapper
        self.mapper = CatBoostWOEMapper(
            self.scorecard_df,
            use_woe=self.use_woe,
            points_column=self.points_column,
        )

        # Generate feature mappings
        self.mapper.generate_feature_mappings()
        self.mapper.calculate_feature_importance()

        # Store the scorecard and tree indices
        self.scorecard = self.scorecard_df
        self.tree_indices = sorted(self.scorecard_df["Tree"].unique())

    def extract_leaf_weights(self) -> pd.DataFrame:
        """
        Extract leaf weights from the model.

        Returns:
            DataFrame containing leaf weights
        """
        if self.model is None:
            raise ValueError("Model not set. Call fit() first.")
        return CatBoostScorecard.extract_leaf_weights(self.model)

    def construct_scorecard(self) -> pd.DataFrame:
        """
        Construct a scorecard from the model and pool.

        Returns:
            DataFrame containing the scorecard information
        """
        if self.scorecard_df is None:
            self._build_scorecard()

        scorecard = self.scorecard_df.copy()

        # Extract feature names and split values from DetailedSplit
        for idx, row in scorecard.iterrows():
            detailed_split = row.get("DetailedSplit")
            if detailed_split and pd.notna(detailed_split):
                # Split the condition string
                parts = str(detailed_split).split(" AND ")
                if parts:
                    # Take the last condition as it's the most specific
                    last_condition = parts[-1]

                    # Extract feature name and split value
                    if " <= " in last_condition:
                        feature, value = last_condition.split(" <= ")
                        scorecard.loc[idx, "Feature"] = feature.strip()
                        scorecard.loc[idx, "Split"] = float(value.strip())
                        scorecard.loc[idx, "Sign"] = "<="
                    elif " > " in last_condition:
                        feature, value = last_condition.split(" > ")
                        scorecard.loc[idx, "Feature"] = feature.strip()
                        scorecard.loc[idx, "Split"] = float(value.strip())
                        scorecard.loc[idx, "Sign"] = ">"
                    elif " = " in last_condition:
                        feature, value = last_condition.split(" = ")
                        scorecard.loc[idx, "Feature"] = feature.strip()
                        scorecard.loc[idx, "Split"] = value.strip().strip("'\"")
                        scorecard.loc[idx, "Sign"] = "="
                    elif " != " in last_condition:
                        feature, value = last_condition.split(" != ")
                        scorecard.loc[idx, "Feature"] = feature.strip()
                        scorecard.loc[idx, "Split"] = value.strip().strip("'\"")
                        scorecard.loc[idx, "Sign"] = "!="

        # Calculate average event rate
        total_events = scorecard["Events"].sum()
        total_count = scorecard["Count"].sum()
        avg_event_rate = total_events / total_count if total_count > 0 else 0.0

        # Calculate WOE and IV
        scorecard["WOE"] = np.log(
            (scorecard["EventRate"] / (1 - scorecard["EventRate"]))
            / (avg_event_rate / (1 - avg_event_rate))
        )
        scorecard["IV"] = (scorecard["EventRate"] - avg_event_rate) * scorecard["WOE"]

        # Calculate CountPct
        scorecard["CountPct"] = (scorecard["Count"] / total_count * 100).fillna(0.0)

        # Return only the basic columns
        return scorecard[
            [
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
                "Conditions",
            ]
        ]

    def get_scorecard(self) -> pd.DataFrame:
        """
        Get the scorecard DataFrame.

        Returns:
            DataFrame containing the scorecard information
        """
        if self.scorecard_df is None:
            raise ValueError("Scorecard not built yet. Call fit() first.")
        return self.scorecard_df

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")
        return self.mapper.feature_importance

    def predict_score(
        self, features: Union[pd.DataFrame, Dict[str, Any]], method: str = "raw"
    ) -> Union[float, np.ndarray]:
        """
        Predict scores using the specified method.

        Args:
            features: DataFrame or dictionary of feature values
            method: Scoring method to use:
                - 'raw': Use original LeafValue
                - 'woe': Use Weight of Evidence values
                - 'pdo': Use points-based scoring

        Returns:
            Series containing predicted scores
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")

        # Ensure we have a valid scorecard
        if self.scorecard is None:
            self.construct_scorecard()

        # For points-based scoring, ensure we have points
        if method == "pdo" and "Points" not in self.scorecard.columns:
            self.create_points()

        # Set the appropriate value column based on method
        if method == "raw":
            self.mapper.value_column = "LeafValue"
            self.mapper.use_woe = False
        elif method == "woe":
            self.mapper.value_column = "WOE"
            self.mapper.use_woe = True
        elif method == "pdo":
            self.mapper.value_column = "Points"
            self.mapper.use_woe = False

        scores = self.mapper.predict_score(features)
        return pd.Series(scores) if isinstance(scores, (float, np.ndarray)) else scores

    def transform(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform features using the scorecard mapping.

        Args:
            features: Input features as DataFrame or dictionary

        Returns:
            Transformed features DataFrame
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        return self.mapper.transform_dataset(features)

    def plot_feature_importance(
        self, figsize: tuple[int, int] = (10, 6), top_n: Optional[int] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            figsize: Figure size
            top_n: Number of top features to plot
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")
        self.mapper.plot_feature_importance(figsize=figsize, top_n=top_n)

    def create_scorecard(self, pdo_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Create a detailed scorecard with points.

        Args:
            pdo_params: Parameters for points doubling odds calculation
                - pdo: Points to Double the Odds (default: 50)
                - target_points: Target score for reference odds (default: 600)
                - target_odds: Reference odds ratio (default: 19)
                - precision_points: Decimal places for points (default: 0)

        Returns:
            DataFrame containing the detailed scorecard
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")
        default_params = {
            "pdo": 50,
            "target_points": 600,
            "target_odds": 19,
            "precision_points": 0,
        }
        if pdo_params:
            default_params |= pdo_params
        return self.mapper.create_scorecard(pdo_params=default_params)

    def get_binned_feature_table(self) -> pd.DataFrame:
        """
        Get a table of binned features and their values.

        Returns:
            DataFrame containing binned feature information with columns:
            Feature, Condition, WOE, Weight, TreeCount, Bin, Value
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")
        table = self.mapper.get_binned_feature_table()
        if "Bin" not in table.columns:
            table["Bin"] = table["Condition"]

        # Add Value column based on WOE or LeafValue
        value_col = self.mapper.get_value_column()
        table["Value"] = table[value_col]

        return table

    def create_points(
        self,
        pdo: float = 50,
        target_points: float = 600,
        target_odds: float = 19,
        precision_points: int = 0,
    ) -> pd.DataFrame:
        """
        Create points for the scorecard using PDO (Points to Double the Odds) scaling.

        Args:
            pdo: Points to Double the Odds (default: 50)
            target_points: Target score for reference odds (default: 600)
            target_odds: Reference odds ratio (default: 19)
            precision_points: Decimal places for points (default: 0)

        Returns:
            DataFrame containing the scorecard with points
        """
        # First get the base scorecard
        scorecard = self.construct_scorecard().copy()

        # Calculate factor and offset for PDO scaling
        factor = pdo / np.log(2)
        offset = target_points - factor * np.log(target_odds)

        # Calculate points based on WOE
        scorecard["Points"] = (factor * scorecard["WOE"] + offset).round(precision_points)

        # Handle NaN and infinite values
        scorecard["Points"] = scorecard["Points"].replace([np.inf, -np.inf], np.nan)
        scorecard["Points"] = scorecard["Points"].fillna(0)  # Replace NaN with 0

        if precision_points <= 0:
            scorecard["Points"] = scorecard["Points"].astype(int)

        # Store the updated scorecard
        self.scorecard = scorecard
        self.scorecard_df = scorecard

        # Update mapper with the new scorecard that includes Points
        self.mapper = CatBoostWOEMapper(
            scorecard,
            use_woe=self.use_woe,
            points_column="Points",
        )
        self.mapper.scorecard = scorecard  # Ensure mapper has the scorecard
        self.mapper.enhanced_scorecard = scorecard  # Store enhanced scorecard
        self.mapper.generate_feature_mappings()
        self.mapper.calculate_feature_importance()

        return scorecard

    def predict_scores(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Predict scores for each tree and total score.

        Args:
            features: Input features as DataFrame or dictionary

        Returns:
            DataFrame with columns for each tree's score and total score
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")

        # Get scores for each tree
        tree_scores = {}
        for tree_idx in self.tree_indices:
            tree_data = self.scorecard[self.scorecard["Tree"] == tree_idx]
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            else:
                features_df = features.copy()

            # Get scores for this tree
            tree_scores[f"Tree_{tree_idx}"] = self.predict_score(features_df, method="raw")

        # Create DataFrame with tree scores
        scores_df = pd.DataFrame(tree_scores)
        scores_df["Score"] = scores_df.sum(axis=1)
        return scores_df

    def generate_sql_query(self, table_name="my_table"):
        """
        Generate an SQL query for deploying the scorecard.

        Parameters
        ----------
        table_name : str, optional (default="my_table")
            The name of the table to query from.

        Returns
        -------
        str
            The SQL query string.
        """
        if self.scorecard_df is None:
            raise ValueError("Scorecard not constructed. Call construct_scorecard() first.")

        # Add scorecard data as VALUES
        values = []
        values.extend(
            f"    SELECT {row['Tree']} AS Tree, {row.get('Node', 0)} AS Node, {row.get('XAddEvidence', 0)} AS XAddEvidence, {row.get('Points', 0)} AS Points"
            for _, row in self.scorecard_df.iterrows()
        )
        cte_parts = [
            "WITH scorecard AS (",
            "  SELECT",
            "    Tree,",
            "    Node,",
            "    XAddEvidence,",
            "    Points",
            "  FROM (",
            *["    " + "\n    UNION ALL\n".join(values), "  ) AS sc_data", ")"],
        ]
        # Main query
        query_parts = [
            "\nSELECT",
            "  t.*,",
            "  SUM(sc.Points) AS Score",
            f"FROM {table_name} t",
            "LEFT JOIN scorecard sc ON 1=1",
        ]

        # Add conditions for each tree and node
        conditions = []
        conditions.extend(
            "  AND ((t.Tree = sc.Tree AND t.Node = sc.Node) OR (sc.Tree != t.Tree))"
            for tree in self.scorecard_df["Tree"].unique()
        )
        query_parts.extend(conditions)

        # Group by all columns from the original table
        query_parts.extend(["GROUP BY", "  t.*"])

        # Combine all parts
        sql_query = "\n".join(cte_parts + query_parts)
        self._sql_query = sql_query
        return sql_query

    @property
    def sql_query(self):
        """
        Get the SQL query for deploying the scorecard.

        Returns
        -------
        str
            The SQL query string.
        """
        if self._sql_query is None:
            self._sql_query = self.generate_sql_query()
        return self._sql_query

    def predict(
        self, features: Union[pd.DataFrame, Dict[str, Any]], method: str = "raw"
    ) -> Union[float, np.ndarray]:
        """
        Make predictions using the scorecard.

        Args:
            features: Input features as DataFrame or dictionary
            method: Prediction method ('raw', 'simplified', or 'detailed')

        Returns:
            Predicted scores
        """
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")
        return self.mapper.predict_score(features, method=method)
