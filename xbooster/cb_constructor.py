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
        use_woe: bool = False,
        points_column: Optional[str] = None,
    ) -> None:
        """
        Initialize the scorecard constructor.

        Args:
            model: Trained CatBoostClassifier
            pool: CatBoost Pool object used for training/validation
            use_woe: If True, use WOE values; if False, use LeafValue (default: False)
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
        
        # Store original mapper and scorecard for raw/woe predictions
        self.original_mapper: Optional[CatBoostWOEMapper] = None
        self.original_scorecard: Optional[pd.DataFrame] = None
        self.points_enabled = False

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
        
        # Store the original objects
        self.original_mapper = self.mapper
        self.original_scorecard = self.scorecard_df

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
                if parts := str(detailed_split).split(" AND "):
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
        clipped_event_rate = scorecard["EventRate"].clip(lower=1e-3, upper=1 - 1e-3)

        # Calculate WOE and IV
        scorecard["WOE"] = np.log(
            (clipped_event_rate / (1 - clipped_event_rate))
            / (avg_event_rate / (1 - avg_event_rate))
        ).fillna(0)
        scorecard["IV"] = (scorecard["EventRate"] - avg_event_rate) * scorecard["WOE"]

        # Calculate CountPct
        scorecard["CountPct"] = (scorecard["Count"] / total_count).fillna(0.0)

        # Return only the basic columns
        return scorecard[
            [
                "Tree",
                "LeafIndex",
                "Feature",
                "Sign",
                "Split",
                "CountPct",
                "Count",
                "NonEvents",
                "Events",
                "EventRate",
                "LeafValue",
                "WOE",
                "IV",
                "DetailedSplit",
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
        if method == "pdo" and not self.points_enabled:
            self.create_points()

        if method == "raw" or method == "woe":
            # Use original mapper for raw and woe methods if available
            original_mapper = self.original_mapper if self.points_enabled else self.mapper

            # Set the appropriate value column based on method
            if method == "raw":
                original_mapper.value_column = "LeafValue"
                original_mapper.use_woe = False
            else:
                original_mapper.value_column = "WOE"
                original_mapper.use_woe = True

            scores = original_mapper.predict_score(features)
        elif method == "pdo":
            # Use current mapper for points method
            self.mapper.value_column = "Points"
            self.mapper.use_woe = False
            scores = self.mapper.predict_score(features)
        else:
            raise ValueError(f"Unknown method: {method}")

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
        # Store original mapper and scorecard if not already stored
        if not self.points_enabled:
            self.original_mapper = self.mapper
            self.original_scorecard = self.scorecard.copy() if self.scorecard is not None else None
        
        # First get the base scorecard
        scorecard = self.construct_scorecard().copy()
        
        # Always use WOE for points calculation to ensure sufficient variance
        value_col = "WOE"
        
        # Base score from average event rate if available
        if "EventRate" in scorecard.columns:
            base_odds = scorecard["EventRate"].mean() / (1 - scorecard["EventRate"].mean())
        else:
            base_odds = target_odds  # fallback
            
        # Factor and Offset
        factor = pdo / np.log(2)
        offset = target_points - factor * np.log(base_odds)

        # Raw contribution score from WOE or LeafValue
        scorecard["RawScore"] = -factor * scorecard[value_col]

        n_trees = len(scorecard["Tree"].unique())
        scorecard["RawScore"] = -factor * scorecard[value_col]
        scorecard["RawScore"] /= n_trees  # Normalize by number of trees

        # Align maximum score within each tree
        scorecard.set_index("Tree", inplace=True)
        tree_max = scorecard.groupby("Tree")["RawScore"].max()
        mean_shift = (tree_max.sum() - offset) / len(tree_max)

        # Calculate points using apply
        scorecard["Points"] = scorecard.apply(
            lambda row: tree_max[row.name] - row["RawScore"] - mean_shift, 
            axis=1
        )
        scorecard.reset_index(inplace=True)

        # Apply rounding
        scorecard["Points"] = scorecard["Points"].round(precision_points)
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
        
        # Mark that points have been created
        self.points_enabled = True
        
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

    def generate_sql_query(self):
        """
        Generate an SQL query for deploying the scorecard.

        Parameters
        ----------
        table_name : str, optional (default="input_data")
            The name of the table to query from.

        Returns
        -------
        str
            The SQL query string.
        """
       
        return ...

    @property
    def sql_query(self):
        """
        Get the SQL query for deploying the scorecard.

        Returns
        -------
        str
            The SQL query string.
        """
        return ...

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
