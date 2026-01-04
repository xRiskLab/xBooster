"""
cb_constructor.py

This module provides a high-level interface for working with CatBoost scorecards.
It combines the functionality of CatBoostScorecard and CatBoostWOEMapper to provide
a streamlined workflow for creating and using scorecards.

Author: Denis Burakov
Github: @deburky
License: MIT
This code is licensed under the MIT License.
Copyright (c) 2025 xRiskLab
"""

import contextlib
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from xbooster.catboost_scorecard import CatBoostScorecard
from xbooster.catboost_wrapper import CatBoostWOEMapper
from xbooster.shap_scorecard import compute_shap_scores, extract_shap_values_cb


class CBScorecardConstructor:
    """
    A high-level interface for working with CatBoost scorecards.
    This class combines the functionality of CatBoostScorecard and CatBoostWOEMapper
    to provide a streamlined workflow for creating and using scorecards.
    """

    def __init__(
        self,
        model: Optional[CatBoostClassifier] = None,
        pool: Optional[Union[Pool, pd.DataFrame]] = None,
        y: Optional[pd.Series] = None,
        use_woe: bool = False,
        points_column: Optional[str] = None,
    ) -> None:
        """
        Initialize the scorecard constructor.

        Args:
            model: Trained CatBoostClassifier
            pool: CatBoost Pool object OR DataFrame (X) for training/validation.
                  If DataFrame is provided, y must also be provided to create Pool automatically.
            y: Labels (required if pool is a DataFrame)
            use_woe: If True, use WOE values; if False, use XAddEvidence (default: False)
            points_column: If provided, use this column for scoring

        Examples:
            # Using Pool object (original API)
            constructor = CBScorecardConstructor(model, pool)

            # Using X, y (consistent with XGBoost/LightGBM API)
            constructor = CBScorecardConstructor(model, X_train, y_train)
        """
        self.model = model
        self.use_woe = use_woe
        self.points_column = points_column

        # Support both Pool object and (X, y) pattern for consistency with XGBoost/LightGBM
        if isinstance(pool, pd.DataFrame) and y is not None:
            # Create Pool from X and y (consistent with XGBoost/LightGBM API)
            # Extract categorical features from model if available
            cat_features = None
            if self.model is not None:
                with contextlib.suppress(AttributeError, RuntimeError):
                    if cat_feature_indices := self.model.get_cat_feature_indices():
                        cat_features = cat_feature_indices
            self.pool = Pool(pool, y, cat_features=cat_features)
            self.X = pool  # Store for get_leafs() method
            self.y = y
        elif isinstance(pool, Pool):
            # Original API: Pool object provided
            self.pool = pool
            self.X = None  # Will be extracted from pool if needed
            self.y = None
        else:
            # No pool provided (lazy initialization)
            self.pool = pool
            self.X = None
            self.y = None

        # Auto-build scorecard if both model and pool are provided
        if self.model is not None and self.pool is not None:
            self._build_scorecard()
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

        # Build column list
        base_columns = [
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
            "XAddEvidence",
            "WOE",
            "IV",
            "DetailedSplit",
        ]

        # Return only the basic columns
        return scorecard[base_columns]

    def get_scorecard(self) -> pd.DataFrame:
        """
        Get the scorecard DataFrame.

        Returns:
            DataFrame containing the scorecard information
        """
        if self.scorecard_df is None:
            raise ValueError("Scorecard not built yet. Call fit() first.")
        return self.scorecard_df

    def get_leafs(
        self,
        X: pd.DataFrame,  # pylint: disable=C0103
        output_type: str = "leaf_index",
    ) -> pd.DataFrame:
        """
        Get leaf indices for a new dataset.

        Args:
            X: Input features DataFrame
            output_type: 'leaf_index' (only supported type for CatBoost)

        Returns:
            DataFrame with columns [tree_0, tree_1, ..., tree_n] containing leaf indices

        Note:
            CatBoost uses calc_leaf_indexes() which returns integer leaf indices.
            The 'margin' output_type is not supported for CatBoost.
        """
        if self.model is None:
            raise ValueError("Model must be set before calling get_leafs()")

        if output_type != "leaf_index":
            raise ValueError(f"CatBoost only supports output_type='leaf_index'. Got: {output_type}")

        # Create Pool from X (no labels needed for leaf prediction)
        # Extract categorical features from model if available
        cat_features = None
        with contextlib.suppress(AttributeError, RuntimeError):
            if cat_feature_indices := self.model.get_cat_feature_indices():
                cat_features = cat_feature_indices
        pool = Pool(X, cat_features=cat_features)
        leaf_indices = self.model.calc_leaf_indexes(pool)

        n_trees = leaf_indices.shape[1]
        _colnames = [f"tree_{i}" for i in range(n_trees)]

        # Return as integer DataFrame (matching XGBoost/LightGBM behavior)
        return pd.DataFrame(leaf_indices, columns=_colnames).astype(int)

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
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        method: Optional[str] = None,
        pdo: float = 50,
        target_points: float = 600,
        target_odds: float = 19,
    ) -> Union[float, np.ndarray]:
        """
        Predict scores using the specified method.

        Args:
            features: DataFrame or dictionary of feature values
            method: Scoring method to use:
                - None (default): Use traditional points-based scoring (scorecard-based)
                - 'raw': Use original XAddEvidence (scorecard-based)
                - 'woe': Use Weight of Evidence values (scorecard-based)
                - 'shap': Use SHAP values directly (computes SHAP on-the-fly, no binning table)
            pdo: Points to Double the Odds (only used for method='shap')
            target_points: Target score for reference odds (only used for method='shap')
            target_odds: Reference odds ratio (only used for method='shap')

        Returns:
            Series containing predicted scores
        """
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Handle SHAP method separately (no binning table needed)
        if method == "shap":
            # Use stored PDO parameters if available (from create_points), otherwise use provided/defaults
            if self.pdo_params is not None:
                pdo = self.pdo_params.get("pdo", pdo)
                target_points = self.pdo_params.get("target_points", target_points)
                target_odds = self.pdo_params.get("target_odds", target_odds)
            return self._predict_score_shap(features, pdo, target_points, target_odds)

        # Default to traditional points-based scoring if method is None
        if method is None:
            method = "pdo"

        # For other methods, use existing scorecard-based approach
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
                original_mapper.value_column = "XAddEvidence"
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
            raise ValueError(
                f"Unknown method: {method}. Use None (default), 'raw', 'woe', or 'shap'."
            )

        return pd.Series(scores) if isinstance(scores, (float, np.ndarray)) else scores

    def _predict_score_shap(
        self,
        features: pd.DataFrame,
        pdo: float = 50,
        target_points: float = 600,
        target_odds: float = 19,
    ) -> pd.Series:
        """
        Predict scores using SHAP values directly (no binning table).

        This method computes SHAP values on-the-fly for input features and scales them
        using PDO formula. Different from scorecard-based methods which use pre-computed
        binned values.

        Args:
            features: DataFrame of feature values
            pdo: Points to Double the Odds
            target_points: Target score for reference odds
            target_odds: Reference odds ratio

        Returns:
            Series of predicted scores
        """
        if self.model is None:
            raise ValueError("Model not set. Call fit() first.")

        # Create Pool for CatBoost
        # For prediction, we don't need labels, but Pool requires them
        # Create dummy labels (will be ignored for SHAP computation)
        dummy_labels = np.zeros(len(features))
        pool = Pool(features, dummy_labels)

        # Extract SHAP values for input features
        shap_values_full = extract_shap_values_cb(
            self.model, pool
        )  # Shape: (n_samples, n_features + 1)
        shap_values = shap_values_full[:, :-1]  # Feature contributions
        base_value = float(np.mean(shap_values_full[:, -1]))  # Base value (expected value)

        # Use the SHAP scorecard computation function
        scorecard_dict = {
            "pdo": pdo,
            "target_points": target_points,
            "target_odds": target_odds,
        }

        # Compute SHAP-based scores using the dedicated function
        scorecard_df = compute_shap_scores(
            shap_values=shap_values,
            base_value=base_value,
            feature_names=features.columns.tolist(),
            scorecard_dict=scorecard_dict,
        )

        # Extract final scores
        scores = scorecard_df["score"]

        return pd.Series(scores, name="Score")

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

        # Add Value column based on WOE or XAddEvidence
        value_col = self.mapper.get_value_column()
        table["Value"] = table[value_col]

        return table

    def create_points(
        self,
        pdo: float = 50,
        target_points: float = 600,
        target_odds: float = 19,
        precision_points: int = 0,
        score_type: str = "WOE",
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

        # Select value column based on score_type
        value_col = "XAddEvidence" if score_type == "XAddEvidence" else "WOE"
        # Get base value based on score_type
        if "EventRate" in scorecard.columns:
            # For WOE/XAddEvidence, use average event rate
            base_odds = scorecard["EventRate"].mean() / (1 - scorecard["EventRate"].mean())
        else:
            base_odds = target_odds  # fallback

        # Factor and Offset
        factor = pdo / np.log(2)
        offset = target_points - factor * np.log(base_odds)

        # Raw contribution score from selected value column
        n_trees = len(scorecard["Tree"].unique())
        # Don't negate here - match XGBoost approach: negate in formula instead
        scorecard["RawScore"] = factor * scorecard[value_col]
        scorecard["RawScore"] /= n_trees  # Normalize by number of trees

        # Align maximum score within each tree
        scorecard.set_index("Tree", inplace=True)
        tree_max = scorecard.groupby("Tree")["RawScore"].max()
        mean_shift = (tree_max.sum() - offset) / len(tree_max)

        # Calculate points using apply
        # Match XGBoost formula: -ScaledScore + var_offsets - shft_base_pts
        # So: -RawScore + tree_max - mean_shift
        # This ensures higher XAddEvidence (higher risk) results in lower scores
        scorecard["Points"] = scorecard.apply(
            lambda row: -row["RawScore"] + tree_max[row.name] - mean_shift, axis=1
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

    def predict_scores(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        method: Optional[str] = None,
        pdo: float = 50,
        target_points: float = 600,
        target_odds: float = 19,
    ) -> pd.DataFrame:
        """
        Predict decomposed scores for a given dataset.

        Args:
            features: Input features as DataFrame or dictionary
            method: Scoring method to use:
                - None (default): Use traditional scorecard-based approach (tree-level decomposition)
                - 'shap': Use SHAP values directly (feature-level decomposition)
            pdo: Points to Double the Odds (only used for method='shap')
            target_points: Target score for reference odds (only used for method='shap')
            target_odds: Reference odds ratio (only used for method='shap')

        Returns:
            DataFrame with decomposed scores (tree-level for default, feature-level for SHAP)
        """
        if method == "shap":
            # Use stored PDO parameters if available (from create_points), otherwise use provided/defaults
            if self.pdo_params is not None:
                pdo = self.pdo_params.get("pdo", pdo)
                target_points = self.pdo_params.get("target_points", target_points)
                target_odds = self.pdo_params.get("target_odds", target_odds)
            return self._predict_scores_shap(features, pdo, target_points, target_odds)

        # Default: use traditional scorecard-based approach (tree-level decomposition)
        if self.mapper is None:
            raise ValueError("Mapper not initialized. Call fit() first.")

        # Get scores for each tree
        tree_scores = {}
        for tree_idx in self.tree_indices:
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

    def _predict_scores_shap(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        pdo: float = 50,
        target_points: float = 600,
        target_odds: float = 19,
    ) -> pd.DataFrame:
        """
        Predict decomposed scores using SHAP values (feature-level decomposition).

        Uses intercept-based scoring where intercept and offset are distributed
        evenly across features, ensuring feature scores sum to the total score.

        Args:
            features: Input features DataFrame or dictionary
            pdo: Points to Double the Odds
            target_points: Target score for reference odds
            target_odds: Reference odds ratio

        Returns:
            DataFrame with feature-level score contributions and total score
        """
        if self.model is None:
            raise ValueError("Model not set. Call fit() first.")

        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()

        # Create Pool for CatBoost
        dummy_labels = np.zeros(len(features_df))
        pool = Pool(features_df, dummy_labels)

        # Extract SHAP values for input features
        shap_values_full = extract_shap_values_cb(
            self.model, pool
        )  # Shape: (n_samples, n_features + 1)
        shap_values = shap_values_full[:, :-1]  # Feature contributions
        base_value = float(np.mean(shap_values_full[:, -1]))  # Base value (expected value)

        # Use the SHAP scorecard computation function
        scorecard_dict = {
            "pdo": pdo,
            "target_points": target_points,
            "target_odds": target_odds,
        }

        return compute_shap_scores(
            shap_values=shap_values,
            base_value=base_value,
            feature_names=features_df.columns.tolist(),
            scorecard_dict=scorecard_dict,
        )

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
