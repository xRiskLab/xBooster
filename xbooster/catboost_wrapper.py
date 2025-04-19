"""
catboost_wrapper.py

This module implements the inference functionality for CatBoost models.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CatBoostWOEMapper:
    """
    Maps features to their WOE/LeafValue from CatBoost scorecard and enables scorecard-based inference.

    This class provides functionality to:
    1. Map features to their corresponding WOE or LeafValue based on tree structure
    2. Calculate feature importance based on WOE/LeafValue contributions
    3. Transform data using the generated mapping
    4. Perform inference (prediction) using sum of WOE/LeafValue across trees
    5. Analyze and visualize feature contributions to the final score
    """

    def __init__(
        self,
        scorecard_df: pd.DataFrame,
        use_woe: bool = False,
        points_column: Optional[str] = None,
    ) -> None:
        """
        Initialize the mapper with a CatBoost scorecard.

        Args:
            scorecard_df: DataFrame containing CatBoost tree structure and metrics
            use_woe: If True, use WOE values; if False, use LeafValue (default: False)
            points_column: If provided, use this column for scoring
        """
        self.scorecard: pd.DataFrame = scorecard_df
        self.use_woe: bool = use_woe
        self.value_column: str = "WOE" if use_woe else "LeafValue"
        self.points_column: Optional[str] = points_column

        self.tree_indices: List[int] = []
        self.feature_mappings: Dict[
            str, Dict[str, Dict[str, Union[List[float], List[int], float]]]
        ] = {}
        self.feature_importance: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.condition_cache: Dict[str, List[str]] = {}

        # Initialize pdo_params attribute
        self.pdo_params: Optional[Dict[str, Any]] = None

        self._preprocess_scorecard()
        self.enhanced_scorecard: Optional[pd.DataFrame] = None

    def get_value_column(self) -> str:
        """Return the column name to use for values (points, WOE, or LeafValue)."""
        return self.points_column or self.value_column

    def _preprocess_scorecard(self) -> None:
        """Extract key information from the scorecard for faster access."""
        self.tree_indices = sorted(self.scorecard["Tree"].unique())
        self.feature_names = self._extract_feature_names()
        self._build_condition_cache()

    def _extract_feature_names(self) -> List[str]:
        """Extract all unique feature names from the conditions in the scorecard."""
        feature_names: Set[str] = set()
        for conditions in self.scorecard["DetailedSplit"].dropna():
            for condition in str(conditions).split(" AND "):
                if feature := self._get_feature_from_condition(condition):
                    feature_names.add(feature)
        return sorted(feature_names)

    @staticmethod
    def _get_feature_from_condition(condition: str) -> Optional[str]:
        """Extract feature name from a condition string."""
        if not condition or not isinstance(condition, str):
            return None

        match = re.match(r"^\s*([\w\d_]+)\s*(<=|>|=|!=|IN|NOT IN)", condition)
        return match[1] if match else None

    def _build_condition_cache(self) -> None:
        """Build a cache of conditions for each feature for faster lookup."""
        for _, row in self.scorecard.iterrows():
            conditions = row["DetailedSplit"]
            if not isinstance(conditions, str):
                continue
            for condition in conditions.split(" AND "):
                if feature := self._get_feature_from_condition(condition):
                    self.condition_cache.setdefault(feature, [])
                    if condition not in self.condition_cache[feature]:
                        self.condition_cache[feature].append(condition)

    def generate_feature_mappings(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Union[List[float], List[int], float]]]]:
        """
        Analyze the scorecard to generate mappings for each feature.
        This maps feature conditions to their corresponding WOE/LeafValue.

        Returns:
            Dictionary of feature mappings
        """
        feature_mappings = defaultdict(
            lambda: defaultdict(lambda: {"value": [], "weight": [], "trees": []})
        )

        value_col = self.get_value_column()

        for _, row in self.scorecard.iterrows():
            tree_idx = row["Tree"]
            conditions = row["DetailedSplit"]
            if not isinstance(conditions, str):
                continue
            value = row[value_col]
            weight = row["Count"]

            for condition in conditions.split(" AND "):
                feature = self._get_feature_from_condition(condition)
                if not feature:
                    continue
                mapping = feature_mappings[feature][condition]
                mapping["value"].append(value)
                mapping["weight"].append(weight)
                mapping["trees"].append(tree_idx)

        self.feature_mappings = feature_mappings
        self._aggregate_feature_mappings()
        return self.feature_mappings

    def _aggregate_feature_mappings(self) -> None:
        """Aggregate values across trees for each feature condition."""
        for _, conditions in self.feature_mappings.items():
            for _, details in conditions.items():
                weights = details["weight"]
                values = details["value"]
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_avg = sum(v * w for v, w in zip(values, weights)) / total_weight
                else:
                    weighted_avg = sum(values) / len(values) if values else 0.0
                details["agg_value"] = weighted_avg
                details["total_weight"] = total_weight
                details["tree_count"] = len(set(details["trees"]))

    def calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance as the sum of WOE values per feature.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # First ensure we're using WOE by forcing value_column to WOE
        original_value_column = self.value_column
        self.value_column = "WOE"

        try:
            # Create a direct feature importance calculation from the scorecard
            feature_woe_sum = self.scorecard.groupby("Feature")["WOE"].sum().abs()
            
            # Normalize to get relative importance
            total = feature_woe_sum.sum()
            importance = (feature_woe_sum / total).to_dict() if total > 0 else {}
            
            self.feature_importance = importance
            return importance
        
        finally:
            # Restore original value column setting
            self.value_column = original_value_column

    def evaluate_condition(self, condition: str, feature_values: Dict[str, Any]) -> bool:
        """
        Evaluate if a condition is satisfied by the given feature values.

        Args:
            condition: String condition from the scorecard
            feature_values: Dictionary of feature name -> value

        Returns:
            Boolean indicating if the condition is satisfied
        """
        feature = self._get_feature_from_condition(condition)
        if not feature or feature not in feature_values:
            return False

        value = feature_values[feature]
        try:
            if " <= " in condition:
                return float(value) <= float(condition.split(" <= ")[1])
            elif " > " in condition:
                return float(value) > float(condition.split(" > ")[1])
            elif " = " in condition:
                return str(value) == condition.split(" = ")[1].strip("'\"")
            elif " != " in condition:
                return str(value) != condition.split(" != ")[1].strip("'\"")
            elif " IN " in condition:
                values_str = condition.split(" IN ")[1].strip("()").split(", ")
                values = [v.strip("'\"") for v in values_str]
                return str(value) in values
            elif " NOT IN " in condition:
                values_str = condition.split(" NOT IN ")[1].strip("()").split(", ")
                values = [v.strip("'\"") for v in values_str]
                return str(value) not in values
        except (ValueError, TypeError, AttributeError):
            return False
        return False

    def transform_instance(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Transform a single instance by mapping features to their WOE/LeafValue.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Dictionary of feature -> transformed value
        """
        transformed = defaultdict(list)
        value_col = self.get_value_column()

        for leaf in self._map_instance_to_leaves(features).values():
            conditions = leaf["DetailedSplit"]
            if not isinstance(conditions, str):
                continue
            value = leaf[value_col]
            for condition in conditions.split(" AND "):
                if feature := self._get_feature_from_condition(condition):
                    transformed[feature].append(value)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in transformed.items()}

    def _map_instance_to_leaves(self, features: Dict[str, Any]) -> Dict[int, pd.Series]:
        """
        Map an instance to its corresponding leaf nodes in each tree.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Dictionary mapping tree index to leaf node
        """
        tree_to_leaf = {}
        for tree_idx in self.tree_indices:
            tree_data = self.scorecard[self.scorecard["Tree"] == tree_idx]
            for _, leaf in tree_data.iterrows():
                conditions = leaf["DetailedSplit"]
                if isinstance(conditions, str) and all(
                    self.evaluate_condition(cond, features) for cond in conditions.split(" AND ")
                ):
                    tree_to_leaf[tree_idx] = leaf
                    break
        return tree_to_leaf

    def transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform an entire dataset by mapping features to WOE/LeafValue.

        Args:
            df: DataFrame with original features

        Returns:
            DataFrame with transformed features
        """
        transformed = pd.DataFrame(index=df.index)
        value_col = self.get_value_column()

        for feature in self.feature_names:
            if feature not in df.columns:
                continue
            transformed[feature] = 0.0
            total_weight = np.zeros(len(df))

            for condition in self.condition_cache.get(feature, []):
                mask = self._evaluate_condition_vectorized(df[feature], condition)
                if mask.any():
                    condition_value = self._get_condition_value(condition, value_col)
                    transformed.loc[mask, feature] += condition_value
                    total_weight[mask] += 1

            non_zero = total_weight > 0
            transformed.loc[non_zero, feature] /= total_weight[non_zero]

        return transformed

    def _evaluate_condition_vectorized(self, series: pd.Series, condition: str) -> pd.Series:
        """
        Evaluate a condition across a series of values (vectorized).

        Args:
            series: Series of values for a single feature
            condition: Condition string

        Returns:
            Boolean series indicating which values satisfy the condition
        """
        try:
            s = series.astype(float)
            if " <= " in condition:
                return s <= float(condition.split(" <= ")[1])
            elif " > " in condition:
                return s > float(condition.split(" > ")[1])
            elif " = " in condition:
                value = condition.split(" = ")[1].strip("'\"")
                return series.astype(str) == value
            elif " != " in condition:
                value = condition.split(" != ")[1].strip("'\"")
                return series.astype(str) != value
            elif " IN " in condition:
                values_str = condition.split(" IN ")[1].strip("()").split(", ")
                values = [v.strip("'\"") for v in values_str]
                return series.astype(str).isin(values)
            elif " NOT IN " in condition:
                values_str = condition.split(" NOT IN ")[1].strip("()").split(", ")
                values = [v.strip("'\"") for v in values_str]
                return ~series.astype(str).isin(values)
        except (ValueError, TypeError, AttributeError):
            return pd.Series(False, index=series.index)
        return pd.Series(False, index=series.index)

    def _get_condition_value(self, condition: str, value_col: str) -> float:
        """
        Get the average value for a specific condition.

        Args:
            condition: Condition string
            value_col: Column name for the value

        Returns:
            Average value for the condition
        """
        relevant_rows = self.scorecard[
            self.scorecard["DetailedSplit"].str.contains(condition, na=False)
        ]
        if relevant_rows.empty:
            return 0.0
        return float(np.average(relevant_rows[value_col], weights=relevant_rows["Count"]))

    def get_binned_feature_table(self) -> pd.DataFrame:
        """
        Create a binned feature table for reporting and analysis.
        Shows how different bins/ranges of each feature map to WOE/LeafValue.

        Returns:
            DataFrame with binned feature information
        """
        if not self.feature_mappings:
            self.generate_feature_mappings()

        value_type = self.get_value_type()
        table_data = [
            {
                "Feature": f,
                "Condition": c,
                value_type: d["agg_value"],
                "Weight": d["total_weight"],
                "TreeCount": d["tree_count"],
            }
            for f, conds in self.feature_mappings.items()
            for c, d in conds.items()
        ]

        table = pd.DataFrame(table_data)
        table["AbsValue"] = table[value_type].abs()
        return table.sort_values(["Feature", "AbsValue"], ascending=[True, False]).drop(
            columns="AbsValue"
        )

    def get_value_type(self) -> str:
        """Return the type of value being used (Points, WOE, or LeafValue)."""
        return (
            "Points" if self.points_column else ("WOE" if self.use_woe else "LeafValue")
        )

    def plot_feature_importance(
        self, figsize: Tuple[int, int] = (12, 8), top_n: Optional[int] = None
    ) -> None:
        """
        Plot feature importance based on WOE/LeafValue magnitudes.

        Args:
            figsize: Tuple for figure size
            top_n: If provided, show only the top N features
        """
        if not self.feature_importance:
            self.calculate_feature_importance()

        # Sort features by importance in ascending order (smallest to largest)
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=False)

        if top_n:
            sorted_features = sorted_features[:top_n]

        features, importance = zip(*sorted_features)

        # Create the plot with monospace font
        plt.figure(figsize=figsize, dpi=100)
        plt.rcParams["font.family"] = "monospace"
        plt.rcParams["font.size"] = 12

        # Create horizontal bar plot with consistent color and black border
        plt.barh(
            range(len(features)),
            importance,
            align="center",
            height=0.8,
            color="#7950f2",  # Purple color from explainer.py
            edgecolor="black",
            linewidth=1,
        )

        # Customize the plot
        plt.yticks(range(len(features)), features)
        plt.xlabel("Relative Importance")
        plt.title("Feature Importance", fontsize=14)

        # Remove spines except vertical
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Adjust margins to make plot area even smaller
        plt.subplots_adjust(left=0.3, right=0.9, top=0.85, bottom=0.15)
        plt.show()

    # Scorecard-based inference
    def create_scorecard(
        self, pdo_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Create a statistically aligned scorecard with accurate points calculation.

        Args:
            pdo_params: Dictionary of points calculation parameters including:
                - pdo: Points to Double the Odds (default: 50)
                - target_points: Target score for reference odds (default: 600)
                - target_odds: Reference odds ratio (default: 19)
                - precision_points: Decimal places for points (default: 0)

        Returns:
            Enhanced scorecard with points column added
        """
        if not self.feature_mappings:
            self.generate_feature_mappings()

        scorecard = self.scorecard.copy()

        self.pdo_params = pdo_params or {
            "pdo": 50,
            "target_points": 600,
            "target_odds": 19,
            "precision_points": 0,
        }

        pdo = self.pdo_params["pdo"]
        target_points = self.pdo_params["target_points"]
        target_odds = self.pdo_params["target_odds"]
        precision_points = self.pdo_params["precision_points"]

        # Determine value source
        value_col = self.get_value_column()

        # Base score from average event rate if available
        if "EventRate" in scorecard.columns:
            base_odds = scorecard["EventRate"].mean() / (
                1 - scorecard["EventRate"].mean()
            )
        else:
            base_odds = target_odds  # fallback

        # Factor and Offset
        factor = pdo / np.log(2)
        offset = target_points - factor * np.log(base_odds)

        # Calculate raw scores from WOE or LeafValue
        n_trees = len(scorecard["Tree"].unique())
        
        # Directly use the value column for raw score calculation
        # We divide by number of trees to keep scores at a consistent scale
        scorecard["RawScore"] = -factor * scorecard[value_col] / n_trees
        
        # Calculate points by adding the offset to the raw scores
        # Each leaf contributes its share to the overall score
        scorecard["Points"] = offset/n_trees + scorecard["RawScore"]
        
        # Apply rounding
        scorecard["Points"] = scorecard["Points"].round(precision_points)
        if precision_points <= 0:
            scorecard["Points"] = scorecard["Points"].astype(int)

        self.enhanced_scorecard = scorecard
        return scorecard

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
            Score(s) based on the specified method
        """
        # Validate method
        if method not in ["raw", "woe", "pdo"]:
            raise ValueError(
                f"Unknown scoring method: {method}. Use 'raw', 'woe', or 'pdo'."
            )

        # Check if we need to create the scorecard first
        if method == "pdo" and not hasattr(self, "enhanced_scorecard"):
            self.create_scorecard()

        # Set the appropriate value column based on method
        original_value_column = self.value_column
        original_use_woe = self.use_woe
        temp_scorecard = None

        try:
            if method == "raw":
                self.value_column = "LeafValue"
                self.use_woe = False
            elif method == "woe":
                self.value_column = "WOE"
                self.use_woe = True
            elif method == "pdo":
                # Use the enhanced scorecard with Points column
                if hasattr(self, "enhanced_scorecard"):
                    temp_scorecard = self.scorecard
                    self.scorecard = self.enhanced_scorecard
                self.value_column = "Points"
                self.use_woe = False

            # Make prediction using the appropriate method
            if isinstance(features, pd.DataFrame):
                scores = self._predict_score_batch(features)
            else:
                scores = self._predict_score_single(features)

            return scores

        finally:
            # Restore original settings
            self.value_column = original_value_column
            self.use_woe = original_use_woe
            if temp_scorecard is not None:
                self.scorecard = temp_scorecard

    def _predict_score_batch(self, df: pd.DataFrame, method: str = "raw") -> np.ndarray:
        """
        Predict scores for a batch of instances using direct pandas filtering.
        """
        scores = np.zeros(len(df))
        value_col = self.get_value_column()

        # For each tree, find the leaf for each instance
        for tree_idx in self.tree_indices:
            tree_data = self.scorecard[self.scorecard["Tree"] == tree_idx]

            # Create a mask of unassigned rows
            unassigned = np.ones(len(df), dtype=bool)
            unassigned_indices = np.arange(len(df))

            # For each leaf in this tree
            for _, leaf in tree_data.iterrows():
                conditions = leaf["DetailedSplit"]
                if not isinstance(conditions, str) or not np.any(unassigned):
                    continue

                # Only process unassigned rows
                subset = df.loc[unassigned_indices[unassigned]]
                if len(subset) == 0:
                    continue

                # Apply conditions directly to subset using pandas filtering
                matches = np.ones(len(subset), dtype=bool)
                valid_filter = True

                for condition in conditions.split(" AND "):
                    feature = self._get_feature_from_condition(condition)
                    if not feature or feature not in df.columns:
                        valid_filter = False
                        break

                    # Apply appropriate filter based on condition type
                    try:
                        if " = " in condition:
                            value = condition.split(" = ")[1].strip("'\"")
                            condition_matches = subset[feature].astype(str) == value
                        elif " != " in condition:
                            value = condition.split(" != ")[1].strip("'\"")
                            condition_matches = subset[feature].astype(str) != value
                        elif " <= " in condition:
                            value = float(condition.split(" <= ")[1])
                            condition_matches = subset[feature].astype(float) <= value
                        elif " > " in condition:
                            value = float(condition.split(" > ")[1])
                            condition_matches = subset[feature].astype(float) > value
                        else:
                            valid_filter = False
                            break

                        matches = matches & condition_matches.values
                    except Exception:
                        valid_filter = False
                        break

                if not valid_filter:
                    continue

                # Get indices of matching rows in the original array
                if np.any(matches):
                    matching_subset_indices = subset.index[matches]
                    matching_positions = np.where(
                        np.isin(unassigned_indices, matching_subset_indices)
                    )[0]

                    # Add the leaf value to matching rows
                    value = leaf[value_col]
                    if method == "raw":
                        # No normalization for raw method
                        pass
                    elif method == "woe" and len(self.tree_indices) > 0:
                        score += leaf[value_col]
                    scores[unassigned_indices[matching_positions]] += value

                    # Mark these as assigned
                    unassigned[matching_positions] = False

            # For unassigned rows, use the default leaf (most common)
            if np.any(unassigned) and not tree_data.empty:
                default_idx = (
                    tree_data["Count"].idxmax()
                    if "Count" in tree_data.columns
                    else tree_data.index[0]
                )
                default_leaf = tree_data.loc[default_idx]
                value = default_leaf[value_col]
                if method == "raw":
                    # No normalization for raw method
                    pass
                elif method == "woe" and len(self.tree_indices) > 0:
                    # For WOE, normalize by tree count
                    value = value / len(self.tree_indices)
                scores[unassigned_indices[unassigned]] += value

        return scores

    def _predict_score_single(self, features: Dict[str, Any], method: str = "raw") -> float:
        """
        Predict score for a single instance with improved handling of categorical features.

        Args:
            features: Dictionary of feature name -> value
            method: Scoring method ('raw', 'woe', or 'pdo')

        Returns:
            Score from summing values across trees
        """
        score = 0.0
        value_col = self.get_value_column()

        for tree_idx in self.tree_indices:
            tree_data = self.scorecard[self.scorecard["Tree"] == tree_idx]
            assigned = False

            # Try each leaf in this tree
            for _, leaf in tree_data.iterrows():
                conditions = leaf["DetailedSplit"]
                if not isinstance(conditions, str):
                    continue

                # Check if all conditions are met
                conditions_met = True
                for condition in conditions.split(" AND "):
                    feature = self._get_feature_from_condition(condition)
                    if not feature or feature not in features:
                        conditions_met = False
                        break

                    feature_value = features[feature]

                    # Evaluate condition based on type
                    try:
                        if " = " in condition:
                            value = condition.split(" = ")[1].strip("'\"")
                            if str(feature_value) != value:
                                conditions_met = False
                                break
                        elif " != " in condition:
                            value = condition.split(" != ")[1].strip("'\"")
                            if str(feature_value) == value:
                                conditions_met = False
                                break
                        elif " <= " in condition:
                            value = float(condition.split(" <= ")[1])
                            if float(feature_value) > value:
                                conditions_met = False
                                break
                        elif " > " in condition:
                            value = float(condition.split(" > ")[1])
                            if float(feature_value) <= value:
                                conditions_met = False
                                break
                    except (ValueError, TypeError):
                        conditions_met = False
                        break

                # If all conditions met, add this leaf's value to the score
                if conditions_met:
                    value = leaf[value_col]
                    if method == "raw":
                        # No normalization for raw method
                        pass
                    elif method == "woe" and len(self.tree_indices) > 0:
                        # For WOE, normalize by tree count
                        value = value / len(self.tree_indices)
                    score += value
                    assigned = True
                    break

            # If no leaf matched, use the default (most common) leaf
            if not assigned and not tree_data.empty:
                default_idx = (
                    tree_data["Count"].idxmax()
                    if "Count" in tree_data.columns
                    else tree_data.index[0]
                )
                value = tree_data.loc[default_idx][value_col]
                if method == "raw":
                    # No normalization for raw method
                    pass
                elif method == "woe" and len(self.tree_indices) > 0:
                    # For WOE, normalize by tree count
                    value = value / len(self.tree_indices)
                score += value

        return score
