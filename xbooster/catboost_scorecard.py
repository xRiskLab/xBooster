"""
CatBoost Scorecard Script
=================================
This module provides functionality to extract data for scorecards from CatBoost models.
It handles both numerical and categorical features with one-hot encoding approach.

Author: Denis Burakov
Github: @deburky
License: MIT
This code is licensed under the MIT License.
Copyright (c) 2025 Denis Burakov

"""

import contextlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa  # pylint: disable=import-error
from catboost import CatBoostClassifier, Pool  # pylint: disable=import-error


# pylint: disable=protected-access
class CatBoostScorecard:
    """
    A class for extracting human-readable scorecards from CatBoost models.
    Handles both numerical and categorical features, including text features.
    Supports both feature-name-based and index-based trees.
    """

    # Class variable to store debug info
    debug_info = {}

    @staticmethod
    def _is_numeric_only_condition(condition: str) -> bool:
        """Check if a condition is numeric-only (like '4' or '3')."""
        try:
            int(condition.strip())
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _extract_feature_names(pool: Pool) -> List[str]:
        """Extract feature names from a CatBoost Pool if available."""
        feature_names = []
        with contextlib.suppress(Exception):
            # Try to access feature names from the pool
            if hasattr(pool, "get_feature_names") and callable(pool.get_feature_names):
                feature_names = pool.get_feature_names()
            # Alternative approach
            elif hasattr(pool, "_feature_names"):
                feature_names = pool._feature_names
        return feature_names

    @staticmethod
    def _get_split_feature_value(condition: str) -> Tuple[str, Any, str]:
        """Extract the feature name and split value from a condition string."""
        # Check if this is a numeric-only condition
        if CatBoostScorecard._is_numeric_only_condition(condition):
            feature_idx = int(condition.strip())
            return f"Feature_{feature_idx}", feature_idx, "index_based"

        # Numerical split with bin=
        if ", bin=" in condition:
            feature, value = condition.split(", bin=")
            return feature.strip(), value.strip(), "numerical"

        # One-hot categorical split with value=
        if ", value=" in condition:
            feature, value = condition.split(", value=")
            return feature.strip(), value.strip(), "categorical_value"

        # Bracket-style categorical split
        if "[" in condition:
            feature = condition.split("[")[0].strip()
            cats = condition.split("[", 1)[1].split("]", 1)[0]
            categories = [c.strip().strip("'\"") for c in cats.split(",")]
            return feature, categories, "categorical_list"

        # Fallback
        feature = condition.split(":")[0].strip()
        return feature, None, "unknown"

    @staticmethod
    def _parse_condition(
        condition: str, bit: str, feature_names: Optional[List[str]] = None
    ) -> str:
        """Parse a CatBoost split condition, handling both string and index-based conditions."""
        is_true = bit == "1"

        # Handle numeric-only condition (index-based)
        if CatBoostScorecard._is_numeric_only_condition(condition):
            feature_idx = int(condition.strip())
            # Try to map to feature name if available
            feature_name = f"Feature_{feature_idx}"
            if feature_names and 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
            return f"{feature_name} {'>' if is_true else '<='} threshold"

        # Handle standard conditions
        feature, value, split_type = CatBoostScorecard._get_split_feature_value(condition)

        if split_type == "numerical":
            return f"{feature} {'>' if is_true else '<='} {value}"

        if split_type == "categorical_value":
            return f"{feature} {'=' if is_true else '!='} '{value}'"

        if split_type == "categorical_list" and isinstance(value, list):
            if len(value) == 1:
                return f"{feature} {'=' if is_true else '!='} '{value[0]}'"
            cat_str = ", ".join(f"'{c}'" for c in value)
            return f"{feature} {'IN' if is_true else 'NOT IN'} ({cat_str})"

        # Fallback
        return f"{feature} {'>' if is_true else '<='}"

    @staticmethod
    def _get_leaf_conditions(cb_obj: object, pool: Pool, tree_idx: int) -> Dict[int, str]:
        """Get leaf conditions for a given tree index, handling both types of trees."""
        split_conditions = cb_obj._get_tree_splits(tree_idx, pool)
        leaf_count = int(cb_obj._get_tree_leaf_counts()[tree_idx])
        tree_depth = leaf_count.bit_length() - 1

        if 2**tree_depth != leaf_count:
            raise ValueError(f"Tree {tree_idx} is not complete binary")

        # Store raw conditions for debugging
        CatBoostScorecard.debug_info[f"tree_{tree_idx}_raw_conditions"] = split_conditions

        # Extract feature names if available
        feature_names = CatBoostScorecard._extract_feature_names(pool)
        CatBoostScorecard.debug_info["feature_names"] = feature_names  # type: ignore

        # Reverse order to match CatBoost's native visualization
        CatBoostScorecard.level_conditions = {tree_idx: list(reversed(split_conditions))}
        # Check if this is an index-based tree
        is_index_based = all(
            CatBoostScorecard._is_numeric_only_condition(cond) for cond in split_conditions if cond
        )
        CatBoostScorecard.debug_info[f"tree_{tree_idx}_is_index_based"] = is_index_based

        # Build path conditions for each leaf
        leaf_conditions = {}
        for leaf_idx in range(leaf_count):
            binary_path = format(leaf_idx, f"0{tree_depth}b")

            # Build full path condition
            path_conditions = []
            for level, bit in enumerate(binary_path):
                # Skip if we're beyond the number of available conditions
                if level >= len(split_conditions):
                    continue

                # Access conditions in reverse order to match native visualization
                condition = split_conditions[tree_depth - 1 - level]
                parsed_condition = CatBoostScorecard._parse_condition(condition, bit, feature_names)
                path_conditions.append(parsed_condition)

            # Join conditions with AND
            leaf_conditions[leaf_idx] = " AND ".join(path_conditions)

        return leaf_conditions

    @classmethod
    def trees_to_scorecard(
        cls,
        model: CatBoostClassifier,
        pool: Pool,
        output_format: str = "pandas",
        debug: bool = False,
    ) -> Union[pd.DataFrame, pa.Table]:
        """
        Extract a scorecard from a trained CatBoost model.

        Args:
            model: Trained CatBoostClassifier
            pool: CatBoost Pool object used for training/validation
            output_format: "pandas" or "arrow"
            debug: Whether to store debug information

        Returns:
            pd.DataFrame or pa.Table: Scorecard with leaf conditions, statistics, and predictions
        """
        # Store the model and pool for later access by visualizer
        cls.model = model
        cls.pool = pool

        # Reset debug info
        cls.debug_info = {"debug_enabled": debug}
        cb_obj = model._object
        tree_count = cb_obj._get_tree_count()
        y = pool.get_label()
        avg_event_rate = float(y.mean())

        # Extract leaf values and conditions
        leaf_records = []
        for tree_idx in range(tree_count):
            leaf_vals = cb_obj._get_tree_leaf_values(tree_idx)
            leaf_conditions = cls._get_leaf_conditions(cb_obj, pool, tree_idx)
            leaf_count = int(cb_obj._get_tree_leaf_counts()[tree_idx])

            for leaf_idx in range(leaf_count):
                val_str = leaf_vals[leaf_idx] if leaf_idx < len(leaf_vals) else "val = 0.0"
                clean_val = float(val_str.replace("val = ", "").strip())
                conditions = leaf_conditions.get(leaf_idx, "")

                # Extract feature and split info from conditions
                feature = None
                sign = None
                split = None

                if conditions:
                    # Try to extract feature and split info from the first condition
                    first_cond = conditions.split(" AND ")[0]
                    if match := re.match(r"([^<>=!]+)\s*([<>=!]+)\s*(.+)", first_cond):
                        feature = match[1].strip()
                        sign = match[2].strip()
                        split = match[3].strip()

                leaf_records.append(
                    {
                        "Tree": tree_idx,
                        "LeafIndex": leaf_idx,
                        "LeafValue": clean_val,
                        "Conditions": conditions,
                        "Feature": feature,
                        "Sign": sign,
                        "Split": split,
                    }
                )

        leaf_df = pd.DataFrame(leaf_records)

        # Get leaf assignments and aggregate statistics
        leaf_assignments = model.calc_leaf_indexes(pool)
        assignments_df = pd.DataFrame(
            leaf_assignments,
            columns=[f"tree_{i}" for i in range(leaf_assignments.shape[1])],
        )

        # Aggregate target stats per leaf
        agg_dfs = []
        for tree_idx in range(tree_count):
            temp_df = pd.DataFrame(
                {
                    "Tree": tree_idx,
                    "LeafIndex": assignments_df[f"tree_{tree_idx}"],
                    "Label": y,
                }
            )
            agg = (
                temp_df.groupby(["Tree", "LeafIndex"])["Label"]
                .agg(Events="sum", Count="count")
                .reset_index()
            )
            agg["NonEvents"] = agg["Count"] - agg["Events"]
            agg_dfs.append(agg)

        # Create complete scorecard
        scorecard_df = (
            pd.DataFrame(
                [
                    {"Tree": tree_idx, "LeafIndex": leaf_idx}
                    for tree_idx in range(tree_count)
                    for leaf_idx in range(int(cb_obj._get_tree_leaf_counts()[tree_idx]))
                ]
            )
            .merge(pd.concat(agg_dfs), on=["Tree", "LeafIndex"], how="left")
            .fillna({"Events": 0, "Count": 0, "NonEvents": 0})
            .merge(leaf_df, on=["Tree", "LeafIndex"], how="left")
        )

        # Calculate event rate
        scorecard_df["EventRate"] = (
            scorecard_df["Events"].astype(float) / scorecard_df["Count"].replace(0, np.nan)
        ).fillna(avg_event_rate)

        # Compute WOE per row
        clipped_event_rate = scorecard_df["EventRate"].clip(lower=1e-3, upper=1 - 1e-3)
        avg_odds = avg_event_rate / (1 - avg_event_rate)
        leaf_odds = clipped_event_rate / (1 - clipped_event_rate)

        # Initial WOE computation
        woe_raw = np.log(leaf_odds / avg_odds)

        # Set WOE = 0 if Events == NonEvents
        is_balanced = scorecard_df["Events"] == scorecard_df["NonEvents"]

        scorecard_df["WOE"] = np.where(is_balanced, 0.0, woe_raw)

        # Handle infinities / NaNs
        scorecard_df["WOE"] = (
            scorecard_df["WOE"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        ).round(4)

        # Calculate IV
        scorecard_df["IV"] = (
            (scorecard_df["WOE"] * (scorecard_df["EventRate"] - avg_event_rate))
            .fillna(0.0)
            .round(4)
        )

        # Calculate xAddEvidence
        scorecard_df["xAddEvidence"] = scorecard_df["LeafValue"]

        # Calculate CountPct
        total_count = scorecard_df["Count"].sum()
        scorecard_df["CountPct"] = (scorecard_df["Count"] / total_count * 100).fillna(0.0)

        # Add DetailedSplit column as a copy of Conditions
        scorecard_df["DetailedSplit"] = scorecard_df["Conditions"]

        scorecard_df = scorecard_df.sort_values(["Tree", "LeafIndex"]).reset_index(drop=True)

        # Rearrange columns in the specified order
        scorecard_df = scorecard_df[
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
                "xAddEvidence",
                "DetailedSplit",
            ]
        ]

        if output_format == "arrow":
            return pa.Table.from_pandas(scorecard_df, preserve_index=False)
        return scorecard_df

    @staticmethod
    def extract_leaf_weights(model: CatBoostClassifier) -> pd.DataFrame:
        """
        Extract leaf weights from a trained CatBoost model.

        Args:
            model: Trained CatBoostClassifier

        Returns:
            DataFrame containing leaf weights with columns:
            'Tree', 'Node', and 'XAddEvidence'
        """
        cb_obj = model._object
        tree_count = cb_obj._get_tree_count()

        records = []
        for tree_idx in range(tree_count):
            leaf_vals = cb_obj._get_tree_leaf_values(tree_idx)
            for node_idx, val_str in enumerate(leaf_vals):
                clean_val = float(val_str.replace("val = ", "").strip())
                records.append(
                    {
                        "Tree": tree_idx,
                        "Node": node_idx,
                        "XAddEvidence": clean_val,
                    }
                )

        return pd.DataFrame(records)
